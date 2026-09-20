"""Prepare CausalVQA-style annotations as unified MetaCausalField JSONL.

The public CausalVQA/Causal-VidQA variants appear in both nested video-level
JSON and flat CSV/JSONL forms. This converter keeps field handling deliberately
forgiving: it normalizes common aliases for media paths, question category,
choices, interventions, counterfactual answers, objects, and bounding boxes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from benchmark_datasets import normalize_edges, parse_bbox, parse_intervention, parse_jsonish, parse_objects


def read_rows(path: Path):
    if path.suffix == ".jsonl":
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix == ".json":
        with open(path) as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        for key in ("samples", "videos", "data", "annotations", "questions"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    if path.suffix == ".csv":
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported annotation format: {path.suffix}")


def resolve_path(value, root: Path):
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    if text.startswith(("http://", "https://", "/")):
        return text
    return str(root / text)


SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "eval": "test",
    "evaluation": "test",
}


def assign_group_splits(
    records: List[Dict[str, Any]],
    *,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Assign deterministic train/val/test labels at media-group level."""
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("train_fraction and val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must leave a test fraction")
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        group_id = str(record.get("group_id") or record["sample_id"])
        groups.setdefault(group_id, []).append(record)
    if len(groups) < 3:
        raise ValueError(
            "At least three distinct video/scene/media groups are required for "
            "non-empty train/val/test splits."
        )

    assigned: Dict[str, str] = {}
    for group_id, group_records in groups.items():
        labels = {
            SPLIT_ALIASES.get(str(record.get("split", "")).strip().lower(), "")
            for record in group_records
            if str(record.get("split", "")).strip()
        }
        if "" in labels:
            raw = sorted({str(record.get("split")) for record in group_records})
            raise ValueError(f"Unsupported explicit split label in group {group_id!r}: {raw}")
        if len(labels) > 1:
            raise ValueError(
                f"Group {group_id!r} spans explicit splits {sorted(labels)}; this leaks shared media."
            )
        if labels:
            assigned[group_id] = next(iter(labels))

    unassigned = sorted(set(groups) - set(assigned))
    random.Random(seed).shuffle(unassigned)
    n_groups = len(groups)
    targets = {
        "train": max(1, int(round(n_groups * train_fraction))),
        "val": max(1, int(round(n_groups * val_fraction))),
    }
    if targets["train"] + targets["val"] >= n_groups:
        overflow = targets["train"] + targets["val"] - (n_groups - 1)
        targets["train"] = max(1, targets["train"] - overflow)
    targets["test"] = n_groups - targets["train"] - targets["val"]

    counts = {name: sum(label == name for label in assigned.values()) for name in targets}
    for group_id in unassigned:
        deficits = {name: targets[name] - counts[name] for name in targets}
        label = max(("train", "val", "test"), key=lambda name: (deficits[name], -counts[name]))
        assigned[group_id] = label
        counts[label] += 1

    if any(counts[name] == 0 for name in ("train", "val", "test")):
        raise ValueError(
            "Explicit split assignments leave an empty partition and cannot be repaired "
            "without moving an explicitly labelled media group."
        )
    for group_id, group_records in groups.items():
        for record in group_records:
            record["split"] = assigned[group_id]
    return records


def choices_from(row: Dict[str, Any]):
    choices = row.get("choices", row.get("options", row.get("answer_choices", [])))
    if isinstance(choices, str):
        parsed = parse_jsonish(choices, None)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [x.strip() for x in choices.replace(";", "|").split("|") if x.strip()]
    return [str(x) for x in choices] if isinstance(choices, list) else []


def record_from_flat(row: Dict[str, Any], root: Path, default_split: str, test_fraction: float, prefix: str = ""):
    sample_id = str(row.get("sample_id", row.get("question_id", row.get("id", ""))))
    if prefix and sample_id:
        sample_id = f"{prefix}_{sample_id}"
    if not sample_id:
        sample_id = hashlib.sha1(json.dumps(row, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]

    del test_fraction  # Retained in the call signature for compatibility.
    split = str(row.get("split") or default_split or "")
    image_path = resolve_path(row.get("image_path") or row.get("image") or row.get("frame_path"), root)
    video_path = resolve_path(row.get("video_path") or row.get("video") or row.get("video_filename"), root)
    cf_image_path = resolve_path(
        row.get("counterfactual_image_path") or row.get("cf_image_path") or row.get("counterfactual_image"),
        root,
    )
    cf_video_path = resolve_path(
        row.get("counterfactual_video_path") or row.get("cf_video_path") or row.get("counterfactual_video"),
        root,
    )
    group_id = str(
        row.get("group_id")
        or row.get("video_id")
        or row.get("scene_id")
        or row.get("clip_id")
        or video_path
        or image_path
        or prefix
        or sample_id
    )
    intervention = parse_intervention(row.get("intervention"))
    if intervention is None:
        target = row.get("intervention_target", row.get("target", row.get("object_id")))
        action = row.get("intervention_type", row.get("action", None))
        if target or action:
            intervention = {"target": str(target) if target is not None else "", "type": str(action or "modify")}

    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "split": split,
        "image_path": image_path,
        "video_path": video_path,
        "counterfactual_image_path": cf_image_path,
        "counterfactual_video_path": cf_video_path,
        "text": str(row.get("caption", row.get("context", ""))),
        "question": str(row.get("question", row.get("query", row.get("text", "")))),
        "answer": str(row.get("answer", row.get("label", row.get("gold_answer", "")))),
        "question_type": str(row.get("question_type", row.get("category", row.get("type", "Reasoning")))),
        "choices": choices_from(row),
        "graph_edges": [{"source": s, "target": t} for s, t in normalize_edges(row.get("graph_edges", row.get("edges")))],
        "intervention": intervention,
        "factual_text": str(row.get("factual_text", "")),
        "counterfactual_text": str(row.get("counterfactual_text", "")),
        "cf_answer": row.get("cf_answer", row.get("counterfactual_answer", row.get("counterfactual_label"))),
        "ood_type": str(row.get("ood_type", row.get("domain", "in_domain"))),
        "object_id": str(row.get("object_id", row.get("target_object", ""))),
        "bbox": parse_bbox(row.get("bbox", row.get("box", row.get("target_bbox")))),
        "objects": parse_objects(row.get("objects", row.get("object_annotations"))),
    }


def flatten_rows(rows: Iterable[Dict[str, Any]], root: Path, default_split: str, test_fraction: float):
    records = []
    for idx, row in enumerate(rows):
        questions = row.get("questions", row.get("qas", row.get("qa")))
        if isinstance(questions, list):
            shared = dict(row)
            shared.pop("questions", None)
            shared.pop("qas", None)
            shared.pop("qa", None)
            prefix = str(row.get("video_id", row.get("id", idx)))
            for question in questions:
                if not isinstance(question, dict):
                    continue
                merged = dict(shared)
                merged.update(question)
                records.append(record_from_flat(merged, root, default_split, test_fraction, prefix=prefix))
        else:
            records.append(record_from_flat(row, root, default_split, test_fraction))
    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare CausalVQA/Causal-VidQA manifest")
    parser.add_argument("--input", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--default_split", default="")
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    root = Path(args.data_root) if args.data_root else input_path.parent
    rows = read_rows(input_path)
    legacy_test_fraction = args.test_fraction
    train_fraction = args.train_fraction
    val_fraction = args.val_fraction
    if legacy_test_fraction is not None:
        if not 0 < legacy_test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1")
        remaining = 1.0 - legacy_test_fraction
        train_fraction = remaining * (args.train_fraction / (args.train_fraction + args.val_fraction))
        val_fraction = remaining - train_fraction
    records = flatten_rows(rows, root, args.default_split, legacy_test_fraction or 0.15)
    records = assign_group_splits(
        records,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} CausalVQA records to {output}")


if __name__ == "__main__":
    main()
