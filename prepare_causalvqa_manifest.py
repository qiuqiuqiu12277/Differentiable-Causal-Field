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


def split_for(sample_id: str, test_fraction: float) -> str:
    value = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if value < test_fraction else "train"


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

    split = str(row.get("split") or default_split or split_for(sample_id, test_fraction))
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
    intervention = parse_intervention(row.get("intervention"))
    if intervention is None:
        target = row.get("intervention_target", row.get("target", row.get("object_id")))
        action = row.get("intervention_type", row.get("action", None))
        if target or action:
            intervention = {"target": str(target) if target is not None else "", "type": str(action or "modify")}

    return {
        "sample_id": sample_id,
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
    parser.add_argument("--test_fraction", type=float, default=0.2)
    args = parser.parse_args()

    input_path = Path(args.input)
    root = Path(args.data_root) if args.data_root else input_path.parent
    rows = read_rows(input_path)
    records = flatten_rows(rows, root, args.default_split, args.test_fraction)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} CausalVQA records to {output}")


if __name__ == "__main__":
    main()
