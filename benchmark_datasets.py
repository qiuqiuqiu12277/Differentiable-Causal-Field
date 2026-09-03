"""Dataset adapters for MetaCausalField paper experiments.

The loaders use a common sample schema so CLEVRER, Causal3DIdent/CITRIS,
Causal-VidQA, MAG, and Lung can share training/evaluation code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from metacausal_field import SimpleTokenizer
from train_metacausal_field import label_to_class, numeric_factor_columns, prepare_metadata


@dataclass
class CausalSample:
    sample_id: str
    split: str
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    counterfactual_image_path: Optional[str] = None
    counterfactual_video_path: Optional[str] = None
    text: str = ""
    question: str = ""
    answer: str = ""
    question_type: str = "Descriptive"
    factors: Dict[str, float] = field(default_factory=dict)
    graph_edges: List[Tuple[str, str]] = field(default_factory=list)
    intervention: Optional[Dict] = None
    factual_text: str = ""
    counterfactual_text: str = ""
    cf_answer: Optional[str] = None
    ood_type: str = "in_domain"
    object_id: Optional[str] = None
    bbox: Optional[List[float]] = None
    mask_path: Optional[str] = None
    feature_key: Optional[str] = None
    choices: List[str] = field(default_factory=list)
    objects: List[Dict[str, Any]] = field(default_factory=list)


def read_manifest(path: str) -> List[Dict]:
    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}. Provide a JSON/JSONL/CSV manifest for the benchmark."
        )
    if manifest.suffix == ".jsonl":
        with open(manifest) as f:
            return [json.loads(line) for line in f if line.strip()]
    if manifest.suffix == ".json":
        with open(manifest) as f:
            payload = json.load(f)
        return payload if isinstance(payload, list) else payload.get("samples", [])
    if manifest.suffix == ".csv":
        return pd.read_csv(manifest).to_dict("records")
    raise ValueError(f"Unsupported manifest format: {manifest.suffix}")


def normalize_edges(edges) -> List[Tuple[str, str]]:
    if edges is None or (isinstance(edges, float) and pd.isna(edges)):
        return []
    if isinstance(edges, str):
        try:
            edges = json.loads(edges)
        except json.JSONDecodeError:
            parsed = []
            for item in edges.split(";"):
                if "->" in item:
                    src, dst = item.split("->", 1)
                    parsed.append((src.strip(), dst.strip()))
            return parsed
    normalized = []
    for edge in edges:
        if isinstance(edge, dict):
            normalized.append((str(edge["source"]), str(edge["target"])))
        elif len(edge) >= 2:
            normalized.append((str(edge[0]), str(edge[1])))
    return normalized


def parse_jsonish(value, default=None):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    return default


def parse_bbox(value) -> Optional[List[float]]:
    parsed = parse_jsonish(value, value)
    if isinstance(parsed, dict):
        if {"x1", "y1", "x2", "y2"} <= set(parsed):
            parsed = [parsed["x1"], parsed["y1"], parsed["x2"], parsed["y2"]]
        elif {"x", "y", "w", "h"} <= set(parsed):
            parsed = [parsed["x"], parsed["y"], parsed["x"] + parsed["w"], parsed["y"] + parsed["h"]]
    if isinstance(parsed, str):
        parts = [p.strip() for p in parsed.replace(";", ",").split(",") if p.strip()]
        parsed = parts
    if isinstance(parsed, (list, tuple)) and len(parsed) >= 4:
        try:
            return [float(x) for x in parsed[:4]]
        except (TypeError, ValueError):
            return None
    return None


def parse_intervention(value) -> Optional[Dict]:
    parsed = parse_jsonish(value, value)
    return parsed if isinstance(parsed, dict) else None


def parse_objects(value) -> List[Dict[str, Any]]:
    parsed = parse_jsonish(value, [])
    if isinstance(parsed, dict):
        parsed = parsed.get("objects", [])
    out = []
    if isinstance(parsed, list):
        for obj in parsed:
            if not isinstance(obj, dict):
                continue
            item = dict(obj)
            item["bbox"] = parse_bbox(item.get("bbox", item.get("box")))
            item["id"] = str(item.get("id", item.get("object_id", item.get("name", len(out)))))
            out.append(item)
    return out


def _resolve_media_path(path_value, root: Path) -> Optional[str]:
    if path_value is None or (isinstance(path_value, float) and pd.isna(path_value)):
        return None
    path = str(path_value)
    if not path:
        return None
    if path.startswith(("http://", "https://", "/")):
        return path
    return str(root / path)


class UnifiedCausalDataset(Dataset):
    def __init__(
        self,
        samples: List[CausalSample],
        tokenizer: SimpleTokenizer,
        transform=None,
        factor_names: Optional[List[str]] = None,
        num_video_frames: int = 8,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.transform = transform
        self.num_video_frames = max(1, int(num_video_frames))
        self.factor_names = factor_names or sorted({name for s in samples for name in s.factors})

    def __len__(self):
        return len(self.samples)

    def _load_image(self, image_path: Optional[str], video_path: Optional[str] = None):
        if image_path:
            try:
                image = Image.open(image_path).convert("RGB")
                return self.transform(image) if self.transform else image
            except Exception:
                pass
        if video_path:
            try:
                from torchvision.io import read_video
                frames, _, _ = read_video(video_path, pts_unit="sec")
                if len(frames) > 0:
                    if self.num_video_frames <= 1:
                        frame = frames[len(frames) // 2].numpy()
                        image = Image.fromarray(frame).convert("RGB")
                        return self.transform(image) if self.transform else image
                    indices = torch.linspace(0, len(frames) - 1, self.num_video_frames).round().long()
                    video_frames = []
                    for frame_idx in indices.tolist():
                        image = Image.fromarray(frames[frame_idx].numpy()).convert("RGB")
                        if self.transform:
                            video_frames.append(self.transform(image))
                        else:
                            from torchvision.transforms.functional import to_tensor
                            video_frames.append(to_tensor(image))
                    return torch.stack(video_frames, dim=0)
            except Exception:
                pass
        image = Image.new("RGB", (224, 224), "white")
        return self.transform(image) if self.transform else image

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample.question or sample.text or sample.factual_text
        answer_text = sample.answer or sample.counterfactual_text or sample.text
        return {
            "sample_id": sample.sample_id,
            "split": sample.split,
            "image": self._load_image(sample.image_path, sample.video_path),
            "cf_image": self._load_image(sample.counterfactual_image_path, sample.counterfactual_video_path)
            if (sample.counterfactual_image_path or sample.counterfactual_video_path)
            else self._load_image(sample.image_path, sample.video_path),
            "image_path": sample.image_path or "",
            "video_path": sample.video_path or "",
            "counterfactual_image_path": sample.counterfactual_image_path or "",
            "counterfactual_video_path": sample.counterfactual_video_path or "",
            "input_ids": torch.tensor(self.tokenizer.encode(text), dtype=torch.long),
            "answer_ids": torch.tensor(self.tokenizer.encode(answer_text), dtype=torch.long),
            "text": text,
            "answer": answer_text,
            "factual_text": sample.factual_text or sample.text,
            "counterfactual_text": sample.counterfactual_text or sample.cf_answer or "",
            "question_type": sample.question_type,
            "factor_targets": torch.tensor(
                [label_to_class(sample.factors.get(name, float("nan"))) for name in self.factor_names],
                dtype=torch.long,
            ),
            "graph_edges": sample.graph_edges,
            "intervention": sample.intervention or {},
            "cf_answer": sample.cf_answer or "",
            "ood_type": sample.ood_type,
            "object_id": sample.object_id or "",
            "bbox": torch.tensor(sample.bbox or [-1, -1, -1, -1], dtype=torch.float32),
            "mask_path": sample.mask_path or "",
            "feature_key": sample.feature_key or sample.image_path or sample.video_path or sample.sample_id,
            "cf_feature_key": sample.counterfactual_image_path or sample.counterfactual_video_path or sample.feature_key or sample.image_path or sample.video_path or sample.sample_id,
            "choices": sample.choices,
            "objects": sample.objects,
        }


def samples_from_manifest(manifest_path: str, data_root: Optional[str] = None, default_split: str = "train") -> List[CausalSample]:
    root = Path(data_root) if data_root else Path(manifest_path).parent
    samples = []
    for i, row in enumerate(read_manifest(manifest_path)):
        image_path = _resolve_media_path(row.get("image_path") or row.get("image") or row.get("frame_path"), root)
        video_path = _resolve_media_path(row.get("video_path") or row.get("video") or row.get("video_filename"), root)
        cf_image_path = _resolve_media_path(
            row.get("counterfactual_image_path") or row.get("cf_image_path") or row.get("counterfactual_image"),
            root,
        )
        cf_video_path = _resolve_media_path(
            row.get("counterfactual_video_path") or row.get("cf_video_path") or row.get("counterfactual_video"),
            root,
        )
        factors = row.get("factors", {})
        factors = parse_jsonish(factors, {})
        choices = row.get("choices", row.get("options", []))
        if isinstance(choices, str):
            try:
                choices = json.loads(choices)
            except json.JSONDecodeError:
                choices = [x.strip() for x in choices.split("|") if x.strip()]
        samples.append(CausalSample(
            sample_id=str(row.get("sample_id", row.get("id", i))),
            split=str(row.get("split", default_split)),
            image_path=image_path,
            video_path=video_path,
            counterfactual_image_path=cf_image_path,
            counterfactual_video_path=cf_video_path,
            text=str(row.get("text", row.get("caption", ""))),
            question=str(row.get("question", "")),
            answer=str(row.get("answer", row.get("label", ""))),
            question_type=str(row.get("question_type", row.get("type", "Descriptive"))),
            factors={str(k): float(v) for k, v in factors.items()} if isinstance(factors, dict) else {},
            graph_edges=normalize_edges(row.get("graph_edges", row.get("edges"))),
            intervention=parse_intervention(row.get("intervention")),
            factual_text=str(row.get("factual_text", row.get("source_text", row.get("text", "")))),
            counterfactual_text=str(row.get("counterfactual_text", row.get("target_text", row.get("cf_text", "")))),
            cf_answer=row.get("cf_answer", row.get("counterfactual_answer")),
            ood_type=str(row.get("ood_type", row.get("domain", "in_domain"))),
            object_id=str(row.get("object_id", "")) if row.get("object_id") is not None else None,
            bbox=parse_bbox(row.get("bbox", row.get("box"))),
            mask_path=row.get("mask_path"),
            feature_key=row.get("feature_key"),
            choices=[str(x) for x in choices] if isinstance(choices, list) else [],
            objects=parse_objects(row.get("objects")),
        ))
    return samples


def load_clevrer(manifest_path: str, data_root: Optional[str] = None) -> List[CausalSample]:
    path = Path(manifest_path)
    if path.suffix not in {".json", ".jsonl"}:
        return samples_from_manifest(manifest_path, data_root)
    root = Path(data_root) if data_root else path.parent
    rows = read_manifest(manifest_path)
    if rows and any(k in rows[0] for k in ("question", "image_path", "video_path")):
        return samples_from_manifest(manifest_path, data_root)
    samples = []
    for vid_idx, video in enumerate(rows):
        if not isinstance(video, dict):
            continue
        video_name = video.get("video_filename", video.get("video", video.get("video_path", f"{vid_idx}.mp4")))
        video_path = _resolve_media_path(video_name, root)
        objects = parse_objects(video.get("objects", video.get("object_annotations")))
        edges = normalize_edges(video.get("graph_edges", video.get("edges")))
        for collision in video.get("collisions", video.get("events", [])) or []:
            if isinstance(collision, dict):
                src = collision.get("object1", collision.get("source"))
                dst = collision.get("object2", collision.get("target"))
                if src is not None and dst is not None:
                    edges.append((str(src), str(dst)))
        for q_idx, q in enumerate(video.get("questions", video.get("qa", [])) or []):
            if not isinstance(q, dict):
                continue
            samples.append(CausalSample(
                sample_id=str(q.get("question_id", f"{vid_idx}_{q_idx}")),
                split=str(q.get("split", video.get("split", "train"))),
                video_path=video_path,
                question=str(q.get("question", q.get("query", ""))),
                answer=str(q.get("answer", q.get("label", ""))),
                question_type=str(q.get("question_type", q.get("type", q.get("program_type", "Descriptive")))),
                graph_edges=normalize_edges(q.get("graph_edges", q.get("edges"))) or edges,
                intervention=parse_intervention(q.get("intervention")),
                cf_answer=q.get("cf_answer", q.get("counterfactual_answer")),
                ood_type=str(q.get("ood_type", q.get("domain", video.get("ood_type", "in_domain")))),
                choices=[str(x) for x in parse_jsonish(q.get("choices", q.get("options")), []) or []],
                objects=objects,
            ))
    return samples


def load_causal3d_citris(manifest_path: str, data_root: Optional[str] = None) -> List[CausalSample]:
    path = Path(manifest_path)
    if path.suffix != ".npz":
        return samples_from_manifest(manifest_path, data_root)
    import numpy as np
    root = Path(data_root) if data_root else path.parent
    data = np.load(path, allow_pickle=True)
    latents = data.get("latents", data.get("targets", None))
    images = data.get("image_paths", data.get("images_path", None))
    split = data.get("split", None)
    factor_names = [str(x) for x in data.get("factor_names", [])] if "factor_names" in data else []
    n = len(latents) if latents is not None else len(images)
    samples = []
    for i in range(n):
        factors = {}
        if latents is not None:
            values = np.asarray(latents[i]).reshape(-1)
            names = factor_names or [f"factor_{j}" for j in range(len(values))]
            factors = {names[j]: float(values[j]) for j in range(min(len(names), len(values)))}
        image_path = None
        if images is not None:
            image_path = _resolve_media_path(str(np.asarray(images[i]).item()), root)
        samples.append(CausalSample(
            sample_id=str(i),
            split=str(np.asarray(split[i]).item()) if split is not None else "train",
            image_path=image_path,
            question="Predict the causal latent factors.",
            answer=json.dumps(factors, ensure_ascii=False),
            question_type="Descriptive",
            factors=factors,
            graph_edges=normalize_edges(data.get("graph_edges", [])),
        ))
    return samples


def load_causal_vidqa(manifest_path: str, data_root: Optional[str] = None) -> List[CausalSample]:
    path = Path(manifest_path)
    root = Path(data_root) if data_root else path.parent
    rows = read_manifest(manifest_path)
    if rows and any(k in rows[0] for k in ("question", "answer", "video_path", "image_path")):
        return samples_from_manifest(manifest_path, data_root)
    samples = []
    for vid_idx, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        video_path = _resolve_media_path(item.get("video_path", item.get("video", item.get("video_filename"))), root)
        objects = parse_objects(item.get("objects"))
        for q_idx, q in enumerate(item.get("questions", item.get("qas", item.get("qa", []))) or []):
            if not isinstance(q, dict):
                continue
            samples.append(CausalSample(
                sample_id=str(q.get("id", q.get("question_id", f"{vid_idx}_{q_idx}"))),
                split=str(q.get("split", item.get("split", "train"))),
                video_path=video_path,
                text=str(item.get("caption", "")),
                question=str(q.get("question", q.get("query", ""))),
                answer=str(q.get("answer", q.get("label", ""))),
                question_type=str(q.get("question_type", q.get("type", "Reasoning"))),
                graph_edges=normalize_edges(q.get("graph_edges", item.get("graph_edges", item.get("edges")))),
                intervention=parse_intervention(q.get("intervention")),
                factual_text=str(q.get("factual_text", "")),
                counterfactual_text=str(q.get("counterfactual_text", "")),
                cf_answer=q.get("cf_answer", q.get("counterfactual_answer")),
                ood_type=str(q.get("ood_type", item.get("ood_type", "in_domain"))),
                object_id=str(q.get("object_id", "")) if q.get("object_id") is not None else None,
                bbox=parse_bbox(q.get("bbox", q.get("box"))),
                choices=[str(x) for x in parse_jsonish(q.get("choices", q.get("options")), []) or []],
                objects=objects,
            ))
    return samples


def load_mag_lung(dataset: str) -> Tuple[List[CausalSample], List[str]]:
    df, _, factor_names = prepare_metadata(dataset)
    graph_file = Path("gold_graphs/mag9_gold_graph.csv" if dataset == "MAG9" else "gold_graphs/lung_gold_graph.csv")
    graph_edges = []
    if graph_file.exists():
        graph_df = pd.read_csv(graph_file)
        graph_edges = [(str(r["source"]), str(r["target"])) for _, r in graph_df.iterrows()]
    samples = []
    for idx, row in df.iterrows():
        factors = {
            name: float(pd.to_numeric(row.get(name), errors="coerce"))
            for name in factor_names
            if not pd.isna(pd.to_numeric(row.get(name), errors="coerce"))
        }
        samples.append(CausalSample(
            sample_id=str(row.get("id", idx)),
            split="train" if idx < int(len(df) * 0.8) else "test",
            image_path=str(row["ImagePath"]),
            text=str(row["Review"]),
            question="Describe the causal factors and predict the score.",
            answer=str(row["Review"]),
            question_type="Descriptive",
            factors=factors,
            graph_edges=graph_edges,
        ))
    return samples, factor_names


def load_benchmark(name: str, manifest_path: Optional[str] = None, data_root: Optional[str] = None) -> Tuple[List[CausalSample], List[str]]:
    if name in {"MAG9", "MAG"}:
        return load_mag_lung("MAG9")
    if name in {"Lung", "Lung4"}:
        return load_mag_lung("Lung")
    if manifest_path is None:
        raise ValueError(f"{name} requires --manifest_path")
    if name == "CLEVRER":
        samples = load_clevrer(manifest_path, data_root)
    elif name in {"Causal3DIdent", "CITRIS"}:
        samples = load_causal3d_citris(manifest_path, data_root)
    elif name == "Causal-VidQA":
        samples = load_causal_vidqa(manifest_path, data_root)
    else:
        raise ValueError(f"Unknown benchmark: {name}")
    factor_names = sorted({name for sample in samples for name in sample.factors})
    return samples, factor_names
