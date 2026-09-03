"""Metrics for MetaCausalField paper experiments."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import re


Edge = Tuple[str, str]


def _as_edge_set(edges: Iterable) -> Set[Edge]:
    out = set()
    for edge in edges or []:
        if isinstance(edge, dict):
            src, dst = edge.get("source"), edge.get("target")
        else:
            src, dst = edge[:2]
        if src is not None and dst is not None and str(src) != str(dst):
            out.add((str(src), str(dst)))
    return out


def precision_recall_f1(tp: int, pred: int, gold: int) -> Tuple[float, float, float]:
    p = tp / pred if pred else 0.0
    r = tp / gold if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def factor_discovery_metrics(pred_factors: Iterable[str], gold_factors: Iterable[str]) -> Dict[str, float]:
    pred = {str(x) for x in pred_factors}
    gold = {str(x) for x in gold_factors}
    p, r, f = precision_recall_f1(len(pred & gold), len(pred), len(gold))
    return {"NP": p, "NR": r, "NF": f}


def edge_discovery_metrics(pred_edges: Iterable, gold_edges: Iterable) -> Dict[str, float]:
    pred = _as_edge_set(pred_edges)
    gold = _as_edge_set(gold_edges)
    p, r, f = precision_recall_f1(len(pred & gold), len(pred), len(gold))
    return {"AP": p, "AR": r, "AF": f}


def eshd(pred_edges: Iterable, gold_edges: Iterable) -> int:
    """Extended structural Hamming distance for directed edge sets.

    Missing/additional edges cost 1. Reversed edges cost 1 instead of 2.
    """
    pred = set(_as_edge_set(pred_edges))
    gold = set(_as_edge_set(gold_edges))
    distance = 0
    for edge in list(gold):
        if edge in pred:
            pred.remove(edge)
            gold.remove(edge)
    for src, dst in list(gold):
        rev = (dst, src)
        if rev in pred:
            distance += 1
            pred.remove(rev)
            gold.remove((src, dst))
    distance += len(pred) + len(gold)
    return distance


def structure_metrics(
    pred_factors: Iterable[str],
    gold_factors: Iterable[str],
    pred_edges: Iterable,
    gold_edges: Iterable,
) -> Dict[str, float]:
    metrics = factor_discovery_metrics(pred_factors, gold_factors)
    metrics.update(edge_discovery_metrics(pred_edges, gold_edges))
    metrics["ESHD"] = float(eshd(pred_edges, gold_edges))
    return metrics


def qa_category_accuracy(records: List[Dict], categories: Optional[List[str]] = None) -> Dict[str, float]:
    buckets = defaultdict(lambda: [0, 0])
    for record in records:
        qtype = str(record.get("question_type", "Descriptive"))
        correct = bool(record.get("correct", False))
        buckets[qtype][1] += 1
        buckets[qtype][0] += int(correct)
        buckets["All"][1] += 1
        buckets["All"][0] += int(correct)
    keys = categories or sorted(k for k in buckets if k != "All") + ["All"]
    return {
        f"{key}_Accuracy": buckets[key][0] / buckets[key][1] if buckets[key][1] else math.nan
        for key in keys
    }


def ood_metrics(records: List[Dict], in_domain_name: str = "in_domain") -> Dict[str, float]:
    buckets = defaultdict(lambda: [0, 0])
    for record in records:
        domain = str(record.get("ood_type", in_domain_name))
        correct = bool(record.get("correct", False))
        buckets[domain][0] += int(correct)
        buckets[domain][1] += 1
    metrics = {}
    for domain, (hits, total) in buckets.items():
        metrics[f"{domain}_Accuracy"] = hits / total if total else math.nan
    ood_values = [
        hits / total
        for domain, (hits, total) in buckets.items()
        if domain != in_domain_name and total
    ]
    metrics["Avg_OOD"] = float(np.mean(ood_values)) if ood_values else math.nan
    in_domain = metrics.get(f"{in_domain_name}_Accuracy", math.nan)
    metrics["Avg_OOD_Drop"] = in_domain - metrics["Avg_OOD"] if not math.isnan(in_domain) and not math.isnan(metrics["Avg_OOD"]) else math.nan
    return metrics


def normalize_answer(answer: str) -> str:
    return " ".join(str(answer).lower().strip().split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def multiple_choice_match(pred: str, gold: str, choices: Optional[List[str]] = None) -> bool:
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True
    if choices:
        letters = "abcdefghijklmnopqrstuvwxyz"
        gold_idx = None
        if len(gold_norm) == 1 and gold_norm in letters[:len(choices)]:
            gold_idx = letters.index(gold_norm)
        else:
            for i, choice in enumerate(choices):
                if normalize_answer(choice) == gold_norm:
                    gold_idx = i
                    break
        if gold_idx is not None:
            patterns = [
                rf"\b{letters[gold_idx]}\b",
                re.escape(normalize_answer(choices[gold_idx])),
            ]
            return any(re.search(pattern, pred_norm) for pattern in patterns)
    # Numeric answers often differ only in formatting.
    try:
        return abs(float(pred_norm) - float(gold_norm)) < 1e-6
    except Exception:
        return False


def answer_correct(pred: str, gold: str, choices: Optional[List[str]] = None) -> bool:
    return multiple_choice_match(pred, gold, choices)


def counterfactual_consistency(records: List[Dict]) -> Dict[str, float]:
    total = len(records)
    if total == 0:
        return {
            "Consistency": math.nan,
            "Causal_Flip_Accuracy": math.nan,
            "Invalid_Transition_Rate": math.nan,
            "Counterfactual_Accuracy": math.nan,
        }

    consistent = 0
    flip_correct = 0
    flip_total = 0
    invalid = 0
    cf_correct = 0
    for record in records:
        factual_pred = normalize_answer(record.get("factual_pred", ""))
        cf_pred = normalize_answer(record.get("cf_pred", ""))
        factual_gold = normalize_answer(record.get("factual_gold", ""))
        cf_gold = normalize_answer(record.get("cf_gold", ""))
        should_flip = bool(record.get("should_flip", factual_gold != cf_gold))
        invalid_transition = bool(record.get("invalid_transition", False))

        cf_correct += int(cf_pred == cf_gold)
        consistent += int((factual_pred == factual_gold) and (cf_pred == cf_gold))
        if should_flip:
            flip_total += 1
            flip_correct += int(cf_pred != factual_pred and cf_pred == cf_gold)
        else:
            flip_total += 1
            flip_correct += int(cf_pred == factual_pred)
        invalid += int(invalid_transition)

    return {
        "Counterfactual_Accuracy": cf_correct / total,
        "Consistency": consistent / total,
        "Causal_Flip_Accuracy": flip_correct / flip_total if flip_total else math.nan,
        "Invalid_Transition_Rate": invalid / total,
    }


def bbox_iou(pred_box: Sequence[float], gold_box: Sequence[float]) -> float:
    px1, py1, px2, py2 = pred_box
    gx1, gy1, gx2, gy2 = gold_box
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    gold_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pred_area + gold_area - inter
    return inter / union if union > 0 else 0.0


def causal_localization_iou(pred_boxes: List[Sequence[float]], gold_boxes: List[Sequence[float]]) -> float:
    if not pred_boxes or not gold_boxes:
        return math.nan
    scores = []
    for gold in gold_boxes:
        scores.append(max(bbox_iou(pred, gold) for pred in pred_boxes))
    return float(np.mean(scores))


def key_object_identification_accuracy(records: List[Dict]) -> float:
    if not records:
        return math.nan
    return sum(str(r.get("pred_object")) == str(r.get("gold_object")) for r in records) / len(records)


def explanation_faithfulness(factual_scores: Sequence[float], ablated_scores: Sequence[float]) -> float:
    if not factual_scores:
        return math.nan
    drops = np.asarray(factual_scores, dtype=float) - np.asarray(ablated_scores, dtype=float)
    return float(np.mean(drops))


class LatencyTimer:
    def __init__(self):
        self.times = []

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.times.append(time.perf_counter() - self.start)

    def summary(self) -> Dict[str, float]:
        if not self.times:
            return {"latency_mean": math.nan, "latency_p95": math.nan}
        return {
            "latency_mean": float(np.mean(self.times)),
            "latency_p95": float(np.percentile(self.times, 95)),
        }


def influence_to_edges(influence: torch.Tensor, labels: List[str], threshold: float = 0.05) -> List[Edge]:
    if influence.dim() == 3:
        influence = influence.mean(dim=0)
    matrix = influence.detach().cpu()
    n = min(matrix.shape[0], len(labels))
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and float(matrix[i, j]) >= threshold:
                edges.append((labels[i], labels[j]))
    return edges


def factor_influence_to_edges(
    factor_influence: torch.Tensor,
    factor_names: List[str],
    threshold: float = 0.05,
    top_k: Optional[int] = None,
) -> List[Edge]:
    """Convert model factor-level influence matrices into named directed edges."""
    if factor_influence.dim() == 3:
        matrix = factor_influence.detach().cpu().mean(dim=0)
    else:
        matrix = factor_influence.detach().cpu()
    n = min(matrix.shape[0], len(factor_names))
    edges = []
    for i in range(n):
        scores = [(j, float(matrix[i, j])) for j in range(n) if i != j]
        if top_k is not None and top_k > 0:
            scores = sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]
        for j, score in scores:
            if score >= threshold:
                edges.append((factor_names[i], factor_names[j]))
    return edges
