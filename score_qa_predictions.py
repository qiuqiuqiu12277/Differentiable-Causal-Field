"""Strict QA scorer for paper benchmark categories and OOD slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmark_datasets import samples_from_manifest
from causal_metrics import answer_correct, counterfactual_consistency, ood_metrics, qa_category_accuracy


def read_predictions(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]
    if p.suffix == ".json":
        with open(p) as f:
            payload = json.load(f)
        return payload if isinstance(payload, list) else payload.get("records", [])
    if p.suffix == ".csv":
        return pd.read_csv(p).to_dict("records")
    raise ValueError(f"Unsupported prediction file: {p}")


def main():
    parser = argparse.ArgumentParser(description="Score QA predictions with exact/multiple-choice/numeric matching")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest_path", default=None, help="Optional manifest to fill missing gold/category/OOD metadata")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output", default="./qa_score.json")
    args = parser.parse_args()

    records = read_predictions(args.predictions)
    manifest = {}
    if args.manifest_path:
        for sample in samples_from_manifest(args.manifest_path, args.data_root):
            manifest[sample.sample_id] = sample

    scored = []
    cf_records = []
    for record in records:
        sample = manifest.get(str(record.get("sample_id", "")))
        gold = record.get("gold", sample.answer if sample else "")
        choices = record.get("choices", sample.choices if sample else [])
        scored_record = dict(record)
        scored_record.setdefault("question_type", sample.question_type if sample else "Descriptive")
        scored_record.setdefault("ood_type", sample.ood_type if sample else "in_domain")
        scored_record["gold"] = gold
        scored_record["correct"] = answer_correct(record.get("pred", ""), gold, choices)
        scored.append(scored_record)
        if "cf_pred" in record or "cf_gold" in record:
            cf_records.append(record)

    metrics = {
        "qa": qa_category_accuracy(scored),
        "ood": ood_metrics(scored),
        "n": len(scored),
    }
    if cf_records:
        metrics["counterfactual"] = counterfactual_consistency(cf_records)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
