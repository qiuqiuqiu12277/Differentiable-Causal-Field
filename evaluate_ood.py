"""Evaluate explicitly annotated, held-out OOD samples.

Synthetic paraphrases are available only as an opt-in plumbing smoke test and
are excluded from every paper-style OOD aggregate.
"""

import argparse
import json
from pathlib import Path

import torch

from benchmark_datasets import load_benchmark
from build_ood_splits import SYNTHETIC_TEMPLATE_SMOKE, select_verified_ood_samples
from causal_metrics import ood_metrics, qa_category_accuracy
from evaluate_metacausal_field import load_model
from frozen_backbones import CachedFrozenBackbone
from three_stage_metacausal_pipeline import run_model_records


def main():
    parser = argparse.ArgumentParser(description="Evaluate OOD splits")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="Causal-VidQA", choices=["CLEVRER", "Causal3DIdent", "CITRIS", "Causal-VidQA", "MAG9", "Lung"])
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output", default="./ood_metrics.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--edge_threshold", type=float, default=0.05)
    parser.add_argument("--num_video_frames", type=int, default=8)
    parser.add_argument("--generation_max_new_tokens", type=int, default=32)
    parser.add_argument("--factor_edge_top_k", type=int, default=None)
    parser.add_argument("--use_frozen_language_tokens", action="store_true")
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument(
        "--allow_missing_media", action="store_true", help="Smoke tests only"
    )
    parser.add_argument(
        "--synthetic_template_smoke",
        action="store_true",
        help="Add smoke-only paraphrases; these are excluded from Avg_OOD and paper metrics.",
    )
    parser.add_argument("--auto_construct", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.auto_construct:
        parser.error(
            "--auto_construct was removed because it fabricated paper OOD labels from in-domain "
            "records. Supply an explicitly annotated held-out manifest, or use "
            "--synthetic_template_smoke for a non-paper plumbing check."
        )

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    samples, factor_names = load_benchmark(args.dataset, args.manifest_path, args.data_root)
    samples = select_verified_ood_samples(
        samples,
        include_in_domain=True,
        synthetic_template_smoke=args.synthetic_template_smoke,
    )
    cached_backbone = (
        CachedFrozenBackbone(
            args.feature_cache,
            allow_smoke_cache=args.allow_missing_media,
        ).to(device)
        if args.feature_cache
        else None
    )
    model, tokenizer, factor_columns, _, _ = load_model(
        args.checkpoint,
        device,
        visual_encoder=cached_backbone is None,
        expected_dataset=args.dataset,
    )
    if any(not (sample.image_path or sample.video_path) and sample.feature_key for sample in samples) and cached_backbone is None:
        raise ValueError("Feature-key-only OOD records require --feature_cache.")
    records, _ = run_model_records(
        args,
        samples,
        model,
        tokenizer,
        factor_columns or factor_names,
        device,
        cached_backbone=cached_backbone,
    )

    paper_records = [
        record
        for record in records
        if record.get("ood_type") != SYNTHETIC_TEMPLATE_SMOKE
    ]
    smoke_records = [
        record
        for record in records
        if record.get("ood_type") == SYNTHETIC_TEMPLATE_SMOKE
    ]
    metrics = {}
    metrics.update(qa_category_accuracy(paper_records))
    metrics.update(ood_metrics(paper_records))
    metrics["Verified_OOD_Count"] = sum(
        record.get("ood_type") not in {"in_domain", SYNTHETIC_TEMPLATE_SMOKE}
        for record in records
    )
    metrics["Synthetic_Template_Smoke_Count"] = len(smoke_records)
    if smoke_records:
        metrics["Synthetic_Template_Smoke_Accuracy_NOT_PAPER_METRIC"] = sum(
            bool(record.get("correct", False)) for record in smoke_records
        ) / len(smoke_records)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
