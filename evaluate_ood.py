"""OOD evaluation for Scene/Object Composition/Template shifts."""

import argparse
import json
from pathlib import Path

import torch

from benchmark_datasets import load_benchmark
from build_ood_splits import make_variants
from causal_metrics import ood_metrics, qa_category_accuracy
from three_stage_metacausal_pipeline import run_model_records
from evaluate_metacausal_field import load_model


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
    parser.add_argument("--auto_construct", action="store_true", help="Construct Scene/Composition/Template/Intervention OOD variants in memory")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    samples, factor_names = load_benchmark(args.dataset, args.manifest_path, args.data_root)
    if args.auto_construct:
        samples = make_variants(samples, include_in_domain=True)
    model, tokenizer, factor_columns, _, _ = load_model(args.checkpoint, device)
    records, _ = run_model_records(args, samples, model, tokenizer, factor_columns or factor_names, device)
    metrics = {}
    metrics.update(qa_category_accuracy(records))
    metrics.update(ood_metrics(records))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
