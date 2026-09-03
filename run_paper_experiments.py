"""Orchestrate paper-style MetaCausalField experiments.

This script wires together the runnable pieces needed for the PDF protocol:
benchmark conversion/validation, frozen MLLM feature extraction, staged
training, three-stage evaluation, OOD construction, and baseline scoring.

It intentionally does not download private/large benchmark files by itself.
Provide official annotations/manifests through the command line and use
--dry_run to inspect the exact commands before launching long jobs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmark_datasets import load_benchmark


def run(cmd, dry_run: bool = False):
    print(" ".join(str(x) for x in cmd))
    if not dry_run:
        subprocess.run([str(x) for x in cmd], check=True)


def validate_manifest(dataset: str, manifest_path: str | None, data_root: str | None, output_dir: Path):
    samples, factors = load_benchmark(dataset, manifest_path, data_root)
    summary = {
        "dataset": dataset,
        "num_samples": len(samples),
        "num_factors": len(factors),
        "splits": sorted({s.split for s in samples}),
        "question_types": sorted({s.question_type for s in samples}),
        "has_counterfactual_answers": sum(bool(s.cf_answer) for s in samples),
        "has_counterfactual_media": sum(bool(s.counterfactual_image_path or s.counterfactual_video_path) for s in samples),
        "has_gold_edges": sum(bool(s.graph_edges) for s in samples),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "manifest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def latest_checkpoint(root: Path) -> Path | None:
    candidates = list(root.glob("**/best_model.pth")) + list(root.glob("**/final_model.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Run paper-style MetaCausalField experiment stages")
    parser.add_argument("--dataset", required=True, choices=["MAG9", "Lung", "CLEVRER", "Causal3DIdent", "CITRIS", "Causal-VidQA"])
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_dir", default="./paper_runs")
    parser.add_argument("--gold_graph", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", default="all", choices=["validate", "features", "train", "eval", "ood", "baseline", "all"])
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--backbone", default="cached", choices=["cached", "resnet", "qwen"])
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument("--extract_backbone", default="qwen", choices=["qwen", "api", "resnet"])
    parser.add_argument("--qwen_model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--download_qwen", action="store_true")
    parser.add_argument("--qwen_cache_dir", default=None)
    parser.add_argument("--qwen_video_input", default="native", choices=["native", "middle_frame"])
    parser.add_argument("--train_qwen_lora", action="store_true")
    parser.add_argument("--qwen_lora_r", type=int, default=8)
    parser.add_argument("--qwen_lora_alpha", type=int, default=16)
    parser.add_argument("--qwen_lora_dropout", type=float, default=0.05)
    parser.add_argument("--qwen_lora_targets", default=None)
    parser.add_argument("--feature_dim", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_propagation_steps", type=int, default=3)
    parser.add_argument("--influence_top_k", type=int, default=None)
    parser.add_argument("--enable_cf_trajectory_loss", action="store_true")
    parser.add_argument("--use_frozen_language_tokens", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline_backend", default="majority", choices=["majority", "random", "gemini", "hf"])
    parser.add_argument("--baseline_model", default=None)
    args = parser.parse_args()

    root = Path(args.output_dir) / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    feature_cache = Path(args.feature_cache) if args.feature_cache else root / f"{args.extract_backbone}_features.pt"

    if args.mode in {"validate", "all"}:
        validate_manifest(args.dataset, args.manifest_path, args.data_root, root)

    needs_feature_cache = args.backbone == "cached" or (args.backbone == "qwen" and args.dataset in {"MAG9", "Lung"})
    if args.mode in {"features", "all"} and needs_feature_cache:
        cmd = [
            sys.executable, "extract_frozen_features.py",
            "--dataset", args.dataset,
            "--backbone", args.extract_backbone,
            "--output", feature_cache,
            "--feature_dim", args.feature_dim,
            "--device", args.device,
        ]
        if args.manifest_path:
            cmd.extend(["--manifest_path", args.manifest_path])
        if args.data_root:
            cmd.extend(["--data_root", args.data_root])
        if args.extract_backbone == "qwen":
            cmd.extend(["--qwen_model", args.qwen_model])
            cmd.extend(["--qwen_video_input", args.qwen_video_input])
            if args.download_qwen:
                cmd.append("--download_qwen")
            if args.qwen_cache_dir:
                cmd.extend(["--qwen_cache_dir", args.qwen_cache_dir])
        run(cmd, args.dry_run)

    checkpoint_hint = root / "training"
    if args.mode in {"train", "all"}:
        if args.dataset in {"MAG9", "Lung"}:
            stage_backbone = "cached" if args.backbone == "qwen" else args.backbone
            cmd = [
                sys.executable, "train_three_stage_metacausal.py",
                "--dataset", args.dataset,
                "--output_dir", checkpoint_hint,
                "--epochs_per_stage", args.epochs,
                "--batch_size", args.batch_size,
                "--feature_dim", args.feature_dim,
                "--num_heads", args.num_heads,
                "--num_propagation_steps", args.num_propagation_steps,
                "--backbone", stage_backbone,
            ]
            if stage_backbone == "cached":
                cmd.extend(["--feature_cache", feature_cache])
            if args.use_frozen_language_tokens:
                cmd.append("--use_frozen_language_tokens")
            if args.influence_top_k is not None:
                cmd.extend(["--influence_top_k", args.influence_top_k])
        else:
            cmd = [
                sys.executable, "train_benchmark_metacausal.py",
                "--dataset", args.dataset,
                "--manifest_path", args.manifest_path,
                "--output_dir", checkpoint_hint,
                "--epochs", args.epochs,
                "--batch_size", args.batch_size,
                "--feature_dim", args.feature_dim,
                "--num_heads", args.num_heads,
                "--num_propagation_steps", args.num_propagation_steps,
                "--backbone", args.backbone,
                "--device", args.device,
            ]
            if args.data_root:
                cmd.extend(["--data_root", args.data_root])
            if needs_feature_cache:
                cmd.extend(["--feature_cache", feature_cache])
            if args.backbone == "qwen":
                cmd.extend(["--qwen_model", args.qwen_model])
                cmd.extend(["--qwen_video_input", args.qwen_video_input])
                if args.download_qwen:
                    cmd.append("--download_qwen")
                if args.qwen_cache_dir:
                    cmd.extend(["--qwen_cache_dir", args.qwen_cache_dir])
                if args.train_qwen_lora:
                    cmd.append("--train_qwen_lora")
                    cmd.extend(["--qwen_lora_r", args.qwen_lora_r])
                    cmd.extend(["--qwen_lora_alpha", args.qwen_lora_alpha])
                    cmd.extend(["--qwen_lora_dropout", args.qwen_lora_dropout])
                    if args.qwen_lora_targets:
                        cmd.extend(["--qwen_lora_targets", args.qwen_lora_targets])
            if args.use_frozen_language_tokens:
                cmd.append("--use_frozen_language_tokens")
            if args.enable_cf_trajectory_loss:
                cmd.append("--enable_cf_trajectory_loss")
            if args.influence_top_k is not None:
                cmd.extend(["--influence_top_k", args.influence_top_k])
        run(cmd, args.dry_run)

    if args.mode in {"ood", "all"} and args.manifest_path:
        ood_manifest = root / "ood_manifest.jsonl"
        run([
            sys.executable, "build_ood_splits.py",
            "--manifest_path", args.manifest_path,
            "--output", ood_manifest,
            *([] if not args.data_root else ["--data_root", args.data_root]),
        ], args.dry_run)

    if args.mode in {"eval", "all"}:
        checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(checkpoint_hint)
        if checkpoint is None:
            print("Evaluation skipped: no checkpoint found yet. Pass --checkpoint <best_model.pth> after training.")
        else:
            cmd = [
                sys.executable, "three_stage_metacausal_pipeline.py",
                "--dataset", args.dataset,
                "--checkpoint", checkpoint,
                "--output_dir", root / "eval",
                "--batch_size", args.batch_size,
                "--device", args.device,
            ]
            if args.manifest_path:
                cmd.extend(["--manifest_path", args.manifest_path])
            if args.data_root:
                cmd.extend(["--data_root", args.data_root])
            if args.gold_graph:
                cmd.extend(["--gold_graph", args.gold_graph])
            if args.backbone == "cached":
                cmd.extend(["--feature_cache", feature_cache])
            if args.use_frozen_language_tokens:
                cmd.append("--use_frozen_language_tokens")
            run(cmd, args.dry_run)

    if args.mode in {"baseline", "all"} and args.manifest_path:
        cmd = [
            sys.executable, "run_baselines.py",
            "--dataset", args.dataset,
            "--manifest_path", args.manifest_path,
            "--backend", args.baseline_backend,
            "--output", root / f"baseline_{args.baseline_backend}.jsonl",
        ]
        if args.data_root:
            cmd.extend(["--data_root", args.data_root])
        if args.baseline_model:
            cmd.extend(["--model", args.baseline_model])
        run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
