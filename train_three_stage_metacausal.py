"""Run the paper-style three-stage MetaCausalField training schedule.

The stages are:
1. factor_discovery: language/score/factor grounding, no CI/CF losses.
2. structure_learning: add cross-environment/feature-noise causal consistency.
3. counterfactual_reasoning: add counterfactual rollout supervision.

Each stage initializes from the previous stage checkpoint and writes its own
training directory under the chosen output root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def latest_checkpoint(stage_dir: Path) -> Path:
    runs = sorted(stage_dir.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No run directory found under {stage_dir}")
    for name in ("best_model.pth", "final_model.pth"):
        path = runs[-1] / name
        if path.exists():
            return path
    checkpoints = sorted(runs[-1].glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under {runs[-1]}")
    return checkpoints[-1]


def run_stage(stage_name: str, base_args, extra_flags, init_checkpoint=None):
    stage_dir = Path(base_args.output_dir) / stage_name
    cmd = [
        sys.executable,
        "train_metacausal_field.py",
        "--dataset",
        base_args.dataset,
        "--output_dir",
        str(stage_dir),
        "--epochs",
        str(base_args.epochs_per_stage),
        "--batch_size",
        str(base_args.batch_size),
        "--feature_dim",
        str(base_args.feature_dim),
        "--num_heads",
        str(base_args.num_heads),
        "--num_propagation_steps",
        str(base_args.num_propagation_steps),
        "--learning_rate",
        str(base_args.learning_rate),
        "--lambda_sparsity",
        str(base_args.lambda_sparsity),
        "--lambda_smoothness",
        str(base_args.lambda_smoothness),
    ]
    if base_args.influence_top_k is not None:
        cmd.extend(["--influence_top_k", str(base_args.influence_top_k)])
    if base_args.backbone:
        cmd.extend(["--backbone", base_args.backbone])
    if base_args.feature_cache:
        cmd.extend(["--feature_cache", base_args.feature_cache])
    if base_args.use_frozen_language_tokens:
        cmd.append("--use_frozen_language_tokens")
    if base_args.freeze_backbone:
        cmd.append("--freeze_backbone")
    if base_args.disable_lm:
        cmd.append("--disable_lm")
    if init_checkpoint:
        cmd.extend(["--init_checkpoint", str(init_checkpoint)])
    cmd.extend(extra_flags)

    print("Running", stage_name, " ".join(cmd))
    subprocess.run(cmd, check=True)
    return latest_checkpoint(stage_dir)


def main():
    parser = argparse.ArgumentParser(description="Three-stage MetaCausalField trainer")
    parser.add_argument("--dataset", default="MAG9", choices=["MAG9", "Lung"])
    parser.add_argument("--output_dir", default="./three_stage_training")
    parser.add_argument("--epochs_per_stage", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_propagation_steps", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lambda_sparsity", type=float, default=0.01)
    parser.add_argument("--lambda_smoothness", type=float, default=0.001)
    parser.add_argument("--influence_top_k", type=int, default=None)
    parser.add_argument("--lambda_consistency", type=float, default=0.3)
    parser.add_argument("--lambda_counterfactual", type=float, default=0.5)
    parser.add_argument("--backbone", default="resnet", choices=["resnet", "cached"])
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument("--use_frozen_language_tokens", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--disable_lm", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt1 = run_stage(
        "factor_discovery",
        args,
        ["--disable_consistency", "--disable_counterfactual", "--lambda_consistency", "0.0", "--lambda_counterfactual", "0.0"],
    )
    ckpt2 = run_stage(
        "structure_learning",
        args,
        ["--disable_counterfactual", "--lambda_consistency", str(args.lambda_consistency), "--lambda_counterfactual", "0.0"],
        init_checkpoint=ckpt1,
    )
    ckpt3 = run_stage(
        "counterfactual_reasoning",
        args,
        ["--lambda_consistency", str(args.lambda_consistency), "--lambda_counterfactual", str(args.lambda_counterfactual)],
        init_checkpoint=ckpt2,
    )

    summary = {
        "factor_discovery_checkpoint": str(ckpt1),
        "structure_learning_checkpoint": str(ckpt2),
        "counterfactual_reasoning_checkpoint": str(ckpt3),
    }
    with open(out / "three_stage_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
