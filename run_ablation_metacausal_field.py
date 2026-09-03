"""Run standard MetaCausalField ablations for a trained checkpoint."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ABLATIONS = {
    "full": [],
    "no_language": ["--disable_language"],
    "no_lm": ["--disable_lm"],
    "no_counterfactual_eval": ["--disable_counterfactual"],
    "no_propagation": ["--disable_propagation"],
}


def main():
    parser = argparse.ArgumentParser(description="Run MetaCausalField ablations")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["Lung", "MAG9"], default=None)
    parser.add_argument("--output_dir", default="./ablation_results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_counterfactual_pairs", type=int, default=None)
    parser.add_argument("--feature_cache", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name, flags in ABLATIONS.items():
        output = output_dir / f"{name}.json"
        cmd = [
            sys.executable,
            "evaluate_metacausal_field.py",
            "--checkpoint",
            args.checkpoint,
            "--output",
            str(output),
            "--device",
            args.device,
            "--batch_size",
            str(args.batch_size),
        ]
        if args.dataset:
            cmd.extend(["--dataset", args.dataset])
        if args.max_counterfactual_pairs:
            cmd.extend(["--max_counterfactual_pairs", str(args.max_counterfactual_pairs)])
        if args.feature_cache:
            cmd.extend(["--feature_cache", args.feature_cache])
        cmd.extend(flags)
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        with open(output) as f:
            summary[name] = json.load(f)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
