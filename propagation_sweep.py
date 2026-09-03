"""Propagation-step K sweep for performance/latency trade-off."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from evaluate_metacausal_field import load_model
from train_metacausal_field import prepare_metadata


def measure_latency(checkpoint: str, k: int, device: str, repeats: int = 20):
    torch_device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    model, _, _, _, _ = load_model(checkpoint, torch_device)
    model.propagation.num_steps = k
    model.eval()
    dummy = torch.randn(1, 49, model.config.feature_dim, device=torch_device)
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(dummy)
            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser(description="Sweep propagation steps")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="Lung", choices=["Lung", "MAG9"])
    parser.add_argument("--output_dir", default="./propagation_sweep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k_values", default="0,1,2,3,4")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--feature_cache", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for k in [int(x) for x in args.k_values.split(",") if x.strip()]:
        output = output_dir / f"k_{k}.json"
        cmd = [
            sys.executable,
            "evaluate_metacausal_field.py",
            "--checkpoint",
            args.checkpoint,
            "--dataset",
            args.dataset,
            "--output",
            str(output),
            "--device",
            args.device,
            "--batch_size",
            str(args.batch_size),
        ]
        if args.feature_cache:
            cmd.extend(["--feature_cache", args.feature_cache])
        if k == 0:
            cmd.append("--disable_propagation")
        subprocess.run(cmd, check=True)
        with open(output) as f:
            metrics = json.load(f)
        metrics["K"] = k
        metrics["latency_mean"] = measure_latency(args.checkpoint, k, args.device)
        summary[f"K={k}"] = metrics

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
