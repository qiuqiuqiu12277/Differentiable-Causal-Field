"""Reproduce MAG/Lung causal discovery tables from pipeline outputs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_dataset(dataset: str, checkpoint: str, output_dir: Path, device: str, batch_size: int):
    ds_dir = output_dir / dataset
    cmd = [
        sys.executable,
        "three_stage_metacausal_pipeline.py",
        "--dataset",
        dataset,
        "--checkpoint",
        checkpoint,
        "--output_dir",
        str(ds_dir),
        "--device",
        device,
        "--batch_size",
        str(batch_size),
    ]
    subprocess.run(cmd, check=True)
    with open(ds_dir / "three_stage_results.json") as f:
        return json.load(f)


def flatten_row(dataset: str, method: str, metrics: dict):
    keys = ["NP", "NR", "NF", "AP", "AR", "AF", "ESHD"]
    return {"Dataset": dataset, "Method": method, **{key: metrics.get(key) for key in keys}}


def main():
    parser = argparse.ArgumentParser(description="Reproduce MAG/Lung tables")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="./paper_table_outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in ["MAG9", "Lung"]:
        result = run_dataset(dataset, args.checkpoint, output_dir, args.device, args.batch_size)
        rows.append(flatten_row(dataset, "FCI", result["structure_learning"]["metrics_fci"]))
        if "metrics_field" in result["structure_learning"]:
            rows.append(flatten_row(dataset, "MetaCausalField", result["structure_learning"]["metrics_field"]))

    table = pd.DataFrame(rows)
    csv_path = output_dir / "mag_lung_causal_discovery.csv"
    tex_path = output_dir / "mag_lung_causal_discovery.tex"
    table.to_csv(csv_path, index=False)
    table.to_latex(tex_path, index=False, float_format="%.4f")
    print(table.to_string(index=False))
    print(f"Saved {csv_path} and {tex_path}")


if __name__ == "__main__":
    main()
