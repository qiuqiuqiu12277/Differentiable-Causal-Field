"""Run a paper-style baseline matrix over unified benchmark manifests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_HF_MODELS = {
    "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava-onevision": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "internvl25": "OpenGVLab/InternVL2_5-8B",
}

DEFAULT_GEMINI_MODELS = {
    "gemini-2.5-flash": "gemini-2.5-flash",
}


def run(cmd, dry_run: bool):
    print(" ".join(str(x) for x in cmd))
    if not dry_run:
        subprocess.run([str(x) for x in cmd], check=True)


def parse_model_specs(items):
    specs = []
    for item in items or []:
        if ":" in item:
            name, model = item.split(":", 1)
        else:
            name, model = item, item
        specs.append((name.strip(), model.strip()))
    return specs


def main():
    parser = argparse.ArgumentParser(description="Run baseline model matrix")
    parser.add_argument("--dataset", required=True, choices=["MAG", "MAG9", "Lung", "Lung4", "CLEVRER", "Causal3DIdent", "CITRIS", "Causal-VidQA"])
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_dir", default="./baseline_matrix")
    parser.add_argument("--include_majority", action="store_true")
    parser.add_argument("--include_random", action="store_true")
    parser.add_argument("--hf_model", action="append", default=[], help="name:model_id or model_id")
    parser.add_argument("--gemini_model", action="append", default=[], help="name:model_name or model_name")
    parser.add_argument("--default_paper_models", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.include_majority:
        jobs.append(("majority", "majority", None))
    if args.include_random:
        jobs.append(("random", "random", None))

    hf_specs = parse_model_specs(args.hf_model)
    gemini_specs = parse_model_specs(args.gemini_model)
    if args.default_paper_models:
        hf_specs.extend(DEFAULT_HF_MODELS.items())
        gemini_specs.extend(DEFAULT_GEMINI_MODELS.items())

    jobs.extend((name, "hf", model) for name, model in hf_specs)
    jobs.extend((name, "gemini", model) for name, model in gemini_specs)

    summary = []
    for name, backend, model in jobs:
        safe_name = name.replace("/", "__").replace(":", "_")
        output = output_dir / f"{safe_name}.jsonl"
        cmd = [
            sys.executable,
            "run_baselines.py",
            "--dataset",
            args.dataset,
            "--backend",
            backend,
            "--output",
            output,
            "--device",
            args.device,
            "--max_new_tokens",
            args.max_new_tokens,
        ]
        if args.manifest_path:
            cmd.extend(["--manifest_path", args.manifest_path])
        if args.data_root:
            cmd.extend(["--data_root", args.data_root])
        if model:
            cmd.extend(["--model", model])
        run(cmd, args.dry_run)
        summary.append({
            "name": name,
            "backend": backend,
            "model": model,
            "predictions": str(output),
            "metrics": str(output.with_suffix(".metrics.json")),
        })

    with open(output_dir / "baseline_matrix_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
