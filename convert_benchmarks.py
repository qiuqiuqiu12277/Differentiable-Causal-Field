"""Convert native benchmark annotations into the unified MetaCausalField manifest.

The loaders accept common CLEVRER, Causal3DIdent/CITRIS, and Causal-VidQA
layouts and emit JSONL records with image/video, QA, intervention, factor, and
gold-graph fields consumed by the training/evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from benchmark_datasets import (
    load_causal3d_citris,
    load_causal_vidqa,
    load_clevrer,
    samples_from_manifest,
)


LOADERS = {
    "CLEVRER": load_clevrer,
    "Causal3DIdent": load_causal3d_citris,
    "CITRIS": load_causal3d_citris,
    "Causal-VidQA": load_causal_vidqa,
    "manifest": samples_from_manifest,
}


def sample_to_record(sample):
    record = asdict(sample)
    record["graph_edges"] = [{"source": s, "target": t} for s, t in sample.graph_edges]
    return record


def main():
    parser = argparse.ArgumentParser(description="Convert benchmark annotations to unified JSONL manifest")
    parser.add_argument("--benchmark", required=True, choices=sorted(LOADERS))
    parser.add_argument("--input", required=True, help="Native annotation file or existing manifest")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samples = LOADERS[args.benchmark](args.input, args.data_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample_to_record(sample), ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} samples to {output}")


if __name__ == "__main__":
    main()
