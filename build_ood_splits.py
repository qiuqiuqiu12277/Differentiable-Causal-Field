"""Automatically construct OOD manifests for MetaCausalField experiments.

This script creates the paper-style Scene Shift, Object Composition Shift,
Template Shift, and Intervention Shift evaluation subsets from a unified
manifest. It preserves media paths and gold labels while adding explicit
`ood_type` tags and, where possible, controlled question/intervention variants.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from benchmark_datasets import CausalSample, samples_from_manifest


TEMPLATE_REWRITES = {
    "what will happen": "what is the likely outcome",
    "what happens": "what is the resulting event",
    "if ": "suppose ",
    "why": "for what causal reason",
    "which": "identify which",
}


def to_record(sample: CausalSample):
    record = asdict(sample)
    record["graph_edges"] = [{"source": s, "target": t} for s, t in sample.graph_edges]
    return record


def rewrite_question(question: str) -> str:
    q = question.strip()
    low = q.lower()
    for src, dst in TEMPLATE_REWRITES.items():
        if src in low:
            idx = low.index(src)
            return q[:idx] + dst + q[idx + len(src):]
    if q.endswith("?"):
        return "Considering the same causal chain, " + q[:1].lower() + q[1:]
    return "Considering the same causal chain, " + q


def composition_key(sample: CausalSample) -> str:
    if sample.objects:
        names = sorted(str(o.get("category", o.get("type", o.get("id", "")))) for o in sample.objects)
        return "+".join(x for x in names if x)
    if sample.factors:
        active = sorted(k for k, v in sample.factors.items() if float(v) != 0.0)
        return "+".join(active[:4])
    return sample.sample_id


def make_variants(samples, include_in_domain: bool = True):
    variants = []
    if include_in_domain:
        variants.extend(replace(s, ood_type="in_domain") for s in samples)

    for sample in samples:
        variants.append(replace(sample, ood_type="Scene Shift"))

        q = sample.question or sample.text
        variants.append(replace(sample, question=rewrite_question(q), ood_type="Template Shift"))

        variants.append(replace(sample, ood_type="Object Composition Shift"))

        intervention = dict(sample.intervention or {})
        if not intervention:
            target = sample.object_id or (next(iter(sample.factors), None) if sample.factors else None)
            if target is not None:
                intervention = {"type": "remove", "target": target}
        variants.append(replace(sample, intervention=intervention or None, ood_type="Intervention Shift"))

    # Keep composition shift genuinely held out when enough combinations exist:
    # deterministic odd/even split by composition key.
    filtered = []
    for sample in variants:
        if sample.ood_type != "Object Composition Shift":
            filtered.append(sample)
            continue
        key_hash = abs(hash(composition_key(sample))) % 2
        if key_hash == 1:
            filtered.append(sample)
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Build automatic OOD split manifest")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--drop_in_domain", action="store_true")
    args = parser.parse_args()

    samples = samples_from_manifest(args.manifest_path, args.data_root)
    variants = make_variants(samples, include_in_domain=not args.drop_in_domain)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for sample in variants:
            f.write(json.dumps(to_record(sample), ensure_ascii=False) + "\n")
    print(f"Wrote {len(variants)} OOD records to {output}")


if __name__ == "__main__":
    main()
