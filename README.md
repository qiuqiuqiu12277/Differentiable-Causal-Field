# InfluenceField

[![arXiv](https://img.shields.io/badge/arXiv-2609.07874-b31b1b.svg)](https://arxiv.org/abs/2609.07874)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![tests](https://github.com/qiuqiuqiu12277/Differentiable-Causal-Field/actions/workflows/tests.yml/badge.svg)](https://github.com/qiuqiuqiu12277/Differentiable-Causal-Field/actions/workflows/tests.yml)

Research code for **InfluenceField: A Differentiable Field with
Interventionally Identifiable Causal Structure for Multimodal World
Modeling**.

InfluenceField inserts an intervention-aware latent field between a visual
encoder and a language decoder. Visual patch features are lifted to a spatial
field, a state-dependent directed operator propagates influence, and localized
field edits are rolled out with the same transition operator.

> **Release status.** This repository currently contains the core field
> prototype, data adapters, evaluation utilities, and a small theorem-aligned
> diagnostic. It is **not yet a self-contained reproduction package for every
> number in the paper**: official datasets, the complete Qwen3-VL training
> stack, paper checkpoints, raw predictions, and full split manifests are not
> bundled. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for an exact status map.

## Scientific scope

The paper's causal result concerns the dependency graph of the **complete
transition operator** under target-aligned interventions and explicit coverage,
regularity, and separation assumptions. Raw attention or influence weights are
not causal merely because they are asymmetric or sparse. The included
synthetic diagnostic makes this distinction executable; MAG/Lung and local
image utilities should be treated as research adapters rather than independent
proof of identifiability.

## Repository status

| Component | Status | Notes |
|---|---:|---|
| Continuous Gaussian field and multi-step propagation | Available | `metacausal_field.py` |
| Local field interventions and counterfactual rollout | Available | Prototype interface |
| Stable, CPU-only identifiability diagnostic | Available | Emits JSON and CSV; not a paper-table reproduction |
| MAG/Lung adapters and graph metrics | Available | Includes legacy MLLM-CD-derived material; see notices |
| CLEVRER/CITRIS/Causal3D/CausalVQA manifest adapters | Partial | Requires official external assets |
| Full three-stage Qwen3-VL paper training | Not bundled | Requires paper-scale data, weights, and compute |
| Paper checkpoints, raw predictions, and frozen manifests | Not bundled | Do not infer paper results from smoke tests |

## Installation

Create an isolated Python 3.10+ environment, then install the local package:

```bash
python -m pip install -e '.[data,dev]'
```

Optional benchmark and MLLM integrations:

```bash
python -m pip install -e '.[benchmarks]'
```

The benchmark extra is intentionally separate because it downloads a much
larger dependency stack. Model weights and datasets are never downloaded by the
unit tests.

## Verified quick start

Run the unit tests:

```bash
pytest
```

Run the quick theorem-aligned diagnostic:

```bash
python -m experiments.synthetic_identifiability \
  --quick \
  --output-dir outputs/synthetic-smoke
```

The command constructs observationally equivalent linear transitions under
0/25/50/100% known-site intervention coverage. It writes:

- `metrics.json`: configuration, scope statement, and aggregate metrics;
- `runs.csv`: seed-level recovery and residual measurements.

At full coverage the constructed ambiguity is restricted to coordinate-wise
scaling and edge support is recovered exactly. At partial coverage only the
covered subgraph is asserted to be preserved. This is a constructive diagnostic
of the theorem's mechanism, not evidence for the full nonlinear multimodal
model.

## Local prototype training

MAG/Lung training remains available for development:

```bash
python train_metacausal_field.py \
  --dataset Lung \
  --epochs 20 \
  --batch_size 8 \
  --feature_dim 256 \
  --num_heads 4 \
  --num_propagation_steps 3
```

The corresponding image directories are not included. Missing media now fail
closed unless a script explicitly enables a synthetic fallback. A successful
run on placeholder images must never be reported as a benchmark result.

For manifest-backed datasets:

```bash
python train_benchmark_metacausal.py \
  --dataset Causal-VidQA \
  --manifest_path /path/to/manifest.jsonl \
  --data_root /path/to/dataset \
  --epochs 5 \
  --batch_size 4
```

The manifest must contain non-empty `train`, `val`, and `test` partitions.
Shared factual or counterfactual media may not cross partitions. The
CausalVQA converter can create deterministic media-group splits:

```bash
python prepare_causalvqa_manifest.py \
  --input /path/to/annotations.json \
  --data_root /path/to/media \
  --output /path/to/manifest.jsonl \
  --seed 42
```

## Integrity guards

- Generative QA metrics use free-running decoding from `<bos>`; teacher-forced
  values are labelled as perplexity only.
- Evaluation restores exact checkpoint split IDs when available and defaults
  to the held-out test partition.
- Missing media, mismatched datasets/backbones, partial checkpoints, and
  duplicate feature-cache keys fail closed.
- Counterfactual supervision accepts only explicit, supported latent edits;
  scenario-name hashes are limited to labelled smoke utilities.
- Unsupervised factor-localization heads are not reported as causal graphs.
- FCI failures are labelled as a correlation smoke fallback, never as FCI.
- OOD evaluation accepts only evidence-backed test/eval/ood records; synthetic
  template paraphrases are opt-in and excluded from paper-style aggregates.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before comparing any local output
with a paper table.

## Main files

- `metacausal_field.py` - field construction, directed propagation,
  interventions, task heads, and losses.
- `experiments/synthetic_identifiability.py` - auditable linear coverage
  diagnostic.
- `train_metacausal_field.py` - local MAG/Lung prototype trainer.
- `train_benchmark_metacausal.py` - unified manifest trainer.
- `benchmark_datasets.py` - dataset schema and media loading.
- `three_stage_metacausal_pipeline.py` - staged structure/QA/CF evaluation.
- `causal_metrics.py` - graph, QA, OOD, and counterfactual metrics.
- `prepare_causalvqa_manifest.py` - annotation-to-manifest conversion.

## Reproducibility rules

1. Use explicit train/validation/test splits and select checkpoints on
   validation only.
2. Record seeds, resolved configuration, input-manifest checksum, checkpoint
   checksum, and raw predictions.
3. Do not call a raw attention matrix an identified causal graph. Evaluate the
   complete transition Jacobian or an explicitly justified surrogate.
4. Do not report paper-table numbers from generated placeholders, teacher-forced
   decoding, or metadata-only "OOD" variants.
5. Label smoke tests, partial reproductions, and full reproductions separately.

## Paper

- [arXiv:2609.07874](https://arxiv.org/abs/2609.07874)

```bibtex
@article{yang2026influencefield,
  title   = {InfluenceField: A Differentiable Field with Interventionally
             Identifiable Causal Structure for Multimodal World Modeling},
  author  = {Yang, Zihao and Wang, Zijia and Huang, Zhiqiu},
  journal = {arXiv preprint arXiv:2609.07874},
  year    = {2026}
}
```

## Provenance and licensing

This repository contains legacy files and data derived from the MLLM-CD
research release. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before
redistributing them. No repository-wide open-source license is asserted here;
see [LICENSE_STATUS.md](LICENSE_STATUS.md).
