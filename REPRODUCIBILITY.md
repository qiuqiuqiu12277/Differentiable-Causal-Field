# Reproducibility status

This document distinguishes code that can be verified from this repository
from experiments that require assets not included in the public tree. It is a
status map, not a claim that missing paper experiments have been reproduced.

## Reproduction levels

### Level 1: local smoke verification

Available without external datasets or model weights:

```bash
python -m pip install -e '.[data,dev]'
pytest
python -m experiments.synthetic_identifiability --quick \
  --output-dir outputs/synthetic-smoke
```

This level checks tensor contracts, loss gradients, deterministic utilities,
and a constructive linear identifiability example. It does not validate
multimodal accuracy.

### Level 2: external-data prototype

The manifest adapters can train the local field model on externally obtained
CLEVRER, CITRIS/Causal3DIdent, or CausalVQA-style assets. A Level 2 result must
include:

- the exact manifest and its SHA-256 digest;
- explicit train/validation/test membership;
- media validation with zero silent fallbacks;
- resolved configuration and random seed;
- checkpoint and raw-prediction digests;
- free-running answer generation for generative metrics.

Level 2 is not automatically equivalent to the paper configuration.

### Level 3: paper-table reproduction

A full paper reproduction additionally requires the paper's Qwen3-VL backbone
configuration, staged CLEVRER/CITRIS/Causal3D training, paired environment
masks, simulator-produced post-intervention trajectories, EMA target encoder,
frozen OOD manifests, capacity-matched controls, and all reported random seeds.

Those assets and complete scripts are not all present in this release. Until
they are added, `run_paper_experiments.py` is an orchestration helper rather
than evidence that Tables 1-15 can be reproduced.

## Paper-to-code map

| Paper element | Current artifact | Status |
|---|---|---|
| Gaussian field construction | `GaussianInterpolation` | Implemented prototype |
| Directed multi-step transition | `DirectionalInfluenceFunction`, `CausalPropagation` | Implemented prototype |
| Local field intervention | `InterventionModule` | Implemented prototype |
| Target-aligned simulator trajectory loss | benchmark training utilities | Partial; exact paper pipeline not bundled |
| Cross-environment foreground-masked invariance | benchmark training utilities | Partial |
| Linear intervention-coverage analysis | `experiments/synthetic_identifiability.py` | Constructive smoke diagnostic |
| Nonlinear theorem assumptions | no executable verifier | Theoretical assumptions, not guaranteed by architecture |
| CausalVQA/NExT-QA paper results | no released checkpoints/raw predictions | Not reproduced here |
| MAG/Lung comparison | legacy adapters and result helpers | Partial; provenance caveats apply |

The default Gaussian bandwidth is interpreted in field-grid units even though
coordinates are stored in `[0,1]`. Checkpoints also record exact split IDs and
whether factor-localization/graph heads received direct supervision. If that
supervision flag is false, named causal-graph metrics are reported as
unavailable rather than computed from a random localization head.

## Required artifacts for reported results

Every future result directory should contain:

```text
run/
|-- config.json
|-- environment.txt
|-- manifest.sha256
|-- checkpoint.sha256
|-- metrics.json
|-- predictions.jsonl
`-- seed.json
```

Numbers should enter the README only after the directory can be regenerated
from a documented command and independently scored from `predictions.jsonl`.
