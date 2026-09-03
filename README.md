# MetaCausalField

This repository contains code for multimodal causal discovery and the
MetaCausalField research prototype. It includes the original MLLM-CD style
pipeline for MAG/Lung causal discovery, plus a differentiable causal field
implementation for intervention-aware multimodal reasoning.

MetaCausalField follows the paper idea of placing a differentiable causal field
between visual encoding and language-conditioned decoding. Visual patch or
frozen MLLM tokens are lifted into a continuous spatial field, propagated with
learned directional influence, and optionally intervened on for counterfactual
rollout.

## What Is Implemented

- Continuous Gaussian field construction from image/video/frozen visual tokens.
- Dynamic directional influence and multi-step causal propagation.
- Field-level interventions for object removal, attribute modification, and
  custom latent edits.
- Factual and counterfactual rollout trajectories with shared transition
  dynamics.
- Language-conditioned field fusion and lightweight text/score/factor heads.
- Factor-level spatial attention and factor-level influence aggregation for
  named causal graph evaluation.
- MAG9 and Lung training/evaluation.
- Unified manifest loaders for CLEVRER, Causal3DIdent/CITRIS, and Causal-VidQA.
- Frozen feature extraction for ResNet, Qwen/Qwen3-VL style models, and closed
  API responses.
- Three-stage paper-style training and evaluation utilities.
- OOD split construction, baseline runners, propagation sweeps, ablations, and
  interpretability metrics.

Large official benchmark assets, pretrained Qwen weights, API credentials, and
paper-table checkpoints are not bundled. To reproduce final numbers, provide the
official datasets and run the experiment commands below.

## Main Files

- `metacausal_field.py`: Core MetaCausalField model, losses, interventions, and
  visualization helpers.
- `train_metacausal_field.py`: MAG9/Lung training.
- `train_benchmark_metacausal.py`: Unified benchmark manifest training.
- `train_three_stage_metacausal.py`: Three-stage MAG9/Lung trainer.
- `three_stage_metacausal_pipeline.py`: Factor discovery, structure evaluation,
  QA, OOD, and counterfactual evaluation.
- `extract_frozen_features.py`: Frozen ResNet/Qwen/API feature cache extraction.
- `run_paper_experiments.py`: Orchestrates validate/features/train/eval/OOD and
  baseline stages.
- `prepare_causalvqa_manifest.py`: Converts common CausalVQA/Causal-VidQA
  annotation layouts into the unified JSONL schema.
- `run_baseline_matrix.py`: Runs batches of majority/random/HF/Gemini baselines.
- `benchmark_datasets.py`: Unified dataset schema and adapters.
- `causal_metrics.py`: Graph, QA, OOD, counterfactual, and interpretability
  metrics.
- `main_MAG.py`, `main_Lung.py`, `utils.py`: Original MLLM-CD style MAG/Lung
  pipeline.

## Data

Bundled tabular files:

- `MAG9.csv`
- `Lung.csv`
- `gold_graphs/mag9_gold_graph.csv`
- `gold_graphs/lung_gold_graph.csv`
- generated counterfactual CSVs under `results/`

Expected local image folders:

- `apple_images_a9/`
- `Lung/`

For external benchmarks, use a JSONL/JSON/CSV manifest. A typical JSONL record:

```json
{
  "sample_id": "example-1",
  "split": "test",
  "image_path": "frames/example.png",
  "video_path": "videos/example.mp4",
  "question": "What happens if the red object is removed?",
  "answer": "The blue object will not move.",
  "question_type": "Counterfactual",
  "factors": {"red_object": 1, "blue_object": 1},
  "graph_edges": [["red_object", "blue_object"]],
  "intervention": {"target": "red_object", "type": "remove"},
  "cf_answer": "The blue object will not move.",
  "counterfactual_image_path": "frames/example_cf.png",
  "ood_type": "Intervention Shift",
  "object_id": "red_object",
  "bbox": [0.1, 0.2, 0.3, 0.4]
}
```

## Installation

Use Python 3.10+ when possible.

```bash
pip install torch torchvision torchaudio
pip install numpy pandas pillow matplotlib tqdm scikit-learn pydot causal-learn
pip install transformers huggingface_hub
```

Closed API feature extraction also requires the relevant API environment
variables used by `gemini_utils.py`.

## Quick Checks

```bash
python -m py_compile \
  metacausal_field.py benchmark_datasets.py frozen_backbones.py \
  extract_frozen_features.py train_metacausal_field.py \
  train_benchmark_metacausal.py train_three_stage_metacausal.py \
  evaluate_metacausal_field.py evaluate_interpretability.py \
  three_stage_metacausal_pipeline.py run_paper_experiments.py \
  prepare_causalvqa_manifest.py run_baseline_matrix.py \
  causal_metrics.py test_metacausal_field.py

python test_metacausal_field.py
python run_paper_experiments.py --dataset MAG9 --mode validate --output_dir /tmp/mcf_validate
```

## MAG9/Lung Training

Train the local ResNet-backed model:

```bash
python train_metacausal_field.py \
  --dataset Lung \
  --epochs 20 \
  --batch_size 8 \
  --feature_dim 256 \
  --num_heads 4 \
  --num_propagation_steps 3
```

Run the paper-style staged schedule:

```bash
python train_three_stage_metacausal.py \
  --dataset MAG9 \
  --epochs_per_stage 3 \
  --batch_size 8 \
  --feature_dim 256 \
  --num_heads 4 \
  --num_propagation_steps 3 \
  --output_dir three_stage_training
```

Evaluate a checkpoint:

```bash
python evaluate_metacausal_field.py \
  --checkpoint outputs/<run>/best_model.pth \
  --dataset Lung
```

## Frozen Qwen/API Features

Download or resolve Qwen weights:

```bash
python download_qwen_weights.py \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --local_dir models/Qwen__Qwen3-VL-8B-Instruct
```

Extract frozen features:

```bash
python extract_frozen_features.py \
  --dataset Lung \
  --backbone qwen \
  --qwen_model models/Qwen__Qwen3-VL-8B-Instruct \
  --output frozen_features/lung_qwen3vl.pt \
  --feature_dim 256
```

Train MetaCausalField on cached frozen tokens:

```bash
python train_metacausal_field.py \
  --dataset Lung \
  --backbone cached \
  --feature_cache frozen_features/lung_qwen3vl.pt \
  --use_frozen_language_tokens \
  --epochs 20
```

For benchmark manifests, direct Qwen-backed training is also available:

```bash
python train_benchmark_metacausal.py \
  --dataset Causal-VidQA \
  --manifest_path data/causal_vidqa/manifest.jsonl \
  --data_root data/causal_vidqa \
  --backbone qwen \
  --qwen_model models/Qwen__Qwen3-VL-8B-Instruct \
  --use_frozen_language_tokens \
  --num_video_frames 8
```

For a closer partially frozen Qwen-style setup, train only LoRA adapters in the
backbone while optimizing MetaCausalField:

```bash
python train_benchmark_metacausal.py \
  --dataset Causal-VidQA \
  --manifest_path data/causal_vidqa/manifest.jsonl \
  --data_root data/causal_vidqa \
  --backbone qwen \
  --qwen_model models/Qwen__Qwen3-VL-8B-Instruct \
  --train_qwen_lora \
  --qwen_lora_r 8 \
  --use_frozen_language_tokens \
  --qwen_video_input native
```

## Benchmark Manifests

Convert native benchmark annotations:

```bash
python convert_benchmarks.py \
  --benchmark CLEVRER \
  --input data/clevrer/questions.json \
  --data_root data/clevrer \
  --output data/clevrer/manifest.jsonl
```

For CausalVQA/Causal-VidQA annotations with mixed nested or flat layouts:

```bash
python prepare_causalvqa_manifest.py \
  --input data/causalvqa/annotations.json \
  --data_root data/causalvqa \
  --output data/causalvqa/manifest.jsonl
```

Train on a unified manifest:

```bash
python train_benchmark_metacausal.py \
  --dataset Causal-VidQA \
  --manifest_path data/causal_vidqa/manifest.jsonl \
  --data_root data/causal_vidqa \
  --epochs 5 \
  --batch_size 4 \
  --num_video_frames 8 \
  --enable_cf_trajectory_loss
```

Build OOD splits:

```bash
python build_ood_splits.py \
  --manifest_path data/causal_vidqa/manifest.jsonl \
  --data_root data/causal_vidqa \
  --output data/causal_vidqa/ood_manifest.jsonl
```

Run the three-stage evaluator:

```bash
python three_stage_metacausal_pipeline.py \
  --dataset Causal-VidQA \
  --manifest_path data/causal_vidqa/manifest.jsonl \
  --data_root data/causal_vidqa \
  --checkpoint benchmark_outputs/<run>/best_model.pth \
  --output_dir three_stage_outputs/causal_vidqa
```

## Paper-Style Orchestration

Inspect commands without launching long jobs:

```bash
python run_paper_experiments.py \
  --dataset MAG9 \
  --mode all \
  --dry_run \
  --backbone qwen \
  --feature_cache frozen_features/mag9_qwen3vl.pt \
  --epochs 1
```

Run individual stages:

```bash
python run_paper_experiments.py --dataset MAG9 --mode validate
python run_paper_experiments.py --dataset MAG9 --mode features --backbone cached
python run_paper_experiments.py --dataset MAG9 --mode train --backbone cached
python run_paper_experiments.py --dataset MAG9 --mode eval --checkpoint <best_model.pth>
```

For external benchmark baselines:

```bash
python run_baselines.py \
  --dataset Causal-VidQA \
  --manifest_path data/causal_vidqa/manifest.jsonl \
  --backend hf \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --output baseline_predictions.jsonl
```

Run a baseline matrix:

```bash
python run_baseline_matrix.py \
  --dataset Causal-VidQA \
  --manifest_path data/causal_vidqa/manifest.jsonl \
  --data_root data/causal_vidqa \
  --default_paper_models \
  --include_majority \
  --output_dir baseline_matrix/causal_vidqa
```

## Inference

Prediction:

```bash
python inference_metacausal_field.py \
  --checkpoint outputs/<run>/best_model.pth \
  --image Lung/1.jpg \
  predict
```

Explanation:

```bash
python inference_metacausal_field.py \
  --checkpoint outputs/<run>/best_model.pth \
  --image Lung/1.jpg \
  explain
```

Counterfactual:

```bash
python inference_metacausal_field.py \
  --checkpoint outputs/<run>/best_model.pth \
  --image Lung/1.jpg \
  counterfactual \
  --scenario smoking
```

## Notes and Limitations

- Qwen/API paths are frozen feature providers. They do not fine-tune the native
  Qwen decoder. `--train_qwen_lora` adds optional LoRA training to the Qwen
  feature provider, but the answer decoder is still the repository's lightweight
  head rather than the native Qwen decoder.
- Counterfactual trajectory supervision is strongest when manifests include
  paired counterfactual images/videos or cached counterfactual features.
- Factor-level graph extraction is learned through factor supervision and
  spatial attention; graph quality depends on data quality and training.
- Dense spatial influence is still quadratic in field size. `--influence_top_k`
  sparsifies the learned graph output after scoring, but does not make the
  attention computation fully sparse.
- The repository provides scripts needed to reproduce paper-style experiments,
  but final paper tables require official benchmark assets, model weights/API
  access, and trained checkpoints.
