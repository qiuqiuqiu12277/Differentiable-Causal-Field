# MetaCausalField Method Completion Status

This repository now contains a runnable end-to-end MetaCausalField prototype for the method described in Sections 3.1-3.4.

## Completed Components

- Real visual encoder: `VisualEncoder` in `metacausal_field.py` uses a ResNet patch backbone and returns spatial visual tokens.
- Language encoder/decoder: `SimpleTokenizer`, `TextEncoder`, and `TextCausalDecoder` provide language conditioning and language-model training.
- Multimodal causal field: `MultimodalCausalField` fuses language tokens into the propagated causal field and predicts score, factors, and text.
- Causal dynamics: `CausalPropagation` performs multi-step field rollout.
- Intervention and counterfactual inference: `InterventionModule` and `counterfactual_forward` support field-level interventions.
- Learning objectives: `MetaCausalLoss` includes LM, score, factor, consistency, counterfactual, sparsity, and smoothness losses.
- Counterfactual supervision: `train_metacausal_field.py` loads `results/*/cf_pd.csv` and supervises counterfactual score/text predictions.
- Cross-environment consistency: training enforces representation consistency under feature perturbations.
- Checkpointing: training saves `best_model.pth`, periodic checkpoints, tokenizer state, factor columns, config, and metrics.
- Metrics: `evaluate_metacausal_field.py` reports validation loss, score MAE, factor accuracy, LM perplexity, and counterfactual metrics.
- Ablations: `run_ablation_metacausal_field.py` evaluates full, no-language, no-LM, no-counterfactual-eval, and no-propagation variants.
- Inference: `inference_metacausal_field.py` supports prediction, causal explanation visualization, and counterfactual visualization.
- Frozen Qwen/API backbones: `frozen_backbones.py` and `extract_frozen_features.py` support Qwen3-VL/Qwen-VL, closed MLLM APIs, and cached frozen features.
- Benchmark loaders: `benchmark_datasets.py` supports manifest-based CLEVRER, Causal3DIdent/CITRIS, Causal-VidQA, plus MAG/Lung adapters.
- Strict three-stage pipeline: `three_stage_metacausal_pipeline.py` implements factor discovery, structure learning, and counterfactual reasoning.
- Paper metrics: `causal_metrics.py` implements NP/NR/NF, AP/AR/AF, ESHD, QA category accuracy, OOD drop, counterfactual consistency, causal flip accuracy, invalid transition rate, localization IoU, key-object accuracy, faithfulness, and latency helpers.
- OOD and interpretability: `evaluate_ood.py` and `evaluate_interpretability.py` provide dedicated evaluation entry points.
- Propagation trade-off: `propagation_sweep.py` runs K-step performance/latency sweeps.
- MAG/Lung table reproduction: `reproduce_mag_lung_tables.py` exports CSV and LaTeX tables from pipeline outputs.

## Main Commands

Train:

```bash
python train_metacausal_field.py --dataset Lung --epochs 20 --batch_size 8
```

Extract frozen Qwen3-VL features, then train only MetaCausalField:

```bash
python download_qwen_weights.py \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --local_dir models/Qwen__Qwen3-VL-8B-Instruct

python extract_frozen_features.py \
  --dataset Lung \
  --backbone qwen \
  --qwen_model models/Qwen__Qwen3-VL-8B-Instruct \
  --output frozen_features/lung_qwen3vl.pt

python train_metacausal_field.py \
  --dataset Lung \
  --backbone cached \
  --feature_cache frozen_features/lung_qwen3vl.pt \
  --epochs 20
```

Alternatively, download and extract in one command:

```bash
python extract_frozen_features.py \
  --dataset Lung \
  --backbone qwen \
  --qwen_model Qwen/Qwen3-VL-8B-Instruct \
  --download_qwen \
  --qwen_cache_dir models/Qwen__Qwen3-VL-8B-Instruct \
  --output frozen_features/lung_qwen3vl.pt
```

Use a closed MLLM API as a frozen backbone:

```bash
python extract_frozen_features.py \
  --dataset MAG9 \
  --backbone api \
  --output frozen_features/mag_api.pt

python train_metacausal_field.py \
  --dataset MAG9 \
  --backbone cached \
  --feature_cache frozen_features/mag_api.pt
```

Evaluate:

```bash
python evaluate_metacausal_field.py --checkpoint outputs/<run>/best_model.pth --dataset Lung
```

Run ablations:

```bash
python run_ablation_metacausal_field.py --checkpoint outputs/<run>/best_model.pth --dataset Lung
```

Run the strict three-stage protocol:

```bash
python three_stage_metacausal_pipeline.py \
  --dataset Lung \
  --checkpoint outputs/<run>/best_model.pth \
  --output_dir three_stage_outputs/lung
```

Evaluate a benchmark manifest:

```bash
python three_stage_metacausal_pipeline.py \
  --dataset Causal-VidQA \
  --manifest_path data/causal_vidqa/manifest.jsonl \
  --data_root data/causal_vidqa \
  --checkpoint outputs/<run>/best_model.pth
```

Run OOD and propagation-step experiments:

```bash
python evaluate_ood.py \
  --checkpoint outputs/<run>/best_model.pth \
  --dataset Causal-VidQA \
  --manifest_path data/causal_vidqa/ood_manifest.jsonl

python propagation_sweep.py \
  --checkpoint outputs/<run>/best_model.pth \
  --dataset Lung \
  --k_values 0,1,2,3,4
```

Reproduce MAG/Lung causal discovery tables:

```bash
python reproduce_mag_lung_tables.py \
  --checkpoint outputs/<run>/best_model.pth \
  --output_dir paper_table_outputs
```

## Manifest Schema

For CLEVRER, Causal3DIdent/CITRIS, and Causal-VidQA, provide JSONL/JSON/CSV rows with these optional fields:

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
  "ood_type": "Scene Shift",
  "object_id": "red_object",
  "bbox": [0.1, 0.2, 0.3, 0.4]
}
```

Predict:

```bash
python inference_metacausal_field.py --checkpoint outputs/<run>/best_model.pth --image Lung/1.jpg predict
```

Counterfactual:

```bash
python inference_metacausal_field.py --checkpoint outputs/<run>/best_model.pth --image Lung/1.jpg counterfactual --scenario smoking
```

## Smoke-Tested Artifact

A small CPU smoke run was generated at:

```text
outputs_smoke/run_20260428_171908/best_model.pth
```

This checkpoint is only a functionality check, not a final trained model for reporting paper-quality results.

## Added Experiment Utilities

Native benchmark conversion:

```bash
python convert_benchmarks.py \
  --benchmark CLEVRER \
  --input data/clevrer/questions.json \
  --data_root data/clevrer \
  --output data/clevrer/manifest.jsonl
```

Benchmark video/QA training:

```bash
python train_benchmark_metacausal.py \
  --dataset Causal-VidQA \
  --manifest_path data/causal_vidqa/manifest.jsonl \
  --data_root data/causal_vidqa \
  --num_video_frames 8 \
  --num_propagation_steps 3
```

Strict three-stage training:

```bash
python train_three_stage_metacausal.py \
  --dataset MAG9 \
  --epochs_per_stage 3 \
  --output_dir three_stage_training
```

OOD construction, baseline, strict scorer, and interpretability:

```bash
python build_ood_splits.py --manifest_path data/causal_vidqa/manifest.jsonl --output data/causal_vidqa/ood_manifest.jsonl
python run_baselines.py --dataset Causal-VidQA --manifest_path data/causal_vidqa/manifest.jsonl --backend majority
python score_qa_predictions.py --predictions baseline_predictions.jsonl --manifest_path data/causal_vidqa/manifest.jsonl
python evaluate_interpretability.py --checkpoint outputs/<run>/best_model.pth --dataset Causal-VidQA --manifest_path data/causal_vidqa/manifest.jsonl
```

Default graph files are available at `gold_graphs/mag9_gold_graph.csv` and `gold_graphs/lung_gold_graph.csv`; replace them with official benchmark gold graphs when available.
