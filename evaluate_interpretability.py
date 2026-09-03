"""Interpretability metrics for causal explanations."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from benchmark_datasets import UnifiedCausalDataset, load_benchmark
from causal_metrics import (
    bbox_iou,
    causal_localization_iou,
    explanation_faithfulness,
    key_object_identification_accuracy,
)
from evaluate_metacausal_field import load_model
from train_metacausal_field import collate


def field_to_bbox(field: torch.Tensor):
    heat = field.abs().mean(dim=-1)
    idx = int(heat.argmax())
    h, w = heat.shape
    y, x = divmod(idx, w)
    return [x / w, y / h, (x + 1) / w, (y + 1) / h]


def bbox_center(box):
    return torch.tensor([[(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]], dtype=torch.float32)


def best_object_for_box(sample, pred_box):
    if sample.objects:
        best = max(
            sample.objects,
            key=lambda obj: bbox_iou(pred_box, obj.get("bbox") or [0, 0, 0, 0]),
        )
        return str(best.get("id", ""))
    if sample.object_id and sample.bbox:
        return str(sample.object_id) if bbox_iou(pred_box, sample.bbox) >= 0.5 else ""
    return str(sample.object_id or "")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate causal explanation interpretability")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="Causal-VidQA", choices=["CLEVRER", "Causal3DIdent", "CITRIS", "Causal-VidQA", "MAG9", "Lung"])
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--human_preference_csv", default=None, help="CSV with columns: sample_id,preferred_method")
    parser.add_argument("--output", default="./interpretability_metrics.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--intervention_radius", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    samples, factor_names = load_benchmark(args.dataset, args.manifest_path, args.data_root)
    model, tokenizer, _, _, _ = load_model(args.checkpoint, device)

    pred_boxes = []
    gold_boxes = []
    object_records = []
    factual_scores = []
    ablated_scores = []

    from torch.utils.data import DataLoader
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = UnifiedCausalDataset(samples, tokenizer, transform, factor_names=factor_names, num_video_frames=8)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)

    for idx, batch in enumerate(loader):
        sample = samples[idx]
        image = batch["image"].to(device)
        text = sample.question or sample.text
        input_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
        visual = model.visual_encoder(image)
        outputs = model(visual, input_ids=input_ids)
        pred_box = field_to_bbox(outputs["field"][0].cpu())
        pred_boxes.append(pred_box)
        if sample.bbox:
            gold_boxes.append(sample.bbox)
        pred_object = best_object_for_box(sample, pred_box)
        object_records.append({
            "pred_object": pred_object,
            "gold_object": sample.object_id or pred_object,
        })
        factual_scores.append(float(outputs["score_pred"][0].cpu()))
        cf_outputs = model.counterfactual_forward(
            visual,
            intervention_type="remove",
            intervention_params={
                "position": bbox_center(pred_box).to(device),
                "radius": model.config.intervention_radius if args.intervention_radius is None else args.intervention_radius,
            },
            input_ids=input_ids,
            num_rollout_steps=model.config.num_propagation_steps,
        )
        ablated_scores.append(float(cf_outputs["score_counterfactual"][0].cpu()))

    metrics = {
        "Causal_Localization_IoU": causal_localization_iou(pred_boxes, gold_boxes) if gold_boxes else float("nan"),
        "Key_Object_Identification_Accuracy": key_object_identification_accuracy(object_records),
        "Explanation_Faithfulness": explanation_faithfulness(factual_scores, ablated_scores),
    }
    if args.human_preference_csv:
        pref = pd.read_csv(args.human_preference_csv)
        metrics["Human_Preference"] = float((pref["preferred_method"] == "MetaCausalField").mean())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
