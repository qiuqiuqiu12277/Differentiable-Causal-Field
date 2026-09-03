"""Evaluate MetaCausalField checkpoints and optional ablations."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from metacausal_field import CausalFieldConfig, MetaCausalLoss, MultimodalCausalField, SimpleTokenizer, VisualEncoder
from frozen_backbones import CachedFrozenBackbone
from train_metacausal_field import (
    CausalDataset,
    CounterfactualPairDataset,
    collate,
    find_counterfactual_file,
    prepare_metadata,
    scenario_directions,
    scenario_positions,
)


def load_model(checkpoint_path, device, disable_propagation=False, visual_encoder=True):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = CausalFieldConfig(**checkpoint["config"])
    tokenizer = SimpleTokenizer.from_state_dict(checkpoint["tokenizer"])
    factor_columns = checkpoint.get("factor_columns", [])
    visual_encoder = VisualEncoder(config.feature_dim, pretrained=False) if visual_encoder else None
    model = MultimodalCausalField(
        config,
        visual_encoder=visual_encoder,
        vocab_size=tokenizer.vocab_size,
        num_factors=len(factor_columns),
        enable_language=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if disable_propagation:
        model.propagation.num_steps = 0
    model.eval()
    criterion = MetaCausalLoss(config)
    return model, tokenizer, factor_columns, criterion, checkpoint


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    cached_backbone = CachedFrozenBackbone(args.feature_cache).to(device) if args.feature_cache else None
    model, tokenizer, factor_columns, criterion, checkpoint = load_model(
        args.checkpoint,
        device,
        disable_propagation=args.disable_propagation,
        visual_encoder=cached_backbone is None,
    )
    dataset_name = args.dataset or checkpoint.get("dataset", "Lung")
    df, _, _ = prepare_metadata(dataset_name)
    split = int(len(df) * args.train_split)
    val_df = df.iloc[split:].reset_index(drop=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_loader = DataLoader(
        CausalDataset(val_df, factor_columns, tokenizer, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    totals = {}
    score_abs = []
    factor_correct = 0
    factor_total = 0
    def backbone_outputs_for(batch):
        if cached_backbone is not None:
            return cached_backbone.encode_by_keys(batch["image_path"], device)
        return {"visual_features": model.visual_encoder(batch["image"].to(device))}

    for batch in tqdm(val_loader, desc="Evaluate"):
        input_ids = batch["input_ids"].to(device)
        backbone_outputs = backbone_outputs_for(batch)
        outputs = model(
            backbone_outputs["visual_features"],
            language_tokens=backbone_outputs.get("language_tokens") if args.use_frozen_language_tokens else None,
            input_ids=None if args.disable_language else input_ids,
            decoder_input_ids=None if args.disable_lm else input_ids[:, :-1],
        )
        targets = {
            "score_targets": batch["score"].to(device),
            "factor_targets": batch["factor_targets"].to(device),
        }
        if not args.disable_lm:
            targets["lm_targets"] = input_ids[:, 1:]
        losses = criterion(outputs, targets)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.cpu())

        score_abs.extend((outputs["score_pred"].cpu() - batch["score"]).abs().tolist())
        if "factor_logits" in outputs:
            pred = outputs["factor_logits"].argmax(dim=-1).cpu()
            target = batch["factor_targets"]
            mask = target.ne(-100)
            factor_correct += int((pred[mask] == target[mask]).sum())
            factor_total += int(mask.sum())

    metrics = {f"val_{k}": v / len(val_loader) for k, v in totals.items()}
    metrics["val_score_mae"] = float(np.mean(score_abs)) if score_abs else math.nan
    metrics["val_factor_acc"] = factor_correct / factor_total if factor_total else math.nan
    if "val_lm" in metrics:
        metrics["val_perplexity"] = float(math.exp(min(metrics["val_lm"], 20.0)))

    if not args.disable_counterfactual:
        cf_file = find_counterfactual_file(dataset_name)
        if cf_file:
            cf_df = pd.read_csv(cf_file)
            if args.max_counterfactual_pairs:
                cf_df = cf_df.iloc[:args.max_counterfactual_pairs]
            cf_loader = DataLoader(
                CounterfactualPairDataset(cf_df, tokenizer, transform),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate,
            )
            cf_abs = []
            cf_lm_losses = []
            for cf_batch in tqdm(cf_loader, desc="Counterfactual eval"):
                input_ids = cf_batch["input_ids"].to(device)
                cf_input_ids = cf_batch["cf_input_ids"].to(device)
                backbone_outputs = backbone_outputs_for(cf_batch)
                outputs = model.counterfactual_forward(
                    backbone_outputs["visual_features"],
                    intervention_type="modify",
                    intervention_params={
                        "position": scenario_positions(cf_batch["scenario"], device),
                        "direction": scenario_directions(cf_batch["scenario"], model.config.feature_dim, device),
                        "radius": model.config.intervention_radius if args.intervention_radius is None else args.intervention_radius,
                    },
                    language_tokens=backbone_outputs.get("language_tokens") if args.use_frozen_language_tokens else None,
                    input_ids=None if args.disable_language else input_ids,
                    decoder_input_ids=None if args.disable_lm else cf_input_ids[:, :-1],
                    num_rollout_steps=model.config.num_propagation_steps,
                )
                cf_abs.extend((outputs["score_counterfactual"].cpu() - cf_batch["cf_score"]).abs().tolist())
                if not args.disable_lm and "lm_logits_counterfactual" in outputs:
                    cf_lm_losses.append(float(F.cross_entropy(
                        outputs["lm_logits_counterfactual"].reshape(-1, outputs["lm_logits_counterfactual"].size(-1)),
                        cf_input_ids[:, 1:].reshape(-1),
                        ignore_index=model.config.pad_token_id,
                    ).cpu()))
            metrics["cf_score_mae"] = float(np.mean(cf_abs)) if cf_abs else math.nan
            if cf_lm_losses:
                metrics["cf_lm"] = float(np.mean(cf_lm_losses))
                metrics["cf_perplexity"] = float(math.exp(min(metrics["cf_lm"], 20.0)))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate MetaCausalField")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["Lung", "MAG9"], default=None)
    parser.add_argument("--output", default="./evaluation_metrics.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--max_counterfactual_pairs", type=int, default=None)
    parser.add_argument("--intervention_radius", type=float, default=None)
    parser.add_argument("--disable_language", action="store_true")
    parser.add_argument("--disable_lm", action="store_true")
    parser.add_argument("--disable_counterfactual", action="store_true")
    parser.add_argument("--disable_propagation", action="store_true")
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument("--use_frozen_language_tokens", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
