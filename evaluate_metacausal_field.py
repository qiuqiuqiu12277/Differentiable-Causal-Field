"""Evaluate InfluenceField checkpoints with leakage-safe data partitions."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from metacausal_field import CausalFieldConfig, MetaCausalLoss, MultimodalCausalField, SimpleTokenizer, VisualEncoder
from frozen_backbones import CachedFrozenBackbone, load_torch_payload


def restore_dataframe_partitions(df, split_record):
    """Restore exact checkpoint partitions and reject dataset drift."""
    if not isinstance(split_record, dict):
        raise ValueError("split_record must be a dictionary")
    id_column = split_record.get("id_column")
    if not id_column or id_column not in df:
        raise ValueError(
            f"Checkpoint split id column {id_column!r} is unavailable in the current dataset."
        )
    split_ids = {
        name: {str(value) for value in split_record.get(f"{name}_ids", [])}
        for name in ("train", "val", "test")
    }
    if any(not values for values in split_ids.values()):
        raise ValueError("Checkpoint split record must contain non-empty train/val/test IDs.")
    if (
        split_ids["train"] & split_ids["val"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["val"] & split_ids["test"]
    ):
        raise ValueError("Checkpoint split record contains overlapping IDs.")

    current_ids = df[id_column].astype(str)
    expected_ids = set().union(*split_ids.values())
    if set(current_ids) != expected_ids:
        missing = sorted(expected_ids - set(current_ids))[:3]
        unexpected = sorted(set(current_ids) - expected_ids)[:3]
        raise ValueError(
            "Dataset content differs from the checkpoint split record: "
            f"missing={missing}, unexpected={unexpected}."
        )
    return tuple(
        df.loc[current_ids.isin(split_ids[name])].reset_index(drop=True)
        for name in ("train", "val", "test")
    )


def load_model(
    checkpoint_path,
    device,
    disable_propagation=False,
    visual_encoder=True,
    num_propagation_steps=None,
    expected_dataset=None,
):
    checkpoint = load_torch_payload(checkpoint_path, map_location=device)
    checkpoint_dataset = checkpoint.get("dataset")
    aliases = {"MAG": "MAG9", "Lung4": "Lung"}
    if expected_dataset and checkpoint_dataset:
        expected = aliases.get(str(expected_dataset), str(expected_dataset))
        actual = aliases.get(str(checkpoint_dataset), str(checkpoint_dataset))
        if expected != actual:
            raise ValueError(
                f"Checkpoint was trained for {checkpoint_dataset!r}, not {expected_dataset!r}."
            )
    checkpoint_backbone = str((checkpoint.get("args") or {}).get("backbone", "resnet"))
    if bool((checkpoint.get("args") or {}).get("train_qwen_lora", False)):
        raise ValueError(
            "This checkpoint includes a trainable Qwen LoRA adapter, which the generic evaluator "
            "cannot reconstruct. Use a dedicated Qwen evaluator or export matched post-training features."
        )
    if checkpoint_backbone in {"cached", "qwen"} and visual_encoder:
        raise ValueError(
            f"Checkpoint backbone is {checkpoint_backbone!r}; evaluation requires the matching "
            "--feature_cache and must not substitute a randomly initialized ResNet."
        )
    if disable_propagation and num_propagation_steps not in (None, 0):
        raise ValueError(
            "disable_propagation cannot be combined with a non-zero "
            "num_propagation_steps override"
        )

    effective_steps = 0 if disable_propagation else num_propagation_steps
    if effective_steps is not None and effective_steps < 0:
        raise ValueError("num_propagation_steps must be non-negative")

    # Build the runtime config from a copy so a sweep override never leaks into
    # the checkpoint metadata returned to callers (or a subsequently saved copy).
    config_values = dict(checkpoint["config"])
    if effective_steps is not None:
        config_values["num_propagation_steps"] = int(effective_steps)
    config = CausalFieldConfig(**config_values)
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
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch would leave random or unused parameters: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}."
        )
    model.eval()
    criterion = MetaCausalLoss(config)
    return model, tokenizer, factor_columns, criterion, checkpoint


@torch.no_grad()
def evaluate(args):
    from torchvision import transforms
    from train_metacausal_field import (
        CausalDataset,
        CounterfactualPairDataset,
        collate,
        find_counterfactual_file,
        intervention_tensors,
        prepare_metadata,
        prepare_counterfactual_supervision,
        seed_worker,
        split_dataframe,
    )

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    cached_backbone = (
        CachedFrozenBackbone(
            args.feature_cache,
            allow_smoke_cache=args.allow_missing_media,
        ).to(device)
        if args.feature_cache
        else None
    )
    model, tokenizer, factor_columns, criterion, checkpoint = load_model(
        args.checkpoint,
        device,
        disable_propagation=args.disable_propagation,
        visual_encoder=cached_backbone is None,
        num_propagation_steps=args.num_propagation_steps,
        expected_dataset=args.dataset,
    )
    dataset_name = args.dataset or checkpoint.get("dataset", "Lung")
    df, _, _ = prepare_metadata(dataset_name)
    checkpoint_args = checkpoint.get("args", {}) or {}
    seed = args.seed if args.seed is not None else int(checkpoint_args.get("seed", 42))
    train_fraction = (
        args.train_split
        if args.train_split is not None
        else float(checkpoint_args.get("train_split", 0.7))
    )
    val_fraction = (
        args.val_split
        if args.val_split is not None
        else float(checkpoint_args.get("val_split", 0.15))
    )
    group_column = args.group_column or checkpoint_args.get("group_column")
    split_record = checkpoint.get("data_split")
    split_source = "checkpoint"
    if split_record is None:
        sidecar = Path(args.checkpoint).resolve().parent / "data_split.json"
        if sidecar.exists():
            with open(sidecar) as handle:
                split_record = json.load(handle)
            split_source = "checkpoint_sidecar"
    if split_record is not None:
        train_df, val_df, test_df = restore_dataframe_partitions(df, split_record)
    else:
        train_df, val_df, test_df = split_dataframe(
            df,
            seed=seed,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            group_column=group_column,
        )
        split_source = "reconstructed_from_checkpoint_args"
    evaluation_df = val_df if args.evaluation_split == "val" else test_df
    metric_prefix = args.evaluation_split
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    evaluation_loader = DataLoader(
        CausalDataset(
            evaluation_df,
            factor_columns,
            tokenizer,
            transform,
            allow_missing_media=args.allow_missing_media,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        worker_init_fn=seed_worker,
    )

    totals = {}
    score_abs = []
    factor_correct = 0
    factor_total = 0
    def backbone_outputs_for(batch):
        if cached_backbone is not None:
            return cached_backbone.encode_by_keys(batch["image_path"], device)
        return {"visual_features": model.visual_encoder(batch["image"].to(device))}

    for batch in tqdm(evaluation_loader, desc=f"Evaluate {args.evaluation_split}"):
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

    metrics = {f"{metric_prefix}_{k}": v / len(evaluation_loader) for k, v in totals.items()}
    metrics[f"{metric_prefix}_score_mae"] = float(np.mean(score_abs)) if score_abs else math.nan
    metrics[f"{metric_prefix}_factor_acc"] = factor_correct / factor_total if factor_total else math.nan
    if f"{metric_prefix}_lm" in metrics:
        metrics[f"{metric_prefix}_teacher_forced_perplexity"] = float(
            math.exp(min(metrics[f"{metric_prefix}_lm"], 20.0))
        )
    metrics["evaluation_protocol"] = {
        "split": args.evaluation_split,
        "num_samples": len(evaluation_df),
        "seed": seed,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "group_column": group_column,
        "allow_missing_media": bool(args.allow_missing_media),
        "split_source": split_source,
    }

    if not args.disable_counterfactual:
        cf_file = find_counterfactual_file(dataset_name)
        if cf_file:
            cf_df, cf_audit = prepare_counterfactual_supervision(pd.read_csv(cf_file))
            source_column = next(
                (name for name in ("Image", "ImagePath", "image_path") if name in cf_df),
                None,
            )
            if source_column and "ImagePath" in evaluation_df:
                heldout_media = set(evaluation_df["ImagePath"].astype(str))
                cf_df = cf_df.loc[
                    cf_df[source_column].astype(str).isin(heldout_media)
                ].reset_index(drop=True)
            elif "split" in cf_df:
                labels = cf_df["split"].astype(str).str.strip().str.lower()
                aliases = {"validation": "val", "dev": "val", "eval": "test"}
                cf_df = cf_df.loc[labels.replace(aliases).eq(args.evaluation_split)].reset_index(drop=True)
            else:
                cf_df = cf_df.iloc[:0].copy()
            if args.max_counterfactual_pairs:
                cf_df = cf_df.iloc[:args.max_counterfactual_pairs]
            cf_audit["selected_evaluation_rows"] = int(len(cf_df))
            metrics["counterfactual_protocol"] = cf_audit
            cf_loader = (
                DataLoader(
                    CounterfactualPairDataset(
                        cf_df,
                        tokenizer,
                        transform,
                        allow_missing_media=args.allow_missing_media,
                    ),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=collate,
                    worker_init_fn=seed_worker,
                )
                if not cf_df.empty
                else None
            )
            cf_abs = []
            cf_lm_losses = []
            for cf_batch in tqdm(cf_loader or [], desc="Counterfactual eval"):
                input_ids = cf_batch["input_ids"].to(device)
                cf_input_ids = cf_batch["cf_input_ids"].to(device)
                backbone_outputs = backbone_outputs_for(cf_batch)
                positions, directions = intervention_tensors(
                    cf_batch["intervention"],
                    cf_batch["scenario"],
                    model.config.feature_dim,
                    device,
                )
                outputs = model.counterfactual_forward(
                    backbone_outputs["visual_features"],
                    intervention_type="modify",
                    intervention_params={
                        "position": positions,
                        "direction": directions,
                        "radius": model.config.intervention_radius if args.intervention_radius is None else args.intervention_radius,
                    },
                    language_tokens=backbone_outputs.get("language_tokens") if args.use_frozen_language_tokens else None,
                    input_ids=None if args.disable_language else input_ids,
                    decoder_input_ids=None if args.disable_lm else cf_input_ids[:, :-1],
                    num_rollout_steps=model.propagation.num_steps,
                )
                score_mask = cf_batch["cf_score_available"]
                if score_mask.any():
                    cf_abs.extend(
                        (
                            outputs["score_counterfactual"].cpu()[score_mask]
                            - cf_batch["cf_score"][score_mask]
                        ).abs().tolist()
                    )
                text_mask = cf_batch["cf_text_available"].to(device)
                if not args.disable_lm and "lm_logits_counterfactual" in outputs and text_mask.any():
                    cf_targets = cf_input_ids[:, 1:].clone()
                    cf_targets[~text_mask] = -100
                    cf_lm_losses.append(float(F.cross_entropy(
                        outputs["lm_logits_counterfactual"].reshape(-1, outputs["lm_logits_counterfactual"].size(-1)),
                        cf_targets.reshape(-1),
                        ignore_index=model.config.pad_token_id,
                    ).cpu()))
            if cf_abs:
                metrics["cf_score_mae"] = float(np.mean(cf_abs))
            if cf_lm_losses:
                metrics["cf_teacher_forced_lm"] = float(np.mean(cf_lm_losses))
                metrics["cf_teacher_forced_perplexity"] = float(
                    math.exp(min(metrics["cf_teacher_forced_lm"], 20.0))
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate InfluenceField")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["Lung", "MAG9"], default=None)
    parser.add_argument("--output", default="./evaluation_metrics.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--evaluation_split", choices=["val", "test"], default="test")
    parser.add_argument("--train_split", type=float, default=None)
    parser.add_argument("--val_split", type=float, default=None)
    parser.add_argument("--group_column", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_counterfactual_pairs", type=int, default=None)
    parser.add_argument("--intervention_radius", type=float, default=None)
    parser.add_argument("--disable_language", action="store_true")
    parser.add_argument("--disable_lm", action="store_true")
    parser.add_argument("--disable_counterfactual", action="store_true")
    propagation_group = parser.add_mutually_exclusive_group()
    propagation_group.add_argument("--disable_propagation", action="store_true")
    propagation_group.add_argument("--num_propagation_steps", type=int, default=None)
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument("--use_frozen_language_tokens", action="store_true")
    parser.add_argument(
        "--allow_missing_media",
        action="store_true",
        help="Use placeholders only for explicit smoke tests; strict media loading is default.",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
