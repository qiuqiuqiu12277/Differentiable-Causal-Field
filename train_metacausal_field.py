"""
Train MetaCausalField end to end.

This script now covers the full research prototype loop:
- real image patch encoding
- language conditioning and language-model decoding
- score/factor supervision
- counterfactual supervision from generated cf_pd.csv files
- validation metrics and checkpoints
"""

import argparse
import json
import math
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from metacausal_field import (
    CausalFieldConfig,
    MetaCausalLoss,
    MultimodalCausalField,
    SimpleTokenizer,
    VisualEncoder,
    visualize_causal_field,
)
from frozen_backbones import CachedFrozenBackbone


TEXT_COLUMNS = {"Review", "description", "Demographics", "History", "ImagePath", "Image", "ImagePrompt"}


def numeric_factor_columns(df: pd.DataFrame) -> List[str]:
    blocked = TEXT_COLUMNS | {"id", "score", "sample_id", "scenario"}
    columns = []
    for col in df.columns:
        if col in blocked:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            columns.append(col)
    return columns


def prepare_metadata(dataset: str) -> tuple[pd.DataFrame, str, List[str]]:
    if dataset == "Lung":
        df = pd.read_csv("Lung.csv")
        df["Review"] = df.get("description", "").fillna("")
        missing_review = df["Review"].str.len() == 0
        df.loc[missing_review, "Review"] = (
            df.get("Demographics", "").fillna("") + " " + df.get("History", "").fillna("")
        )
        df["ImagePath"] = df["id"].map(lambda x: f"./Lung/{int(x)}.jpg")
        image_dir = "./Lung"
    elif dataset == "MAG9":
        df = pd.read_csv("MAG9.csv")
        if "ImagePath" not in df.columns:
            df["ImagePath"] = [f"./apple_images_a9/apple_{idx}.png" for idx in range(len(df))]
        image_dir = "./apple_images_a9"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    df["Review"] = df["Review"].fillna("").astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    return df.reset_index(drop=True), image_dir, numeric_factor_columns(df)


def find_counterfactual_file(dataset: str) -> Optional[Path]:
    candidates = sorted(Path("results").glob(f"{dataset}*/cf_pd.csv"))
    return candidates[-1] if candidates else None


def label_to_class(value) -> int:
    if pd.isna(value):
        return -100
    value = int(float(value))
    if value < 0:
        return 0
    if value > 0:
        return 2
    return 1


def load_image(path: str, fallback_size=(224, 224)) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", fallback_size, color="white")


class CausalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, factor_columns: List[str], tokenizer: SimpleTokenizer, transform):
        self.df = df.reset_index(drop=True)
        self.factor_columns = factor_columns
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.transform(load_image(str(row["ImagePath"])))
        input_ids = torch.tensor(self.tokenizer.encode(row["Review"]), dtype=torch.long)
        factor_targets = torch.tensor(
            [label_to_class(row[col]) if col in row else -100 for col in self.factor_columns],
            dtype=torch.long,
        )
        return {
            "image": image,
            "input_ids": input_ids,
            "score": torch.tensor(float(row["score"]), dtype=torch.float32),
            "factor_targets": factor_targets,
            "review": row["Review"],
            "image_path": row["ImagePath"],
            "feature_key": row["ImagePath"],
        }


class CounterfactualPairDataset(Dataset):
    def __init__(self, cf_df: pd.DataFrame, tokenizer: SimpleTokenizer, transform, max_pairs: Optional[int] = None):
        self.cf_df = cf_df.reset_index(drop=True)
        if max_pairs:
            self.cf_df = self.cf_df.iloc[:max_pairs].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.cf_df)

    def __getitem__(self, idx):
        row = self.cf_df.iloc[idx]
        image_path = str(row.get("Image", row.get("ImagePath", "")))
        cf_image_path = str(row.get(
            "CounterfactualImage",
            row.get("counterfactual_image", row.get("cf_image", image_path)),
        ))
        source_text = str(row.get("Review", ""))
        target_text = str(row.get("Review", ""))
        scenario = str(row.get("scenario", "counterfactual"))
        return {
            "image": self.transform(load_image(image_path)),
            "cf_image": self.transform(load_image(cf_image_path)),
            "input_ids": torch.tensor(self.tokenizer.encode(source_text), dtype=torch.long),
            "cf_input_ids": torch.tensor(self.tokenizer.encode(target_text), dtype=torch.long),
            "cf_score": torch.tensor(float(pd.to_numeric(row.get("score", 0.0), errors="coerce")), dtype=torch.float32),
            "scenario": scenario,
            "image_path": image_path,
            "cf_image_path": cf_image_path,
            "feature_key": image_path,
            "cf_feature_key": cf_image_path,
        }


def collate(batch):
    out = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out


def scenario_positions(scenarios: List[str], device: torch.device) -> torch.Tensor:
    coords = []
    for scenario in scenarios:
        seed = abs(hash(scenario))
        x = 0.15 + 0.7 * ((seed % 997) / 996.0)
        y = 0.15 + 0.7 * (((seed // 997) % 991) / 990.0)
        coords.append([x, y])
    return torch.tensor(coords, dtype=torch.float32, device=device)


def scenario_directions(scenarios: List[str], feature_dim: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(feature_dim, dtype=torch.float32, device=device).unsqueeze(0)
    seeds = torch.tensor([abs(hash(s)) % 10007 for s in scenarios], dtype=torch.float32, device=device).unsqueeze(1)
    directions = torch.sin(base * 0.017 + seeds * 0.001)
    return F.normalize(directions, dim=-1)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.meta_df, _, self.factor_columns = prepare_metadata(args.dataset)
        split = int(len(self.meta_df) * args.train_split)
        train_df = self.meta_df.iloc[:split].reset_index(drop=True)
        val_df = self.meta_df.iloc[split:].reset_index(drop=True)

        cf_file = find_counterfactual_file(args.dataset)
        cf_df = pd.read_csv(cf_file) if cf_file else pd.DataFrame()
        texts = train_df["Review"].tolist() + val_df["Review"].tolist()
        if not cf_df.empty and "Review" in cf_df:
            texts.extend(cf_df["Review"].fillna("").astype(str).tolist())
        self.tokenizer = SimpleTokenizer.build(
            texts,
            vocab_size=args.vocab_size,
            max_length=args.max_text_length,
        )

        self.config = CausalFieldConfig(
            feature_dim=args.feature_dim,
            num_heads=args.num_heads,
            num_propagation_steps=args.num_propagation_steps,
            intervention_radius=args.intervention_radius,
            dropout=args.dropout,
            vocab_size=self.tokenizer.vocab_size,
            max_text_length=args.max_text_length,
            lambda_consistency=args.lambda_consistency,
            lambda_counterfactual=args.lambda_counterfactual,
            lambda_sparsity=args.lambda_sparsity,
            lambda_smoothness=args.lambda_smoothness,
            influence_top_k=args.influence_top_k,
        )
        self.cached_backbone = None
        if args.backbone == "cached":
            if not args.feature_cache:
                raise ValueError("--feature_cache is required when --backbone cached")
            self.cached_backbone = CachedFrozenBackbone(args.feature_cache, feature_dim=args.feature_dim)
            args.feature_dim = self.cached_backbone.feature_dim
            self.config.feature_dim = self.cached_backbone.feature_dim
            self.visual_encoder = None
        else:
            self.visual_encoder = VisualEncoder(
                feature_dim=args.feature_dim,
                pretrained=args.pretrained_backbone,
                train_backbone=not args.freeze_backbone,
            )
        self.model = MultimodalCausalField(
            self.config,
            visual_encoder=self.visual_encoder,
            vocab_size=self.tokenizer.vocab_size,
            num_factors=len(self.factor_columns),
            enable_language=not args.disable_language,
        ).to(self.device)
        if args.init_checkpoint:
            checkpoint = torch.load(args.init_checkpoint, map_location=self.device, weights_only=False)
            state = checkpoint.get("model_state_dict", checkpoint)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(
                f"Initialized from {args.init_checkpoint} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )

        self.criterion = MetaCausalLoss(self.config)
        self.criterion.lambda_consistency = args.lambda_consistency
        self.criterion.lambda_counterfactual = args.lambda_counterfactual

        self.train_loader = DataLoader(
            CausalDataset(train_df, self.factor_columns, self.tokenizer, self.transform),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate,
        )
        self.val_loader = DataLoader(
            CausalDataset(val_df, self.factor_columns, self.tokenizer, self.transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
        )
        if cf_file and not cf_df.empty:
            self.cf_loader = DataLoader(
                CounterfactualPairDataset(cf_df, self.tokenizer, self.transform, args.max_counterfactual_pairs),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate,
            )
            print(f"Loaded counterfactual supervision: {cf_file} ({len(self.cf_loader.dataset)} pairs)")
        else:
            self.cf_loader = None
            print("No counterfactual CSV found; counterfactual supervised loss will be skipped.")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, args.epochs))
        self.best_metric = float("inf")

        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"Vocab size: {self.tokenizer.vocab_size}, factors: {self.factor_columns}")
        print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    def extract_backbone_features(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.cached_backbone is not None:
            return self.cached_backbone.encode_by_keys(batch["feature_key"], self.device)
        return {"visual_features": self.visual_encoder(batch["image"].to(self.device))}

    def extract_visual_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.extract_backbone_features(batch)["visual_features"]

    def run_batch(self, batch: Dict[str, torch.Tensor], train: bool) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.device)
        decoder_input_ids = input_ids[:, :-1]
        lm_targets = input_ids[:, 1:]

        backbone_outputs = self.extract_backbone_features(batch)
        visual_features = backbone_outputs["visual_features"]
        language_tokens = backbone_outputs.get("language_tokens") if self.args.use_frozen_language_tokens else None
        outputs = self.model(
            visual_features,
            language_tokens=language_tokens,
            input_ids=None if self.args.disable_language else input_ids,
            decoder_input_ids=None if self.args.disable_lm else decoder_input_ids,
        )
        targets = {
            "score_targets": batch["score"].to(self.device),
            "factor_targets": batch["factor_targets"].to(self.device),
        }
        if not self.args.disable_lm:
            targets["lm_targets"] = lm_targets

        losses = self.criterion(outputs, targets)

        if not self.args.disable_consistency:
            noise = torch.randn_like(visual_features) * self.args.feature_noise_std
            aug_outputs = self.model(
                visual_features + noise,
                language_tokens=language_tokens,
                input_ids=None if self.args.disable_language else input_ids,
            )
            losses["consistency"] = self.criterion.causal_consistency_loss(outputs["field"], aug_outputs["field"])
            losses["total"] = losses["total"] + self.args.lambda_consistency * losses["consistency"]

        return losses

    def run_counterfactual_batch(self, cf_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = cf_batch["input_ids"].to(self.device)
        cf_input_ids = cf_batch["cf_input_ids"].to(self.device)
        backbone_outputs = self.extract_backbone_features(cf_batch)
        visual_features = backbone_outputs["visual_features"]
        language_tokens = backbone_outputs.get("language_tokens") if self.args.use_frozen_language_tokens else None
        cf_outputs = self.model.counterfactual_forward(
            visual_features,
            intervention_type="modify",
            intervention_params={
                "position": scenario_positions(cf_batch["scenario"], self.device),
                "direction": scenario_directions(cf_batch["scenario"], self.config.feature_dim, self.device),
                "radius": self.args.intervention_radius,
            },
            language_tokens=language_tokens,
            input_ids=None if self.args.disable_language else input_ids,
            decoder_input_ids=None if self.args.disable_lm else cf_input_ids[:, :-1],
            num_rollout_steps=self.args.num_propagation_steps,
        )
        targets = {"cf_score_targets": cf_batch["cf_score"].to(self.device)}
        if not self.args.disable_lm:
            targets["cf_lm_targets"] = cf_input_ids[:, 1:]
        if self.args.enable_cf_trajectory_loss:
            teacher_batch = dict(cf_batch)
            teacher_batch["image"] = cf_batch["cf_image"]
            teacher_batch["feature_key"] = cf_batch["cf_feature_key"]
            with torch.no_grad():
                teacher_features = self.extract_backbone_features(teacher_batch)["visual_features"]
                teacher_outputs = self.model(teacher_features)
            targets["cf_target_trajectory"] = [
                state.detach() for state in teacher_outputs.get("field_trajectory", [])[1:]
            ]
        losses = self.criterion(cf_outputs, targets)
        return losses

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {}
        cf_iter = iter(self.cf_loader) if self.cf_loader else None
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            losses = self.run_batch(batch, train=True)
            if cf_iter is not None and not self.args.disable_counterfactual:
                try:
                    cf_batch = next(cf_iter)
                except StopIteration:
                    cf_iter = iter(self.cf_loader)
                    cf_batch = next(cf_iter)
                cf_losses = self.run_counterfactual_batch(cf_batch)
                losses["counterfactual_supervised"] = cf_losses["total"]
                losses["total"] = losses["total"] + cf_losses["total"]

            self.optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            pbar.set_postfix(loss=float(losses["total"].detach().cpu()))

        return {key: value / len(self.train_loader) for key, value in totals.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {}
        score_abs = []
        factor_correct = 0
        factor_total = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            losses = self.run_batch(batch, train=False)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())

            input_ids = batch["input_ids"].to(self.device)
            backbone_outputs = self.extract_backbone_features(batch)
            outputs = self.model(
                backbone_outputs["visual_features"],
                language_tokens=backbone_outputs.get("language_tokens") if self.args.use_frozen_language_tokens else None,
                input_ids=None if self.args.disable_language else input_ids,
            )
            score_abs.extend((outputs["score_pred"].cpu() - batch["score"]).abs().tolist())
            if "factor_logits" in outputs:
                pred = outputs["factor_logits"].argmax(dim=-1).cpu()
                target = batch["factor_targets"]
                mask = target.ne(-100)
                factor_correct += int((pred[mask] == target[mask]).sum())
                factor_total += int(mask.sum())

        metrics = {key: value / len(self.val_loader) for key, value in totals.items()}
        metrics["score_mae"] = float(np.mean(score_abs)) if score_abs else math.nan
        metrics["factor_acc"] = factor_correct / factor_total if factor_total else math.nan
        if "lm" in metrics:
            metrics["perplexity"] = float(math.exp(min(metrics["lm"], 20.0)))
        return metrics

    def save_checkpoint(self, name: str, metrics: Dict[str, float], epoch: int):
        checkpoint = {
            "epoch": epoch,
            "config": asdict(self.config),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "tokenizer": self.tokenizer.state_dict(),
            "factor_columns": self.factor_columns,
            "dataset": self.args.dataset,
            "metrics": metrics,
            "args": vars(self.args),
        }
        path = self.output_dir / name
        torch.save(checkpoint, path)
        return path

    def visualize(self, epoch: int):
        batch = next(iter(self.val_loader))
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                self.extract_visual_features({key: value[:1] if isinstance(value, torch.Tensor) else value[:1] for key, value in batch.items()}),
                input_ids=None if self.args.disable_language else batch["input_ids"][:1].to(self.device),
            )
        visualize_causal_field(
            outputs["field"][0].cpu(),
            outputs["influence_matrix"][0].cpu(),
            str(self.output_dir / f"field_epoch_{epoch}.png"),
        )

    def train(self):
        all_metrics = []
        for epoch in range(1, self.args.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()

            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            all_metrics.append(record)
            with open(self.output_dir / "metrics.json", "w") as f:
                json.dump(all_metrics, f, indent=2)

            print(f"Epoch {epoch}: train={train_metrics} val={val_metrics}")
            metric = val_metrics.get("score_mae", val_metrics.get("total", float("inf")))
            if metric < self.best_metric:
                self.best_metric = metric
                path = self.save_checkpoint("best_model.pth", val_metrics, epoch)
                print(f"Saved best checkpoint: {path}")
            if epoch % self.args.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth", val_metrics, epoch)
            if epoch % self.args.vis_interval == 0:
                self.visualize(epoch)

        final_metrics = all_metrics[-1]["val"] if all_metrics else {}
        self.save_checkpoint("final_model.pth", final_metrics, self.args.epochs)


def main():
    parser = argparse.ArgumentParser(description="Train MetaCausalField")
    parser.add_argument("--dataset", default="Lung", choices=["Lung", "MAG9"])
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_propagation_steps", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--max_text_length", type=int, default=96)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_consistency", type=float, default=0.3)
    parser.add_argument("--lambda_counterfactual", type=float, default=0.5)
    parser.add_argument("--lambda_sparsity", type=float, default=0.01)
    parser.add_argument("--lambda_smoothness", type=float, default=0.001)
    parser.add_argument("--influence_top_k", type=int, default=None)
    parser.add_argument("--feature_noise_std", type=float, default=0.03)
    parser.add_argument("--intervention_radius", type=float, default=2.0)
    parser.add_argument("--max_counterfactual_pairs", type=int, default=None)
    parser.add_argument("--pretrained_backbone", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--backbone", default="resnet", choices=["resnet", "cached"])
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument("--use_frozen_language_tokens", action="store_true")
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--disable_language", action="store_true")
    parser.add_argument("--disable_lm", action="store_true")
    parser.add_argument("--disable_consistency", action="store_true")
    parser.add_argument("--disable_counterfactual", action="store_true")
    parser.add_argument("--enable_cf_trajectory_loss", action="store_true")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--vis_interval", type=int, default=5)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
