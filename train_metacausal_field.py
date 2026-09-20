"""
Train the local InfluenceField prototype end to end.

This script now covers the full research prototype loop:
- real image patch encoding
- language conditioning and language-model decoding
- score/factor supervision
- counterfactual supervision from generated cf_pd.csv files
- validation metrics and checkpoints
"""

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from frozen_backbones import CachedFrozenBackbone, load_torch_payload


TEXT_COLUMNS = {"Review", "description", "Demographics", "History", "ImagePath", "Image", "ImagePrompt"}


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG used by the local training/evaluation scripts."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # Older PyTorch versions do not support warn_only.
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """Give NumPy/Python deterministic, distinct seeds in each data worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def stable_int(value: Any, modulo: Optional[int] = None) -> int:
    """Return a process-independent integer derived from ``value``."""
    if isinstance(value, (dict, list, tuple)):
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    else:
        payload = str(value)
    number = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    return number % modulo if modulo else number


def split_dataframe(
    df: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    group_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic, group-disjoint train/validation/test partitions.

    If an explicit ``split`` column is present it is respected. Otherwise whole
    groups are shuffled with a seeded generator. The test partition is never
    folded into validation or training.
    """
    if len(df) < 3:
        raise ValueError("At least three rows/groups are required for train/val/test splitting.")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < val_fraction < 1.0:
        raise ValueError("train_fraction and val_fraction must be between 0 and 1.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must leave a non-empty test fraction.")

    if group_column is None:
        group_column = next(
            (
                name
                for name in (
                    "group_id", "video_id", "scene_id", "patient_id",
                    "subject_id", "sample_id", "id",
                )
                if name in df.columns
            ),
            None,
        )

    if "split" in df.columns and df["split"].notna().any():
        aliases = {"validation": "val", "dev": "val", "eval": "test"}
        labels = df["split"].fillna("").astype(str).str.strip().str.lower().replace(aliases)
        unknown = sorted(set(labels) - {"train", "val", "test"})
        if unknown:
            raise ValueError(f"Unsupported explicit split labels: {unknown}")
        partitions = tuple(
            df.loc[labels.eq(name)].reset_index(drop=True)
            for name in ("train", "val", "test")
        )
        if any(part.empty for part in partitions):
            raise ValueError("Explicit split column must contain non-empty train, val, and test partitions.")
        if group_column:
            group_sets = [set(part[group_column].astype(str)) for part in partitions]
            if (group_sets[0] & group_sets[1]) or (group_sets[0] & group_sets[2]) or (group_sets[1] & group_sets[2]):
                raise ValueError(
                    f"Explicit partitions leak groups from {group_column!r} across train/val/test."
                )
        return partitions  # type: ignore[return-value]

    groups = (
        df[group_column].astype(str)
        if group_column
        else pd.Series(df.index.astype(str), index=df.index)
    )
    unique_groups = np.asarray(sorted(groups.unique().tolist()), dtype=object)
    if len(unique_groups) < 3:
        raise ValueError("At least three distinct groups are required for leakage-safe splitting.")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_train = max(1, int(round(n_groups * train_fraction)))
    n_val = max(1, int(round(n_groups * val_fraction)))
    if n_train + n_val >= n_groups:
        overflow = n_train + n_val - (n_groups - 1)
        if n_train >= n_val and n_train - overflow >= 1:
            n_train -= overflow
        else:
            n_val -= overflow
    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val:])
    if not train_groups or not val_groups or not test_groups:
        raise RuntimeError("Unable to produce non-empty train/val/test group partitions.")

    return tuple(
        df.loc[groups.isin(selected)].reset_index(drop=True)
        for selected in (train_groups, val_groups, test_groups)
    )  # type: ignore[return-value]


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


CF_TARGET_TEXT_COLUMNS = (
    "CounterfactualReview",
    "counterfactual_text",
    "cf_text",
    "target_text",
)
CF_TARGET_SCORE_COLUMNS = (
    "CounterfactualScore",
    "counterfactual_score",
    "cf_score",
    "target_score",
)
CF_TARGET_MEDIA_COLUMNS = (
    "CounterfactualImage",
    "counterfactual_image",
    "cf_image",
)


def _nonempty_series(df: pd.DataFrame, columns: Tuple[str, ...]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for column in columns:
        if column in df:
            mask |= df[column].notna() & df[column].astype(str).str.strip().ne("")
    return mask


def is_explicit_modify_intervention(
    value: Any,
    *,
    feature_dim: Optional[int] = None,
    require_position: bool = True,
) -> bool:
    """Return whether a latent ``modify`` intervention is fully specified."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return False
    if not isinstance(value, dict) or not value:
        return False
    intervention_type = str(value.get("type", value.get("intervention_type", "modify"))).lower()
    if intervention_type not in {"modify", "attribute", "attribute_modify"}:
        return False
    raw_position = value.get("position", value.get("center"))
    if raw_position is None and {"x", "y"} <= set(value):
        raw_position = [value["x"], value["y"]]
    if require_position and not (
        isinstance(raw_position, (list, tuple)) and len(raw_position) >= 2
    ):
        return False
    direction = value.get("direction", value.get("attribute_direction"))
    if not isinstance(direction, (list, tuple)) or not direction:
        return False
    if feature_dim is not None and len(direction) != feature_dim:
        return False
    return True


def prepare_counterfactual_supervision(
    cf_df: pd.DataFrame,
    *,
    require_trajectory_media: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Keep only counterfactual rows with explicit, auditable targets.

    Legacy ``cf_pd.csv`` files often contain a generated row named ``Review``
    and a generic ``score`` but no factual/counterfactual pair or grounded
    intervention.  Treating those columns as paired supervision silently trains
    on invented labels, so generic names are deliberately not accepted here.
    """
    if cf_df.empty:
        return cf_df.copy(), {"input_rows": 0, "usable_rows": 0}

    source_media = _nonempty_series(cf_df, ("Image", "ImagePath", "image_path"))
    target_text = _nonempty_series(cf_df, CF_TARGET_TEXT_COLUMNS)
    target_score = pd.Series(False, index=cf_df.index)
    for column in CF_TARGET_SCORE_COLUMNS:
        if column in cf_df:
            target_score |= pd.to_numeric(cf_df[column], errors="coerce").notna()
    target_media = _nonempty_series(cf_df, CF_TARGET_MEDIA_COLUMNS)
    intervention = (
        cf_df["intervention"].map(is_explicit_modify_intervention)
        if "intervention" in cf_df
        else pd.Series(False, index=cf_df.index)
    )
    has_target = target_media if require_trajectory_media else (target_text | target_score)
    usable = source_media & intervention & has_target
    audit = {
        "input_rows": int(len(cf_df)),
        "usable_rows": int(usable.sum()),
        "missing_source_media": int((~source_media).sum()),
        "missing_explicit_intervention": int((~intervention).sum()),
        "missing_explicit_target": int((~has_target).sum()),
    }
    return cf_df.loc[usable].reset_index(drop=True), audit


def label_to_class(value) -> int:
    if pd.isna(value):
        return -100
    value = int(float(value))
    if value < 0:
        return 0
    if value > 0:
        return 2
    return 1


def load_image(
    path: str,
    fallback_size=(224, 224),
    *,
    allow_missing_media: bool = False,
) -> Image.Image:
    """Load an image, failing loudly unless a synthetic fallback is requested."""
    if not path:
        if allow_missing_media:
            return Image.new("RGB", fallback_size, color="white")
        raise FileNotFoundError("Image path is empty. Use --allow_missing_media only for smoke tests.")
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except FileNotFoundError:
        if allow_missing_media:
            return Image.new("RGB", fallback_size, color="white")
        raise FileNotFoundError(
            f"Required image not found: {path}. "
            "Use --allow_missing_media only for explicit synthetic smoke tests."
        ) from None
    except Exception as exc:
        if allow_missing_media:
            return Image.new("RGB", fallback_size, color="white")
        raise RuntimeError(f"Failed to decode required image {path}: {exc}") from exc


class CausalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        tokenizer: SimpleTokenizer,
        transform,
        allow_missing_media: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.factor_columns = factor_columns
        self.tokenizer = tokenizer
        self.transform = transform
        self.allow_missing_media = allow_missing_media

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.transform(load_image(
            str(row["ImagePath"]),
            allow_missing_media=self.allow_missing_media,
        ))
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
    def __init__(
        self,
        cf_df: pd.DataFrame,
        tokenizer: SimpleTokenizer,
        transform,
        max_pairs: Optional[int] = None,
        allow_missing_media: bool = False,
    ):
        self.cf_df = cf_df.reset_index(drop=True)
        if max_pairs:
            self.cf_df = self.cf_df.iloc[:max_pairs].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.allow_missing_media = allow_missing_media

    def __len__(self):
        return len(self.cf_df)

    def __getitem__(self, idx):
        row = self.cf_df.iloc[idx]
        image_path = str(row.get("Image", row.get("ImagePath", "")))
        raw_cf_image_path = row.get(
            "CounterfactualImage",
            row.get("counterfactual_image", row.get("cf_image", "")),
        )
        cf_image_path = "" if pd.isna(raw_cf_image_path) else str(raw_cf_image_path).strip()
        source_text = str(row.get("Review", row.get("source_text", "")))
        target_text = next(
            (
                str(row[column])
                for column in CF_TARGET_TEXT_COLUMNS
                if column in row and pd.notna(row[column]) and str(row[column]).strip()
            ),
            "",
        )
        raw_cf_score = next(
            (
                pd.to_numeric(row[column], errors="coerce")
                for column in CF_TARGET_SCORE_COLUMNS
                if column in row and pd.notna(row[column])
            ),
            np.nan,
        )
        scenario = str(row.get("scenario", "counterfactual"))
        intervention = row.get("intervention", {})
        if isinstance(intervention, str):
            try:
                intervention = json.loads(intervention)
            except json.JSONDecodeError:
                intervention = {}
        if not isinstance(intervention, dict):
            intervention = {}
        image = self.transform(load_image(image_path, allow_missing_media=self.allow_missing_media))
        cf_image = (
            self.transform(load_image(cf_image_path, allow_missing_media=self.allow_missing_media))
            if cf_image_path
            else image.clone()
        )
        return {
            "image": image,
            "cf_image": cf_image,
            "input_ids": torch.tensor(self.tokenizer.encode(source_text), dtype=torch.long),
            "cf_input_ids": torch.tensor(self.tokenizer.encode(target_text), dtype=torch.long),
            "cf_text_available": torch.tensor(bool(target_text.strip()), dtype=torch.bool),
            "cf_score": torch.tensor(float(raw_cf_score), dtype=torch.float32),
            "cf_score_available": torch.tensor(bool(pd.notna(raw_cf_score)), dtype=torch.bool),
            "scenario": scenario,
            "intervention": intervention,
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
        seed = stable_int(scenario)
        x = 0.15 + 0.7 * ((seed % 997) / 996.0)
        y = 0.15 + 0.7 * (((seed // 997) % 991) / 990.0)
        coords.append([x, y])
    return torch.tensor(coords, dtype=torch.float32, device=device)


def scenario_directions(scenarios: List[str], feature_dim: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(feature_dim, dtype=torch.float32, device=device).unsqueeze(0)
    seeds = torch.tensor([stable_int(s, 10007) for s in scenarios], dtype=torch.float32, device=device).unsqueeze(1)
    directions = torch.sin(base * 0.017 + seeds * 0.001)
    return F.normalize(directions, dim=-1)


def intervention_tensors(
    interventions: List[Dict[str, Any]],
    scenarios: List[str],
    feature_dim: int,
    device: torch.device,
    *,
    allow_synthetic_fallback: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolve explicit latent edits, optionally enabling a labelled smoke fallback."""
    stable_labels = [
        json.dumps(item, sort_keys=True, default=str) if item else scenario
        for item, scenario in zip(interventions, scenarios)
    ]
    positions = scenario_positions(stable_labels, device)
    directions = scenario_directions(stable_labels, feature_dim, device)
    invalid = []
    for idx, intervention in enumerate(interventions):
        if not is_explicit_modify_intervention(
            intervention,
            feature_dim=feature_dim,
            require_position=True,
        ):
            invalid.append(idx)
            continue
        raw_position = intervention.get("position", intervention.get("center"))
        if raw_position is None and {"x", "y"} <= set(intervention):
            raw_position = [intervention["x"], intervention["y"]]
        if isinstance(raw_position, (list, tuple)) and len(raw_position) >= 2:
            positions[idx] = torch.tensor(
                raw_position[:2], dtype=torch.float32, device=device
            ).clamp(0.0, 1.0)
        raw_direction = intervention.get("direction", intervention.get("attribute_direction"))
        if isinstance(raw_direction, (list, tuple)) and len(raw_direction) == feature_dim:
            vector = torch.tensor(raw_direction, dtype=torch.float32, device=device)
            if torch.linalg.vector_norm(vector) > 0:
                directions[idx] = F.normalize(vector, dim=0)
    if invalid and not allow_synthetic_fallback:
        raise ValueError(
            "Counterfactual modify interventions require normalized position and an explicit "
            f"direction of length {feature_dim}; invalid batch indices: {invalid[:8]}. "
            "Synthetic hash-based edits are available only in explicit smoke-test paths."
        )
    return positions, directions


class Trainer:
    def __init__(self, args):
        self.args = args
        set_global_seed(args.seed, deterministic=not args.allow_nondeterministic)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.meta_df, _, self.factor_columns = prepare_metadata(args.dataset)
        train_df, val_df, test_df = split_dataframe(
            self.meta_df,
            seed=args.seed,
            train_fraction=args.train_split,
            val_fraction=args.val_split,
            group_column=args.group_column,
        )

        cf_file = find_counterfactual_file(args.dataset)
        cf_df = pd.read_csv(cf_file) if cf_file else pd.DataFrame()
        if not cf_df.empty:
            cf_df, cf_audit = prepare_counterfactual_supervision(
                cf_df,
                require_trajectory_media=args.enable_cf_trajectory_loss,
            )
            cf_source_column = next(
                (name for name in ("Image", "ImagePath", "image_path") if name in cf_df),
                None,
            )
            if cf_source_column and "ImagePath" in train_df:
                train_media = set(train_df["ImagePath"].astype(str))
                cf_train_df = cf_df.loc[
                    cf_df[cf_source_column].astype(str).isin(train_media)
                ].reset_index(drop=True)
            elif "split" in cf_df:
                labels = cf_df["split"].astype(str).str.strip().str.lower()
                cf_train_df = cf_df.loc[labels.eq("train")].reset_index(drop=True)
            else:
                # Without a join key or an explicit split, using the rows could
                # leak validation/test sources into training.
                cf_train_df = cf_df.iloc[:0].copy()
            cf_audit["selected_training_rows"] = int(len(cf_train_df))
            print(f"Counterfactual supervision audit: {cf_audit}")
        else:
            cf_train_df = cf_df

        # Vocabulary construction is a learned preprocessing step: fit it only
        # on the training partition to avoid validation/test leakage.
        texts = train_df["Review"].tolist()
        for column in ("Review", "CounterfactualReview", "counterfactual_text", "target_text"):
            if not cf_train_df.empty and column in cf_train_df:
                texts.extend(cf_train_df[column].fillna("").astype(str).tolist())
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
            self.cached_backbone = CachedFrozenBackbone(
                args.feature_cache,
                feature_dim=args.feature_dim,
                allow_smoke_cache=args.allow_missing_media,
            )
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
            checkpoint = load_torch_payload(args.init_checkpoint, map_location=self.device)
            state = checkpoint.get("model_state_dict", checkpoint)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if (missing or unexpected) and not args.allow_partial_init:
                raise RuntimeError(
                    "Initialization checkpoint/model mismatch. Pass --allow_partial_init only "
                    "for an intentional transfer-learning experiment. "
                    f"missing={missing[:8]}, unexpected={unexpected[:8]}"
                )
            print(
                f"Initialized from {args.init_checkpoint} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )

        self.criterion = MetaCausalLoss(self.config)
        self.criterion.lambda_consistency = args.lambda_consistency
        self.criterion.lambda_counterfactual = args.lambda_counterfactual

        train_generator = torch.Generator().manual_seed(args.seed)
        cf_generator = torch.Generator().manual_seed(args.seed + 1)
        dataset_kwargs = {"allow_missing_media": args.allow_missing_media}
        self.train_loader = DataLoader(
            CausalDataset(
                train_df, self.factor_columns, self.tokenizer, self.transform, **dataset_kwargs,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate,
            worker_init_fn=seed_worker,
            generator=train_generator,
        )
        self.val_loader = DataLoader(
            CausalDataset(
                val_df, self.factor_columns, self.tokenizer, self.transform, **dataset_kwargs,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            worker_init_fn=seed_worker,
        )
        self.test_loader = DataLoader(
            CausalDataset(
                test_df, self.factor_columns, self.tokenizer, self.transform, **dataset_kwargs,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            worker_init_fn=seed_worker,
        )
        if cf_file and not cf_train_df.empty:
            self.cf_loader = DataLoader(
                CounterfactualPairDataset(
                    cf_train_df,
                    self.tokenizer,
                    self.transform,
                    args.max_counterfactual_pairs,
                    allow_missing_media=args.allow_missing_media,
                ),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate,
                worker_init_fn=seed_worker,
                generator=cf_generator,
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
        id_column = next(
            (name for name in (args.group_column, "id", "sample_id", "ImagePath") if name and name in self.meta_df),
            None,
        )

        def partition_ids(partition):
            return (
                partition[id_column].astype(str).tolist()
                if id_column
                else partition.index.astype(str).tolist()
            )

        split_record = {
            "seed": args.seed,
            "id_column": id_column,
            "train_ids": partition_ids(train_df),
            "val_ids": partition_ids(val_df),
            "test_ids": partition_ids(test_df),
        }
        self.split_record = split_record
        with open(self.output_dir / "data_split.json", "w") as handle:
            json.dump(split_record, handle, indent=2)
        print(
            f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, "
            f"held-out Test samples: {len(test_df)}"
        )

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

        if train and not self.args.disable_consistency:
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
        positions, directions = intervention_tensors(
            cf_batch["intervention"],
            cf_batch["scenario"],
            self.config.feature_dim,
            self.device,
        )
        cf_outputs = self.model.counterfactual_forward(
            visual_features,
            intervention_type="modify",
            intervention_params={
                "position": positions,
                "direction": directions,
                "radius": self.args.intervention_radius,
            },
            language_tokens=language_tokens,
            input_ids=None if self.args.disable_language else input_ids,
            decoder_input_ids=None if self.args.disable_lm else cf_input_ids[:, :-1],
            num_rollout_steps=self.args.num_propagation_steps,
        )
        targets = {}
        if cf_batch["cf_score_available"].any():
            targets["cf_score_targets"] = cf_batch["cf_score"].to(self.device)
        if not self.args.disable_lm and cf_batch["cf_text_available"].any():
            cf_lm_targets = cf_input_ids[:, 1:].clone()
            cf_lm_targets[
                ~cf_batch["cf_text_available"].to(self.device)
            ] = self.config.pad_token_id
            targets["cf_lm_targets"] = cf_lm_targets
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
    def evaluate_loader(self, loader: DataLoader, description: str) -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {}
        score_abs = []
        factor_correct = 0
        factor_total = 0

        for batch in tqdm(loader, desc=description):
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

        metrics = {key: value / len(loader) for key, value in totals.items()}
        metrics["score_mae"] = float(np.mean(score_abs)) if score_abs else math.nan
        metrics["factor_acc"] = factor_correct / factor_total if factor_total else math.nan
        if "lm" in metrics:
            metrics["perplexity"] = float(math.exp(min(metrics["lm"], 20.0)))
        return metrics

    def validate(self) -> Dict[str, float]:
        return self.evaluate_loader(self.val_loader, "Validation")

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
            "data_split": self.split_record,
            "supervision": {
                "factor_labels": bool(self.factor_columns),
                "factor_localizer": False,
                "factor_graph": False,
                "note": "Factor class labels do not supervise factor-to-position localization.",
            },
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
        best_path = self.output_dir / "best_model.pth"
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
        if best_path.exists():
            checkpoint = load_torch_payload(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            test_metrics = self.evaluate_loader(self.test_loader, "Held-out test")
            with open(self.output_dir / "test_metrics.json", "w") as handle:
                json.dump(
                    {
                        "selected_epoch": checkpoint["epoch"],
                        "selection_metric": "validation score_mae (fallback: total loss)",
                        "test_metrics": test_metrics,
                    },
                    handle,
                    indent=2,
                )
            print(
                f"Evaluated held-out test once using validation-selected epoch "
                f"{checkpoint['epoch']}."
            )


def main():
    parser = argparse.ArgumentParser(description="Train MetaCausalField")
    parser.add_argument("--dataset", default="Lung", choices=["Lung", "MAG9"])
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--group_column", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_nondeterministic",
        action="store_true",
        help="Permit faster nondeterministic kernels; RNGs are still seeded.",
    )
    parser.add_argument(
        "--allow_missing_media",
        action="store_true",
        help="Use white placeholder images for smoke tests only (never for reported metrics).",
    )
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
    parser.add_argument(
        "--allow_partial_init",
        action="store_true",
        help="Allow intentional partial checkpoint loading for transfer experiments.",
    )
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
