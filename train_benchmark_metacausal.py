"""Train MetaCausalField directly on unified benchmark manifests.

This is the runnable path for CLEVRER, Causal3DIdent/CITRIS, and Causal-VidQA:
it supports image/video samples, language-conditioned decoding, factor targets,
counterfactual paired answers in the manifest, and checkpoint export compatible
with the evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from benchmark_datasets import CausalSample, UnifiedCausalDataset, load_benchmark
from frozen_backbones import (
    CachedFrozenBackbone,
    QwenVLFrozenBackbone,
    download_qwen_weights,
    load_torch_payload,
)
from metacausal_field import (
    CausalFieldConfig,
    MetaCausalLoss,
    MultimodalCausalField,
    SimpleTokenizer,
    VisualEncoder,
)
from train_metacausal_field import (
    collate,
    is_explicit_modify_intervention,
    seed_worker,
    set_global_seed,
)

_SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "eval": "test",
}


def _sample_leakage_keys(sample: CausalSample) -> set[str]:
    """Include factual and counterfactual media in leakage checks."""
    values = {
        sample.group_id,
        sample.video_path,
        sample.image_path,
        sample.counterfactual_video_path,
        sample.counterfactual_image_path,
    }
    keys = {str(value) for value in values if value}
    return keys or {str(sample.sample_id)}


def partition_samples(samples):
    """Require explicit, non-empty, group-disjoint train/val/test splits.

    Benchmark test data must never be repurposed as validation data. Media-level
    keys are checked as well as sample IDs so multiple questions about one video
    cannot silently straddle partitions.
    """
    partitions = {"train": [], "val": [], "test": []}
    unknown = set()
    for sample in samples:
        raw_label = str(sample.split or "").strip().lower()
        label = _SPLIT_ALIASES.get(raw_label)
        if label is None:
            unknown.add(raw_label or "<empty>")
            continue
        partitions[label].append(sample)
    if unknown:
        raise ValueError(f"Unsupported benchmark split labels: {sorted(unknown)}")
    missing = [name for name, partition in partitions.items() if not partition]
    if missing:
        raise ValueError(
            "Benchmark manifests must provide non-empty, explicit train/val/test "
            f"partitions; missing: {', '.join(missing)}."
        )

    id_sets = {
        name: {str(sample.sample_id) for sample in partition}
        for name, partition in partitions.items()
    }
    group_sets = {
        name: set().union(*(_sample_leakage_keys(sample) for sample in partition))
        for name, partition in partitions.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        duplicate_ids = id_sets[left] & id_sets[right]
        duplicate_groups = group_sets[left] & group_sets[right]
        if duplicate_ids or duplicate_groups:
            details = []
            if duplicate_ids:
                details.append(f"sample_ids={sorted(duplicate_ids)[:5]}")
            if duplicate_groups:
                details.append(f"media_groups={sorted(duplicate_groups)[:5]}")
            raise ValueError(
                f"Data leakage between {left} and {right}: " + "; ".join(details)
            )
    return partitions["train"], partitions["val"], partitions["test"]


def build_tokenizer(samples, vocab_size, max_length):
    texts = []
    for sample in samples:
        texts.extend([
            sample.text,
            sample.question,
            sample.answer,
            sample.factual_text,
            sample.counterfactual_text,
            sample.cf_answer or "",
        ])
    return SimpleTokenizer.build(texts, vocab_size=vocab_size, max_length=max_length)


def trainable_module_state_dict(module):
    """Capture only trainable adapter parameters from an optional backbone."""
    if module is None:
        return None
    trainable_names = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    if not trainable_names:
        return None
    return {
        name: tensor.detach().cpu()
        for name, tensor in module.state_dict().items()
        if name in trainable_names
    }


def make_benchmark_loader(
    samples,
    tokenizer,
    transform,
    factor_names,
    *,
    num_video_frames,
    batch_size,
    shuffle,
    num_workers,
    seed,
    allow_missing_media,
    skip_media_loading=False,
):
    """Build a loader whose shuffling and worker RNGs are reproducible."""
    dataset = UnifiedCausalDataset(
        samples,
        tokenizer=tokenizer,
        transform=transform,
        factor_names=factor_names,
        num_video_frames=num_video_frames,
        allow_missing_media=allow_missing_media,
        skip_media_loading=skip_media_loading,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def benchmark_intervention_tensors(batch, feature_dim, device):
    """Resolve explicit edits, using neutral values only for masked batch slots."""
    if feature_dim < 1:
        raise ValueError("feature_dim must be positive")
    interventions = batch.get("intervention", [{} for _ in batch["sample_id"]])
    normalized_interventions = [item if isinstance(item, dict) else {} for item in interventions]
    batch_size = len(normalized_interventions)
    positions = torch.full((batch_size, 2), 0.5, dtype=torch.float32, device=device)
    directions = torch.zeros((batch_size, feature_dim), dtype=torch.float32, device=device)
    directions[:, 0] = 1.0
    for index, intervention in enumerate(normalized_interventions):
        raw_position = intervention.get("position", intervention.get("center"))
        if raw_position is None and {"x", "y"} <= set(intervention):
            raw_position = [intervention["x"], intervention["y"]]
        if isinstance(raw_position, (list, tuple)) and len(raw_position) >= 2:
            try:
                positions[index] = torch.tensor(
                    [float(raw_position[0]), float(raw_position[1])],
                    dtype=torch.float32,
                    device=device,
                ).clamp(0.0, 1.0)
            except (TypeError, ValueError):
                pass
        raw_direction = intervention.get(
            "direction",
            intervention.get("attribute_direction"),
        )
        if isinstance(raw_direction, (list, tuple)) and len(raw_direction) == feature_dim:
            try:
                vector = torch.tensor(raw_direction, dtype=torch.float32, device=device)
                norm = torch.linalg.vector_norm(vector)
                if torch.isfinite(norm) and norm > 0:
                    directions[index] = vector / norm
            except (TypeError, ValueError):
                pass
    has_explicit_position = torch.tensor(
        [
            bool(
                item.get("position") is not None
                or item.get("center") is not None
                or {"x", "y"} <= set(item)
            )
            for item in normalized_interventions
        ],
        dtype=torch.bool,
        device=device,
    )

    boxes = batch["bbox"].to(device)
    valid = (
        (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
        & torch.all((boxes >= 0.0) & (boxes <= 1.0), dim=1)
    )
    centers = torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], dim=-1)
    use_sample_bbox = valid & ~has_explicit_position
    positions = torch.where(use_sample_bbox.unsqueeze(-1), centers, positions)

    for i, obj_id in enumerate(batch["object_id"]):
        target = str(obj_id or "")
        intervention = normalized_interventions[i]
        if isinstance(intervention, dict):
            target = str(intervention.get("target", intervention.get("object_id", target)))
        found = None
        for obj in batch.get("objects", [[]])[i] or []:
            if not isinstance(obj, dict):
                continue
            names = {
                str(obj.get("id", "")),
                str(obj.get("object_id", "")),
                str(obj.get("name", "")),
                str(obj.get("category", "")),
                str(obj.get("type", "")),
            }
            box = obj.get("bbox")
            if target and target in names and box and len(box) >= 4:
                values = [float(value) for value in box[:4]]
                if all(0.0 <= value <= 1.0 for value in values):
                    found = [
                        (values[0] + values[2]) / 2.0,
                        (values[1] + values[3]) / 2.0,
                    ]
                    break
        if found is not None and not valid[i] and not has_explicit_position[i]:
            positions[i] = torch.tensor(found, dtype=torch.float32, device=device)
    return positions.clamp(0.0, 1.0), directions


def load_media_frame(
    image_path: str = "",
    video_path: str = "",
    *,
    allow_missing_media: bool = False,
):
    errors = []
    if image_path:
        try:
            with Image.open(image_path) as source:
                return source.convert("RGB")
        except (OSError, ValueError) as exc:
            errors.append(f"image={image_path!r}: {exc}")
    if video_path:
        try:
            from torchvision.io import read_video
            frames, _, _ = read_video(video_path, pts_unit="sec")
            if len(frames) > 0:
                return Image.fromarray(frames[len(frames) // 2].numpy()).convert("RGB")
            errors.append(f"video={video_path!r}: contained no frames")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            errors.append(f"video={video_path!r}: {exc}")
    if allow_missing_media:
        return Image.new("RGB", (224, 224), "white")
    requested = ", ".join(path for path in (image_path, video_path) if path) or "<empty path>"
    details = "; ".join(errors)
    raise FileNotFoundError(
        f"Required media could not be loaded ({requested}). {details} "
        "Pass --allow_missing_media only for explicit smoke tests."
    )


def backbone_features(model, frozen_backbone, batch, args, device):
    if args.backbone == "qwen":
        texts = batch["text"] if "text" in batch else batch["answer"]
        image_paths = batch["image_path"]
        video_paths = batch["video_path"]
        if (
            args.qwen_video_input == "native"
            and all(not path for path in image_paths)
            and all(bool(path) for path in video_paths)
        ):
            outputs = frozen_backbone.encode_video(video_paths, texts, device)
        else:
            images = [
                load_media_frame(
                    image_path,
                    video_path,
                    allow_missing_media=args.allow_missing_media,
                )
                for image_path, video_path in zip(image_paths, video_paths)
            ]
            outputs = frozen_backbone.encode_pil(images, texts, device)
        return outputs["visual_features"], outputs.get("language_tokens")
    if frozen_backbone is not None:
        outputs = frozen_backbone.encode_by_keys(batch["feature_key"], device)
        return outputs["visual_features"], outputs.get("language_tokens")
    return model.visual_encoder(batch["image"].to(device)), None


def run_epoch(
    model,
    criterion,
    loader,
    optimizer,
    args,
    device,
    train=True,
    frozen_backbone=None,
    phase=None,
):
    model.train(train)
    if frozen_backbone is not None:
        if getattr(frozen_backbone, "trainable_lora", False):
            frozen_backbone.train(train)
        else:
            frozen_backbone.eval()
    totals = {}
    for batch in tqdm(loader, desc=phase or ("train" if train else "val")):
        with torch.set_grad_enabled(train):
            input_ids = batch["input_ids"].to(device)
            answer_ids = batch["answer_ids"].to(device)
            visual, frozen_language_tokens = backbone_features(model, frozen_backbone, batch, args, device)
            language_tokens = frozen_language_tokens if args.use_frozen_language_tokens else None
            outputs = model(
                visual,
                language_tokens=language_tokens,
                input_ids=input_ids,
                decoder_input_ids=answer_ids[:, :-1],
            )
            targets = {
                "lm_targets": answer_ids[:, 1:],
                "factor_targets": batch["factor_targets"].to(device),
            }
            losses = criterion(outputs, targets)

            if train and not args.disable_consistency:
                noise = torch.randn_like(visual) * args.feature_noise_std
                aug_outputs = model(visual + noise, language_tokens=language_tokens, input_ids=input_ids)
                losses["consistency"] = criterion.causal_consistency_loss(outputs["field"], aug_outputs["field"])
                losses["total"] = losses["total"] + args.lambda_consistency * losses["consistency"]

            cf_mask = []
            for index, (answer, intervention) in enumerate(
                zip(batch["cf_answer"], batch["intervention"])
            ):
                box = batch["bbox"][index]
                normalized_box = bool(
                    box[2] > box[0]
                    and box[3] > box[1]
                    and torch.all((box >= 0.0) & (box <= 1.0))
                )
                explicit_edit = is_explicit_modify_intervention(
                    intervention,
                    feature_dim=model.config.feature_dim,
                    require_position=True,
                )
                bbox_aligned_edit = normalized_box and is_explicit_modify_intervention(
                    intervention,
                    feature_dim=model.config.feature_dim,
                    require_position=False,
                )
                cf_mask.append(bool(answer) and (explicit_edit or bbox_aligned_edit))
            if any(cf_mask) and not args.disable_counterfactual:
                cf_texts = [cf if cf else ans for cf, ans in zip(batch["cf_answer"], batch["answer"])]
                cf_ids = torch.tensor([loader.dataset.tokenizer.encode(t) for t in cf_texts], dtype=torch.long, device=device)
                positions, directions = benchmark_intervention_tensors(
                    batch,
                    model.config.feature_dim,
                    device,
                )
                cf_outputs = model.counterfactual_forward(
                    visual,
                    intervention_type="modify",
                    intervention_params={
                        "position": positions,
                        "direction": directions,
                        "radius": args.intervention_radius,
                    },
                    language_tokens=language_tokens,
                    input_ids=input_ids,
                    decoder_input_ids=cf_ids[:, :-1],
                    num_rollout_steps=model.config.num_propagation_steps,
                )
                cf_targets = cf_ids[:, 1:].clone()
                valid_cf_mask = torch.tensor(cf_mask, dtype=torch.bool, device=device)
                cf_targets[~valid_cf_mask] = model.config.pad_token_id
                cf_losses = criterion(cf_outputs, {"cf_lm_targets": cf_targets})
                has_cf_media = [
                    bool(image_path or video_path)
                    for image_path, video_path in zip(
                        batch.get("counterfactual_image_path", []),
                        batch.get("counterfactual_video_path", []),
                    )
                ]
                if (
                    args.enable_cf_trajectory_loss
                    and all(cf_mask)
                    and has_cf_media
                    and all(has_cf_media)
                ):
                    teacher_batch = dict(batch)
                    teacher_batch["image"] = batch["cf_image"]
                    teacher_batch["feature_key"] = batch["cf_feature_key"]
                    teacher_batch["image_path"] = batch["counterfactual_image_path"]
                    teacher_batch["video_path"] = batch["counterfactual_video_path"]
                    with torch.no_grad():
                        teacher_visual, _ = backbone_features(model, frozen_backbone, teacher_batch, args, device)
                        teacher_outputs = model(teacher_visual)
                    trajectory_losses = criterion(cf_outputs, {
                        "cf_target_trajectory": [
                            state.detach() for state in teacher_outputs.get("field_trajectory", [])[1:]
                        ]
                    })
                    for key, value in trajectory_losses.items():
                        cf_losses[f"trajectory_{key}"] = value
                    cf_losses["total"] = cf_losses["total"] + trajectory_losses["total"]
                losses["counterfactual_supervised"] = cf_losses["total"]
                losses["total"] = losses["total"] + cf_losses["total"]

        if train:
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            clip_params = list(model.parameters())
            if frozen_backbone is not None and getattr(frozen_backbone, "trainable_lora", False):
                clip_params.extend(param for param in frozen_backbone.parameters() if param.requires_grad)
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            optimizer.step()

        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
    return {k: v / max(1, len(loader)) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Train MetaCausalField on benchmark manifest")
    parser.add_argument("--dataset", required=True, choices=["CLEVRER", "Causal3DIdent", "CITRIS", "Causal-VidQA"])
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_dir", default="./benchmark_outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_nondeterministic",
        action="store_true",
        help="Allow nondeterministic kernels. Seeded RNGs are still used.",
    )
    parser.add_argument(
        "--allow_missing_media",
        action="store_true",
        help="Use blank media only for explicit smoke tests; disabled by default.",
    )
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_propagation_steps", type=int, default=3)
    parser.add_argument("--num_video_frames", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--max_text_length", type=int, default=96)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lambda_consistency", type=float, default=0.3)
    parser.add_argument("--lambda_counterfactual", type=float, default=0.5)
    parser.add_argument("--lambda_sparsity", type=float, default=0.01)
    parser.add_argument("--lambda_smoothness", type=float, default=0.001)
    parser.add_argument("--influence_top_k", type=int, default=None)
    parser.add_argument("--feature_noise_std", type=float, default=0.03)
    parser.add_argument("--intervention_radius", type=float, default=2.0)
    parser.add_argument("--backbone", default="resnet", choices=["resnet", "cached", "qwen"])
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument("--qwen_model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--download_qwen", action="store_true")
    parser.add_argument("--qwen_cache_dir", default=None)
    parser.add_argument("--qwen_revision", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--qwen_video_input", default="native", choices=["native", "middle_frame"])
    parser.add_argument("--train_qwen_lora", action="store_true")
    parser.add_argument("--qwen_lora_r", type=int, default=8)
    parser.add_argument("--qwen_lora_alpha", type=int, default=16)
    parser.add_argument("--qwen_lora_dropout", type=float, default=0.05)
    parser.add_argument("--qwen_lora_targets", default=None, help="Comma-separated LoRA target modules.")
    parser.add_argument("--use_frozen_language_tokens", action="store_true")
    parser.add_argument("--disable_consistency", action="store_true")
    parser.add_argument("--disable_counterfactual", action="store_true")
    parser.add_argument("--enable_cf_trajectory_loss", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_global_seed(args.seed, deterministic=not args.allow_nondeterministic)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    samples, manifest_factor_names = load_benchmark(
        args.dataset,
        args.manifest_path,
        args.data_root,
        seed=args.seed,
    )
    train_samples, val_samples, test_samples = partition_samples(samples)
    # The output-head vocabulary is learned metadata.  Derive it from training
    # only; test-only factor names remain available solely as gold references in
    # separate evaluation code.
    factor_names = sorted({name for sample in train_samples for name in sample.factors})
    heldout_only_factors = sorted(set(manifest_factor_names) - set(factor_names))
    if heldout_only_factors:
        print(f"Held-out-only factor names excluded from training heads: {heldout_only_factors}")
    # Vocabulary construction is learned preprocessing, so it is fit on train
    # only. Validation/test words map to <unk> instead of leaking into the model.
    tokenizer = build_tokenizer(train_samples, args.vocab_size, args.max_text_length)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    loader_kwargs = {
        "tokenizer": tokenizer,
        "transform": transform,
        "factor_names": factor_names,
        "num_video_frames": args.num_video_frames,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "allow_missing_media": args.allow_missing_media,
        "skip_media_loading": args.backbone == "cached",
    }
    train_loader = make_benchmark_loader(
        train_samples,
        shuffle=True,
        seed=args.seed,
        **loader_kwargs,
    )
    val_loader = make_benchmark_loader(
        val_samples,
        shuffle=False,
        seed=args.seed + 1,
        **loader_kwargs,
    )
    test_loader = make_benchmark_loader(
        test_samples,
        shuffle=False,
        seed=args.seed + 2,
        **loader_kwargs,
    )

    config = CausalFieldConfig(
        feature_dim=args.feature_dim,
        num_heads=args.num_heads,
        num_propagation_steps=args.num_propagation_steps,
        intervention_radius=args.intervention_radius,
        vocab_size=tokenizer.vocab_size,
        max_text_length=args.max_text_length,
        lambda_consistency=args.lambda_consistency,
        lambda_counterfactual=args.lambda_counterfactual,
        lambda_sparsity=args.lambda_sparsity,
        lambda_smoothness=args.lambda_smoothness,
        influence_top_k=args.influence_top_k,
    )
    frozen_backbone = None
    if args.backbone == "cached":
        if not args.feature_cache:
            raise ValueError("--feature_cache is required when --backbone cached")
        frozen_backbone = CachedFrozenBackbone(
            args.feature_cache,
            feature_dim=args.feature_dim,
            allow_smoke_cache=args.allow_missing_media,
        ).to(device)
        config.feature_dim = frozen_backbone.feature_dim
        args.feature_dim = frozen_backbone.feature_dim
    elif args.backbone == "qwen":
        qwen_path = args.qwen_model
        if args.download_qwen:
            qwen_path = download_qwen_weights(
                model_id=args.qwen_model,
                local_dir=args.qwen_cache_dir,
                revision=args.qwen_revision,
                local_files_only=args.local_files_only,
            )
        frozen_backbone = QwenVLFrozenBackbone(
            qwen_path,
            feature_dim=args.feature_dim,
            revision=args.qwen_revision if not Path(qwen_path).exists() else None,
            local_files_only=args.local_files_only,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            trainable_lora=args.train_qwen_lora,
            lora_r=args.qwen_lora_r,
            lora_alpha=args.qwen_lora_alpha,
            lora_dropout=args.qwen_lora_dropout,
            lora_target_modules=[x.strip() for x in args.qwen_lora_targets.split(",") if x.strip()]
            if args.qwen_lora_targets
            else None,
        ).to(device)
        config.feature_dim = frozen_backbone.feature_dim
        args.feature_dim = frozen_backbone.feature_dim

    model = MultimodalCausalField(
        config,
        visual_encoder=None if frozen_backbone is not None else VisualEncoder(args.feature_dim),
        vocab_size=tokenizer.vocab_size,
        num_factors=len(factor_names),
        enable_language=True,
    ).to(device)
    criterion = MetaCausalLoss(config)
    criterion.lambda_counterfactual = args.lambda_counterfactual
    trainable_params = list(model.parameters())
    if frozen_backbone is not None and getattr(frozen_backbone, "trainable_lora", False):
        trainable_params.extend(param for param in frozen_backbone.parameters() if param.requires_grad)
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)

    output_dir = Path(args.output_dir) / f"run_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_record = {
        "seed": args.seed,
        "id_column": "sample_id",
        "group_key": "video_path || image_path || feature_key || sample_id",
        "train_ids": [sample.sample_id for sample in train_samples],
        "val_ids": [sample.sample_id for sample in val_samples],
        "test_ids": [sample.sample_id for sample in test_samples],
    }
    with open(output_dir / "data_split.json", "w") as f:
        json.dump(split_record, f, indent=2)
    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            args,
            device,
            train=True,
            frozen_backbone=frozen_backbone,
            phase="train",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                criterion,
                val_loader,
                optimizer,
                args,
                device,
                train=False,
                frozen_backbone=frozen_backbone,
                phase="val",
            )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(history, f, indent=2)
        metric = val_metrics.get("total", np.inf)
        if metric < best:
            best = metric
            torch.save({
                "epoch": epoch,
                "config": asdict(config),
                "model_state_dict": model.state_dict(),
                "frozen_backbone_trainable_state_dict": trainable_module_state_dict(frozen_backbone),
                "optimizer_state_dict": optimizer.state_dict(),
                "tokenizer": tokenizer.state_dict(),
                "factor_columns": factor_names,
                "dataset": args.dataset,
                "data_split": split_record,
                "supervision": {
                    "factor_labels": bool(factor_names),
                    "factor_localizer": False,
                    "factor_graph": False,
                    "note": "Factor class labels do not supervise factor-to-position localization.",
                },
                "metrics": val_metrics,
                "args": vars(args),
            }, output_dir / "best_model.pth")
    torch.save({
        "epoch": args.epochs,
        "config": asdict(config),
        "model_state_dict": model.state_dict(),
        "frozen_backbone_trainable_state_dict": trainable_module_state_dict(frozen_backbone),
        "optimizer_state_dict": optimizer.state_dict(),
        "tokenizer": tokenizer.state_dict(),
        "factor_columns": factor_names,
        "dataset": args.dataset,
        "data_split": split_record,
        "supervision": {
            "factor_labels": bool(factor_names),
            "factor_localizer": False,
            "factor_graph": False,
            "note": "Factor class labels do not supervise factor-to-position localization.",
        },
        "metrics": history[-1]["val"] if history else {},
        "args": vars(args),
    }, output_dir / "final_model.pth")

    # The test partition is touched exactly once, after validation has selected
    # the checkpoint. Its result is reported separately and cannot select a model.
    best_checkpoint = load_torch_payload(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    adapter_state = best_checkpoint.get("frozen_backbone_trainable_state_dict")
    if frozen_backbone is not None and adapter_state:
        expected_trainable = {
            name
            for name, parameter in frozen_backbone.named_parameters()
            if parameter.requires_grad
        }
        missing_trainable = expected_trainable - set(adapter_state)
        if missing_trainable:
            raise RuntimeError(
                "Checkpoint is missing trained backbone adapter parameters: "
                f"{sorted(missing_trainable)[:8]}"
            )
        _, unexpected = frozen_backbone.load_state_dict(adapter_state, strict=False)
        if unexpected:
            raise RuntimeError(
                "Checkpoint contains unknown backbone adapter parameters: "
                f"{unexpected[:8]}"
            )
    with torch.no_grad():
        test_metrics = run_epoch(
            model,
            criterion,
            test_loader,
            optimizer,
            args,
            device,
            train=False,
            frozen_backbone=frozen_backbone,
            phase="test",
        )
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(
            {
                "selected_epoch": best_checkpoint["epoch"],
                "selection_metric": "validation total loss",
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
        )
    print(
        f"Wrote checkpoints to {output_dir}; selected epoch {best_checkpoint['epoch']} "
        "on validation and evaluated the held-out test partition once."
    )


if __name__ == "__main__":
    main()
