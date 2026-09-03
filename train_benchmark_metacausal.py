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
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from benchmark_datasets import UnifiedCausalDataset, load_benchmark
from frozen_backbones import CachedFrozenBackbone, QwenVLFrozenBackbone, download_qwen_weights
from metacausal_field import CausalFieldConfig, MetaCausalLoss, MultimodalCausalField, SimpleTokenizer, VisualEncoder
from train_metacausal_field import collate, scenario_directions


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


def intervention_positions(batch, device):
    boxes = batch["bbox"].to(device)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    centers = torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], dim=-1)
    if valid.all():
        return centers
    fallback = []
    for i, (sample_id, obj_id) in enumerate(zip(batch["sample_id"], batch["object_id"])):
        target = str(obj_id or "")
        intervention = batch.get("intervention", [{}])[i] if "intervention" in batch else {}
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
                found = [(float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0]
                break
        if found is not None:
            fallback.append(found)
            continue
        seed = abs(hash(str(sample_id) + target))
        fallback.append([0.15 + 0.7 * ((seed % 997) / 996.0), 0.15 + 0.7 * (((seed // 997) % 991) / 990.0)])
    fallback = torch.tensor(fallback, dtype=torch.float32, device=device)
    return torch.where(valid.unsqueeze(-1), centers, fallback)


def load_media_frame(image_path: str = "", video_path: str = ""):
    if image_path:
        try:
            return Image.open(image_path).convert("RGB")
        except Exception:
            pass
    if video_path:
        try:
            from torchvision.io import read_video
            frames, _, _ = read_video(video_path, pts_unit="sec")
            if len(frames) > 0:
                return Image.fromarray(frames[len(frames) // 2].numpy()).convert("RGB")
        except Exception:
            pass
    return Image.new("RGB", (224, 224), "white")


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
                load_media_frame(image_path, video_path)
                for image_path, video_path in zip(image_paths, video_paths)
            ]
            outputs = frozen_backbone.encode_pil(images, texts, device)
        return outputs["visual_features"], outputs.get("language_tokens")
    if frozen_backbone is not None:
        outputs = frozen_backbone.encode_by_keys(batch["feature_key"], device)
        return outputs["visual_features"], outputs.get("language_tokens")
    return model.visual_encoder(batch["image"].to(device)), None


def run_epoch(model, criterion, loader, optimizer, args, device, train=True, frozen_backbone=None):
    model.train(train)
    if frozen_backbone is not None:
        if getattr(frozen_backbone, "trainable_lora", False):
            frozen_backbone.train(train)
        else:
            frozen_backbone.eval()
    totals = {}
    for batch in tqdm(loader, desc="train" if train else "val"):
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

            if not args.disable_consistency:
                noise = torch.randn_like(visual) * args.feature_noise_std
                aug_outputs = model(visual + noise, language_tokens=language_tokens, input_ids=input_ids)
                losses["consistency"] = criterion.causal_consistency_loss(outputs["field"], aug_outputs["field"])
                losses["total"] = losses["total"] + args.lambda_consistency * losses["consistency"]

            cf_mask = [bool(x) for x in batch["cf_answer"]]
            if any(cf_mask) and not args.disable_counterfactual:
                cf_texts = [cf if cf else ans for cf, ans in zip(batch["cf_answer"], batch["answer"])]
                cf_ids = torch.tensor([loader.dataset.tokenizer.encode(t) for t in cf_texts], dtype=torch.long, device=device)
                cf_outputs = model.counterfactual_forward(
                    visual,
                    intervention_type="modify",
                    intervention_params={
                        "position": intervention_positions(batch, device),
                        "direction": scenario_directions(batch["sample_id"], model.config.feature_dim, device),
                        "radius": args.intervention_radius,
                    },
                    language_tokens=language_tokens,
                    input_ids=input_ids,
                    decoder_input_ids=cf_ids[:, :-1],
                    num_rollout_steps=model.config.num_propagation_steps,
                )
                cf_losses = criterion(cf_outputs, {"cf_lm_targets": cf_ids[:, 1:]})
                if args.enable_cf_trajectory_loss and any(batch.get("counterfactual_image_path", [])):
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

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    samples, factor_names = load_benchmark(args.dataset, args.manifest_path, args.data_root)
    train_samples = [s for s in samples if s.split in {"train", "val", "validation"}] or samples
    val_samples = [s for s in samples if s.split in {"test", "eval"}] or samples[: max(1, len(samples) // 5)]
    tokenizer = build_tokenizer(samples, args.vocab_size, args.max_text_length)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_loader = DataLoader(
        UnifiedCausalDataset(train_samples, tokenizer, transform, factor_names, args.num_video_frames),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        UnifiedCausalDataset(val_samples, tokenizer, transform, factor_names, args.num_video_frames),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
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
        frozen_backbone = CachedFrozenBackbone(args.feature_cache, feature_dim=args.feature_dim).to(device)
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

    output_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, criterion, train_loader, optimizer, args, device, train=True, frozen_backbone=frozen_backbone)
        with torch.no_grad():
            val_metrics = run_epoch(model, criterion, val_loader, optimizer, args, device, train=False, frozen_backbone=frozen_backbone)
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
                "optimizer_state_dict": optimizer.state_dict(),
                "tokenizer": tokenizer.state_dict(),
                "factor_columns": factor_names,
                "dataset": args.dataset,
                "metrics": val_metrics,
                "args": vars(args),
            }, output_dir / "best_model.pth")
    torch.save({
        "epoch": args.epochs,
        "config": asdict(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tokenizer": tokenizer.state_dict(),
        "factor_columns": factor_names,
        "dataset": args.dataset,
        "metrics": history[-1]["val"] if history else {},
        "args": vars(args),
    }, output_dir / "final_model.pth")
    print(f"Wrote checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
