"""Extract frozen Qwen/API/cache-ready features for MetaCausalField training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from benchmark_datasets import load_benchmark
from frozen_backbones import (
    ClosedAPIFrozenBackbone,
    QwenVLFrozenBackbone,
    ResNetTextFrozenBackbone,
    download_qwen_weights,
)
from metacausal_field import SimpleTokenizer


def load_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), "white")


def load_media_frame(image_path: str = "", video_path: str = ""):
    if image_path:
        return load_image(image_path)
    if video_path:
        try:
            from torchvision.io import read_video
            frames, _, _ = read_video(video_path, pts_unit="sec")
            if len(frames) > 0:
                return Image.fromarray(frames[len(frames) // 2].numpy()).convert("RGB")
        except Exception:
            pass
    return Image.new("RGB", (224, 224), "white")


def main():
    parser = argparse.ArgumentParser(description="Extract frozen MLLM features")
    parser.add_argument("--dataset", default="MAG9", choices=["MAG", "MAG9", "Lung", "Lung4", "CLEVRER", "Causal3DIdent", "CITRIS", "Causal-VidQA"])
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--backbone", default="qwen", choices=["qwen", "api", "resnet"])
    parser.add_argument("--qwen_model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--download_qwen", action="store_true", help="Download/resolve Qwen weights before loading.")
    parser.add_argument("--qwen_cache_dir", default=None, help="Local directory for downloaded Qwen weights.")
    parser.add_argument("--qwen_revision", default=None, help="Optional HuggingFace model revision/commit.")
    parser.add_argument("--hf_token", default=None, help="Optional HuggingFace token; falls back to HF_TOKEN env.")
    parser.add_argument("--local_files_only", action="store_true", help="Load only local model files.")
    parser.add_argument("--device_map", default="auto", help="Transformers device_map for Qwen loading.")
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--qwen_video_input", default="native", choices=["native", "middle_frame"])
    parser.add_argument("--output", default="./frozen_feature_cache.pt")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--max_text_length", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples, _ = load_benchmark(args.dataset, args.manifest_path, args.data_root)
    texts = [(s.question or s.text or s.answer) for s in samples]
    tokenizer = SimpleTokenizer.build(texts, vocab_size=args.vocab_size, max_length=args.max_text_length)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    if args.backbone == "qwen":
        qwen_path = args.qwen_model
        if args.download_qwen:
            qwen_path = download_qwen_weights(
                model_id=args.qwen_model,
                local_dir=args.qwen_cache_dir,
                revision=args.qwen_revision,
                token=args.hf_token,
                local_files_only=args.local_files_only,
            )
            print(f"Resolved Qwen weights: {qwen_path}")
        backbone = QwenVLFrozenBackbone(
            qwen_path,
            feature_dim=args.feature_dim,
            revision=args.qwen_revision if not Path(qwen_path).exists() else None,
            local_files_only=args.local_files_only,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
        )
    elif args.backbone == "api":
        from gemini_utils import generate
        backbone = ClosedAPIFrozenBackbone(generate, tokenizer, args.feature_dim, cache_path=str(Path(args.output).with_suffix(".api.pt")))
    else:
        backbone = ResNetTextFrozenBackbone(args.feature_dim, tokenizer)
    backbone.to(device)
    backbone.eval()

    items = {}
    for sample in tqdm(samples, desc="Extract frozen features"):
        text = sample.question or sample.text or sample.answer
        key = sample.feature_key or sample.image_path or sample.video_path or sample.sample_id
        with torch.no_grad():
            if args.backbone == "qwen":
                if args.qwen_video_input == "native" and sample.video_path and not sample.image_path:
                    outputs = backbone.encode_video([sample.video_path], [text], device)
                else:
                    outputs = backbone.encode_pil(
                        [load_media_frame(sample.image_path or "", sample.video_path or "")],
                        [text],
                        device,
                    )
            elif args.backbone == "api":
                outputs = backbone.encode_paths([sample.image_path or sample.video_path or ""], [text], device)
            else:
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                image = transform(load_media_frame(sample.image_path or "", sample.video_path or "")).unsqueeze(0).to(device)
                outputs = backbone.encode(image, [text])
        items[str(key)] = {
            "visual_features": outputs["visual_features"][0].detach().cpu(),
        }
        if "language_tokens" in outputs:
            items[str(key)]["language_tokens"] = outputs["language_tokens"][0].detach().cpu()

    payload = {
        "feature_dim": args.feature_dim,
        "tokenizer": tokenizer.state_dict(),
        "items": items,
        "dataset": args.dataset,
        "backbone": args.backbone,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"Saved frozen feature cache with {len(items)} items: {output}")


if __name__ == "__main__":
    main()
