"""Extract auditable frozen multimodal features for InfluenceField training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from tqdm import tqdm

from benchmark_datasets import (
    CausalSample,
    counterfactual_feature_key,
    load_benchmark,
    sample_feature_key,
)
from frozen_backbones import (
    ClosedAPIFrozenBackbone,
    QwenVLFrozenBackbone,
    ResNetTextFrozenBackbone,
    download_qwen_weights,
)
from metacausal_field import SimpleTokenizer


TRAIN_SPLITS = {"train"}


def load_media_frame(
    image_path: str = "",
    video_path: str = "",
    *,
    allow_missing_media: bool = False,
) -> Image.Image:
    """Load an image or a video's middle frame without silent substitution."""
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
        "Pass --allow_missing_media only for an explicitly labelled smoke cache."
    )


def build_tokenizer(
    samples: Iterable[CausalSample],
    *,
    vocab_size: int,
    max_length: int,
) -> SimpleTokenizer:
    """Fit learned text preprocessing on the training partition only."""
    training_texts = [
        sample.question or sample.text or sample.answer
        for sample in samples
        if str(sample.split).strip().lower() in TRAIN_SPLITS
    ]
    if not training_texts:
        raise ValueError("Feature extraction requires a non-empty explicit training partition.")
    return SimpleTokenizer.build(
        training_texts,
        vocab_size=vocab_size,
        max_length=max_length,
    )


def build_feature_requests(samples: Iterable[CausalSample]) -> list[dict]:
    """Create unique factual/counterfactual cache entries before expensive work."""
    requests = []
    owners: dict[str, str] = {}
    for sample in samples:
        text = sample.question or sample.text or sample.answer
        factual_key = sample_feature_key(sample)
        candidates = [
            {
                "key": factual_key,
                "sample_id": str(sample.sample_id),
                "kind": "factual",
                "image_path": sample.image_path or "",
                "video_path": sample.video_path or "",
                "text": text,
            }
        ]
        if sample.counterfactual_image_path or sample.counterfactual_video_path:
            candidates.append(
                {
                    "key": counterfactual_feature_key(sample),
                    "sample_id": str(sample.sample_id),
                    "kind": "counterfactual",
                    "image_path": sample.counterfactual_image_path or "",
                    "video_path": sample.counterfactual_video_path or "",
                    "text": text,
                }
            )
        for request in candidates:
            key = str(request["key"])
            if not key:
                raise ValueError(f"Sample {sample.sample_id!r} has an empty feature-cache key.")
            if key in owners:
                raise ValueError(
                    f"Duplicate feature-cache key {key!r} for {owners[key]} and "
                    f"{request['sample_id']}:{request['kind']}. Use unique sample/feature IDs."
                )
            owners[key] = f"{request['sample_id']}:{request['kind']}"
            requests.append(request)
    if not requests:
        raise ValueError("No samples were available for feature extraction.")
    return requests


def _local_path_exists(path: str) -> bool:
    return bool(path) and (
        path.startswith(("http://", "https://")) or Path(path).is_file()
    )


def _encode_request(backbone, request, args, device, resnet_transform=None):
    image_path = request["image_path"]
    video_path = request["video_path"]
    text = request["text"]
    if args.backbone == "qwen":
        use_native_video = (
            args.qwen_video_input == "native"
            and bool(video_path)
            and not image_path
            and _local_path_exists(video_path)
        )
        if use_native_video:
            return backbone.encode_video([video_path], [text], device)
        image = load_media_frame(
            image_path,
            video_path,
            allow_missing_media=args.allow_missing_media,
        )
        return backbone.encode_pil([image], [text], device)
    if args.backbone == "api":
        path = image_path or video_path
        if not _local_path_exists(path) and not args.allow_missing_media:
            raise FileNotFoundError(
                f"Required API media path is missing or invalid: {path or '<empty path>'!r}."
            )
        return backbone.encode_paths([path], [text], device)

    if resnet_transform is None:
        raise RuntimeError("ResNet transform was not initialized")
    image = resnet_transform(
        load_media_frame(
            image_path,
            video_path,
            allow_missing_media=args.allow_missing_media,
        )
    ).unsqueeze(0).to(device)
    return backbone.encode(image, [text])


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow_missing_media",
        action="store_true",
        help="Use white media only for an explicitly labelled smoke cache.",
    )
    parser.add_argument(
        "--allow_api_text_smoke",
        action="store_true",
        help="Allow response-text surrogates when a closed API exposes no visual features.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    samples, _ = load_benchmark(
        args.dataset,
        args.manifest_path,
        args.data_root,
        seed=args.seed,
    )
    requests = build_feature_requests(samples)
    tokenizer = build_tokenizer(
        samples,
        vocab_size=args.vocab_size,
        max_length=args.max_text_length,
    )
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

        backbone = ClosedAPIFrozenBackbone(
            generate,
            tokenizer,
            args.feature_dim,
            cache_path=str(Path(args.output).with_suffix(".api.pt")),
            allow_response_text_smoke=args.allow_api_text_smoke,
        )
    else:
        backbone = ResNetTextFrozenBackbone(args.feature_dim, tokenizer)
    backbone.to(device)
    backbone.eval()

    resnet_transform = None
    if args.backbone == "resnet":
        from torchvision import transforms

        resnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    items = {}
    contains_api_text_surrogates = False
    for request in tqdm(requests, desc="Extract frozen features"):
        with torch.no_grad():
            outputs = _encode_request(
                backbone,
                request,
                args,
                device,
                resnet_transform=resnet_transform,
            )
        if "visual_features" not in outputs:
            raise RuntimeError(f"Backbone returned no visual_features for key {request['key']!r}")
        item = {
            "visual_features": outputs["visual_features"][0].detach().cpu(),
            "sample_id": request["sample_id"],
            "kind": request["kind"],
        }
        if "language_tokens" in outputs:
            item["language_tokens"] = outputs["language_tokens"][0].detach().cpu()
        surrogate = bool(outputs.get("response_text_surrogate_smoke", False))
        if surrogate:
            item["response_text_surrogate_smoke"] = True
            contains_api_text_surrogates = True
        items[request["key"]] = item

    payload = {
        "schema_version": 2,
        "feature_dim": args.feature_dim,
        "tokenizer": tokenizer.state_dict(),
        "tokenizer_fit_split": "train",
        "items": items,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "seed": args.seed,
        "smoke_cache": bool(args.allow_missing_media or contains_api_text_surrogates),
        "allow_missing_media": bool(args.allow_missing_media),
        "contains_api_text_surrogates": contains_api_text_surrogates,
        "qwen_model": args.qwen_model if args.backbone == "qwen" else None,
        "qwen_revision": args.qwen_revision if args.backbone == "qwen" else None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"Saved frozen feature cache with {len(items)} unique items: {output}")


if __name__ == "__main__":
    main()
