"""Frozen multimodal backbone adapters for MetaCausalField.

The adapters share one interface: given images and text, return visual feature
tokens and optional text feature tokens without updating the backbone. Heavy
external resources such as Qwen3-VL-8B weights or closed API responses are
loaded only when explicitly requested.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from metacausal_field import SimpleTokenizer, TextEncoder, VisualEncoder


def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def download_qwen_weights(
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    local_dir: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
) -> str:
    """Download or resolve Qwen/Qwen3-VL weights via HuggingFace Hub.

    Returns a local directory that can be passed to `from_pretrained`.
    If `model_id` is already a local path, it is returned unchanged.
    """
    model_path = Path(model_id).expanduser()
    if model_path.exists():
        return str(model_path)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise ImportError(
            "Downloading Qwen weights requires `huggingface_hub`. "
            "Install it with `pip install huggingface_hub`, or pass a local --qwen_model path."
        ) from exc

    resolved_dir = local_dir
    if resolved_dir is None:
        safe_name = model_id.replace("/", "__")
        resolved_dir = str(Path("models") / safe_name)

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=resolved_dir,
        local_dir_use_symlinks=False,
        token=token,
        local_files_only=local_files_only,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "*.safetensors",
            "*.bin",
            "*.py",
            "tokenizer.*",
            "merges.txt",
            "vocab.*",
            "preprocessor_config.json",
            "processor_config.json",
            "generation_config.json",
        ],
    )


class FrozenBackbone(nn.Module):
    """Base class for frozen multimodal feature extractors."""

    feature_dim: int

    def encode(self, images: Optional[torch.Tensor], texts: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, images: Optional[torch.Tensor], texts: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        return self.encode(images, texts)


class ResNetTextFrozenBackbone(FrozenBackbone):
    """Local frozen backbone used for smoke tests and small datasets."""

    def __init__(self, feature_dim: int, tokenizer: SimpleTokenizer, train_backbone: bool = False):
        super().__init__()
        self.feature_dim = feature_dim
        self.visual = VisualEncoder(feature_dim=feature_dim, pretrained=False, train_backbone=train_backbone)
        self.text = TextEncoder(
            vocab_size=tokenizer.vocab_size,
            feature_dim=feature_dim,
            max_length=tokenizer.max_length,
            dropout=0.0,
        )
        self.tokenizer = tokenizer
        if not train_backbone:
            for param in self.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def encode(self, images: Optional[torch.Tensor], texts: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        outputs = {}
        if images is not None:
            outputs["visual_features"] = self.visual(images)
        if texts is not None:
            ids = torch.tensor(
                [self.tokenizer.encode(text) for text in texts],
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            outputs["language_tokens"] = self.text(ids)
        return outputs


class CachedFrozenBackbone(FrozenBackbone):
    """Loads precomputed frozen visual/text tokens from a .pt or JSONL cache.

    Expected .pt format:
        {
          "feature_dim": 512,
          "items": {
             "<key>": {"visual_features": Tensor[N,C], "language_tokens": Tensor[L,C]}
          }
        }

    The key is either an explicit sample id or sha256(image_path, text).
    """

    def __init__(self, cache_path: str, feature_dim: Optional[int] = None):
        super().__init__()
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Feature cache not found: {self.cache_path}")
        if self.cache_path.suffix == ".pt":
            payload = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        else:
            payload = {"items": {}}
            with open(self.cache_path) as f:
                for line in f:
                    item = json.loads(line)
                    key = item.pop("key")
                    payload["items"][key] = {
                        name: torch.tensor(value, dtype=torch.float32)
                        for name, value in item.items()
                        if name in {"visual_features", "language_tokens"}
                    }
            payload["feature_dim"] = feature_dim
        self.items = payload["items"]
        self.feature_dim = int(payload.get("feature_dim") or feature_dim or self._infer_dim())

    def _infer_dim(self) -> int:
        first = next(iter(self.items.values()))
        tensor = first.get("visual_features")
        if tensor is None:
            tensor = first.get("language_tokens")
        return int(tensor.shape[-1])

    @torch.no_grad()
    def encode_by_keys(self, keys: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        visual = []
        text = []
        for key in keys:
            if key not in self.items:
                raise KeyError(f"Missing frozen feature cache key: {key}")
            item = self.items[key]
            if "visual_features" in item:
                visual.append(item["visual_features"])
            if "language_tokens" in item:
                text.append(item["language_tokens"])
        outputs = {}
        if visual:
            outputs["visual_features"] = _pad_token_sequences(visual).to(device)
        if text:
            outputs["language_tokens"] = _pad_token_sequences(text).to(device)
        return outputs


def _pad_token_sequences(sequences: List[torch.Tensor]) -> torch.Tensor:
    if not sequences:
        raise ValueError("Cannot pad an empty sequence list")
    max_len = max(seq.shape[0] for seq in sequences)
    dim = sequences[0].shape[-1]
    out = sequences[0].new_zeros(len(sequences), max_len, dim)
    for i, seq in enumerate(sequences):
        if seq.numel() > 0:
            out[i, :seq.shape[0]] = seq
    return out


def _special_token_ids(processor, model) -> Tuple[set, set]:
    tokenizer = getattr(processor, "tokenizer", processor)
    image_ids = set()
    text_exclude = set()
    for attr in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        value = getattr(getattr(model, "config", None), attr, None)
        if value is not None:
            image_ids.add(int(value))
    for token in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>", "<image>", "<video>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
                image_ids.add(int(token_id))
        except Exception:
            pass
    for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            text_exclude.add(int(value))
    text_exclude |= image_ids
    return image_ids, text_exclude


class QwenVLFrozenBackbone(FrozenBackbone):
    """Frozen Qwen-VL/Qwen3-VL adapter.

    This adapter intentionally imports transformers lazily. It supports current
    HuggingFace auto classes where Qwen-VL checkpoints expose hidden states. If
    a local install cannot load the requested checkpoint, the raised error tells
    the user to install/update transformers or provide a cached feature file.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        feature_dim: Optional[int] = None,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        device_map: Optional[str] = "auto",
        torch_dtype: Optional[str] = "auto",
        trainable_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            raise ImportError(
                "Qwen frozen backbone requires transformers with AutoModelForImageTextToText. "
                "Install/update transformers or use --backbone cached."
            ) from exc

        if torch_dtype == "auto":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif torch_dtype in {"float16", "fp16"}:
            dtype = torch.float16
        elif torch_dtype in {"bfloat16", "bf16"}:
            dtype = torch.bfloat16
        elif torch_dtype in {"float32", "fp32"}:
            dtype = torch.float32
        else:
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

        self.uses_device_map = bool(device_map and torch.cuda.is_available())
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            revision=revision,
            torch_dtype=dtype,
            device_map=device_map if torch.cuda.is_available() else None,
            local_files_only=local_files_only,
            trust_remote_code=True,
            output_hidden_states=True,
        )
        self.trainable_lora = bool(trainable_lora)
        if self.trainable_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except Exception as exc:
                raise ImportError(
                    "Trainable Qwen LoRA requires `peft`. Install it with `pip install peft`, "
                    "or omit --train_qwen_lora."
                ) from exc
            target_modules = lora_target_modules or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.train()
        else:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)
        if hidden_size is None and hasattr(self.model.config, "text_config"):
            hidden_size = getattr(self.model.config.text_config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = feature_dim
        self.hidden_size = int(hidden_size)
        self.feature_dim = int(feature_dim or hidden_size)
        self.proj = None
        if self.feature_dim != self.hidden_size:
            self.proj = nn.Linear(self.hidden_size, self.feature_dim, bias=False)
            generator = torch.Generator().manual_seed(17)
            nn.init.orthogonal_(self.proj.weight, gain=1.0)
            for param in self.proj.parameters():
                param.requires_grad = self.trainable_lora
        self.image_token_ids, self.text_exclude_ids = _special_token_ids(self.processor, self.model)

    def to(self, *args, **kwargs):
        if self.uses_device_map:
            if self.proj is not None:
                self.proj.to(*args, **kwargs)
            return self
        return super().to(*args, **kwargs)

    def _project(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            return tokens
        return self.proj(tokens)

    def _encode_inputs(self, inputs, device: torch.device) -> Dict[str, torch.Tensor]:
        inputs = inputs.to(device)
        context = nullcontext() if self.trainable_lora else torch.no_grad()
        with context:
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = self._project(outputs.hidden_states[-1].float())
        input_ids = inputs.get("input_ids")
        visual_sequences = []
        text_sequences = []
        for b in range(hidden.shape[0]):
            ids = input_ids[b] if input_ids is not None else None
            if ids is None:
                visual_sequences.append(hidden[b])
                text_sequences.append(hidden[b])
                continue
            image_mask = torch.zeros_like(ids, dtype=torch.bool)
            for token_id in self.image_token_ids:
                image_mask |= ids.eq(token_id)
            text_mask = torch.ones_like(ids, dtype=torch.bool)
            for token_id in self.text_exclude_ids:
                text_mask &= ~ids.eq(token_id)
            if image_mask.any():
                visual_sequences.append(hidden[b, image_mask])
            else:
                # Some Qwen variants replace image placeholders before the LM.
                # Fall back to all non-special tokens instead of fabricating blank features.
                visual_sequences.append(hidden[b, text_mask])
            text_sequences.append(hidden[b, text_mask])
        return {
            "visual_features": _pad_token_sequences(visual_sequences),
            "language_tokens": _pad_token_sequences(text_sequences),
        }

    def encode_pil(self, images: List, texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        return self._encode_inputs(inputs, device)

    def encode_video(
        self,
        videos: List[Union[str, List]],
        texts: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        inputs = self.processor(text=texts, videos=videos, return_tensors="pt", padding=True)
        return self._encode_inputs(inputs, device)

    def encode_media(
        self,
        images: Optional[List] = None,
        videos: Optional[List[Union[str, List]]] = None,
        texts: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        texts = texts or [""] * max(len(images or []), len(videos or []), 1)
        device = device or next(self.parameters()).device
        kwargs = {"text": texts, "return_tensors": "pt", "padding": True}
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
        inputs = self.processor(**kwargs)
        return self._encode_inputs(inputs, device)


class ClosedAPIFrozenBackbone(FrozenBackbone):
    """Closed MLLM API adapter with persistent feature cache.

    Most closed MLLM APIs do not expose internal embeddings. This adapter uses
    an explicit response/embedding cache protocol: if the API response contains
    `visual_features` or `language_tokens`, those frozen embeddings are used.
    Otherwise it stores the raw response and derives deterministic text tokens
    from the frozen response text, making the limitation visible in the cache.
    """

    def __init__(
        self,
        api_generate: Callable,
        tokenizer: SimpleTokenizer,
        feature_dim: int,
        cache_path: str = "./api_feature_cache.pt",
    ):
        super().__init__()
        self.api_generate = api_generate
        self.tokenizer = tokenizer
        self.feature_dim = feature_dim
        self.text_encoder = TextEncoder(tokenizer.vocab_size, feature_dim, tokenizer.max_length, dropout=0.0)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.cache_path = Path(cache_path)
        self.cache = torch.load(self.cache_path, map_location="cpu", weights_only=False) if self.cache_path.exists() else {}

    def _response_for(self, image_path: str, text: str) -> Dict:
        key = _hash_key(image_path, text)
        if key not in self.cache:
            response = self.api_generate(prompt=text, img_path=[image_path])
            if not isinstance(response, dict):
                response = {"text": str(response)}
            self.cache[key] = response
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.cache, self.cache_path)
        return self.cache[key]

    @torch.no_grad()
    def encode_paths(self, image_paths: List[str], texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        responses = [self._response_for(path, text) for path, text in zip(image_paths, texts)]
        if all("visual_features" in r for r in responses):
            visual = torch.stack([torch.tensor(r["visual_features"], dtype=torch.float32) for r in responses]).to(device)
        else:
            ids = torch.tensor(
                [self.tokenizer.encode(r.get("text", "")) for r in responses],
                dtype=torch.long,
                device=device,
            )
            visual = self.text_encoder(ids)
        if all("language_tokens" in r for r in responses):
            language = torch.stack([torch.tensor(r["language_tokens"], dtype=torch.float32) for r in responses]).to(device)
        else:
            ids = torch.tensor(
                [self.tokenizer.encode(r.get("text", "")) for r in responses],
                dtype=torch.long,
                device=device,
            )
            language = self.text_encoder(ids)
        return {
            "visual_features": visual,
            "language_tokens": language,
            "api_responses": [r.get("text", "") for r in responses],
        }


def build_frozen_backbone(
    backbone: str,
    feature_dim: int,
    tokenizer: SimpleTokenizer,
    qwen_model: str = "Qwen/Qwen3-VL-8B-Instruct",
    cache_path: Optional[str] = None,
    api_generate: Optional[Callable] = None,
) -> FrozenBackbone:
    if backbone == "resnet":
        return ResNetTextFrozenBackbone(feature_dim, tokenizer)
    if backbone == "qwen":
        return QwenVLFrozenBackbone(qwen_model, feature_dim)
    if backbone == "cached":
        if cache_path is None:
            raise ValueError("--feature_cache is required when --backbone cached")
        return CachedFrozenBackbone(cache_path, feature_dim)
    if backbone == "api":
        if api_generate is None:
            raise ValueError("api_generate callable is required when --backbone api")
        return ClosedAPIFrozenBackbone(api_generate, tokenizer, feature_dim, cache_path or "./api_feature_cache.pt")
    raise ValueError(f"Unknown frozen backbone: {backbone}")
