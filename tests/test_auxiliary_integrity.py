from __future__ import annotations

import random
import sys
import types
from pathlib import Path

import pytest
import torch

try:
    import torchvision  # noqa: F401
except (ImportError, RuntimeError):
    for module_name in [name for name in sys.modules if name.startswith("torchvision")]:
        del sys.modules[module_name]
    torchvision_stub = types.ModuleType("torchvision")
    torchvision_stub.transforms = types.ModuleType("torchvision.transforms")
    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.transforms"] = torchvision_stub.transforms

from benchmark_datasets import CausalSample
from extract_frozen_features import (
    build_feature_requests,
    build_tokenizer,
    load_media_frame,
)
from frozen_backbones import CachedFrozenBackbone, ClosedAPIFrozenBackbone
from metacausal_field import SimpleTokenizer
from run_baselines import partition_baseline_samples, predict_random


def _sample(sample_id: str, split: str, **kwargs) -> CausalSample:
    return CausalSample(sample_id=sample_id, split=split, **kwargs)


def test_feature_tokenizer_is_fit_on_training_text_only():
    samples = [
        _sample("train", "train", question="trainingword"),
        _sample("val", "val", question="validationword"),
        _sample("test", "test", question="heldoutword"),
    ]

    tokenizer = build_tokenizer(samples, vocab_size=32, max_length=8)

    assert "trainingword" in tokenizer.token_to_id
    assert "validationword" not in tokenizer.token_to_id
    assert "heldoutword" not in tokenizer.token_to_id


def test_feature_requests_include_counterfactual_media_and_reject_collisions():
    sample = _sample(
        "question-1",
        "train",
        image_path="factual.png",
        counterfactual_image_path="counterfactual.png",
    )
    requests = build_feature_requests([sample])
    assert [request["key"] for request in requests] == [
        "question-1",
        "question-1::counterfactual",
    ]

    with pytest.raises(ValueError, match="Duplicate feature-cache key"):
        build_feature_requests([sample, _sample("question-1", "test")])


def test_feature_media_loading_fails_closed(tmp_path: Path):
    missing = str(tmp_path / "missing.png")

    with pytest.raises(FileNotFoundError, match="Required media"):
        load_media_frame(missing)

    smoke = load_media_frame(missing, allow_missing_media=True)
    assert smoke.size == (224, 224)


def test_cached_backbone_rejects_smoke_or_inconsistent_cache(tmp_path: Path):
    smoke_path = tmp_path / "smoke.pt"
    torch.save(
        {
            "feature_dim": 4,
            "smoke_cache": True,
            "items": {"one": {"visual_features": torch.zeros(2, 4)}},
        },
        smoke_path,
    )
    with pytest.raises(ValueError, match="labelled as a smoke cache"):
        CachedFrozenBackbone(str(smoke_path))
    assert CachedFrozenBackbone(str(smoke_path), allow_smoke_cache=True).feature_dim == 4

    inconsistent_path = tmp_path / "inconsistent.pt"
    torch.save(
        {
            "feature_dim": 4,
            "items": {
                "one": {
                    "visual_features": torch.zeros(2, 4),
                    "language_tokens": torch.zeros(2, 4),
                },
                "two": {"visual_features": torch.zeros(2, 4)},
            },
        },
        inconsistent_path,
    )
    with pytest.raises(ValueError, match="mixes entries"):
        CachedFrozenBackbone(str(inconsistent_path))


def test_closed_api_requires_real_visual_features_unless_smoke_is_explicit(tmp_path: Path):
    tokenizer = SimpleTokenizer.build(["response"], vocab_size=16, max_length=6)

    def text_only_api(**_):
        return {"text": "response"}

    strict = ClosedAPIFrozenBackbone(
        text_only_api,
        tokenizer,
        feature_dim=4,
        cache_path=str(tmp_path / "strict.pt"),
    )
    with pytest.raises(RuntimeError, match="did not return visual_features"):
        strict.encode_paths(["image.png"], ["question"], torch.device("cpu"))

    smoke = ClosedAPIFrozenBackbone(
        text_only_api,
        tokenizer,
        feature_dim=4,
        cache_path=str(tmp_path / "smoke-api.pt"),
        allow_response_text_smoke=True,
    )
    outputs = smoke.encode_paths(["image.png"], ["question"], torch.device("cpu"))
    assert outputs["response_text_surrogate_smoke"] is True


def test_baseline_partitions_exclude_validation_and_require_explicit_test():
    samples = [
        _sample("train", "train", image_path="train.png", answer="train answer"),
        _sample("val", "validation", image_path="val.png", answer="val answer"),
        _sample("test", "eval", image_path="test.png", answer="test answer"),
    ]

    train, test = partition_baseline_samples(samples, require_train=True)

    assert [sample.sample_id for sample in train] == ["train"]
    assert [sample.sample_id for sample in test] == ["test"]
    with pytest.raises(ValueError, match="explicit test"):
        partition_baseline_samples(samples[:2], require_train=True)


def test_random_baseline_is_seeded_and_samples_candidates():
    sample = _sample("test", "test", choices=["a", "b", "c"])
    first_rng = random.Random(9)
    second_rng = random.Random(9)

    first = [predict_random(sample, first_rng, []) for _ in range(12)]
    second = [predict_random(sample, second_rng, []) for _ in range(12)]

    assert first == second
    assert len(set(first)) > 1
