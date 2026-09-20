from __future__ import annotations

import ast
import inspect
import sys
import types
from pathlib import Path

import pytest
import torch
import pandas as pd

try:
    import torchvision  # noqa: F401
except (ImportError, RuntimeError):  # Unit tests here do not instantiate VisualEncoder.
    for module_name in [name for name in sys.modules if name.startswith("torchvision")]:
        del sys.modules[module_name]
    torchvision_stub = types.ModuleType("torchvision")
    torchvision_stub.transforms = types.ModuleType("torchvision.transforms")
    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.transforms"] = torchvision_stub.transforms

from benchmark_datasets import CausalSample
from evaluate_metacausal_field import restore_dataframe_partitions
from metacausal_field import SimpleTokenizer
from train_benchmark_metacausal import (
    benchmark_intervention_tensors,
    build_tokenizer,
    load_media_frame,
    main,
    make_benchmark_loader,
    partition_samples,
)
from train_metacausal_field import intervention_tensors, prepare_counterfactual_supervision


def _sample(sample_id: str, split: str, *, media: str | None = None, text: str = ""):
    return CausalSample(
        sample_id=sample_id,
        split=split,
        image_path=media,
        question=text,
        answer=text,
    )


def test_partition_samples_is_strict_and_accepts_split_aliases():
    samples = [
        _sample("train-id", "train", media="train.png"),
        _sample("val-id", "validation", media="val.png"),
        _sample("test-id", "eval", media="test.png"),
    ]

    train, val, test = partition_samples(samples)

    assert [sample.sample_id for sample in train] == ["train-id"]
    assert [sample.sample_id for sample in val] == ["val-id"]
    assert [sample.sample_id for sample in test] == ["test-id"]


def test_partition_samples_rejects_missing_or_leaking_partitions():
    with pytest.raises(ValueError, match="missing: test"):
        partition_samples([
            _sample("train-id", "train", media="train.png"),
            _sample("val-id", "val", media="val.png"),
        ])

    with pytest.raises(ValueError, match="Data leakage"):
        partition_samples([
            _sample("train-id", "train", media="shared-video.mp4"),
            _sample("val-id", "val", media="shared-video.mp4"),
            _sample("test-id", "test", media="test.mp4"),
        ])

    train = _sample("train-id", "train", media="train.mp4")
    train.counterfactual_video_path = "heldout.mp4"
    with pytest.raises(ValueError, match="Data leakage"):
        partition_samples([
            train,
            _sample("val-id", "val", media="val.mp4"),
            _sample("test-id", "test", media="heldout.mp4"),
        ])


def test_tokenizer_can_be_fit_on_train_only_without_heldout_vocabulary():
    train = [_sample("train", "train", text="trainingtoken")]
    heldout = _sample("test", "test", text="heldouttoken")

    tokenizer = build_tokenizer(train, vocab_size=32, max_length=8)

    assert "trainingtoken" in tokenizer.token_to_id
    assert "heldouttoken" not in tokenizer.token_to_id
    assert tokenizer.token_to_id[SimpleTokenizer.UNK] in tokenizer.encode(heldout.question)


def test_media_loading_fails_closed_unless_smoke_fallback_is_explicit(tmp_path: Path):
    missing = str(tmp_path / "missing.png")

    with pytest.raises(FileNotFoundError, match="Required media"):
        load_media_frame(missing)

    fallback = load_media_frame(missing, allow_missing_media=True)
    assert fallback.mode == "RGB"
    assert fallback.size == (224, 224)


def test_masked_intervention_defaults_are_neutral_and_explicit_metadata_wins():
    batch = {
        "sample_id": ["sample-a", "sample-b"],
        "object_id": ["", ""],
        "intervention": [
            {},
            {"position": [0.2, 0.4], "direction": [1.0, 0.0, 0.0, 0.0]},
        ],
        "bbox": torch.tensor([
            [-1.0, -1.0, -1.0, -1.0],
            [0.7, 0.7, 0.9, 0.9],
        ]),
        "objects": [[], []],
    }

    first_positions, first_directions = benchmark_intervention_tensors(
        batch, feature_dim=4, device=torch.device("cpu")
    )
    second_positions, second_directions = benchmark_intervention_tensors(
        batch, feature_dim=4, device=torch.device("cpu")
    )

    assert torch.equal(first_positions, second_positions)
    assert torch.equal(first_directions, second_directions)
    assert torch.allclose(first_positions[0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(first_directions[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(first_positions[1], torch.tensor([0.2, 0.4]))
    assert torch.allclose(first_directions[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="require normalized position"):
        intervention_tensors(
            [{}],
            ["sample-a"],
            feature_dim=4,
            device=torch.device("cpu"),
        )


def test_seeded_loader_repeats_shuffle_order():
    samples = [
        _sample(f"sample-{index}", "train", text=f"text {index}")
        for index in range(8)
    ]
    tokenizer = build_tokenizer(samples, vocab_size=64, max_length=8)
    loader_args = {
        "tokenizer": tokenizer,
        "transform": lambda _: torch.zeros(3, 2, 2),
        "factor_names": [],
        "num_video_frames": 1,
        "batch_size": 3,
        "shuffle": True,
        "num_workers": 0,
        "seed": 123,
        "allow_missing_media": True,
    }

    first = make_benchmark_loader(samples, **loader_args)
    second = make_benchmark_loader(samples, **loader_args)
    first_order = [sample_id for batch in first for sample_id in batch["sample_id"]]
    second_order = [sample_id for batch in second for sample_id in batch["sample_id"]]

    assert first_order == second_order


def test_epoch_selection_loop_does_not_touch_test_loader():
    tree = ast.parse(inspect.getsource(main))
    epoch_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "epoch"
    ]

    assert len(epoch_loops) == 1
    names_used_for_selection = {
        node.id for node in ast.walk(epoch_loops[0]) if isinstance(node, ast.Name)
    }
    assert "val_loader" in names_used_for_selection
    assert "test_loader" not in names_used_for_selection


def test_legacy_counterfactual_rows_are_not_treated_as_paired_supervision():
    legacy = pd.DataFrame([
        {
            "Image": "source.png",
            "Review": "generated text with an ambiguous role",
            "score": 3.0,
            "scenario": "remove object",
        }
    ])
    usable, audit = prepare_counterfactual_supervision(legacy)

    assert usable.empty
    assert audit["missing_explicit_intervention"] == 1
    assert audit["missing_explicit_target"] == 1

    explicit = legacy.assign(
        intervention=['{"position": [0.2, 0.4], "direction": [1, 0]}'],
        CounterfactualReview=["explicit target"],
    )
    usable, audit = prepare_counterfactual_supervision(explicit)
    assert len(usable) == 1
    assert audit["usable_rows"] == 1


def test_checkpoint_split_record_restores_exact_partitions_and_rejects_drift():
    frame = pd.DataFrame({"sample_id": ["c", "a", "b"], "value": [3, 1, 2]})
    record = {
        "id_column": "sample_id",
        "train_ids": ["a"],
        "val_ids": ["b"],
        "test_ids": ["c"],
    }
    train, val, test = restore_dataframe_partitions(frame, record)
    assert train["sample_id"].tolist() == ["a"]
    assert val["sample_id"].tolist() == ["b"]
    assert test["sample_id"].tolist() == ["c"]

    drifted = frame.assign(sample_id=["c", "a", "new"])
    with pytest.raises(ValueError, match="differs from the checkpoint"):
        restore_dataframe_partitions(drifted, record)
