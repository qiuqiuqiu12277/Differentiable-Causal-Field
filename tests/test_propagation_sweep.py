"""Regression tests for propagation-step evaluation overrides."""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import evaluate_metacausal_field
import propagation_sweep
from metacausal_field import CausalFieldConfig, MultimodalCausalField, SimpleTokenizer


def _checkpoint(num_propagation_steps=3):
    config = CausalFieldConfig(
        spatial_size=(2, 2),
        feature_dim=4,
        num_heads=1,
        num_propagation_steps=num_propagation_steps,
        dropout=0.0,
        vocab_size=4,
        max_text_length=8,
    )
    tokenizer = SimpleTokenizer(max_length=config.max_text_length)
    model = MultimodalCausalField(
        config,
        visual_encoder=None,
        vocab_size=tokenizer.vocab_size,
        enable_language=True,
    )
    return {
        "config": asdict(config),
        "tokenizer": tokenizer.state_dict(),
        "factor_columns": [],
        "model_state_dict": model.state_dict(),
    }


@pytest.mark.parametrize("runtime_steps", [0, 1, 4])
def test_load_model_overrides_runtime_steps_without_mutating_checkpoint(monkeypatch, runtime_steps):
    checkpoint = _checkpoint(num_propagation_steps=3)
    monkeypatch.setattr(
        evaluate_metacausal_field.torch,
        "load",
        lambda *args, **kwargs: checkpoint,
    )

    model, _, _, _, returned_checkpoint = evaluate_metacausal_field.load_model(
        "unused.pt",
        torch.device("cpu"),
        visual_encoder=False,
        num_propagation_steps=runtime_steps,
    )

    assert model.config.num_propagation_steps == runtime_steps
    assert model.propagation.num_steps == runtime_steps
    outputs = model(torch.randn(1, 4, model.config.feature_dim))
    assert len(outputs["field_trajectory"]) == runtime_steps + 1

    assert returned_checkpoint is checkpoint
    assert returned_checkpoint["config"]["num_propagation_steps"] == 3


def test_latency_loads_model_with_the_requested_runtime_steps(monkeypatch):
    requested_steps = []

    class IdentityModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(feature_dim=2)

        def forward(self, inputs):
            return inputs

    def fake_load_model(checkpoint, device, **kwargs):
        requested_steps.append(kwargs["num_propagation_steps"])
        return IdentityModel(), None, None, None, None

    monkeypatch.setattr(propagation_sweep, "load_model", fake_load_model)

    latency = propagation_sweep.measure_latency("unused.pt", 5, "cpu", repeats=1)

    assert requested_steps == [5]
    assert latency >= 0.0


def test_sweep_passes_every_k_to_evaluation(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, check):
        assert check is True
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({"val_score_mae": 0.25}))

    monkeypatch.setattr(propagation_sweep.subprocess, "run", fake_run)
    monkeypatch.setattr(
        propagation_sweep,
        "measure_latency",
        lambda checkpoint, k, device: float(k),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "propagation_sweep.py",
            "--checkpoint",
            "model.pt",
            "--output_dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--k_values",
            "0,2,5",
        ],
    )

    propagation_sweep.main()

    assert len(commands) == 3
    assert [
        command[command.index("--num_propagation_steps") + 1]
        for command in commands
    ] == ["0", "2", "5"]
    assert all("--disable_propagation" not in command for command in commands)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert list(summary) == ["K=0", "K=2", "K=5"]
    assert [summary[key]["K"] for key in summary] == [0, 2, 5]
