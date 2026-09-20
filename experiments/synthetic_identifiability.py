"""Constructive diagnostic for intervention coverage and graph ambiguity.

This module is intentionally small and CPU-only.  It demonstrates the linear
similarity ambiguity discussed in the InfluenceField paper:

    y_t = H z_t,       z_{t+1} = A_* z_t,
    y_{t+1} = (H A_* H^{-1}) y_t.

Observational prediction is exact for every invertible mixing ``H``, although
the support of ``H A_* H^{-1}`` can differ from the support of ``A_*``.  Known
site interventions restrict covered coordinates to diagonal rescalings.  The
diagnostic constructs that solution family at 0/25/50/100 percent coverage and
reports graph recovery inside and outside the covered set.

It is a theorem-aligned smoke experiment, not a reproduction of any paper
table.  It does not train the multimodal model or use CausalVQA.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class DiagnosticConfig:
    num_nodes: int = 64
    num_blocks: int = 4
    seeds: int = 5
    samples: int = 256
    within_block_probability: float = 0.12
    between_block_probability: float = 0.06
    spectral_radius: float = 0.90
    coverages: tuple[float, ...] = (0.0, 0.25, 0.50, 1.0)
    support_tolerance: float = 1e-8

    def validate(self) -> None:
        if self.num_nodes < 4:
            raise ValueError("num_nodes must be at least 4")
        if self.num_blocks < 1 or self.num_nodes % self.num_blocks:
            raise ValueError("num_nodes must be divisible by num_blocks")
        if self.seeds < 1 or self.samples < 1:
            raise ValueError("seeds and samples must be positive")
        if any(not 0.0 <= value <= 1.0 for value in self.coverages):
            raise ValueError("coverages must lie in [0, 1]")


def _sample_stable_transition(config: DiagnosticConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample the ordered-block transition used by the diagnostic.

    Matrix entry ``A[target, source]`` represents the edge ``source -> target``.
    Between-block edges point from earlier blocks to later blocks.  Self terms
    stabilize the transition but are excluded from graph metrics.
    """

    k = config.num_nodes
    block_size = k // config.num_blocks
    for _ in range(200):
        transition = np.zeros((k, k), dtype=np.float64)
        for source in range(k):
            source_block = source // block_size
            for target in range(k):
                if source == target:
                    continue
                target_block = target // block_size
                if source_block == target_block:
                    probability = config.within_block_probability
                elif source_block < target_block:
                    probability = config.between_block_probability
                else:
                    probability = 0.0
                if rng.random() < probability:
                    magnitude = rng.uniform(0.25, 0.80)
                    transition[target, source] = magnitude * rng.choice((-1.0, 1.0))

        np.fill_diagonal(transition, 2.0)
        radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
        transition *= config.spectral_radius / max(radius, 1e-12)
        if np.linalg.matrix_rank(transition) == k:
            return transition
    raise RuntimeError("could not sample an invertible stable transition")


def _covered_indices(config: DiagnosticConfig, coverage: float) -> np.ndarray:
    """Use complete trailing blocks, matching the paper's closed-set diagnostic."""

    block_size = config.num_nodes // config.num_blocks
    blocks = int(round(coverage * config.num_blocks))
    blocks = min(config.num_blocks, max(0, blocks))
    first = config.num_nodes - blocks * block_size
    return np.arange(first, config.num_nodes, dtype=np.int64)


def _mixing_for_coverage(
    config: DiagnosticConfig,
    covered: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct an observationally equivalent mixing allowed by coverage.

    Covered coordinates receive only a nonzero diagonal scaling.  Uncovered
    coordinates may mix through an arbitrary well-conditioned invertible block.
    """

    k = config.num_nodes
    mixing = np.eye(k, dtype=np.float64)
    is_covered = np.zeros(k, dtype=bool)
    is_covered[covered] = True
    uncovered = np.flatnonzero(~is_covered)

    if uncovered.size:
        dense = rng.normal(size=(uncovered.size, uncovered.size))
        q, _ = np.linalg.qr(dense)
        scales = rng.uniform(0.70, 1.30, size=uncovered.size)
        mixing[np.ix_(uncovered, uncovered)] = q @ np.diag(scales)

    if covered.size:
        mixing[covered, covered] = rng.uniform(0.70, 1.30, size=covered.size)
    return mixing


def _edge_mask(matrix: np.ndarray, tolerance: float) -> np.ndarray:
    mask = np.abs(matrix) > tolerance
    np.fill_diagonal(mask, False)
    return mask


def _edge_f1(predicted: np.ndarray, truth: np.ndarray, nodes: Optional[np.ndarray] = None) -> float:
    if nodes is not None:
        if nodes.size < 2:
            return 1.0
        predicted = predicted[np.ix_(nodes, nodes)]
        truth = truth[np.ix_(nodes, nodes)]
    tp = int(np.logical_and(predicted, truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    if tp == fp == fn == 0:
        return 1.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def run_diagnostic(config: DiagnosticConfig) -> tuple[list[dict], dict]:
    config.validate()
    rows: list[dict] = []

    for seed in range(config.seeds):
        rng = np.random.default_rng(seed)
        transition = _sample_stable_transition(config, rng)
        true_support = _edge_mask(transition, config.support_tolerance)
        states = rng.normal(size=(config.num_nodes, config.samples))

        for coverage in config.coverages:
            covered = _covered_indices(config, coverage)
            mixing = _mixing_for_coverage(config, covered, rng)
            learned_transition = mixing @ transition @ np.linalg.inv(mixing)

            learned_states = mixing @ states
            predicted_next = learned_transition @ learned_states
            target_next = mixing @ transition @ states
            observational_rmse = float(np.sqrt(np.mean((predicted_next - target_next) ** 2)))

            predicted_support = _edge_mask(learned_transition, config.support_tolerance)
            overall_f1 = _edge_f1(predicted_support, true_support)
            within_covered_f1 = (
                _edge_f1(predicted_support, true_support, covered) if covered.size else None
            )

            if covered.size:
                errors = []
                for index in covered:
                    latent_effect = mixing[:, index]
                    coordinate_edit = np.zeros(config.num_nodes)
                    coordinate_edit[index] = mixing[index, index]
                    errors.append(np.linalg.norm(latent_effect - coordinate_edit))
                intervention_alignment_error: Optional[float] = float(np.mean(errors))
            else:
                intervention_alignment_error = None

            rows.append(
                {
                    "seed": seed,
                    "coverage": float(coverage),
                    "covered_nodes": int(covered.size),
                    "overall_edge_f1": overall_f1,
                    "within_covered_edge_f1": within_covered_f1,
                    "observational_fit_rmse": observational_rmse,
                    "covered_intervention_alignment_error": intervention_alignment_error,
                }
            )

    summaries = []
    for coverage in config.coverages:
        group = [row for row in rows if row["coverage"] == float(coverage)]
        summary = {"coverage": float(coverage), "runs": len(group)}
        for key in (
            "overall_edge_f1",
            "within_covered_edge_f1",
            "observational_fit_rmse",
            "covered_intervention_alignment_error",
        ):
            values = [float(row[key]) for row in group if row[key] is not None]
            summary[f"{key}_mean"] = float(np.mean(values)) if values else None
            summary[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
        summaries.append(summary)

    payload = {
        "experiment": "constructive_linear_intervention_coverage_diagnostic",
        "scope": (
            "CPU-only theorem-aligned diagnostic; not a reproduction of the paper's "
            "multimodal experiments or reported tables."
        ),
        "config": asdict(config),
        "summary": summaries,
    }
    return rows, payload


def write_results(rows: Iterable[dict], payload: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_identifiability"))
    parser.add_argument("--num-nodes", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--quick", action="store_true", help="Run an 8-node, two-seed smoke diagnostic.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.num_nodes, args.num_blocks, args.seeds, args.samples = 8, 4, 2, 32
    config = DiagnosticConfig(
        num_nodes=args.num_nodes,
        num_blocks=args.num_blocks,
        seeds=args.seeds,
        samples=args.samples,
    )
    rows, payload = run_diagnostic(config)
    write_results(rows, payload, args.output_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
