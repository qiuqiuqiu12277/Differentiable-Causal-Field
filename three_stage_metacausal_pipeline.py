"""Strict three-stage MetaCausalField experiment pipeline.

Stages:
1. factor_discovery
2. structure_learning
3. counterfactual_reasoning

This script is intentionally explicit because the paper describes a staged
protocol rather than only end-to-end training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from benchmark_datasets import UnifiedCausalDataset, load_benchmark
from frozen_backbones import CachedFrozenBackbone
from causal_metrics import (
    answer_correct,
    counterfactual_consistency,
    factor_influence_to_edges,
    ood_metrics,
    qa_category_accuracy,
    structure_metrics,
)
from evaluate_metacausal_field import load_model
from train_metacausal_field import (
    collate,
    intervention_tensors,
    is_explicit_modify_intervention,
    set_global_seed,
)


def strict_sample_partitions(samples):
    """Return disjoint train/val/test lists and reject ambiguous protocols."""
    aliases = {"validation": "val", "dev": "val", "eval": "test"}
    partitions = {"train": [], "val": [], "test": []}
    for sample in samples:
        label = aliases.get(str(sample.split).strip().lower(), str(sample.split).strip().lower())
        if label not in partitions:
            raise ValueError(
                f"Sample {sample.sample_id!r} has unsupported split {sample.split!r}; "
                "expected train, val, or test."
            )
        partitions[label].append(sample)
    if not partitions["train"] or not partitions["test"]:
        raise ValueError("The pipeline requires non-empty, explicit train and test partitions.")

    def leakage_keys(sample):
        values = {
            sample.group_id,
            sample.video_path,
            sample.image_path,
            sample.counterfactual_video_path,
            sample.counterfactual_image_path,
        }
        return {str(value) for value in values if value} or {str(sample.sample_id)}

    group_sets = {
        name: set().union(*(leakage_keys(sample) for sample in rows))
        for name, rows in partitions.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            preview = sorted(str(x) for x in overlap)[:3]
            raise ValueError(f"Media/group leakage between {left} and {right}: {preview}")
    return partitions["train"], partitions["val"], partitions["test"]


def discover_factors_from_dataset(samples, min_variance: float = 1e-8) -> List[str]:
    names = sorted({name for sample in samples for name in sample.factors})
    discovered = []
    for name in names:
        values = [sample.factors.get(name) for sample in samples if name in sample.factors]
        if len(values) > 1 and float(np.nanvar(values)) > min_variance:
            discovered.append(name)
    return discovered


def learn_structure_fci(samples, factors: List[str]) -> Tuple[List[Tuple[str, str]], str]:
    rows = []
    for sample in samples:
        if sample.factors:
            rows.append([sample.factors.get(name, 0.0) for name in factors])
    if len(rows) < 3 or len(factors) < 2:
        return [], "unavailable_insufficient_training_data"
    data = np.asarray(rows, dtype=float)
    try:
        from causallearn.search.ConstraintBased.FCI import fci
        graph, _ = fci(data, alpha=0.05, independence_test_method="kci", verbose=False)
    except Exception as exc:
        corr = np.corrcoef(data, rowvar=False)
        edges = []
        for i in range(len(factors)):
            for j in range(len(factors)):
                if i != j and abs(corr[i, j]) > 0.25:
                    edges.append((factors[i], factors[j]))
        return edges, f"correlation_smoke_fallback:{type(exc).__name__}"

    from causallearn.graph.Endpoint import Endpoint
    edges = []
    for edge in graph.get_graph_edges():
        n1 = edge.get_node1().get_name()
        n2 = edge.get_node2().get_name()
        try:
            i = int(n1.replace("X", "")) - 1
            j = int(n2.replace("X", "")) - 1
        except ValueError:
            continue
        if 0 <= i < len(factors) and 0 <= j < len(factors):
            e1 = edge.get_endpoint1()
            e2 = edge.get_endpoint2()
            if e1 == Endpoint.TAIL and e2 == Endpoint.ARROW:
                edges.append((factors[i], factors[j]))
            elif e2 == Endpoint.TAIL and e1 == Endpoint.ARROW:
                edges.append((factors[j], factors[i]))
            elif e1 == Endpoint.CIRCLE and e2 == Endpoint.ARROW:
                edges.append((factors[i], factors[j]))
            elif e2 == Endpoint.CIRCLE and e1 == Endpoint.ARROW:
                edges.append((factors[j], factors[i]))
            else:
                # Undirected/ambiguous PAG edge: retain both possible directions
                # so downstream ESHD penalizes uncertainty without dropping signal.
                edges.append((factors[i], factors[j]))
                edges.append((factors[j], factors[i]))
    return edges, "fci"


def gold_edges_from_samples(samples) -> List[Tuple[str, str]]:
    counts = {}
    for sample in samples:
        for edge in sample.graph_edges:
            counts[edge] = counts.get(edge, 0) + 1
    return [edge for edge, count in counts.items() if count > 0]


def load_gold_edges(path: str) -> List[Tuple[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Gold graph file not found: {p}")
    if p.suffix == ".csv":
        df = pd.read_csv(p)
        src_col = "source" if "source" in df.columns else df.columns[0]
        dst_col = "target" if "target" in df.columns else df.columns[1]
        return [(str(r[src_col]), str(r[dst_col])) for _, r in df.iterrows()]
    if p.suffix in {".json", ".jsonl"}:
        from benchmark_datasets import read_manifest, normalize_edges
        rows = read_manifest(str(p))
        if isinstance(rows, list) and rows and "source" in rows[0]:
            return [(str(r["source"]), str(r["target"])) for r in rows]
        edges = []
        for row in rows:
            edges.extend(normalize_edges(row.get("graph_edges", row.get("edges"))))
        return edges
    if p.suffix == ".dot":
        import pydot
        graphs = pydot.graph_from_dot_file(str(p))
        edges = []
        for graph in graphs or []:
            for edge in graph.get_edges():
                edges.append((edge.get_source().strip('"'), edge.get_destination().strip('"')))
        return edges
    raise ValueError(f"Unsupported gold graph format: {p.suffix}")


@torch.no_grad()
def encode_model_visual(model, cached_backbone, batch, device):
    if cached_backbone is not None:
        return cached_backbone.encode_by_keys(batch["feature_key"], device)
    return {"visual_features": model.visual_encoder(batch["image"].to(device))}


def run_model_records(
    args,
    samples,
    model,
    tokenizer,
    factor_names,
    device,
    cached_backbone=None,
    factor_graph_supervised=False,
):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = UnifiedCausalDataset(
        samples,
        tokenizer,
        transform,
        factor_names=factor_names,
        num_video_frames=args.num_video_frames,
        allow_missing_media=getattr(args, "allow_missing_media", False),
        skip_media_loading=cached_backbone is not None,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    records = []
    influence_edges = [] if factor_graph_supervised else None
    for batch in tqdm(loader, desc="Stage model inference"):
        input_ids = batch["input_ids"].to(device)
        backbone_outputs = encode_model_visual(model, cached_backbone, batch, device)
        outputs = model.generate_text(
            backbone_outputs["visual_features"],
            language_tokens=backbone_outputs.get("language_tokens") if args.use_frozen_language_tokens else None,
            input_ids=input_ids,
            max_new_tokens=args.generation_max_new_tokens,
        )
        generated = outputs["generated_ids"].detach().cpu()
        for i in range(len(batch["sample_id"])):
            pred_text = tokenizer.decode(generated[i])
            gold = batch["answer"][i]
            correct = answer_correct(pred_text, gold, batch["choices"][i] if "choices" in batch else None)
            records.append({
                "sample_id": batch["sample_id"][i],
                "question_type": batch["question_type"][i],
                "ood_type": batch["ood_type"][i],
                "pred": pred_text,
                "gold": gold,
                "correct": correct,
            })
        if factor_graph_supervised:
            if "factor_influence_matrix" not in outputs or not factor_names:
                raise RuntimeError(
                    "Checkpoint declares factor-graph supervision but factor influence output is unavailable."
                )
            influence_edges.extend(
                factor_influence_to_edges(
                    outputs["factor_influence_matrix"],
                    factor_names,
                    threshold=args.edge_threshold,
                    top_k=args.factor_edge_top_k,
                )
            )
    return records, influence_edges


@torch.no_grad()
def run_counterfactual_records(args, model, tokenizer, device, dataset_name: str, samples=None, factor_names=None, cached_backbone=None):
    del dataset_name
    # A counterfactual answer without an explicit intervention is not an
    # auditable pair: hashing a scenario name into a field edit would measure a
    # synthetic plumbing heuristic, not counterfactual reasoning.
    manifest_pairs = [
        sample
        for sample in (samples or [])
        if sample.answer
        and sample.cf_answer
        and is_explicit_modify_intervention(
            sample.intervention,
            feature_dim=model.config.feature_dim,
            require_position=True,
        )
    ]
    if args.max_counterfactual_pairs:
        manifest_pairs = manifest_pairs[:args.max_counterfactual_pairs]
    if manifest_pairs:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        loader = DataLoader(
            UnifiedCausalDataset(
                manifest_pairs,
                tokenizer,
                transform,
                factor_names=factor_names,
                num_video_frames=args.num_video_frames,
                allow_missing_media=getattr(args, "allow_missing_media", False),
                skip_media_loading=cached_backbone is not None,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        records = []
        for batch in tqdm(loader, desc="Manifest counterfactual"):
            input_ids = batch["input_ids"].to(device)
            backbone_outputs = encode_model_visual(model, cached_backbone, batch, device)
            visual = backbone_outputs["visual_features"]
            language_tokens = backbone_outputs.get("language_tokens") if args.use_frozen_language_tokens else None
            factual = model.generate_text(
                visual,
                language_tokens=language_tokens,
                input_ids=input_ids,
                max_new_tokens=args.generation_max_new_tokens,
            )
            factual_ids = factual["generated_ids"].cpu()
            positions, directions = intervention_tensors(
                batch["intervention"],
                batch["sample_id"],
                model.config.feature_dim,
                device,
            )
            cf_outputs = model.generate_counterfactual_text(
                visual,
                intervention_type="modify",
                intervention_params={
                    "position": positions,
                    "direction": directions,
                    "radius": model.config.intervention_radius if args.intervention_radius is None else args.intervention_radius,
                },
                language_tokens=language_tokens,
                input_ids=input_ids,
                num_rollout_steps=model.config.num_propagation_steps,
                max_new_tokens=args.generation_max_new_tokens,
            )
            pred_cf_ids = cf_outputs["generated_ids_counterfactual"].cpu()
            for i in range(len(batch["sample_id"])):
                factual_pred = tokenizer.decode(factual_ids[i])
                cf_pred = tokenizer.decode(pred_cf_ids[i])
                records.append({
                    "factual_pred": factual_pred,
                    "cf_pred": cf_pred,
                    "factual_gold": batch["answer"][i],
                    "cf_gold": batch["cf_answer"][i],
                    "should_flip": normalize_bool(batch["answer"][i] != batch["cf_answer"][i]),
                    "invalid_transition": not torch.isfinite(cf_outputs["score_counterfactual"][i]).item(),
                })
        return records

    return []


def normalize_bool(value) -> bool:
    return bool(value)


def run_pipeline(args):
    set_global_seed(getattr(args, "seed", 42))
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    samples, manifest_factor_names = load_benchmark(
        args.dataset,
        args.manifest_path,
        args.data_root,
        seed=getattr(args, "seed", 42),
    )
    train_samples, val_samples, test_samples = strict_sample_partitions(samples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    discovered_factors = discover_factors_from_dataset(train_samples)
    gold_factors = manifest_factor_names or discovered_factors
    results["factor_discovery"] = {
        "discovered_factors": discovered_factors,
        "gold_factors": gold_factors,
        "status": "available" if discovered_factors else "unavailable_no_varying_training_factors",
    }

    fci_edges, structure_method = learn_structure_fci(train_samples, discovered_factors)
    gold_edges = load_gold_edges(args.gold_graph) if args.gold_graph else gold_edges_from_samples(samples)
    results["structure_learning"] = {
        "gold_edges": gold_edges,
        "classical_method": structure_method,
    }
    if structure_method == "fci":
        results["structure_learning"]["pred_edges_fci"] = fci_edges
        results["structure_learning"]["metrics_fci"] = structure_metrics(
            discovered_factors, gold_factors, fci_edges, gold_edges,
        )
    elif structure_method.startswith("correlation_smoke_fallback"):
        results["structure_learning"]["pred_edges_correlation_smoke"] = fci_edges
        results["structure_learning"]["metrics_correlation_smoke"] = structure_metrics(
            discovered_factors, gold_factors, fci_edges, gold_edges,
        )
    else:
        results["structure_learning"]["metrics_fci"] = {
            "status": "unavailable",
            "reason": structure_method,
        }

    if args.checkpoint:
        cached_backbone = (
            CachedFrozenBackbone(
                args.feature_cache,
                allow_smoke_cache=args.allow_missing_media,
            ).to(device)
            if args.feature_cache
            else None
        )
        model, tokenizer, factor_columns, _, loaded_checkpoint = load_model(
            args.checkpoint,
            device,
            visual_encoder=cached_backbone is None,
            expected_dataset=args.dataset,
        )
        factor_names = factor_columns or discovered_factors
        factor_graph_supervised = bool(
            loaded_checkpoint.get("supervision", {}).get("factor_localizer")
            and loaded_checkpoint.get("supervision", {}).get("factor_graph")
        )
        qa_records, influence_edges = run_model_records(
            args,
            test_samples,
            model,
            tokenizer,
            factor_names,
            device,
            cached_backbone=cached_backbone,
            factor_graph_supervised=factor_graph_supervised,
        )
        if factor_graph_supervised:
            results["structure_learning"]["pred_edges_field"] = influence_edges
            results["structure_learning"]["metrics_field"] = structure_metrics(
                factor_names,
                gold_factors,
                influence_edges,
                gold_edges,
            )
        else:
            results["structure_learning"]["metrics_field"] = {
                "status": "unavailable",
                "reason": (
                    "Checkpoint does not contain supervised factor localization/graph metadata; "
                    "patch influence weights are not reported as a named causal graph."
                ),
            }
        results["qa"] = qa_category_accuracy(qa_records)
        results["ood"] = ood_metrics(qa_records)
        cf_records = run_counterfactual_records(
            args,
            model,
            tokenizer,
            device,
            args.dataset,
            test_samples,
            factor_names,
            cached_backbone=cached_backbone,
        )
        results["counterfactual_reasoning"] = counterfactual_consistency(cf_records)
    else:
        results["qa"] = {}
        results["ood"] = {}
        results["counterfactual_reasoning"] = {}

    with open(output_dir / "three_stage_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Strict three-stage MetaCausalField pipeline")
    parser.add_argument("--dataset", default="MAG9", choices=["MAG", "MAG9", "Lung", "Lung4", "CLEVRER", "Causal3DIdent", "CITRIS", "Causal-VidQA"])
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument("--use_frozen_language_tokens", action="store_true")
    parser.add_argument("--gold_graph", default=None)
    parser.add_argument("--output_dir", default="./three_stage_outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--edge_threshold", type=float, default=0.05)
    parser.add_argument("--factor_edge_top_k", type=int, default=None)
    parser.add_argument("--intervention_radius", type=float, default=None)
    parser.add_argument("--max_counterfactual_pairs", type=int, default=32)
    parser.add_argument("--num_video_frames", type=int, default=8)
    parser.add_argument(
        "--generation_max_new_tokens",
        type=int,
        default=32,
        help="Maximum autoregressive answer length; gold answer prefixes are never used.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_missing_media",
        action="store_true",
        help="Use white placeholder media for smoke tests only.",
    )
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
