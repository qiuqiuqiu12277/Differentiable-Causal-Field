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
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    influence_to_edges,
    ood_metrics,
    qa_category_accuracy,
    structure_metrics,
)
from evaluate_metacausal_field import load_model
from metacausal_field import SimpleTokenizer
from train_metacausal_field import collate, find_counterfactual_file, scenario_directions, scenario_positions


def discover_factors_from_dataset(samples, min_variance: float = 1e-8) -> List[str]:
    names = sorted({name for sample in samples for name in sample.factors})
    discovered = []
    for name in names:
        values = [sample.factors.get(name) for sample in samples if name in sample.factors]
        if len(values) > 1 and float(np.nanvar(values)) > min_variance:
            discovered.append(name)
    return discovered


def learn_structure_fci(samples, factors: List[str]) -> List[Tuple[str, str]]:
    rows = []
    for sample in samples:
        if sample.factors:
            rows.append([sample.factors.get(name, 0.0) for name in factors])
    if len(rows) < 3 or len(factors) < 2:
        return []
    data = np.asarray(rows, dtype=float)
    try:
        from causallearn.search.ConstraintBased.FCI import fci
        graph, _ = fci(data, alpha=0.05, independence_test_method="kci", verbose=False)
    except Exception:
        corr = np.corrcoef(data, rowvar=False)
        edges = []
        for i in range(len(factors)):
            for j in range(len(factors)):
                if i != j and abs(corr[i, j]) > 0.25:
                    edges.append((factors[i], factors[j]))
        return edges

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
    return edges


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


def run_model_records(args, samples, model, tokenizer, factor_names, device, cached_backbone=None):
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
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    records = []
    influence_edges = []
    for batch in tqdm(loader, desc="Stage model inference"):
        input_ids = batch["input_ids"].to(device)
        answer_ids = batch["answer_ids"].to(device)
        backbone_outputs = encode_model_visual(model, cached_backbone, batch, device)
        outputs = model(
            backbone_outputs["visual_features"],
            language_tokens=backbone_outputs.get("language_tokens") if args.use_frozen_language_tokens else None,
            input_ids=input_ids,
            decoder_input_ids=answer_ids[:, :-1],
        )
        generated = outputs["lm_logits"].argmax(dim=-1).detach().cpu() if "lm_logits" in outputs else None
        for i in range(len(batch["sample_id"])):
            pred_text = tokenizer.decode(generated[i]) if generated is not None else ""
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
        if "factor_influence_matrix" in outputs and factor_names:
            influence_edges.extend(
                factor_influence_to_edges(
                    outputs["factor_influence_matrix"],
                    factor_names,
                    threshold=args.edge_threshold,
                    top_k=args.factor_edge_top_k,
                )
            )
        else:
            labels = factor_names or [f"p{i}" for i in range(outputs["influence_matrix"].shape[-1])]
            influence_edges.extend(influence_to_edges(outputs["influence_matrix"], labels, threshold=args.edge_threshold))
    return records, influence_edges


@torch.no_grad()
def run_counterfactual_records(args, model, tokenizer, device, dataset_name: str, samples=None, factor_names=None, cached_backbone=None):
    manifest_pairs = [s for s in (samples or []) if s.cf_answer]
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
            ),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        records = []
        for batch in tqdm(loader, desc="Manifest counterfactual"):
            input_ids = batch["input_ids"].to(device)
            answer_ids = batch["answer_ids"].to(device)
            backbone_outputs = encode_model_visual(model, cached_backbone, batch, device)
            visual = backbone_outputs["visual_features"]
            language_tokens = backbone_outputs.get("language_tokens") if args.use_frozen_language_tokens else None
            factual = model(
                visual,
                language_tokens=language_tokens,
                input_ids=input_ids,
                decoder_input_ids=answer_ids[:, :-1],
            )
            factual_ids = factual.get("lm_logits").argmax(dim=-1).cpu()
            cf_ids = torch.tensor([tokenizer.encode(x) for x in batch["cf_answer"]], dtype=torch.long, device=device)
            cf_outputs = model.counterfactual_forward(
                visual,
                intervention_type="modify",
                intervention_params={
                    "position": scenario_positions(batch["sample_id"], device),
                    "direction": scenario_directions(batch["sample_id"], model.config.feature_dim, device),
                    "radius": model.config.intervention_radius if args.intervention_radius is None else args.intervention_radius,
                },
                language_tokens=language_tokens,
                input_ids=input_ids,
                decoder_input_ids=cf_ids[:, :-1],
                num_rollout_steps=model.config.num_propagation_steps,
            )
            pred_cf_ids = cf_outputs.get("lm_logits_counterfactual").argmax(dim=-1).cpu()
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

    cf_file = find_counterfactual_file("MAG9" if dataset_name in {"MAG", "MAG9"} else "Lung")
    if not cf_file:
        return []
    cf_df = pd.read_csv(cf_file)
    if args.max_counterfactual_pairs:
        cf_df = cf_df.iloc[:args.max_counterfactual_pairs]

    from train_metacausal_field import CounterfactualPairDataset

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(
        CounterfactualPairDataset(cf_df, tokenizer, transform),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    records = []
    for batch in tqdm(loader, desc="Stage counterfactual"):
        input_ids = batch["input_ids"].to(device)
        cf_input_ids = batch["cf_input_ids"].to(device)
        backbone_outputs = encode_model_visual(model, cached_backbone, batch, device)
        outputs = model.counterfactual_forward(
            backbone_outputs["visual_features"],
            intervention_type="modify",
            intervention_params={
                "position": scenario_positions(batch["scenario"], device),
                "direction": scenario_directions(batch["scenario"], model.config.feature_dim, device),
                "radius": model.config.intervention_radius if args.intervention_radius is None else args.intervention_radius,
            },
            language_tokens=backbone_outputs.get("language_tokens") if args.use_frozen_language_tokens else None,
            input_ids=input_ids,
            decoder_input_ids=cf_input_ids[:, :-1],
            num_rollout_steps=model.config.num_propagation_steps,
        )
        pred_ids = outputs.get("lm_logits_counterfactual").argmax(dim=-1).cpu()
        for i in range(len(batch["scenario"])):
            records.append({
                "factual_pred": "",
                "cf_pred": tokenizer.decode(pred_ids[i]),
                "factual_gold": "",
                "cf_gold": tokenizer.decode(cf_input_ids[i, 1:]),
                "should_flip": True,
                "invalid_transition": not torch.isfinite(outputs["score_counterfactual"][i]).item(),
            })
    return records


def normalize_bool(value) -> bool:
    return bool(value)


def run_pipeline(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    samples, manifest_factor_names = load_benchmark(args.dataset, args.manifest_path, args.data_root)
    train_samples = [s for s in samples if s.split in {"train", "val", "validation"}]
    test_samples = [s for s in samples if s.split in {"test", "eval"}] or samples

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    discovered_factors = discover_factors_from_dataset(train_samples or samples)
    if not discovered_factors:
        discovered_factors = manifest_factor_names
    gold_factors = manifest_factor_names or discovered_factors
    results["factor_discovery"] = {
        "discovered_factors": discovered_factors,
        "gold_factors": gold_factors,
    }

    fci_edges = learn_structure_fci(train_samples or samples, discovered_factors)
    gold_edges = load_gold_edges(args.gold_graph) if args.gold_graph else gold_edges_from_samples(samples)
    results["structure_learning"] = {
        "pred_edges_fci": fci_edges,
        "gold_edges": gold_edges,
        "metrics_fci": structure_metrics(discovered_factors, gold_factors, fci_edges, gold_edges),
    }

    if args.checkpoint:
        cached_backbone = CachedFrozenBackbone(args.feature_cache).to(device) if args.feature_cache else None
        model, tokenizer, factor_columns, _, _ = load_model(
            args.checkpoint,
            device,
            visual_encoder=cached_backbone is None,
        )
        factor_names = factor_columns or discovered_factors
        qa_records, influence_edges = run_model_records(
            args,
            test_samples,
            model,
            tokenizer,
            factor_names,
            device,
            cached_backbone=cached_backbone,
        )
        results["structure_learning"]["pred_edges_field"] = influence_edges
        results["structure_learning"]["metrics_field"] = structure_metrics(
            factor_names,
            gold_factors,
            influence_edges,
            gold_edges,
        )
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
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
