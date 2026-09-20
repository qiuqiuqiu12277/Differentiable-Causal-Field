"""Run leakage-resistant baseline models on unified benchmark manifests."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_datasets import CausalSample, load_benchmark
from causal_metrics import answer_correct, ood_metrics, qa_category_accuracy


_SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "eval": "test",
    "evaluation": "test",
    "ood": "test",
}


def build_prompt(sample):
    choices = ""
    if sample.choices:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        choices = "\nChoices:\n" + "\n".join(
            f"{letters[i]}. {choice}" for i, choice in enumerate(sample.choices)
        )
    return (
        "Answer the visual causal reasoning question. "
        "Return only the final answer, without explanation.\n"
        f"Question: {sample.question or sample.text}{choices}"
    )


def _leakage_keys(sample: CausalSample) -> set[str]:
    values = {
        sample.group_id,
        sample.image_path,
        sample.video_path,
        sample.counterfactual_image_path,
        sample.counterfactual_video_path,
    }
    keys = {str(value) for value in values if value}
    return keys or {str(sample.sample_id)}


def partition_baseline_samples(samples, *, require_train: bool) -> tuple[list, list]:
    """Return train/test only; validation never contributes baseline statistics."""
    partitions = {"train": [], "val": [], "test": []}
    unknown = set()
    for sample in samples:
        raw = str(sample.split or "").strip().lower()
        label = _SPLIT_ALIASES.get(raw)
        if label is None:
            unknown.add(raw or "<empty>")
        else:
            partitions[label].append(sample)
    if unknown:
        raise ValueError(f"Unsupported baseline split labels: {sorted(unknown)}")
    if require_train and not partitions["train"]:
        raise ValueError("The majority/random baselines require a non-empty training partition.")
    if not partitions["test"]:
        raise ValueError("Baseline evaluation requires a non-empty explicit test/eval partition.")

    sample_ids = {
        name: {str(sample.sample_id) for sample in partition}
        for name, partition in partitions.items()
    }
    groups = {
        name: set().union(*(_leakage_keys(sample) for sample in partition))
        if partition
        else set()
        for name, partition in partitions.items()
    }
    for left in ("train", "val"):
        duplicate_ids = sample_ids[left] & sample_ids["test"]
        duplicate_groups = groups[left] & groups["test"]
        if duplicate_ids or duplicate_groups:
            raise ValueError(
                f"Data leakage between {left} and test: "
                f"sample_ids={sorted(duplicate_ids)[:5]}, "
                f"media_groups={sorted(duplicate_groups)[:5]}"
            )
    return partitions["train"], partitions["test"]


def predict_majority(sample, majority_by_type, global_majority):
    return majority_by_type.get(sample.question_type, global_majority)


def predict_random(sample, rng: random.Random, answer_pool: list[str]):
    candidates = list(sample.choices) or answer_pool
    return rng.choice(candidates) if candidates else ""


def predict_gemini(sample, model_name):
    from gemini_utils import generate

    media = []
    if sample.image_path:
        media.append(sample.image_path)
    if sample.video_path:
        media.append(sample.video_path)
    kwargs = {"img_path": media or None}
    if model_name:
        kwargs["model"] = model_name
    result = generate(build_prompt(sample), **kwargs)
    if not isinstance(result, dict):
        raise RuntimeError(f"Gemini returned no structured prediction for sample {sample.sample_id!r}")
    return result.get("text", "")


def load_hf_model(model_name, device):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=device if device == "auto" else None,
    )
    if device != "auto":
        model.to(torch.device(device))
    model.eval()
    return processor, model


def predict_hf(sample, processor, model, device, max_new_tokens):
    import torch
    from PIL import Image

    content = []
    image = None
    if sample.image_path:
        try:
            with Image.open(sample.image_path) as source:
                image = source.convert("RGB")
        except (OSError, ValueError) as exc:
            raise FileNotFoundError(
                f"Unable to load baseline image {sample.image_path!r}"
            ) from exc
        content.append({"type": "image", "image": image})
    elif sample.video_path:
        if not Path(sample.video_path).is_file():
            raise FileNotFoundError(f"Unable to load baseline video {sample.video_path!r}")
        content.append({"type": "video", "video": sample.video_path})
    content.append({"type": "text", "text": build_prompt(sample)})
    messages = [{"role": "user", "content": content}]
    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            images=[image] if image is not None else None,
            videos=[sample.video_path] if sample.video_path and image is None else None,
            return_tensors="pt",
        )
    except Exception as exc:
        raise RuntimeError(
            f"The selected HF processor could not encode sample {sample.sample_id!r}; "
            "refusing to silently drop its visual input."
        ) from exc
    if device != "auto":
        inputs = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )[0]


def main():
    parser = argparse.ArgumentParser(description="Run baseline predictions")
    parser.add_argument("--dataset", required=True, choices=["MAG", "MAG9", "Lung", "Lung4", "CLEVRER", "Causal3DIdent", "CITRIS", "Causal-VidQA"])
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--backend", default="majority", choices=["majority", "random", "gemini", "hf"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./baseline_predictions.jsonl")
    args = parser.parse_args()

    samples, _ = load_benchmark(
        args.dataset,
        args.manifest_path,
        args.data_root,
        seed=args.seed,
    )
    require_train = args.backend in {"majority", "random"}
    train, test = partition_baseline_samples(samples, require_train=require_train)

    answer_pool = [sample.answer for sample in train if sample.answer]
    answer_counts = Counter(answer_pool)
    global_majority = answer_counts.most_common(1)[0][0] if answer_counts else ""
    by_type = defaultdict(Counter)
    for sample in train:
        if sample.answer:
            by_type[sample.question_type][sample.answer] += 1
    majority_by_type = {
        question_type: counts.most_common(1)[0][0]
        for question_type, counts in by_type.items()
    }
    rng = random.Random(args.seed)

    hf = None
    if args.backend == "hf":
        hf = load_hf_model(
            args.model or "Qwen/Qwen3-VL-8B-Instruct",
            args.device,
        )

    records = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for sample in test:
            if args.backend == "majority":
                prediction = predict_majority(sample, majority_by_type, global_majority)
            elif args.backend == "random":
                prediction = predict_random(sample, rng, answer_pool)
            elif args.backend == "gemini":
                prediction = predict_gemini(sample, args.model)
            else:
                prediction = predict_hf(
                    sample,
                    hf[0],
                    hf[1],
                    args.device,
                    args.max_new_tokens,
                )
            record = {
                "sample_id": sample.sample_id,
                "question_type": sample.question_type,
                "ood_type": sample.ood_type,
                "pred": prediction,
                "gold": sample.answer,
                "correct": answer_correct(prediction, sample.answer, sample.choices),
            }
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = {
        "protocol": {
            "backend": args.backend,
            "seed": args.seed,
            "train_count": len(train),
            "test_count": len(test),
            "model_selection_data_used": False,
        },
        "qa": qa_category_accuracy(records),
        "ood": ood_metrics(records),
    }
    metrics_path = output.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
