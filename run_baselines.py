"""Run baseline MLLM/API/HF models on unified benchmark manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_datasets import load_benchmark
from causal_metrics import answer_correct, ood_metrics, qa_category_accuracy


def build_prompt(sample):
    choices = ""
    if sample.choices:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        choices = "\nChoices:\n" + "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(sample.choices))
    return (
        "Answer the visual causal reasoning question. "
        "Return only the final answer, without explanation.\n"
        f"Question: {sample.question or sample.text}{choices}"
    )


def predict_majority(sample, majority_by_type, global_majority):
    return majority_by_type.get(sample.question_type, global_majority)


def predict_random(sample):
    if sample.choices:
        return sample.choices[0]
    return ""


def predict_gemini(sample, model_name):
    from gemini_utils import generate
    media = []
    if sample.image_path:
        media.append(sample.image_path)
    if sample.video_path:
        media.append(sample.video_path)
    result = generate(build_prompt(sample), img_path=media or None, model=model_name)
    return (result or {}).get("text", "")


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
    if sample.image_path:
        content.append({"type": "image", "image": Image.open(sample.image_path).convert("RGB")})
    elif sample.video_path:
        content.append({"type": "video", "video": sample.video_path})
    content.append({"type": "text", "text": build_prompt(sample)})
    messages = [{"role": "user", "content": content}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[content[0]["image"]] if sample.image_path else None,
            videos=[sample.video_path] if sample.video_path and not sample.image_path else None,
            return_tensors="pt",
        )
    except Exception:
        inputs = processor(text=build_prompt(sample), images=Image.open(sample.image_path).convert("RGB") if sample.image_path else None, return_tensors="pt")
    if device != "auto":
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser(description="Run baseline predictions")
    parser.add_argument("--dataset", required=True, choices=["MAG", "MAG9", "Lung", "Lung4", "CLEVRER", "Causal3DIdent", "CITRIS", "Causal-VidQA"])
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--backend", default="majority", choices=["majority", "random", "gemini", "hf"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--output", default="./baseline_predictions.jsonl")
    args = parser.parse_args()

    samples, _ = load_benchmark(args.dataset, args.manifest_path, args.data_root)
    train = [s for s in samples if s.split in {"train", "val", "validation"}]
    test = [s for s in samples if s.split in {"test", "eval"}] or samples

    answer_counts = Counter(s.answer for s in train if s.answer)
    global_majority = answer_counts.most_common(1)[0][0] if answer_counts else ""
    by_type = defaultdict(Counter)
    for s in train:
        if s.answer:
            by_type[s.question_type][s.answer] += 1
    majority_by_type = {k: c.most_common(1)[0][0] for k, c in by_type.items()}

    hf = None
    if args.backend == "hf":
        hf = load_hf_model(args.model or "Qwen/Qwen3-VL-8B-Instruct", args.device)

    records = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for sample in test:
            if args.backend == "majority":
                pred = predict_majority(sample, majority_by_type, global_majority)
            elif args.backend == "random":
                pred = predict_random(sample)
            elif args.backend == "gemini":
                pred = predict_gemini(sample, args.model)
            else:
                pred = predict_hf(sample, hf[0], hf[1], args.device, args.max_new_tokens)
            record = {
                "sample_id": sample.sample_id,
                "question_type": sample.question_type,
                "ood_type": sample.ood_type,
                "pred": pred,
                "gold": sample.answer,
                "correct": answer_correct(pred, sample.answer, sample.choices),
            }
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = {"qa": qa_category_accuracy(records), "ood": ood_metrics(records)}
    metrics_path = output.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
