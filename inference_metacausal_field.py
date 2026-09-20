"""Inference, explanation, and counterfactual analysis for InfluenceField."""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from metacausal_field import (
    CausalFieldConfig,
    MultimodalCausalField,
    SimpleTokenizer,
    VisualEncoder,
    compute_causal_effects,
    extract_causal_graph,
)
from train_metacausal_field import (
    intervention_tensors,
    scenario_directions,
    scenario_positions,
)
from frozen_backbones import CachedFrozenBackbone, load_torch_payload


class MetaCausalInference:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        feature_cache: str = None,
        allow_missing_media: bool = False,
        allow_synthetic_intervention_smoke: bool = False,
        generation_max_new_tokens: int = 32,
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.allow_missing_media = bool(allow_missing_media)
        self.allow_synthetic_intervention_smoke = bool(allow_synthetic_intervention_smoke)
        self.generation_max_new_tokens = int(generation_max_new_tokens)
        checkpoint = load_torch_payload(checkpoint_path, map_location=self.device)
        checkpoint_args = checkpoint.get("args", {}) or {}
        checkpoint_backbone = str(checkpoint_args.get("backbone", "resnet"))
        if checkpoint_args.get("train_qwen_lora"):
            raise ValueError(
                "Generic inference cannot reconstruct the checkpoint's trained Qwen LoRA adapter."
            )
        if checkpoint_backbone in {"cached", "qwen"} and not feature_cache:
            raise ValueError(
                f"Checkpoint backbone is {checkpoint_backbone!r}; pass the matching feature cache "
                "instead of substituting a random ResNet."
            )
        config_dict = checkpoint.get("config", {})
        self.config = CausalFieldConfig(**config_dict) if isinstance(config_dict, dict) else config_dict
        self.tokenizer = SimpleTokenizer.from_state_dict(checkpoint["tokenizer"])
        self.factor_columns = checkpoint.get("factor_columns", [])

        self.cached_backbone = (
            CachedFrozenBackbone(
                feature_cache,
                allow_smoke_cache=self.allow_missing_media,
            ).to(self.device)
            if feature_cache
            else None
        )
        self.visual_encoder = VisualEncoder(feature_dim=self.config.feature_dim, pretrained=False) if self.cached_backbone is None else None
        self.model = MultimodalCausalField(
            self.config,
            visual_encoder=self.visual_encoder,
            vocab_size=self.tokenizer.vocab_size,
            num_factors=len(self.factor_columns),
            enable_language=True,
        ).to(self.device)
        missing, unexpected = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint/model mismatch would leave random or unused parameters: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}."
            )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_image(self, image_path: str) -> Image.Image:
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as exc:
            if self.allow_missing_media:
                return Image.new("RGB", (224, 224), color="white")
            raise RuntimeError(
                f"Unable to load image {image_path!r}. Pass --allow_missing_media only for smoke tests."
            ) from exc

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        return self.transform(self.load_image(image_path)).unsqueeze(0).to(self.device)

    def encode_text(self, text: str) -> torch.Tensor:
        return torch.tensor([self.tokenizer.encode(text)], dtype=torch.long, device=self.device)

    def visual_features(
        self,
        image_path: str,
        image: torch.Tensor,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        if self.cached_backbone is not None:
            key = feature_key or image_path
            return self.cached_backbone.encode_by_keys([key], self.device)["visual_features"]
        return self.visual_encoder(image)

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        prompt: str = "describe the causal state",
        feature_key: Optional[str] = None,
    ) -> Dict:
        image = self.preprocess_image(image_path)
        input_ids = self.encode_text(prompt)
        visual_features = self.visual_features(image_path, image, feature_key)
        outputs = self.model.generate_text(
            visual_features,
            input_ids=input_ids,
            max_new_tokens=self.generation_max_new_tokens,
        )

        factors = {}
        if "factor_logits" in outputs:
            pred = outputs["factor_logits"].argmax(dim=-1)[0].cpu().tolist()
            label_map = {0: -1, 1: 0, 2: 1}
            factors = {name: label_map[int(value)] for name, value in zip(self.factor_columns, pred)}

        generated = self.tokenizer.decode(outputs["generated_ids"][0])

        return {
            "score": float(outputs["score_pred"][0].cpu()),
            "factors": factors,
            "generated_text": generated,
            "field": outputs["field"][0].cpu(),
            "initial_field": outputs["initial_field"][0].cpu(),
            "influence_matrix": outputs["influence_matrix"][0].cpu(),
        }

    @torch.no_grad()
    def generate_counterfactual(
        self,
        image_path: str,
        prompt: str,
        *,
        position: Optional[list[float]] = None,
        direction: Optional[list[float]] = None,
        scenario: Optional[str] = None,
        radius: Optional[float] = None,
        feature_key: Optional[str] = None,
    ) -> Dict:
        image = self.preprocess_image(image_path)
        input_ids = self.encode_text(prompt)
        visual_features = self.visual_features(image_path, image, feature_key)
        if position is not None or direction is not None:
            intervention = {"position": position, "direction": direction}
            positions, directions = intervention_tensors(
                [intervention],
                ["explicit-inference-intervention"],
                self.config.feature_dim,
                self.device,
            )
            intervention_source = "explicit"
        elif scenario and self.allow_synthetic_intervention_smoke:
            positions = scenario_positions([scenario], self.device)
            directions = scenario_directions(
                [scenario],
                self.config.feature_dim,
                self.device,
            )
            intervention_source = "synthetic_hash_smoke"
        else:
            raise ValueError(
                "Counterfactual inference requires --position X Y and an explicit "
                f"--direction with {self.config.feature_dim} values. Scenario-string hashing "
                "is available only with --allow_synthetic_intervention_smoke."
            )
        cf_outputs = self.model.generate_counterfactual_text(
            visual_features,
            intervention_type="modify",
            intervention_params={
                "position": positions,
                "direction": directions,
                "radius": self.config.intervention_radius if radius is None else radius,
            },
            input_ids=input_ids,
            num_rollout_steps=self.config.num_propagation_steps,
            max_new_tokens=self.generation_max_new_tokens,
        )

        generated = self.tokenizer.decode(cf_outputs["generated_ids_counterfactual"][0])

        return {
            "score_factual": float(cf_outputs["score_factual"][0].cpu()),
            "score_counterfactual": float(cf_outputs["score_counterfactual"][0].cpu()),
            "generated_counterfactual_text": generated,
            "intervention_source": intervention_source,
            "field_factual": cf_outputs["field_factual"][0].cpu(),
            "field_counterfactual": cf_outputs["field_counterfactual"][0].cpu(),
            "intervention_mask": cf_outputs["intervention_mask"][0].cpu(),
        }

    def visualize_causal_explanation(
        self,
        image_path: str,
        prompt: str,
        save_path: Optional[str] = None,
        feature_key: Optional[str] = None,
    ):
        result = self.predict(image_path, prompt, feature_key)
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes[0, 0].imshow(self.load_image(image_path))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        field_intensity = result["field"].mean(dim=-1).numpy()
        im1 = axes[0, 1].imshow(field_intensity, cmap="viridis")
        axes[0, 1].set_title("Causal Field Intensity")
        axes[0, 1].axis("off")
        plt.colorbar(im1, ax=axes[0, 1])

        field_delta = (result["field"] - result["initial_field"]).abs().mean(dim=-1).numpy()
        im2 = axes[0, 2].imshow(field_delta, cmap="magma")
        axes[0, 2].set_title("Propagation Delta")
        axes[0, 2].axis("off")
        plt.colorbar(im2, ax=axes[0, 2])

        im3 = axes[1, 0].imshow(result["influence_matrix"].numpy(), cmap="Blues")
        axes[1, 0].set_title("Directional Influence")
        plt.colorbar(im3, ax=axes[1, 0])

        graph = extract_causal_graph(result["influence_matrix"], threshold=0.05)
        im4 = axes[1, 1].imshow(graph, cmap="binary")
        axes[1, 1].set_title("Thresholded Influence Support")
        axes[1, 1].axis("off")
        plt.colorbar(im4, ax=axes[1, 1])

        effects = compute_causal_effects(
            result["field"].flatten(0, 1),
            result["influence_matrix"],
            intervention_position=result["influence_matrix"].shape[0] // 2,
        )
        h, w, _ = result["field"].shape
        im5 = axes[1, 2].imshow(effects.reshape(h, w), cmap="hot")
        axes[1, 2].set_title("Effects From Center")
        axes[1, 2].axis("off")
        plt.colorbar(im5, ax=axes[1, 2])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig, result

    def visualize_counterfactual(
        self,
        image_path: str,
        prompt: str,
        *,
        position: Optional[list[float]] = None,
        direction: Optional[list[float]] = None,
        scenario: Optional[str] = None,
        radius: Optional[float] = None,
        save_path: Optional[str] = None,
        feature_key: Optional[str] = None,
    ):
        result = self.generate_counterfactual(
            image_path,
            prompt,
            position=position,
            direction=direction,
            scenario=scenario,
            radius=radius,
            feature_key=feature_key,
        )
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(self.load_image(image_path))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")
        im1 = axes[0, 1].imshow(result["intervention_mask"].numpy(), cmap="Reds")
        axes[0, 1].set_title("Intervention Mask")
        axes[0, 1].axis("off")
        plt.colorbar(im1, ax=axes[0, 1])
        im2 = axes[1, 0].imshow(result["field_factual"].mean(dim=-1).numpy(), cmap="viridis")
        axes[1, 0].set_title("Factual Field")
        axes[1, 0].axis("off")
        plt.colorbar(im2, ax=axes[1, 0])
        im3 = axes[1, 1].imshow(result["field_counterfactual"].mean(dim=-1).numpy(), cmap="viridis")
        axes[1, 1].set_title("Counterfactual Field")
        axes[1, 1].axis("off")
        plt.colorbar(im3, ax=axes[1, 1])
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig, result


def main():
    parser = argparse.ArgumentParser(description="InfluenceField inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="describe the causal state")
    parser.add_argument("--output_dir", default="./inference_results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument(
        "--feature_key",
        default=None,
        help="Sample-specific key in --feature_cache (defaults to the image path for legacy caches).",
    )
    parser.add_argument("--generation_max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--allow_missing_media",
        action="store_true",
        help="Use a white placeholder only for smoke tests; strict loading is the default.",
    )
    parser.add_argument(
        "--allow_synthetic_intervention_smoke",
        action="store_true",
        help="Allow deterministic scenario-string edits for plumbing tests only.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("predict")
    subparsers.add_parser("explain")
    cf_parser = subparsers.add_parser("counterfactual")
    cf_parser.add_argument("--position", type=float, nargs=2, default=None, metavar=("X", "Y"))
    cf_parser.add_argument("--direction", type=float, nargs="+", default=None)
    cf_parser.add_argument(
        "--scenario",
        default=None,
        help="Smoke-only string used only with --allow_synthetic_intervention_smoke.",
    )
    cf_parser.add_argument("--radius", type=float, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = MetaCausalInference(
        args.checkpoint,
        args.device,
        args.feature_cache,
        allow_missing_media=args.allow_missing_media,
        allow_synthetic_intervention_smoke=args.allow_synthetic_intervention_smoke,
        generation_max_new_tokens=args.generation_max_new_tokens,
    )

    if args.command == "predict":
        result = engine.predict(args.image, args.prompt, args.feature_key)
        serializable = {k: v for k, v in result.items() if k not in {"field", "initial_field", "influence_matrix"}}
        print(json.dumps(serializable, indent=2))
    elif args.command == "explain":
        save_path = output_dir / f"explanation_{Path(args.image).stem}.png"
        _, result = engine.visualize_causal_explanation(
            args.image,
            args.prompt,
            str(save_path),
            args.feature_key,
        )
        print(json.dumps({"score": result["score"], "factors": result["factors"], "figure": str(save_path)}, indent=2))
    elif args.command == "counterfactual":
        save_path = output_dir / f"counterfactual_{Path(args.image).stem}.png"
        _, result = engine.visualize_counterfactual(
            args.image,
            args.prompt,
            position=args.position,
            direction=args.direction,
            scenario=args.scenario,
            radius=args.radius,
            save_path=str(save_path),
            feature_key=args.feature_key,
        )
        print(json.dumps({
            "score_factual": result["score_factual"],
            "score_counterfactual": result["score_counterfactual"],
            "generated_counterfactual_text": result["generated_counterfactual_text"],
            "intervention_source": result["intervention_source"],
            "figure": str(save_path),
        }, indent=2))


if __name__ == "__main__":
    main()
