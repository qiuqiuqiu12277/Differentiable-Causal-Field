"""Validate and export evidence-backed OOD evaluation records.

This module deliberately does *not* turn an in-domain example into a
paper-style OOD example by changing only its question or ``ood_type`` field.
Scene, composition, template, and intervention shifts must already be present
as held-out records in the input manifest. The checks here cannot prove that a
dataset designer chose a scientifically valid split, but they do prevent the
old metadata-only construction from silently masquerading as one.

An opt-in synthetic template rewrite is retained solely for pipeline smoke
tests. Those records use the unmistakable ``synthetic_template_smoke`` label,
which must never be reported as a paper OOD result.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, replace
from pathlib import Path

from benchmark_datasets import CausalSample, samples_from_manifest

PAPER_OOD_TYPES = (
    "Scene Shift",
    "Object Composition Shift",
    "Template Shift",
    "Intervention Shift",
)
SYNTHETIC_TEMPLATE_SMOKE = "synthetic_template_smoke"

_IN_DOMAIN_ALIASES = {"", "id", "in domain", "indomain", "nan", "none", "null"}
_OOD_ALIASES = {
    "scene shift": "Scene Shift",
    "object composition shift": "Object Composition Shift",
    "composition shift": "Object Composition Shift",
    "template shift": "Template Shift",
    "intervention shift": "Intervention Shift",
}
_HELD_OUT_SPLITS = {
    "test",
    "eval",
    "evaluation",
    "ood",
}

TEMPLATE_REWRITES = {
    "what will happen": "what is the likely outcome",
    "what happens": "what is the resulting event",
    "if ": "suppose ",
    "why": "for what causal reason",
    "which": "identify which",
}


class OODIntegrityError(ValueError):
    """Raised when a manifest cannot support a defensible OOD evaluation."""


def _normalise_label(value: object) -> str:
    return " ".join(
        str(value or "").strip().lower().replace("_", "-").replace("-", " ").split()
    )


def canonical_ood_type(value: object) -> str | None:
    """Return the canonical paper label, ``None`` for ID, or raise for typos."""

    normalised = _normalise_label(value)
    if normalised in _IN_DOMAIN_ALIASES:
        return None
    if normalised == _normalise_label(SYNTHETIC_TEMPLATE_SMOKE):
        return SYNTHETIC_TEMPLATE_SMOKE
    if normalised not in _OOD_ALIASES:
        allowed = ", ".join(PAPER_OOD_TYPES)
        raise OODIntegrityError(
            f"Unknown ood_type {value!r}. Use 'in_domain' or one of: {allowed}. "
            "Do not invent a new label by relabelling an in-domain record."
        )
    return _OOD_ALIASES[normalised]


def to_record(sample: CausalSample):
    record = asdict(sample)
    record["graph_edges"] = [{"source": s, "target": t} for s, t in sample.graph_edges]
    return record


def rewrite_question(question: str) -> str:
    """Create a superficial paraphrase for an explicitly named smoke test."""

    q = question.strip()
    low = q.lower()
    for src, dst in TEMPLATE_REWRITES.items():
        if src in low:
            idx = low.index(src)
            return q[:idx] + dst + q[idx + len(src) :]
    if q.endswith("?"):
        return "Considering the same causal chain, " + q[:1].lower() + q[1:]
    return "Considering the same causal chain, " + q


def _media_reference(sample: CausalSample) -> str | None:
    return sample.image_path or sample.video_path or sample.feature_key


def _validate_local_media(sample: CausalSample) -> None:
    """Verify local primary media when a cached feature key is not supplied."""

    reference = sample.image_path or sample.video_path
    if not reference and not sample.feature_key:
        raise OODIntegrityError(
            f"Record {sample.sample_id!r} has no image_path, video_path, or feature_key. "
            "A label alone is not evidence for an OOD example."
        )
    references = [reference] if reference else []
    references.extend(
        value
        for value in (
            sample.counterfactual_image_path,
            sample.counterfactual_video_path,
        )
        if value
    )
    for media_path in references:
        if str(media_path).startswith(("http://", "https://")):
            raise OODIntegrityError(
                f"Record {sample.sample_id!r} uses remote media {media_path!r}; copy the official "
                "asset locally so its existence can be verified."
            )
        if not Path(media_path).is_file():
            raise OODIntegrityError(
                f"Record {sample.sample_id!r} references missing media: {media_path}. "
                "Provide the real held-out asset (or an existing feature_key), rather than a placeholder."
            )


def _validate_ood_evidence(sample: CausalSample, canonical_type: str) -> None:
    if _normalise_label(sample.split) not in _HELD_OUT_SPLITS:
        raise OODIntegrityError(
            f"OOD record {sample.sample_id!r} is in split {sample.split!r}. "
            "Final OOD records must be in test/eval/ood, never train or model-selection validation."
        )
    if not _media_reference(sample):
        raise OODIntegrityError(
            f"OOD record {sample.sample_id!r} has no real media or feature reference; "
            "changing ood_type metadata is not an OOD construction."
        )
    if canonical_type == "Object Composition Shift" and not (
        sample.objects or sample.factors
    ):
        raise OODIntegrityError(
            f"Object Composition Shift record {sample.sample_id!r} has neither object annotations "
            "nor factors with which to substantiate the composition."
        )
    if canonical_type == "Template Shift" and not (sample.question or sample.text):
        raise OODIntegrityError(
            f"Template Shift record {sample.sample_id!r} has no question/template text."
        )
    if canonical_type == "Intervention Shift":
        if not sample.intervention:
            raise OODIntegrityError(
                f"Intervention Shift record {sample.sample_id!r} has no explicit intervention."
            )
        has_counterfactual_target = bool(
            sample.counterfactual_image_path
            or sample.counterfactual_video_path
            or sample.counterfactual_text
            or sample.cf_answer
        )
        if not has_counterfactual_target:
            raise OODIntegrityError(
                f"Intervention Shift record {sample.sample_id!r} has no counterfactual media or "
                "counterfactual target. An intervention dictionary alone is insufficient evidence."
            )


def _observation_identity(sample: CausalSample) -> str | None:
    value = _media_reference(sample)
    return str(value) if value else None


def select_verified_ood_samples(
    samples: Iterable[CausalSample],
    *,
    include_in_domain: bool = True,
    synthetic_template_smoke: bool = False,
) -> list[CausalSample]:
    """Select pre-existing held-out OOD records and optionally add smoke records.

    Scene and composition shifts are rejected when they reuse an observation
    that is also labelled in-domain in the same manifest. Template and
    intervention questions may legitimately share factual media, but still
    require the type-specific evidence checked above.
    """

    source = list(samples)
    if not source:
        raise OODIntegrityError("The manifest contains no samples.")

    labelled = []
    in_domain = []
    for sample in source:
        label = canonical_ood_type(sample.ood_type)
        if label == SYNTHETIC_TEMPLATE_SMOKE:
            raise OODIntegrityError(
                f"Input record {sample.sample_id!r} already uses the reserved smoke-only label "
                f"{SYNTHETIC_TEMPLATE_SMOKE!r}. Generate smoke records with the explicit CLI flag instead."
            )
        if label is None:
            in_domain.append(replace(sample, ood_type="in_domain"))
            continue
        canonical = replace(sample, ood_type=label)
        _validate_ood_evidence(canonical, label)
        _validate_local_media(canonical)
        labelled.append(canonical)

    id_observations = {_observation_identity(s) for s in in_domain}
    for sample in labelled:
        if sample.ood_type in {"Scene Shift", "Object Composition Shift"}:
            identity = _observation_identity(sample)
            if identity and identity in id_observations:
                raise OODIntegrityError(
                    f"{sample.ood_type} record {sample.sample_id!r} reuses in-domain observation "
                    f"{identity!r}. Supply a genuinely held-out scene/composition asset."
                )

    if not labelled and not synthetic_template_smoke:
        allowed = ", ".join(PAPER_OOD_TYPES)
        raise OODIntegrityError(
            "No evidence-backed OOD records were found. Annotate official held-out records with "
            f"an explicit ood_type ({allowed}) and the required media/targets. This tool will not "
            "fabricate paper OOD sets by editing questions or metadata."
        )

    selected: list[CausalSample] = []
    if include_in_domain:
        held_out_id = [
            s for s in in_domain if _normalise_label(s.split) in _HELD_OUT_SPLITS
        ]
        for sample in held_out_id:
            _validate_local_media(sample)
        selected.extend(held_out_id)
    selected.extend(labelled)

    if synthetic_template_smoke:
        smoke_sources = [
            s for s in in_domain if _normalise_label(s.split) in _HELD_OUT_SPLITS
        ]
        if not smoke_sources:
            raise OODIntegrityError(
                "Synthetic template smoke mode needs at least one held-out in-domain source record."
            )
        for sample in smoke_sources:
            _validate_local_media(sample)
        selected.extend(
            replace(
                sample,
                sample_id=f"{sample.sample_id}::synthetic-template-smoke",
                question=rewrite_question(sample.question or sample.text),
                ood_type=SYNTHETIC_TEMPLATE_SMOKE,
            )
            for sample in smoke_sources
        )
    return selected


def make_variants(
    samples: Iterable[CausalSample],
    include_in_domain: bool = True,
    *,
    synthetic_template_smoke: bool = False,
) -> list[CausalSample]:
    """Backward-compatible name for strict selection (not variant fabrication)."""

    return select_verified_ood_samples(
        samples,
        include_in_domain=include_in_domain,
        synthetic_template_smoke=synthetic_template_smoke,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate and export explicitly annotated, evidence-backed OOD records"
    )
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--drop_in_domain", action="store_true")
    parser.add_argument(
        "--synthetic_template_smoke",
        action="store_true",
        help=(
            "Also create clearly labelled synthetic_template_smoke paraphrases. "
            "This is a plumbing check, not a paper OOD benchmark."
        ),
    )
    args = parser.parse_args()

    samples = samples_from_manifest(args.manifest_path, args.data_root)
    selected = select_verified_ood_samples(
        samples,
        include_in_domain=not args.drop_in_domain,
        synthetic_template_smoke=args.synthetic_template_smoke,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.writelines(
            json.dumps(to_record(sample), ensure_ascii=False) + "\n"
            for sample in selected
        )
    real_ood = sum(sample.ood_type in PAPER_OOD_TYPES for sample in selected)
    smoke = sum(sample.ood_type == SYNTHETIC_TEMPLATE_SMOKE for sample in selected)
    print(
        f"Wrote {len(selected)} records to {output} "
        f"({real_ood} verified OOD, {smoke} synthetic smoke-only)"
    )


if __name__ == "__main__":
    main()
