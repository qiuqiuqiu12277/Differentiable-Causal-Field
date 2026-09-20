from pathlib import Path

import pytest

from benchmark_datasets import CausalSample
from build_ood_splits import (
    PAPER_OOD_TYPES,
    SYNTHETIC_TEMPLATE_SMOKE,
    OODIntegrityError,
    select_verified_ood_samples,
)


def media(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    path.write_bytes(b"official-test-asset")
    return str(path)


def sample(tmp_path: Path, sample_id: str, **kwargs) -> CausalSample:
    defaults = {
        "split": "test",
        "image_path": media(tmp_path, f"{sample_id}.png"),
        "question": "What happens next?",
        "answer": "It moves.",
        "ood_type": "in_domain",
    }
    defaults.update(kwargs)
    return CausalSample(sample_id=sample_id, **defaults)


def test_unannotated_manifest_is_not_fabricated(tmp_path):
    records = [sample(tmp_path, "id")]

    with pytest.raises(OODIntegrityError, match="will not fabricate"):
        select_verified_ood_samples(records)


def test_real_held_out_records_are_selected_without_mutating_content(tmp_path):
    id_record = sample(tmp_path, "id")
    scene = sample(tmp_path, "scene", ood_type="scene_shift")
    composition = sample(
        tmp_path,
        "composition",
        ood_type="Object Composition Shift",
        objects=[{"id": "red-cube", "category": "cube"}],
    )
    template = sample(
        tmp_path,
        "template",
        ood_type="Template Shift",
        question="Which object caused the collision?",
    )
    intervention = sample(
        tmp_path,
        "intervention",
        ood_type="Intervention Shift",
        intervention={"type": "remove", "target": "red-cube"},
        cf_answer="The sphere does not move.",
    )

    selected = select_verified_ood_samples(
        [id_record, scene, composition, template, intervention]
    )

    assert selected[0] == id_record
    assert {record.ood_type for record in selected[1:]} == set(PAPER_OOD_TYPES)
    assert (
        next(record for record in selected if record.sample_id == "template").question
        == template.question
    )


def test_ood_only_manifest_does_not_require_an_id_comparison_record(tmp_path):
    template = sample(tmp_path, "template-only", ood_type="Template Shift")

    selected = select_verified_ood_samples([template])

    assert selected == [template]


def test_scene_or_composition_shift_cannot_relabel_id_media(tmp_path):
    shared = media(tmp_path, "shared.png")
    id_record = sample(tmp_path, "id", image_path=shared)
    fake_scene = sample(
        tmp_path, "fake-scene", image_path=shared, ood_type="Scene Shift"
    )

    with pytest.raises(OODIntegrityError, match="reuses in-domain observation"):
        select_verified_ood_samples([id_record, fake_scene])


def test_intervention_label_requires_real_target_evidence(tmp_path):
    id_record = sample(tmp_path, "id")
    metadata_only = sample(
        tmp_path,
        "metadata-only",
        ood_type="Intervention Shift",
        intervention={"type": "remove", "target": "cube"},
    )

    with pytest.raises(
        OODIntegrityError, match="no counterfactual media or counterfactual target"
    ):
        select_verified_ood_samples([id_record, metadata_only])


def test_ood_records_must_be_held_out(tmp_path):
    leaked = sample(tmp_path, "leaked", split="train", ood_type="Template Shift")

    with pytest.raises(OODIntegrityError, match="never train"):
        select_verified_ood_samples([leaked], include_in_domain=False)


def test_missing_primary_or_counterfactual_media_is_rejected(tmp_path):
    missing_primary = CausalSample(
        sample_id="missing-primary",
        split="test",
        image_path=str(tmp_path / "does-not-exist.png"),
        question="What happens?",
        ood_type="Template Shift",
    )
    with pytest.raises(OODIntegrityError, match="missing media"):
        select_verified_ood_samples([missing_primary], include_in_domain=False)

    missing_cf = sample(
        tmp_path,
        "missing-cf",
        ood_type="Intervention Shift",
        intervention={"type": "remove", "target": "cube"},
        counterfactual_image_path=str(tmp_path / "does-not-exist-cf.png"),
        cf_answer="No collision.",
    )
    with pytest.raises(OODIntegrityError, match="missing media"):
        select_verified_ood_samples([missing_cf], include_in_domain=False)


def test_synthetic_template_mode_is_explicit_and_never_uses_paper_labels(tmp_path):
    id_record = sample(tmp_path, "id")
    selected = select_verified_ood_samples(
        [id_record],
        synthetic_template_smoke=True,
    )

    assert [record.ood_type for record in selected] == [
        "in_domain",
        SYNTHETIC_TEMPLATE_SMOKE,
    ]
    smoke = selected[1]
    assert smoke.sample_id.endswith("::synthetic-template-smoke")
    assert smoke.question != id_record.question
    assert smoke.ood_type not in PAPER_OOD_TYPES


def test_reserved_smoke_label_cannot_enter_through_manifest(tmp_path):
    prelabelled = sample(tmp_path, "prelabelled", ood_type=SYNTHETIC_TEMPLATE_SMOKE)

    with pytest.raises(OODIntegrityError, match="reserved smoke-only label"):
        select_verified_ood_samples([prelabelled], include_in_domain=False)
