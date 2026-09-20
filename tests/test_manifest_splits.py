from __future__ import annotations

import copy
import sys
import types

import pytest

try:
    import torchvision  # noqa: F401
except (ImportError, RuntimeError):
    for module_name in [name for name in sys.modules if name.startswith("torchvision")]:
        del sys.modules[module_name]
    torchvision_stub = types.ModuleType("torchvision")
    torchvision_stub.transforms = types.ModuleType("torchvision.transforms")
    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.transforms"] = torchvision_stub.transforms

from prepare_causalvqa_manifest import assign_group_splits


def _records(num_groups=6):
    return [
        {
            "sample_id": f"video-{group}-question-{question}",
            "group_id": f"video-{group}",
            "split": "",
        }
        for group in range(num_groups)
        for question in range(2)
    ]


def test_group_split_is_deterministic_nonempty_and_keeps_video_questions_together():
    first = assign_group_splits(copy.deepcopy(_records()), seed=17)
    second = assign_group_splits(copy.deepcopy(_records()), seed=17)

    assert [row["split"] for row in first] == [row["split"] for row in second]
    assert {row["split"] for row in first} == {"train", "val", "test"}
    by_group = {}
    for row in first:
        by_group.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(labels) == 1 for labels in by_group.values())


def test_group_split_rejects_too_few_groups_and_explicit_media_leakage():
    with pytest.raises(ValueError, match="At least three distinct"):
        assign_group_splits(_records(num_groups=2))

    leaking = _records(num_groups=3)
    leaking[0]["split"] = "train"
    leaking[1]["split"] = "test"
    with pytest.raises(ValueError, match="leaks shared media"):
        assign_group_splits(leaking)
