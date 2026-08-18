from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.audit_robot_layer_long_video import LOWER_METRICS, UPPER_METRICS, _summary


def test_summary_reuses_frozen_limits_verbatim() -> None:
    np = pytest.importorskip("numpy")
    rows = []
    for frame in range(4):
        rows.append(
            {
                "frame": frame,
                **{name: 5.0 for name in UPPER_METRICS},
                **{name: 5.0 for name in LOWER_METRICS},
                "contact_required": False,
                "contact_observed": False,
            }
        )
    frozen = {
        **{name: 9.0 for name in UPPER_METRICS},
        **{name: 1.0 for name in LOWER_METRICS},
    }
    args = SimpleNamespace(
        anchor_start=0,
        anchor_end_exclusive=2,
        late_start=2,
        allowed_late_violation_fraction=0.0,
        required_contact_recall=1.0,
        persistent_grasp_start=-1,
        persistent_grasp_end_exclusive=-1,
        required_persistent_grasp_recall=1.0,
    )

    summary = _summary(np, rows, args, frozen_limits=frozen)

    assert summary["limits_fit_only_on_anchor_frames"] == frozen
    assert summary["image_space_contract_pass"] is True


def test_summary_rejects_incomplete_frozen_limits() -> None:
    np = pytest.importorskip("numpy")
    rows = [
        {
            "frame": frame,
            **{name: 5.0 for name in UPPER_METRICS},
            **{name: 5.0 for name in LOWER_METRICS},
            "contact_required": False,
            "contact_observed": False,
        }
        for frame in range(4)
    ]
    args = SimpleNamespace(
        anchor_start=0,
        anchor_end_exclusive=2,
        late_start=2,
        allowed_late_violation_fraction=0.0,
        required_contact_recall=1.0,
        persistent_grasp_start=-1,
        persistent_grasp_end_exclusive=-1,
        required_persistent_grasp_recall=1.0,
    )

    with pytest.raises(ValueError, match="omit metrics"):
        _summary(np, rows, args, frozen_limits={})
