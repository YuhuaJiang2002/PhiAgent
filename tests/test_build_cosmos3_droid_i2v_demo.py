from __future__ import annotations

import pytest

from scripts.build_cosmos3_droid_i2v_demo import (
    generation_title,
    require_accepted_evaluation,
)


def test_demo_requires_strict_accepted_validation() -> None:
    accepted = {
        "method": "phiagent_cosmos3_droid_multiview_i2v_strict_validation",
        "status": "WORKING",
        "accepted": True,
        "allowed_split": "validation",
    }
    require_accepted_evaluation(accepted)
    for key, value in (("accepted", False), ("status", "PARTIAL")):
        rejected = {**accepted, key: value}
        with pytest.raises(ValueError, match="unaccepted"):
            require_accepted_evaluation(rejected)


def test_first_frame_is_never_labeled_as_generated() -> None:
    first_title, first_subtitle = generation_title(0)
    generated_title, generated_subtitle = generation_title(1)
    assert first_title == "REAL CONDITION (FRAME 1)"
    assert "MODEL INPUT" in first_subtitle
    assert generated_title == "OUR GENERATED VIDEO"
    assert "NO REAL FUTURE INPUT" in generated_subtitle
