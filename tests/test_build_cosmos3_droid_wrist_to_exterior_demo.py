from __future__ import annotations

import pytest

from scripts.build_cosmos3_droid_wrist_to_exterior_demo import (
    require_accepted_evaluation,
)


def _accepted() -> dict[str, object]:
    return {
        "method": "phiagent_cosmos3_droid_wrist_only_to_exterior_strict_validation",
        "status": "WORKING",
        "accepted": True,
        "disclosure": {
            "pure_wrist_only_claim": True,
            "condition_contains_third_person_pixels": False,
        },
    }


def test_demo_accepts_only_strict_true_wrist_only_evaluation() -> None:
    require_accepted_evaluation(_accepted())
    bad = _accepted()
    bad["accepted"] = False
    with pytest.raises(ValueError, match="unaccepted"):
        require_accepted_evaluation(bad)


def test_demo_rejects_anchor_conditioned_claim() -> None:
    bad = _accepted()
    bad["disclosure"]["condition_contains_third_person_pixels"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="third-person pixels"):
        require_accepted_evaluation(bad)
