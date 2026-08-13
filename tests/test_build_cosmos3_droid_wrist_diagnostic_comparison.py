from __future__ import annotations

import pytest

from scripts.build_cosmos3_droid_wrist_diagnostic_comparison import (
    validate_diagnostic_evaluation,
)


def test_rejected_strict_evaluation_is_diagnostic_only() -> None:
    status, accepted = validate_diagnostic_evaluation(
        {
            "method": "phiagent_cosmos3_droid_wrist_only_to_exterior_strict_validation",
            "status": "PARTIAL",
            "accepted": False,
            "disclosure": {
                "pure_wrist_only_claim": True,
                "condition_contains_third_person_pixels": False,
            },
        }
    )
    assert status == "PARTIAL"
    assert accepted is False


def test_diagnostic_rejects_undisclosed_third_person_condition() -> None:
    with pytest.raises(ValueError, match="third-person"):
        validate_diagnostic_evaluation(
            {
                "method": "phiagent_cosmos3_droid_wrist_only_to_exterior_strict_validation",
                "status": "PARTIAL",
                "accepted": False,
                "disclosure": {
                    "pure_wrist_only_claim": True,
                    "condition_contains_third_person_pixels": True,
                },
            }
        )
