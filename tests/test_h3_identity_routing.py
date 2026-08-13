from __future__ import annotations

import pytest

from phiagent.agent.h3_identity_routing import (
    ACTION_CONTEXT_KEYS,
    METRIC_CONTEXT_KEYS,
    REQUIRED_PROMOTION_GATES,
    decide_identity_delivery,
    require_matched_delivery_context,
)


def _assessment(
    *, passed: bool, failed: list[str] | None = None
) -> dict[str, object]:
    failed = failed or []
    return {
        "status": "accepted" if passed else "rejected",
        "honest_status": "WORKING" if passed else "PARTIAL",
        "assessment": {
            "passed": passed,
            "gates": {gate: gate not in failed for gate in REQUIRED_PROMOTION_GATES},
            "failed_gates": failed,
        },
    }


def _metrics(seed: str = "a") -> dict[str, object]:
    hashes = {
        key: (seed * 64 if key == "source" else chr(ord(seed) + index + 1) * 64)
        for index, key in enumerate(ACTION_CONTEXT_KEYS)
    }
    inputs = {
        key: {"sha256": chr(ord(seed) + index + 6) * 64}
        for index, key in enumerate(METRIC_CONTEXT_KEYS)
    }
    return {
        "inputs": inputs,
        "action_evidence": {"matched_context_sha256": hashes},
    }


def test_rejected_candidate_requires_task_bound_fallback_assessment() -> None:
    decision = decide_identity_delivery(
        _assessment(passed=False, failed=["topology", "motion_non_regression"]),
        _assessment(passed=True),
    )

    assert decision.route == "task_bound_fallback"
    assert "motion_non_regression" in decision.failed_candidate_gates


def test_candidate_requires_status_honesty_and_every_gate() -> None:
    decision = decide_identity_delivery(
        _assessment(passed=True),
        _assessment(passed=False, failed=["topology"]),
    )

    assert decision.route == "candidate"


def test_regressing_fallback_blocks_delivery() -> None:
    decision = decide_identity_delivery(
        _assessment(passed=False, failed=["topology"]),
        _assessment(
            passed=False,
            failed=["motion_non_regression", "action_non_regression"],
        ),
    )

    assert decision.route == "blocked"
    assert "motion_non_regression" in decision.failed_fallback_gates
    assert "action_non_regression" in decision.failed_fallback_gates


def test_legacy_fallback_manifest_cannot_be_used_as_assessment() -> None:
    legacy = {
        "status": "accepted",
        "honest_status": "WORKING",
        "acceptance": {"action_direction_correspondence_passed": True},
    }

    with pytest.raises(ValueError, match="fallback.assessment"):
        decide_identity_delivery(_assessment(passed=False, failed=["topology"]), legacy)


def test_partial_gate_list_cannot_claim_full_promotion() -> None:
    incomplete = _assessment(passed=True)
    incomplete["assessment"]["gates"] = {"topology": True}  # type: ignore[index]

    with pytest.raises(ValueError, match="missing required gates"):
        decide_identity_delivery(incomplete, _assessment(passed=False, failed=["topology"]))


def test_delivery_metrics_must_share_exact_action_context() -> None:
    candidate = _metrics("a")
    fallback = _metrics("a")

    assert require_matched_delivery_context(candidate, fallback)["action:source"] == "a" * 64

    fallback["action_evidence"]["matched_context_sha256"]["motion_reference"] = "f" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="different delivery context"):
        require_matched_delivery_context(candidate, fallback)
