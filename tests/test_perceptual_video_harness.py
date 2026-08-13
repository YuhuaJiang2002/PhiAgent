from __future__ import annotations

from phiagent.agent.perceptual_video_harness import (
    PERCEPTUAL_DEMO_GATES,
    PerceptualCandidate,
    foundation_model_roles,
    select_display_candidate,
)


def _candidate(candidate_id: str, failed: str | None = None) -> PerceptualCandidate:
    return PerceptualCandidate(
        candidate_id=candidate_id,
        gates=tuple((gate, gate != failed) for gate in PERCEPTUAL_DEMO_GATES),
        utility=100.0 if failed else 1.0,
        wall_seconds=2.0,
        evidence_path=f"outputs/{candidate_id}/manifest.json",
    )


def test_mean_utility_cannot_override_one_failed_visual_gate() -> None:
    decision = select_display_candidate(
        [_candidate("high-score-blur", "intermittent_hand_smear_absent")]
    )
    assert decision["status"] == "PARTIAL"
    assert decision["selected_candidate"] is None


def test_display_ready_is_explicitly_not_physical_evidence() -> None:
    decision = select_display_candidate([_candidate("accepted")])
    assert decision["status"] == "DISPLAY_READY"
    assert decision["selected_candidate"] == "accepted"
    assert decision["physical_evidence"] is False
    assert "contact force" in decision["physical_claims_disallowed"]


def test_model_roles_include_generator_repair_critic_and_locked_state() -> None:
    roles = foundation_model_roles()
    names = {row["role"] for row in roles}
    assert "motion_and_character_replacement" in names
    assert "masked_local_video_repair" in names
    assert "semantic_failure_miner" in names
    assert "immutable_state_layers" in names
