from __future__ import annotations

import pytest

from phiagent.agent.flower_task_adaptation import (
    PHASE_CANDIDATE_GATES,
    BimanualPhase,
    ContactConstraint,
    EvidenceKind,
    FlowerInstanceSpec,
    FlowerTaskContract,
    HandPhase,
    HandSide,
    OcclusionOrder,
    PhaseCandidateEvaluation,
    select_immutable_phase_candidates,
)


def _contract(*, stable: bool = False, adapter: str | None = None) -> FlowerTaskContract:
    evidence_kind = EvidenceKind.INSTANCE_MASK if stable else EvidenceKind.UNION_MASK_PROXY
    evidence = ("instance-track.npz",) if stable else ("union-track.npz",)
    return FlowerTaskContract(
        frame_count=10,
        fps=24.0,
        adapter_checkpoint_sha256=adapter,
        instances=(
            FlowerInstanceSpec("bouquet", 10, 10, evidence_kind, stable, evidence=evidence),
        ),
        contacts=(
            ContactConstraint(
                "left-hold",
                HandSide.LEFT,
                "bouquet",
                2,
                8,
                HandPhase.HOLD,
                evidence_kind,
                1.0 if stable else 0.7,
                (
                    OcclusionOrder.FLOWER_BEHIND_HAND
                    if stable
                    else OcclusionOrder.DEPTH_TRACK_REQUIRED
                ),
                evidence=evidence,
            ),
        ),
        phases=(
            BimanualPhase("approach", 0, 2, HandPhase.APPROACH, HandPhase.OBSERVE),
            BimanualPhase(
                "grasp", 2, 3, HandPhase.GRASP, HandPhase.OBSERVE, "bouquet"
            ),
            BimanualPhase(
                "manipulate", 3, 7, HandPhase.MANIPULATE, HandPhase.OBSERVE, "bouquet"
            ),
            BimanualPhase(
                "release", 7, 8, HandPhase.RELEASE, HandPhase.OBSERVE, "bouquet"
            ),
            BimanualPhase("retract", 8, 10, HandPhase.RETRACT, HandPhase.OBSERVE),
        ),
    )


def test_union_mask_contact_is_an_explicit_claim_blocker() -> None:
    contract = _contract()

    assert not contract.claim_ready
    assert "flower_identity_unverified:bouquet" in contract.claim_blockers
    assert "contact_is_proxy:left-hold" in contract.claim_blockers
    assert "occlusion_depth_unverified:left-hold" in contract.claim_blockers


def test_instance_contact_and_hashed_adapter_can_be_claim_ready() -> None:
    contract = _contract(stable=True, adapter="a" * 64)

    assert contract.claim_ready
    assert contract.claim_blockers == ()


def test_phases_must_cover_every_frame_without_gaps() -> None:
    contract = _contract()
    phases = list(contract.phases)
    phases[1] = BimanualPhase(
        "grasp", 3, 4, HandPhase.GRASP, HandPhase.OBSERVE, "bouquet"
    )

    with pytest.raises(ValueError, match="contiguously"):
        FlowerTaskContract(
            frame_count=10,
            fps=24.0,
            instances=contract.instances,
            contacts=contract.contacts,
            phases=tuple(phases),
        )


def test_contact_phase_requires_named_flower_instance() -> None:
    with pytest.raises(ValueError, match="flower_id"):
        BimanualPhase("bad", 0, 2, HandPhase.HOLD, HandPhase.OBSERVE)


def _candidate(phase: str, candidate: str, value: float) -> PhaseCandidateEvaluation:
    return PhaseCandidateEvaluation(
        phase,
        candidate,
        {name: value for name in PHASE_CANDIDATE_GATES},
        (f"evaluation:{candidate}",),
    )


def test_phase_selection_keeps_one_complete_candidate_and_rejects_missing_gate() -> None:
    thresholds = {name: 0.8 for name in PHASE_CANDIDATE_GATES}
    evaluations = (
        _candidate("grasp", "seed-1", 0.82),
        _candidate("grasp", "seed-2", 0.91),
        _candidate("release", "seed-3", 0.79),
    )

    assert select_immutable_phase_candidates(evaluations[:2], ("grasp",), thresholds) == {
        "grasp": "seed-2"
    }
    with pytest.raises(ValueError, match="release"):
        select_immutable_phase_candidates(evaluations, ("grasp", "release"), thresholds)
