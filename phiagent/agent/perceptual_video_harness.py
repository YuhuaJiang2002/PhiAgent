"""Fail-closed promotion for perceptually plausible long-video demos.

This contract is intentionally separate from the physical-contact contract.  A
candidate may be suitable for a clearly labelled synthetic demo without being
metric camera, robot telemetry, depth, or force evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


PERCEPTUAL_DEMO_GATES = (
    "duration_at_least_20_seconds",
    "full_video_decodes",
    "native_background_locked",
    "flower_pixels_locked",
    "flower_response_not_frozen",
    "human_residue_absent",
    "canonical_hand_topology_locked",
    "intermittent_hand_smear_absent",
    "long_term_robot_identity_stable",
    "adversarial_attacks_detected",
    "high_resolution_review_pass",
)


@dataclass(frozen=True)
class PerceptualCandidate:
    candidate_id: str
    gates: tuple[tuple[str, bool], ...]
    utility: float
    wall_seconds: float
    evidence_path: str

    def validate(self) -> None:
        if not self.candidate_id.strip() or not self.evidence_path.strip():
            raise ValueError("candidate ID and evidence path must be non-empty")
        gate_map = dict(self.gates)
        if len(gate_map) != len(self.gates):
            raise ValueError("candidate gate names must be unique")
        if set(gate_map) != set(PERCEPTUAL_DEMO_GATES):
            missing = sorted(set(PERCEPTUAL_DEMO_GATES) - set(gate_map))
            extra = sorted(set(gate_map) - set(PERCEPTUAL_DEMO_GATES))
            raise ValueError(f"gate contract mismatch: missing={missing}, extra={extra}")
        if self.wall_seconds < 0:
            raise ValueError("wall time must be non-negative")

    @property
    def passed(self) -> bool:
        return all(value for _, value in self.gates)


def select_display_candidate(
    candidates: Iterable[PerceptualCandidate],
) -> dict[str, object]:
    """Select only a candidate that passes every visible-quality gate.

    Utility is allowed to rank candidates only after every hard gate passes.  A
    high average score can therefore never hide one malformed hand frame or a
    failed high-resolution veto.
    """

    rows = list(candidates)
    if not rows:
        raise ValueError("at least one perceptual candidate is required")
    seen = set()
    summaries = []
    eligible = []
    for row in rows:
        row.validate()
        if row.candidate_id in seen:
            raise ValueError(f"duplicate candidate ID: {row.candidate_id}")
        seen.add(row.candidate_id)
        failed = [name for name, passed in row.gates if not passed]
        summary = {
            "candidate_id": row.candidate_id,
            "all_visual_gates_pass": not failed,
            "failed_gates": failed,
            "utility": row.utility,
            "wall_seconds": row.wall_seconds,
            "evidence_path": row.evidence_path,
        }
        summaries.append(summary)
        if not failed:
            eligible.append(summary)
    winner = (
        max(
            eligible,
            key=lambda item: (
                float(item["utility"]),
                -float(item["wall_seconds"]),
                str(item["candidate_id"]),
            ),
        )
        if eligible
        else None
    )
    return {
        "status": "DISPLAY_READY" if winner else "PARTIAL",
        "selected_candidate": winner["candidate_id"] if winner else None,
        "candidates": summaries,
        "claim_scope": "perceptually plausible synthetic video data",
        "physical_evidence": False,
        "physical_claims_disallowed": (
            "metric depth, calibrated camera, exact q/qdot, contact force, "
            "force closure, and real-robot executability"
        ),
    }


def foundation_model_roles() -> tuple[dict[str, str], ...]:
    """Return the architecture roles used by the long-video harness."""

    return (
        {
            "role": "motion_and_character_replacement",
            "model": "Wan2.2-Animate-14B",
            "use": "generate the robot appearance and inherit source body motion",
        },
        {
            "role": "masked_local_video_repair",
            "model": "Wan2.1-VACE-1.3B",
            "use": "propose bounded failed-window repairs; never overwrite locked objects",
        },
        {
            "role": "source_anchored_causal_window_editor",
            "model": "JoyAI-Video-Edit-16B",
            "use": (
                "edit measured 1+8n-frame seam/contact windows with first-chunk "
                "sink and recent causal KV; proposal only, followed by immutable-state projection"
            ),
        },
        {
            "role": "long_horizon_challenger",
            "model": "LongCat-Video / SkyReels-V2 / MAGI-1",
            "use": "future continuation challenger, admitted only after task-local gates pass",
        },
        {
            "role": "semantic_failure_miner",
            "model": "Qwen3-VL-4B and 8B",
            "use": "propose high-resolution failure windows; never act as physical truth",
        },
        {
            "role": "immutable_state_layers",
            "model": "deterministic source and canonical-topology compositor",
            "use": "lock real flowers, background, response motion, and hand topology",
        },
    )
