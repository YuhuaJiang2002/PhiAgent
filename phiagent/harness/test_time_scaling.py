"""Hard-gate-aware test-time scaling for action-conditioned video proposals.

Scaling may spend more inference steps, seeds, or repair rounds.  It may never
relax a threshold or allow a mean score to override a failed physical gate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from phiagent.acwm.adapters import ACWMVideoRenderer
from phiagent.acwm.schema import ACWMCase
from phiagent.agent.acwm import (
    ACWMCandidate,
    ACWMProposal,
    ACWMThresholds,
)
from phiagent.harness.task_reasoning import (
    TaskReasoningPlan,
    validate_task_reasoning_plan,
)


@dataclass(frozen=True)
class ScalingRound:
    candidate_count: int
    inference_steps: int
    seed_offset: int
    purpose: str

    def __post_init__(self) -> None:
        if self.candidate_count < 1:
            raise ValueError("scaling candidate_count must be positive")
        if self.inference_steps < 1:
            raise ValueError("scaling inference_steps must be positive")
        if self.seed_offset < 0:
            raise ValueError("scaling seed_offset must be non-negative")
        if not self.purpose.strip():
            raise ValueError("scaling round purpose is required")


@dataclass(frozen=True)
class TestTimeScalingPolicy:
    __test__ = False

    rounds: tuple[ScalingRound, ...]
    seed_stride: int = 1009
    maximum_candidates: int = 12
    threshold_policy: str = "frozen_fail_closed"

    def __post_init__(self) -> None:
        if not self.rounds:
            raise ValueError("test-time scaling requires at least one round")
        if self.seed_stride < 1:
            raise ValueError("test-time scaling seed_stride must be positive")
        if self.maximum_candidates < 1:
            raise ValueError("test-time scaling maximum_candidates must be positive")
        if sum(item.candidate_count for item in self.rounds) > self.maximum_candidates:
            raise ValueError("test-time scaling rounds exceed maximum_candidates")
        if self.threshold_policy != "frozen_fail_closed":
            raise ValueError("test-time scaling cannot relax or aggregate hard gates")
        steps = tuple(item.inference_steps for item in self.rounds)
        if any(right < left for left, right in zip(steps, steps[1:])):
            raise ValueError("test-time scaling inference steps must be non-decreasing")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TestTimeScalingPolicy:
        raw_rounds = payload.get("rounds")
        if not isinstance(raw_rounds, list) or not raw_rounds:
            raise ValueError("test-time scaling policy requires a rounds array")
        rounds = tuple(
            ScalingRound(
                candidate_count=int(item["candidate_count"]),
                inference_steps=int(item["inference_steps"]),
                seed_offset=int(item["seed_offset"]),
                purpose=str(item["purpose"]),
            )
            for item in raw_rounds
            if isinstance(item, dict)
        )
        if len(rounds) != len(raw_rounds):
            raise ValueError("test-time scaling round must be an object")
        return cls(
            rounds=rounds,
            seed_stride=int(payload.get("seed_stride", 1009)),
            maximum_candidates=int(payload.get("maximum_candidates", 12)),
            threshold_policy=str(payload.get("threshold_policy", "frozen_fail_closed")),
        )


def load_test_time_scaling_policy(path: Path) -> TestTimeScalingPolicy:
    payload = json.loads(path.expanduser().resolve().read_text())
    if not isinstance(payload, dict):
        raise ValueError("test-time scaling policy must contain one JSON object")
    return TestTimeScalingPolicy.from_dict(payload)


def compile_task_reasoning_prompt(plan: TaskReasoningPlan) -> str:
    """Compile the hash-bound plan into an H3-readable phase program."""

    validated = validate_task_reasoning_plan(plan)
    phases = "\n".join(
        (
            f"- {phase.start_seconds:.3f}-{phase.end_seconds:.3f}s "
            f"[{phase.phase_id}; {phase.speed_class}]: {phase.language_directive} "
            f"Hard gates: {', '.join(phase.gate_ids)}."
        )
        for phase in validated.phases
    )
    constraints = "\n".join(f"- {item}" for item in validated.global_constraints)
    gates = "\n".join(
        f"- {gate.gate_id}: {gate.description}"
        for gate in validated.verification_gates
    )
    return (
        "\n\n[Hash-bound physical/task language plan]\n"
        f"Plan SHA-256: {validated.plan_sha256}\n"
        f"Normalized task: {validated.language_analysis.normalized_instruction}\n"
        "Render one continuous shot. Do not use cuts, dissolves, crossfades, state swaps, "
        "or keyframe morphing. The following timeline is causal and mandatory:\n"
        f"{phases}\n"
        "Global invariants:\n"
        f"{constraints}\n"
        "Non-negotiable hard gates; any failed gate rejects the candidate:\n"
        f"{gates}\n"
        f"Claim boundary: {validated.claim_boundary}"
    )


def _supported_backend(
    case: ACWMCase,
    renderers: Mapping[str, ACWMVideoRenderer],
    preferred: Sequence[str],
) -> str:
    for name in preferred:
        renderer = renderers.get(name)
        if renderer is not None and renderer.supports(case).supported:
            return name
    raise ValueError(f"no preferred backend supports {case.case_id}")


def initial_scaled_proposals(
    cases: Sequence[ACWMCase],
    renderers: Mapping[str, ACWMVideoRenderer],
    *,
    policy: TestTimeScalingPolicy,
    base_seed: int,
    backend_preference: Sequence[str] = ("minimax-h3", "oscar", "bwm", "kinema4d"),
) -> tuple[ACWMProposal, ...]:
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    first = policy.rounds[0]
    proposals: list[ACWMProposal] = []
    for case in cases:
        backend = _supported_backend(case, renderers, backend_preference)
        for index in range(first.candidate_count):
            proposals.append(
                ACWMProposal(
                    case_id=case.case_id,
                    backend=backend,
                    seed=base_seed + first.seed_offset + index * policy.seed_stride,
                    num_inference_steps=first.inference_steps,
                    guidance_scale=1.0 if backend == "minimax-h3" else 6.0,
                )
            )
    return tuple(proposals)


_HARD_GATE_REPAIRS = {
    "viewer_left_sleeve_length_conserved": (
        "Preserve the viewer-left cuff-to-shoulder material polyline and every segment length; "
        "fold by continuous rotation and bending, never by shrinking or morphing."
    ),
    "viewer_right_sleeve_length_conserved": (
        "Preserve the viewer-right cuff-to-shoulder material polyline and every segment length; "
        "fold by continuous rotation and bending, never by shrinking or morphing."
    ),
    "viewer_left_fold_precedes_viewer_right_fold": (
        "Finish and visibly settle the viewer-left sleeve before initiating any viewer-right sleeve motion."
    ),
    "contact_precedes_cloth_motion": (
        "Show stabilizing shoulder contact and cuff-side grasp before the contacted cloth begins moving."
    ),
    "no_teleportation_or_crossfade": (
        "Use continuous material trajectories only; prohibit hard cuts, dissolves, crossfades, and single-frame jumps."
    ),
    "body_fold_after_both_sleeves": (
        "Keep the shirt body fixed until both sleeves are folded and settled."
    ),
    "bundle_move_after_body_fold": (
        "Move the bundle aside only after one compact layered rectangle is complete."
    ),
    "camera_and_background_static": (
        "Lock the input camera, table, glass, cables, surrounding garments, lighting, and background."
    ),
}


def _failed_hard_gate_ids(history: Sequence[ACWMCandidate], case_id: str) -> tuple[str, ...]:
    failures: set[str] = set()
    for candidate in history:
        if candidate.proposal.case_id != case_id:
            continue
        for diagnosis in candidate.scorecard.diagnoses:
            prefix = "hard_gate:"
            if diagnosis.startswith(prefix):
                failures.add(diagnosis.removeprefix(prefix))
    return tuple(sorted(failures))


class HardGateTestTimeScalingRepairAgent:
    """Spend more compute on diagnosed failures without changing acceptance gates."""

    def __init__(
        self,
        policy: TestTimeScalingPolicy,
        *,
        base_seed: int,
        backend_preference: Sequence[str] = (
            "minimax-h3",
            "oscar",
            "bwm",
            "kinema4d",
        ),
    ) -> None:
        if base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        self.policy = policy
        self.base_seed = base_seed
        self.backend_preference = tuple(backend_preference)

    def propose(
        self,
        *,
        cases: Mapping[str, ACWMCase],
        renderers: Mapping[str, ACWMVideoRenderer],
        history: tuple[ACWMCandidate, ...],
        thresholds: ACWMThresholds,
    ) -> tuple[ACWMProposal, ...]:
        # Thresholds are read-only inputs from the controller; this agent never mutates them.
        next_round_index = max(item.round_index for item in history) + 1
        if next_round_index >= len(self.policy.rounds):
            return ()
        scaling_round = self.policy.rounds[next_round_index]
        proposals: list[ACWMProposal] = []
        for case_id, case in cases.items():
            case_history = tuple(
                item for item in history if item.proposal.case_id == case_id
            )
            if any(
                diagnosis.startswith("evaluator_error:")
                for item in case_history
                for diagnosis in item.scorecard.diagnoses
            ):
                continue
            if any(item.scorecard.automatic_gates_pass(thresholds) for item in case_history):
                continue
            backend = _supported_backend(case, renderers, self.backend_preference)
            hard_gate_ids = _failed_hard_gate_ids(case_history, case_id)
            repair_text = " ".join(
                _HARD_GATE_REPAIRS.get(
                    gate_id,
                    f"Correct hard gate {gate_id} without weakening any other gate.",
                )
                for gate_id in hard_gate_ids
            )
            if not repair_text:
                repair_text = (
                    "Increase temporal and material consistency while preserving every frozen hard gate."
                )
            for index in range(scaling_round.candidate_count):
                proposals.append(
                    ACWMProposal(
                        case_id=case_id,
                        backend=backend,
                        seed=(
                            self.base_seed
                            + scaling_round.seed_offset
                            + index * self.policy.seed_stride
                        ),
                        num_inference_steps=scaling_round.inference_steps,
                        guidance_scale=1.0 if backend == "minimax-h3" else 6.0,
                        prompt_suffix=repair_text,
                    )
                )
        return tuple(proposals)
