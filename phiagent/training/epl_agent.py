"""Deterministic EPL-conditioned repair-policy training examples."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import IntEnum

from phiagent.physical_language.schema import ContactState, ManipulationPhase

PHASES = tuple(ManipulationPhase)
CONTACT_STATES = tuple(ContactState)


class RepairAction(IntEnum):
    CLAMP_LIMITS = 0
    SMOOTH_TRAJECTORY = 1
    RETIME_TRAJECTORY = 2
    SHIFT_ALIGNMENT = 3
    CONTACT_SAFE_REPLAN = 4
    ACCEPT = 5


@dataclass(frozen=True)
class EPLPolicyExample:
    phase: ManipulationPhase
    contact_state: ContactState
    confidence: float
    hand_aperture_m: float
    diagnostics: tuple[float, ...]
    action: RepairAction

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("example confidence must be finite and in [0, 1]")
        if not math.isfinite(self.hand_aperture_m) or self.hand_aperture_m < 0:
            raise ValueError("example hand aperture must be finite and non-negative")
        if len(self.diagnostics) != 8 or not all(
            math.isfinite(value) for value in self.diagnostics
        ):
            raise ValueError("example diagnostics must contain eight finite values")


def _jitter(generator: random.Random, value: float, scale: float = 0.025) -> float:
    return max(0.0, value + generator.gauss(0.0, scale))


def generate_policy_examples(count: int, seed: int) -> tuple[EPLPolicyExample, ...]:
    """Generate balanced diagnostic cases with EPL-dependent ambiguous repairs."""

    if count < len(RepairAction):
        raise ValueError("policy dataset must contain at least one example per action family")
    generator = random.Random(seed)
    examples: list[EPLPolicyExample] = []
    for index in range(count):
        family = index % len(RepairAction)
        phase = generator.choice(PHASES)
        contact = generator.choice(CONTACT_STATES)
        confidence = generator.uniform(0.55, 1.0)
        aperture = generator.uniform(0.005, 0.12)
        diagnostics = [generator.uniform(0.0, 0.035) for _ in range(8)]

        if family == 0:
            diagnostics[0] = _jitter(generator, 1.0)
            diagnostics[6] = _jitter(generator, 0.65)
            action = RepairAction.CLAMP_LIMITS
        elif family == 1:
            phase = generator.choice(
                (
                    ManipulationPhase.IDLE,
                    ManipulationPhase.APPROACH,
                    ManipulationPhase.PREGRASP,
                    ManipulationPhase.RELEASE,
                    ManipulationPhase.RETRACT,
                )
            )
            contact = generator.choice(
                (ContactState.NONE, ContactState.CANDIDATE, ContactState.UNKNOWN)
            )
            diagnostics[1] = _jitter(generator, 0.85)
            diagnostics[4] = _jitter(generator, 0.25)
            action = RepairAction.SMOOTH_TRAJECTORY
        elif family == 2:
            diagnostics[2] = _jitter(generator, 0.9)
            action = RepairAction.RETIME_TRAJECTORY
        elif family == 3:
            phase = generator.choice(
                (
                    ManipulationPhase.IDLE,
                    ManipulationPhase.APPROACH,
                    ManipulationPhase.PREGRASP,
                    ManipulationPhase.RETRACT,
                )
            )
            contact = generator.choice(
                (ContactState.NONE, ContactState.CANDIDATE, ContactState.UNKNOWN)
            )
            diagnostics[3] = _jitter(generator, 0.9)
            action = RepairAction.SHIFT_ALIGNMENT
        elif family == 4:
            ambiguous_kind = (index // len(RepairAction)) % 3
            if ambiguous_kind == 0:
                phase = generator.choice(
                    (ManipulationPhase.GRASP, ManipulationPhase.MANIPULATE)
                )
                contact = generator.choice(
                    (ContactState.STABLE, ContactState.SLIPPING)
                )
                diagnostics[1] = _jitter(generator, 0.85)
                diagnostics[4] = _jitter(generator, 0.25)
            elif ambiguous_kind == 1:
                phase = generator.choice(
                    (
                        ManipulationPhase.GRASP,
                        ManipulationPhase.MANIPULATE,
                        ManipulationPhase.RELEASE,
                    )
                )
                contact = ContactState.STABLE
                diagnostics[3] = _jitter(generator, 0.9)
            else:
                diagnostics[4] = _jitter(generator, 0.95)
                diagnostics[7] = _jitter(generator, 0.7)
            action = RepairAction.CONTACT_SAFE_REPLAN
        else:
            diagnostics[5] = _jitter(generator, 0.02)
            action = RepairAction.ACCEPT

        examples.append(
            EPLPolicyExample(
                phase=phase,
                contact_state=contact,
                confidence=confidence,
                hand_aperture_m=aperture,
                diagnostics=tuple(diagnostics),
                action=action,
            )
        )
    generator.shuffle(examples)
    return tuple(examples)


def encode_example(example: EPLPolicyExample, include_epl: bool = True) -> tuple[float, ...]:
    phase_features = tuple(float(example.phase is phase) for phase in PHASES)
    contact_features = tuple(
        float(example.contact_state is state) for state in CONTACT_STATES
    )
    epl_features = (
        *phase_features,
        *contact_features,
        example.confidence,
        min(example.hand_aperture_m / 0.2, 1.0),
    )
    if not include_epl:
        epl_features = (0.0,) * len(epl_features)
    return (*epl_features, *example.diagnostics)


def feature_names() -> tuple[str, ...]:
    return (
        *(f"phase:{phase.value}" for phase in PHASES),
        *(f"contact:{state.value}" for state in CONTACT_STATES),
        "confidence",
        "hand_aperture_normalized",
        "joint_limit_violation",
        "position_noise",
        "temporal_scale_error",
        "timing_shift",
        "contact_loss",
        "collision",
        "reachability_failure",
        "slip",
    )
