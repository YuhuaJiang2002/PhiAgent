"""Strategy-conditioned physical and language planning for T-shirt folding.

The base T-shirt planner intentionally retains its historical left-then-right,
place-left behavior.  This plugin expands the same typed request into a bounded
set of independently hash-bound alternatives so generation and evaluation can
never silently exchange one action order for another.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .task_reasoning import (
    SCHEMA_VERSION,
    TSHIRT_FOLD_TASK,
    LanguageAnalysis,
    ReasoningFinding,
    ReasoningPluginDescriptor,
    TaskPhase,
    TaskReasoningPlan,
    TaskReasoningRequest,
    VerificationGate,
    _canonical_sha256,
    _detect_language,
)


LEFT_THEN_RIGHT = "left_then_right"
RIGHT_THEN_LEFT = "right_then_left"
SIMULTANEOUS = "simultaneous"
VIEWER_LEFT = "viewer_left"
VIEWER_RIGHT = "viewer_right"

_SLEEVE_ORDERS = {LEFT_THEN_RIGHT, RIGHT_THEN_LEFT, SIMULTANEOUS}
_BUNDLE_PLACEMENTS = {VIEWER_LEFT, VIEWER_RIGHT}
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class TshirtFoldStrategy:
    """One explicit sleeve-order and terminal-placement choice."""

    sleeve_order: str
    bundle_placement: str

    def __post_init__(self) -> None:
        if self.sleeve_order not in _SLEEVE_ORDERS:
            raise ValueError(f"unsupported T-shirt sleeve order: {self.sleeve_order}")
        if self.bundle_placement not in _BUNDLE_PLACEMENTS:
            raise ValueError(
                f"unsupported T-shirt bundle placement: {self.bundle_placement}"
            )

    @property
    def strategy_id(self) -> str:
        order = {
            LEFT_THEN_RIGHT: "left-then-right",
            RIGHT_THEN_LEFT: "right-then-left",
            SIMULTANEOUS: "simultaneous",
        }[self.sleeve_order]
        side = "place-left" if self.bundle_placement == VIEWER_LEFT else "place-right"
        return f"{order}-{side}"

    @property
    def ordered_sleeves(self) -> tuple[str, str] | None:
        if self.sleeve_order == LEFT_THEN_RIGHT:
            return VIEWER_LEFT, VIEWER_RIGHT
        if self.sleeve_order == RIGHT_THEN_LEFT:
            return VIEWER_RIGHT, VIEWER_LEFT
        return None

    def to_dict(self) -> dict[str, str]:
        return {
            "strategy_id": self.strategy_id,
            "sleeve_order": self.sleeve_order,
            "bundle_placement": self.bundle_placement,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TshirtFoldStrategy":
        strategy = cls(
            sleeve_order=str(payload["sleeve_order"]),
            bundle_placement=str(payload["bundle_placement"]),
        )
        declared_id = payload.get("strategy_id")
        if declared_id is not None and declared_id != strategy.strategy_id:
            raise ValueError("T-shirt strategy_id does not match its choices")
        return strategy


def all_tshirt_fold_strategies() -> tuple[TshirtFoldStrategy, ...]:
    """Return the frozen six-way strategy matrix in a stable order."""

    return tuple(
        TshirtFoldStrategy(order, placement)
        for order in (LEFT_THEN_RIGHT, RIGHT_THEN_LEFT, SIMULTANEOUS)
        for placement in (VIEWER_LEFT, VIEWER_RIGHT)
    )


def _variant_task_id(task_id: str, strategy: TshirtFoldStrategy) -> str:
    suffix = f"--{strategy.strategy_id}"
    prefix = task_id[: 128 - len(suffix)].rstrip(".-_")
    candidate = f"{prefix}{suffix}"
    if not _SAFE_ID.fullmatch(candidate):
        raise ValueError("strategy-expanded T-shirt task id is not filesystem-safe")
    return candidate


def _boundaries(duration_seconds: float, weights: Sequence[float]) -> tuple[float, ...]:
    if not weights or any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("T-shirt phase weights must be finite and positive")
    total = sum(weights)
    elapsed = 0.0
    values = [0.0]
    for weight in weights[:-1]:
        elapsed += weight
        values.append(round(duration_seconds * elapsed / total, 6))
    values.append(duration_seconds)
    return tuple(values)


def _side_text(side: str) -> str:
    return "viewer-left" if side == VIEWER_LEFT else "viewer-right"


def _order_gate(strategy: TshirtFoldStrategy) -> VerificationGate:
    if strategy.sleeve_order == SIMULTANEOUS:
        return VerificationGate(
            "both_sleeves_fold_synchronously",
            (
                "Both sleeve material tracks start within the frozen synchronization "
                "tolerance, remain bilaterally contact-driven, and settle before body motion."
            ),
            "automatic_proxy",
        )
    first, second = strategy.ordered_sleeves or ()
    first_text = _side_text(first)
    second_text = _side_text(second)
    return VerificationGate(
        f"{first}_fold_precedes_{second}_fold",
        (
            f"The {first_text} sleeve completes and settles before the "
            f"{second_text} sleeve starts folding."
        ),
        "automatic_proxy",
    )


def _gates(strategy: TshirtFoldStrategy) -> tuple[VerificationGate, ...]:
    destination = _side_text(strategy.bundle_placement)
    return (
        VerificationGate(
            "exact_first_frame",
            "Decoded frame zero preserves the supplied first-frame pixels within tolerance.",
            "automatic_proxy",
        ),
        VerificationGate(
            "single_shirt_identity",
            "Exactly one original T-shirt persists without duplication or material substitution.",
            "native_resolution_human_review",
        ),
        VerificationGate(
            "viewer_left_sleeve_length_conserved",
            (
                "The viewer-left cuff-to-shoulder material polyline stays within all "
                "frozen total-length and segment-deformation bounds."
            ),
            "automatic_proxy",
        ),
        VerificationGate(
            "viewer_right_sleeve_length_conserved",
            (
                "The viewer-right cuff-to-shoulder material polyline stays within all "
                "frozen total-length and segment-deformation bounds."
            ),
            "automatic_proxy",
        ),
        VerificationGate(
            "viewer_left_sleeve_folds_inward",
            (
                "The viewer-left sleeve moves the frozen minimum distance toward the "
                "original torso center during its assigned fold window."
            ),
            "automatic_proxy",
        ),
        VerificationGate(
            "viewer_right_sleeve_folds_inward",
            (
                "The viewer-right sleeve moves the frozen minimum distance toward the "
                "original torso center during its assigned fold window."
            ),
            "automatic_proxy",
        ),
        VerificationGate(
            "cuff_and_shoulder_identity_persistent",
            "Both cuffs and both shoulder seams retain material identity through occlusion.",
            "native_resolution_human_review",
        ),
        VerificationGate(
            "contact_precedes_cloth_motion",
            "Every moving sleeve has visible gripper contact before its material track moves.",
            "native_resolution_human_review",
        ),
        _order_gate(strategy),
        VerificationGate(
            "no_teleportation_or_crossfade",
            (
                "Every tracked cloth and gripper point obeys the frozen per-point "
                "single-frame displacement bound, with no cut, dissolve, or crossfade."
            ),
            "automatic_proxy",
        ),
        VerificationGate(
            "body_fold_after_both_sleeves",
            "The lower body starts folding only after both sleeves are folded and settled.",
            "automatic_proxy",
        ),
        VerificationGate(
            "bundle_move_after_body_fold",
            (
                f"The completed compact bundle moves toward {destination} only after "
                "the body fold is complete."
            ),
            "automatic_proxy",
        ),
        VerificationGate(
            "bundle_moves_as_one_material",
            (
                f"Both sleeves and the torso move coherently toward {destination}, "
                "with component disagreement below the frozen bound."
            ),
            "automatic_proxy",
        ),
        VerificationGate(
            "camera_and_background_static",
            "The camera, table, glass, cables, surrounding garments, and lighting remain fixed.",
            "automatic_proxy",
        ),
        VerificationGate(
            "terminal_compact_bundle_stable",
            (
                f"One compact bundle remains at {destination} through the terminal hold "
                "without rebound, unfolding, or drift."
            ),
            "automatic_proxy",
        ),
    )


def _phase(
    phase_id: str,
    start: float,
    end: float,
    objective: str,
    directive: str,
    frame: str,
    speed: str,
    preconditions: tuple[str, ...],
    invariants: tuple[str, ...],
    postconditions: tuple[str, ...],
    gate_ids: tuple[str, ...],
) -> TaskPhase:
    return TaskPhase(
        phase_id=phase_id,
        start_seconds=start,
        end_seconds=end,
        objective=objective,
        language_directive=directive,
        motion_frame=frame,
        speed_class=speed,
        preconditions=preconditions,
        invariants=invariants,
        postconditions=postconditions,
        gate_ids=gate_ids,
    )


def _sequential_phases(
    request: TaskReasoningRequest,
    strategy: TshirtFoldStrategy,
) -> tuple[TaskPhase, ...]:
    first, second = strategy.ordered_sleeves or ()
    first_text = _side_text(first)
    second_text = _side_text(second)
    destination = _side_text(strategy.bundle_placement)
    order_gate = f"{first}_fold_precedes_{second}_fold"
    b = _boundaries(
        request.duration_seconds,
        (0.05, 0.10, 0.17, 0.06, 0.10, 0.16, 0.16, 0.10, 0.07, 0.03),
    )
    frame = request.coordinate_frame
    return (
        _phase(
            "initial_state_hold",
            b[0],
            b[1],
            "Bind the exact source scene and material identities before motion.",
            "Hold the exact first frame; neither robot nor any cloth material moves.",
            frame,
            "stationary",
            ("The supplied first frame is the sole initial-state authority.",),
            ("No camera, background, robot, or cloth motion.",),
            ("Both sleeves, cuffs, shoulder seams, body, and manipulators are bound.",),
            ("exact_first_frame", "single_shirt_identity", "camera_and_background_static"),
        ),
        _phase(
            f"establish_{first}_two_point_contact",
            b[1],
            b[2],
            f"Establish causal support and cuff contact on the {first_text} sleeve.",
            (
                f"Approach the {first_text} sleeve without moving it. One gripper stabilizes "
                "its shoulder-body junction while the other visibly grasps near its cuff."
            ),
            frame,
            "fine",
            (f"The {first_text} sleeve is flat and motionless.",),
            (f"No {first_text} cloth motion before contact; the {second_text} sleeve stays still.",),
            ("A supported shoulder region and cuff-side grasp are visible.",),
            ("contact_precedes_cloth_motion", "cuff_and_shoulder_identity_persistent"),
        ),
        _phase(
            f"fold_{first}_sleeve",
            b[2],
            b[3],
            f"Fold the {first_text} sleeve inward without changing material length.",
            (
                f"Guide the complete {first_text} sleeve inward through one continuous "
                "shoulder-seam fold arc. Preserve cuff-to-shoulder arclength and every "
                "material segment; never shrink, stretch, dissolve, or teleport cloth."
            ),
            frame,
            "slow",
            (f"Supported contact on the {first_text} sleeve is established.",),
            (f"The {second_text} sleeve and shirt body remain fixed.",),
            (f"The {first_text} sleeve lies inward with unchanged length and identity.",),
            (
                f"{first}_sleeve_length_conserved",
                f"{first}_sleeve_folds_inward",
                "contact_precedes_cloth_motion",
                "no_teleportation_or_crossfade",
            ),
        ),
        _phase(
            f"settle_{first}_sleeve",
            b[3],
            b[4],
            "Settle the first sleeve before manipulating the second.",
            (
                f"Hold the folded {first_text} sleeve motionless while the "
                f"{second_text} sleeve remains completely unchanged."
            ),
            frame,
            "stationary",
            (f"The {first_text} sleeve reached its inward terminal pose.",),
            ("No rebound, unfolding, or second-sleeve motion.",),
            ("The first fold is visibly settled.",),
            (f"{first}_sleeve_length_conserved", order_gate),
        ),
        _phase(
            f"establish_{second}_two_point_contact",
            b[4],
            b[5],
            f"Establish causal support and cuff contact on the {second_text} sleeve.",
            (
                f"Keep the {first_text} fold fixed. Stabilize the {second_text} "
                "shoulder-body junction and visibly grasp near its cuff before motion."
            ),
            frame,
            "fine",
            (f"The {first_text} fold is settled and the {second_text} sleeve is flat.",),
            (f"No {second_text} cloth motion before visible contact.",),
            ("A supported shoulder region and cuff-side grasp are visible.",),
            ("contact_precedes_cloth_motion", order_gate),
        ),
        _phase(
            f"fold_{second}_sleeve",
            b[5],
            b[6],
            f"Fold the {second_text} sleeve inward without changing material length.",
            (
                f"Guide the complete {second_text} sleeve inward through one continuous "
                f"shoulder-seam fold arc while the {first_text} fold stays fixed. Preserve "
                "both sleeves' cuff-to-shoulder material segments."
            ),
            frame,
            "slow",
            (f"Supported contact on the {second_text} sleeve follows the settled first fold.",),
            ("The first fold stays fixed; no cut, crossfade, shrinkage, or regrowth.",),
            ("Both sleeves lie inward, retain length, and are visibly settled.",),
            (
                f"{second}_sleeve_length_conserved",
                f"{second}_sleeve_folds_inward",
                f"{first}_sleeve_length_conserved",
                "cuff_and_shoulder_identity_persistent",
                "no_teleportation_or_crossfade",
                order_gate,
            ),
        ),
        *_common_terminal_phases(
            frame=frame,
            boundaries=b,
            offset=6,
            destination=destination,
            strategy=strategy,
        ),
    )


def _simultaneous_phases(
    request: TaskReasoningRequest,
    strategy: TshirtFoldStrategy,
) -> tuple[TaskPhase, ...]:
    destination = _side_text(strategy.bundle_placement)
    b = _boundaries(
        request.duration_seconds,
        (0.05, 0.14, 0.27, 0.07, 0.17, 0.11, 0.13, 0.06),
    )
    frame = request.coordinate_frame
    return (
        _phase(
            "initial_state_hold",
            b[0],
            b[1],
            "Bind the exact source scene and material identities before motion.",
            "Hold the exact first frame; neither robot nor any cloth material moves.",
            frame,
            "stationary",
            ("The supplied first frame is the sole initial-state authority.",),
            ("No camera, background, robot, or cloth motion.",),
            ("Both sleeves, cuffs, shoulder seams, body, and manipulators are bound.",),
            ("exact_first_frame", "single_shirt_identity", "camera_and_background_static"),
        ),
        _phase(
            "establish_bilateral_cuff_contacts",
            b[1],
            b[2],
            "Establish one causal cuff-side grasp on each sleeve.",
            (
                "The lower-left and upper-right grippers approach opposite cuffs without "
                "moving cloth, then close visibly and hold. The torso and both shoulder "
                "seams remain table-supported; do not invent extra hands or unsupported cloth."
            ),
            frame,
            "fine",
            ("Both sleeves and the table-supported torso are flat and motionless.",),
            ("Neither sleeve moves before both cuff contacts are established.",),
            ("Each original cuff has one visible grasp while both shoulder seams stay attached.",),
            ("contact_precedes_cloth_motion", "cuff_and_shoulder_identity_persistent"),
        ),
        _phase(
            "fold_both_sleeves_synchronously",
            b[2],
            b[3],
            "Fold both sleeves inward together without changing either material length.",
            (
                "Both grippers move inward as a synchronized pair, guiding each complete "
                "sleeve through a continuous shoulder-seam fold arc while the torso remains "
                "table-supported. Preserve both cuff-to-shoulder arclengths and segment "
                "identities; never shrink, stretch, merge, dissolve, or teleport cloth."
            ),
            frame,
            "slow",
            ("Both cuff contacts are established and the torso is table-supported.",),
            ("Bilateral onset remains synchronized; the gray body stays fixed.",),
            ("Both sleeves reach separate inward poses with unchanged lengths.",),
            (
                "viewer_left_sleeve_length_conserved",
                "viewer_right_sleeve_length_conserved",
                "viewer_left_sleeve_folds_inward",
                "viewer_right_sleeve_folds_inward",
                "contact_precedes_cloth_motion",
                "both_sleeves_fold_synchronously",
                "no_teleportation_or_crossfade",
            ),
        ),
        _phase(
            "settle_both_sleeves",
            b[3],
            b[4],
            "Settle both synchronous folds before body motion.",
            "Hold both folded sleeves separately and motionless; release only after they settle.",
            frame,
            "stationary",
            ("Both sleeves reached their inward terminal poses together.",),
            ("No rebound, unfolding, sleeve merger, or premature body motion.",),
            ("Both sleeve folds are stable before the lower hem approach.",),
            (
                "viewer_left_sleeve_length_conserved",
                "viewer_right_sleeve_length_conserved",
                "both_sleeves_fold_synchronously",
            ),
        ),
        *_common_terminal_phases(
            frame=frame,
            boundaries=b,
            offset=4,
            destination=destination,
            strategy=strategy,
        ),
    )


def _common_terminal_phases(
    *,
    frame: str,
    boundaries: Sequence[float],
    offset: int,
    destination: str,
    strategy: TshirtFoldStrategy,
) -> tuple[TaskPhase, ...]:
    return (
        _phase(
            "fold_body_bottom_to_top",
            boundaries[offset],
            boundaries[offset + 1],
            "Fold the lower torso upward only after both sleeves settle.",
            (
                "After both sleeves are stable, establish contact on the lower hem and "
                "guide the complete gray body upward through one continuous horizontal fold."
            ),
            frame,
            "slow",
            ("Both sleeve folds are complete and settled.",),
            ("No sleeve unfolding, cloth duplication, teleportation, or camera motion.",),
            ("One compact layered rectangle exists at the center.",),
            (
                "body_fold_after_both_sleeves",
                "single_shirt_identity",
                "no_teleportation_or_crossfade",
            ),
        ),
        _phase(
            "compress_bundle_without_stretch",
            boundaries[offset + 1],
            boundaries[offset + 2],
            "Settle the layered rectangle without changing cloth material.",
            "Use gentle contact to square the bundle; do not stretch, shorten, erase, or add cloth.",
            frame,
            "fine",
            ("One layered rectangular bundle exists at the center.",),
            ("Both sleeves retain their original material length inside the layers.",),
            ("The compact bundle is stable and ready for transport.",),
            (
                "viewer_left_sleeve_length_conserved",
                "viewer_right_sleeve_length_conserved",
                "terminal_compact_bundle_stable",
            ),
        ),
        _phase(
            f"move_folded_bundle_{strategy.bundle_placement}",
            boundaries[offset + 2],
            boundaries[offset + 3],
            f"Move the completed bundle to the {destination} side as one object.",
            (
                f"Only after folding is complete, smoothly lift or slide the entire compact "
                f"bundle toward the clear {destination} side. Preserve its layered shape."
            ),
            frame,
            "slow",
            ("The body fold is complete and the compact bundle is stable.",),
            ("The bundle moves as one object without reopening or a single-frame jump.",),
            (f"The compact bundle reaches the {destination} side.",),
            (
                "bundle_move_after_body_fold",
                "bundle_moves_as_one_material",
                "no_teleportation_or_crossfade",
                "camera_and_background_static",
            ),
        ),
        _phase(
            "terminal_bundle_hold",
            boundaries[offset + 3],
            boundaries[offset + 4],
            "Hold the completed fold at the selected side for inspection.",
            (
                f"Stop all motion and hold one compact folded shirt at the "
                f"{destination} side through the last frame."
            ),
            frame,
            "stationary",
            (f"The compact bundle is at the {destination} side.",),
            ("No drift, rebound, unfolding, identity change, or background motion.",),
            ("The selected terminal state remains stable through the final frame.",),
            (
                "terminal_compact_bundle_stable",
                "single_shirt_identity",
                "camera_and_background_static",
            ),
        ),
    )


def _language(
    request: TaskReasoningRequest,
    strategy: TshirtFoldStrategy,
    phases: Sequence[TaskPhase],
) -> LanguageAnalysis:
    destination = _side_text(strategy.bundle_placement)
    if strategy.sleeve_order == SIMULTANEOUS:
        sleeve_instruction = (
            "establish bilateral cuff contacts; fold both sleeves inward synchronously "
            "while the torso remains table-supported; settle both sleeves"
        )
        order_relation = "both sleeve motion onsets remain within the frozen synchronization window"
    else:
        first, second = strategy.ordered_sleeves or ()
        first_text = _side_text(first)
        second_text = _side_text(second)
        sleeve_instruction = (
            f"establish supported contact and fold the {first_text} sleeve; settle it; "
            f"then establish supported contact and fold the {second_text} sleeve"
        )
        order_relation = f"the {first_text} fold settles before the {second_text} fold begins"
    return LanguageAnalysis(
        source_language=_detect_language(request.instruction),
        normalized_instruction=(
            f"Preserve the exact first frame; {sleeve_instruction}; fold the body "
            f"bottom-to-top; compact the bundle; move it to {destination}; hold."
        ),
        ordered_actions=tuple(phase.phase_id for phase in phases),
        spatial_relations=(
            "viewer-left and viewer-right are fixed directions in the named camera frame",
            "each cuff remains bound to its original shoulder seam by a material polyline",
            order_relation,
            f"the completed compact bundle terminates at {destination}",
        ),
        temporal_modifiers=(
            "contact before cloth motion",
            (
                "bilateral sleeve motion and settling before body fold"
                if strategy.sleeve_order == SIMULTANEOUS
                else "first sleeve fold and settling before second sleeve motion"
            ),
            "both sleeves before body fold",
            "body fold before bundle transport",
            "terminal hold after transport",
        ),
        ambiguity_resolutions=(
            "Left and right are viewer-relative in the named camera frame, never robot-base sides.",
            (
                "Sleeve length is tracked camera-frame material-polyline arclength with "
                "segment-level bounds; any missing or violating track fails closed."
            ),
            (
                "Simultaneous means synchronized observable sleeve-motion onset, not "
                "identical robot joint trajectories."
            ),
            (
                "A compact visual fold is not evidence of force control, calibrated 3-D "
                "cloth state, collision safety, or real-robot feasibility."
            ),
        ),
    )


def _findings(strategy: TshirtFoldStrategy) -> tuple[ReasoningFinding, ...]:
    support = (
        ReasoningFinding(
            "bilateral_table_supported_fold",
            "contact_reasoning",
            "INFERRED",
            "Two grippers can fold opposite sleeves together only while the torso remains supported.",
            (
                "With one gripper per cuff, inventing two additional shoulder contacts would "
                "violate the observed two-arm embodiment."
            ),
            "Require both cuff contacts first and keep the torso and shoulder seams table-supported.",
        )
        if strategy.sleeve_order == SIMULTANEOUS
        else ReasoningFinding(
            "supported_sequential_fold",
            "contact_reasoning",
            "INFERRED",
            "Sequential folding permits one cuff grasp and one shoulder-side stabilizing contact.",
            "Visible support makes cloth motion causal and reduces unsupported generative deformation.",
            "Require supported contact before each sleeve moves and a settled dwell between sleeves.",
        )
    )
    return (
        ReasoningFinding(
            "sleeves_require_material_identity",
            "cloth_conservation",
            "INFERRED",
            "A sleeve cannot physically shrink or grow while being folded.",
            "Generative morphing can shorten a sleeve even when endpoints look plausible.",
            "Track cuff, intermediate material points, and shoulder; fail on missing tracks or drift.",
        ),
        support,
        ReasoningFinding(
            "strategy_must_bind_evaluation",
            "task_reasoning",
            "INFERRED",
            "Sleeve order and terminal placement define different valid task strategies.",
            "A candidate from one strategy must not pass another strategy's order or direction gate.",
            "Bind every strategy to a distinct task id, plan hash, prompt, and evaluation contract.",
        ),
        ReasoningFinding(
            "physical_measurements_unavailable",
            "claim_boundary",
            "UNAVAILABLE",
            "The visual harness has no calibrated cloth mesh, force, tactile data, or joint trajectory.",
            "Image-space consistency cannot establish physical execution or safety.",
            "Treat MiniMax-H3 as a proposal model and retain human veto gates.",
        ),
    )


class TshirtFoldStrategyReasoningPlugin:
    """Expand a T-shirt request into one or all six bounded task strategies."""

    descriptor = ReasoningPluginDescriptor(
        name="tshirt-fold-strategy-language-planner",
        version="1.0.0",
        stage="reasoning",
        description=(
            "Expands T-shirt folding language into strategy-specific causal phases, "
            "material-conservation gates, and terminal-placement alternatives."
        ),
        capabilities=(
            "language_analysis",
            "task_expansion",
            "strategy_branching",
            "cloth_material_conservation",
            "contact_causality",
            "test_time_scaling_feedback",
            TSHIRT_FOLD_TASK,
        ),
        deterministic=True,
        heavyweight=False,
    )

    def analyze(self, request: TaskReasoningRequest) -> TaskReasoningPlan:
        """Analyze one request using the explicitly stated or conservative default strategy."""

        text = " ".join((request.instruction, *request.user_constraints)).lower()
        order = LEFT_THEN_RIGHT
        if any(token in text for token in ("同时", "一起叠", "simultaneous", "together")):
            order = SIMULTANEOUS
        elif any(token in text for token in ("右边袖子先", "右袖先", "right sleeve first")):
            order = RIGHT_THEN_LEFT
        placement = (
            VIEWER_RIGHT
            if any(
                token in text
                for token in ("放到右边", "移到右边", "place right", "viewer-right side")
            )
            else VIEWER_LEFT
        )
        return self.analyze_strategy(request, TshirtFoldStrategy(order, placement))

    def analyze_candidates(
        self,
        request: TaskReasoningRequest,
        strategies: Sequence[TshirtFoldStrategy] | None = None,
    ) -> tuple[TaskReasoningPlan, ...]:
        selected = tuple(strategies or all_tshirt_fold_strategies())
        if not selected or len({item.strategy_id for item in selected}) != len(selected):
            raise ValueError("T-shirt strategy candidates must be non-empty and unique")
        return tuple(self.analyze_strategy(request, strategy) for strategy in selected)

    def analyze_strategy(
        self,
        request: TaskReasoningRequest,
        strategy: TshirtFoldStrategy,
    ) -> TaskReasoningPlan:
        if request.task_type != TSHIRT_FOLD_TASK:
            raise ValueError(f"unsupported task type: {request.task_type}")
        if request.duration_seconds < 4.0:
            raise ValueError("continuous two-arm T-shirt folding requires at least 4 seconds")
        phases = (
            _simultaneous_phases(request, strategy)
            if strategy.sleeve_order == SIMULTANEOUS
            else _sequential_phases(request, strategy)
        )
        destination = _side_text(strategy.bundle_placement)
        unlocked: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "plugin": asdict(self.descriptor),
            "task_id": _variant_task_id(request.task_id, strategy),
            "task_type": request.task_type,
            "coordinate_frame": request.coordinate_frame,
            "duration_seconds": request.duration_seconds,
            "language_analysis": asdict(_language(request, strategy, phases)),
            "physical_analysis": [asdict(item) for item in _findings(strategy)],
            "phases": [asdict(item) for item in phases],
            "global_constraints": [
                "Use only the named camera frame; viewer-left and viewer-right never change meaning.",
                "Preserve one shirt, both original sleeves, cuffs, shoulder seams, and two manipulators.",
                "A sleeve never moves before visible causal contact.",
                "Both sleeve material polylines and all tracked segments must remain within frozen bounds.",
                "Every cloth and gripper point must obey the per-point temporal displacement gate.",
                "Never let a mean score, seed count, or preference override one failed hard gate.",
                f"The completed bundle terminates at {destination}; the opposite direction is a task failure.",
                "The camera, table, glass, cables, surrounding garments, lighting, and background remain fixed.",
                "Do not claim metric geometry, force, safety, joint feasibility, or real-robot success.",
            ],
            "verification_gates": [asdict(item) for item in _gates(strategy)],
            "claim_boundary": (
                "This strategy constrains a continuous generated camera-pixel proposal. "
                "Sleeve material length, causal order or synchronization, per-point continuity, "
                f"body-fold order, and terminal placement at {destination} are fail-closed gates. "
                "They are not calibrated 3-D cloth geometry, force evidence, robot joint commands, "
                "collision safety, or physical execution."
            ),
        }
        return TaskReasoningPlan.from_dict(
            {**unlocked, "plan_sha256": _canonical_sha256(unlocked)}
        )


def strategy_from_plan(plan: TaskReasoningPlan) -> TshirtFoldStrategy:
    """Recover and validate the strategy encoded by a generated phase plan."""

    phase_ids = tuple(phase.phase_id for phase in plan.phases)
    if "fold_both_sleeves_synchronously" in phase_ids:
        order = SIMULTANEOUS
    else:
        left_index = phase_ids.index("fold_viewer_left_sleeve")
        right_index = phase_ids.index("fold_viewer_right_sleeve")
        order = LEFT_THEN_RIGHT if left_index < right_index else RIGHT_THEN_LEFT
    if "move_folded_bundle_viewer_left" in phase_ids:
        placement = VIEWER_LEFT
    elif "move_folded_bundle_viewer_right" in phase_ids:
        placement = VIEWER_RIGHT
    else:
        raise ValueError("T-shirt strategy plan has no terminal bundle-placement phase")
    strategy = TshirtFoldStrategy(order, placement)
    if f"--{strategy.strategy_id}" not in plan.task_id:
        raise ValueError("T-shirt plan task id does not bind the encoded strategy")
    return strategy
