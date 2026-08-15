"""CPU-only evaluation of tracked hand--object interactions in named frames.

This module deliberately consumes geometry tracks rather than image tensors, so it
can be paired with visual object-instance metrics without imposing a perception
backend or GPU dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class FramePoint3D:
    """A metric point whose coordinate frame is explicit."""

    frame: str
    x_m: float
    y_m: float
    z_m: float

    def __post_init__(self) -> None:
        if not self.frame.strip():
            raise ValueError("coordinate frame must be non-empty")
        if not all(math.isfinite(value) for value in (self.x_m, self.y_m, self.z_m)):
            raise ValueError("point coordinates must be finite")

    def distance_to(self, other: FramePoint3D) -> float:
        if self.frame != other.frame:
            raise ValueError("points must use the same coordinate frame")
        return math.dist((self.x_m, self.y_m, self.z_m), (other.x_m, other.y_m, other.z_m))

    def minus(self, other: FramePoint3D) -> tuple[float, float, float]:
        if self.frame != other.frame:
            raise ValueError("points must use the same coordinate frame")
        return (self.x_m - other.x_m, self.y_m - other.y_m, self.z_m - other.z_m)


@dataclass(frozen=True)
class TrackedObjectObservation:
    """One detector observation; repeated IDs in a frame represent duplication."""

    object_id: str
    position: FramePoint3D | None
    visible: bool = True
    state: str = "unknown"

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError("object_id must be non-empty")
        if not self.state.strip():
            raise ValueError("object state must be non-empty")
        if self.visible and self.position is None:
            raise ValueError("a visible object observation requires a position")


@dataclass(frozen=True)
class InteractionFrame:
    """Synchronous hand and object observations at one timestamp."""

    timestamp_s: float
    hand_position: FramePoint3D
    objects: tuple[TrackedObjectObservation, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("interaction timestamp must be finite and non-negative")


@dataclass(frozen=True)
class InteractionTrace:
    """A tracked interaction, all expressed in one named coordinate frame."""

    target_object_id: str
    coordinate_frame: str
    frames: tuple[InteractionFrame, ...]

    def __post_init__(self) -> None:
        if not self.target_object_id.strip():
            raise ValueError("target_object_id must be non-empty")
        if not self.coordinate_frame.strip():
            raise ValueError("coordinate_frame must be non-empty")
        if len(self.frames) < 2:
            raise ValueError("interaction traces require at least two frames")
        timestamps = tuple(frame.timestamp_s for frame in self.frames)
        if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
            raise ValueError("interaction timestamps must be strictly increasing")
        for frame in self.frames:
            if frame.hand_position.frame != self.coordinate_frame:
                raise ValueError("hand positions must match the trace coordinate frame")
            for observation in frame.objects:
                if (
                    observation.position is not None
                    and observation.position.frame != self.coordinate_frame
                ):
                    raise ValueError("object positions must match the trace coordinate frame")


@dataclass(frozen=True)
class InteractionExpectation:
    """Reference interaction and the required terminal symbolic state."""

    reference: InteractionTrace
    terminal_state: str | None = None

    def __post_init__(self) -> None:
        terminal_state = self.terminal_state
        observations, missing, duplicates = _target_series(self.reference)
        if (
            missing
            or duplicates
            or any(
                observation is None or not observation.visible or observation.position is None
                for observation in observations
            )
        ):
            raise ValueError("reference must contain one visible target in every frame")
        if terminal_state is None:
            final = observations[-1]
            assert final is not None
            terminal_state = final.state
            object.__setattr__(self, "terminal_state", terminal_state)
        if not terminal_state.strip():
            raise ValueError("terminal_state must be non-empty")


@dataclass(frozen=True)
class InteractionEvaluationConfig:
    """Physical tolerances and fail-closed score thresholds, in metres and seconds."""

    contact_distance_m: float = 0.03
    trajectory_tolerance_m: float = 0.03
    distance_tolerance_m: float = 0.02
    coupling_tolerance_m: float = 0.02
    terminal_tolerance_m: float = 0.03
    motion_threshold_m: float = 0.01
    maximum_object_step_m: float = 0.20
    causal_tolerance_s: float = 0.03
    causal_score_decay_s: float = 0.10
    timing_tolerance_s: float = 0.05
    normalization_m: float = 0.10
    minimum_identity_score: float = 1.0
    minimum_visibility_coverage: float = 1.0
    minimum_trajectory_score: float = 0.75
    minimum_terminal_score: float = 1.0
    minimum_distance_score: float = 0.75
    minimum_contact_score: float = 0.75
    minimum_contact_timing_score: float = 0.75
    minimum_coupling_score: float = 0.75
    minimum_causal_score: float = 1.0

    def __post_init__(self) -> None:
        positive = (
            "contact_distance_m",
            "trajectory_tolerance_m",
            "distance_tolerance_m",
            "coupling_tolerance_m",
            "terminal_tolerance_m",
            "motion_threshold_m",
            "maximum_object_step_m",
            "causal_score_decay_s",
            "timing_tolerance_s",
            "normalization_m",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.causal_tolerance_s) or self.causal_tolerance_s < 0:
            raise ValueError("causal_tolerance_s must be finite and non-negative")
        for field in fields(self):
            if field.name.startswith("minimum_"):
                value = getattr(self, field.name)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{field.name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class InteractionDiagnostics:
    """Raw, serializable measurements used to make the scorecard auditable."""

    identity_valid_frames: int
    visible_target_frames: int
    missing_target_frames: tuple[int, ...]
    duplicate_target_frames: tuple[int, ...]
    candidate_hand_object_distances_m: tuple[float | None, ...]
    reference_hand_object_distances_m: tuple[float | None, ...]
    candidate_contact: tuple[bool, ...]
    reference_contact: tuple[bool, ...]
    candidate_contact_onset_s: float | None
    candidate_contact_offset_s: float | None
    reference_contact_onset_s: float | None
    reference_contact_offset_s: float | None
    object_motion_onset_s: float | None
    coupling_errors_m: tuple[float, ...]
    teleport_frames: tuple[int, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class InteractionScorecard:
    """Named [0, 1] scores plus a strict pass/fail decision and measurements."""

    identity_score: float
    visibility_coverage: float
    relative_trajectory_score: float
    terminal_state_score: float
    hand_object_distance_score: float
    contact_agreement_score: float
    contact_timing_score: float
    motion_coupling_score: float
    causal_order_score: float
    continuity_score: float
    manipulation_score: float
    passed: bool
    diagnostics: InteractionDiagnostics

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name.endswith("_score") or field.name in {
                "visibility_coverage",
                "continuity_score",
            }:
                value = getattr(self, field.name)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{field.name} must be finite and in [0, 1]")


def _single_target_observation(
    frame: InteractionFrame, target_object_id: str
) -> TrackedObjectObservation | None:
    matches = tuple(item for item in frame.objects if item.object_id == target_object_id)
    return matches[0] if len(matches) == 1 else None


def _target_series(
    trace: InteractionTrace,
) -> tuple[tuple[TrackedObjectObservation | None, ...], tuple[int, ...], tuple[int, ...]]:
    observations: list[TrackedObjectObservation | None] = []
    missing: list[int] = []
    duplicates: list[int] = []
    for index, frame in enumerate(trace.frames):
        matches = tuple(item for item in frame.objects if item.object_id == trace.target_object_id)
        if not matches:
            missing.append(index)
            observations.append(None)
        elif len(matches) > 1:
            duplicates.append(index)
            observations.append(None)
        else:
            observations.append(matches[0])
    return tuple(observations), tuple(missing), tuple(duplicates)


def _distances(
    trace: InteractionTrace, observations: tuple[TrackedObjectObservation | None, ...]
) -> tuple[float | None, ...]:
    return tuple(
        (
            frame.hand_position.distance_to(observation.position)
            if observation is not None and observation.visible and observation.position is not None
            else None
        )
        for frame, observation in zip(trace.frames, observations)
    )


def _onset_and_offset(
    timestamps: tuple[float, ...], values: tuple[bool, ...]
) -> tuple[float | None, float | None]:
    indices = tuple(index for index, value in enumerate(values) if value)
    if not indices:
        return None, None
    return timestamps[indices[0]], timestamps[indices[-1]]


def _mean_score(errors: tuple[float, ...], tolerance: float) -> float:
    if not errors:
        return 0.0
    return sum(max(0.0, 1.0 - error / tolerance) for error in errors) / len(errors)


def _contact_iou(candidate: tuple[bool, ...], reference: tuple[bool, ...]) -> float:
    union = sum(left or right for left, right in zip(candidate, reference))
    if union == 0:
        return 0.0
    return sum(left and right for left, right in zip(candidate, reference)) / union


def _timing_score(
    candidate_onset: float | None,
    candidate_offset: float | None,
    reference_onset: float | None,
    reference_offset: float | None,
    tolerance_s: float,
) -> float:
    if None in (candidate_onset, candidate_offset, reference_onset, reference_offset):
        return 0.0
    errors = (
        abs(candidate_onset - reference_onset),  # type: ignore[operator]
        abs(candidate_offset - reference_offset),  # type: ignore[operator]
    )
    return _mean_score(errors, tolerance_s)


def _relative_trajectory_errors(
    trace: InteractionTrace,
    observations: tuple[TrackedObjectObservation | None, ...],
    reference: InteractionTrace,
    reference_observations: tuple[TrackedObjectObservation | None, ...],
) -> tuple[float, ...]:
    errors: list[float] = []
    for frame, observation, ref_frame, ref_observation in zip(
        trace.frames, observations, reference.frames, reference_observations
    ):
        if (
            observation is None
            or not observation.visible
            or observation.position is None
            or ref_observation is None
            or not ref_observation.visible
            or ref_observation.position is None
        ):
            continue
        relative = observation.position.minus(frame.hand_position)
        expected = ref_observation.position.minus(ref_frame.hand_position)
        errors.append(math.dist(relative, expected))
    return tuple(errors)


def evaluate_interaction(
    trace: InteractionTrace,
    expectation: InteractionExpectation,
    config: InteractionEvaluationConfig = InteractionEvaluationConfig(),
) -> InteractionScorecard:
    """Evaluate whether ``trace`` performs the reference object interaction.

    The reference supplies desired relative object motion and contact timing while
    the scorecard keeps coverage, identity, causality, coupling, and continuity
    independently inspectable.  Essential gates are combined fail-closed.
    """

    reference = expectation.reference
    if trace.target_object_id != reference.target_object_id:
        raise ValueError("trace and reference target_object_id must match")
    if trace.coordinate_frame != reference.coordinate_frame:
        raise ValueError("trace and reference coordinate frames must match")
    if len(trace.frames) != len(reference.frames):
        raise ValueError("trace and reference sequence lengths must match")
    if any(
        abs(actual.timestamp_s - expected.timestamp_s) > 1e-6
        for actual, expected in zip(trace.frames, reference.frames)
    ):
        raise ValueError("trace and reference timestamps must be aligned")

    observations, missing, duplicates = _target_series(trace)
    reference_observations, _, _ = _target_series(reference)
    count = len(trace.frames)
    valid = tuple(
        observation is not None and observation.visible and observation.position is not None
        for observation in observations
    )
    visible = sum(valid)
    identity = sum(observation is not None for observation in observations)
    identity_score = identity / count
    visibility_coverage = visible / count
    distances = _distances(trace, observations)
    reference_distances = _distances(reference, reference_observations)
    candidate_contact = tuple(
        distance is not None and distance <= config.contact_distance_m for distance in distances
    )
    reference_contact = tuple(
        distance is not None and distance <= config.contact_distance_m
        for distance in reference_distances
    )
    timestamps = tuple(frame.timestamp_s for frame in trace.frames)
    contact_onset, contact_offset = _onset_and_offset(timestamps, candidate_contact)
    reference_onset, reference_offset = _onset_and_offset(timestamps, reference_contact)

    distance_errors = tuple(
        abs(actual - expected)
        for actual, expected in zip(distances, reference_distances)
        if actual is not None and expected is not None
    )
    relative_errors = _relative_trajectory_errors(
        trace, observations, reference, reference_observations
    )
    trajectory_score = _mean_score(relative_errors, config.trajectory_tolerance_m)
    distance_score = _mean_score(distance_errors, config.distance_tolerance_m)
    contact_score = _contact_iou(candidate_contact, reference_contact)
    contact_timing_score = _timing_score(
        contact_onset, contact_offset, reference_onset, reference_offset, config.timing_tolerance_s
    )

    coupling_errors: list[float] = []
    teleports: list[int] = []
    motion_onset: float | None = None
    coupled_motion = False
    for index in range(1, count):
        previous = observations[index - 1]
        current = observations[index]
        if (
            previous is None
            or current is None
            or not previous.visible
            or not current.visible
            or previous.position is None
            or current.position is None
        ):
            continue
        object_delta = current.position.minus(previous.position)
        object_step = math.dist(object_delta, (0.0, 0.0, 0.0))
        if object_step > config.maximum_object_step_m:
            teleports.append(index)
        if object_step > config.motion_threshold_m and motion_onset is None:
            motion_onset = timestamps[index]
        if candidate_contact[index - 1] and candidate_contact[index]:
            hand_delta = trace.frames[index].hand_position.minus(
                trace.frames[index - 1].hand_position
            )
            coupling_error = math.dist(object_delta, hand_delta)
            coupling_errors.append(coupling_error)
            if (
                object_step > config.motion_threshold_m
                and coupling_error <= config.coupling_tolerance_m
            ):
                coupled_motion = True
    coupling_score = _mean_score(tuple(coupling_errors), config.coupling_tolerance_m)

    final = observations[-1]
    reference_final = reference_observations[-1]
    terminal_matches = (
        final is not None
        and final.visible
        and final.position is not None
        and final.state == expectation.terminal_state
        and reference_final is not None
        and reference_final.visible
        and reference_final.position is not None
        and final.position.distance_to(reference_final.position) <= config.terminal_tolerance_m
    )
    terminal_score = 1.0 if terminal_matches else 0.0
    if contact_onset is None or motion_onset is None:
        causal_score = 0.0
    elif motion_onset >= contact_onset - config.causal_tolerance_s:
        causal_score = 1.0
    else:
        causal_score = max(
            0.0,
            1.0
            - (contact_onset - config.causal_tolerance_s - motion_onset)
            / config.causal_score_decay_s,
        )
    continuity_score = 0.0 if teleports else 1.0
    manipulation_score = 1.0 if (any(candidate_contact) and coupled_motion) else 0.0

    reasons: list[str] = []
    if missing:
        reasons.append("target object is missing in one or more frames")
    if duplicates:
        reasons.append("target object is duplicated in one or more frames")
    if teleports:
        reasons.append("target object has a discontinuous teleportation step")
    if not any(candidate_contact):
        reasons.append("target object is visible but never contacts the hand")
    if any(candidate_contact) and not coupled_motion:
        reasons.append("no coupled hand-object motion was observed while grasped")
    if causal_score < config.minimum_causal_score:
        reasons.append("object manipulation motion begins before contact")
    if not terminal_matches:
        reasons.append("target object does not reach the required terminal state")

    diagnostics = InteractionDiagnostics(
        identity_valid_frames=identity,
        visible_target_frames=visible,
        missing_target_frames=missing,
        duplicate_target_frames=duplicates,
        candidate_hand_object_distances_m=distances,
        reference_hand_object_distances_m=reference_distances,
        candidate_contact=candidate_contact,
        reference_contact=reference_contact,
        candidate_contact_onset_s=contact_onset,
        candidate_contact_offset_s=contact_offset,
        reference_contact_onset_s=reference_onset,
        reference_contact_offset_s=reference_offset,
        object_motion_onset_s=motion_onset,
        coupling_errors_m=tuple(coupling_errors),
        teleport_frames=tuple(teleports),
        reasons=tuple(reasons),
    )
    passed = (
        identity_score >= config.minimum_identity_score
        and visibility_coverage >= config.minimum_visibility_coverage
        and trajectory_score >= config.minimum_trajectory_score
        and terminal_score >= config.minimum_terminal_score
        and distance_score >= config.minimum_distance_score
        and contact_score >= config.minimum_contact_score
        and contact_timing_score >= config.minimum_contact_timing_score
        and coupling_score >= config.minimum_coupling_score
        and causal_score >= config.minimum_causal_score
        and continuity_score == 1.0
        and manipulation_score == 1.0
    )
    return InteractionScorecard(
        identity_score=identity_score,
        visibility_coverage=visibility_coverage,
        relative_trajectory_score=trajectory_score,
        terminal_state_score=terminal_score,
        hand_object_distance_score=distance_score,
        contact_agreement_score=contact_score,
        contact_timing_score=contact_timing_score,
        motion_coupling_score=coupling_score,
        causal_order_score=causal_score,
        continuity_score=continuity_score,
        manipulation_score=manipulation_score,
        passed=passed,
        diagnostics=diagnostics,
    )
