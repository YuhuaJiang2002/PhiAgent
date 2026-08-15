"""Dependency-free metrics for action trajectories and EPL annotations."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isclose, isfinite, sqrt

Vector = tuple[float, ...]
_TIME_ABS_TOLERANCE = 1e-9


def _unit_score(value: float, name: str) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _finite(value: float, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _times_close(first: float, second: float) -> bool:
    return isclose(first, second, rel_tol=0.0, abs_tol=_TIME_ABS_TOLERANCE)


def _validate_frame(frame: str, name: str) -> None:
    if not isinstance(frame, str) or not frame.strip():
        raise ValueError(f"{name} must be a non-empty named coordinate frame")


@dataclass(frozen=True)
class Trajectory:
    """A sampled Cartesian trajectory in one explicit coordinate frame."""

    timestamps: tuple[float, ...]
    positions: tuple[Vector, ...]
    frame: str

    def __post_init__(self) -> None:
        _validate_frame(self.frame, "trajectory frame")
        if len(self.timestamps) != len(self.positions):
            raise ValueError("trajectory timestamps and positions must have aligned lengths")
        if len(self.timestamps) < 2:
            raise ValueError("trajectory requires at least two states")
        dimensions: int | None = None
        previous_time: float | None = None
        for index, (timestamp, position) in enumerate(zip(self.timestamps, self.positions)):
            _finite(timestamp, f"trajectory timestamp {index}")
            if previous_time is not None and timestamp <= previous_time:
                raise ValueError("trajectory timestamps must be strictly monotonic")
            previous_time = timestamp
            if not position:
                raise ValueError(f"trajectory position {index} cannot be empty")
            if dimensions is None:
                dimensions = len(position)
            elif len(position) != dimensions:
                raise ValueError("trajectory positions must have a consistent dimension")
            for axis, coordinate in enumerate(position):
                _finite(coordinate, f"trajectory position {index} coordinate {axis}")


@dataclass(frozen=True)
class EPLPhase:
    """A named EPL phase interval on the same clock as a trajectory."""

    label: str
    start_time: float
    end_time: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("EPL phase label must be non-empty")
        _finite(self.start_time, "EPL phase start_time")
        _finite(self.end_time, "EPL phase end_time")
        if self.end_time <= self.start_time:
            raise ValueError("EPL phase end_time must be after start_time")


@dataclass(frozen=True)
class EPLTimeline:
    """An ordered, non-overlapping temporal segmentation of an action."""

    phases: tuple[EPLPhase, ...]

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("EPL timeline cannot be empty")
        previous_end: float | None = None
        for phase in self.phases:
            if (
                previous_end is not None
                and phase.start_time < previous_end
                and not _times_close(phase.start_time, previous_end)
            ):
                raise ValueError("EPL phases must not overlap")
            previous_end = phase.end_time


@dataclass(frozen=True)
class ContactEvent:
    """An active contact interval for a named contact relation."""

    label: str
    onset_time: float
    offset_time: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("contact event label must be non-empty")
        _finite(self.onset_time, "contact event onset_time")
        _finite(self.offset_time, "contact event offset_time")
        if self.offset_time <= self.onset_time:
            raise ValueError("contact event offset_time must be after onset_time")


@dataclass(frozen=True)
class ContactTimeline:
    """Contact events; different labels may be active concurrently."""

    events: tuple[ContactEvent, ...] = ()

    def __post_init__(self) -> None:
        previous_offsets: dict[str, float] = {}
        for event in sorted(self.events, key=lambda item: (item.label, item.onset_time)):
            previous_offset = previous_offsets.get(event.label)
            if (
                previous_offset is not None
                and event.onset_time < previous_offset
                and not _times_close(event.onset_time, previous_offset)
            ):
                raise ValueError("contact events with the same label must not overlap")
            previous_offsets[event.label] = event.offset_time


@dataclass(frozen=True)
class TaskMotionInput:
    """All action and EPL evidence needed to evaluate one execution."""

    trajectory: Trajectory
    phases: EPLTimeline
    contacts: ContactTimeline = ContactTimeline()


@dataclass(frozen=True)
class ActionDiagnostics:
    dtw_mean_normalized_distance: float
    terminal_normalized_error: float
    reference_duration: float
    candidate_duration: float
    temporal_overlap_duration: float

    def __post_init__(self) -> None:
        for name in (
            "dtw_mean_normalized_distance",
            "terminal_normalized_error",
            "reference_duration",
            "candidate_duration",
            "temporal_overlap_duration",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class PhaseDiagnostics:
    boundary_mean_absolute_error: float
    labels_compared: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isfinite(self.boundary_mean_absolute_error) or self.boundary_mean_absolute_error < 0:
            raise ValueError("boundary_mean_absolute_error must be finite and non-negative")


@dataclass(frozen=True)
class ContactDiagnostics:
    onset_offset_mean_absolute_error: float
    matched_event_count: int
    reference_event_count: int
    candidate_event_count: int

    def __post_init__(self) -> None:
        if (
            not isfinite(self.onset_offset_mean_absolute_error)
            or self.onset_offset_mean_absolute_error < 0
        ):
            raise ValueError("onset_offset_mean_absolute_error must be finite and non-negative")
        for name in ("matched_event_count", "reference_event_count", "candidate_event_count"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class TaskMotionScorecard:
    """Immutable normalized scores plus unnormalized evidence for audit."""

    trajectory_similarity: float
    direction_progress_adherence: float
    terminal_state_accuracy: float
    horizon_coverage: float
    phase_macro_f1: float
    phase_boundary_score: float
    contact_state_f1: float
    contact_timing_score: float
    action_adherence: float
    phase_agreement: float
    contact_agreement: float
    overall_score: float
    action_diagnostics: ActionDiagnostics
    phase_diagnostics: PhaseDiagnostics
    contact_diagnostics: ContactDiagnostics

    def __post_init__(self) -> None:
        for name in (
            "trajectory_similarity",
            "direction_progress_adherence",
            "terminal_state_accuracy",
            "horizon_coverage",
            "phase_macro_f1",
            "phase_boundary_score",
            "contact_state_f1",
            "contact_timing_score",
            "action_adherence",
            "phase_agreement",
            "contact_agreement",
            "overall_score",
        ):
            _unit_score(getattr(self, name), name)


def _distance(left: Vector, right: Vector) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _trajectory_dtw(reference: Trajectory, candidate: Trajectory, scale: float) -> float:
    width = len(candidate.positions)
    costs = [(float("inf"), 0)] * (width + 1)
    costs[0] = (0.0, 0)
    for reference_position in reference.positions:
        next_costs = [(float("inf"), 0)] * (width + 1)
        for candidate_index, candidate_position in enumerate(candidate.positions, start=1):
            previous = min(
                costs[candidate_index],
                next_costs[candidate_index - 1],
                costs[candidate_index - 1],
                key=lambda item: item[0],
            )
            next_costs[candidate_index] = (
                previous[0] + _distance(reference_position, candidate_position) / scale,
                previous[1] + 1,
            )
        costs = next_costs
    total_cost, steps = costs[-1]
    return total_cost / steps


def _interpolate(trajectory: Trajectory, timestamp: float) -> Vector:
    if timestamp <= trajectory.timestamps[0]:
        return trajectory.positions[0]
    if timestamp >= trajectory.timestamps[-1]:
        return trajectory.positions[-1]
    for index, next_time in enumerate(trajectory.timestamps[1:], start=1):
        if timestamp <= next_time:
            prior_time = trajectory.timestamps[index - 1]
            fraction = (timestamp - prior_time) / (next_time - prior_time)
            return tuple(
                prior + fraction * (following - prior)
                for prior, following in zip(
                    trajectory.positions[index - 1], trajectory.positions[index]
                )
            )
    raise AssertionError("timestamp should be bracketed by trajectory samples")


def _direction_progress(reference: Trajectory, candidate: Trajectory, scale: float) -> float:
    reference_delta = tuple(
        end - start for start, end in zip(reference.positions[0], reference.positions[-1])
    )
    candidate_delta = tuple(
        end - start for start, end in zip(candidate.positions[0], candidate.positions[-1])
    )
    reference_length = _distance(reference.positions[0], reference.positions[-1])
    candidate_length = _distance(candidate.positions[0], candidate.positions[-1])
    if reference_length < 1e-12:
        direction_score = 1.0 if candidate_length < 1e-12 else 0.0
    elif candidate_length < 1e-12:
        direction_score = 0.0
    else:
        cosine = sum(a * b for a, b in zip(reference_delta, candidate_delta))
        cosine /= reference_length * candidate_length
        direction_score = (
            max(0.0, min(1.0, cosine))
            * min(reference_length, candidate_length)
            / max(reference_length, candidate_length)
        )

    reference_start = reference.timestamps[0]
    reference_duration = reference.timestamps[-1] - reference_start
    candidate_start = candidate.timestamps[0]
    candidate_duration = candidate.timestamps[-1] - candidate_start
    errors = []
    for timestamp, position in zip(reference.timestamps, reference.positions):
        fraction = (timestamp - reference_start) / reference_duration
        candidate_position = _interpolate(
            candidate, candidate_start + fraction * candidate_duration
        )
        errors.append(_distance(position, candidate_position) / scale)
    progress_score = exp(-sum(errors) / len(errors))
    return min(direction_score, progress_score)


def _overlap_duration(
    left_start: float, left_end: float, right_start: float, right_end: float
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _validate_timeline_bounds(timeline: EPLTimeline, trajectory: Trajectory, role: str) -> None:
    start, end = trajectory.timestamps[0], trajectory.timestamps[-1]
    if not _times_close(timeline.phases[0].start_time, start) or not _times_close(
        timeline.phases[-1].end_time, end
    ):
        raise ValueError(f"{role} EPL timeline must cover its trajectory horizon exactly")
    for previous, following in zip(timeline.phases, timeline.phases[1:]):
        if not _times_close(previous.end_time, following.start_time):
            raise ValueError(f"{role} EPL timeline must be contiguous")


def _validate_contact_bounds(timeline: ContactTimeline, trajectory: Trajectory, role: str) -> None:
    start, end = trajectory.timestamps[0], trajectory.timestamps[-1]
    for event in timeline.events:
        if (
            event.onset_time < start
            and not _times_close(event.onset_time, start)
        ) or (
            event.offset_time > end
            and not _times_close(event.offset_time, end)
        ):
            raise ValueError(
                f"{role} contact event {event.label!r} must lie within its trajectory horizon"
            )


def _phase_metrics(
    reference: EPLTimeline, candidate: EPLTimeline, duration: float
) -> tuple[float, float, PhaseDiagnostics]:
    labels = tuple(sorted({phase.label for phase in reference.phases + candidate.phases}))
    f1_scores = []
    for label in labels:
        reference_duration = sum(
            phase.end_time - phase.start_time for phase in reference.phases if phase.label == label
        )
        candidate_duration = sum(
            phase.end_time - phase.start_time for phase in candidate.phases if phase.label == label
        )
        overlap = sum(
            _overlap_duration(left.start_time, left.end_time, right.start_time, right.end_time)
            for left in reference.phases
            if left.label == label
            for right in candidate.phases
            if right.label == label
        )
        denominator = reference_duration + candidate_duration
        f1_scores.append(1.0 if denominator == 0 else 2.0 * overlap / denominator)
    reference_boundaries = tuple(phase.end_time for phase in reference.phases[:-1])
    candidate_boundaries = tuple(phase.end_time for phase in candidate.phases[:-1])
    if not reference_boundaries and not candidate_boundaries:
        boundary_error = 0.0
    elif not reference_boundaries or not candidate_boundaries:
        boundary_error = duration
    else:
        errors = [
            min(abs(boundary - other) for other in candidate_boundaries)
            for boundary in reference_boundaries
        ]
        errors.extend(
            min(abs(boundary - other) for other in reference_boundaries)
            for boundary in candidate_boundaries
        )
        boundary_error = sum(errors) / len(errors)
    return (
        sum(f1_scores) / len(f1_scores),
        exp(-boundary_error / duration),
        PhaseDiagnostics(boundary_error, labels),
    )


def _event_duration(events: tuple[ContactEvent, ...], label: str) -> float:
    return sum(event.offset_time - event.onset_time for event in events if event.label == label)


def _contact_metrics(
    reference: ContactTimeline, candidate: ContactTimeline, duration: float
) -> tuple[float, float, ContactDiagnostics]:
    labels = {event.label for event in reference.events + candidate.events}
    if not labels:
        diagnostics = ContactDiagnostics(0.0, 0, 0, 0)
        return 1.0, 1.0, diagnostics
    true_positive = false_positive = false_negative = 0.0
    for label in labels:
        reference_events = tuple(event for event in reference.events if event.label == label)
        candidate_events = tuple(event for event in candidate.events if event.label == label)
        overlap = sum(
            _overlap_duration(
                left.onset_time, left.offset_time, right.onset_time, right.offset_time
            )
            for left in reference_events
            for right in candidate_events
        )
        reference_duration = _event_duration(reference.events, label)
        candidate_duration = _event_duration(candidate.events, label)
        true_positive += overlap
        false_positive += candidate_duration - overlap
        false_negative += reference_duration - overlap
    denominator = 2.0 * true_positive + false_positive + false_negative
    state_f1 = 1.0 if denominator == 0.0 else 2.0 * true_positive / denominator

    matched: list[tuple[ContactEvent, ContactEvent]] = []
    unmatched = 0
    for label in labels:
        available = [event for event in candidate.events if event.label == label]
        for reference_event in (event for event in reference.events if event.label == label):
            if not available:
                unmatched += 1
                continue
            nearest = min(
                available,
                key=lambda event: abs(event.onset_time - reference_event.onset_time)
                + abs(event.offset_time - reference_event.offset_time),
            )
            available.remove(nearest)
            matched.append((reference_event, nearest))
        unmatched += len(available)
    if unmatched:
        timing_error = duration
    elif not matched:
        timing_error = 0.0
    else:
        timing_error = sum(
            (
                abs(reference_event.onset_time - candidate_event.onset_time)
                + abs(reference_event.offset_time - candidate_event.offset_time)
            )
            / 2.0
            for reference_event, candidate_event in matched
        ) / len(matched)
    diagnostics = ContactDiagnostics(
        timing_error, len(matched), len(reference.events), len(candidate.events)
    )
    return state_f1, exp(-timing_error / duration), diagnostics


def evaluate_task_motion(
    reference: TaskMotionInput,
    candidate: TaskMotionInput,
    *,
    spatial_normalization: float,
) -> TaskMotionScorecard:
    """Evaluate action execution without allowing strong dimensions to compensate failures."""

    _finite(spatial_normalization, "spatial_normalization")
    if spatial_normalization <= 0.0:
        raise ValueError("spatial_normalization must be positive")
    reference_trajectory = reference.trajectory
    candidate_trajectory = candidate.trajectory
    if reference_trajectory.frame != candidate_trajectory.frame:
        raise ValueError(
            "trajectory frame mismatch: "
            f"reference={reference_trajectory.frame!r}, candidate={candidate_trajectory.frame!r}"
        )
    if len(reference_trajectory.positions[0]) != len(candidate_trajectory.positions[0]):
        raise ValueError("trajectory position dimensions must match")
    _validate_timeline_bounds(reference.phases, reference_trajectory, "reference")
    _validate_timeline_bounds(candidate.phases, candidate_trajectory, "candidate")
    _validate_contact_bounds(reference.contacts, reference_trajectory, "reference")
    _validate_contact_bounds(candidate.contacts, candidate_trajectory, "candidate")

    reference_start, reference_end = (
        reference_trajectory.timestamps[0],
        reference_trajectory.timestamps[-1],
    )
    candidate_start, candidate_end = (
        candidate_trajectory.timestamps[0],
        candidate_trajectory.timestamps[-1],
    )
    reference_duration = reference_end - reference_start
    candidate_duration = candidate_end - candidate_start
    overlap = _overlap_duration(reference_start, reference_end, candidate_start, candidate_end)
    trajectory_distance = _trajectory_dtw(
        reference_trajectory, candidate_trajectory, spatial_normalization
    )
    trajectory_similarity = exp(-trajectory_distance)
    direction_progress = _direction_progress(
        reference_trajectory, candidate_trajectory, spatial_normalization
    )
    terminal_error = (
        _distance(reference_trajectory.positions[-1], candidate_trajectory.positions[-1])
        / spatial_normalization
    )
    terminal_accuracy = exp(-terminal_error)
    coverage = min(1.0, max(0.0, overlap / reference_duration))
    phase_f1, phase_boundary, phase_diagnostics = _phase_metrics(
        reference.phases, candidate.phases, reference_duration
    )
    contact_f1, contact_timing, contact_diagnostics = _contact_metrics(
        reference.contacts, candidate.contacts, reference_duration
    )
    action_adherence = min(trajectory_similarity, direction_progress, terminal_accuracy, coverage)
    phase_agreement = min(phase_f1, phase_boundary)
    contact_agreement = min(contact_f1, contact_timing)
    return TaskMotionScorecard(
        trajectory_similarity=trajectory_similarity,
        direction_progress_adherence=direction_progress,
        terminal_state_accuracy=terminal_accuracy,
        horizon_coverage=coverage,
        phase_macro_f1=phase_f1,
        phase_boundary_score=phase_boundary,
        contact_state_f1=contact_f1,
        contact_timing_score=contact_timing,
        action_adherence=action_adherence,
        phase_agreement=phase_agreement,
        contact_agreement=contact_agreement,
        overall_score=min(action_adherence, phase_agreement, contact_agreement),
        action_diagnostics=ActionDiagnostics(
            trajectory_distance,
            terminal_error,
            reference_duration,
            candidate_duration,
            overlap,
        ),
        phase_diagnostics=phase_diagnostics,
        contact_diagnostics=contact_diagnostics,
    )
