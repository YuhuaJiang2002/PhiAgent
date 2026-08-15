"""Dependency-free structural evaluation for image-plane robot embodiments.

The inputs deliberately retain their image coordinate frame.  This evaluator
does not infer relationships between camera, world, or robot-base coordinates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable


def _finite_unit(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _name(value: str, description: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")


def _unique_names(values: tuple[str, ...], description: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{description} must contain non-empty names")
    if len(set(values)) != len(values):
        raise ValueError(f"{description} must not contain duplicates")


@dataclass(frozen=True)
class ImageMask:
    """One non-empty binary robot component in a named image coordinate frame."""

    coordinate_frame: str
    width: int
    height: int
    pixels: frozenset[tuple[int, int]]

    def __post_init__(self) -> None:
        _name(self.coordinate_frame, "coordinate_frame")
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width <= 0
            or self.height <= 0
        ):
            raise ValueError("mask dimensions must be positive integers")
        try:
            pixels = frozenset(self.pixels)
        except TypeError as error:
            raise ValueError("mask pixels must be coordinate pairs") from error
        if not pixels:
            raise ValueError("a robot component mask cannot be empty")
        for pixel in pixels:
            if (
                not isinstance(pixel, tuple)
                or len(pixel) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in pixel)
            ):
                raise ValueError("mask pixels must be integer (x, y) pairs")
            x, y = pixel
            if not 0 <= x < self.width or not 0 <= y < self.height:
                raise ValueError("mask pixel lies outside mask dimensions")
        object.__setattr__(self, "pixels", pixels)

    @property
    def area(self) -> int:
        return len(self.pixels)

    @property
    def connected_components(self) -> int:
        """Return the number of 4-connected regions without image dependencies."""

        remaining = set(self.pixels)
        count = 0
        while remaining:
            count += 1
            pending = [remaining.pop()]
            while pending:
                x, y = pending.pop()
                for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        pending.append(neighbor)
        return count


@dataclass(frozen=True)
class ImageLandmark:
    """A named, finite 2-D point measured in an explicit image frame."""

    name: str
    coordinate_frame: str
    x: float
    y: float

    def __post_init__(self) -> None:
        _name(self.name, "landmark name")
        _name(self.coordinate_frame, "landmark coordinate_frame")
        try:
            finite = math.isfinite(self.x) and math.isfinite(self.y)
        except TypeError as error:
            raise ValueError("landmark coordinates must be finite") from error
        if not finite:
            raise ValueError("landmark coordinates must be finite")

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class KinematicLink:
    """A named landmark-to-landmark link, optionally with a known pixel length."""

    name: str
    start_landmark: str
    end_landmark: str
    expected_length: float | None = None

    def __post_init__(self) -> None:
        _name(self.name, "link name")
        _name(self.start_landmark, "link start landmark")
        _name(self.end_landmark, "link end landmark")
        if self.start_landmark == self.end_landmark:
            raise ValueError("a link must join two distinct landmarks")
        if self.expected_length is not None:
            try:
                valid_length = math.isfinite(self.expected_length) and self.expected_length > 0
            except TypeError as error:
                raise ValueError("expected link length must be finite and positive") from error
            if not valid_length:
                raise ValueError("expected link length must be finite and positive")


@dataclass(frozen=True)
class EmbodimentFrame:
    """Detected robot components and optional landmarks for one video frame."""

    frame_index: int
    coordinate_frame: str
    components: tuple[ImageMask, ...]
    landmarks: tuple[ImageLandmark, ...] = ()
    target_id: str | None = None

    def __post_init__(self) -> None:
        _name(self.coordinate_frame, "frame coordinate_frame")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise ValueError("frame_index must be an integer")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        components = tuple(self.components)
        landmarks = tuple(self.landmarks)
        if any(component.coordinate_frame != self.coordinate_frame for component in components):
            raise ValueError("component coordinate frame does not match frame coordinate frame")
        if any(landmark.coordinate_frame != self.coordinate_frame for landmark in landmarks):
            raise ValueError("landmark coordinate frame does not match frame coordinate frame")
        _unique_names(tuple(landmark.name for landmark in landmarks), "frame landmark names")
        if self.target_id is not None:
            _name(self.target_id, "target_id")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "landmarks", landmarks)


@dataclass(frozen=True)
class EmbodimentSequence:
    """A time-ordered image-plane embodiment track and its optional skeleton."""

    coordinate_frame: str
    frames: tuple[EmbodimentFrame, ...]
    landmark_names: tuple[str, ...] = ()
    links: tuple[KinematicLink, ...] = ()

    def __post_init__(self) -> None:
        _name(self.coordinate_frame, "sequence coordinate_frame")
        frames = tuple(self.frames)
        links = tuple(self.links)
        declared_names = tuple(self.landmark_names)
        if len(frames) < 2:
            raise ValueError("embodiment evaluation requires at least two frames")
        if any(frame.coordinate_frame != self.coordinate_frame for frame in frames):
            raise ValueError("frame coordinate frame does not match sequence coordinate frame")
        indices = tuple(frame.frame_index for frame in frames)
        if any(current <= previous for previous, current in zip(indices, indices[1:])):
            raise ValueError("frame indices must be strictly increasing")
        _unique_names(tuple(link.name for link in links), "link names")
        if not declared_names:
            declared_names = tuple(
                sorted(
                    {landmark.name for frame in frames for landmark in frame.landmarks}
                    | {
                        endpoint
                        for link in links
                        for endpoint in (link.start_landmark, link.end_landmark)
                    }
                )
            )
        _unique_names(declared_names, "sequence landmark names")
        known_names = set(declared_names)
        if any(
            landmark.name not in known_names for frame in frames for landmark in frame.landmarks
        ):
            raise ValueError("frame landmark is not declared by the sequence")
        if any(
            endpoint not in known_names
            for link in links
            for endpoint in (link.start_landmark, link.end_landmark)
        ):
            raise ValueError("link refers to an undeclared landmark")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "landmark_names", declared_names)


@dataclass(frozen=True)
class EmbodimentEvaluationConfig:
    """Thresholds and essential evidence required for a target action."""

    required_landmarks: tuple[str, ...] = ()
    expected_components: int = 1
    require_target_identity: bool = True
    expect_articulation: bool = True
    minimum_articulation_displacement: float = 0.05
    articulation_window: int = 2
    maximum_area_relative_change: float = 0.25
    maximum_link_relative_drift: float = 0.15

    def __post_init__(self) -> None:
        required_landmarks = tuple(self.required_landmarks)
        _unique_names(required_landmarks, "required landmark names")
        if (
            isinstance(self.expected_components, bool)
            or not isinstance(self.expected_components, int)
            or self.expected_components <= 0
        ):
            raise ValueError("expected_components must be a positive integer")
        if not isinstance(self.require_target_identity, bool):
            raise ValueError("require_target_identity must be a boolean")
        if not isinstance(self.expect_articulation, bool):
            raise ValueError("expect_articulation must be a boolean")
        if (
            isinstance(self.articulation_window, bool)
            or not isinstance(self.articulation_window, int)
            or self.articulation_window < 2
        ):
            raise ValueError("articulation_window must be an integer of at least two frames")
        for value, name in (
            (self.minimum_articulation_displacement, "minimum_articulation_displacement"),
            (self.maximum_area_relative_change, "maximum_area_relative_change"),
            (self.maximum_link_relative_drift, "maximum_link_relative_drift"),
        ):
            try:
                valid = math.isfinite(value) and value > 0
            except TypeError as error:
                raise ValueError(f"{name} must be finite and positive") from error
            if not valid:
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(self, "required_landmarks", required_landmarks)


@dataclass(frozen=True)
class EmbodimentDiagnostics:
    """Raw evidence retained beside normalized scores for later audit."""

    component_counts: tuple[int, ...]
    connected_component_counts: tuple[int, ...]
    component_areas: tuple[int, ...]
    missing_landmarks: tuple[tuple[str, ...], ...]
    link_relative_drifts: tuple[tuple[str, float], ...]
    target_ids: tuple[str | None, ...]
    articulation_displacements: tuple[float, ...]
    sustained_articulation_displacements: tuple[float, ...]


@dataclass(frozen=True)
class EmbodimentScorecard:
    """Essential, [0,1]-bounded scores; each is minimum/tail-sensitive."""

    topology: float
    area_stability: float
    landmark_tracking: float
    geometry: float
    target_identity: float
    articulation: float
    essential: float
    diagnostics: EmbodimentDiagnostics

    def __post_init__(self) -> None:
        for name in (
            "topology",
            "area_stability",
            "landmark_tracking",
            "geometry",
            "target_identity",
            "articulation",
            "essential",
        ):
            _finite_unit(getattr(self, name), name)

    def scores(self) -> dict[str, float]:
        return {
            "topology": self.topology,
            "area_stability": self.area_stability,
            "landmark_tracking": self.landmark_tracking,
            "geometry": self.geometry,
            "target_identity": self.target_identity,
            "articulation": self.articulation,
            "essential": self.essential,
        }

    def as_dict(self) -> dict[str, object]:
        return {"scores": self.scores(), "diagnostics": asdict(self.diagnostics)}


def _landmark_map(frame: EmbodimentFrame) -> dict[str, ImageLandmark]:
    return {landmark.name: landmark for landmark in frame.landmarks}


def _relative_configuration(
    points: Iterable[ImageLandmark],
) -> tuple[tuple[float, float], ...] | None:
    values = tuple(points)
    if len(values) < 2:
        return None
    center_x = sum(point.x for point in values) / len(values)
    center_y = sum(point.y for point in values) / len(values)
    centered = tuple((point.x - center_x, point.y - center_y) for point in values)
    scale = math.sqrt(sum(x * x + y * y for x, y in centered) / len(centered))
    if scale <= 1e-12:
        return None
    return tuple((x / scale, y / scale) for x, y in centered)


def _configuration_distance(
    first: tuple[tuple[float, float], ...], second: tuple[tuple[float, float], ...]
) -> float:
    dot = sum(ax * bx + ay * by for (ax, ay), (bx, by) in zip(first, second))
    cross = sum(ax * by - ay * bx for (ax, ay), (bx, by) in zip(first, second))
    angle = math.atan2(cross, dot)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return math.sqrt(
        sum(
            (cosine * ax - sine * ay - bx) ** 2
            + (sine * ax + cosine * ay - by) ** 2
            for (ax, ay), (bx, by) in zip(first, second)
        )
        / len(first)
    )


def _bounded_deficit(observed: float, allowed: float) -> float:
    return max(0.0, min(1.0, 1.0 - observed / allowed))


def evaluate_embodiment(
    sequence: EmbodimentSequence,
    config: EmbodimentEvaluationConfig = EmbodimentEvaluationConfig(),
) -> EmbodimentScorecard:
    """Evaluate topology and target-consistent articulated motion.

    Scores use the worst observed frame/window rather than an average, so a
    short terminal failure cannot be masked by an otherwise coherent sequence.
    """

    required = tuple(
        sorted(
            set(config.required_landmarks)
            | {
                endpoint
                for link in sequence.links
                for endpoint in (link.start_landmark, link.end_landmark)
            }
        )
    )
    unknown_required = set(required) - set(sequence.landmark_names)
    if unknown_required:
        raise ValueError("required landmark is not declared by the sequence")

    component_counts = tuple(len(frame.components) for frame in sequence.frames)
    connected_counts = tuple(
        sum(component.connected_components for component in frame.components)
        for frame in sequence.frames
    )
    areas = tuple(
        sum(component.area for component in frame.components) for frame in sequence.frames
    )
    topology = (
        1.0
        if all(
            count == config.expected_components and connected == config.expected_components
            for count, connected in zip(component_counts, connected_counts)
        )
        else 0.0
    )

    if any(area == 0 for area in areas):
        area_stability = 0.0
    else:
        reference_area = areas[0]
        maximum_area_change = max(abs(area - reference_area) / reference_area for area in areas)
        area_stability = _bounded_deficit(maximum_area_change, config.maximum_area_relative_change)

    maps = tuple(_landmark_map(frame) for frame in sequence.frames)
    missing = tuple(tuple(name for name in required if name not in points) for points in maps)
    landmark_tracking = 1.0 if not required or all(not names for names in missing) else 0.0

    link_drifts: list[tuple[str, float]] = []
    geometry = 1.0
    for link in sequence.links:
        lengths: list[float | None] = []
        for points in maps:
            start, end = points.get(link.start_landmark), points.get(link.end_landmark)
            lengths.append(math.dist(start.xy, end.xy) if start and end else None)
        reference = link.expected_length
        if reference is None:
            reference = next((length for length in lengths if length is not None), None)
        if reference is None or reference <= 1e-12 or any(length is None for length in lengths):
            drift = math.inf
        else:
            drift = max(
                abs(length - reference) / reference for length in lengths if length is not None
            )
        link_drifts.append((link.name, drift))
        geometry = min(
            geometry,
            0.0
            if not math.isfinite(drift)
            else _bounded_deficit(drift, config.maximum_link_relative_drift),
        )

    target_ids = tuple(frame.target_id for frame in sequence.frames)
    known_ids = tuple(target_id for target_id in target_ids if target_id is not None)
    target_identity = (
        1.0
        if not config.require_target_identity
        or (len(known_ids) == len(target_ids) and len(set(known_ids)) == 1)
        else 0.0
    )

    displacements: tuple[float, ...] = ()
    sustained_displacements: tuple[float, ...] = ()
    if not config.expect_articulation:
        articulation = 1.0
    elif len(required) < 2 or any(names for names in missing):
        articulation = 0.0
    else:
        configurations = tuple(
            _relative_configuration(points[name] for name in required) for points in maps
        )
        if any(configuration is None for configuration in configurations):
            articulation = 0.0
        else:
            valid = tuple(
                configuration for configuration in configurations if configuration is not None
            )
            displacements = tuple(
                _configuration_distance(previous, current)
                for previous, current in zip(valid, valid[1:])
            )
            window_offset = config.articulation_window - 1
            sustained_displacements = tuple(
                _configuration_distance(valid[index], valid[index + window_offset])
                for index in range(len(valid) - window_offset)
            )
            articulation = min(
                1.0,
                (
                    sorted(sustained_displacements)[(len(sustained_displacements) - 1) // 2]
                    if sustained_displacements
                    else 0.0
                )
                / config.minimum_articulation_displacement,
            )

    diagnostics = EmbodimentDiagnostics(
        component_counts=component_counts,
        connected_component_counts=connected_counts,
        component_areas=areas,
        missing_landmarks=missing,
        link_relative_drifts=tuple(link_drifts),
        target_ids=target_ids,
        articulation_displacements=displacements,
        sustained_articulation_displacements=sustained_displacements,
    )
    values = (topology, area_stability, landmark_tracking, geometry, target_identity, articulation)
    return EmbodimentScorecard(*values, essential=min(values), diagnostics=diagnostics)


# Descriptive aliases for integrations that use "target embodiment" terminology.
RobotMask = ImageMask
TargetEmbodimentFrame = EmbodimentFrame
TargetEmbodimentSequence = EmbodimentSequence
TargetEmbodimentConfig = EmbodimentEvaluationConfig
evaluate_target_embodiment = evaluate_embodiment
