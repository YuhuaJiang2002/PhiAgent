"""Deterministic scene routing for localized embodiment replacement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class EntityRole(str, Enum):
    SUBJECT = "subject"
    OBJECT = "object"


class ReplacementGranularity(str, Enum):
    HAND = "hand"
    HAND_FOREARM = "hand_forearm"
    FULL_BODY = "full_body"


@dataclass(frozen=True)
class NormalizedBox:
    """Axis-aligned image box in normalized image coordinates."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("normalized box values must be finite")
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("normalized box must have a non-negative origin and positive size")
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("normalized box must lie inside the image")

    def interpolate(self, other: NormalizedBox, weight: float) -> NormalizedBox:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("interpolation weight must be in [0, 1]")
        return NormalizedBox(
            x=self.x + (other.x - self.x) * weight,
            y=self.y + (other.y - self.y) * weight,
            width=self.width + (other.width - self.width) * weight,
            height=self.height + (other.height - self.height) * weight,
        )

    def intersects(self, other: NormalizedBox) -> bool:
        return not (
            self.x + self.width <= other.x
            or other.x + other.width <= self.x
            or self.y + self.height <= other.y
            or other.y + other.height <= self.y
        )


@dataclass(frozen=True)
class TrackKeyframe:
    frame_index: int
    box: NormalizedBox
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("track frame index must be non-negative")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("track confidence must be finite and in [0, 1]")


@dataclass(frozen=True)
class Shot:
    shot_id: str
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if not self.shot_id.strip():
            raise ValueError("shot ID must be non-empty")
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("shot frame range is invalid")

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame


@dataclass(frozen=True)
class TrackSegment:
    entity_id: str
    shot_id: str
    role: EntityRole
    keyframes: tuple[TrackKeyframe, ...]
    side: str | None = None

    def __post_init__(self) -> None:
        if not self.entity_id.strip() or not self.shot_id.strip():
            raise ValueError("track entity and shot IDs must be non-empty")
        if not self.keyframes:
            raise ValueError("track segment requires at least one keyframe")
        frame_indices = tuple(keyframe.frame_index for keyframe in self.keyframes)
        if tuple(sorted(frame_indices)) != frame_indices or len(set(frame_indices)) != len(
            frame_indices
        ):
            raise ValueError("track keyframes must have unique increasing frame indices")
        if self.side not in {None, "left", "right"}:
            raise ValueError("track side must be left, right, or None")
        if self.role is EntityRole.OBJECT and self.side is not None:
            raise ValueError("object tracks cannot declare handedness")

    def sample(self, frame_index: int, maximum_carry_frames: int) -> TrackKeyframe | None:
        before = [item for item in self.keyframes if item.frame_index <= frame_index]
        after = [item for item in self.keyframes if item.frame_index >= frame_index]
        if before and after:
            left = before[-1]
            right = after[0]
            if left.frame_index == right.frame_index:
                return left
            weight = (frame_index - left.frame_index) / (
                right.frame_index - left.frame_index
            )
            return TrackKeyframe(
                frame_index=frame_index,
                box=left.box.interpolate(right.box, weight),
                confidence=min(left.confidence, right.confidence),
            )
        nearest = before[-1] if before else after[0]
        if abs(frame_index - nearest.frame_index) > maximum_carry_frames:
            return None
        return TrackKeyframe(frame_index, nearest.box, nearest.confidence)


@dataclass(frozen=True)
class ReplacementSpec:
    source_entity_id: str
    target_identity: str
    granularity: ReplacementGranularity

    def __post_init__(self) -> None:
        if not self.source_entity_id.strip() or not self.target_identity.strip():
            raise ValueError("replacement source and target identity must be non-empty")


@dataclass(frozen=True)
class ReplacementOperation:
    source_entity_id: str
    target_identity: str
    granularity: ReplacementGranularity
    side: str | None
    box: NormalizedBox
    confidence: float


@dataclass(frozen=True)
class ProtectedObjectOperation:
    entity_id: str
    box: NormalizedBox
    confidence: float
    overlaps_replacement: bool


@dataclass(frozen=True)
class RouteDiagnostic:
    code: str
    entity_id: str
    message: str


@dataclass(frozen=True)
class FrameReplacementRoute:
    frame_index: int
    shot_id: str
    replacements: tuple[ReplacementOperation, ...]
    protected_objects: tuple[ProtectedObjectOperation, ...]
    diagnostics: tuple[RouteDiagnostic, ...]


@dataclass(frozen=True)
class SceneReplacementPlan:
    shots: tuple[Shot, ...]
    tracks: tuple[TrackSegment, ...]
    replacements: tuple[ReplacementSpec, ...]
    protected_object_ids: tuple[str, ...] = ()
    maximum_carry_frames: int = 2
    minimum_confidence: float = 0.5

    def __post_init__(self) -> None:
        if not self.shots:
            raise ValueError("scene replacement plan requires at least one shot")
        ordered_shots = tuple(sorted(self.shots, key=lambda shot: shot.start_frame))
        if ordered_shots != self.shots:
            raise ValueError("shots must be ordered by start frame")
        for previous, current in zip(self.shots, self.shots[1:]):
            if current.start_frame <= previous.end_frame:
                raise ValueError("shots must not overlap")
        shot_by_id = {shot.shot_id: shot for shot in self.shots}
        if len(shot_by_id) != len(self.shots):
            raise ValueError("shot IDs must be unique")
        if self.maximum_carry_frames < 0:
            raise ValueError("maximum carry frames must be non-negative")
        if not math.isfinite(self.minimum_confidence) or not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum confidence must be finite and in [0, 1]")

        segment_keys: set[tuple[str, str]] = set()
        roles_by_entity: dict[str, EntityRole] = {}
        for track in self.tracks:
            if track.shot_id not in shot_by_id:
                raise ValueError(f"track uses unknown shot: {track.shot_id}")
            key = (track.entity_id, track.shot_id)
            if key in segment_keys:
                raise ValueError(f"duplicate track segment: {track.entity_id}/{track.shot_id}")
            segment_keys.add(key)
            shot = shot_by_id[track.shot_id]
            if any(not shot.contains(item.frame_index) for item in track.keyframes):
                raise ValueError(f"track keyframe lies outside shot: {track.entity_id}")
            previous_role = roles_by_entity.setdefault(track.entity_id, track.role)
            if previous_role is not track.role:
                raise ValueError(f"entity role changes between shots: {track.entity_id}")

        replacement_ids = [item.source_entity_id for item in self.replacements]
        if len(set(replacement_ids)) != len(replacement_ids):
            raise ValueError("each source entity can have only one replacement")
        for entity_id in replacement_ids:
            if roles_by_entity.get(entity_id) is not EntityRole.SUBJECT:
                raise ValueError(f"replacement source is not a tracked subject: {entity_id}")
        if len(set(self.protected_object_ids)) != len(self.protected_object_ids):
            raise ValueError("protected object IDs must be unique")
        for entity_id in self.protected_object_ids:
            if roles_by_entity.get(entity_id) is not EntityRole.OBJECT:
                raise ValueError(f"protected entity is not a tracked object: {entity_id}")

    def route_frame(self, frame_index: int) -> FrameReplacementRoute:
        if frame_index < 0:
            raise ValueError("frame index must be non-negative")
        shot = next((item for item in self.shots if item.contains(frame_index)), None)
        if shot is None:
            raise ValueError(f"frame {frame_index} does not belong to a declared shot")
        segments = {
            track.entity_id: track
            for track in self.tracks
            if track.shot_id == shot.shot_id
        }
        replacements: list[ReplacementOperation] = []
        diagnostics: list[RouteDiagnostic] = []
        for spec in self.replacements:
            segment = segments.get(spec.source_entity_id)
            if segment is None:
                diagnostics.append(
                    RouteDiagnostic(
                        "subject_not_in_shot",
                        spec.source_entity_id,
                        f"{spec.source_entity_id} has no track in shot {shot.shot_id}",
                    )
                )
                continue
            sample = segment.sample(frame_index, self.maximum_carry_frames)
            if sample is None:
                diagnostics.append(
                    RouteDiagnostic(
                        "subject_track_missing",
                        spec.source_entity_id,
                        f"{spec.source_entity_id} has no nearby track at frame {frame_index}",
                    )
                )
                continue
            if sample.confidence < self.minimum_confidence:
                diagnostics.append(
                    RouteDiagnostic(
                        "subject_confidence_low",
                        spec.source_entity_id,
                        f"{spec.source_entity_id} confidence {sample.confidence:.3f} is below "
                        f"{self.minimum_confidence:.3f}",
                    )
                )
                continue
            replacements.append(
                ReplacementOperation(
                    source_entity_id=spec.source_entity_id,
                    target_identity=spec.target_identity,
                    granularity=spec.granularity,
                    side=segment.side,
                    box=sample.box,
                    confidence=sample.confidence,
                )
            )

        protected_objects: list[ProtectedObjectOperation] = []
        for entity_id in self.protected_object_ids:
            segment = segments.get(entity_id)
            sample = (
                segment.sample(frame_index, self.maximum_carry_frames)
                if segment is not None
                else None
            )
            if sample is None:
                diagnostics.append(
                    RouteDiagnostic(
                        "protected_object_missing",
                        entity_id,
                        f"{entity_id} has no nearby track at frame {frame_index}",
                    )
                )
                continue
            if sample.confidence < self.minimum_confidence:
                diagnostics.append(
                    RouteDiagnostic(
                        "protected_object_confidence_low",
                        entity_id,
                        f"{entity_id} confidence {sample.confidence:.3f} is below "
                        f"{self.minimum_confidence:.3f}",
                    )
                )
                continue
            protected_objects.append(
                ProtectedObjectOperation(
                    entity_id=entity_id,
                    box=sample.box,
                    confidence=sample.confidence,
                    overlaps_replacement=any(
                        sample.box.intersects(replacement.box)
                        for replacement in replacements
                    ),
                )
            )
        return FrameReplacementRoute(
            frame_index=frame_index,
            shot_id=shot.shot_id,
            replacements=tuple(replacements),
            protected_objects=tuple(protected_objects),
            diagnostics=tuple(diagnostics),
        )
