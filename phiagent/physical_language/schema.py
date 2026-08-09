"""Typed, frame-explicit schema for Embodied Physical Language (EPL)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

Vector3 = tuple[float, float, float]
QuaternionXYZW = tuple[float, float, float, float]


def _finite(values: Iterable[float], label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _vector3(values: Iterable[float], label: str) -> Vector3:
    result = _finite(values, label)
    if len(result) != 3:
        raise ValueError(f"{label} must have length 3")
    return result  # type: ignore[return-value]


def _quaternion(values: Iterable[float], label: str) -> QuaternionXYZW:
    result = _finite(values, label)
    if len(result) != 4:
        raise ValueError(f"{label} must have length 4")
    norm = math.sqrt(sum(value * value for value in result))
    if norm < 1e-12:
        raise ValueError(f"{label} cannot be the zero quaternion")
    if abs(norm - 1.0) > 1e-4:
        raise ValueError(f"{label} must be unit length; got norm {norm:.8f}")
    normalized = tuple(value / norm for value in result)
    return normalized  # type: ignore[return-value]


def _confidence(value: float, label: str = "confidence") -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return value


class FrameKind(str, Enum):
    CAMERA = "camera"
    WORLD = "world"
    ROBOT_BASE = "robot_base"
    END_EFFECTOR = "end_effector"
    HUMAN_WRIST = "human_wrist"
    OBJECT = "object"
    CUSTOM = "custom"


@dataclass(frozen=True, order=True)
class FrameRef:
    """A closed frame kind plus an instance name for multi-object scenes."""

    kind: FrameKind
    name: str = "default"

    def __post_init__(self) -> None:
        clean = self.name.strip()
        if not clean:
            raise ValueError("frame name cannot be empty")
        if any(character.isspace() for character in clean):
            raise ValueError("frame name cannot contain whitespace")
        object.__setattr__(self, "name", clean)

    @property
    def key(self) -> str:
        return f"{self.kind.value}:{self.name}"

    @classmethod
    def parse(cls, key: str) -> FrameRef:
        kind, separator, name = key.partition(":")
        if not separator:
            raise ValueError(f"invalid frame key {key!r}; expected kind:name")
        return cls(FrameKind(kind), name)


def _quat_multiply(left: QuaternionXYZW, right: QuaternionXYZW) -> QuaternionXYZW:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    result = (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )
    norm = math.sqrt(sum(value * value for value in result))
    return tuple(value / norm for value in result)  # type: ignore[return-value]


def _quat_rotate(quaternion: QuaternionXYZW, point: Vector3) -> Vector3:
    x, y, z, w = quaternion
    px, py, pz = point
    tx = 2.0 * (y * pz - z * py)
    ty = 2.0 * (z * px - x * pz)
    tz = 2.0 * (x * py - y * px)
    return (
        px + w * tx + (y * tz - z * ty),
        py + w * ty + (z * tx - x * tz),
        pz + w * tz + (x * ty - y * tx),
    )


@dataclass(frozen=True)
class PoseSE3:
    """Rigid transform target_T_source with translation in metres."""

    source_frame: FrameRef
    target_frame: FrameRef
    translation_m: Vector3
    quaternion_xyzw: QuaternionXYZW
    confidence: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "translation_m", _vector3(self.translation_m, "translation_m")
        )
        object.__setattr__(
            self,
            "quaternion_xyzw",
            _quaternion(self.quaternion_xyzw, "quaternion_xyzw"),
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))

    @classmethod
    def identity(cls, frame: FrameRef, confidence: float = 1.0) -> PoseSE3:
        return cls(frame, frame, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), confidence)

    def inverse(self) -> PoseSE3:
        x, y, z, w = self.quaternion_xyzw
        inverse_quaternion = (-x, -y, -z, w)
        negated = tuple(-value for value in self.translation_m)
        inverse_translation = _quat_rotate(
            inverse_quaternion, negated  # type: ignore[arg-type]
        )
        return PoseSE3(
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            translation_m=inverse_translation,
            quaternion_xyzw=inverse_quaternion,
            confidence=self.confidence,
        )

    def compose(self, right: PoseSE3) -> PoseSE3:
        """Return self @ right, checking target/source frame continuity."""

        if self.source_frame != right.target_frame:
            raise ValueError(
                "cannot compose transforms: "
                f"left source {self.source_frame.key} != "
                f"right target {right.target_frame.key}"
            )
        rotated = _quat_rotate(self.quaternion_xyzw, right.translation_m)
        translation = tuple(
            left + right_value for left, right_value in zip(self.translation_m, rotated)
        )
        return PoseSE3(
            source_frame=right.source_frame,
            target_frame=self.target_frame,
            translation_m=translation,  # type: ignore[arg-type]
            quaternion_xyzw=_quat_multiply(
                self.quaternion_xyzw, right.quaternion_xyzw
            ),
            confidence=min(self.confidence, right.confidence),
        )

    def transform_point(self, point: Point3D) -> Point3D:
        if point.frame != self.source_frame:
            raise ValueError(
                f"point is in {point.frame.key}, transform source is {self.source_frame.key}"
            )
        rotated = _quat_rotate(self.quaternion_xyzw, point.xyz_m)
        translated = tuple(
            left + right for left, right in zip(self.translation_m, rotated)
        )
        return Point3D(
            frame=self.target_frame,
            xyz_m=translated,  # type: ignore[arg-type]
            confidence=min(self.confidence, point.confidence),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame": self.source_frame.key,
            "target_frame": self.target_frame.key,
            "translation_m": list(self.translation_m),
            "quaternion_xyzw": list(self.quaternion_xyzw),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PoseSE3:
        return cls(
            source_frame=FrameRef.parse(str(payload["source_frame"])),
            target_frame=FrameRef.parse(str(payload["target_frame"])),
            translation_m=tuple(payload["translation_m"]),
            quaternion_xyzw=tuple(payload["quaternion_xyzw"]),
            confidence=float(payload.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class Point3D:
    frame: FrameRef
    xyz_m: Vector3
    confidence: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "xyz_m", _vector3(self.xyz_m, "xyz_m"))
        object.__setattr__(self, "confidence", _confidence(self.confidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame.key,
            "xyz_m": list(self.xyz_m),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Point3D:
        return cls(
            frame=FrameRef.parse(str(payload["frame"])),
            xyz_m=tuple(payload["xyz_m"]),
            confidence=float(payload.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class MotionSE3:
    """Relative motion expressed in one named frame."""

    expressed_in: FrameRef
    translation_m: Vector3
    quaternion_xyzw: QuaternionXYZW

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "translation_m", _vector3(self.translation_m, "translation_m")
        )
        object.__setattr__(
            self,
            "quaternion_xyzw",
            _quaternion(self.quaternion_xyzw, "quaternion_xyzw"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "expressed_in": self.expressed_in.key,
            "translation_m": list(self.translation_m),
            "quaternion_xyzw": list(self.quaternion_xyzw),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MotionSE3:
        return cls(
            expressed_in=FrameRef.parse(str(payload["expressed_in"])),
            translation_m=tuple(payload["translation_m"]),
            quaternion_xyzw=tuple(payload["quaternion_xyzw"]),
        )


class ManipulationPhase(str, Enum):
    IDLE = "idle"
    APPROACH = "approach"
    PREGRASP = "pregrasp"
    GRASP = "grasp"
    MANIPULATE = "manipulate"
    RELEASE = "release"
    RETRACT = "retract"


class ContactState(str, Enum):
    NONE = "none"
    CANDIDATE = "candidate"
    STABLE = "stable"
    SLIPPING = "slipping"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.subject.strip() or not self.predicate.strip() or not self.object.strip():
            raise ValueError("relation subject, predicate, and object must be non-empty")
        object.__setattr__(self, "confidence", _confidence(self.confidence))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Relation:
        return cls(
            subject=str(payload["subject"]),
            predicate=str(payload["predicate"]),
            object=str(payload["object"]),
            confidence=float(payload["confidence"]),
        )


@dataclass(frozen=True)
class EPLChunk:
    """Continuous EPL state for one non-empty temporal interval."""

    start_s: float
    end_s: float
    phase: ManipulationPhase
    eef_delta: MotionSE3
    wrist_pose: PoseSE3
    fingertips: tuple[Point3D, Point3D, Point3D, Point3D, Point3D]
    hand_aperture_m: float
    hand_articulation: tuple[float, ...]
    contact_state: ContactState
    contact_points: tuple[Point3D, ...]
    object_pose: PoseSE3 | None
    object_delta: MotionSE3 | None
    object_state_changes: tuple[str, ...]
    relations: tuple[Relation, ...]
    confidence: float

    def __post_init__(self) -> None:
        start, end = float(self.start_s), float(self.end_s)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("EPL chunk requires finite 0 <= start_s < end_s")
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)
        if len(self.fingertips) != 5:
            raise ValueError("EPL chunk requires exactly five fingertips")
        aperture = float(self.hand_aperture_m)
        if not math.isfinite(aperture) or aperture < 0:
            raise ValueError("hand_aperture_m must be finite and non-negative")
        object.__setattr__(self, "hand_aperture_m", aperture)
        articulation = _finite(self.hand_articulation, "hand_articulation")
        object.__setattr__(self, "hand_articulation", articulation)
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        if self.object_pose is None and self.object_delta is not None:
            raise ValueError("object_delta cannot be present without object_pose")

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "phase": self.phase.value,
            "eef_delta": self.eef_delta.to_dict(),
            "wrist_pose": self.wrist_pose.to_dict(),
            "fingertips": [point.to_dict() for point in self.fingertips],
            "hand_aperture_m": self.hand_aperture_m,
            "hand_articulation": list(self.hand_articulation),
            "contact_state": self.contact_state.value,
            "contact_points": [point.to_dict() for point in self.contact_points],
            "object_pose": self.object_pose.to_dict() if self.object_pose else None,
            "object_delta": self.object_delta.to_dict() if self.object_delta else None,
            "object_state_changes": list(self.object_state_changes),
            "relations": [asdict(relation) for relation in self.relations],
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EPLChunk:
        fingertips = tuple(Point3D.from_dict(point) for point in payload["fingertips"])
        if len(fingertips) != 5:
            raise ValueError("serialized EPL chunk requires exactly five fingertips")
        return cls(
            start_s=float(payload["start_s"]),
            end_s=float(payload["end_s"]),
            phase=ManipulationPhase(str(payload["phase"])),
            eef_delta=MotionSE3.from_dict(payload["eef_delta"]),
            wrist_pose=PoseSE3.from_dict(payload["wrist_pose"]),
            fingertips=fingertips,  # type: ignore[arg-type]
            hand_aperture_m=float(payload["hand_aperture_m"]),
            hand_articulation=tuple(payload.get("hand_articulation", [])),
            contact_state=ContactState(str(payload["contact_state"])),
            contact_points=tuple(
                Point3D.from_dict(point) for point in payload.get("contact_points", [])
            ),
            object_pose=(
                PoseSE3.from_dict(payload["object_pose"])
                if payload.get("object_pose") is not None
                else None
            ),
            object_delta=(
                MotionSE3.from_dict(payload["object_delta"])
                if payload.get("object_delta") is not None
                else None
            ),
            object_state_changes=tuple(payload.get("object_state_changes", [])),
            relations=tuple(
                Relation.from_dict(relation) for relation in payload.get("relations", [])
            ),
            confidence=float(payload["confidence"]),
        )


@dataclass(frozen=True)
class EPLSequence:
    schema_version: str
    source_video: str
    chunks: tuple[EPLChunk, ...]
    conventions: str = (
        "right-handed; translation metres; quaternion XYZW; transforms target_T_source"
    )

    def __post_init__(self) -> None:
        if self.schema_version != "0.1.0":
            raise ValueError(f"unsupported EPL schema version {self.schema_version!r}")
        if not self.source_video:
            raise ValueError("source_video cannot be empty")
        previous_end = 0.0
        for index, chunk in enumerate(self.chunks):
            if index and chunk.start_s < previous_end - 1e-9:
                raise ValueError("EPL chunks must be sorted and non-overlapping")
            previous_end = chunk.end_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_video": self.source_video,
            "conventions": self.conventions,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EPLSequence:
        return cls(
            schema_version=str(payload["schema_version"]),
            source_video=str(payload["source_video"]),
            conventions=str(payload.get("conventions", "")),
            chunks=tuple(EPLChunk.from_dict(chunk) for chunk in payload["chunks"]),
        )

    @classmethod
    def from_json(cls, path: Path) -> EPLSequence:
        return cls.from_dict(json.loads(path.read_text()))
