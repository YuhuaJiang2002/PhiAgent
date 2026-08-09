"""Backend-independent timestamped perception outputs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from phiagent.physical_language.schema import Point3D, PoseSE3

FINGERTIP_INDICES = (4, 8, 12, 16, 20)


@dataclass(frozen=True)
class HandObservation:
    timestamp_s: float
    wrist_pose: PoseSE3
    keypoints_3d: tuple[Point3D, ...]
    articulation: tuple[float, ...]
    confidence: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("hand timestamp must be finite and non-negative")
        if len(self.keypoints_3d) != 21:
            raise ValueError("hand observation requires exactly 21 keypoints")
        frames = {point.frame for point in self.keypoints_3d}
        if frames != {self.wrist_pose.target_frame}:
            raise ValueError("hand keypoints and wrist pose must share one target frame")
        if not 0 <= self.confidence <= 1 or not math.isfinite(self.confidence):
            raise ValueError("hand confidence must be finite and in [0, 1]")
        if not all(math.isfinite(value) for value in self.articulation):
            raise ValueError("hand articulation must be finite")

    @property
    def fingertips(self) -> tuple[Point3D, Point3D, Point3D, Point3D, Point3D]:
        points = tuple(self.keypoints_3d[index] for index in FINGERTIP_INDICES)
        return points  # type: ignore[return-value]

    @property
    def aperture_m(self) -> float:
        thumb, index = self.fingertips[:2]
        return math.dist(thumb.xyz_m, index.xyz_m)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": self.timestamp_s,
            "wrist_pose": self.wrist_pose.to_dict(),
            "keypoints_3d": [point.to_dict() for point in self.keypoints_3d],
            "articulation": list(self.articulation),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HandObservation:
        return cls(
            timestamp_s=float(payload["timestamp_s"]),
            wrist_pose=PoseSE3.from_dict(payload["wrist_pose"]),
            keypoints_3d=tuple(
                Point3D.from_dict(point) for point in payload["keypoints_3d"]
            ),
            articulation=tuple(payload.get("articulation", [])),
            confidence=float(payload["confidence"]),
        )


@dataclass(frozen=True)
class ObjectObservation:
    timestamp_s: float
    object_id: str
    pose: PoseSE3
    state: str
    confidence: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("object timestamp must be finite and non-negative")
        if not self.object_id.strip() or not self.state.strip():
            raise ValueError("object_id and state must be non-empty")
        if not 0 <= self.confidence <= 1 or not math.isfinite(self.confidence):
            raise ValueError("object confidence must be finite and in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": self.timestamp_s,
            "object_id": self.object_id,
            "pose": self.pose.to_dict(),
            "state": self.state,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObjectObservation:
        return cls(
            timestamp_s=float(payload["timestamp_s"]),
            object_id=str(payload["object_id"]),
            pose=PoseSE3.from_dict(payload["pose"]),
            state=str(payload.get("state", "unknown")),
            confidence=float(payload["confidence"]),
        )


@dataclass(frozen=True)
class PerceptionSequence:
    schema_version: str
    hands: tuple[HandObservation, ...]
    objects: tuple[ObjectObservation | None, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "0.1.0":
            raise ValueError("unsupported perception schema")
        if len(self.hands) < 2:
            raise ValueError("at least two hand observations are required")
        if len(self.objects) != len(self.hands):
            raise ValueError("hand and object observation sequences must be aligned")
        times = tuple(hand.timestamp_s for hand in self.hands)
        if any(current <= previous for previous, current in zip(times, times[1:])):
            raise ValueError("observation timestamps must be strictly increasing")
        for hand, object_observation in zip(self.hands, self.objects):
            if (
                object_observation is not None
                and abs(object_observation.timestamp_s - hand.timestamp_s) > 1e-6
            ):
                raise ValueError("hand and object timestamps must be aligned")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hands": [hand.to_dict() for hand in self.hands],
            "objects": [
                observation.to_dict() if observation is not None else None
                for observation in self.objects
            ],
        }

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PerceptionSequence:
        return cls(
            schema_version=str(payload["schema_version"]),
            hands=tuple(HandObservation.from_dict(hand) for hand in payload["hands"]),
            objects=tuple(
                ObjectObservation.from_dict(observation)
                if observation is not None
                else None
                for observation in payload["objects"]
            ),
        )

    @classmethod
    def from_json(cls, path: Path) -> PerceptionSequence:
        return cls.from_dict(json.loads(path.read_text()))
