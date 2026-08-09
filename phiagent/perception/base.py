"""Backend contracts for human-hand and object teacher trackers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from phiagent.perception.schema import HandObservation, ObjectObservation
from phiagent.physical_language.schema import FrameRef


class HandTracker(Protocol):
    def track(self, video_path: Path, camera_frame: FrameRef) -> tuple[HandObservation, ...]:
        """Return timestamped 3D hand observations in an explicit camera frame."""


class ObjectTracker(Protocol):
    def track(
        self, video_path: Path, camera_frame: FrameRef
    ) -> tuple[ObjectObservation | None, ...]:
        """Return timestamped object poses, preserving missing observations."""
