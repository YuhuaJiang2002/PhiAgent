"""License-isolated importer for official FoundationPose 4x4 pose outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from phiagent.perception.geometry import rotation_matrix_to_quaternion
from phiagent.perception.schema import ObjectObservation
from phiagent.physical_language.schema import FrameKind, FrameRef, PoseSE3


class FoundationPoseOutputReader:
    """Read object-in-camera matrices without importing restricted NVIDIA code."""

    def load(
        self,
        pose_directory: Path,
        timestamps_s: Iterable[float],
        object_id: str,
        camera_frame: FrameRef,
        confidence: float,
        state: str = "tracked",
    ) -> tuple[ObjectObservation, ...]:
        if camera_frame.kind is not FrameKind.CAMERA:
            raise ValueError("FoundationPose output target must be a camera frame")
        paths = sorted(pose_directory.glob("*.txt"))
        timestamps = tuple(float(value) for value in timestamps_s)
        if len(paths) != len(timestamps):
            raise ValueError(
                "FoundationPose file count and timestamp count differ: "
                f"{len(paths)} != {len(timestamps)}"
            )
        object_frame = FrameRef(FrameKind.OBJECT, object_id)
        observations = []
        for path, timestamp in zip(paths, timestamps):
            values = tuple(float(value) for value in path.read_text().split())
            if len(values) != 16:
                raise ValueError(f"FoundationPose matrix must contain 16 values: {path}")
            matrix = tuple(tuple(values[row * 4 : row * 4 + 4]) for row in range(4))
            if any(
                abs(value - expected) > 1e-6
                for value, expected in zip(matrix[3], (0, 0, 0, 1))
            ):
                raise ValueError(f"FoundationPose matrix has an invalid final row: {path}")
            quaternion = rotation_matrix_to_quaternion(row[:3] for row in matrix[:3])
            observations.append(
                ObjectObservation(
                    timestamp_s=timestamp,
                    object_id=object_id,
                    pose=PoseSE3(
                        source_frame=object_frame,
                        target_frame=camera_frame,
                        translation_m=(matrix[0][3], matrix[1][3], matrix[2][3]),
                        quaternion_xyzw=quaternion,
                        confidence=confidence,
                    ),
                    state=state,
                    confidence=confidence,
                )
            )
        return tuple(observations)
