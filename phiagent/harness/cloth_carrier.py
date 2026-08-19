"""Deterministic camera-frame carrier for ordered T-shirt folding.

The carrier is an image-space control signal for a proposal model.  Rigid
sleeve rotations preserve every frozen material-polyline segment exactly; the
result is not metric 3-D cloth, contact, force, or executable robot evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


Point = tuple[float, float]


def smoothstep(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("smoothstep input must be finite")
    clipped = min(max(value, 0.0), 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def phase_progress(frame: int, start: int, end: int) -> float:
    if frame < 0 or start < 0 or end <= start:
        raise ValueError("carrier frame window is invalid")
    return smoothstep((frame - start) / (end - start))


def rigid_transform_points(
    points: Sequence[Point],
    *,
    pivot: Point,
    angle_degrees: float,
    translation: Point = (0.0, 0.0),
) -> tuple[Point, ...]:
    if not math.isfinite(angle_degrees) or any(
        not math.isfinite(value) for point in (*points, pivot, translation) for value in point
    ):
        raise ValueError("carrier geometry must be finite")
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    px, py = pivot
    tx, ty = translation
    return tuple(
        (
            px + cosine * (x - px) - sine * (y - py) + tx,
            py + sine * (x - px) + cosine * (y - py) + ty,
        )
        for x, y in points
    )


def polyline_segment_lengths(points: Sequence[Point]) -> tuple[float, ...]:
    if len(points) < 2:
        raise ValueError("material polyline requires at least two points")
    return tuple(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(points, points[1:])
    )


@dataclass(frozen=True)
class TshirtCarrierGeometry:
    coordinate_frame: str
    viewer_left_polygon: tuple[Point, ...]
    viewer_right_polygon: tuple[Point, ...]
    body_polygon: tuple[Point, ...]
    viewer_left_material: tuple[Point, ...]
    viewer_right_material: tuple[Point, ...]
    viewer_left_pivot: Point
    viewer_right_pivot: Point
    # OpenCV's camera-pixel rotation convention has y increasing downward.
    # These signs rotate each distal sleeve tip inward over the shirt torso.
    viewer_left_angle_degrees: float = 96.0
    viewer_right_angle_degrees: float = -73.0
    bundle_translation: Point = (-42.0, 0.0)

    def __post_init__(self) -> None:
        if not self.coordinate_frame.startswith("camera:"):
            raise ValueError("cloth carrier geometry requires a named camera frame")
        if min(
            len(self.viewer_left_polygon),
            len(self.viewer_right_polygon),
            len(self.body_polygon),
        ) < 3:
            raise ValueError("carrier layer polygons require at least three points")
        polyline_segment_lengths(self.viewer_left_material)
        polyline_segment_lengths(self.viewer_right_material)

    def sleeve_material_at(self, frame: int) -> dict[str, tuple[Point, ...]]:
        left_progress = phase_progress(frame, 20, 40)
        right_progress = phase_progress(frame, 60, 80)
        move_progress = phase_progress(frame, 111, 121)
        translation = (
            self.bundle_translation[0] * move_progress,
            self.bundle_translation[1] * move_progress,
        )
        return {
            "viewer_left": rigid_transform_points(
                self.viewer_left_material,
                pivot=self.viewer_left_pivot,
                angle_degrees=self.viewer_left_angle_degrees * left_progress,
                translation=translation,
            ),
            "viewer_right": rigid_transform_points(
                self.viewer_right_material,
                pivot=self.viewer_right_pivot,
                angle_degrees=self.viewer_right_angle_degrees * right_progress,
                translation=translation,
            ),
        }


TSHIRT_832X480_CARRIER = TshirtCarrierGeometry(
    coordinate_frame="camera:tshirt_fold_832x480_pixels",
    viewer_left_polygon=(
        (221.0, 192.0),
        (244.0, 164.0),
        (278.0, 132.0),
        (307.0, 116.0),
        (335.0, 135.0),
        (307.0, 158.0),
        (274.0, 191.0),
        (248.0, 220.0),
    ),
    viewer_right_polygon=(
        (353.0, 136.0),
        (374.0, 107.0),
        (397.0, 96.0),
        (414.0, 114.0),
        (455.0, 122.0),
        (431.0, 148.0),
        (391.0, 147.0),
    ),
    body_polygon=(
        (286.0, 139.0),
        (313.0, 121.0),
        (350.0, 132.0),
        (398.0, 139.0),
        (440.0, 153.0),
        (478.0, 181.0),
        (516.0, 207.0),
        (509.0, 231.0),
        (482.0, 263.0),
        (439.0, 285.0),
        (389.0, 277.0),
        (346.0, 251.0),
        (312.0, 225.0),
        (290.0, 187.0),
    ),
    viewer_left_material=(
        (246.0, 184.0),
        (260.0, 176.0),
        (276.0, 166.0),
        (292.0, 153.0),
        (309.0, 141.0),
    ),
    viewer_right_material=(
        (444.0, 124.0),
        (423.0, 126.0),
        (402.0, 130.0),
        (380.0, 134.0),
        (355.0, 136.0),
    ),
    viewer_left_pivot=(309.0, 141.0),
    viewer_right_pivot=(355.0, 136.0),
)


def write_carrier_contract(path: Path, geometry: TshirtCarrierGeometry) -> None:
    """Write candidate-independent material paths for experiment provenance."""

    import json

    baseline = {
        side: list(polyline_segment_lengths(points))
        for side, points in (
            ("viewer_left", geometry.viewer_left_material),
            ("viewer_right", geometry.viewer_right_material),
        )
    }
    terminal = geometry.sleeve_material_at(123)
    terminal_lengths = {
        side: list(polyline_segment_lengths(points)) for side, points in terminal.items()
    }
    payload = {
        "schema_version": "1.0.0",
        "coordinate_frame": geometry.coordinate_frame,
        "method": "rigid_sleeve_rotation_then_body_fold_then_bundle_translation",
        "phase_windows": {
            "viewer_left_sleeve": [20, 40],
            "viewer_right_sleeve": [60, 80],
            "body_fold": [80, 105],
            "bundle_move": [111, 121],
        },
        "baseline_segment_lengths_pixels": baseline,
        "terminal_segment_lengths_pixels": terminal_lengths,
        "claim_boundary": (
            "The carrier preserves declared camera-pixel sleeve segments analytically. "
            "It is a proposal-control signal, not metric 3-D cloth or robot evidence."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
