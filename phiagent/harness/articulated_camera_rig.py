"""Connected camera-frame robot carriers for the T-shirt folding proposal.

The rigs are synthetic planar image-space mechanisms, not identified real
robots.  They provide complete connected joint trajectories for an H3 motion
condition while preserving an explicit evidence boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence

from phiagent.harness.cloth_carrier import (
    Point,
    TSHIRT_832X480_CARRIER,
    TshirtCarrierGeometry,
    phase_progress,
)


def _distance(left: Point, right: Point) -> float:
    return math.hypot(right[0] - left[0], right[1] - left[1])


def _interpolate(left: Point, right: Point, progress: float) -> Point:
    return (
        left[0] + (right[0] - left[0]) * progress,
        left[1] + (right[1] - left[1]) * progress,
    )


def _normalized(vector: Point) -> Point:
    length = math.hypot(*vector)
    if length <= 1e-12:
        raise ValueError("cannot normalize a zero-length camera-frame vector")
    return vector[0] / length, vector[1] / length


@dataclass(frozen=True)
class PlanarArmRig:
    rig_id: str
    coordinate_frame: str
    reference_nodes_xy: tuple[Point, ...]

    def __post_init__(self) -> None:
        if not self.rig_id.strip():
            raise ValueError("planar rig requires a non-empty identity")
        if not self.coordinate_frame.startswith("camera:"):
            raise ValueError("planar rig requires a named camera frame")
        if len(self.reference_nodes_xy) < 3:
            raise ValueError("planar rig requires at least two connected links")
        if any(
            not all(math.isfinite(value) for value in point)
            for point in self.reference_nodes_xy
        ):
            raise ValueError("planar rig nodes must be finite")
        if any(length <= 1e-6 for length in self.link_lengths_pixels):
            raise ValueError("planar rig links must have positive length")

    @property
    def link_lengths_pixels(self) -> tuple[float, ...]:
        return tuple(
            _distance(left, right)
            for left, right in zip(self.reference_nodes_xy, self.reference_nodes_xy[1:])
        )


@dataclass(frozen=True)
class PlanarRigFrame:
    frame_index: int
    nodes_xy: tuple[Point, ...]
    q_radians: tuple[float, ...]
    qdot_radians_per_second: tuple[float, ...]
    target_tip_xy: Point
    contact_entity: str | None


@dataclass(frozen=True)
class DualArmCarrierTrajectory:
    coordinate_frame: str
    fps: float
    frame_count: int
    rigs: Mapping[str, PlanarArmRig]
    frames: Mapping[str, tuple[PlanarRigFrame, ...]]
    maximum_link_length_error_pixels: float
    maximum_joint_step_radians: float
    maximum_tip_step_pixels: float
    mean_tip_error_pixels: float


LOWER_LEFT_RIG = PlanarArmRig(
    rig_id="synthetic_camera_planar:lower_left_robot",
    coordinate_frame="camera:tshirt_fold_832x480_pixels",
    reference_nodes_xy=(
        (300.0, 468.0),
        (174.0, 395.0),
        (287.0, 339.0),
        (313.0, 305.0),
        (421.0, 280.0),
    ),
)


UPPER_RIGHT_RIG = PlanarArmRig(
    rig_id="synthetic_camera_planar:upper_right_robot",
    coordinate_frame="camera:tshirt_fold_832x480_pixels",
    reference_nodes_xy=(
        (643.0, 353.0),
        (640.0, 146.0),
        (598.0, 124.0),
        (525.0, 194.0),
    ),
)


def solve_fabrik(
    rig: PlanarArmRig,
    target_tip_xy: Point,
    *,
    initial_nodes_xy: Sequence[Point] | None = None,
    tolerance_pixels: float = 1e-7,
    maximum_iterations: int = 64,
) -> tuple[Point, ...]:
    """Solve one fixed-base planar chain while preserving every link length."""

    if tolerance_pixels <= 0 or maximum_iterations < 1:
        raise ValueError("FABRIK tolerances and iteration count must be positive")
    if not all(math.isfinite(value) for value in target_tip_xy):
        raise ValueError("FABRIK target must be finite")
    source = initial_nodes_xy or rig.reference_nodes_xy
    if len(source) != len(rig.reference_nodes_xy):
        raise ValueError("FABRIK initial node count does not match the rig")
    nodes = [[float(x), float(y)] for x, y in source]
    base = rig.reference_nodes_xy[0]
    lengths = rig.link_lengths_pixels
    reach = sum(lengths)
    base_distance = _distance(base, target_tip_xy)
    if base_distance >= reach:
        direction = _normalized(
            (target_tip_xy[0] - base[0], target_tip_xy[1] - base[1])
        )
        nodes[0] = [*base]
        for index, length in enumerate(lengths, start=1):
            nodes[index] = [
                nodes[index - 1][0] + direction[0] * length,
                nodes[index - 1][1] + direction[1] * length,
            ]
        return tuple((point[0], point[1]) for point in nodes)

    for _ in range(maximum_iterations):
        nodes[-1] = [*target_tip_xy]
        for index in range(len(nodes) - 2, -1, -1):
            direction = _normalized(
                (
                    nodes[index][0] - nodes[index + 1][0],
                    nodes[index][1] - nodes[index + 1][1],
                )
            )
            nodes[index] = [
                nodes[index + 1][0] + direction[0] * lengths[index],
                nodes[index + 1][1] + direction[1] * lengths[index],
            ]
        nodes[0] = [*base]
        for index, length in enumerate(lengths):
            direction = _normalized(
                (
                    nodes[index + 1][0] - nodes[index][0],
                    nodes[index + 1][1] - nodes[index][1],
                )
            )
            nodes[index + 1] = [
                nodes[index][0] + direction[0] * length,
                nodes[index][1] + direction[1] * length,
            ]
        if _distance(tuple(nodes[-1]), target_tip_xy) <= tolerance_pixels:
            break
    return tuple((point[0], point[1]) for point in nodes)


def _absolute_angles(nodes: Sequence[Point]) -> tuple[float, ...]:
    return tuple(
        math.atan2(right[1] - left[1], right[0] - left[0])
        for left, right in zip(nodes, nodes[1:])
    )


def _unwrap(previous: Sequence[float], current: Sequence[float]) -> tuple[float, ...]:
    if not previous:
        return tuple(current)
    result = []
    for old, new in zip(previous, current):
        delta = (new - old + math.pi) % (2.0 * math.pi) - math.pi
        result.append(old + delta)
    return tuple(result)


def tshirt_gripper_targets(
    frame: int,
    *,
    geometry: TshirtCarrierGeometry = TSHIRT_832X480_CARRIER,
) -> dict[str, tuple[Point, str | None]]:
    """Compile contact-first tip targets in the frozen T-shirt camera frame."""

    if frame < 0 or frame >= 124:
        raise ValueError("T-shirt carrier frame must be in [0, 123]")
    sleeves = geometry.sleeve_material_at(frame)
    lower_initial = LOWER_LEFT_RIG.reference_nodes_xy[-1]
    upper_initial = UPPER_RIGHT_RIG.reference_nodes_xy[-1]
    lower_cuff_initial = geometry.viewer_left_material[0]
    upper_cuff_initial = geometry.viewer_right_material[0]
    move = phase_progress(frame, 111, 121)
    bundle_dx = geometry.bundle_translation[0] * move

    if frame < 20:
        lower = _interpolate(
            lower_initial,
            lower_cuff_initial,
            phase_progress(frame, 2, 20),
        )
        lower_contact = None
    elif frame <= 42:
        lower = sleeves["viewer_left"][0]
        lower_contact = "viewer_left_sleeve"
    elif frame < 78:
        lower = _interpolate(
            sleeves["viewer_left"][0],
            (350.0, 260.0),
            phase_progress(frame, 42, 78),
        )
        lower_contact = None
    elif frame < 106:
        lower = _interpolate(
            (350.0, 260.0),
            (350.0, 205.0),
            phase_progress(frame, 88, 106),
        )
        lower_contact = "shirt_body"
    else:
        lower = (350.0 + bundle_dx, 205.0)
        lower_contact = "shirt_bundle"

    if frame < 60:
        upper = _interpolate(
            upper_initial,
            upper_cuff_initial,
            phase_progress(frame, 46, 60),
        )
        upper_contact = None
    elif frame <= 82:
        upper = sleeves["viewer_right"][0]
        upper_contact = "viewer_right_sleeve"
    elif frame < 106:
        upper = _interpolate(
            sleeves["viewer_right"][0],
            (470.0, 205.0),
            phase_progress(frame, 82, 106),
        )
        upper_contact = "shirt_body"
    else:
        upper = (470.0 + bundle_dx, 205.0)
        upper_contact = "shirt_bundle"
    return {
        "lower_left": (lower, lower_contact),
        "upper_right": (upper, upper_contact),
    }


def compile_tshirt_dual_arm_trajectory(
    *,
    frame_count: int = 124,
    fps: float = 24.0,
    geometry: TshirtCarrierGeometry = TSHIRT_832X480_CARRIER,
) -> DualArmCarrierTrajectory:
    if frame_count != 124 or fps <= 0:
        raise ValueError("the frozen T-shirt carrier requires 124 frames and positive FPS")
    rigs = {"lower_left": LOWER_LEFT_RIG, "upper_right": UPPER_RIGHT_RIG}
    results: dict[str, list[PlanarRigFrame]] = {name: [] for name in rigs}
    previous_nodes = {
        name: rig.reference_nodes_xy for name, rig in rigs.items()
    }
    previous_q: dict[str, tuple[float, ...]] = {name: () for name in rigs}
    maximum_link_error = 0.0
    maximum_joint_step = 0.0
    maximum_tip_step = 0.0
    tip_errors: list[float] = []
    for frame_index in range(frame_count):
        targets = tshirt_gripper_targets(frame_index, geometry=geometry)
        for name, rig in rigs.items():
            target, contact = targets[name]
            nodes = solve_fabrik(
                rig,
                target,
                initial_nodes_xy=previous_nodes[name],
            )
            q = _unwrap(previous_q[name], _absolute_angles(nodes))
            if results[name]:
                previous_frame = results[name][-1]
                qdot = tuple(
                    (right - left) * fps
                    for left, right in zip(previous_frame.q_radians, q)
                )
                maximum_joint_step = max(
                    maximum_joint_step,
                    max(abs(right - left) for left, right in zip(previous_frame.q_radians, q)),
                )
                maximum_tip_step = max(
                    maximum_tip_step,
                    _distance(previous_frame.nodes_xy[-1], nodes[-1]),
                )
            else:
                qdot = tuple(0.0 for _ in q)
            link_errors = tuple(
                abs(observed - expected)
                for observed, expected in zip(
                    (_distance(left, right) for left, right in zip(nodes, nodes[1:])),
                    rig.link_lengths_pixels,
                )
            )
            maximum_link_error = max(maximum_link_error, *link_errors)
            tip_errors.append(_distance(nodes[-1], target))
            results[name].append(
                PlanarRigFrame(
                    frame_index=frame_index,
                    nodes_xy=nodes,
                    q_radians=q,
                    qdot_radians_per_second=qdot,
                    target_tip_xy=target,
                    contact_entity=contact,
                )
            )
            previous_nodes[name] = nodes
            previous_q[name] = q
    return DualArmCarrierTrajectory(
        coordinate_frame=geometry.coordinate_frame,
        fps=fps,
        frame_count=frame_count,
        rigs=rigs,
        frames={name: tuple(frames) for name, frames in results.items()},
        maximum_link_length_error_pixels=maximum_link_error,
        maximum_joint_step_radians=maximum_joint_step,
        maximum_tip_step_pixels=maximum_tip_step,
        mean_tip_error_pixels=fmean(tip_errors),
    )
