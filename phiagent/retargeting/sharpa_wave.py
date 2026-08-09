"""Geometry-based retargeting from 21-point hand observations to Sharpa Wave."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from phiagent.data.schema import EmbodimentDescriptor, RobotTrajectory
from phiagent.perception.schema import HandObservation, PerceptionSequence
from phiagent.retargeting.base import RetargetingResult

Vector3 = tuple[float, float, float]

SHARPA_WAVE_JOINT_SUFFIXES = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)


def _sub(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _unit(vector: Vector3, label: str) -> Vector3:
    norm = math.sqrt(_dot(vector, vector))
    if norm < 1e-8:
        raise ValueError(f"cannot retarget degenerate hand geometry: {label}")
    return tuple(value / norm for value in vector)  # type: ignore[return-value]


def _bend(first: Vector3, middle: Vector3, last: Vector3) -> float:
    incoming = _unit(_sub(middle, first), "zero-length incoming finger segment")
    outgoing = _unit(_sub(last, middle), "zero-length outgoing finger segment")
    return math.acos(max(-1.0, min(1.0, _dot(incoming, outgoing))))


def _mcp_angles(
    wrist: Vector3,
    mcp: Vector3,
    pip: Vector3,
    palm_normal: Vector3,
) -> tuple[float, float]:
    proximal = _unit(_sub(pip, mcp), "zero-length proximal finger segment")
    reference = _unit(_sub(mcp, wrist), "MCP coincides with wrist")
    normal_component = _dot(proximal, palm_normal)
    flexion = math.atan2(
        abs(normal_component),
        math.sqrt(max(0.0, 1.0 - normal_component * normal_component)),
    )
    projected = _sub(
        proximal,
        tuple(normal_component * value for value in palm_normal),  # type: ignore[arg-type]
    )
    projected = _unit(projected, "proximal segment is normal to palm")
    abduction = math.atan2(
        _dot(_cross(reference, projected), palm_normal),
        max(-1.0, min(1.0, _dot(reference, projected))),
    )
    return flexion, abduction


def load_sharpa_wave_embodiment(
    model_xml: Path, side: str = "right"
) -> EmbodimentDescriptor:
    """Load the named 22-DOF contract from an official Sharpa Wave MJCF file."""

    if side not in {"left", "right"}:
        raise ValueError("Sharpa Wave side must be 'left' or 'right'")
    if not model_xml.is_file():
        raise ValueError(f"Sharpa Wave model does not exist: {model_xml}")
    root = ET.parse(model_xml).getroot()
    joints = {element.get("name"): element for element in root.findall(".//joint")}
    names = tuple(f"{side}_{suffix}" for suffix in SHARPA_WAVE_JOINT_SUFFIXES)
    lower: list[float] = []
    upper: list[float] = []
    for name in names:
        element = joints.get(name)
        if element is None:
            raise ValueError(f"Sharpa Wave model is missing joint {name!r}")
        raw_range = element.get("range", "").split()
        if len(raw_range) != 2:
            raise ValueError(f"Sharpa Wave joint {name!r} needs a two-value range")
        low, high = (float(value) for value in raw_range)
        lower.append(low)
        upper.append(high)
    return EmbodimentDescriptor(
        name=f"sharpa_wave_{side}_22dof",
        joint_names=names,
        lower_limits_rad=tuple(lower),
        upper_limits_rad=tuple(upper),
        end_effector_frame=f"{side}_hand_C_MC",
        urdf_path=str(model_xml.resolve()),
    )


class SharpaWaveRetargeter:
    """Map frame-invariant hand angles to the official Sharpa Wave joint order."""

    def __init__(self, embodiment: EmbodimentDescriptor, side: str = "right") -> None:
        if side not in {"left", "right"}:
            raise ValueError("Sharpa Wave side must be 'left' or 'right'")
        expected = tuple(f"{side}_{suffix}" for suffix in SHARPA_WAVE_JOINT_SUFFIXES)
        if embodiment.joint_names != expected:
            raise ValueError("embodiment does not use the official Sharpa Wave joint order")
        self.embodiment = embodiment
        self.side = side

    def _targets(self, hand: HandObservation) -> tuple[float, ...]:
        if hand.wrist_pose.source_frame.name != self.side:
            raise ValueError(
                f"{self.side} Sharpa target received "
                f"{hand.wrist_pose.source_frame.name} hand observations"
            )
        points = tuple(point.xyz_m for point in hand.keypoints_3d)
        wrist = points[0]
        across = _unit(_sub(points[5], points[17]), "index and pinky MCP coincide")
        forward_seed = _unit(_sub(points[9], wrist), "middle MCP coincides with wrist")
        palm_normal = _unit(_cross(across, forward_seed), "palm basis is collinear")

        thumb_fe, thumb_aa = _mcp_angles(wrist, points[1], points[2], palm_normal)
        values = [
            thumb_fe,
            thumb_aa,
            _bend(points[1], points[2], points[3]),
            0.0,
            _bend(points[2], points[3], points[4]),
        ]
        for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16)):
            flexion, abduction = _mcp_angles(
                wrist, points[mcp], points[pip], palm_normal
            )
            values.extend(
                (
                    flexion,
                    abduction,
                    _bend(points[mcp], points[pip], points[dip]),
                    _bend(points[pip], points[dip], points[tip]),
                )
            )
        pinky_flexion, pinky_abduction = _mcp_angles(
            wrist, points[17], points[18], palm_normal
        )
        values.extend(
            (
                0.0,
                pinky_flexion,
                pinky_abduction,
                _bend(points[17], points[18], points[19]),
                _bend(points[18], points[19], points[20]),
            )
        )
        return tuple(
            max(low, min(high, value))
            for value, low, high in zip(
                values,
                self.embodiment.lower_limits_rad,
                self.embodiment.upper_limits_rad,
            )
        )

    def retarget(self, observations: PerceptionSequence) -> RetargetingResult:
        if len(observations.hands) < 2:
            raise ValueError("Sharpa Wave retargeting requires at least two hand frames")
        trajectory = RobotTrajectory(
            schema_version="0.1.0",
            embodiment=self.embodiment,
            timestamps_s=tuple(hand.timestamp_s for hand in observations.hands),
            joint_positions_rad=tuple(
                self._targets(hand) for hand in observations.hands
            ),
        )
        return RetargetingResult(trajectory, ())
