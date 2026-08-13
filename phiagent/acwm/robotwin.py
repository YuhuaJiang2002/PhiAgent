"""Pure RoboTwin-to-BWM data contracts.

The source dataset stores each bimanual end-effector state as two
``xyz + quaternion_xyzw + gripper`` vectors (16 values).  BWM consumes two
``xyz + Euler XYZ + gripper`` vectors (14 values).  Keeping this conversion in
the lightweight PhiAgent package makes the frame and rotation convention
testable without importing PyArrow, Torch, or a simulator.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Sequence

ROBOTWIN_LEROBOT_REPOSITORY = "Traly/Robotwin2.0-lerobot"
ROBOTWIN_LEROBOT_REVISION = "f89cf979382d18f4f281e9c20370dfafbed9d35b"
ROBOTWIN_FPS = 30
ROBOTWIN_EMBODIMENTS = (
    "franka",
    "arx-x5",
    "aloha-agilex",
    "piper",
    "ur5",
)

BWM_EEF_CHANNELS = (
    "left_eef_pos_x_m",
    "left_eef_pos_y_m",
    "left_eef_pos_z_m",
    "left_eef_rot_euler_x_rad",
    "left_eef_rot_euler_y_rad",
    "left_eef_rot_euler_z_rad",
    "left_gripper_open",
    "right_eef_pos_x_m",
    "right_eef_pos_y_m",
    "right_eef_pos_z_m",
    "right_eef_rot_euler_x_rad",
    "right_eef_rot_euler_y_rad",
    "right_eef_rot_euler_z_rad",
    "right_gripper_open",
)


def parse_robotwin_task(text: str) -> tuple[str, str, str]:
    """Return ``(embodiment, task, instruction)`` from a RoboTwin prompt."""

    match = re.fullmatch(r"\[([^]]+)]\s*([^:]+):\s*(.+)", text.strip())
    if match is None:
        raise ValueError(f"invalid RoboTwin task text: {text!r}")
    embodiment, task, instruction = (part.strip() for part in match.groups())
    if embodiment not in ROBOTWIN_EMBODIMENTS:
        raise ValueError(f"unknown RoboTwin embodiment: {embodiment}")
    if not re.fullmatch(r"[a-z0-9_]+", task):
        raise ValueError(f"invalid RoboTwin task identifier: {task}")
    return embodiment, task, instruction


def grouped_split(
    embodiment: str,
    task: str,
    *,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> str:
    """Assign an entire embodiment/task group to one deterministic split.

    Grouping prevents paraphrases and randomized episodes of the same
    embodiment/task pair from appearing in both training and evaluation.
    """

    if embodiment not in ROBOTWIN_EMBODIMENTS:
        raise ValueError(f"unknown RoboTwin embodiment: {embodiment}")
    if seed < 0:
        raise ValueError("split seed must be non-negative")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if not 0 <= validation_fraction < 1 - train_fraction:
        raise ValueError("validation_fraction leaves no test partition")
    key = f"{seed}:{embodiment}:{task}".encode()
    fraction = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    if fraction < train_fraction:
        return "train"
    if fraction < train_fraction + validation_fraction:
        return "validation"
    return "test"


def _quaternion_xyzw_to_euler_xyz(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 4:
        raise ValueError("a quaternion must contain x, y, z, w")
    x, y, z, w = (float(value) for value in values)
    if any(not math.isfinite(value) for value in (x, y, z, w)):
        raise ValueError("quaternion values must be finite")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-8:
        raise ValueError("cannot convert a zero-norm quaternion")
    x, y, z, w = (value / norm for value in (x, y, z, w))

    # Roll-pitch-yaw about X-Y-Z, matching BWM's published Euler channel order.
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch_argument = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, pitch_argument)))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def eef16_to_bwm14(values: Sequence[float]) -> tuple[float, ...]:
    """Convert one absolute bimanual EE pose from RoboTwin to BWM channels."""

    if len(values) != 16:
        raise ValueError(f"RoboTwin EE state must have 16 values, found {len(values)}")
    source = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in source):
        raise ValueError("RoboTwin EE state values must be finite")
    result: list[float] = []
    for offset in (0, 8):
        result.extend(source[offset : offset + 3])
        result.extend(_quaternion_xyzw_to_euler_xyz(source[offset + 3 : offset + 7]))
        gripper = source[offset + 7]
        if not -1e-6 <= gripper <= 1 + 1e-6:
            raise ValueError(f"gripper value is outside [0, 1]: {gripper}")
        result.append(max(0.0, min(1.0, gripper)))
    return tuple(result)


def overlapping_clip_starts(length: int, *, num_frames: int = 57, history: int = 9) -> tuple[int, ...]:
    """Return clip starts with exactly ``history`` frames of adjacent overlap."""

    if num_frames < 2 or not 0 <= history < num_frames:
        raise ValueError("history must satisfy 0 <= history < num_frames")
    if length < num_frames:
        return ()
    stride = num_frames - history
    starts = list(range(0, length - num_frames + 1, stride))
    terminal = length - num_frames
    if starts[-1] != terminal:
        starts.append(terminal)
    return tuple(starts)


@dataclass(frozen=True)
class RoboTwinEpisode:
    episode_index: int
    embodiment: str
    task: str
    instruction: str
    length: int
    data_path: str
    video_path: str
    data_start_frame: int
    video_start_frame: int
    coordinate_frame: str

    def __post_init__(self) -> None:
        if self.episode_index < 0 or self.length <= 0:
            raise ValueError("episode index and length must be positive")
        if self.embodiment not in ROBOTWIN_EMBODIMENTS:
            raise ValueError(f"unknown RoboTwin embodiment: {self.embodiment}")
        if not self.task or not self.instruction:
            raise ValueError("episode task and instruction must be non-empty")
        if self.data_start_frame < 0 or self.video_start_frame < 0:
            raise ValueError("packed-file frame offsets must be non-negative")
        if not self.coordinate_frame.startswith("robot_base:"):
            raise ValueError("absolute EEF poses require a named robot-base frame")
        if not self.data_path.endswith(".parquet") or not self.video_path.endswith(".mp4"):
            raise ValueError("episode paths must point to Parquet actions and MP4 video")

    @property
    def group(self) -> str:
        return f"{self.embodiment}/{self.task}"


def bwm_clip_record(
    episode: RoboTwinEpisode,
    *,
    clip_start: int,
    num_frames: int = 57,
) -> dict[str, object]:
    """Compile one frame-explicit BWM metadata row.

    Video and action offsets stay separate.  The public BWM training script
    currently overwrites the action offsets with video offsets; the PhiAgent
    launcher validates that its compatibility patch is present before training.
    """

    if clip_start < 0 or clip_start + num_frames > episode.length:
        raise ValueError("clip lies outside the episode")
    return {
        "episode_index": episode.episode_index,
        "embodiment": episode.embodiment,
        "task": episode.task,
        "prompt": episode.instruction,
        "coordinate_frame": episode.coordinate_frame,
        "representation": "eef_absolute",
        "channels": list(BWM_EEF_CHANNELS),
        "length": num_frames,
        "video": {
            "data": episode.video_path,
            "start_frame": episode.video_start_frame + clip_start,
            "end_frame": episode.video_start_frame + clip_start + num_frames - 1,
        },
        "action": {
            "data": episode.data_path,
            "start_frame": episode.data_start_frame + clip_start,
            "end_frame": episode.data_start_frame + clip_start + num_frames - 1,
        },
    }
