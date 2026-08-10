"""Backend-independent inputs for action-conditioned video world models.

The contract deliberately separates image-pixel skeletons from robot-base
actions.  A backend may only consume the representation it was trained for;
screen-space paths are never silently relabelled as metric end-effector poses.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class ActionRepresentation(str, Enum):
    KINEMATIC_SKELETON_2D = "kinematic_skeleton_2d"
    EEF_ABSOLUTE = "eef_absolute"
    EEF_DELTA = "eef_delta"
    JOINT_ABSOLUTE = "joint_absolute"
    JOINT_DELTA = "joint_delta"
    ROBOT_POINTMAP = "robot_pointmap"

    @property
    def requires_camera_frame(self) -> bool:
        return self in {
            ActionRepresentation.KINEMATIC_SKELETON_2D,
            ActionRepresentation.ROBOT_POINTMAP,
        }

    @property
    def requires_robot_base_frame(self) -> bool:
        return self in {
            ActionRepresentation.EEF_ABSOLUTE,
            ActionRepresentation.EEF_DELTA,
            ActionRepresentation.JOINT_ABSOLUTE,
            ActionRepresentation.JOINT_DELTA,
        }


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"{label} does not exist or is empty: {resolved}")
    return resolved


@dataclass(frozen=True)
class ACWMActionCondition:
    """One frame-explicit action condition with a named coordinate frame."""

    label: str
    instruction: str
    timeline: str
    representation: ActionRepresentation
    coordinate_frame: str
    timestamps_s: tuple[float, ...]
    channels: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    visual_condition: Path | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", self.label):
            raise ValueError("action label must be filesystem-safe lowercase text")
        if not self.instruction.strip() or not self.timeline.strip():
            raise ValueError("action instruction and timeline must be non-empty")
        if self.representation.requires_camera_frame and not self.coordinate_frame.startswith(
            "camera:"
        ):
            raise ValueError(
                f"{self.representation.value} requires a named camera frame"
            )
        if self.representation.requires_robot_base_frame and not self.coordinate_frame.startswith(
            "robot_base:"
        ):
            raise ValueError(
                f"{self.representation.value} requires a named robot-base frame"
            )
        if len(self.timestamps_s) < 2:
            raise ValueError("an action condition requires at least two frames")
        if any(not math.isfinite(value) or value < 0 for value in self.timestamps_s):
            raise ValueError("action timestamps must be finite and non-negative")
        if any(current <= previous for previous, current in zip(
            self.timestamps_s, self.timestamps_s[1:]
        )):
            raise ValueError("action timestamps must be strictly increasing")
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError("action channels must be non-empty and unique")
        if len(self.values) != len(self.timestamps_s):
            raise ValueError("action values and timestamps must have equal lengths")
        for row in self.values:
            if len(row) != len(self.channels):
                raise ValueError("every action row must match the declared channel count")
            if any(not math.isfinite(value) for value in row):
                raise ValueError("action values must be finite")
        if self.representation is ActionRepresentation.KINEMATIC_SKELETON_2D:
            if self.visual_condition is None:
                raise ValueError("2-D skeleton actions require a skeleton video")
            object.__setattr__(
                self,
                "visual_condition",
                _require_file(self.visual_condition, "skeleton video"),
            )
        elif self.visual_condition is not None:
            object.__setattr__(
                self,
                "visual_condition",
                _require_file(self.visual_condition, "visual action condition"),
            )

    @property
    def fps(self) -> float:
        periods = [
            current - previous
            for previous, current in zip(self.timestamps_s, self.timestamps_s[1:])
        ]
        mean_period = sum(periods) / len(periods)
        if any(abs(period - mean_period) > 1e-4 for period in periods):
            raise ValueError("action timestamps must be uniformly sampled for video generation")
        return 1.0 / mean_period

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        visual: str | None = None
        if self.visual_condition is not None:
            resolved = self.visual_condition.expanduser().resolve()
            visual = (
                os.path.relpath(resolved, relative_to.resolve())
                if relative_to is not None
                else str(resolved)
            )
        return {
            "schema_version": "1.0.0",
            "label": self.label,
            "instruction": self.instruction,
            "timeline": self.timeline,
            "representation": self.representation.value,
            "coordinate_frame": self.coordinate_frame,
            "timestamps_s": list(self.timestamps_s),
            "channels": list(self.channels),
            "values": [list(row) for row in self.values],
            "visual_condition": visual,
        }

    def to_json(self, path: Path) -> None:
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(relative_to=path.parent), indent=2, sort_keys=True)
            + "\n"
        )
        temporary.replace(path)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, base_dir: Path) -> "ACWMActionCondition":
        visual_value = payload.get("visual_condition")
        visual = None
        if visual_value is not None:
            visual = Path(str(visual_value)).expanduser()
            if not visual.is_absolute():
                visual = base_dir / visual
        return cls(
            label=str(payload["label"]),
            instruction=str(payload["instruction"]),
            timeline=str(payload["timeline"]),
            representation=ActionRepresentation(str(payload["representation"])),
            coordinate_frame=str(payload["coordinate_frame"]),
            timestamps_s=tuple(float(value) for value in payload["timestamps_s"]),
            channels=tuple(str(value) for value in payload["channels"]),
            values=tuple(
                tuple(float(value) for value in row) for row in payload["values"]
            ),
            visual_condition=visual,
        )

    @classmethod
    def from_json(cls, path: Path) -> "ACWMActionCondition":
        resolved = _require_file(path, "action condition")
        payload = json.loads(resolved.read_text())
        if not isinstance(payload, dict):
            raise ValueError("action condition JSON must contain one object")
        return cls.from_dict(payload, base_dir=resolved.parent)


@dataclass(frozen=True)
class ACWMCase:
    """A real-scene initial state paired with one counterfactual action."""

    case_id: str
    first_frame: Path
    source_video: Path
    action: ACWMActionCondition
    prompt: str
    auxiliary_inputs: tuple[tuple[str, Path], ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", self.case_id):
            raise ValueError("case_id must be filesystem-safe lowercase text")
        _require_file(self.first_frame, "first frame")
        _require_file(self.source_video, "real-scene source video")
        if not self.prompt.strip():
            raise ValueError("AC-WM prompt must be non-empty")
        keys = [key for key, _ in self.auxiliary_inputs]
        if len(set(keys)) != len(keys) or any(not key.strip() for key in keys):
            raise ValueError("auxiliary input keys must be non-empty and unique")
        for key, path in self.auxiliary_inputs:
            _require_file(path, f"auxiliary input {key}")

    @property
    def assets(self) -> dict[str, Path]:
        return dict(self.auxiliary_inputs)
