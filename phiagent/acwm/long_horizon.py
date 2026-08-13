"""Frame-explicit long-horizon action contracts for windowed AC-WM video.

This module is deliberately lightweight.  It validates language/action state
across overlapping video-model windows without importing a video framework,
NumPy, PyTorch, or CUDA.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


_SAFE_LABEL = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


@dataclass(frozen=True)
class LongActionPhase:
    """One half-open action phase in seconds with explicit object state."""

    name: str
    start_s: float
    end_s: float
    description: str
    entry_object_holder: str
    exit_object_holder: str

    def __post_init__(self) -> None:
        if not _SAFE_LABEL.fullmatch(self.name):
            raise ValueError("phase name must be filesystem-safe lowercase text")
        if not all(math.isfinite(value) for value in (self.start_s, self.end_s)):
            raise ValueError("phase times must be finite")
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("phase interval must have positive duration")
        if not self.description.strip():
            raise ValueError("phase description must be non-empty")
        for value in (self.entry_object_holder, self.exit_object_holder):
            if not _SAFE_LABEL.fullmatch(value):
                raise ValueError(
                    "object-holder states must be explicit filesystem-safe lowercase text"
                )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LongActionPhase":
        return cls(
            name=str(payload["name"]),
            start_s=float(payload["start_s"]),
            end_s=float(payload["end_s"]),
            description=str(payload["description"]),
            entry_object_holder=str(payload["entry_object_holder"]),
            exit_object_holder=str(payload["exit_object_holder"]),
        )


@dataclass(frozen=True)
class LongActionWindow:
    """One model window in a shared absolute camera timeline."""

    index: int
    start_frame: int
    frame_count: int
    fps: int
    entry_object_holder: str
    exit_object_holder: str
    active_phases: tuple[str, ...]
    timeline: str
    coordinate_frame: str
    object_name: str

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.frame_count

    @property
    def start_s(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_s(self) -> float:
        return self.end_frame / self.fps

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "end_frame_exclusive": self.end_frame,
            "start_s": self.start_s,
            "end_s_exclusive": self.end_s,
            "coordinate_frame": self.coordinate_frame,
            "object_name": self.object_name,
        }


@dataclass(frozen=True)
class LongHorizonAction:
    """A complete action whose phase and object state survive model windows."""

    label: str
    instruction: str
    duration_s: float
    phases: tuple[LongActionPhase, ...]

    def __post_init__(self) -> None:
        if not _SAFE_LABEL.fullmatch(self.label):
            raise ValueError("action label must be filesystem-safe lowercase text")
        if not self.instruction.strip():
            raise ValueError("action instruction must be non-empty")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("action duration must be finite and positive")
        if not self.phases:
            raise ValueError("long action requires at least one phase")
        tolerance = 1e-6
        if abs(self.phases[0].start_s) > tolerance:
            raise ValueError("the first phase must start at zero")
        if abs(self.phases[-1].end_s - self.duration_s) > tolerance:
            raise ValueError("the final phase must end at the action duration")
        for previous, current in zip(self.phases, self.phases[1:]):
            if abs(previous.end_s - current.start_s) > tolerance:
                raise ValueError("action phases must be contiguous without gaps or overlaps")
            if previous.exit_object_holder != current.entry_object_holder:
                raise ValueError(
                    "object-holder state must be continuous across action phases"
                )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LongHorizonAction":
        phases = payload.get("phases")
        if not isinstance(phases, list):
            raise ValueError("long action requires a phases list")
        return cls(
            label=str(payload["label"]),
            instruction=str(payload["instruction"]),
            duration_s=float(payload["duration_s"]),
            phases=tuple(LongActionPhase.from_dict(item) for item in phases),
        )

    def object_holder_at(self, time_s: float, *, after: bool = False) -> str:
        if not math.isfinite(time_s) or not 0 <= time_s <= self.duration_s:
            raise ValueError("state query time is outside the action")
        if abs(time_s - self.duration_s) <= 1e-9:
            return self.phases[-1].exit_object_holder
        for phase in self.phases:
            if phase.start_s <= time_s < phase.end_s:
                if after and abs(time_s - phase.start_s) <= 1e-9:
                    return phase.entry_object_holder
                return phase.entry_object_holder
        raise AssertionError("action phase lookup failed")

    def compile_windows(
        self,
        *,
        total_frames: int,
        fps: int,
        window_frames: int,
        overlap_frames: int,
        coordinate_frame: str = "camera:source_pixels",
        object_name: str = "object",
    ) -> tuple[LongActionWindow, ...]:
        if total_frames <= 0 or fps <= 0 or window_frames <= 0:
            raise ValueError("frame counts and fps must be positive")
        if not 0 <= overlap_frames < window_frames:
            raise ValueError("overlap must be in [0, window_frames)")
        if abs(total_frames / fps - self.duration_s) > 1e-6:
            raise ValueError("frame timeline does not match the action duration")
        if total_frames < window_frames:
            raise ValueError("long action must span at least one complete model window")
        stride = window_frames - overlap_frames
        starts = list(range(0, total_frames - window_frames + 1, stride))
        final_start = total_frames - window_frames
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        windows = []
        for index, start in enumerate(starts):
            end = start + window_frames
            start_s = start / fps
            end_s = end / fps
            phases = tuple(
                phase
                for phase in self.phases
                if phase.start_s < end_s and phase.end_s > start_s
            )
            local_timeline = "; ".join(
                (
                    f"absolute {phase.start_s:.3f}-{phase.end_s:.3f} s: "
                    f"{phase.description} [object {phase.entry_object_holder}"
                    f" -> {phase.exit_object_holder}]"
                )
                for phase in phases
            )
            entry_holder = self.object_holder_at(start_s)
            exit_holder = self.object_holder_at(min(self.duration_s, end_s))
            state_prefix = (
                f"This is window {index + 1}/{len(starts)} at absolute "
                f"{start_s:.3f}-{end_s:.3f} s. Enter with the {object_name} held or "
                f"supported by {entry_holder}; leave with it held or supported by "
                f"{exit_holder}. Do not reset the robot, {object_name}, grasp, object "
                "state, or camera at the window boundary. "
            )
            windows.append(
                LongActionWindow(
                    index=index,
                    start_frame=start,
                    frame_count=window_frames,
                    fps=fps,
                    entry_object_holder=entry_holder,
                    exit_object_holder=exit_holder,
                    active_phases=tuple(phase.name for phase in phases),
                    timeline=state_prefix + local_timeline,
                    coordinate_frame=coordinate_frame,
                    object_name=object_name,
                )
            )
        for previous, current in zip(windows, windows[1:]):
            if current.start_frame >= previous.end_frame:
                raise ValueError("long action windows must overlap")
        return tuple(windows)


@dataclass(frozen=True)
class LongHorizonActionSet:
    schema_version: str
    scene: str
    coordinate_frame: str
    object_name: str
    actions: tuple[LongHorizonAction, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported long-action schema")
        if not self.scene.strip():
            raise ValueError("long-action scene must be non-empty")
        if not self.coordinate_frame.startswith("camera:"):
            raise ValueError("long visual action controls require a named camera frame")
        if not _SAFE_LABEL.fullmatch(self.object_name):
            raise ValueError("long-action object_name must be filesystem-safe lowercase text")
        if not self.actions:
            raise ValueError("long-action set must contain actions")
        labels = [action.label for action in self.actions]
        if len(labels) != len(set(labels)):
            raise ValueError("long-action labels must be unique")
        durations = {action.duration_s for action in self.actions}
        if len(durations) != 1:
            raise ValueError("matched action comparison requires equal durations")

    @classmethod
    def load(cls, path: Path) -> "LongHorizonActionSet":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or not isinstance(payload.get("actions"), list):
            raise ValueError("long-action manifest must contain an actions list")
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            scene=str(payload.get("scene", "")),
            coordinate_frame=str(payload.get("coordinate_frame", "")),
            object_name=str(payload.get("object_name", "object")),
            actions=tuple(LongHorizonAction.from_dict(item) for item in payload["actions"]),
        )

    @property
    def duration_s(self) -> float:
        return self.actions[0].duration_s

    def compile_matched_windows(
        self,
        *,
        total_frames: int,
        fps: int,
        window_frames: int,
        overlap_frames: int,
    ) -> tuple[tuple[LongActionWindow, ...], ...]:
        result = tuple(
            action.compile_windows(
                total_frames=total_frames,
                fps=fps,
                window_frames=window_frames,
                overlap_frames=overlap_frames,
                coordinate_frame=self.coordinate_frame,
                object_name=self.object_name,
            )
            for action in self.actions
        )
        geometry = {
            tuple((item.start_frame, item.frame_count) for item in windows)
            for windows in result
        }
        if len(geometry) != 1:
            raise ValueError("matched actions compiled to different window geometry")
        return result


def window_action_manifest(
    actions: Sequence[LongHorizonAction],
    windows: Sequence[LongActionWindow],
) -> dict[str, object]:
    if len(actions) != len(windows):
        raise ValueError("one compiled window is required per action")
    indices = {window.index for window in windows}
    if len(indices) != 1:
        raise ValueError("window action manifest requires a shared window index")
    return {
        "schema_version": "1.0.0",
        "actions": [
            {
                "label": action.label,
                "instruction": action.instruction,
                "timeline": window.timeline,
                "long_horizon_state": window.to_dict(),
            }
            for action, window in zip(actions, windows)
        ],
    }
