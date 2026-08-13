"""Exact numeric action contracts for BWM video generation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phiagent.acwm.robotwin import BWM_EEF_CHANNELS
from phiagent.acwm.schema import ACWMActionCondition, ActionRepresentation
from phiagent.acwm.worldarena import WORLD_ARENA_EEF_QUATERNION_CHANNELS

BWM_ACTION_FPS = 24.0
BWM_ACTION_FRAMES = 57


@dataclass(frozen=True)
class NumericActionChannel:
    name: str
    arm: str
    quantity: str
    axis: str | None
    unit: str
    minimum: float | None = None
    maximum: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "arm": self.arm,
            "quantity": self.quantity,
            "axis": self.axis,
            "unit": self.unit,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


BWM_EEF_CHANNEL_SPECS = (
    NumericActionChannel("left_eef_pos_x_m", "left", "position", "x", "m"),
    NumericActionChannel("left_eef_pos_y_m", "left", "position", "y", "m"),
    NumericActionChannel("left_eef_pos_z_m", "left", "position", "z", "m"),
    NumericActionChannel(
        "left_eef_rot_euler_x_rad",
        "left",
        "rotation",
        "x",
        "rad",
        minimum=-math.pi,
        maximum=math.pi,
    ),
    NumericActionChannel(
        "left_eef_rot_euler_y_rad",
        "left",
        "rotation",
        "y",
        "rad",
        minimum=-math.pi,
        maximum=math.pi,
    ),
    NumericActionChannel(
        "left_eef_rot_euler_z_rad",
        "left",
        "rotation",
        "z",
        "rad",
        minimum=-math.pi,
        maximum=math.pi,
    ),
    NumericActionChannel(
        "left_gripper_open",
        "left",
        "gripper",
        None,
        "normalized",
        minimum=-1.0,
        maximum=1.0,
    ),
    NumericActionChannel("right_eef_pos_x_m", "right", "position", "x", "m"),
    NumericActionChannel("right_eef_pos_y_m", "right", "position", "y", "m"),
    NumericActionChannel("right_eef_pos_z_m", "right", "position", "z", "m"),
    NumericActionChannel(
        "right_eef_rot_euler_x_rad",
        "right",
        "rotation",
        "x",
        "rad",
        minimum=-math.pi,
        maximum=math.pi,
    ),
    NumericActionChannel(
        "right_eef_rot_euler_y_rad",
        "right",
        "rotation",
        "y",
        "rad",
        minimum=-math.pi,
        maximum=math.pi,
    ),
    NumericActionChannel(
        "right_eef_rot_euler_z_rad",
        "right",
        "rotation",
        "z",
        "rad",
        minimum=-math.pi,
        maximum=math.pi,
    ),
    NumericActionChannel(
        "right_gripper_open",
        "right",
        "gripper",
        None,
        "normalized",
        minimum=-1.0,
        maximum=1.0,
    ),
)

if tuple(spec.name for spec in BWM_EEF_CHANNEL_SPECS) != BWM_EEF_CHANNELS:
    raise RuntimeError("numeric BWM channel specifications are out of order")

WORLD_ARENA_EEF_QUATERNION_SPECS = (
    NumericActionChannel("left_eef_pos_x_m", "left", "position", "x", "m"),
    NumericActionChannel("left_eef_pos_y_m", "left", "position", "y", "m"),
    NumericActionChannel("left_eef_pos_z_m", "left", "position", "z", "m"),
    NumericActionChannel(
        "left_eef_quaternion_x", "left", "quaternion", "x", "unit", -1.0, 1.0
    ),
    NumericActionChannel(
        "left_eef_quaternion_y", "left", "quaternion", "y", "unit", -1.0, 1.0
    ),
    NumericActionChannel(
        "left_eef_quaternion_z", "left", "quaternion", "z", "unit", -1.0, 1.0
    ),
    NumericActionChannel(
        "left_eef_quaternion_w", "left", "quaternion", "w", "unit", -1.0, 1.0
    ),
    NumericActionChannel("right_eef_pos_x_m", "right", "position", "x", "m"),
    NumericActionChannel("right_eef_pos_y_m", "right", "position", "y", "m"),
    NumericActionChannel("right_eef_pos_z_m", "right", "position", "z", "m"),
    NumericActionChannel(
        "right_eef_quaternion_x", "right", "quaternion", "x", "unit", -1.0, 1.0
    ),
    NumericActionChannel(
        "right_eef_quaternion_y", "right", "quaternion", "y", "unit", -1.0, 1.0
    ),
    NumericActionChannel(
        "right_eef_quaternion_z", "right", "quaternion", "z", "unit", -1.0, 1.0
    ),
    NumericActionChannel(
        "right_eef_quaternion_w", "right", "quaternion", "w", "unit", -1.0, 1.0
    ),
)

if (
    tuple(spec.name for spec in WORLD_ARENA_EEF_QUATERNION_SPECS)
    != WORLD_ARENA_EEF_QUATERNION_CHANNELS
):
    raise RuntimeError("WorldArena quaternion channel specifications are out of order")

_CHANNEL_PROFILES = {
    BWM_EEF_CHANNELS: BWM_EEF_CHANNEL_SPECS,
    WORLD_ARENA_EEF_QUATERNION_CHANNELS: WORLD_ARENA_EEF_QUATERNION_SPECS,
}


def numeric_action_channel_specs(
    channels: Sequence[str],
) -> tuple[NumericActionChannel, ...]:
    try:
        return _CHANNEL_PROFILES[tuple(channels)]
    except KeyError as exc:
        raise ValueError("numeric action channels do not match a supported BWM profile") from exc


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _value_row(
    values: object,
    label: str,
    *,
    specs: Sequence[NumericActionChannel] = BWM_EEF_CHANNEL_SPECS,
) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must contain 14 numeric values")
    row = tuple(_number(value, f"{label}[{index}]") for index, value in enumerate(values))
    if len(row) != len(BWM_EEF_CHANNELS):
        raise ValueError(f"{label} must contain exactly 14 values")
    for value, spec in zip(row, specs):
        if spec.minimum is not None and value < spec.minimum:
            raise ValueError(f"{spec.name} must be >= {spec.minimum:.9g}")
        if spec.maximum is not None and value > spec.maximum:
            raise ValueError(f"{spec.name} must be <= {spec.maximum:.9g}")
    return row


def _validate_quaternion_rows(
    rows: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-3,
) -> None:
    for frame, row in enumerate(rows):
        for arm, start in (("left", 3), ("right", 10)):
            norm = math.sqrt(sum(float(row[index]) ** 2 for index in range(start, start + 4)))
            if abs(norm - 1.0) > tolerance:
                raise ValueError(
                    f"{arm} EEF quaternion at frame {frame} has norm {norm:.9g}; "
                    f"expected 1 within {tolerance:g}"
                )


def _slerp(
    start: Sequence[float], end: Sequence[float], alpha: float
) -> tuple[float, float, float, float]:
    first = tuple(float(value) for value in start)
    second = tuple(float(value) for value in end)
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm < 1e-12 or second_norm < 1e-12:
        raise ValueError("cannot interpolate a zero-norm EEF quaternion")
    first = tuple(value / first_norm for value in first)
    second = tuple(value / second_norm for value in second)
    dot = sum(left * right for left, right in zip(first, second))
    if dot < 0:
        second = tuple(-value for value in second)
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        result = tuple(left + alpha * (right - left) for left, right in zip(first, second))
        norm = math.sqrt(sum(value * value for value in result))
        normalized = tuple(value / norm for value in result)
        return normalized[0], normalized[1], normalized[2], normalized[3]
    theta = math.acos(dot)
    sine = math.sin(theta)
    left_weight = math.sin((1.0 - alpha) * theta) / sine
    right_weight = math.sin(alpha * theta) / sine
    result = tuple(
        left_weight * left + right_weight * right
        for left, right in zip(first, second)
    )
    return result[0], result[1], result[2], result[3]


@dataclass(frozen=True)
class NumericActionKeyframe:
    frame: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.frame, bool) or not isinstance(self.frame, int) or self.frame < 0:
            raise ValueError("numeric action keyframe indices must be non-negative integers")
        object.__setattr__(self, "values", _value_row(self.values, f"keyframe {self.frame}"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object], *, index: int) -> "NumericActionKeyframe":
        unknown = set(payload) - {"frame", "values"}
        if unknown:
            raise ValueError(f"keyframe {index} has unknown fields: {sorted(unknown)}")
        if "frame" not in payload or "values" not in payload:
            raise ValueError(f"keyframe {index} requires frame and values")
        frame = payload["frame"]
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError(f"keyframe {index} frame must be an integer")
        return cls(frame=frame, values=_value_row(payload["values"], f"keyframe {index} values"))


@dataclass(frozen=True)
class CompiledNumericAction:
    condition: ACWMActionCondition
    prompt: str
    source_mode: str

    @property
    def summary(self) -> dict[str, object]:
        return summarize_numeric_action(self.condition, source_mode=self.source_mode)


def _interpolate_keyframes(
    keyframes: Sequence[NumericActionKeyframe],
    *,
    num_frames: int,
    channels: Sequence[str],
) -> tuple[tuple[float, ...], ...]:
    ordered = tuple(sorted(keyframes, key=lambda item: item.frame))
    if len(ordered) < 2:
        raise ValueError("numeric action interpolation requires at least two keyframes")
    if len({item.frame for item in ordered}) != len(ordered):
        raise ValueError("numeric action keyframe indices must be unique")
    if ordered[0].frame != 0 or ordered[-1].frame != num_frames - 1:
        raise ValueError(
            f"numeric action keyframes must include frames 0 and {num_frames - 1}"
        )
    if any(item.frame >= num_frames for item in ordered):
        raise ValueError(f"numeric action keyframes must lie inside 0..{num_frames - 1}")

    rows: list[tuple[float, ...] | None] = [None] * num_frames
    for start, end in zip(ordered, ordered[1:]):
        span = end.frame - start.frame
        for frame in range(start.frame, end.frame + 1):
            alpha = (frame - start.frame) / span
            rows[frame] = tuple(
                start_value + (end_value - start_value) * alpha
                for start_value, end_value in zip(start.values, end.values)
            )
            if tuple(channels) == WORLD_ARENA_EEF_QUATERNION_CHANNELS:
                interpolated = list(rows[frame])
                interpolated[3:7] = _slerp(start.values[3:7], end.values[3:7], alpha)
                interpolated[10:14] = _slerp(
                    start.values[10:14], end.values[10:14], alpha
                )
                rows[frame] = tuple(interpolated)
    if any(row is None for row in rows):
        raise RuntimeError("numeric action interpolation left an uninitialized frame")
    return tuple(row for row in rows if row is not None)


def _timestamps(*, num_frames: int, fps: float) -> tuple[float, ...]:
    if num_frames < 2:
        raise ValueError("numeric action generation requires at least two frames")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("numeric action FPS must be finite and positive")
    return tuple(index / fps for index in range(num_frames))


def compile_bwm_eef_action(
    *,
    label: str,
    instruction: str,
    prompt: str,
    coordinate_frame: str,
    samples: Sequence[Sequence[object]] | None = None,
    keyframes: Sequence[NumericActionKeyframe] | None = None,
    channels: Sequence[str] = BWM_EEF_CHANNELS,
    num_frames: int = BWM_ACTION_FRAMES,
    fps: float = BWM_ACTION_FPS,
) -> CompiledNumericAction:
    """Compile exact samples or piecewise-linear keyframes into BWM's native contract."""

    normalized_instruction = " ".join(instruction.split())
    normalized_prompt = " ".join(prompt.split())
    if not 8 <= len(normalized_instruction) <= 700:
        raise ValueError("instruction must contain 8-700 characters")
    if not 8 <= len(normalized_prompt) <= 1600:
        raise ValueError("prompt must contain 8-1600 characters")
    if not coordinate_frame.startswith("robot_base:"):
        raise ValueError("numeric BWM EEF actions require a named robot_base frame")
    if (samples is None) == (keyframes is None):
        raise ValueError("provide exactly one of samples or keyframes")
    action_channels = tuple(channels)
    specs = numeric_action_channel_specs(action_channels)

    if samples is not None:
        rows = tuple(
            _value_row(row, f"sample {index}", specs=specs)
            for index, row in enumerate(samples)
        )
        if len(rows) != num_frames:
            raise ValueError(f"exact numeric action samples must contain {num_frames} rows")
        source_mode = "exact_samples"
        timeline = (
            f"{num_frames} exact per-frame EEF samples at {fps:.6g} Hz; "
            "no interpolation or coordinate conversion applied."
        )
    else:
        assert keyframes is not None
        rows = _interpolate_keyframes(
            keyframes,
            num_frames=num_frames,
            channels=action_channels,
        )
        source_mode = (
            "piecewise_linear_position_slerp_quaternion_keyframes"
            if action_channels == WORLD_ARENA_EEF_QUATERNION_CHANNELS
            else "piecewise_linear_keyframes"
        )
        timeline = (
            f"{len(keyframes)} exact EEF keyframes compiled to {num_frames} frames at "
            f"{fps:.6g} Hz by "
            + (
                "piecewise-linear position and quaternion SLERP."
                if action_channels == WORLD_ARENA_EEF_QUATERNION_CHANNELS
                else "channel-wise piecewise-linear interpolation."
            )
        )
        rows = tuple(
            _value_row(row, f"interpolated frame {index}", specs=specs)
            for index, row in enumerate(rows)
        )
    if action_channels == WORLD_ARENA_EEF_QUATERNION_CHANNELS:
        _validate_quaternion_rows(rows)

    condition = ACWMActionCondition(
        label=label,
        instruction=normalized_instruction,
        timeline=timeline,
        representation=ActionRepresentation.EEF_ABSOLUTE,
        coordinate_frame=coordinate_frame,
        timestamps_s=_timestamps(num_frames=num_frames, fps=fps),
        channels=action_channels,
        values=rows,
    )
    return CompiledNumericAction(
        condition=condition,
        prompt=normalized_prompt,
        source_mode=source_mode,
    )


def compile_bwm_eef_payload(
    payload: Mapping[str, Any],
    *,
    label: str,
    coordinate_frame: str,
    channels: Sequence[str] = BWM_EEF_CHANNELS,
    num_frames: int = BWM_ACTION_FRAMES,
    fps: float = BWM_ACTION_FPS,
) -> CompiledNumericAction:
    """Parse the strict JSON payload accepted by the numeric-action job API."""

    unknown = set(payload) - {"instruction", "prompt", "samples", "keyframes"}
    if unknown:
        raise ValueError(f"numeric action request has unknown fields: {sorted(unknown)}")
    instruction = payload.get("instruction")
    if not isinstance(instruction, str):
        raise ValueError("numeric action request requires a string instruction")
    prompt = payload.get("prompt", instruction)
    if not isinstance(prompt, str):
        raise ValueError("numeric action prompt must be a string")

    raw_samples = payload.get("samples")
    raw_keyframes = payload.get("keyframes")
    if (raw_samples is None) == (raw_keyframes is None):
        raise ValueError("numeric action request requires exactly one of samples or keyframes")
    samples: tuple[tuple[float, ...], ...] | None = None
    keyframes: tuple[NumericActionKeyframe, ...] | None = None
    if raw_samples is not None:
        if not isinstance(raw_samples, Sequence) or isinstance(raw_samples, (str, bytes)):
            raise ValueError("samples must be a JSON array")
        samples = tuple(
            _value_row(
                row,
                f"sample {index}",
                specs=numeric_action_channel_specs(channels),
            )
            for index, row in enumerate(raw_samples)
        )
    if raw_keyframes is not None:
        if not isinstance(raw_keyframes, Sequence) or isinstance(
            raw_keyframes, (str, bytes)
        ):
            raise ValueError("keyframes must be a JSON array")
        parsed = []
        for index, item in enumerate(raw_keyframes):
            if not isinstance(item, Mapping):
                raise ValueError(f"keyframe {index} must be a JSON object")
            parsed.append(NumericActionKeyframe.from_dict(item, index=index))
        keyframes = tuple(parsed)

    return compile_bwm_eef_action(
        label=label,
        instruction=instruction,
        prompt=prompt,
        coordinate_frame=coordinate_frame,
        samples=samples,
        keyframes=keyframes,
        channels=channels,
        num_frames=num_frames,
        fps=fps,
    )


def summarize_numeric_action(
    condition: ACWMActionCondition, *, source_mode: str
) -> dict[str, object]:
    if condition.representation is not ActionRepresentation.EEF_ABSOLUTE:
        raise ValueError("numeric action summary requires eef_absolute")
    numeric_action_channel_specs(condition.channels)
    maximum_steps = tuple(
        max(
            abs(current[index] - previous[index])
            for previous, current in zip(condition.values, condition.values[1:])
        )
        for index in range(len(condition.channels))
    )
    return {
        "representation": condition.representation.value,
        "coordinate_frame": condition.coordinate_frame,
        "source_mode": source_mode,
        "frames": len(condition.values),
        "fps": condition.fps,
        "duration_s": condition.timestamps_s[-1],
        "channels": list(condition.channels),
        "start_values": list(condition.values[0]),
        "end_values": list(condition.values[-1]),
        "delta_values": [
            end - start for start, end in zip(condition.values[0], condition.values[-1])
        ],
        "maximum_abs_step": list(maximum_steps),
    }


@dataclass(frozen=True)
class NumericActionStatistics:
    path: Path
    coordinate_frame: str
    channels: tuple[str, ...]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    p01: tuple[float, ...]
    p99: tuple[float, ...]

    def __post_init__(self) -> None:
        numeric_action_channel_specs(self.channels)
        if not self.coordinate_frame.startswith("robot_base:"):
            raise ValueError("numeric action statistics require a named robot_base frame")
        for name, values in (
            ("min", self.minimum),
            ("max", self.maximum),
            ("p01", self.p01),
            ("p99", self.p99),
        ):
            if len(values) != len(self.channels) or any(
                not math.isfinite(value) for value in values
            ):
                raise ValueError(f"numeric action statistics {name} must contain 14 values")
        if any(low > high for low, high in zip(self.minimum, self.maximum)):
            raise ValueError("numeric action statistics require min <= max per channel")

    @classmethod
    def from_json(cls, path: Path) -> "NumericActionStatistics":
        resolved = path.expanduser().resolve()
        if not resolved.is_file() or resolved.stat().st_size == 0:
            raise ValueError(f"numeric action statistics do not exist: {resolved}")
        payload = json.loads(resolved.read_text())
        if not isinstance(payload, Mapping):
            raise ValueError("numeric action statistics must contain one JSON object")
        entry = payload.get("state_pose")
        if not isinstance(entry, Mapping):
            raise ValueError("numeric action statistics require state_pose")
        channels = entry.get("channels")
        if not isinstance(channels, Sequence) or isinstance(channels, (str, bytes)):
            raise ValueError("numeric action statistics require channel names")
        return cls(
            path=resolved,
            coordinate_frame=str(entry.get("coordinate_frame", "")),
            channels=tuple(str(value) for value in channels),
            minimum=tuple(float(value) for value in entry["min"]),
            maximum=tuple(float(value) for value in entry["max"]),
            p01=tuple(float(value) for value in entry.get("p01", entry["min"])),
            p99=tuple(float(value) for value in entry.get("p99", entry["max"])),
        )

    def validate(
        self,
        condition: ACWMActionCondition,
        *,
        require_within_minmax: bool = True,
    ) -> dict[str, object]:
        if condition.coordinate_frame != self.coordinate_frame:
            raise ValueError("action coordinate frame does not match action statistics")
        if tuple(condition.channels) != self.channels:
            raise ValueError("action channels do not match action statistics")
        outside_minmax = []
        outside_percentiles = []
        for frame, row in enumerate(condition.values):
            for channel, value in enumerate(row):
                if value < self.minimum[channel] or value > self.maximum[channel]:
                    outside_minmax.append((frame, channel, value))
                if value < self.p01[channel] or value > self.p99[channel]:
                    outside_percentiles.append((frame, channel, value))
        if require_within_minmax and outside_minmax:
            frame, channel, value = outside_minmax[0]
            raise ValueError(
                f"action frame {frame} channel {self.channels[channel]}={value:.9g} "
                "lies outside matching training min/max"
            )
        total = len(condition.values) * len(condition.channels)
        return {
            "path": str(self.path),
            "coordinate_frame": self.coordinate_frame,
            "channels": list(self.channels),
            "outside_minmax_count": len(outside_minmax),
            "outside_p01_p99_count": len(outside_percentiles),
            "outside_p01_p99_fraction": len(outside_percentiles) / total,
        }
