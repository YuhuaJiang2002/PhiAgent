"""Apply a material-only edit through a tracked hand-instance mask."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from phiagent.evaluation.object_instance import (
    NormalizedROI,
    ObjectTrackerConfig,
    RGBFrames,
    route_object_preservation,
)


@dataclass(frozen=True)
class GraphiteHandConfig:
    blue_scale: float = 0.18
    green_scale: float = 0.22
    red_scale: float = 0.25
    opacity: float = 0.94

    def __post_init__(self) -> None:
        values = (self.blue_scale, self.green_scale, self.red_scale, self.opacity)
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("graphite material values must be between zero and one")
        if self.opacity == 0:
            raise ValueError("graphite material opacity must be positive")


@dataclass(frozen=True)
class SudoHandConfig:
    shell_blue: int = 232
    shell_green: int = 235
    shell_red: int = 238
    shading: float = 0.25
    dark_threshold: int = 105
    opacity: float = 0.94

    def __post_init__(self) -> None:
        channels = (self.shell_blue, self.shell_green, self.shell_red)
        if any(not 0 <= value <= 255 for value in channels):
            raise ValueError("Sudo shell channels must be between 0 and 255")
        if not 0 <= self.shading <= 1 or not 0 < self.opacity <= 1:
            raise ValueError("Sudo shading/opacity values must be in range")
        if not 0 <= self.dark_threshold <= 255:
            raise ValueError("Sudo dark threshold must be between 0 and 255")


@dataclass(frozen=True)
class SudoRobotConfig(SudoHandConfig):
    dark_threshold: int = 60
    eye_radius_fraction: float = 0.012

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.eye_radius_fraction < 0.1:
            raise ValueError("Sudo eye radius fraction must be in (0, 0.1)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _apply_hand_style(
    *,
    candidate: Path,
    hand_mask: Path,
    output: Path,
    object_roi: NormalizedROI,
    config: GraphiteHandConfig | SudoHandConfig,
    style_name: str,
) -> Path:
    inputs = {
        "candidate": candidate.expanduser().resolve(),
        "hand_mask": hand_mask.expanduser().resolve(),
    }
    for label, path in inputs.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")
    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    if destination.suffix.lower() != ".mp4":
        raise ValueError("styled hand output must be an .mp4 file")

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("tracked hand styling requires OpenCV and NumPy") from exc

    def read_video(path: Path) -> tuple[list[object], float, int, int]:
        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames: list[object] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        if not frames or fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"could not decode video: {path}")
        return frames, fps, width, height

    frames, fps, width, height = read_video(inputs["candidate"])
    masks, mask_fps, mask_width, mask_height = read_video(inputs["hand_mask"])
    if (mask_width, mask_height) != (width, height) or abs(mask_fps - fps) > 1e-3:
        raise ValueError("hand mask dimensions and FPS must match the candidate")
    if len(masks) < len(frames):
        raise ValueError("hand mask has fewer frames than the candidate")

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    output_frames: list[object] = []
    for frame, mask_frame in zip(frames, masks):
        mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )
        alpha = (
            cv2.GaussianBlur(mask, (9, 9), 0).astype(np.float32)[..., None]
            / 255
            * config.opacity
        )
        luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if isinstance(config, GraphiteHandConfig):
            material = np.stack(
                (
                    luminance * config.blue_scale,
                    luminance * config.green_scale,
                    luminance * config.red_scale,
                ),
                axis=2,
            )
        else:
            light = luminance[..., None] * config.shading
            shell = np.array(
                [config.shell_blue, config.shell_green, config.shell_red],
                dtype=np.float32,
            )
            shell_material = np.clip(
                shell * (1 - config.shading) + light,
                0,
                255,
            )
            dark_material = np.stack(
                (
                    luminance * 0.12,
                    luminance * 0.14,
                    luminance * 0.16,
                ),
                axis=2,
            )
            dark_weight = np.clip(
                (config.dark_threshold + 25 - luminance) / 50,
                0,
                1,
            )[..., None]
            material = (
                shell_material * (1 - dark_weight) + dark_material * dark_weight
            )
        styled = np.clip(
            frame.astype(np.float32) * (1 - alpha) + material * alpha,
            0,
            255,
        ).astype(np.uint8)
        writer.write(styled)
        output_frames.append(styled)
    writer.release()

    source_rgb = RGBFrames(
        tuple(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes() for frame in frames),
        width,
        height,
    )
    output_rgb = RGBFrames(
        tuple(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()
            for frame in output_frames
        ),
        width,
        height,
    )
    route = route_object_preservation(
        source_rgb,
        output_rgb,
        ObjectTrackerConfig(initial_roi=object_roi),
        maximum_candidate_area_ratio=3.0,
    )
    metadata = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": (
            f"sam2_hand_instance_{style_name}_material_edit_"
            "with_object_confidence_routing"
        ),
        "config": asdict(config),
        "object_roi": asdict(object_roi),
        "route": asdict(route),
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in inputs.items()
        },
        "output": {"path": str(destination), "sha256": _sha256(destination)},
    }
    metadata_path = destination.with_suffix(".style.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata_path


def apply_graphite_hand_style(
    *,
    candidate: Path,
    hand_mask: Path,
    output: Path,
    object_roi: NormalizedROI,
    config: GraphiteHandConfig,
) -> Path:
    return _apply_hand_style(
        candidate=candidate,
        hand_mask=hand_mask,
        output=output,
        object_roi=object_roi,
        config=config,
        style_name="graphite",
    )


def apply_sudo_hand_style(
    *,
    candidate: Path,
    hand_mask: Path,
    output: Path,
    object_roi: NormalizedROI,
    config: SudoHandConfig,
) -> Path:
    return _apply_hand_style(
        candidate=candidate,
        hand_mask=hand_mask,
        output=output,
        object_roi=object_roi,
        config=config,
        style_name="sudo_r1_inspired",
    )


def apply_sudo_robot_style(
    *,
    candidate: Path,
    robot_mask: Path,
    output: Path,
    object_roi: NormalizedROI,
    config: SudoRobotConfig,
) -> Path:
    metadata_path = _apply_hand_style(
        candidate=candidate,
        hand_mask=robot_mask,
        output=output,
        object_roi=object_roi,
        config=config,
        style_name="sudo_r1_full_robot_inspired",
    )
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Sudo full-robot styling requires OpenCV and NumPy") from exc

    styled_capture = cv2.VideoCapture(str(output))
    mask_capture = cv2.VideoCapture(str(robot_mask))
    fps = float(styled_capture.get(cv2.CAP_PROP_FPS))
    width = int(styled_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(styled_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temporary_video = output.with_suffix(".sudo.tmp.mp4")
    writer = cv2.VideoWriter(
        str(temporary_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    while True:
        frame_ok, frame = styled_capture.read()
        mask_ok, mask_frame = mask_capture.read()
        if not (frame_ok and mask_ok):
            break
        robot = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY) > 127
        head = np.zeros((height, width), np.uint8)
        cv2.ellipse(
            head,
            (round(0.5 * width), round(0.18 * height)),
            (round(0.065 * width), round(0.15 * height)),
            0,
            0,
            360,
            255,
            -1,
        )
        head_region = (head > 0) & robot
        top_cleanup = (
            robot
            & (np.arange(height)[:, None] < round(0.08 * height))
            & ~head_region
        )
        frame[top_cleanup] = (8, 8, 8)
        head_luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        head_shell = np.stack(
            (
                config.shell_blue * 0.72 + head_luminance * 0.28,
                config.shell_green * 0.72 + head_luminance * 0.28,
                config.shell_red * 0.72 + head_luminance * 0.28,
            ),
            axis=2,
        )
        frame[head_region] = np.clip(head_shell[head_region], 0, 255).astype(np.uint8)
        chest = np.zeros((height, width), np.uint8)
        cv2.rectangle(
            chest,
            (round(0.43 * width), round(0.43 * height)),
            (round(0.57 * width), round(0.72 * height)),
            255,
            -1,
        )
        chest = cv2.GaussianBlur(chest, (21, 21), 0)
        chest_alpha = (
            chest.astype(np.float32)[..., None] / 255 * robot[..., None] * 0.78
        )
        frame = np.clip(frame.astype(np.float32) * (1 - chest_alpha), 0, 255).astype(
            np.uint8
        )
        eye_radius = round(config.eye_radius_fraction * width)
        eye_y = round(0.16 * height)
        for eye_x in (round(0.475 * width), round(0.525 * width)):
            cv2.circle(frame, (eye_x, eye_y), eye_radius, (15, 15, 15), -1)
            cv2.circle(
                frame,
                (eye_x - eye_radius // 3, eye_y - eye_radius // 3),
                max(1, eye_radius // 4),
                (90, 90, 90),
                -1,
            )
        writer.write(frame)
    styled_capture.release()
    mask_capture.release()
    writer.release()
    temporary_video.replace(output)

    metadata = json.loads(metadata_path.read_text())
    metadata["method"] = (
        "sam2_full_robot_sudo_r1_inspired_edit_with_object_confidence_routing"
    )
    metadata["output"]["sha256"] = _sha256(output)
    metadata["full_robot_details"] = {
        "white_shell": True,
        "black_joint_preservation": True,
        "dual_camera_eyes": True,
        "black_chest_cavity": True,
        "exact_geometry_claimed": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata_path
