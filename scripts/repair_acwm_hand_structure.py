#!/usr/bin/env python3
"""Lock an AC-WM robot hand to one canonical 2-D topology and scale."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.acwm.schema import ACWMActionCondition  # noqa: E402
from phiagent.rendering.minimax_h3 import file_sha256  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fixed_scale_transform(
    reference_contact: tuple[float, float],
    reference_elbow: tuple[float, float],
    target_contact: tuple[float, float],
    target_elbow: tuple[float, float],
    *,
    scale: float = 1.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return a source-to-target similarity with an explicit fixed scale."""

    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("hand scale must be finite and positive")
    source_angle = math.atan2(
        reference_elbow[1] - reference_contact[1],
        reference_elbow[0] - reference_contact[0],
    )
    target_angle = math.atan2(
        target_elbow[1] - target_contact[1],
        target_elbow[0] - target_contact[0],
    )
    angle = target_angle - source_angle
    cosine = math.cos(angle) * scale
    sine = math.sin(angle) * scale
    translate_x = target_contact[0] - (cosine * reference_contact[0] - sine * reference_contact[1])
    translate_y = target_contact[1] - (sine * reference_contact[0] + cosine * reference_contact[1])
    return (
        (cosine, -sine, translate_x),
        (sine, cosine, translate_y),
    )


def _video_writer(ffmpeg: str, output: Path, width: int, height: int, fps: float) -> Any:
    return subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--canonical-image", type=Path, required=True)
    parser.add_argument("--canonical-mask", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/acwm-hand-structure"))
    parser.add_argument("--run-id")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--removal-dilation", type=int, default=29)
    parser.add_argument("--inpaint-radius", type=float, default=3.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = {
        "candidate": args.candidate.expanduser().resolve(),
        "condition": args.condition.expanduser().resolve(),
        "canonical_image": args.canonical_image.expanduser().resolve(),
        "canonical_mask": args.canonical_mask.expanduser().resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing structure-repair inputs: {missing}")
    if args.removal_dilation < 3 or args.removal_dilation % 2 == 0:
        raise ValueError("removal dilation must be an odd integer >= 3")
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    )
    experiment = args.output_root.expanduser().resolve() / run_id
    if experiment.exists():
        raise FileExistsError(f"experiment already exists: {experiment}")
    for relative in ("input", "output", "review", "provenance"):
        (experiment / relative).mkdir(parents=True, exist_ok=True)
    frozen_source = experiment / "provenance" / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), frozen_source)

    import cv2
    import numpy as np

    condition = ACWMActionCondition.from_json(paths["condition"])
    channel = {name: index for index, name in enumerate(condition.channels)}
    required_channels = {
        "elbow_x_px",
        "elbow_y_px",
        "wrist_x_px",
        "wrist_y_px",
    }
    if missing_channels := required_channels - channel.keys():
        raise ValueError(f"condition lacks hand-lock channels: {sorted(missing_channels)}")

    capture = cv2.VideoCapture(str(paths["candidate"]))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {paths['candidate']}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if expected_frames != len(condition.values):
        raise ValueError(
            f"candidate has {expected_frames} frames but action has {len(condition.values)}"
        )
    canonical = cv2.imread(str(paths["canonical_image"]), cv2.IMREAD_COLOR)
    canonical_mask = cv2.imread(str(paths["canonical_mask"]), cv2.IMREAD_GRAYSCALE)
    if canonical is None or canonical_mask is None:
        raise RuntimeError("cannot decode canonical hand assets")
    if canonical.shape[:2] != (height, width) or canonical_mask.shape != (height, width):
        raise ValueError("canonical assets and candidate must have identical dimensions")
    canonical_binary = np.where(canonical_mask >= 128, 255, 0).astype(np.uint8)
    canonical_components = cv2.connectedComponents(canonical_binary)[0] - 1
    if canonical_components != 1:
        raise ValueError("canonical hand mask must contain exactly one connected component")
    canonical_alpha = cv2.GaussianBlur(canonical_binary, (5, 5), 0)

    reference = condition.values[0]
    reference_contact = (
        reference[channel["wrist_x_px"]],
        reference[channel["wrist_y_px"]],
    )
    reference_elbow = (
        reference[channel["elbow_x_px"]],
        reference[channel["elbow_y_px"]],
    )
    output_video = experiment / "output" / "structure-locked.mp4"
    writer = _video_writer(args.ffmpeg, output_video, width, height, fps)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (args.removal_dilation, args.removal_dilation),
    )
    frames: list[Any] = []
    hand_areas = []
    component_counts = []
    changed_outside_support = 0
    protected_object_changed = 0
    edit_fractions = []
    transforms = []
    try:
        for frame_index, values in enumerate(condition.values):
            ok, raw = capture.read()
            if not ok:
                raise RuntimeError(f"candidate ended at frame {frame_index}")
            target_contact = (
                values[channel["wrist_x_px"]],
                values[channel["wrist_y_px"]],
            )
            target_elbow = (
                values[channel["elbow_x_px"]],
                values[channel["elbow_y_px"]],
            )
            matrix_tuple = fixed_scale_transform(
                reference_contact,
                reference_elbow,
                target_contact,
                target_elbow,
                scale=args.scale,
            )
            matrix = np.asarray(matrix_tuple, dtype=np.float32)
            warped_hand = cv2.warpAffine(
                canonical,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            warped_alpha = cv2.warpAffine(
                canonical_alpha,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            hard_alpha = np.where(warped_alpha >= 128, 255, 0).astype(np.uint8)
            components = cv2.connectedComponents(hard_alpha)[0] - 1
            if components > 1:
                raise RuntimeError(
                    f"rigid projection broke canonical topology at frame {frame_index}"
                )
            removal = cv2.dilate(hard_alpha, kernel)
            hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
            yellow = cv2.inRange(
                hsv,
                np.asarray((10, 55, 45), dtype=np.uint8),
                np.asarray((48, 255, 255), dtype=np.uint8),
            )
            yellow = cv2.morphologyEx(
                yellow,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            removal_without_object = removal.copy()
            removal_without_object[yellow > 0] = 0
            clean = cv2.inpaint(
                raw,
                removal_without_object,
                args.inpaint_radius,
                cv2.INPAINT_TELEA,
            )
            alpha = warped_alpha.astype(np.float32)[:, :, None] / 255.0
            repaired = np.clip(
                warped_hand.astype(np.float32) * alpha + clean.astype(np.float32) * (1.0 - alpha),
                0,
                255,
            ).astype(np.uint8)
            protected = cv2.dilate(
                yellow,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            repaired[protected > 0] = raw[protected > 0]
            support = np.logical_or(removal > 0, warped_alpha > 0)
            outside = np.logical_not(support)
            changed_outside_support += int(np.count_nonzero(repaired[outside] != raw[outside]))
            protected_object_changed += int(
                np.count_nonzero(repaired[protected > 0] != raw[protected > 0])
            )
            hand_areas.append(int(np.count_nonzero(hard_alpha)))
            component_counts.append(components)
            edit_fractions.append(float(np.mean(repaired != raw)))
            transforms.append(
                {
                    "frame": frame_index,
                    "target_contact_xy": target_contact,
                    "target_elbow_xy": target_elbow,
                    "matrix": matrix.tolist(),
                    "scale": args.scale,
                }
            )
            frames.append(repaired)
            assert writer.stdin is not None
            writer.stdin.write(repaired.tobytes())
    finally:
        capture.release()
        if writer.stdin is not None:
            writer.stdin.close()
        returncode = writer.wait()
    if returncode:
        raise RuntimeError(f"ffmpeg failed with exit {returncode}")

    decoded = cv2.VideoCapture(str(output_video))
    decoded_count = int(decoded.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_fps = float(decoded.get(cv2.CAP_PROP_FPS))
    decoded_width = int(decoded.get(cv2.CAP_PROP_FRAME_WIDTH))
    decoded_height = int(decoded.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded.release()
    if (decoded_count, decoded_width, decoded_height) != (
        expected_frames,
        width,
        height,
    ):
        raise RuntimeError("encoded structure-locked video failed alignment verification")

    samples = np.linspace(0, len(frames) - 1, 12).round().astype(int)
    tiles = []
    for index in samples:
        tile = cv2.resize(frames[index], (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile,
            f"FRAME {index:02d} / TOPOLOGY LOCKED",
            (9, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 245, 190),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    storyboard = np.vstack([np.hstack(tiles[row : row + 4]) for row in range(0, 12, 4)])
    storyboard_path = experiment / "review" / "storyboard.jpg"
    cv2.imwrite(str(storyboard_path), storyboard)
    transform_path = experiment / "output" / "transforms.json"
    _write_json(transform_path, transforms)

    minimum_area = min(hand_areas)
    maximum_area = max(hand_areas)
    area_ratio = float(maximum_area / max(1, minimum_area))
    accepted = (
        max(component_counts) == 1
        and area_ratio <= 1.08
        and changed_outside_support == 0
        and protected_object_changed == 0
        and decoded_fps == fps
    )
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted_structure_lock" if accepted else "rejected_structure_lock",
        "honest_status": "WORKING" if accepted else "PARTIAL",
        "method": "sam2_canonical_hand_fixed_topology_action_projection",
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "gpu": {"used": False, "reason": "deterministic CPU structure projection"},
        "coordinate_frame": condition.coordinate_frame,
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()
        },
        "configuration": {
            "fixed_scale": args.scale,
            "removal_dilation": args.removal_dilation,
            "inpaint_radius": args.inpaint_radius,
            "reference_contact_xy": reference_contact,
            "reference_elbow_xy": reference_elbow,
        },
        "acceptance": {
            "accepted": accepted,
            "canonical_connected_components": canonical_components,
            "maximum_projected_connected_components": max(component_counts),
            "hand_area_min_pixels": minimum_area,
            "hand_area_max_pixels": maximum_area,
            "hand_area_max_over_min": area_ratio,
            "required_area_ratio_maximum": 1.08,
            "fixed_scale_min": args.scale,
            "fixed_scale_max": args.scale,
            "changed_channels_outside_declared_support_preencode": changed_outside_support,
            "changed_protected_object_channels_preencode": protected_object_changed,
            "decoded_frames": decoded_count,
            "decoded_fps": decoded_fps,
        },
        "outputs": {
            "video": {
                "path": str(output_video),
                "sha256": file_sha256(output_video),
            },
            "storyboard": {
                "path": str(storyboard_path),
                "sha256": file_sha256(storyboard_path),
            },
            "transforms": {
                "path": str(transform_path),
                "sha256": file_sha256(transform_path),
            },
        },
        "mean_changed_channel_fraction_preencode": float(np.mean(edit_fractions)),
        "limitations": [
            "The repaired hand is a rigid 2-D canonical projection; finger articulation is intentionally frozen to prevent topology drift.",
            "This improves image-space morphology but does not validate 3-D kinematics, contact force, or real-robot execution.",
            "The original OSCAR output remains the model result; this artifact is explicitly a structure-locked repair.",
        ],
        "execution_source": {
            "path": str(frozen_source),
            "sha256": file_sha256(frozen_source),
        },
    }
    _write_json(experiment / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
