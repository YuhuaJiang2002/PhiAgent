#!/usr/bin/env python3
"""Export per-frame source human hand and arm corridors from MediaPipe pose."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.minimax_h3 import file_sha256  # noqa: E402


LANDMARK_IDS = tuple(range(11, 23))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state() -> dict[str, object]:
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "available": status.returncode == 0,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
        "error": status.stderr.strip() if status.returncode != 0 else None,
    }


def _interpolate_tracks(np: Any, tracks: Any) -> tuple[Any, int]:
    result = tracks.copy()
    frame_axis = np.arange(len(result), dtype=np.float32)
    missing_frames = int(np.count_nonzero(~np.isfinite(result).all(axis=(1, 2))))
    for joint in range(result.shape[1]):
        for coordinate in range(2):
            values = result[:, joint, coordinate]
            valid = np.isfinite(values)
            if not np.any(valid):
                raise RuntimeError(f"pose landmark {LANDMARK_IDS[joint]} was never detected")
            result[:, joint, coordinate] = np.interp(
                frame_axis,
                frame_axis[valid],
                values[valid],
            )
    return result, missing_frames


def _draw_capsule(cv2: Any, mask: Any, start: Any, end: Any, radius: int) -> None:
    a = tuple(round(float(value)) for value in start)
    b = tuple(round(float(value)) for value in end)
    cv2.line(mask, a, b, 255, radius * 2, cv2.LINE_AA)
    cv2.circle(mask, a, radius, 255, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(mask, b, radius, 255, cv2.FILLED, cv2.LINE_AA)


def _render_masks(
    cv2: Any,
    np: Any,
    tracks: Any,
    *,
    height: int,
    width: int,
    hand_radius: int,
    forearm_radius: int,
    upper_arm_radius: int,
) -> tuple[Any, Any, Any]:
    hands = np.zeros((len(tracks), height, width), dtype=np.uint8)
    forearms = np.zeros_like(hands)
    arms = np.zeros_like(hands)
    joint = {landmark_id: index for index, landmark_id in enumerate(LANDMARK_IDS)}
    for frame_index, points in enumerate(tracks):
        for elbow, wrist, pinky, index_tip, thumb in (
            (13, 15, 17, 19, 21),
            (14, 16, 18, 20, 22),
        ):
            for endpoint in (pinky, index_tip, thumb):
                _draw_capsule(
                    cv2,
                    hands[frame_index],
                    points[joint[wrist]],
                    points[joint[endpoint]],
                    hand_radius,
                )
            _draw_capsule(
                cv2,
                forearms[frame_index],
                points[joint[elbow]],
                points[joint[wrist]],
                forearm_radius,
            )
        arms[frame_index] = cv2.bitwise_or(
            hands[frame_index], forearms[frame_index]
        )
        for shoulder, elbow in ((11, 13), (12, 14)):
            _draw_capsule(
                cv2,
                arms[frame_index],
                points[joint[shoulder]],
                points[joint[elbow]],
                upper_arm_radius,
            )
    return hands > 0, forearms > 0, arms > 0


def _pack(np: Any, masks: Any) -> Any:
    return np.packbits(masks.reshape(len(masks), -1), axis=1, bitorder="little")


def _review_sheet(
    cv2: Any,
    np: Any,
    *,
    video: Path,
    hands: Any,
    forearms: Any,
    arms: Any,
    count: int,
) -> Any:
    indices = np.linspace(0, len(hands) - 1, count, dtype=np.int32)
    capture = cv2.VideoCapture(str(video))
    frames: dict[int, Any] = {}
    target_set = set(int(index) for index in indices)
    index = 0
    while target_set:
        ok, frame = capture.read()
        if not ok:
            break
        if index in target_set:
            frames[index] = frame
            target_set.remove(index)
        index += 1
    capture.release()
    rows = []
    for index in indices:
        frame = frames[int(index)]
        overlay = frame.copy()
        overlay[arms[index]] = np.rint(
            0.38 * overlay[arms[index]] + 0.62 * np.asarray([255, 80, 40])
        ).astype(np.uint8)
        overlay[forearms[index]] = np.rint(
            0.30 * overlay[forearms[index]] + 0.70 * np.asarray([40, 210, 255])
        ).astype(np.uint8)
        overlay[hands[index]] = np.rint(
            0.25 * overlay[hands[index]] + 0.75 * np.asarray([40, 40, 255])
        ).astype(np.uint8)
        cv2.putText(
            overlay,
            f"frame {int(index)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rows.append(overlay)
    columns = 4
    row_groups = [
        rows[offset : offset + columns]
        for offset in range(0, len(rows), columns)
    ]
    while len(row_groups[-1]) < columns:
        row_groups[-1].append(np.zeros_like(rows[0]))
    return cv2.vconcat([cv2.hconcat(group) for group in row_groups])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hand-radius", type=int, default=15)
    parser.add_argument("--forearm-radius", type=int, default=18)
    parser.add_argument("--upper-arm-radius", type=int, default=20)
    parser.add_argument("--review-frame-count", type=int, default=28)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


def main() -> int:
    args = _parser().parse_args()
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    source = args.source_video.expanduser().resolve()
    pose_model = args.pose_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "assets").mkdir()
    (output_dir / "review").mkdir()
    (output_dir / "logs").mkdir()
    if not source.is_file() or not pose_model.is_file():
        raise FileNotFoundError("source video and pose model must exist")
    if min(args.hand_radius, args.forearm_radius, args.upper_arm_radius) <= 0:
        raise ValueError("all limb radii must be positive")

    command = " ".join(shlex.quote(value) for value in sys.argv)
    (output_dir / "command.sh").write_text(command + "\n")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.20,
        min_pose_presence_confidence=0.20,
        min_tracking_confidence=0.20,
        output_segmentation_masks=False,
    )
    tracks = np.full((frame_count, len(LANDMARK_IDS), 2), np.nan, dtype=np.float32)
    detected = 0
    capture = cv2.VideoCapture(str(source))
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = landmarker.detect_for_video(image, round(index * 1000.0 / fps))
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                tracks[index] = np.asarray(
                    [
                        (landmarks[item].x * width, landmarks[item].y * height)
                        for item in LANDMARK_IDS
                    ],
                    dtype=np.float32,
                )
                detected += 1
            index += 1
    capture.release()
    if index != frame_count:
        frame_count = index
        tracks = tracks[:index]
    tracks, interpolated_frames = _interpolate_tracks(np, tracks)
    hands, forearms, arms = _render_masks(
        cv2,
        np,
        tracks,
        height=height,
        width=width,
        hand_radius=args.hand_radius,
        forearm_radius=args.forearm_radius,
        upper_arm_radius=args.upper_arm_radius,
    )

    masks_path = output_dir / "assets" / "source-pose-limb-masks-packed.npz"
    np.savez_compressed(
        masks_path,
        hands_packed=_pack(np, hands),
        forearms_packed=_pack(np, forearms),
        arms_packed=_pack(np, arms),
        landmarks_xy=tracks,
        landmark_ids=np.asarray(LANDMARK_IDS, dtype=np.int32),
        height=np.asarray(height, dtype=np.int32),
        width=np.asarray(width, dtype=np.int32),
        bitorder=np.asarray("little"),
    )
    review = _review_sheet(
        cv2,
        np,
        video=source,
        hands=hands,
        forearms=forearms,
        arms=arms,
        count=args.review_frame_count,
    )
    review_path = output_dir / "review" / "source-pose-limb-mask-review.jpg"
    cv2.imwrite(str(review_path), review, [cv2.IMWRITE_JPEG_QUALITY, 93])

    fractions = {
        "hands_mean": float(np.mean(hands)),
        "hands_min": float(np.min(np.mean(hands, axis=(1, 2)))),
        "hands_max": float(np.max(np.mean(hands, axis=(1, 2)))),
        "forearms_mean": float(np.mean(forearms)),
        "arms_mean": float(np.mean(arms)),
    }
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "scope": "source human hand/arm negative-mask layer generation",
        "command": command,
        "seed": args.seed,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform()},
        "gpu": {"required": False, "selected_physical_gpu": None},
        "git": _git_state(),
        "packages": {
            name: _package_version(name)
            for name in ("mediapipe", "numpy", "opencv-python-headless")
        },
        "inputs": {
            "source_video": {"path": str(source), "sha256": file_sha256(source)},
            "pose_model": {"path": str(pose_model), "sha256": file_sha256(pose_model)},
        },
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "frames": frame_count,
        },
        "method": {
            "landmarks": list(LANDMARK_IDS),
            "hand_radius": args.hand_radius,
            "forearm_radius": args.forearm_radius,
            "upper_arm_radius": args.upper_arm_radius,
            "coordinate_frame": "camera:source_video_pixels",
            "use": (
                "negative constraint for source-flower instance segmentation; "
                "not a direct final-frame erasure mask"
            ),
        },
        "metrics": {
            "pose_detected_frames": detected,
            "pose_interpolated_frames": interpolated_frames,
            **fractions,
        },
        "outputs": {
            "packed_masks": {
                "path": str(masks_path),
                "sha256": file_sha256(masks_path),
            },
            "review": {"path": str(review_path), "sha256": file_sha256(review_path)},
        },
        "honest_status": (
            "PARTIAL: deterministic limb corridors were exported; flower-instance "
            "segmentation and end-to-end video acceptance remain pending."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    summary = {"output_dir": str(output_dir), **manifest["metrics"]}
    (output_dir / "logs" / "run.log").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
