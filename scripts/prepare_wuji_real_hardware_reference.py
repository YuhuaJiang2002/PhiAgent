#!/usr/bin/env python3
"""Prepare hash-bound real-Wuji appearance inputs for video replacement.

The hardware pixels come from an explicitly licensed upstream recording.  This
script only extracts and crops those pixels; it does not render a robot model or
generate a synthetic reference image.  Heavy video dependencies remain behind
the command-line entry point so importing :mod:`phiagent` stays lightweight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UPSTREAM_REPOSITORY = "https://github.com/wuji-technology/wuji-hand-teleop"
UPSTREAM_REVISION = "647801345a6a27dec5cbf56280ce63bb8b2f6a32"
UPSTREAM_MEDIA_PATH = "docs/teleop-demo.mp4"
UPSTREAM_LICENSE = "MIT"
SOURCE_FRAME_ZERO_HAND_ENVELOPE = (605, 153, 891, 400)
SOURCE_SCENE_REFERENCE_PLACEMENT = (610, 150, 270, 238)
DEFAULT_PROMPT = (
    "主体外观：真实相机拍摄的物理 WUJI 舞姬灵巧手，黑色柔性表皮的完整五指，"
    "白色工业机器人前臂，真实材料纹理、接缝、螺钉、线缆、光照和相机噪声。"
    "严格只替换输入中的人手与前臂，五指逐帧遵循输入手势。背景：保持原始显示器、"
    "桌面、物体、构图、相机和光照完全稳定。不要仿真渲染，不要 CGI，不要蓝色金属"
    "骨架，不要人类皮肤，不要额外手指或额外手臂。"
)


def box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    """Return IoU for two ``left, top, right, bottom`` camera-pixel boxes."""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_crop(value: str) -> tuple[int, int, int, int]:
    fields = value.split(":")
    if len(fields) != 4:
        raise ValueError("crop must be WIDTH:HEIGHT:X:Y")
    try:
        width, height, x, y = (int(field) for field in fields)
    except ValueError as error:
        raise ValueError("crop fields must be integers") from error
    if width <= 0 or height <= 0 or x < 0 or y < 0:
        raise ValueError("crop dimensions must be positive and offsets non-negative")
    return width, height, x, y


def extract_reference_filter(
    *, frame_index: int, crop: tuple[int, int, int, int], output_size: tuple[int, int]
) -> str:
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    crop_width, crop_height, crop_x, crop_y = crop
    output_width, output_height = output_size
    if output_width <= 0 or output_height <= 0:
        raise ValueError("output dimensions must be positive")
    return (
        f"select=eq(n\\,{frame_index}),"
        f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},"
        f"scale={output_width}:{output_height}:flags=lanczos"
    )


def build_source_scene_reference(
    cv2: Any, np: Any, frame: Any, clean_plate: Any
) -> tuple[Any, Any]:
    """Place real photographed Wuji hand pixels inside the target scene.

    The image-space masks are fixed for frame 30 of the hash-bound 1080-square
    upstream recording.  Only the black compliant hand and its photographed
    wrist socket are retained.  Deliberately excluding the upstream arm, test
    stand, cables, and background prevents the appearance reference from
    transplanting the laboratory rig into the target scene.
    """

    if frame.shape[:2] != (1080, 1080):
        raise ValueError("source-scene reference expects the pinned 1080x1080 recording")
    height, width = frame.shape[:2]
    probable = np.zeros((height, width), dtype=np.uint8)
    hand_polygon = np.asarray(
        [
            (268, 574),
            (331, 548),
            (392, 554),
            (433, 579),
            (510, 646),
            (513, 688),
            (480, 719),
            (438, 730),
            (414, 714),
            (393, 732),
            (365, 733),
            (345, 717),
            (321, 731),
            (289, 719),
            (265, 692),
            (258, 642),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(probable, [hand_polygon], 255)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[..., 2]
    black_hand = np.where((probable > 0) & (value < 142), 255, 0).astype(np.uint8)
    black_hand = cv2.morphologyEx(
        black_hand,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(black_hand, 8)
    if count < 2:
        raise RuntimeError("real hardware segmentation produced no foreground")
    seed_label = int(labels[620, 365])
    if seed_label == 0:
        seed_label = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    hand = np.where(labels == seed_label, 255, 0).astype(np.uint8)

    # Preserve the physical wrist socket behind the palm.  This is a tight
    # matte, not a rectangular crop: its vertices follow the photographed
    # circular housing and intentionally stop before the robot arm and cables.
    wrist = np.zeros((height, width), dtype=np.uint8)
    wrist_polygon = np.asarray(
        [
            (279, 575),
            (286, 552),
            (311, 535),
            (350, 528),
            (389, 539),
            (422, 562),
            (435, 587),
            (421, 612),
            (395, 598),
            (377, 575),
            (337, 566),
            (306, 578),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(wrist, [wrist_polygon], 255)
    binary = cv2.bitwise_or(hand, wrist)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    x, y, crop_width, crop_height = cv2.boundingRect(binary)
    if crop_width < 180 or crop_height < 150:
        raise RuntimeError("real hardware segmentation is implausibly small")
    subject = frame[y : y + crop_height, x : x + crop_width]
    alpha = binary[y : y + crop_height, x : x + crop_width]
    subject = cv2.rotate(subject, cv2.ROTATE_180)
    alpha = cv2.rotate(alpha, cv2.ROTATE_180)
    left, top, target_width, target_height = SOURCE_SCENE_REFERENCE_PLACEMENT
    subject = cv2.resize(subject, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    alpha = cv2.resize(alpha, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
    plate_height, plate_width = clean_plate.shape[:2]
    # The canonical hardware pixels must occupy frame zero's detected
    # source-camera support.  Placing them in identity-image coordinates makes
    # Animate preserve a second, stationary hand instead of transferring the
    # source motion.
    placed_box = (left, top, left + target_width, top + target_height)
    alignment_iou = box_iou(placed_box, SOURCE_FRAME_ZERO_HAND_ENVELOPE)
    if alignment_iou < 0.75:
        raise RuntimeError(f"source-camera reference alignment failed: IoU={alignment_iou}")
    if left + target_width > plate_width or top + target_height > plate_height:
        raise ValueError("real hardware cutout does not fit the clean plate")
    reference = clean_plate.copy()
    weight = alpha.astype(np.float32)[..., None] / 255.0
    region = reference[top : top + target_height, left : left + target_width]
    reference[top : top + target_height, left : left + target_width] = np.clip(
        np.rint(subject.astype(np.float32) * weight + region * (1.0 - weight)),
        0,
        255,
    ).astype(np.uint8)
    full_alpha = np.zeros((plate_height, plate_width), dtype=np.uint8)
    full_alpha[top : top + target_height, left : left + target_width] = alpha
    return reference, full_alpha


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _video_info(ffprobe: Path, video: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _git_state(root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-video", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--clean-plate", type=Path)
    parser.add_argument("--expected-hardware-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-clean-plate-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=30)
    parser.add_argument("--crop", default="640:360:0:390")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--anchor-frames", type=int, default=80)
    parser.add_argument(
        "--reference-mode",
        choices=("crop", "source_scene"),
        default="source_scene",
    )
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/opt/homebrew/bin/ffprobe"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    hardware_video = args.hardware_video.expanduser().resolve()
    source_video = args.source_video.expanduser().resolve()
    clean_plate_path = (
        args.clean_plate.expanduser().resolve() if args.clean_plate is not None else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    ffprobe = args.ffprobe.expanduser().resolve()
    for label, path in (
        ("hardware video", hardware_video),
        ("source video", source_video),
        ("FFmpeg", ffmpeg),
        ("FFprobe", ffprobe),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} does not exist or is empty: {path}")
    if args.reference_mode == "source_scene":
        if clean_plate_path is None or not clean_plate_path.is_file():
            raise ValueError("source_scene reference mode requires --clean-plate")
        if not args.expected_clean_plate_sha256:
            raise ValueError(
                "source_scene reference mode requires --expected-clean-plate-sha256"
            )
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if args.fps <= 0 or args.anchor_frames <= 0:
        raise ValueError("fps and anchor-frames must be positive")
    crop = parse_crop(args.crop)
    hardware_hash = sha256_file(hardware_video)
    source_hash = sha256_file(source_video)
    if hardware_hash != args.expected_hardware_sha256:
        raise ValueError("hardware-video SHA-256 mismatch")
    if source_hash != args.expected_source_sha256:
        raise ValueError("source-video SHA-256 mismatch")
    clean_plate_hash = (
        sha256_file(clean_plate_path) if clean_plate_path is not None else None
    )
    if (
        args.reference_mode == "source_scene"
        and clean_plate_hash != args.expected_clean_plate_sha256
    ):
        raise ValueError("clean-plate SHA-256 mismatch")

    output_dir.mkdir(parents=True)
    reference = output_dir / "real-wuji-reference.png"
    reference_alpha = output_dir / "real-wuji-reference-alpha.png"
    source_24fps = output_dir / "source-24fps.mp4"
    anchor = output_dir / "source-first80-placeholder.mp4"
    prompt = output_dir / "prompt.txt"
    commands: list[list[str]] = []

    if args.reference_mode == "crop":
        reference_command = [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-i",
            str(hardware_video),
            "-vf",
            extract_reference_filter(
                frame_index=args.frame_index,
                crop=crop,
                output_size=(args.width, args.height),
            ),
            "-frames:v",
            "1",
            str(reference),
        ]
        subprocess.run(reference_command, check=True)
        commands.append(reference_command)
    else:
        import cv2
        import numpy as np

        capture = cv2.VideoCapture(str(hardware_video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"cannot decode hardware frame {args.frame_index}")
        assert clean_plate_path is not None
        clean_plate = cv2.imread(str(clean_plate_path), cv2.IMREAD_COLOR)
        if clean_plate is None:
            raise RuntimeError("cannot decode clean plate")
        scene_reference, full_alpha = build_source_scene_reference(
            cv2, np, frame, clean_plate
        )
        if not cv2.imwrite(str(reference), scene_reference):
            raise RuntimeError("failed to write source-scene hardware reference")
        if not cv2.imwrite(str(reference_alpha), full_alpha):
            raise RuntimeError("failed to write hardware reference alpha")

    source_command = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-i",
        str(source_video),
        "-vf",
        f"fps={args.fps}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "10",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(source_24fps),
    ]
    subprocess.run(source_command, check=True)
    commands.append(source_command)

    anchor_command = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-i",
        str(source_24fps),
        "-vf",
        f"trim=end_frame={args.anchor_frames},setpts=PTS-STARTPTS",
        "-frames:v",
        str(args.anchor_frames),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "10",
        "-pix_fmt",
        "yuv420p",
        str(anchor),
    ]
    subprocess.run(anchor_command, check=True)
    commands.append(anchor_command)
    prompt.write_text(DEFAULT_PROMPT + "\n")

    root = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim": (
            "Real physical Wuji hardware footage supplies the appearance reference; "
            "downstream motion remains model-generated visual synthesis."
        ),
        "hardware_appearance_source": {
            "repository": UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
            "path": UPSTREAM_MEDIA_PATH,
            "permalink": (
                f"{UPSTREAM_REPOSITORY}/blob/{UPSTREAM_REVISION}/{UPSTREAM_MEDIA_PATH}"
            ),
            "license": UPSTREAM_LICENSE,
            "local_path": str(hardware_video),
            "sha256": hardware_hash,
            "frame_index": args.frame_index,
            "reference_mode": args.reference_mode,
            "crop_xywh": (
                [crop[2], crop[3], crop[0], crop[1]]
                if args.reference_mode == "crop"
                else None
            ),
            "clean_plate": str(clean_plate_path) if clean_plate_path else None,
            "clean_plate_sha256": clean_plate_hash,
            "pixel_operation": (
                "exact frame extraction, crop, and Lanczos resize only"
                if args.reference_mode == "crop"
                else "fixed silhouette matte, connected-component selection, "
                "180-degree rotation, deterministic resize, and alpha composite; "
                "no generated pixels"
            ),
            "source_camera_alignment": (
                {
                    "coordinate_frame": "source_camera_frame_0_pixels",
                    "detected_hand_envelope_ltrb": list(
                        SOURCE_FRAME_ZERO_HAND_ENVELOPE
                    ),
                    "reference_placement_xywh": list(
                        SOURCE_SCENE_REFERENCE_PLACEMENT
                    ),
                    "box_iou": box_iou(
                        (
                            SOURCE_SCENE_REFERENCE_PLACEMENT[0],
                            SOURCE_SCENE_REFERENCE_PLACEMENT[1],
                            SOURCE_SCENE_REFERENCE_PLACEMENT[0]
                            + SOURCE_SCENE_REFERENCE_PLACEMENT[2],
                            SOURCE_SCENE_REFERENCE_PLACEMENT[1]
                            + SOURCE_SCENE_REFERENCE_PLACEMENT[3],
                        ),
                        SOURCE_FRAME_ZERO_HAND_ENVELOPE,
                    ),
                    "required_box_iou": 0.75,
                }
                if args.reference_mode == "source_scene"
                else None
            ),
        },
        "source": {
            "path": str(source_video),
            "sha256": source_hash,
            "prepared_path": str(source_24fps),
            "prepared_sha256": sha256_file(source_24fps),
            "info": _video_info(ffprobe, source_24fps),
        },
        "outputs": {
            "reference": str(reference),
            "reference_sha256": sha256_file(reference),
            "reference_alpha": (
                str(reference_alpha) if reference_alpha.is_file() else None
            ),
            "reference_alpha_sha256": (
                sha256_file(reference_alpha) if reference_alpha.is_file() else None
            ),
            "quality_anchor_placeholder": str(anchor),
            "quality_anchor_placeholder_sha256": sha256_file(anchor),
            "prompt": str(prompt),
            "prompt_sha256": sha256_file(prompt),
        },
        "limitations": [
            "The reference is genuine hardware footage, but the replacement video is synthetic.",
            "The upstream hand wears its real black compliant cover; this is not "
            "an exposed-skeleton appearance.",
            "A real appearance reference does not establish hardware execution, "
            "contact force, or metric depth.",
        ],
        "commands": [shlex.join(command) for command in commands],
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git": _git_state(root),
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
