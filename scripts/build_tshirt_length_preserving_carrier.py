#!/usr/bin/env python3
"""Build a continuous sleeve-length-preserving H3 control carrier."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.schema import ACWMActionCondition  # noqa: E402
from phiagent.harness.cloth_carrier import (  # noqa: E402
    TSHIRT_832X480_CARRIER,
    phase_progress,
    write_carrier_contract,
)
from phiagent.harness.provenance import capture_provenance, write_json_atomic  # noqa: E402
from phiagent.rendering.minimax_h3 import file_sha256  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-condition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser


def _polygon_mask(cv2, np, shape, points):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    polygon = np.asarray(points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255, lineType=cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (5, 5), 0)


def _warp(cv2, image, mask, matrix, size):
    warped_image = cv2.warpAffine(
        image,
        matrix,
        size,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_image, warped_mask


def _composite(np, canvas, layer, mask):
    alpha = mask.astype(np.float32)[..., None] / 255.0
    return np.clip(canvas * (1.0 - alpha) + layer * alpha, 0, 255).astype(np.uint8)


def _rotation_matrix(cv2, pivot, angle, translation):
    matrix = cv2.getRotationMatrix2D(pivot, angle, 1.0)
    matrix[0, 2] += translation[0]
    matrix[1, 2] += translation[1]
    return matrix


def _render_frames(source: Path):
    import cv2
    import numpy as np

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (480, 832):
        raise ValueError("carrier source must be a readable 832x480 image")
    geometry = TSHIRT_832X480_CARRIER
    left_mask = _polygon_mask(cv2, np, image.shape, geometry.viewer_left_polygon)
    right_mask = _polygon_mask(cv2, np, image.shape, geometry.viewer_right_polygon)
    body_mask = _polygon_mask(cv2, np, image.shape, geometry.body_polygon)
    union = np.maximum(np.maximum(left_mask, right_mask), body_mask)
    background = cv2.inpaint(image, union, 9, cv2.INPAINT_TELEA)
    height, width = image.shape[:2]
    size = (width, height)
    identity = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    upper_clip = np.zeros((height, width), dtype=np.uint8)
    upper_clip[:211, :] = 255
    lower_clip = np.zeros((height, width), dtype=np.uint8)
    lower_clip[190:, :] = 255
    upper_body_mask = cv2.bitwise_and(body_mask, upper_clip)
    lower_body_mask = cv2.bitwise_and(body_mask, lower_clip)

    frames = []
    for frame_index in range(124):
        left = phase_progress(frame_index, 20, 40)
        right = phase_progress(frame_index, 60, 80)
        body = phase_progress(frame_index, 80, 105)
        move = phase_progress(frame_index, 111, 121)
        translation = (
            geometry.bundle_translation[0] * move,
            geometry.bundle_translation[1] * move,
        )
        left_matrix = _rotation_matrix(
            cv2,
            geometry.viewer_left_pivot,
            geometry.viewer_left_angle_degrees * left,
            translation,
        )
        right_matrix = _rotation_matrix(
            cv2,
            geometry.viewer_right_pivot,
            geometry.viewer_right_angle_degrees * right,
            translation,
        )
        upper_matrix = identity.copy()
        upper_matrix[:, 2] += translation
        lower_scale = 1.0 - 0.48 * body
        lower_matrix = np.asarray(
            [
                [1.0, 0.0, translation[0]],
                [0.0, lower_scale, 190.0 * (1.0 - lower_scale) - 10.0 * body],
            ],
            dtype=np.float32,
        )
        canvas = background.copy()
        if body <= 0.0:
            body_layer, body_alpha = _warp(cv2, image, body_mask, upper_matrix, size)
            canvas = _composite(np, canvas, body_layer, body_alpha)
        else:
            upper_layer, upper_alpha = _warp(
                cv2, image, upper_body_mask, upper_matrix, size
            )
            lower_layer, lower_alpha = _warp(
                cv2, image, lower_body_mask, lower_matrix, size
            )
            canvas = _composite(np, canvas, upper_layer, upper_alpha)
            canvas = _composite(np, canvas, lower_layer, lower_alpha)
        left_layer, left_alpha = _warp(cv2, image, left_mask, left_matrix, size)
        right_layer, right_alpha = _warp(cv2, image, right_mask, right_matrix, size)
        canvas = _composite(np, canvas, left_layer, left_alpha)
        canvas = _composite(np, canvas, right_layer, right_alpha)
        reconstruction = phase_progress(frame_index, 12, 20)
        if reconstruction < 1.0:
            canvas = np.clip(
                image * (1.0 - reconstruction) + canvas * reconstruction,
                0,
                255,
            ).astype(np.uint8)
        frames.append(canvas)
    frames[0] = image.copy()
    return frames


def _encode_carrier(frames, output: Path) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the carrier")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "832x480",
        "-r",
        "24",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "12",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the T-shirt carrier")
    return command


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    base = args.base_condition.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"carrier condition output already exists: {output}")
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("base condition manifest must contain one JSON object")
    shutil.copytree(base, output)
    source = output / str(manifest["first_frame"])
    carrier = output / "control" / "length-preserving-cloth-carrier.mp4"
    frames = _render_frames(source)
    ffmpeg_command = _encode_carrier(frames, carrier)
    contract = output / "control" / "carrier-contract.json"
    write_carrier_contract(contract, TSHIRT_832X480_CARRIER)
    variant = manifest["variants"][0]
    condition_path = output / str(variant["condition"])
    condition = ACWMActionCondition.from_json(condition_path)
    condition = ACWMActionCondition(
        label=condition.label,
        instruction=condition.instruction,
        timeline=condition.timeline,
        representation=condition.representation,
        coordinate_frame=condition.coordinate_frame,
        timestamps_s=condition.timestamps_s,
        channels=condition.channels,
        values=condition.values,
        visual_condition=carrier,
    )
    condition.to_json(condition_path)
    prompt = str(variant["prompt"]).replace(
        "<Video 1> is a static scene/camera identity reference only. It deliberately contains no target motion and no edited target states. Do not copy its stillness; synthesize the new continuous manipulation from the appended hash-bound task plan.",
        "<Video 1> is a continuous camera-frame cloth-motion carrier. Follow its left-sleeve rigid rotation, right-sleeve rigid rotation, lower-body fold, and final viewer-left bundle translation frame by frame. Infer continuous two-arm gripper contact from the task plan. Preserve the real scene and robot identity from the pictures; do not render carrier seams or self-moving cloth.",
    )
    prompt_path = output / str(variant["prompt_file"])
    prompt_path.write_text(prompt)
    manifest.update(
        {
            **capture_provenance(project_root, [sys.executable, *sys.argv], args.seed),
            "status": "carrier_condition_compiled",
            "honest_status": "NOT STARTED",
            "method": "continuous_rigid_sleeve_carrier_plus_hash_bound_plan",
            "carrier_contract": str(contract.relative_to(output)),
            "carrier_contract_sha256": file_sha256(contract),
            "carrier_video": str(carrier.relative_to(output)),
            "carrier_video_sha256": file_sha256(carrier),
            "carrier_ffmpeg_command": ffmpeg_command,
        }
    )
    variant.update(
        {
            "condition_sha256": file_sha256(condition_path),
            "prompt": prompt,
            "prompt_sha256": file_sha256(prompt_path),
            "control_video": str(carrier.relative_to(output)),
            "control_video_sha256": file_sha256(carrier),
        }
    )
    write_json_atomic(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "manifest": str(output / "manifest.json"),
                "carrier": str(carrier),
                "carrier_sha256": file_sha256(carrier),
                "contract": str(contract),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
