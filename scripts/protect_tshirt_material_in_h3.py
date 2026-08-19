#!/usr/bin/env python3
"""Restore the analytic cloth layer after an H3 appearance refinement.

The H3 candidate remains authoritative outside the moving shirt support.  Inside
that support, pixels come from the connected articulated carrier whose sleeve
motion is a rigid camera-frame transform.  This is a bounded hybrid rendering
step, not evidence of metric 3-D cloth or real-robot dynamics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.harness.cloth_carrier import (  # noqa: E402
    TSHIRT_832X480_CARRIER,
    phase_progress,
)
from phiagent.harness.provenance import capture_provenance, write_json_atomic  # noqa: E402
from phiagent.rendering.minimax_h3 import file_sha256  # noqa: E402
from scripts.build_tshirt_length_preserving_carrier import (  # noqa: E402
    _encode_carrier,
    _polygon_mask,
    _rotation_matrix,
    _warp,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h3-candidate", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--carrier-manifest", type=Path, required=True)
    parser.add_argument("--candidate-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--feather-sigma", type=float, default=1.25)
    return parser


def _read_video(cv2, path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != 124 or any(frame.shape[:2] != (480, 832) for frame in frames):
        raise ValueError("protected T-shirt videos must contain 124 832x480 frames")
    if abs(fps - 24.0) > 1e-3:
        raise ValueError("protected T-shirt videos must run at 24 FPS")
    return frames


def _material_masks(cv2, np, shape):
    geometry = TSHIRT_832X480_CARRIER
    left_mask = _polygon_mask(cv2, np, shape, geometry.viewer_left_polygon)
    right_mask = _polygon_mask(cv2, np, shape, geometry.viewer_right_polygon)
    body_mask = _polygon_mask(cv2, np, shape, geometry.body_polygon)
    height, width = shape[:2]
    size = (width, height)
    identity = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    upper_clip = np.zeros((height, width), dtype=np.uint8)
    upper_clip[:211, :] = 255
    lower_clip = np.zeros((height, width), dtype=np.uint8)
    lower_clip[190:, :] = 255
    upper_body_mask = cv2.bitwise_and(body_mask, upper_clip)
    lower_body_mask = cv2.bitwise_and(body_mask, lower_clip)
    result = []
    for frame_index in range(124):
        left = phase_progress(frame_index, 20, 40)
        right = phase_progress(frame_index, 60, 80)
        body = phase_progress(frame_index, 88, 106)
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
            (
                (1.0, 0.0, translation[0]),
                (0.0, lower_scale, 190.0 * (1.0 - lower_scale) - 10.0 * body),
            ),
            dtype=np.float32,
        )
        _, left_alpha = _warp(cv2, np.zeros(shape, dtype=np.uint8), left_mask, left_matrix, size)
        _, right_alpha = _warp(
            cv2, np.zeros(shape, dtype=np.uint8), right_mask, right_matrix, size
        )
        if body <= 0.0:
            _, body_alpha = _warp(
                cv2, np.zeros(shape, dtype=np.uint8), body_mask, upper_matrix, size
            )
        else:
            _, upper_alpha = _warp(
                cv2,
                np.zeros(shape, dtype=np.uint8),
                upper_body_mask,
                upper_matrix,
                size,
            )
            _, lower_alpha = _warp(
                cv2,
                np.zeros(shape, dtype=np.uint8),
                lower_body_mask,
                lower_matrix,
                size,
            )
            body_alpha = np.maximum(upper_alpha, lower_alpha)
        result.append(np.maximum.reduce((left_alpha, right_alpha, body_alpha)))
    return result


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    candidate = args.h3_candidate.expanduser().resolve()
    carrier = args.carrier.expanduser().resolve()
    carrier_manifest_path = args.carrier_manifest.expanduser().resolve()
    candidate_evidence = args.candidate_evidence.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"protected-material output already exists: {output}")
    if args.feather_sigma < 0 or args.feather_sigma > 3:
        raise ValueError("feather sigma must remain in [0, 3]")
    for path in (candidate, carrier, carrier_manifest_path, candidate_evidence):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    carrier_manifest = json.loads(carrier_manifest_path.read_text())
    if not all(carrier_manifest.get("articulated_trajectory_gates", {}).values()):
        raise ValueError("carrier manifest does not pass every analytic trajectory gate")

    import cv2
    import numpy as np

    candidate_frames = _read_video(cv2, candidate)
    carrier_frames = _read_video(cv2, carrier)
    masks = _material_masks(cv2, np, carrier_frames[0].shape)
    protected = []
    for candidate_frame, carrier_frame, mask in zip(
        candidate_frames, carrier_frames, masks
    ):
        alpha = mask.astype(np.float32) / 255.0
        if args.feather_sigma:
            alpha = cv2.GaussianBlur(alpha, (0, 0), args.feather_sigma).clip(0.0, 1.0)
        frame = np.rint(
            carrier_frame.astype(np.float32) * alpha[..., None]
            + candidate_frame.astype(np.float32) * (1.0 - alpha[..., None])
        ).clip(0, 255).astype(np.uint8)
        protected.append(frame)
    protected[0] = carrier_frames[0].copy()

    output.mkdir(parents=True)
    video = output / "h3-with-protected-analytic-shirt.mp4"
    ffmpeg_command = _encode_carrier(protected, video)
    mask_video = output / "protected-shirt-support.mp4"
    mask_rgb = [cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) for mask in masks]
    mask_ffmpeg_command = _encode_carrier(mask_rgb, mask_video)
    evidence = json.loads(candidate_evidence.read_text())
    manifest = {
        **capture_provenance(project_root, [sys.executable, *sys.argv], args.seed),
        "schema_version": "1.0.0",
        "status": "protected_hybrid_pending_independent_evaluation",
        "honest_status": "PARTIAL",
        "method": "minimax-h3-outside-plus-analytic-material-support-inside",
        "output": str(video.relative_to(output)),
        "output_sha256": file_sha256(video),
        "candidate": str(candidate),
        "candidate_sha256": file_sha256(candidate),
        "carrier": str(carrier),
        "carrier_sha256": file_sha256(carrier),
        "carrier_manifest": str(carrier_manifest_path),
        "carrier_manifest_sha256": file_sha256(carrier_manifest_path),
        "candidate_evidence": str(candidate_evidence),
        "candidate_evidence_sha256": file_sha256(candidate_evidence),
        "candidate_failed_gates": [
            key for key, passed in evidence["gate_results"].items() if not passed
        ],
        "mask_video": str(mask_video.relative_to(output)),
        "mask_video_sha256": file_sha256(mask_video),
        "feather_sigma": args.feather_sigma,
        "ffmpeg_command": ffmpeg_command,
        "mask_ffmpeg_command": mask_ffmpeg_command,
        "analytic_gates": carrier_manifest["articulated_trajectory_gates"],
        "claim_boundary": (
            "The cloth support is restored from a 2-D analytic carrier after H3. "
            "This preserves declared camera-pixel material paths but is not metric "
            "3-D cloth, force, collision safety, or executable robot control evidence."
        ),
    }
    write_json_atomic(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
