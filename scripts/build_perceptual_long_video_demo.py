#!/usr/bin/env python3
"""Build a native-resolution, topology-locked long-video display candidate.

The Wan candidate supplies robot appearance and motion.  The real source owns
the background, flowers, and flower response.  Only frames diagnosed by the
independent hand audit receive canonical robot-hand sprites, transformed from
one reference frame by the source wrist-to-hand direction.  The result is a
perceptual synthetic demo and is not metric contact evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.agent.perceptual_video_harness import foundation_model_roles  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--wan-candidate", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--person-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument("--pose-limbs", type=Path, required=True)
    parser.add_argument("--failure-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--reference-frame", type=int, default=276)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--temporal-padding", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    result: dict[str, object] = {}
    for label, command in {
        "head": ["git", "rev-parse", "--verify", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
    }.items():
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        result[label] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    return result


def _package_versions() -> dict[str, str | None]:
    result = {}
    for name in ("numpy", "opencv-python-headless"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _load_packed(np: Any, path: Path, key: str) -> tuple[Any, int, int, str]:
    payload = np.load(path, allow_pickle=False)
    return (
        payload[key],
        int(payload["height"]),
        int(payload["width"]),
        str(payload["bitorder"]),
    )


def _unpack(np: Any, payload: tuple[Any, int, int, str], index: int) -> Any:
    packed, height, width, bitorder = payload
    flat = np.unpackbits(packed[index], bitorder=bitorder)[: height * width]
    return flat.reshape(height, width).astype(np.uint8)


def _aligned_mask_to_native(cv2: Any, np: Any, mask: Any, width: int, height: int) -> Any:
    """Map camera:source_aligned_832x480 back to camera:source_native_1280x720."""

    if mask.shape != (480, 832):
        raise ValueError("aligned masks must be 832x480")
    canvas = np.zeros((480, 854), dtype=np.uint8)
    canvas[:, 11:843] = mask
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_NEAREST) > 0


def _aligned_points_to_native(np: Any, points: Any) -> Any:
    result = points.astype(np.float32).copy()
    result[..., 0] = (result[..., 0] + 11.0) * (1280.0 / 854.0)
    result[..., 1] = result[..., 1] * (720.0 / 480.0)
    return result


def _side_points(np: Any, landmarks: Any, mapping: dict[int, int], side: str) -> Any:
    ids = (15, 17, 19, 21) if side == "left" else (16, 18, 20, 22)
    return np.asarray([landmarks[mapping[item]] for item in ids], dtype=np.float32)


def _side_forearm(np: Any, landmarks: Any, mapping: dict[int, int], side: str) -> Any:
    ids = (13, 15) if side == "left" else (14, 16)
    return np.asarray([landmarks[mapping[item]] for item in ids], dtype=np.float32)


def _similarity(np: Any, reference: Any, target: Any) -> tuple[Any, float, float]:
    ref_elbow, ref_wrist = reference
    target_elbow, target_wrist = target
    ref_vector = ref_wrist - ref_elbow
    target_vector = target_wrist - target_elbow
    ref_norm = float(np.linalg.norm(ref_vector))
    target_norm = float(np.linalg.norm(target_vector))
    if ref_norm < 1e-4 or target_norm < 1e-4:
        raise ValueError("wrist-to-hand direction is degenerate")
    scale = target_norm / ref_norm
    angle = float(
        np.arctan2(target_vector[1], target_vector[0])
        - np.arctan2(ref_vector[1], ref_vector[0])
    )
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    linear = scale * np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    translation = target_wrist - linear @ ref_wrist
    matrix = np.concatenate((linear, translation[:, None]), axis=1)
    return matrix, scale, float(np.degrees(angle))


def _hand_corridor(cv2: Any, np: Any, points: Any, shape: tuple[int, int]) -> Any:
    mask = np.zeros(shape, dtype=np.uint8)
    wrist = tuple(np.rint(points[0]).astype(int))
    for endpoint in points[1:]:
        tip = tuple(np.rint(endpoint).astype(int))
        cv2.line(mask, wrist, tip, 255, 38, cv2.LINE_AA)
        cv2.circle(mask, tip, 19, 255, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(mask, wrist, 26, 255, cv2.FILLED, cv2.LINE_AA)
    return mask


def _canonical_sprites(
    cv2: Any,
    np: Any,
    reference: Any,
    source_reference: Any,
    ref_landmarks: Any,
    mapping: dict[int, int],
) -> dict[str, dict[str, Any]]:
    difference = np.max(
        np.abs(reference.astype(np.int16) - source_reference.astype(np.int16)), axis=2
    )
    sprites = {}
    for side in ("left", "right"):
        points = _side_points(np, ref_landmarks, mapping, side)
        corridor = _hand_corridor(cv2, np, points, reference.shape[:2])
        alpha = ((difference >= 12) & (corridor > 0)).astype(np.uint8) * 255
        alpha = cv2.morphologyEx(
            alpha,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        alpha = cv2.dilate(
            alpha,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
        if int(np.count_nonzero(alpha >= 32)) < 500:
            raise RuntimeError(f"canonical {side} hand sprite is unexpectedly empty")
        sprites[side] = {"image": reference, "alpha": alpha, "points": points}
        sprites[side]["forearm"] = _side_forearm(
            np, ref_landmarks, mapping, side
        )
    return sprites


def _repair_weights(raw_failures: list[int], padding: int, frames: int) -> dict[int, float]:
    result: dict[int, float] = {}
    for failure in raw_failures:
        for offset in range(-padding, padding + 1):
            index = failure + offset
            if 0 <= index < frames:
                weight = 1.0 - abs(offset) / (padding + 1.0)
                result[index] = max(result.get(index, 0.0), weight)
    return result


def _encoder(ffmpeg: Path, output: Path, width: int, height: int, fps: float, lossless: bool) -> Any:
    codec = ["-c:v", "ffv1", "-level", "3"] if lossless else [
        "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p"
    ]
    return subprocess.Popen(
        [
            str(ffmpeg),
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
            *codec,
            str(output),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _finish_encoder(process: Any, label: str) -> str:
    assert process.stdin is not None
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    returncode = process.wait()
    if returncode:
        raise RuntimeError(f"{label} encoder failed with {returncode}: {stderr[-2000:]}")
    return stderr


def _sheet(cv2: Any, np: Any, frames: list[tuple[int, Any]], columns: int, label: str) -> Any:
    tiles = []
    for index, frame in frames:
        tile = frame.copy()
        cv2.putText(
            tile,
            f"{label}  f{index:03d}  {index / 24.0:05.2f}s",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (15, 15, 15),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            f"{label}  f{index:03d}  {index / 24.0:05.2f}s",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    blank = np.zeros_like(tiles[0])
    while len(tiles) % columns:
        tiles.append(blank)
    rows = [cv2.hconcat(tiles[start : start + columns]) for start in range(0, len(tiles), columns)]
    return cv2.vconcat(rows)


def main() -> int:
    args = _parser().parse_args()
    import cv2
    import numpy as np

    paths = {
        "source_video": args.source_video.expanduser().resolve(),
        "wan_candidate": args.wan_candidate.expanduser().resolve(),
        "reference_image": args.reference_image.expanduser().resolve(),
        "person_masks": args.person_masks.expanduser().resolve(),
        "flower_masks": args.flower_masks.expanduser().resolve(),
        "pose_limbs": args.pose_limbs.expanduser().resolve(),
        "failure_manifest": args.failure_manifest.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{name} is missing or empty: {path}")
    if args.expected_frames <= 0 or args.fps <= 0 or args.temporal_padding < 0:
        raise ValueError("frame count/FPS must be positive and padding non-negative")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    (output / "review").mkdir()

    failure_payload = json.loads(paths["failure_manifest"].read_text())
    raw_failures = [
        int(value) for value in failure_payload["audit_source"]["raw_failed_frames"]
    ]
    repair_weights = _repair_weights(
        raw_failures, args.temporal_padding, args.expected_frames
    )
    person_payload = _load_packed(np, paths["person_masks"], "packed")
    flower_payload = _load_packed(np, paths["flower_masks"], "packed")
    pose = np.load(paths["pose_limbs"], allow_pickle=False)
    landmarks = _aligned_points_to_native(np, pose["landmarks_xy"])
    landmark_ids = [int(value) for value in pose["landmark_ids"]]
    mapping = {value: index for index, value in enumerate(landmark_ids)}

    source_capture = cv2.VideoCapture(str(paths["source_video"]))
    candidate_capture = cv2.VideoCapture(str(paths["wan_candidate"]))
    if not source_capture.isOpened() or not candidate_capture.isOpened():
        raise RuntimeError("could not open source or Wan candidate")
    source_shape = (
        int(source_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(source_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(source_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    candidate_shape = (
        int(candidate_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(candidate_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(candidate_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if source_shape != (1280, 720, args.expected_frames):
        raise RuntimeError(f"source must be 1280x720x{args.expected_frames}: {source_shape}")
    if candidate_shape != (624, 352, args.expected_frames):
        raise RuntimeError(f"candidate must be 624x352x{args.expected_frames}: {candidate_shape}")

    reference = cv2.imread(str(paths["reference_image"]), cv2.IMREAD_COLOR)
    if reference is None or reference.shape[:2] != (720, 1280):
        raise RuntimeError("reference image must decode as 1280x720")
    source_capture.set(cv2.CAP_PROP_POS_FRAMES, args.reference_frame)
    ok, source_reference = source_capture.read()
    if not ok:
        raise RuntimeError("could not decode source reference frame")
    source_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    sprites = _canonical_sprites(
        cv2,
        np,
        reference,
        source_reference,
        landmarks[args.reference_frame],
        mapping,
    )
    for side, sprite in sprites.items():
        cv2.imwrite(str(output / "review" / f"canonical-{side}-alpha.png"), sprite["alpha"])

    lossless_path = output / "perceptual-demo-27p5s-lossless.mkv"
    review_path = output / "perceptual-demo-27p5s-720p.mp4"
    lossless_writer = _encoder(paths["ffmpeg"], lossless_path, 1280, 720, args.fps, True)
    review_writer = _encoder(paths["ffmpeg"], review_path, 1280, 720, args.fps, False)
    assert lossless_writer.stdin is not None and review_writer.stdin is not None

    uniform_indices = set(int(value) for value in np.linspace(0, args.expected_frames - 1, 24))
    failure_review_indices = set(raw_failures)
    full_review: list[tuple[int, Any]] = []
    hand_review: list[tuple[int, Any]] = []
    previous_source = None
    source_flower_motion = []
    flower_exact = 0
    flower_total = 0
    background_exact = 0
    background_total = 0
    person_changed = 0
    person_total = 0
    transform_rows = []
    started = time.perf_counter()
    for index in range(args.expected_frames):
        source_ok, source = source_capture.read()
        candidate_ok, candidate = candidate_capture.read()
        if not source_ok or not candidate_ok:
            raise RuntimeError(f"decode stopped at frame {index}")
        candidate_native = cv2.resize(candidate, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
        person = _aligned_mask_to_native(
            cv2, np, _unpack(np, person_payload, index), 1280, 720
        )
        flower = _aligned_mask_to_native(
            cv2, np, _unpack(np, flower_payload, index), 1280, 720
        )
        support = cv2.dilate(
            person.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37)),
        )
        alpha = cv2.GaussianBlur(support, (0, 0), 2.0).astype(np.float32) / 255.0
        alpha[flower] = 0.0
        composed = np.rint(
            candidate_native.astype(np.float32) * alpha[..., None]
            + source.astype(np.float32) * (1.0 - alpha[..., None])
        ).astype(np.uint8)

        repair_weight = repair_weights.get(index, 0.0)
        frame_transforms = {}
        hand_edit = np.zeros((720, 1280), dtype=bool)
        if repair_weight > 0:
            for side, sprite in sprites.items():
                current_forearm = _side_forearm(np, landmarks[index], mapping, side)
                matrix, scale, angle = _similarity(
                    np, sprite["forearm"], current_forearm
                )
                warped_image = cv2.warpAffine(
                    sprite["image"],
                    matrix,
                    (1280, 720),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                warped_alpha = cv2.warpAffine(
                    sprite["alpha"],
                    matrix,
                    (1280, 720),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                ).astype(np.float32) / 255.0
                hand_alpha = warped_alpha * repair_weight
                hand_alpha[flower] = 0.0
                hand_edit |= hand_alpha >= 0.01
                composed = np.rint(
                    warped_image.astype(np.float32) * hand_alpha[..., None]
                    + composed.astype(np.float32) * (1.0 - hand_alpha[..., None])
                ).astype(np.uint8)
                frame_transforms[side] = {
                    "scale": scale,
                    "rotation_degrees": angle,
                    "alpha_pixels": int(np.count_nonzero(hand_alpha >= 0.25)),
                }
            transform_rows.append(
                {
                    "frame": index,
                    "weight": repair_weight,
                    "hands": frame_transforms,
                }
            )

        composed[flower] = source[flower]
        flower_exact += int(np.count_nonzero(np.all(composed[flower] == source[flower], axis=1)))
        flower_total += int(np.count_nonzero(flower))
        protected_background = ~(support > 0) & ~flower & ~hand_edit
        background_exact += int(
            np.count_nonzero(
                np.all(composed[protected_background] == source[protected_background], axis=1)
            )
        )
        background_total += int(np.count_nonzero(protected_background))
        changed_support = (support > 0) & ~flower
        person_total += int(np.count_nonzero(changed_support))
        person_changed += int(
            np.count_nonzero(
                np.max(
                    np.abs(
                        composed[changed_support].astype(np.int16)
                        - source[changed_support].astype(np.int16)
                    ),
                    axis=1,
                )
                >= 12
            )
        )
        if previous_source is not None:
            motion_mask = flower
            if np.any(motion_mask):
                delta = np.mean(
                    np.abs(
                        source[motion_mask].astype(np.float32)
                        - previous_source[motion_mask].astype(np.float32)
                    )
                )
                source_flower_motion.append(float(delta))
        previous_source = source
        lossless_writer.stdin.write(composed.tobytes())
        review_writer.stdin.write(composed.tobytes())

        if index in uniform_indices:
            full_review.append(
                (index, cv2.resize(composed, (480, 270), interpolation=cv2.INTER_AREA))
            )
        if index in failure_review_indices:
            points = np.concatenate(
                [
                    _side_points(np, landmarks[index], mapping, "left"),
                    _side_points(np, landmarks[index], mapping, "right"),
                ],
                axis=0,
            )
            center = np.mean(points, axis=0)
            x0 = max(0, min(1280 - 640, int(round(center[0])) - 320))
            y0 = max(0, min(720 - 360, int(round(center[1])) - 180))
            hand_review.append((index, composed[y0 : y0 + 360, x0 : x0 + 640]))

    source_capture.release()
    candidate_capture.release()
    lossless_stderr = _finish_encoder(lossless_writer, "lossless")
    review_stderr = _finish_encoder(review_writer, "review")
    wall_seconds = time.perf_counter() - started

    full_sheet = _sheet(cv2, np, full_review, columns=4, label="720P FULL")
    hand_sheet = _sheet(cv2, np, hand_review, columns=3, label="TOPOLOGY LOCK")
    full_sheet_path = output / "review" / "uniform-full-resolution-24.jpg"
    hand_sheet_path = output / "review" / "all-repaired-hands.jpg"
    cv2.imwrite(str(full_sheet_path), full_sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    cv2.imwrite(str(hand_sheet_path), hand_sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])

    motion = np.asarray(source_flower_motion, dtype=np.float32)
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "method": "wan_motion_plus_native_source_state_plus_canonical_hand_topology",
        "claim_scope": "perceptually plausible synthetic display data",
        "physical_evidence": False,
        "command": sys.argv,
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": _package_versions(),
        "git": _git_state(),
        "foundation_model_roles": list(foundation_model_roles()),
        "coordinate_frames": {
            "source": "camera:source_native_1280x720",
            "packed_masks": "camera:source_aligned_832x480",
            "wan_candidate": "camera:wan_output_624x352",
            "timeline": f"absolute_frame_index:full_source_{args.expected_frames}",
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
            if name != "ffmpeg"
        },
        "config": {
            "reference_frame": args.reference_frame,
            "expected_frames": args.expected_frames,
            "fps": args.fps,
            "temporal_padding": args.temporal_padding,
            "raw_failure_frames": raw_failures,
            "repair_frames": sorted(repair_weights),
        },
        "metrics": {
            "video_seconds": args.expected_frames / args.fps,
            "frames": args.expected_frames,
            "native_width": 1280,
            "native_height": 720,
            "wall_seconds": wall_seconds,
            "compositor_fps": args.expected_frames / wall_seconds,
            "realtime_factor": wall_seconds / (args.expected_frames / args.fps),
            "raw_failure_coverage": sum(frame in repair_weights for frame in raw_failures)
            / len(raw_failures),
            "repair_frame_count": len(repair_weights),
            "flower_exact_fraction_before_encode": flower_exact / flower_total,
            "native_background_exact_fraction_before_encode": background_exact
            / background_total,
            "source_person_changed_fraction_before_encode": person_changed / person_total,
            "source_flower_motion_delta_mean": float(motion.mean()),
            "source_flower_motion_delta_p05": float(np.quantile(motion, 0.05)),
            "source_flower_dynamic_frame_fraction": float(np.mean(motion >= 1.0)),
        },
        "hand_transforms": transform_rows,
        "outputs": {
            "lossless": {"path": str(lossless_path), "sha256": _sha256(lossless_path)},
            "review_video": {"path": str(review_path), "sha256": _sha256(review_path)},
            "uniform_review": {"path": str(full_sheet_path), "sha256": _sha256(full_sheet_path)},
            "hand_review": {"path": str(hand_sheet_path), "sha256": _sha256(hand_sheet_path)},
        },
        "limitations": [
            "The hand layer is a 2-D canonical-topology display repair, not an articulated 3-D reconstruction.",
            "The real flower response is preserved from the input capture rather than predicted by the robot model.",
            "High-resolution human review and adversarial audit remain vetoes before DISPLAY_READY.",
            "No depth, force, telemetry, force closure, or real-robot executability is claimed.",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "logs" / "encoders.json").write_text(
        json.dumps(
            {"lossless_stderr": lossless_stderr, "review_stderr": review_stderr},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"output": str(output), **manifest["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
