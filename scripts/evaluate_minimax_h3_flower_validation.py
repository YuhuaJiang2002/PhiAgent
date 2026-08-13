#!/usr/bin/env python3
"""Evaluate and repair a MiniMax-H3 flower replacement in bounded rounds."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.agent.epl_video_evolution import (  # noqa: E402
    PhaseScore,
    ReplacementScorecard,
    ReplacementThresholds,
)
from phiagent.physical_language.schema import ManipulationPhase  # noqa: E402
from phiagent.training.flower_repair_policy import (  # noqa: E402
    FlowerRepairPolicy,
    NonRegressionContract,
)


@dataclass(frozen=True)
class RepairRound:
    name: str
    hard_background_lock: bool
    mask_dilation_pixels: int
    mask_feather_pixels: float
    restore_source_flowers: bool
    flower_dilation_pixels: int
    face_box_margin_pixels: int
    exclude_source_face_from_flower_restore: bool


ROUNDS = (
    RepairRound("raw-h3", False, 0, 0.0, False, 0, 0, False),
    RepairRound("tracked-mask-background-lock", True, 3, 1.0, False, 0, 0, False),
    RepairRound("tracked-mask-plus-flower-restore", True, 3, 1.0, True, 2, 0, False),
    RepairRound("face-safe-background-lock", True, 3, 1.0, False, 0, 12, True),
    RepairRound("face-safe-plus-flower-restore", True, 3, 1.0, True, 2, 12, True),
)

FACE_REPLACEMENT_THRESHOLD = 0.95


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-h3", type=Path, required=True)
    parser.add_argument(
        "--motion-reference",
        type=Path,
        help="Optional explicit action-control video used instead of source-person motion.",
    )
    parser.add_argument("--robot-reference", type=Path, required=True)
    parser.add_argument("--anchor-mask", type=Path, required=True)
    parser.add_argument("--backend-metadata", type=Path, required=True)
    parser.add_argument("--candidate-label", default="minimax_h3_nf4_ref2va")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-frame", type=int, default=60)
    parser.add_argument("--source-start-frame", type=int, default=216)
    parser.add_argument("--full-source-frames", type=int, default=660)
    parser.add_argument("--flow-width", type=int, default=208)
    parser.add_argument("--metric-stride", type=int, default=3)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument(
        "--repair-policy",
        type=Path,
        help=(
            "Optional learned flower-repair checkpoint. The evaluator first scores the raw "
            "candidate, then renders ranked repairs until one passes capability non-regression."
        ),
    )
    parser.add_argument(
        "--action-override",
        action="store_true",
        help="Do not treat divergence from the source person's motion as a failure.",
    )
    return parser


def _action_override_accepted(
    scorecard: ReplacementScorecard,
    thresholds: ReplacementThresholds,
    face_replacement: float,
    require_motion: bool = False,
) -> bool:
    safety = (
        scorecard.background_lock >= thresholds.background_lock
        and scorecard.object_lock >= thresholds.object_lock
        and scorecard.subject_replacement >= thresholds.subject_replacement
        and scorecard.robot_identity >= thresholds.robot_identity
        and face_replacement >= FACE_REPLACEMENT_THRESHOLD
    )
    return safety and (
        not require_motion
        or (
            scorecard.motion_preservation >= thresholds.motion_preservation
            and scorecard.temporal_consistency >= thresholds.temporal_consistency
            and scorecard.epl_minimum >= thresholds.epl_minimum
        )
    )


def _action_override_margin(
    scorecard: ReplacementScorecard,
    thresholds: ReplacementThresholds,
    face_replacement: float,
    require_motion: bool = False,
) -> float:
    margins = [
        scorecard.background_lock - thresholds.background_lock,
        scorecard.object_lock - thresholds.object_lock,
        scorecard.subject_replacement - thresholds.subject_replacement,
        scorecard.robot_identity - thresholds.robot_identity,
        face_replacement - FACE_REPLACEMENT_THRESHOLD,
    ]
    if require_motion:
        margins.extend(
            (
                scorecard.motion_preservation - thresholds.motion_preservation,
                scorecard.temporal_consistency - thresholds.temporal_consistency,
                scorecard.epl_minimum - thresholds.epl_minimum,
            )
        )
    return min(margins)


def _decode(cv2: Any, path: Path) -> tuple[list[Any], dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {path}")
    info = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "reported_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < 3:
        raise RuntimeError(f"decoded too few frames from {path}: {len(frames)}")
    info["decoded_frames"] = len(frames)
    return frames, info


def _align_motion_reference(
    cv2: Any,
    frames: list[Any],
    info: dict[str, float | int],
    target_info: dict[str, float | int],
) -> tuple[list[Any], dict[str, object]]:
    """Map a control clip into H3 output pixels with an explicit 2-D transform."""
    if int(info["decoded_frames"]) != int(target_info["decoded_frames"]):
        raise RuntimeError(
            "motion-reference frame-count mismatch: "
            f"{info['decoded_frames']} vs {target_info['decoded_frames']}"
        )
    if not math.isclose(float(info["fps"]), float(target_info["fps"]), abs_tol=1e-3):
        raise RuntimeError(
            f"motion-reference fps mismatch: {info['fps']} vs {target_info['fps']}"
        )
    source_width, source_height = int(info["width"]), int(info["height"])
    target_width, target_height = int(target_info["width"]), int(target_info["height"])
    transform: dict[str, object] = {
        "from": "camera:source_anchor_pixels",
        "to": "camera:H3_output_pixels",
        "operation": "identity",
        "scale_x": target_width / source_width,
        "scale_y": target_height / source_height,
        "normalized_positions_preserved": True,
    }
    if (source_width, source_height) == (target_width, target_height):
        return frames, transform
    interpolation = (
        cv2.INTER_AREA
        if target_width <= source_width and target_height <= source_height
        else cv2.INTER_LANCZOS4
    )
    transform["operation"] = "independent_axis_scale"
    transform["interpolation"] = (
        "opencv:INTER_AREA" if interpolation == cv2.INTER_AREA else "opencv:INTER_LANCZOS4"
    )
    aligned = [
        cv2.resize(frame, (target_width, target_height), interpolation=interpolation)
        for frame in frames
    ]
    return aligned, transform


def _preprocess_reference(cv2: Any, image: Any, width: int, height: int, nearest: bool) -> Any:
    scaled_width = round(width * image.shape[0] / height)
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LANCZOS4
    resized = cv2.resize(image, (scaled_width, height), interpolation=interpolation)
    start = max(0, (scaled_width - width) // 2)
    return resized[:, start : start + width]


def _track_masks(
    cv2: Any,
    np: Any,
    source_frames: list[Any],
    anchor_mask: Any,
    anchor_frame: int,
    flow_width: int,
) -> tuple[list[Any], dict[str, float]]:
    height, width = anchor_mask.shape
    flow_height = round(height * flow_width / width)
    anchor_gray = cv2.cvtColor(
        cv2.resize(source_frames[anchor_frame], (flow_width, flow_height), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    )
    small_anchor_mask = cv2.resize(
        anchor_mask, (flow_width, flow_height), interpolation=cv2.INTER_NEAREST
    )
    small_x, small_y = np.meshgrid(
        np.arange(flow_width, dtype=np.float32),
        np.arange(flow_height, dtype=np.float32),
    )
    tracked = []
    coverages = []
    for frame in source_frames:
        current_gray = cv2.cvtColor(
            cv2.resize(frame, (flow_width, flow_height), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        flow = cv2.calcOpticalFlowFarneback(
            current_gray,
            anchor_gray,
            None,
            0.5,
            4,
            25,
            4,
            7,
            1.5,
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        flow = cv2.GaussianBlur(flow, (7, 7), 0)
        magnitude = np.linalg.norm(flow, axis=2)
        scale = np.minimum(1.0, 10.0 / np.maximum(magnitude, 1e-6))
        flow *= scale[..., None]
        warped_small = cv2.remap(
            small_anchor_mask,
            small_x + flow[..., 0],
            small_y + flow[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        mask = cv2.resize(warped_small, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = (mask >= 72).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        tracked.append(mask)
        coverages.append(float(np.count_nonzero(mask) / mask.size))
    return tracked, {
        "minimum_coverage": min(coverages),
        "mean_coverage": sum(coverages) / len(coverages),
        "maximum_coverage": max(coverages),
    }


def _flower_mask(
    cv2: Any,
    np: Any,
    frame: Any,
    dilation: int,
    exclude_skin_like: bool,
) -> Any:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (hue >= 26) & (hue <= 96) & (saturation >= 44) & (value >= 25)
    pink = (hue >= 145) & (hue <= 179) & (saturation >= 85) & (value >= 45)
    yellow = (hue >= 12) & (hue <= 35) & (saturation >= 80) & (value >= 65)
    height, width = frame.shape[:2]
    scene = np.zeros((height, width), dtype=bool)
    scene[round(height * 0.27) :, round(width * 0.31) :] = True
    scene[round(height * 0.61) :, :] = True
    scene[round(height * 0.36) :, round(width * 0.72) :] = True
    selected = (green | pink | yellow) & scene
    if exclude_skin_like:
        _, cr, cb = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb))
        skin_like = (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135)
        selected &= ~skin_like
    mask = selected.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if dilation:
        size = dilation * 2 + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return mask


def _detect_source_faces(
    cv2: Any,
    source_frames: list[Any],
) -> tuple[list[tuple[int, int, int, int] | None], dict[str, object]]:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"cannot load OpenCV face detector: {cascade_path}")
    boxes: list[tuple[int, int, int, int] | None] = []
    for frame in source_frames:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(35, 35),
        )
        plausible = [
            tuple(int(value) for value in box)
            for box in detections
            if box[0] + box[2] / 2 > width * 0.55
            and box[1] + box[3] / 2 < height * 0.5
        ]
        boxes.append(max(plausible, key=lambda box: box[2] * box[3]) if plausible else None)
    return boxes, {
        "detector": str(cascade_path),
        "detected_frames": sum(box is not None for box in boxes),
        "total_frames": len(boxes),
        "selection": "largest frontal-face box with center in upper-right source-person region",
    }


def _expand_with_face_box(
    cv2: Any,
    mask: Any,
    face_box: tuple[int, int, int, int] | None,
    margin: int,
) -> Any:
    if face_box is None or margin <= 0:
        return mask
    x, y, width, height = face_box
    expanded = mask.copy()
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(mask.shape[1], x + width + margin)
    bottom = min(mask.shape[0], y + height + margin)
    cv2.rectangle(expanded, (left, top), (right - 1, bottom - 1), 255, thickness=-1)
    return expanded


def _candidate_frames(
    cv2: Any,
    np: Any,
    source: list[Any],
    raw: list[Any],
    tracked_masks: list[Any],
    face_boxes: list[tuple[int, int, int, int] | None],
    repair: RepairRound,
) -> tuple[list[Any], list[Any], list[Any]]:
    candidates, allowed_masks, base_flower_masks = [], [], []
    for source_frame, raw_frame, tracked, face_box in zip(
        source, raw, tracked_masks, face_boxes
    ):
        mask = _expand_with_face_box(
            cv2, tracked, face_box, repair.face_box_margin_pixels
        )
        if repair.mask_dilation_pixels:
            size = repair.mask_dilation_pixels * 2 + 1
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        if repair.hard_background_lock:
            if repair.mask_feather_pixels:
                alpha = cv2.GaussianBlur(mask, (0, 0), repair.mask_feather_pixels).astype(np.float32) / 255.0
            else:
                alpha = mask.astype(np.float32) / 255.0
            candidate = np.rint(
                source_frame.astype(np.float32) * (1.0 - alpha[..., None])
                + raw_frame.astype(np.float32) * alpha[..., None]
            ).astype(np.uint8)
            margin = math.ceil(repair.mask_feather_pixels * 3)
            allowed = mask
            if margin:
                size = margin * 2 + 1
                allowed = cv2.dilate(
                    allowed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                )
        else:
            candidate = raw_frame.copy()
            allowed = mask
        base_flowers = _flower_mask(
            cv2,
            np,
            source_frame,
            0,
            repair.exclude_source_face_from_flower_restore,
        )
        if repair.restore_source_flowers:
            protected = _flower_mask(
                cv2,
                np,
                source_frame,
                repair.flower_dilation_pixels,
                repair.exclude_source_face_from_flower_restore,
            )
            if repair.exclude_source_face_from_flower_restore and face_box is not None:
                x, y, box_width, box_height = face_box
                margin = repair.face_box_margin_pixels
                left, top = max(0, x - margin), max(0, y - margin)
                right = min(protected.shape[1], x + box_width + margin)
                bottom = min(protected.shape[0], y + box_height + margin)
                protected[top:bottom, left:right] = 0
                base_flowers[top:bottom, left:right] = 0
            candidate[protected > 0] = source_frame[protected > 0]
        candidates.append(candidate)
        allowed_masks.append(allowed)
        base_flower_masks.append(base_flowers)
    return candidates, allowed_masks, base_flower_masks


def _cosine(np: Any, first: Any, second: Any) -> float:
    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-9:
        return 1.0 if float(np.linalg.norm(a - b)) < 1e-9 else 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / denominator)))


def _structure(np: Any, first: Any, second: Any) -> float:
    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()
    a_centered, b_centered = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    correlation = float(np.dot(a_centered, b_centered) / denominator) if denominator > 1e-9 else 0.0
    mae = float(np.mean(np.abs(a - b)))
    return max(0.0, min(1.0, 0.6 * ((correlation + 1.0) / 2.0) + 0.4 * (1.0 - mae / 255.0)))


def _phase(absolute_frame: int, full_frames: int) -> ManipulationPhase:
    progress = absolute_frame / max(1, full_frames - 1)
    if progress < 0.12:
        return ManipulationPhase.APPROACH
    if progress < 0.25:
        return ManipulationPhase.PREGRASP
    if progress < 0.38:
        return ManipulationPhase.GRASP
    if progress < 0.72:
        return ManipulationPhase.MANIPULATE
    if progress < 0.86:
        return ManipulationPhase.RELEASE
    return ManipulationPhase.RETRACT


def _score(
    cv2: Any,
    np: Any,
    source: list[Any],
    motion_reference: list[Any],
    candidate: list[Any],
    tracked_masks: list[Any],
    allowed_masks: list[Any],
    flower_masks: list[Any],
    robot_reference: Any,
    face_boxes: list[tuple[int, int, int, int] | None],
    anchor_frame: int,
    source_start_frame: int,
    full_source_frames: int,
    flow_width: int,
    metric_stride: int,
) -> tuple[ReplacementScorecard, dict[str, object]]:
    background, objects, replacement = [], [], []
    face_replacement = []
    for source_frame, candidate_frame, tracked, allowed, flowers, face_box in zip(
        source, candidate, tracked_masks, allowed_masks, flower_masks, face_boxes
    ):
        outside = allowed == 0
        background.append(
            float(np.count_nonzero(np.all(candidate_frame[outside] == source_frame[outside], axis=1)))
            / max(1, int(np.count_nonzero(outside)))
        )
        object_region = (flowers > 0) & (tracked > 0)
        objects.append(
            float(np.count_nonzero(np.all(candidate_frame[object_region] == source_frame[object_region], axis=1)))
            / max(1, int(np.count_nonzero(object_region)))
        )
        subject_region = (tracked > 0) & ~(flowers > 0)
        replacement.append(
            1.0
            - float(np.count_nonzero(np.all(candidate_frame[subject_region] == source_frame[subject_region], axis=1)))
            / max(1, int(np.count_nonzero(subject_region)))
        )
        if face_box is not None:
            x, y, box_width, box_height = face_box
            source_face = source_frame[y : y + box_height, x : x + box_width]
            candidate_face = candidate_frame[y : y + box_height, x : x + box_width]
            face_replacement.append(
                1.0
                - float(
                    np.count_nonzero(np.all(candidate_face == source_face, axis=2))
                )
                / max(1, box_width * box_height)
            )

    identity_region = (tracked_masks[anchor_frame] > 0) & ~(flower_masks[anchor_frame] > 0)
    identity = _structure(
        np,
        cv2.cvtColor(candidate[anchor_frame], cv2.COLOR_BGR2GRAY)[identity_region],
        cv2.cvtColor(robot_reference, cv2.COLOR_BGR2GRAY)[identity_region],
    )
    width = source[0].shape[1]
    height = source[0].shape[0]
    flow_height = round(height * flow_width / width)
    motion_values, temporal_values = [], []
    phase_values: dict[ManipulationPhase, list[tuple[float, float]]] = {}
    previous_reference = previous_candidate = None
    for index in range(0, len(source), metric_stride):
        current_reference = cv2.cvtColor(
            cv2.resize(
                motion_reference[index],
                (flow_width, flow_height),
                interpolation=cv2.INTER_AREA,
            ),
            cv2.COLOR_BGR2GRAY,
        )
        current_candidate = cv2.cvtColor(
            cv2.resize(candidate[index], (flow_width, flow_height), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        if previous_reference is not None and previous_candidate is not None:
            mask = cv2.resize(
                tracked_masks[index], (flow_width, flow_height), interpolation=cv2.INTER_NEAREST
            ) > 0
            source_motion = cv2.absdiff(current_reference, previous_reference)[mask]
            candidate_motion = cv2.absdiff(current_candidate, previous_candidate)[mask]
            cosine = _cosine(np, source_motion, candidate_motion)
            source_energy = float(np.mean(source_motion))
            candidate_energy = float(np.mean(candidate_motion))
            energy_ratio = min(
                (candidate_energy + 1e-3) / (source_energy + 1e-3),
                (source_energy + 1e-3) / (candidate_energy + 1e-3),
            )
            motion = math.sqrt(max(0.0, cosine * energy_ratio))
            residual = float(np.mean(np.abs(candidate_motion.astype(np.float32) - source_motion.astype(np.float32))))
            temporal = math.exp(-residual / 32.0)
            phase = _phase(source_start_frame + index, full_source_frames)
            motion_values.append(motion)
            temporal_values.append(temporal)
            phase_values.setdefault(phase, []).append((motion, temporal))
        previous_reference, previous_candidate = current_reference, current_candidate
    phase_scores = tuple(
        PhaseScore(
            phase=phase,
            motion_preservation=sum(value[0] for value in values) / len(values),
            temporal_consistency=sum(value[1] for value in values) / len(values),
            samples=len(values),
        )
        for phase, values in sorted(phase_values.items(), key=lambda item: item[0].value)
    )
    scorecard = ReplacementScorecard(
        background_lock=sum(background) / len(background),
        object_lock=sum(objects) / len(objects),
        subject_replacement=sum(replacement) / len(replacement),
        robot_identity=identity,
        motion_preservation=sum(motion_values) / len(motion_values),
        temporal_consistency=sum(temporal_values) / len(temporal_values),
        phase_scores=phase_scores,
    )
    return scorecard, {
        "identity_proxy": "anchor-frame masked grayscale structure similarity to robot reference",
        "background_proxy": "pre-encode exact RGB equality outside allowed tracked mask",
        "object_proxy": "pre-encode exact RGB equality on conservative flower/stem pixels inside tracked mask",
        "subject_proxy": "pre-encode changed-pixel ratio inside tracked mask excluding protected flowers",
        "source_face_replacement": (
            sum(face_replacement) / len(face_replacement) if face_replacement else 0.0
        ),
        "source_face_replacement_threshold": FACE_REPLACEMENT_THRESHOLD,
        "source_face_proxy": "pre-encode changed-pixel ratio inside detected source-face boxes",
        "metric_stride": metric_stride,
        "flow_analysis_size": [flow_width, flow_height],
        "motion_reference_proxy": (
            "masked adjacent-frame grayscale motion similarity to the explicit motion reference"
        ),
    }


def _write_video(ffmpeg: Path, frames: list[Any], output: Path, fps: float) -> None:
    height, width = frames[0].shape[:2]
    process = subprocess.Popen(
        [
            str(ffmpeg), "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "-", "-an",
            "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed to encode {output}")


def main() -> int:
    args = _parser().parse_args()
    paths = {
        "source": args.source.expanduser().resolve(),
        "raw_h3": args.raw_h3.expanduser().resolve(),
        "robot_reference": args.robot_reference.expanduser().resolve(),
        "anchor_mask": args.anchor_mask.expanduser().resolve(),
        "backend_metadata": args.backend_metadata.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    if args.motion_reference is not None:
        paths["motion_reference"] = args.motion_reference.expanduser().resolve()
    if args.repair_policy is not None:
        paths["repair_policy"] = args.repair_policy.expanduser().resolve()
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} does not exist or is empty: {path}")
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "evolution.json"
    if manifest_path.exists():
        raise FileExistsError(f"evaluation already exists: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    import cv2
    import numpy as np

    source, source_info = _decode(cv2, paths["source"])
    raw, raw_info = _decode(cv2, paths["raw_h3"])
    if len(source) != len(raw) or source_info["width"] != raw_info["width"] or source_info["height"] != raw_info["height"]:
        raise RuntimeError(f"source/raw alignment mismatch: {source_info} vs {raw_info}")
    motion_reference = source
    motion_reference_info = source_info
    motion_reference_transform: dict[str, object] = {
        "from": "camera:source_clip_pixels",
        "to": "camera:H3_output_pixels",
        "operation": "identity",
        "scale_x": 1.0,
        "scale_y": 1.0,
        "normalized_positions_preserved": True,
    }
    if "motion_reference" in paths:
        motion_reference, motion_reference_info = _decode(cv2, paths["motion_reference"])
        motion_reference, motion_reference_transform = _align_motion_reference(
            cv2, motion_reference, motion_reference_info, source_info
        )
    if not 0 <= args.anchor_frame < len(source):
        raise ValueError("anchor-frame is outside the decoded clip")
    width, height = int(source_info["width"]), int(source_info["height"])
    mask_image = cv2.imread(str(paths["anchor_mask"]), cv2.IMREAD_GRAYSCALE)
    robot_image = cv2.imread(str(paths["robot_reference"]), cv2.IMREAD_COLOR)
    if mask_image is None or robot_image is None:
        raise RuntimeError("cannot decode anchor mask or robot reference")
    anchor_mask = _preprocess_reference(cv2, mask_image, width, height, True)
    anchor_mask = (anchor_mask >= 127).astype(np.uint8) * 255
    robot_reference = _preprocess_reference(cv2, robot_image, width, height, False)
    cv2.imwrite(str(output_dir / "anchor-mask-preprocessed.png"), anchor_mask)
    cv2.imwrite(str(output_dir / "robot-reference-preprocessed.png"), robot_reference)
    tracked_masks, tracking = _track_masks(
        cv2, np, source, anchor_mask, args.anchor_frame, args.flow_width
    )
    face_boxes, face_tracking = _detect_source_faces(cv2, source)
    thresholds = ReplacementThresholds()
    round_records = []
    rendered: list[tuple[Path, ReplacementScorecard, RepairRound, float]] = []
    repair_policy = (
        FlowerRepairPolicy.load(paths["repair_policy"])
        if "repair_policy" in paths
        else None
    )
    non_regression_contract = (
        repair_policy.non_regression_contract
        if repair_policy is not None
        else NonRegressionContract()
    )
    repair_plan = list(ROUNDS)
    policy_selection: dict[str, object] | None = None
    policy_queue: list[tuple[RepairRound, float]] = []
    raw_scorecard: ReplacementScorecard | None = None
    for index, repair in enumerate(repair_plan):
        round_dir = output_dir / "rounds" / f"round-{index:02d}-{repair.name}"
        round_dir.mkdir(parents=True)
        frames, allowed, flowers = _candidate_frames(
            cv2, np, source, raw, tracked_masks, face_boxes, repair
        )
        scorecard, evidence = _score(
            cv2,
            np,
            source,
            motion_reference,
            frames,
            tracked_masks,
            allowed,
            flowers,
            robot_reference,
            face_boxes,
            args.anchor_frame,
            args.source_start_frame,
            args.full_source_frames,
            args.flow_width,
            args.metric_stride,
        )
        output = round_dir / "candidate.mp4"
        _write_video(paths["ffmpeg"], frames, output, float(source_info["fps"]))
        cv2.imwrite(str(round_dir / "anchor-allowed-mask.png"), allowed[args.anchor_frame])
        cv2.imwrite(str(round_dir / "anchor-object-mask.png"), flowers[args.anchor_frame])
        diagnoses = []
        actions = []
        if scorecard.background_lock < thresholds.background_lock:
            diagnoses.append("pixels outside the tracked subject region changed")
            actions.append("copy every outside-mask pixel from the aligned source frame")
        if scorecard.object_lock < thresholds.object_lock:
            diagnoses.append("protected flower/stem pixels changed")
            actions.append("restore conservative flower/stem pixels from the aligned source frame")
        if scorecard.robot_identity < thresholds.robot_identity:
            diagnoses.append("robot identity proxy is below threshold")
            actions.append("regenerate H3 with stronger robot-reference retention")
        if (
            (not args.action_override or "motion_reference" in paths)
            and scorecard.motion_preservation < thresholds.motion_preservation
        ):
            diagnoses.append(
                "explicit action-control motion proxy is below threshold"
                if "motion_reference" in paths
                else "source manipulation motion proxy is below threshold"
            )
            actions.append("regenerate H3 with stronger Video 1 motion transfer")
        if (
            (not args.action_override or "motion_reference" in paths)
            and scorecard.temporal_consistency < thresholds.temporal_consistency
        ):
            diagnoses.append("motion-reference temporal proxy is below threshold")
            actions.append("regenerate H3 with more steps or a stronger motion reference")
        face_replacement = float(evidence["source_face_replacement"])
        if face_replacement < FACE_REPLACEMENT_THRESHOLD:
            diagnoses.append("detected source face pixels remain in the candidate")
            actions.append("union detected source-face boxes into the replacement support")
        if raw_scorecard is None:
            raw_scorecard = scorecard
        non_regression = non_regression_contract.assess(
            raw_scorecard.to_dict(), scorecard.to_dict()
        )
        if not non_regression.passed:
            regressed = [
                field
                for field, excess in non_regression.excess_regressions
                if excess > 0
            ]
            diagnoses.append(
                "candidate violates capability non-regression: " + ", ".join(regressed)
            )
            actions.append("reject repair and evaluate the next bounded candidate")
        record = {
            "round": index,
            "repair": asdict(repair),
            "scorecard": scorecard.to_dict(),
            "accepted": (
                _action_override_accepted(
                    scorecard,
                    thresholds,
                    face_replacement,
                    require_motion="motion_reference" in paths,
                )
                if args.action_override
                else thresholds.accepted(scorecard)
                and face_replacement >= FACE_REPLACEMENT_THRESHOLD
            ),
            "constraint_margin": (
                _action_override_margin(
                    scorecard,
                    thresholds,
                    face_replacement,
                    require_motion="motion_reference" in paths,
                )
                if args.action_override
                else thresholds.constraint_margin(scorecard)
            ),
            "diagnoses": diagnoses,
            "actions": actions,
            "evidence": evidence,
            "non_regression": non_regression.to_dict(),
            "output": str(output),
            "output_sha256": _sha256(output),
        }
        _write_json(round_dir / "metrics.json", record)
        round_records.append(record)
        rendered.append((output, scorecard, repair, face_replacement))
        if index == 0 and repair_policy is not None:
            repair_payloads = [asdict(item) for item in ROUNDS[1:]]
            ranking = repair_policy.rank(scorecard.to_dict(), repair_payloads)
            policy_queue = [
                (
                    next(item for item in ROUNDS[1:] if item.name == candidate["name"]),
                    prediction,
                )
                for candidate, prediction in ranking
            ]
            first_repair, _ = policy_queue.pop(0)
            repair_plan[1:] = [first_repair]
            policy_selection = {
                "checkpoint": str(paths["repair_policy"]),
                "checkpoint_sha256": _sha256(paths["repair_policy"]),
                "training_actions": list(repair_policy.training_actions),
                "held_out_action": repair_policy.held_out_action,
                "objective": repair_policy.objective,
                "non_regression_contract": non_regression_contract.to_dict(),
                "selected_repair": None,
                "ranked_predictions": [
                    {
                        "repair": str(candidate["name"]),
                        "predicted_constrained_utility": prediction,
                    }
                    for candidate, prediction in ranking
                ],
                "attempts": [],
            }
        elif index > 0 and repair_policy is not None:
            assert policy_selection is not None
            attempts = policy_selection["attempts"]
            assert isinstance(attempts, list)
            attempts.append(
                {
                    "repair": repair.name,
                    "non_regression": non_regression.to_dict(),
                }
            )
            if non_regression.passed:
                policy_selection["selected_repair"] = repair.name
            elif policy_queue:
                next_repair, _ = policy_queue.pop(0)
                repair_plan.append(next_repair)

    if policy_selection is not None:
        policy_selection["candidate_evaluations_saved"] = len(ROUNDS) - len(rendered)
        policy_selection["evaluated_candidates"] = len(rendered)

    non_regressing_rendered = [
        item
        for index, item in enumerate(rendered)
        if bool(round_records[index]["non_regression"]["passed"])
    ]
    safety = [
        item
        for item in non_regressing_rendered
        if item[1].background_lock >= thresholds.background_lock
        and item[1].object_lock >= thresholds.object_lock
        and item[1].subject_replacement >= thresholds.subject_replacement
        and item[3] >= FACE_REPLACEMENT_THRESHOLD
    ]
    human_safe = [
        item
        for item in non_regressing_rendered
        if item[1].background_lock >= thresholds.background_lock
        and item[1].subject_replacement >= thresholds.subject_replacement
        and item[3] >= FACE_REPLACEMENT_THRESHOLD
    ]
    pool = safety or human_safe or non_regressing_rendered
    if args.action_override:
        best = max(
            pool,
            key=lambda item: (
                item[1].motion_preservation if "motion_reference" in paths else 0.0,
                item[1].temporal_consistency if "motion_reference" in paths else 0.0,
                item[1].robot_identity,
                item[1].subject_replacement,
                item[1].object_lock,
                item[1].background_lock,
            ),
        )
    else:
        best = max(
            pool,
            key=lambda item: (
                item[1].epl_minimum,
                item[1].motion_preservation,
                item[1].robot_identity,
                item[1].mean_score,
            ),
        )
    final = output_dir / "final-background-locked.mp4"
    shutil.copy2(best[0], final)
    comparison = output_dir / "source-vs-raw-vs-final.mp4"
    subprocess.run(
        [
            str(paths["ffmpeg"]), "-v", "error", "-i", str(paths["source"]),
            "-i", str(paths["raw_h3"]), "-i", str(final), "-filter_complex",
            f"[0:v]scale={width}:{height}[a];[1:v]scale={width}:{height}[b];[2:v]scale={width}:{height}[c];[a][b][c]hstack=inputs=3[v]",
            "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(comparison),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(paths["ffmpeg"]), "-v", "error", "-i", str(comparison),
            "-vf", "fps=1,scale=1497:-1,tile=2x3:padding=4:margin=4:color=black",
            "-frames:v", "1", "-q:v", "2", str(output_dir / "storyboard.jpg"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(paths["ffmpeg"]), "-v", "error", "-ss", "2.5", "-i", str(comparison),
            "-frames:v", "1", "-q:v", "2", str(output_dir / "poster.jpg"),
        ],
        check=True,
    )
    accepted = (
        _action_override_accepted(
            best[1],
            thresholds,
            best[3],
            require_motion="motion_reference" in paths,
        )
        if args.action_override
        else thresholds.accepted(best[1]) and best[3] >= FACE_REPLACEMENT_THRESHOLD
    )
    evolution_decision = (
        "ACCEPT"
        if accepted
        else (
            "REGENERATE_WORLD_MODEL_CANDIDATE"
            if repair_policy is not None
            else "REJECT_REPAIR_SET"
        )
    )
    if policy_selection is not None:
        policy_selection["strict_task_accepted"] = accepted
        policy_selection["requires_regeneration"] = not accepted
    packages = {}
    for name in ("numpy", "opencv-python", "opencv-python-headless"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "schema_version": "1.0.0",
        "method": f"{args.candidate_label}_plus_tracked_mask_agentic_repairs",
        "status": "accepted" if accepted else "rejected",
        "honest_status": "WORKING" if accepted else "PARTIAL",
        "evolution_decision": evolution_decision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "packages": packages,
        "inputs": {label: {"path": str(path), "sha256": _sha256(path)} for label, path in paths.items()},
        "source_info": source_info,
        "raw_info": raw_info,
        "motion_reference_info": motion_reference_info,
        "motion_reference_transform": motion_reference_transform,
        "coordinate_frames": {
            "source": "camera:source_clip_pixels",
            "raw_h3": "camera:H3_output_pixels aligned by frame index",
            "motion_reference": (
                "camera:source_anchor_pixels transformed explicitly to camera:H3_output_pixels"
                if "motion_reference" in paths
                else "camera:source_clip_pixels"
            ),
            "mask_flow": "current source camera pixels -> anchor source camera pixels",
        },
        "tracking": tracking,
        "source_face_tracking": face_tracking,
        "thresholds": asdict(thresholds),
        "source_face_replacement_threshold": FACE_REPLACEMENT_THRESHOLD,
        "action_override": args.action_override,
        "explicit_motion_reference": "motion_reference" in paths,
        "non_regression_contract": non_regression_contract.to_dict(),
        "learned_repair_policy": policy_selection,
        "rounds": round_records,
        "best_round": next(index for index, item in enumerate(rendered) if item is best),
        "best_repair": asdict(best[2]),
        "best_scorecard": best[1].to_dict(),
        "best_source_face_replacement": best[3],
        "outputs": {
            "final": str(final),
            "final_sha256": _sha256(final),
            "comparison": str(comparison),
            "comparison_sha256": _sha256(comparison),
            "storyboard": str(output_dir / "storyboard.jpg"),
            "poster": str(output_dir / "poster.jpg"),
        },
        "limitations": [
            (
                "H3 uses the third-party NF4 quantization rather than the official BF16 weights."
                if args.candidate_label.startswith("minimax_h3_nf4")
                else "This control candidate is not a MiniMax-H3 output."
            ),
            "Robot identity is a deterministic masked grayscale structure proxy and requires visual review.",
            "Subject replacement measures source-pixel change, not a learned human detector.",
            "Source-face replacement uses an OpenCV frontal-face proxy and requires visual review.",
            "Flower protection uses conservative HSV segmentation and can miss pale or fully occluded petals.",
            "Pixel-lock metrics are measured before lossy H.264 encoding.",
            "EPL phase labels are mapped from the clip's absolute frames in the 660-frame source timeline.",
            (
                "Explicit action-control motion is an acceptance target and replaces source-person "
                "motion in the scorecard."
                if "motion_reference" in paths
                else (
                    "Source-motion preservation and its derived temporal proxy are diagnostic only; "
                    "they are excluded from acceptance and best-round selection because the language "
                    "condition intentionally overrides the source person's action."
                    if args.action_override
                    else "Source motion is an acceptance target for this replacement run."
                )
            ),
            (
                "The learned repair policy was trained on a small same-scene candidate archive; "
                "its held-action result does not establish new-scene generalization."
                if repair_policy is not None
                else "No learned repair policy was used; all five fixed repair rounds were audited."
            ),
            "Final candidate selection excludes every repair that violates the recorded capability non-regression contract.",
        ],
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"evaluation": str(output_dir), "status": manifest["status"], "scorecard": manifest["best_scorecard"]}, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
