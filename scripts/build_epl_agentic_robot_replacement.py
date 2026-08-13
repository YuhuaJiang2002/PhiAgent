#!/usr/bin/env python3
"""Build a background-locked human-to-robot comparison with EPL repairs.

The generative model supplies one robot identity/lighting anchor.  Every video
candidate is then constructed locally: source pixels are copied verbatim first,
and only a tracked subject mask may be replaced.  A deterministic EPL agent
scores phase-local motion and temporal behavior and mutates one bounded set of
render parameters per round.
"""

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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.agent.epl_video_evolution import (  # noqa: E402
    EPLVideoEvolutionAgent,
    PhaseScore,
    ReplacementParameters,
    ReplacementScorecard,
    ReplacementThresholds,
)
from phiagent.physical_language.schema import ManipulationPhase  # noqa: E402


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


def _git_state(project_root: Path) -> dict[str, object]:
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=project_root,
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


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "opencv-python-headless", "opencv-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _new_experiment(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = root / f"{stamp}-{uuid4().hex[:8]}"
    experiment.mkdir()
    return experiment


def _odd(value: int) -> int:
    return max(1, value if value % 2 else value + 1)


def _source_info(cv2: Any, source: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {source}")
    result = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    if result["width"] <= 0 or result["height"] <= 0:
        raise RuntimeError("source video dimensions are invalid")
    if result["fps"] <= 0 or result["frames"] <= 1:
        raise RuntimeError("source video timing is invalid")
    return result


def _read_frame(cv2: Any, source: Path, frame_index: int) -> Any:
    capture = cv2.VideoCapture(str(source))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot decode source frame {frame_index}")
    return frame


def _largest_components(cv2: Any, np: Any, mask: Any, minimum_area: int) -> Any:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    result = np.zeros(mask.shape, dtype=np.uint8)
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= minimum_area:
            result[labels == component] = 255
    return result


def _build_anchor_mask(
    cv2: Any,
    np: Any,
    source_anchor: Any,
    robot_anchor: Any,
    prior_mask: Any,
    semantic_mask: Any,
) -> tuple[Any, dict[str, float | int]]:
    delta = cv2.absdiff(source_anchor, robot_anchor)
    delta = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
    delta = cv2.GaussianBlur(delta, (5, 5), 0)
    changed = (delta >= 14).astype(np.uint8) * 255
    height, width = changed.shape
    complete_subject_roi = np.zeros(changed.shape, dtype=np.uint8)
    complete_subject_roi[
        round(height * 0.035) : round(height * 0.93),
        round(width * 0.47) : round(width * 0.88),
    ] = 255
    semantic_mask = cv2.bitwise_and(semantic_mask, complete_subject_roi)
    semantic_mask = cv2.morphologyEx(
        semantic_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    semantic_mask = _largest_components(
        cv2,
        np,
        semantic_mask,
        minimum_area=max(64, round(semantic_mask.size * 0.01)),
    )
    semantic_neighborhood = cv2.dilate(
        semantic_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )
    changed = cv2.bitwise_and(changed, semantic_neighborhood)
    changed = cv2.bitwise_or(changed, semantic_mask)
    changed = cv2.morphologyEx(
        changed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    )
    changed = _largest_components(
        cv2,
        np,
        changed,
        minimum_area=max(64, round(changed.size * 0.00012)),
    )
    changed = cv2.dilate(
        changed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    changed = cv2.bitwise_and(changed, complete_subject_roi)
    coverage = float(np.count_nonzero(changed) / changed.size)
    prior_coverage = float(np.count_nonzero(prior_mask) / prior_mask.size)
    if coverage < 0.04 or coverage > 0.35:
        raise RuntimeError(f"anchor replacement mask coverage is implausible: {coverage:.4f}")
    return changed, {
        "threshold": 14,
        "coverage": coverage,
        "prior_person_union_coverage": prior_coverage,
        "historical_prior_used_as_hard_boundary": False,
        "imagegen_semantic_mask_used_as_hard_boundary": True,
        "complete_subject_roi_xyxy_normalized": [0.47, 0.035, 0.88, 0.93],
        "coordinate_frame": "camera:source_pixels",
    }


def _object_mask(cv2: Any, np: Any, frame: Any, dilation: int) -> Any:
    """Conservative flower/stem mask; skin and the pale shirt are excluded."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (hue >= 26) & (hue <= 96) & (saturation >= 44) & (value >= 25)
    pink = (hue >= 145) & (hue <= 179) & (saturation >= 85) & (value >= 45)
    yellow = (hue >= 12) & (hue <= 35) & (saturation >= 80) & (value >= 65)
    height, width = frame.shape[:2]
    scene_region = np.zeros((height, width), dtype=bool)
    scene_region[round(height * 0.27) :, round(width * 0.31) :] = True
    scene_region[round(height * 0.61) :, :] = True
    scene_region[round(height * 0.36) :, round(width * 0.72) :] = True
    mask = ((green | pink | yellow) & scene_region).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if dilation:
        size = _odd(dilation * 2 + 1)
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        )
    return mask


def _phase(frame_index: int, frame_count: int) -> ManipulationPhase:
    progress = frame_index / max(1, frame_count - 1)
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


def _writer(ffmpeg: str, output: Path, width: int, height: int, fps: float) -> Any:
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def _cosine_score(np: Any, first: Any, second: Any) -> float:
    first_values = first.astype(np.float64).ravel()
    second_values = second.astype(np.float64).ravel()
    denominator = float(np.linalg.norm(first_values) * np.linalg.norm(second_values))
    if denominator < 1e-9:
        return 1.0 if float(np.linalg.norm(first_values - second_values)) < 1e-9 else 0.0
    return max(0.0, min(1.0, float(np.dot(first_values, second_values) / denominator)))


def _render_candidate(
    *,
    cv2: Any,
    np: Any,
    ffmpeg: str,
    source: Path,
    output: Path,
    source_info: dict[str, float | int],
    source_anchor: Any,
    robot_anchor: Any,
    anchor_mask: Any,
    parameters: ReplacementParameters,
    flow_width: int,
    metric_stride: int,
) -> tuple[ReplacementScorecard, dict[str, object]]:
    width = int(source_info["width"])
    height = int(source_info["height"])
    fps = float(source_info["fps"])
    frame_count = int(source_info["frames"])
    flow_height = max(2, round(height * flow_width / width))
    anchor_small = cv2.resize(source_anchor, (flow_width, flow_height), interpolation=cv2.INTER_AREA)
    anchor_gray = cv2.cvtColor(anchor_small, cv2.COLOR_BGR2GRAY)
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    writer = _writer(ffmpeg, output, width, height, fps)
    capture = cv2.VideoCapture(str(source))
    phase_metrics: dict[ManipulationPhase, list[tuple[float, float]]] = {}
    background_scores: list[float] = []
    object_scores: list[float] = []
    replacement_scores: list[float] = []
    identity_scores: list[float] = []
    motion_scores: list[float] = []
    temporal_scores: list[float] = []
    previous_source_small = None
    previous_candidate_small = None
    decoded = 0
    maximum_flow = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current_small = cv2.resize(frame, (flow_width, flow_height), interpolation=cv2.INTER_AREA)
            current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
            if parameters.flow_strength > 0:
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
                if parameters.flow_blur_pixels > 1:
                    flow = cv2.GaussianBlur(
                        flow,
                        (parameters.flow_blur_pixels, parameters.flow_blur_pixels),
                        0,
                    )
                clip_at_flow_scale = parameters.flow_clip_pixels * flow_width / width
                magnitude = np.linalg.norm(flow, axis=2)
                maximum_flow = max(maximum_flow, float(magnitude.max() * width / flow_width))
                scale = np.minimum(1.0, clip_at_flow_scale / np.maximum(magnitude, 1e-6))
                flow *= scale[..., None]
                full_flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
                full_flow[..., 0] *= width / flow_width
                full_flow[..., 1] *= height / flow_height
                motion_gate = cv2.absdiff(current_gray, anchor_gray)
                motion_gate = (motion_gate >= 7).astype(np.uint8) * 255
                motion_gate = cv2.bitwise_and(
                    motion_gate,
                    cv2.resize(
                        anchor_mask,
                        (flow_width, flow_height),
                        interpolation=cv2.INTER_NEAREST,
                    ),
                )
                motion_gate = cv2.dilate(
                    motion_gate,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                )
                motion_gate = cv2.GaussianBlur(motion_gate, (15, 15), 0)
                full_gate = cv2.resize(
                    motion_gate, (width, height), interpolation=cv2.INTER_LINEAR
                ).astype(np.float32) / 255.0
                map_x = (
                    grid_x
                    + full_flow[..., 0] * parameters.flow_strength * full_gate
                )
                map_y = (
                    grid_y
                    + full_flow[..., 1] * parameters.flow_strength * full_gate
                )
                warped_robot = cv2.remap(
                    robot_anchor,
                    map_x,
                    map_y,
                    cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT101,
                )
                warped_mask = cv2.remap(
                    anchor_mask,
                    map_x,
                    map_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
            else:
                warped_robot = robot_anchor
                warped_mask = anchor_mask

            binary_mask = cv2.bitwise_or(
                (warped_mask >= 96).astype(np.uint8) * 255,
                anchor_mask,
            )
            if parameters.mask_dilation_pixels:
                size = _odd(parameters.mask_dilation_pixels * 2 + 1)
                binary_mask = cv2.dilate(
                    binary_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
                )
            if parameters.mask_feather_pixels:
                alpha = cv2.GaussianBlur(
                    binary_mask,
                    (0, 0),
                    parameters.mask_feather_pixels,
                ).astype(np.float32) / 255.0
            else:
                alpha = binary_mask.astype(np.float32) / 255.0
            candidate = np.rint(
                frame.astype(np.float32) * (1.0 - alpha[..., None])
                + warped_robot.astype(np.float32) * alpha[..., None]
            ).astype(np.uint8)
            base_object_mask = _object_mask(cv2, np, frame, 0)
            protected_object_mask = base_object_mask
            if parameters.object_dilation_pixels:
                size = _odd(parameters.object_dilation_pixels * 2 + 1)
                protected_object_mask = cv2.dilate(
                    base_object_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
                )
            if parameters.protect_objects:
                candidate[protected_object_mask > 0] = frame[protected_object_mask > 0]

            feather_margin = math.ceil(parameters.mask_feather_pixels * 3)
            allowed_mask = binary_mask
            if feather_margin:
                size = _odd(feather_margin * 2 + 1)
                allowed_mask = cv2.dilate(
                    allowed_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
                )
            outside = allowed_mask == 0
            background_scores.append(
                float(np.count_nonzero(np.all(candidate[outside] == frame[outside], axis=1)))
                / max(1, int(np.count_nonzero(outside)))
            )
            object_region = (base_object_mask > 0) & (allowed_mask > 0)
            object_scores.append(
                float(
                    np.count_nonzero(
                        np.all(candidate[object_region] == frame[object_region], axis=1)
                    )
                )
                / max(1, int(np.count_nonzero(object_region)))
            )
            subject_region = (binary_mask > 0) & ~(protected_object_mask > 0)
            replacement_scores.append(
                1.0
                - float(
                    np.count_nonzero(
                        np.all(candidate[subject_region] == frame[subject_region], axis=1)
                    )
                )
                / max(1, int(np.count_nonzero(subject_region)))
            )
            identity_region = (anchor_mask > 0) & ~(protected_object_mask > 0)
            identity_residual = float(
                np.mean(
                    np.abs(
                        candidate[identity_region].astype(np.float32)
                        - robot_anchor[identity_region].astype(np.float32)
                    )
                )
            )
            identity_scores.append(math.exp(-identity_residual / 64.0))

            if decoded % metric_stride == 0:
                candidate_small = cv2.resize(
                    candidate, (flow_width, flow_height), interpolation=cv2.INTER_AREA
                )
                candidate_gray = cv2.cvtColor(candidate_small, cv2.COLOR_BGR2GRAY)
                if previous_source_small is not None and previous_candidate_small is not None:
                    small_mask = cv2.resize(
                        binary_mask, (flow_width, flow_height), interpolation=cv2.INTER_NEAREST
                    ) > 0
                    source_motion = cv2.absdiff(current_gray, previous_source_small)
                    candidate_motion = cv2.absdiff(candidate_gray, previous_candidate_small)
                    if np.count_nonzero(small_mask):
                        source_values = source_motion[small_mask]
                        candidate_values = candidate_motion[small_mask]
                        cosine = _cosine_score(np, source_values, candidate_values)
                        source_energy = float(np.mean(source_values))
                        candidate_energy = float(np.mean(candidate_values))
                        energy_ratio = min(
                            (candidate_energy + 1e-3) / (source_energy + 1e-3),
                            (source_energy + 1e-3) / (candidate_energy + 1e-3),
                        )
                        motion_score = math.sqrt(max(0.0, cosine * energy_ratio))
                        residual = float(
                            np.mean(
                                np.abs(
                                    candidate_values.astype(np.float32)
                                    - source_values.astype(np.float32)
                                )
                            )
                        )
                        temporal_score = math.exp(-residual / 32.0)
                        motion_scores.append(motion_score)
                        temporal_scores.append(temporal_score)
                        phase_metrics.setdefault(_phase(decoded, frame_count), []).append(
                            (motion_score, temporal_score)
                        )
                previous_source_small = current_gray
                previous_candidate_small = candidate_gray

            assert writer.stdin is not None
            writer.stdin.write(candidate.tobytes())
            decoded += 1
    finally:
        capture.release()
        if writer.stdin is not None:
            writer.stdin.close()
        return_code = writer.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg candidate writer failed with code {return_code}")

    if decoded != frame_count:
        raise RuntimeError(f"candidate decoded {decoded}/{frame_count} source frames")
    if not motion_scores or not phase_metrics:
        raise RuntimeError("candidate evaluation did not collect temporal samples")
    phase_scores = tuple(
        PhaseScore(
            phase=phase,
            motion_preservation=sum(value[0] for value in values) / len(values),
            temporal_consistency=sum(value[1] for value in values) / len(values),
            samples=len(values),
        )
        for phase, values in sorted(phase_metrics.items(), key=lambda item: item[0].value)
    )
    scorecard = ReplacementScorecard(
        background_lock=sum(background_scores) / len(background_scores),
        object_lock=sum(object_scores) / len(object_scores),
        subject_replacement=sum(replacement_scores) / len(replacement_scores),
        robot_identity=sum(identity_scores) / len(identity_scores),
        motion_preservation=sum(motion_scores) / len(motion_scores),
        temporal_consistency=sum(temporal_scores) / len(temporal_scores),
        phase_scores=phase_scores,
    )
    evidence = {
        "decoded_frames": decoded,
        "metric_stride": metric_stride,
        "flow_analysis_size": [flow_width, flow_height],
        "maximum_unclipped_flow_pixels": maximum_flow,
        "background_lock_audit": (
            "pre-encode exact RGB equality outside the per-frame allowed replacement mask"
        ),
        "object_lock_audit": (
            "pre-encode exact RGB equality on conservative flower/stem pixels intersecting "
            "the replacement region"
        ),
        "coordinate_frame": parameters.coordinate_frame,
    }
    return scorecard, evidence


def _build_comparison(
    ffmpeg: str,
    source: Path,
    candidate: Path,
    output: Path,
    width: int,
    height: int,
) -> None:
    filter_graph = (
        f"[0:v]scale={width}:{height}[left];"
        f"[1:v]scale={width}:{height}[right];"
        "[left][right]hstack=inputs=2[v]"
    )
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(source),
            "-i",
            str(candidate),
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def _render_review_assets(ffmpeg: str, comparison: Path, output_dir: Path) -> None:
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-ss",
            "11.5",
            "-i",
            str(comparison),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_dir / "poster.jpg"),
        ],
        check=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(comparison),
            "-vf",
            "fps=1/4,scale=960:-1,tile=2x4:padding=4:margin=4:color=black",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_dir / "storyboard.jpg"),
        ],
        check=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--robot-anchor", type=Path, required=True)
    parser.add_argument("--prior-person-mask", type=Path, required=True)
    parser.add_argument("--semantic-person-mask", type=Path, required=True)
    parser.add_argument("--anchor-prompt-file", type=Path, required=True)
    parser.add_argument("--semantic-mask-prompt-file", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--anchor-seconds", type=float, default=11.5)
    parser.add_argument("--maximum-rounds", type=int, default=4)
    parser.add_argument("--flow-width", type=int, default=320)
    parser.add_argument("--metric-stride", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-finalization", type=Path)
    return parser


def _finalize_existing(args: argparse.Namespace, paths: dict[str, Path]) -> int:
    experiment = args.resume_finalization.expanduser().resolve()
    trace_path = experiment / "trace.json"
    if not trace_path.is_file():
        raise ValueError(f"resume trace does not exist: {trace_path}")
    trace = json.loads(trace_path.read_text())
    rounds = trace.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("resume trace contains no completed rounds")
    threshold_values = trace["thresholds"]
    safety_fields = (
        "background_lock",
        "object_lock",
        "subject_replacement",
        "robot_identity",
    )
    safety_rounds = [
        item
        for item in rounds
        if all(
            float(item["scorecard"][field]) >= float(threshold_values[field])
            for field in safety_fields
        )
    ]
    if safety_rounds:
        best_round = max(
            safety_rounds,
            key=lambda item: (
                float(item["scorecard"]["epl_minimum"]),
                float(item["scorecard"]["motion_preservation"]),
                float(item["scorecard"]["robot_identity"]),
                float(item["scorecard"]["mean_score"]),
            ),
        )
        selection_rule = (
            "pass background/object/subject/identity safety gates, then maximize "
            "EPL phase minimum and motion preservation"
        )
    else:
        best_round = max(
            rounds,
            key=lambda item: (
                float(item["scorecard"]["robot_identity"]),
                float(item["constraint_margin"]),
                float(item["scorecard"]["mean_score"]),
            ),
        )
        selection_rule = (
            "no safety-valid candidate; maximize robot identity before constraint margin"
        )
    best_scorecard = best_round["scorecard"]
    accepted = all(
        float(best_scorecard[field]) >= float(value)
        for field, value in threshold_values.items()
    )
    source_info = trace["source_video"]
    final_dir = experiment / "final"
    final_dir.mkdir(exist_ok=True)
    final_video = final_dir / "robot-replacement.mp4"
    shutil.copy2(Path(best_round["output"]), final_video)
    comparison = final_dir / "human-vs-robot-comparison.mp4"
    _build_comparison(
        str(paths["ffmpeg"]),
        paths["source"],
        final_video,
        comparison,
        int(source_info["width"]),
        int(source_info["height"]),
    )
    _render_review_assets(str(paths["ffmpeg"]), comparison, final_dir)
    for output in (final_video, comparison):
        subprocess.run(
            [str(paths["ffmpeg"]), "-v", "error", "-i", str(output), "-f", "null", "-"],
            check=True,
        )
    trace.pop("error", None)
    trace.update(
        {
            "status": "accepted" if accepted else "rejected",
            "honest_status": "WORKING" if accepted else "PARTIAL",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "best_round": int(best_round["round"]),
            "best_scorecard": best_scorecard,
            "best_parameters": best_round["parameters"],
            "selection_rule": selection_rule,
            "safety_valid_candidate_count": len(safety_rounds),
            "comparison_layout": "left original, right robot; no burned labels",
            "outputs": {
                "robot_replacement": str(final_video),
                "robot_replacement_sha256": _sha256(final_video),
                "comparison": str(comparison),
                "comparison_sha256": _sha256(comparison),
                "poster": str(final_dir / "poster.jpg"),
                "storyboard": str(final_dir / "storyboard.jpg"),
            },
            "acceptance": {
                "real_input_full_clip_decoded": True,
                "output_full_clip_decoded": True,
                "all_non_subject_pixels_copied_from_current_source_frame_before_encoding": True,
                "flower_and_stem_source_pixels_restored": bool(
                    best_round["parameters"]["protect_objects"]
                ),
                "thresholds_passed": accepted,
            },
            "limitations": [
                "This is a proxy demo, not official PhiZero inference and not real-robot execution.",
                "The selected safe candidate keeps the robot body at the generated anchor pose; it does not pass the motion-transfer or EPL phase-minimum gates.",
                "The rejected motion-evolved candidate deformed the robot identity and was not selected.",
                "Flower/stem protection is conservative HSV segmentation and can miss pale or fully occluded petals.",
                "Pixel-lock metrics are measured before lossy H.264 encoding; decoded MP4 pixels may differ slightly because of the codec.",
                "EPL phases are deterministic timeline segments for phase-local evaluation, not inferred contact labels.",
            ],
        }
    )
    _write_json(trace_path, trace)
    _write_json(final_dir / "manifest.json", trace)
    print(
        json.dumps(
            {
                "experiment": str(experiment),
                "status": trace["status"],
                "honest_status": trace["honest_status"],
                "best_round": trace["best_round"],
                "comparison": str(comparison),
                "scorecard": best_scorecard,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if accepted else 2


def main() -> int:
    args = _parser().parse_args()
    if args.maximum_rounds < 2:
        raise ValueError("maximum-rounds must be at least two for evolutionary repair")
    if args.flow_width < 64 or args.metric_stride < 1:
        raise ValueError("flow-width must be >= 64 and metric-stride must be positive")
    if not math.isfinite(args.anchor_seconds) or args.anchor_seconds < 0:
        raise ValueError("anchor-seconds must be finite and non-negative")
    project_root = Path(__file__).resolve().parents[1]
    paths = {
        "source": args.source.expanduser().resolve(),
        "robot_anchor": args.robot_anchor.expanduser().resolve(),
        "prior_person_mask": args.prior_person_mask.expanduser().resolve(),
        "semantic_person_mask": args.semantic_person_mask.expanduser().resolve(),
        "anchor_prompt_file": args.anchor_prompt_file.expanduser().resolve(),
        "semantic_mask_prompt_file": args.semantic_mask_prompt_file.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")

    if args.resume_finalization is not None:
        return _finalize_existing(args, paths)

    experiment = _new_experiment(args.experiment_root.expanduser().resolve())
    trace_path = experiment / "trace.json"
    try:
        import cv2
        import numpy as np

        np.random.seed(args.seed)
        source_info = _source_info(cv2, paths["source"])
        width = int(source_info["width"])
        height = int(source_info["height"])
        anchor_index = min(
            int(source_info["frames"]) - 1,
            round(args.anchor_seconds * float(source_info["fps"])),
        )
        source_anchor = _read_frame(cv2, paths["source"], anchor_index)
        robot_anchor = cv2.imread(str(paths["robot_anchor"]), cv2.IMREAD_COLOR)
        prior_mask = cv2.imread(str(paths["prior_person_mask"]), cv2.IMREAD_GRAYSCALE)
        semantic_mask = cv2.imread(
            str(paths["semantic_person_mask"]), cv2.IMREAD_GRAYSCALE
        )
        if robot_anchor is None or prior_mask is None or semantic_mask is None:
            raise RuntimeError("cannot decode robot anchor or person masks")
        robot_anchor = cv2.resize(
            robot_anchor, (width, height), interpolation=cv2.INTER_LANCZOS4
        )
        prior_mask = cv2.resize(
            prior_mask, (width, height), interpolation=cv2.INTER_NEAREST
        )
        prior_mask = (prior_mask >= 127).astype(np.uint8) * 255
        semantic_mask = cv2.resize(
            semantic_mask, (width, height), interpolation=cv2.INTER_LANCZOS4
        )
        semantic_mask = (semantic_mask >= 127).astype(np.uint8) * 255
        anchor_mask, mask_evidence = _build_anchor_mask(
            cv2, np, source_anchor, robot_anchor, prior_mask, semantic_mask
        )
        assets = experiment / "assets"
        assets.mkdir()
        cv2.imwrite(str(assets / "source-anchor.png"), source_anchor)
        cv2.imwrite(str(assets / "robot-anchor-resized.png"), robot_anchor)
        cv2.imwrite(str(assets / "localized-replacement-mask.png"), anchor_mask)
        overlay = source_anchor.copy()
        overlay[anchor_mask > 0] = np.rint(
            0.45 * source_anchor[anchor_mask > 0]
            + 0.55 * np.asarray((40, 70, 245), dtype=np.float64)
        ).astype(np.uint8)
        cv2.imwrite(str(assets / "mask-overlay.jpg"), overlay)

        thresholds = ReplacementThresholds()
        trace: dict[str, object] = {
            "schema_version": "1.0.0",
            "status": "running",
            "method": "imagegen_anchor_plus_epl_conditioned_agentic_multi_round_evolution",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": _package_versions(),
            "seed": args.seed,
            "gpu": {
                "used": False,
                "cuda_visible_devices": None,
                "reason": "local deterministic OpenCV compositor runs on CPU; image anchor was generated before this run",
            },
            "git": _git_state(project_root),
            "inputs": {
                **{label: str(path) for label, path in paths.items()},
                **{f"{label}_sha256": _sha256(path) for label, path in paths.items()},
                "robot_anchor_role": "imagegen precise-object-edit identity and lighting anchor",
                "anchor_prompt": paths["anchor_prompt_file"].read_text().strip(),
                "semantic_mask_prompt": paths[
                    "semantic_mask_prompt_file"
                ].read_text().strip(),
            },
            "source_video": {
                **source_info,
                "duration_seconds": int(source_info["frames"]) / float(source_info["fps"]),
                "anchor_frame": anchor_index,
                "anchor_seconds": args.anchor_seconds,
            },
            "mask_evidence": mask_evidence,
            "thresholds": asdict(thresholds),
            "maximum_rounds": args.maximum_rounds,
            "coordinate_frames": {
                "source": "camera:source_pixels",
                "robot_anchor": "camera:source_pixels after explicit resize",
                "optical_flow": "target-frame camera pixels -> anchor-frame camera pixels",
            },
            "rounds": [],
        }
        _write_json(trace_path, trace)

        if args.preflight_only:
            trace.update(
                {
                    "status": "preflight_complete",
                    "honest_status": "PARTIAL",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "limitations": ["No video candidate was rendered in preflight-only mode."],
                }
            )
            _write_json(trace_path, trace)
            print(json.dumps({"experiment": str(experiment), "mask_evidence": mask_evidence}))
            return 0

        agent = EPLVideoEvolutionAgent()
        parameters = ReplacementParameters()
        candidates: list[tuple[Path, ReplacementScorecard, ReplacementParameters]] = []
        attempted: set[ReplacementParameters] = set()
        for round_index in range(args.maximum_rounds):
            if parameters in attempted:
                raise RuntimeError("EPL repair repeated an already-rendered parameter set")
            attempted.add(parameters)
            round_dir = experiment / "rounds" / f"round-{round_index:02d}"
            round_dir.mkdir(parents=True)
            output = round_dir / "candidate.mp4"
            print(
                json.dumps(
                    {
                        "event": "render_round",
                        "round": round_index,
                        "parameters": parameters.to_dict(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            scorecard, evidence = _render_candidate(
                cv2=cv2,
                np=np,
                ffmpeg=str(paths["ffmpeg"]),
                source=paths["source"],
                output=output,
                source_info=source_info,
                source_anchor=source_anchor,
                robot_anchor=robot_anchor,
                anchor_mask=anchor_mask,
                parameters=parameters,
                flow_width=args.flow_width,
                metric_stride=args.metric_stride,
            )
            accepted = thresholds.accepted(scorecard)
            decision = agent.propose(parameters, scorecard, thresholds)
            round_payload = {
                "round": round_index,
                "parameters": parameters.to_dict(),
                "scorecard": scorecard.to_dict(),
                "constraint_margin": thresholds.constraint_margin(scorecard),
                "accepted": accepted,
                "diagnoses": list(decision.diagnoses),
                "actions": list(decision.actions),
                "evidence": evidence,
                "output": str(output),
                "output_sha256": _sha256(output),
            }
            _write_json(round_dir / "metrics.json", round_payload)
            candidates.append((output, scorecard, parameters))
            trace["rounds"] = [
                json.loads(
                    (experiment / "rounds" / f"round-{index:02d}" / "metrics.json").read_text()
                )
                for index in range(len(candidates))
            ]
            _write_json(trace_path, trace)
            print(
                json.dumps(
                    {
                        "event": "scored_round",
                        "round": round_index,
                        "accepted": accepted,
                        "scorecard": scorecard.to_dict(),
                        "actions": decision.actions,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if accepted:
                break
            parameters = decision.parameters

        safety_candidates = tuple(
            (index, candidate)
            for index, candidate in enumerate(candidates)
            if candidate[1].background_lock >= thresholds.background_lock
            and candidate[1].object_lock >= thresholds.object_lock
            and candidate[1].subject_replacement >= thresholds.subject_replacement
            and candidate[1].robot_identity >= thresholds.robot_identity
        )
        if safety_candidates:
            best_index, best = max(
                safety_candidates,
                key=lambda item: (
                    item[1][1].epl_minimum,
                    item[1][1].motion_preservation,
                    item[1][1].robot_identity,
                    item[1][1].mean_score,
                ),
            )
            selection_rule = (
                "pass background/object/subject/identity safety gates, then maximize "
                "EPL phase minimum and motion preservation"
            )
        else:
            best_index, best = max(
                enumerate(candidates),
                key=lambda item: (
                    item[1][1].robot_identity,
                    thresholds.constraint_margin(item[1][1]),
                    item[1][1].mean_score,
                ),
            )
            selection_rule = (
                "no safety-valid candidate; maximize robot identity before constraint margin"
            )
        final_dir = experiment / "final"
        final_dir.mkdir()
        final_video = final_dir / "robot-replacement.mp4"
        shutil.copy2(best[0], final_video)
        comparison = final_dir / "human-vs-robot-comparison.mp4"
        _build_comparison(
            str(paths["ffmpeg"]),
            paths["source"],
            final_video,
            comparison,
            width,
            height,
        )
        _render_review_assets(str(paths["ffmpeg"]), comparison, final_dir)
        for output in (final_video, comparison):
            subprocess.run(
                [str(paths["ffmpeg"]), "-v", "error", "-i", str(output), "-f", "null", "-"],
                check=True,
            )
        accepted = thresholds.accepted(best[1])
        trace.update(
            {
                "status": "accepted" if accepted else "rejected",
                "honest_status": "WORKING" if accepted else "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "best_round": best_index,
                "best_scorecard": best[1].to_dict(),
                "best_parameters": best[2].to_dict(),
                "selection_rule": selection_rule,
                "safety_valid_candidate_count": len(safety_candidates),
                "outputs": {
                    "robot_replacement": str(final_video),
                    "robot_replacement_sha256": _sha256(final_video),
                    "comparison": str(comparison),
                    "comparison_sha256": _sha256(comparison),
                    "poster": str(final_dir / "poster.jpg"),
                    "storyboard": str(final_dir / "storyboard.jpg"),
                },
                "acceptance": {
                    "real_input_full_clip_decoded": True,
                    "output_full_clip_decoded": True,
                    "all_non_subject_pixels_copied_from_current_source_frame_before_encoding": True,
                    "flower_and_stem_source_pixels_restored": best[2].protect_objects,
                    "thresholds_passed": accepted,
                },
                "limitations": [
                    "This is a proxy demo, not official PhiZero inference and not real-robot execution.",
                    "The robot uses one image-generated identity/lighting anchor; optical flow transfers source motion without a video diffusion checkpoint.",
                    "Flower/stem protection is conservative HSV segmentation and can miss pale or fully occluded petals.",
                    "Pixel-lock metrics are measured before lossy H.264 encoding; decoded MP4 pixels may differ slightly because of the codec.",
                    "EPL phases are deterministic timeline segments for phase-local evaluation, not inferred contact labels.",
                ],
            }
        )
        _write_json(trace_path, trace)
        _write_json(final_dir / "manifest.json", trace)
        print(
            json.dumps(
                {
                    "experiment": str(experiment),
                    "status": trace["status"],
                    "honest_status": trace["honest_status"],
                    "best_round": best_index,
                    "comparison": str(comparison),
                    "scorecard": best[1].to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if accepted else 2
    except Exception as exc:
        if trace_path.exists():
            trace = json.loads(trace_path.read_text())
            trace.update(
                {
                    "status": "failed",
                    "honest_status": "BLOCKED",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_json(trace_path, trace)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
