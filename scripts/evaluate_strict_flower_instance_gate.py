#!/usr/bin/env python3
"""Evaluate named flower identities and explicit robot-hand contact segments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--aligned-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _longest_false_run(values: Any) -> int:
    longest = current = 0
    for value in values:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _load_video(cv2: Any, path: Path) -> tuple[list[Any], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video contains no frames: {path}")
    return frames, fps


def _load_track_run(np: Any, experiment: Path, run_name: str) -> dict[str, Any]:
    run = experiment / run_name
    manifest_path = run / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing track manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("coordinate_frame") != "camera:source_video_pixels":
        raise ValueError(f"track {run_name} has an unexpected coordinate frame")
    packed = [path for path in run.glob("*.npz") if path.is_file()]
    if len(packed) != 1:
        raise ValueError(f"track {run_name} must contain exactly one packed NPZ")
    expected_hash = manifest["outputs"]["packed_masks"]["sha256"]
    if _sha256(packed[0]) != expected_hash:
        raise ValueError(f"track hash mismatch: {packed[0]}")
    data = np.load(packed[0])
    ids = data["instance_ids"].astype(str).tolist()
    height = int(data["height"])
    width = int(data["width"])
    indices = data["source_frame_indices"].astype(int).tolist()
    packed_masks = data["masks_packed"]
    flat = np.unpackbits(packed_masks, axis=2, bitorder=str(data["bitorder"]))
    masks = flat[:, :, : height * width].reshape(len(ids), len(indices), height, width).astype(bool)
    return {
        "run_name": run_name,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "packed_path": packed[0],
        "ids": ids,
        "indices": indices,
        "height": height,
        "width": width,
        "masks": {instance_id: masks[index] for index, instance_id in enumerate(ids)},
    }


def _centroids(np: Any, masks: Any) -> Any:
    rows = []
    for mask in masks:
        ys, xs = np.where(mask)
        rows.append(
            [
                float(np.mean(xs)) if len(xs) else float("nan"),
                float(np.mean(ys)) if len(ys) else float("nan"),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _interpolate_centroids(np: Any, values: Any) -> Any:
    result = values.copy()
    positions = np.arange(len(result))
    for axis in range(2):
        valid = np.isfinite(result[:, axis])
        if not np.any(valid):
            raise ValueError("cannot interpolate a completely empty track")
        result[:, axis] = np.interp(positions, positions[valid], result[valid, axis])
    return result


def _track_metrics(np: Any, masks: Any) -> tuple[dict[str, Any], Any]:
    areas = np.asarray([int(np.count_nonzero(mask)) for mask in masks], dtype=np.int64)
    visible = areas > 0
    raw_centroids = _centroids(np, masks)
    centroids = _interpolate_centroids(np, raw_centroids)
    steps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    ious = []
    adjacent_area_ratios = []
    for frame_index, (first, second) in enumerate(zip(masks, masks[1:])):
        if not np.any(first) or not np.any(second):
            continue
        first_area = areas[frame_index]
        second_area = areas[frame_index + 1]
        adjacent_area_ratios.append(
            float(max(first_area, second_area) / min(first_area, second_area))
        )
        ious.append(
            float(np.count_nonzero(first & second) / max(1, np.count_nonzero(first | second)))
        )
    nonempty = areas[visible]
    return (
        {
            "visible_frames": int(np.count_nonzero(visible)),
            "total_frames": int(len(visible)),
            "visible_fraction": float(np.mean(visible)),
            "maximum_occlusion_gap_frames": _longest_false_run(visible),
            "area_pixels_min_nonempty": int(np.min(nonempty)),
            "area_pixels_max": int(np.max(nonempty)),
            "area_ratio_max_min_nonempty": float(np.max(nonempty) / np.min(nonempty)),
            "adjacent_area_ratio_max_visible": float(np.max(adjacent_area_ratios)),
            "centroid_step_max_pixels": float(np.max(steps)),
            "adjacent_iou_mean_visible": float(np.mean(ious)),
            "adjacent_iou_min_visible": float(np.min(ious)),
        },
        centroids,
    )


def _hs_histogram(cv2: Any, np: Any, frame: Any, mask: Any) -> Any:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], mask.astype(np.uint8), [18, 8], [0, 180, 0, 256]
    ).reshape(-1).astype(np.float64)
    return histogram / max(1e-9, float(np.linalg.norm(histogram)))


def _git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args], cwd=project_root, capture_output=True, text=True
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {"head": run("rev-parse", "HEAD"), "status_porcelain": run("status", "--porcelain")}


def main() -> int:
    args = _parser().parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    source_video = args.source_video.expanduser().resolve()
    candidate_video = args.candidate_video.expanduser().resolve()
    aligned_path = args.aligned_evaluation.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for path in (experiment, config_path, source_video, candidate_video, aligned_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {output}")

    import cv2
    import numpy as np

    config = json.loads(config_path.read_text())
    inherited_config_path = None
    if "inherits" in config:
        inherited_config_path = (config_path.parent / config["inherits"]).resolve()
        inherited = json.loads(inherited_config_path.read_text())
        inherited["thresholds"].update(config.get("threshold_overrides", {}))
        inherited["schema_version"] = config.get("schema_version")
        inherited["inheritance"] = {
            "base_config": str(inherited_config_path),
            "threshold_overrides": config.get("threshold_overrides", {}),
        }
        config = inherited
    if config.get("schema_version") != "1.0.0":
        raise ValueError("strict gate config must use schema_version 1.0.0")
    if config.get("coordinate_frame") != "camera:source_video_pixels":
        raise ValueError("strict gate coordinates must use camera:source_video_pixels")
    candidate_hash = _sha256(candidate_video)
    if candidate_hash != config["candidate_sha256"]:
        raise ValueError("candidate hash does not match the strict gate contract")

    source_frames, source_fps = _load_video(cv2, source_video)
    candidate_frames, candidate_fps = _load_video(cv2, candidate_video)
    expected_indices = list(range(*config["window_local_frame_range"]))
    if len(source_frames) != len(expected_indices) or len(candidate_frames) != len(expected_indices):
        raise ValueError("source, candidate, and strict gate frame counts must match")
    if abs(source_fps - candidate_fps) > 1e-6:
        raise ValueError("source and candidate FPS must match")
    height, width = source_frames[0].shape[:2]
    if candidate_frames[0].shape[:2] != (height, width):
        raise ValueError("source and candidate dimensions must match")

    run_cache: dict[str, dict[str, Any]] = {}

    def track(run_name: str) -> dict[str, Any]:
        if run_name not in run_cache:
            run_cache[run_name] = _load_track_run(np, experiment, run_name)
            loaded = run_cache[run_name]
            if loaded["indices"] != expected_indices:
                raise ValueError(f"track {run_name} frame indices do not match the gate")
            if (loaded["height"], loaded["width"]) != (height, width):
                raise ValueError(f"track {run_name} dimensions do not match the videos")
        return run_cache[run_name]

    thresholds = config["thresholds"]
    flower_metrics: dict[str, Any] = {}
    source_masks: dict[str, Any] = {}
    candidate_masks: dict[str, Any] = {}
    source_centroids: dict[str, Any] = {}
    candidate_centroids: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    flower_ids = [row["instance_id"] for row in config["flower_instances"]]
    for row in config["flower_instances"]:
        instance_id = row["instance_id"]
        source_run = track(row["source_track_run"])
        candidate_run = track(row["candidate_track_run"])
        if instance_id not in source_run["masks"] or instance_id not in candidate_run["masks"]:
            raise KeyError(f"missing named flower track: {instance_id}")
        source_masks[instance_id] = source_run["masks"][instance_id]
        candidate_masks[instance_id] = candidate_run["masks"][instance_id]
        source_metric, source_centroid = _track_metrics(np, source_masks[instance_id])
        candidate_metric, candidate_centroid = _track_metrics(np, candidate_masks[instance_id])
        source_centroids[instance_id] = source_centroid
        candidate_centroids[instance_id] = candidate_centroid

        delta = candidate_centroid - source_centroid
        median_offset = np.median(delta, axis=0)
        centered_residual = np.linalg.norm(delta - median_offset, axis=1)
        source_velocity = np.diff(source_centroid, axis=0)
        candidate_velocity = np.diff(candidate_centroid, axis=0)
        moving = np.linalg.norm(source_velocity, axis=1) > 0.2
        cosines = np.sum(source_velocity[moving] * candidate_velocity[moving], axis=1) / (
            np.linalg.norm(source_velocity[moving], axis=1)
            * np.linalg.norm(candidate_velocity[moving], axis=1)
            + 1e-6
        )
        histogram_cosines = []
        for frame_index in expected_indices:
            source_mask = source_masks[instance_id][frame_index]
            candidate_mask = candidate_masks[instance_id][frame_index]
            if not np.any(source_mask) or not np.any(candidate_mask):
                continue
            histogram_cosines.append(
                float(
                    _hs_histogram(cv2, np, source_frames[frame_index], source_mask)
                    @ _hs_histogram(cv2, np, candidate_frames[frame_index], candidate_mask)
                )
            )
        identity_metric = {
            "median_offset_xy_pixels": median_offset.tolist(),
            "offset_centered_trajectory_p90_pixels": float(
                np.percentile(centered_residual, 90)
            ),
            "positive_velocity_cosine_fraction": float(np.mean(cosines > 0)),
            "median_hs_histogram_cosine": float(np.median(histogram_cosines)),
            "histogram_comparison_frames": len(histogram_cosines),
        }
        flower_metrics[instance_id] = {
            "source": source_metric,
            "candidate": candidate_metric,
            "source_candidate_identity": identity_metric,
        }
        prefix = f"flower:{instance_id}"
        gates[f"{prefix}:source_all_frames"] = source_metric["visible_fraction"] == 1.0
        gates[f"{prefix}:candidate_visible_fraction"] = (
            candidate_metric["visible_fraction"]
            >= thresholds["minimum_candidate_visible_fraction"]
        )
        gates[f"{prefix}:occlusion_gap"] = (
            candidate_metric["maximum_occlusion_gap_frames"]
            <= row["maximum_occlusion_gap_frames"]
        )
        gates[f"{prefix}:centroid_step"] = (
            candidate_metric["centroid_step_max_pixels"]
            <= thresholds["maximum_flower_centroid_step_pixels"]
        )
        gates[f"{prefix}:adjacent_iou"] = (
            candidate_metric["adjacent_iou_mean_visible"]
            >= thresholds["minimum_flower_adjacent_iou_mean"]
        )
        gates[f"{prefix}:adjacent_area_ratio"] = (
            candidate_metric["adjacent_area_ratio_max_visible"]
            <= thresholds["maximum_adjacent_area_ratio"]
        )
        gates[f"{prefix}:trajectory"] = (
            identity_metric["offset_centered_trajectory_p90_pixels"]
            <= thresholds["maximum_offset_centered_trajectory_p90_pixels"]
        )
        gates[f"{prefix}:velocity_direction"] = (
            identity_metric["positive_velocity_cosine_fraction"]
            >= thresholds["minimum_positive_velocity_cosine_fraction"]
        )
        gates[f"{prefix}:color_identity"] = (
            identity_metric["median_hs_histogram_cosine"]
            >= thresholds["minimum_median_hs_histogram_cosine"]
        )

    pairwise_metrics = {}
    for left_index, left_id in enumerate(flower_ids):
        for right_id in flower_ids[left_index + 1 :]:
            source_order = np.sign(
                source_centroids[right_id][:, 0] - source_centroids[left_id][:, 0]
            )
            candidate_order = np.sign(
                candidate_centroids[right_id][:, 0] - candidate_centroids[left_id][:, 0]
            )
            agreement = float(np.mean(source_order == candidate_order))
            key = f"{left_id}__{right_id}"
            pairwise_metrics[key] = {"horizontal_order_agreement": agreement}
            gates[f"pairwise:{key}:horizontal_order"] = (
                agreement >= thresholds["minimum_pairwise_horizontal_order_agreement"]
            )

    hand_run = track(config["robot_hand_track_run"])
    hand_metrics = {}
    hand_masks = {}
    for hand_id in config["robot_hands"]:
        if hand_id not in hand_run["masks"]:
            raise KeyError(f"missing robot hand track: {hand_id}")
        hand_masks[hand_id] = hand_run["masks"][hand_id]
        metric, _ = _track_metrics(np, hand_masks[hand_id])
        hand_metrics[hand_id] = metric
        prefix = f"hand:{hand_id}"
        gates[f"{prefix}:all_frames"] = metric["visible_fraction"] == 1.0
        gates[f"{prefix}:centroid_step"] = (
            metric["centroid_step_max_pixels"]
            <= thresholds["maximum_hand_centroid_step_pixels"]
        )
        gates[f"{prefix}:adjacent_iou"] = (
            metric["adjacent_iou_mean_visible"]
            >= thresholds["minimum_hand_adjacent_iou_mean"]
        )

    contact_rows = []
    contact_metrics = []
    for segment_index, segment in enumerate(config["contact_segments"]):
        hand_id = segment["hand_id"]
        flower_id = segment["flower_instance_id"]
        start = segment["local_frame_start"]
        end = segment["local_frame_end_exclusive"]
        distances = []
        visible = 0
        for local_frame in range(start, end):
            hand_mask = hand_masks[hand_id][local_frame]
            flower_mask = candidate_masks[flower_id][local_frame]
            distance = None
            if np.any(hand_mask) and np.any(flower_mask):
                visible += 1
                field = cv2.distanceTransform(
                    (~hand_mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
                )
                distance = float(np.min(field[flower_mask]))
                distances.append(distance)
            contact_rows.append(
                {
                    "segment_index": segment_index,
                    "local_frame": local_frame,
                    "global_source_frame": config["global_source_frame_offset"] + local_frame,
                    "hand_id": hand_id,
                    "flower_instance_id": flower_id,
                    "distance_pixels": distance,
                    "flower_visible": bool(np.any(flower_mask)),
                }
            )
        visible_fraction = visible / (end - start)
        p90 = float(np.percentile(distances, 90))
        maximum = float(np.max(distances))
        metric = {
            **segment,
            "visible_frames": visible,
            "total_frames": end - start,
            "visible_fraction": visible_fraction,
            "distance_median_pixels": float(np.median(distances)),
            "distance_p90_pixels": p90,
            "distance_max_pixels": maximum,
        }
        contact_metrics.append(metric)
        prefix = f"contact:{segment_index}:{hand_id}:{flower_id}"
        gates[f"{prefix}:visibility"] = (
            visible_fraction >= thresholds["minimum_contact_visible_fraction"]
        )
        gates[f"{prefix}:p90_distance"] = (
            p90 <= thresholds["maximum_contact_distance_p90_pixels"]
        )
        gates[f"{prefix}:max_distance"] = (
            maximum <= thresholds["maximum_contact_distance_pixels"]
        )

    aligned = json.loads(aligned_path.read_text())
    aligned_metrics = aligned["metrics"]
    gates["aligned:identity"] = (
        aligned_metrics["target_identity"] >= thresholds["minimum_aligned_identity_score"]
    )
    gates["aligned:motion"] = (
        aligned_metrics["motion_preservation"] >= thresholds["minimum_aligned_motion_score"]
    )
    gates["aligned:temporal"] = (
        aligned_metrics["temporal_consistency"] >= thresholds["minimum_aligned_temporal_score"]
    )
    review = config["semantic_review"]
    reviewed = review["reviewed_local_frames"]
    gates["semantic:dense_review"] = (
        len(reviewed) >= 40 and reviewed[0] == 0 and reviewed[-1] == expected_indices[-1]
    )
    for key in (
        "complete_human_removal",
        "two_mechanical_hands_visible",
        "named_flower_identity_order_preserved",
        "contact_pairs_visually_plausible",
    ):
        gates[f"semantic:{key}"] = review.get(key) is True

    output.mkdir(parents=True)
    review_dir = output / "review"
    review_dir.mkdir()
    colors = {
        "cream-flower-left-01": (30, 220, 255),
        "gold-flower-middle-03": (0, 165, 255),
        "pink-flower-right-04": (220, 80, 220),
        "lime-flower-middle-02": (80, 235, 100),
        "pink-flower-right-03": (220, 80, 220),
        "robot-hand-upper-left-01": (255, 180, 30),
        "robot-hand-lower-right-02": (80, 255, 80),
    }

    def render(frame_index: int) -> Any:
        frame = candidate_frames[frame_index].copy()
        for instance_id, masks in {**candidate_masks, **hand_masks}.items():
            mask = masks[frame_index]
            if not np.any(mask):
                continue
            color = np.asarray(colors.get(instance_id, (255, 255, 255)), dtype=np.float64)
            frame[mask] = (0.58 * frame[mask] + 0.42 * color).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, tuple(int(v) for v in color), 2)
        active = [
            row
            for row in config["contact_segments"]
            if row["local_frame_start"] <= frame_index < row["local_frame_end_exclusive"]
        ]
        cv2.putText(
            frame,
            f"local {frame_index:02d} / source {config['global_source_frame_offset'] + frame_index}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        for row_index, row in enumerate(active):
            cv2.putText(
                frame,
                f"{row['hand_id']} -> {row['flower_instance_id']}",
                (12, 49 + row_index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return frame

    writer = cv2.VideoWriter(
        str(review_dir / "strict-object-contact-overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        candidate_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("could not create review video")
    for frame_index in expected_indices:
        writer.write(render(frame_index))
    writer.release()

    review_indices = np.unique(
        np.rint(np.linspace(0, len(expected_indices) - 1, 24)).astype(np.int32)
    )
    cells = [
        cv2.resize(render(int(index)), (448, 256), interpolation=cv2.INTER_AREA)
        for index in review_indices
    ]
    sheet = cv2.vconcat(
        [cv2.hconcat(cells[index : index + 4]) for index in range(0, len(cells), 4)]
    )
    cv2.imwrite(str(review_dir / "strict-object-contact-storyboard.jpg"), sheet)

    all_passed = all(gates.values())
    geometry_gates = {
        key: value for key, value in gates.items() if key != "aligned:temporal"
    }
    geometry_all_passed = all(geometry_gates.values())
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if all_passed else "PARTIAL",
        "decision": "ALLOW_FULL_FILM_EXPANSION" if all_passed else "HOLD_FULL_FILM_EXPANSION",
        "all_gates_passed": all_passed,
        "geometry_all_gates_passed": geometry_all_passed,
        "geometry_decision": (
            "ALLOW_TEMPORAL_REPAIR_ONLY"
            if geometry_all_passed and not all_passed
            else "GEOMETRY_ACCEPTED"
            if geometry_all_passed
            else "HOLD_GEOMETRY"
        ),
        "coordinate_frame": config["coordinate_frame"],
        "window_local_frame_range": config["window_local_frame_range"],
        "global_source_frame_range_inclusive": [
            config["global_source_frame_offset"],
            config["global_source_frame_offset"] + expected_indices[-1],
        ],
        "flower_metrics": flower_metrics,
        "pairwise_metrics": pairwise_metrics,
        "hand_metrics": hand_metrics,
        "contact_metrics": contact_metrics,
        "aligned_metrics": {
            "target_identity": aligned_metrics["target_identity"],
            "motion_preservation": aligned_metrics["motion_preservation"],
            "temporal_consistency": aligned_metrics["temporal_consistency"],
        },
        "semantic_review": review,
        "thresholds": thresholds,
        "gates": gates,
        "geometry_gates": geometry_gates,
        "limitations": [
            "Three visually separable held flowers are tracked; this is not a claim that every background flower has an instance ID.",
            "The cream flower has an explicit two-frame occlusion and is re-identified with a second prompt; empty masks are preserved during the occlusion.",
            "Contact is image-plane mask distance in camera pixels, not force or 3D tactile sensing.",
            "Semantic removal checks are a dense Codex visual audit, not external human annotation."
        ],
    }
    (output / "gate-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "contact-pairs.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "coordinate_frame": config["coordinate_frame"],
                "segments": contact_metrics,
                "frames": contact_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    project_root = Path(__file__).resolve().parents[1]
    artifacts = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts[str(path.relative_to(output))] = {
                "path": str(path),
                "sha256": _sha256(path),
            }
    manifest = {
        "schema_version": "1.0.0",
        "created_at": report["created_at"],
        "status": report["status"],
        "decision": report["decision"],
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {"numpy": np.__version__, "opencv": cv2.__version__},
        "git": _git_state(project_root),
        "inputs": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "inherited_config": (
                {
                    "path": str(inherited_config_path),
                    "sha256": _sha256(inherited_config_path),
                }
                if inherited_config_path is not None
                else None
            ),
            "source_video": {"path": str(source_video), "sha256": _sha256(source_video)},
            "candidate_video": {"path": str(candidate_video), "sha256": candidate_hash},
            "aligned_evaluation": {"path": str(aligned_path), "sha256": _sha256(aligned_path)},
            "track_runs": {
                name: {
                    "manifest": str(row["manifest_path"]),
                    "manifest_sha256": _sha256(row["manifest_path"]),
                    "packed_masks": str(row["packed_path"]),
                    "packed_masks_sha256": _sha256(row["packed_path"]),
                }
                for name, row in sorted(run_cache.items())
            },
        },
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output), "status": report["status"], "failed_gates": [key for key, value in gates.items() if not value]}, indent=2))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
