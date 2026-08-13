#!/usr/bin/env python3
"""Build real flower/contact paired supervision and a geometry-only candidate.

The H3 candidate supplies the already reviewed complete robot replacement.  One
explicitly tracked source flower is restored by immutable instance ID, while
the generated hand remains in front at the measured contact pixels.  The same
frames are exported as a target/control/mask pair for later task adaptation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--robot-candidate-video", type=Path, required=True)
    parser.add_argument("--stem-instances", type=Path, required=True)
    parser.add_argument("--source-pose-limbs", type=Path, required=True)
    parser.add_argument("--source-flower-union", type=Path, required=True)
    parser.add_argument("--source-person-masks", type=Path, required=True)
    parser.add_argument("--robot-hand-instances", type=Path, required=True)
    parser.add_argument("--expected-stem-id", default="active-pink-stem-01")
    parser.add_argument(
        "--expected-hand-ids",
        nargs=2,
        default=["robot-hand-upper-01", "robot-hand-lower-02"],
    )
    parser.add_argument(
        "--source-frame-range",
        type=int,
        nargs=2,
        metavar=("START", "END_EXCLUSIVE"),
        help="Optional contiguous subset of the aligned instance tracks",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--hand-radius", type=int, default=34)
    parser.add_argument("--contact-threshold", type=float, default=8.0)
    parser.add_argument(
        "--support-contact-threshold",
        type=float,
        help="Optional separate maximum distance for support-hand/bouquet contact",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unpack(np: Any, path: Path, key: str, frames: int, height: int, width: int) -> Any:
    payload = np.load(path)
    packed = payload[key]
    unpacked = np.unpackbits(packed, axis=-1, bitorder=str(payload["bitorder"]))
    return unpacked[..., : height * width].reshape(frames, height, width).astype(bool)


def closest_mask_points(cv2: Any, np: Any, first: Any, second: Any) -> dict[str, Any]:
    if not bool(first.any()) or not bool(second.any()):
        raise ValueError("closest-point masks must both be non-empty")
    overlap = first & second
    if bool(overlap.any()):
        ys, xs = np.where(overlap)
        point = [float(np.mean(xs)), float(np.mean(ys))]
        return {
            "distance_pixels": 0.0,
            "first_xy": point,
            "second_xy": point,
            "overlap_pixels": int(np.count_nonzero(overlap)),
        }
    distance_to_second, labels = cv2.distanceTransformWithLabels(
        (~second).astype(np.uint8),
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    first_y, first_x = np.where(first)
    values = distance_to_second[first]
    selected = int(np.argmin(values))
    point_first = (int(first_x[selected]), int(first_y[selected]))
    label = int(labels[point_first[1], point_first[0]])
    second_y, second_x = np.where(second)
    # OpenCV labels foreground-zero pixels in scan order starting from one.
    label_index = max(0, min(len(second_x) - 1, label - 1))
    point_second = (int(second_x[label_index]), int(second_y[label_index]))
    return {
        "distance_pixels": float(values[selected]),
        "first_xy": [float(point_first[0]), float(point_first[1])],
        "second_xy": [float(point_second[0]), float(point_second[1])],
        "overlap_pixels": 0,
    }


def compose_exact_flower_instance(
    cv2: Any,
    np: Any,
    source: Any,
    candidate: Any,
    flower_mask: Any,
    foreground_hand: Any,
) -> tuple[Any, Any, Any]:
    """Restore the visible source instance, retaining robot pixels at grip overlap."""

    contact_band = np.logical_and(
        foreground_hand,
        cv2.dilate(flower_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0,
    )
    # Keep an explicit non-contact buffer.  On NumPy 2.x/Python 3.14, a
    # temporary ``~contact_band`` in a chained ufunc can reuse the input buffer,
    # changing the later identity audit from the contact band to its complement.
    non_contact = np.logical_not(contact_band).copy()
    restore = np.logical_and(flower_mask, non_contact)
    output = candidate.copy()
    output[restore] = source[restore]
    core = cv2.erode(flower_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    exact_audit = np.logical_and(core, non_contact)
    return output, restore, exact_audit


def _skin_like(cv2: Any, np: Any, frame: Any) -> Any:
    _, cr, cb = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb))
    blue, green, red = cv2.split(frame.astype(np.float32))
    return (
        (cr >= 132)
        & (cr <= 180)
        & (cb >= 75)
        & (cb <= 135)
        & (red > green * 1.03)
        & (green > blue * 0.90)
    )


def _read_frames(cv2: Any, path: Path, indices: list[int]) -> tuple[list[Any], dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    info = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"decode failed at {path} frame {index}")
        frames.append(frame)
    capture.release()
    return frames, info


def _encode(ffmpeg: Path, np: Any, frames: Any, path: Path, fps: int, pixel_format: str) -> None:
    height, width = frames.shape[1:3]
    command = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pixel_format,
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
        "15",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    completed = subprocess.run(command, input=np.ascontiguousarray(frames).tobytes(), check=False)
    if completed.returncode:
        raise RuntimeError(f"ffmpeg failed to encode {path}")


def select_source_frame_positions(
    source_indices: list[int], frame_range: list[int] | None
) -> tuple[list[int], list[int]]:
    if frame_range is None:
        return source_indices, list(range(len(source_indices)))
    start, end = frame_range
    if start < 0 or end <= start:
        raise ValueError("source-frame-range must satisfy 0 <= START < END_EXCLUSIVE")
    positions_by_frame = {frame: position for position, frame in enumerate(source_indices)}
    requested = list(range(start, end))
    missing = [frame for frame in requested if frame not in positions_by_frame]
    if missing:
        raise ValueError(
            f"source-frame-range is not fully represented by the tracks; missing {missing[:8]}"
        )
    return requested, [positions_by_frame[frame] for frame in requested]


def _review_sheet(
    cv2: Any,
    np: Any,
    rows: list[list[Any]],
    labels: list[str],
    source_indices: list[int],
) -> Any:
    rendered_rows = []
    review_indices = np.unique(
        np.rint(np.linspace(0, len(rows) - 1, min(40, len(rows)))).astype(np.int32)
    )
    for local_index in review_indices:
        cells = rows[int(local_index)]
        rendered_cells = []
        for label, frame in zip(labels, cells):
            item = cv2.resize(frame, (416, 240), interpolation=cv2.INTER_AREA)
            cv2.putText(
                item,
                f"{label} source={source_indices[int(local_index)]}",
                (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            rendered_cells.append(item)
        rendered_rows.append(cv2.hconcat(rendered_cells))
    return cv2.vconcat(rendered_rows)


def main() -> int:
    args = _parser().parse_args()
    support_threshold = (
        args.contact_threshold
        if args.support_contact_threshold is None
        else args.support_contact_threshold
    )
    if (
        args.fps <= 0
        or args.hand_radius <= 0
        or args.contact_threshold < 0
        or support_threshold < 0
    ):
        raise ValueError("FPS/radius must be positive and contact threshold non-negative")
    paths = {
        "source_video": args.source_video.expanduser().resolve(),
        "robot_candidate_video": args.robot_candidate_video.expanduser().resolve(),
        "stem_instances": args.stem_instances.expanduser().resolve(),
        "source_pose_limbs": args.source_pose_limbs.expanduser().resolve(),
        "source_flower_union": args.source_flower_union.expanduser().resolve(),
        "source_person_masks": args.source_person_masks.expanduser().resolve(),
        "robot_hand_instances": args.robot_hand_instances.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty {name}: {path}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")

    import cv2
    import numpy as np

    stem_payload = np.load(paths["stem_instances"])
    full_source_indices = [
        int(value) for value in stem_payload["source_frame_indices"]
    ]
    source_indices, selected_positions = select_source_frame_positions(
        full_source_indices, args.source_frame_range
    )
    instance_ids = [str(value) for value in stem_payload["instance_ids"]]
    if instance_ids != [args.expected_stem_id]:
        raise RuntimeError(
            f"expected one {args.expected_stem_id} track, got {instance_ids}"
        )
    height, width = int(stem_payload["height"]), int(stem_payload["width"])
    stems = np.unpackbits(
        stem_payload["masks_packed"], axis=2, bitorder=str(stem_payload["bitorder"])
    )[..., : height * width].reshape(
        1, len(full_source_indices), height, width
    ).astype(bool)[0, selected_positions]
    source, source_info = _read_frames(cv2, paths["source_video"], source_indices)
    robot, robot_info = _read_frames(cv2, paths["robot_candidate_video"], source_indices)
    if source_info != robot_info or (width, height) != (
        source_info["width"], source_info["height"]
    ):
        raise RuntimeError("source, robot candidate, and instance coordinates are not aligned")

    pose = np.load(paths["source_pose_limbs"])
    ids = [int(value) for value in pose["landmark_ids"]]
    landmarks = pose["landmarks_xy"]
    shoulder_indices = [ids.index(11), ids.index(12)]
    pose_arms = _unpack(
        np, paths["source_pose_limbs"], "arms_packed", 660, height, width
    )[source_indices]
    hands_payload = np.load(paths["robot_hand_instances"])
    hand_instance_ids = [str(value) for value in hands_payload["instance_ids"]]
    expected_hand_ids = list(args.expected_hand_ids)
    if hand_instance_ids != expected_hand_ids:
        raise RuntimeError(
            f"expected prompted hands {expected_hand_ids}, got {hand_instance_ids}"
        )
    if [int(value) for value in hands_payload["source_frame_indices"]] != full_source_indices:
        raise RuntimeError("robot hand tracks and flower stem use different source frames")
    prompted_hands = np.unpackbits(
        hands_payload["masks_packed"],
        axis=2,
        bitorder=str(hands_payload["bitorder"]),
    )[..., : height * width].reshape(
        2, len(full_source_indices), height, width
    ).astype(bool)[:, selected_positions]
    flower_union = _unpack(
        np, paths["source_flower_union"], "packed", 660, height, width
    )[source_indices]
    person = _unpack(
        np, paths["source_person_masks"], "packed", 660, height, width
    )[source_indices]

    hand_rows = {name: [] for name in expected_hand_ids}
    hand_masks = {
        name: [mask for mask in prompted_hands[index]]
        for index, name in enumerate(expected_hand_ids)
    }
    for local_index, source_index in enumerate(source_indices):
        for hand_name in expected_hand_ids:
            mask = hand_masks[hand_name][local_index]
            ys, xs = np.where(mask)
            center = [float(np.mean(xs)), float(np.mean(ys))]
            hand_rows[hand_name].append(
                {
                    "center_xy": center,
                    "visible_pixels": int(np.count_nonzero(mask)),
                    "active_contact": closest_mask_points(cv2, np, mask, stems[local_index]),
                    "bouquet_contact": closest_mask_points(
                        cv2, np, mask, flower_union[local_index]
                    ),
                }
            )
    median_distances = {
        side: float(
            np.median([row["active_contact"]["distance_pixels"] for row in rows])
        )
        for side, rows in hand_rows.items()
    }
    active_side = min(median_distances, key=median_distances.get)
    support_side = next(name for name in expected_hand_ids if name != active_side)

    final_frames, controls, edit_masks, review_rows = [], [], [], []
    exact_fractions, skin_retention, source_skin_counts = [], [], []
    unexpected_person_edit_counts = []
    unexpected_person_edit_masks = []
    retained_skin_component_maxima = []
    retained_skin_component_rows = []
    contact_rows = []
    for local_index, source_index in enumerate(source_indices):
        foreground = hand_masks[active_side][local_index]
        final, restore, exact = compose_exact_flower_instance(
            cv2,
            np,
            source[local_index],
            robot[local_index],
            stems[local_index],
            foreground,
        )
        exact_fraction = float(
            np.mean(np.all(final[exact] == source[local_index][exact], axis=1))
        )
        exact_fractions.append(exact_fraction)
        shoulders = landmarks[source_index, shoulder_indices]
        shoulder_center = np.mean(shoulders, axis=0)
        shoulder_span = float(np.linalg.norm(shoulders[0] - shoulders[1]))
        face = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(
            face,
            (
                round(float(shoulder_center[0])),
                round(float(shoulder_center[1] - shoulder_span * 0.56)),
            ),
            (
                max(18, round(shoulder_span * 0.30)),
                max(24, round(shoulder_span * 0.42)),
            ),
            0,
            0,
            360,
            255,
            -1,
        )
        audited_human_region = pose_arms[local_index] | (face > 0)
        protected_interaction = (
            cv2.dilate(
                flower_union[local_index].astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
            )
            > 0
        )
        protected_interaction |= (
            cv2.dilate(
                stems[local_index].astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
            )
            > 0
        )
        protected_interaction |= hand_masks[expected_hand_ids[0]][local_index]
        protected_interaction |= hand_masks[expected_hand_ids[1]][local_index]
        unprotected_interaction = np.logical_not(protected_interaction).copy()
        source_skin = np.logical_and(
            np.logical_and(
                _skin_like(cv2, np, source[local_index]),
                person[local_index],
            ),
            np.logical_and(audited_human_region, unprotected_interaction),
        )
        source_skin_counts.append(int(np.count_nonzero(source_skin)))
        person_audit_region = np.logical_and(
            person[local_index], audited_human_region
        ).copy()
        person_audit_region = np.logical_and(
            person_audit_region, unprotected_interaction
        ).copy()
        unexpected_person_edit = np.logical_and(
            np.any(final != robot[local_index], axis=2).copy(),
            person_audit_region,
        )
        unexpected_person_edit_counts.append(
            int(np.count_nonzero(unexpected_person_edit))
        )
        unexpected_person_edit_masks.append(unexpected_person_edit)
        retained = np.all(
            np.abs(
                final.astype(np.int16) - source[local_index].astype(np.int16)
            )
            <= 12,
            axis=2,
        )
        skin_retention.append(
            float(np.count_nonzero(retained & source_skin) / max(1, np.count_nonzero(source_skin)))
        )
        retained_skin = retained & source_skin
        component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
            retained_skin.astype(np.uint8)
        )
        if component_count > 1:
            largest_component_index = max(
                range(1, component_count),
                key=lambda index: int(component_stats[index, cv2.CC_STAT_AREA]),
            )
            largest_retained_skin_component = int(
                component_stats[largest_component_index, cv2.CC_STAT_AREA]
            )
            largest_component_box = {
                "x": int(component_stats[largest_component_index, cv2.CC_STAT_LEFT]),
                "y": int(component_stats[largest_component_index, cv2.CC_STAT_TOP]),
                "width": int(component_stats[largest_component_index, cv2.CC_STAT_WIDTH]),
                "height": int(component_stats[largest_component_index, cv2.CC_STAT_HEIGHT]),
            }
        else:
            largest_retained_skin_component = 0
            largest_component_box = {"x": 0, "y": 0, "width": 0, "height": 0}
        retained_skin_component_maxima.append(largest_retained_skin_component)
        retained_skin_component_rows.append(
            {
                "local_frame": local_index,
                "source_frame": source_index,
                "largest_component_pixels": largest_retained_skin_component,
                "largest_component_box_xywh": largest_component_box,
            }
        )
        geometry = (
            hand_masks[expected_hand_ids[0]][local_index]
            | hand_masks[expected_hand_ids[1]][local_index]
        ).astype(np.uint8) * 255
        edges = cv2.Canny(geometry, 50, 130)
        control = np.repeat(edges[..., None], 3, axis=2)
        control[stems[local_index]] = np.asarray([40, 220, 40], dtype=np.uint8)
        active_contact = hand_rows[active_side][local_index]["active_contact"]
        support_contact = hand_rows[support_side][local_index]["bouquet_contact"]
        for point, color in (
            (active_contact["first_xy"], (30, 30, 255)),
            (support_contact["first_xy"], (255, 80, 30)),
        ):
            cv2.circle(control, tuple(round(value) for value in point), 8, color, -1)
        edit = person[local_index] & ~(
            cv2.dilate(stems[local_index].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        )
        overlay = final.copy()
        overlay[stems[local_index]] = np.rint(
            0.35 * overlay[stems[local_index]]
            + 0.65 * np.asarray([30, 40, 250])
        ).astype(np.uint8)
        for hand_name, color in zip(
            expected_hand_ids, ((255, 150, 30), (30, 220, 255))
        ):
            point = hand_rows[hand_name][local_index]["active_contact"]["first_xy"]
            cv2.circle(overlay, tuple(round(value) for value in point), 7, color, 2)
        human_audit = final.copy()
        human_audit[unexpected_person_edit] = np.asarray(
            [20, 20, 255], dtype=np.uint8
        )
        final_frames.append(final)
        controls.append(control)
        edit_masks.append(edit.astype(np.uint8) * 255)
        review_rows.append(
            [source[local_index], robot[local_index], overlay, human_audit, control]
        )
        contact_rows.append(
            {
                "local_frame": local_index,
                "source_frame": source_index,
                "active_instance_id": args.expected_stem_id,
                "active_hand": active_side,
                "support_hand": support_side,
                "occlusion_order": "robot_hand_front_at_contact; source_flower_front_elsewhere",
                expected_hand_ids[0]: hand_rows[expected_hand_ids[0]][local_index],
                expected_hand_ids[1]: hand_rows[expected_hand_ids[1]][local_index],
                "restored_instance_pixels": int(np.count_nonzero(restore)),
                "exact_identity_audit_pixels": int(np.count_nonzero(exact)),
            }
        )

    final_array = np.stack(final_frames)
    control_array = np.stack(controls)
    mask_array = np.stack(edit_masks)
    source_array = np.stack(source)
    output.mkdir(parents=True)
    candidate_path = output / "geometry-candidate.mp4"
    source_path = output / "source-window.mp4"
    control_path = output / "contact-control.mp4"
    edit_mask_path = output / "regional-edit-mask.mp4"
    reference_path = output / "robot-reference.png"
    human_audit_mask_path = output / "human-audit-person-region-edits-packed.npz"
    _encode(paths["ffmpeg"], np, final_array, candidate_path, args.fps, "bgr24")
    _encode(paths["ffmpeg"], np, source_array, source_path, args.fps, "bgr24")
    _encode(paths["ffmpeg"], np, control_array, control_path, args.fps, "bgr24")
    _encode(paths["ffmpeg"], np, mask_array, edit_mask_path, args.fps, "gray")
    cv2.imwrite(str(reference_path), final_array[0])
    person_edit_array = np.stack(unexpected_person_edit_masks).astype(np.uint8)
    np.savez_compressed(
        human_audit_mask_path,
        masks_packed=np.packbits(
            person_edit_array.reshape(len(source_indices), -1), axis=1, bitorder="little"
        ),
        source_frame_indices=np.asarray(source_indices, dtype=np.int32),
        height=np.asarray(height, dtype=np.int32),
        width=np.asarray(width, dtype=np.int32),
        bitorder=np.asarray("little"),
    )
    review_path = output / "geometry-contact-review.jpg"
    cv2.imwrite(
        str(review_path),
        _review_sheet(
            cv2,
            np,
            review_rows,
            ["SOURCE", "H3", "FINAL+CONTACT", "HUMAN-AUDIT", "CONTROL"],
            source_indices,
        ),
        [cv2.IMWRITE_JPEG_QUALITY, 93],
    )
    pairs_path = output / "contact-pairs.json"
    pairs_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "coordinate_frame": "camera:source_video_pixels",
                "active_hand": active_side,
                "support_hand": support_side,
                "frames": contact_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    active_distances = [
        float(row[active_side]["active_contact"]["distance_pixels"])
        for row in contact_rows
    ]
    support_distances = [
        float(row[support_side]["bouquet_contact"]["distance_pixels"])
        for row in contact_rows
    ]
    hand_minimums = {
        side: min(int(row["visible_pixels"]) for row in hand_rows[side])
        for side in expected_hand_ids
    }
    automatic_gates = {
        "complete_human_removal_proxy": max(unexpected_person_edit_counts) == 0,
        "two_robot_hands_visible_proxy": min(hand_minimums.values()) >= 80,
        "active_stem_contact_proxy": max(active_distances) <= args.contact_threshold,
        "support_bouquet_contact_proxy": max(support_distances) <= support_threshold,
        "active_flower_identity_exact_before_encode": min(exact_fractions) == 1.0,
    }
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "method": "immutable_h3_candidate_plus_real_stem_identity_and_contact_pairing",
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "seed": args.seed,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "gpu": {"used": False, "reason": "deterministic CPU supervision composition"},
        "coordinate_frames": {
            "image": "camera:source_video_pixels",
            "robot_masks": "camera:generated_video_pixels aligned to source",
            "flower": f"object:{args.expected_stem_id} observed in camera:source_video_pixels",
        },
        "source_frame_indices": source_indices,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
        },
        "contact_assignment": {
            "active_hand": active_side,
            "support_hand": support_side,
            "active_median_distance_by_side": median_distances,
            "active_contact_threshold_pixels": args.contact_threshold,
            "support_contact_threshold_pixels": support_threshold,
        },
        "metrics": {
            "active_contact_distance_max_pixels": max(active_distances),
            "active_contact_distance_mean_pixels": float(np.mean(active_distances)),
            "support_contact_distance_max_pixels": max(support_distances),
            "support_contact_distance_mean_pixels": float(np.mean(support_distances)),
            "robot_hand_visible_pixels_min": hand_minimums,
            "source_skin_retention_fraction_mean": float(np.mean(skin_retention)),
            "source_skin_retention_fraction_max": max(skin_retention),
            "audited_source_skin_pixels_min": min(source_skin_counts),
            "retained_source_skin_component_max_pixels": max(
                retained_skin_component_maxima
            ),
            "retained_source_skin_largest_components": retained_skin_component_rows,
            "unexpected_person_region_edit_pixels_max": max(
                unexpected_person_edit_counts
            ),
            "unexpected_person_region_edit_pixels_total": sum(
                unexpected_person_edit_counts
            ),
            "active_flower_exact_fraction_min_before_encode": min(exact_fractions),
            "edit_mask_fraction_min": float(np.min(np.mean(mask_array > 0, axis=(1, 2)))),
            "edit_mask_fraction_max": float(np.max(np.mean(mask_array > 0, axis=(1, 2)))),
        },
        "automatic_gates": automatic_gates,
        "all_automatic_gates_pass": all(automatic_gates.values()),
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in {
                "geometry_candidate": candidate_path,
                "source_window": source_path,
                "contact_control": control_path,
                "regional_edit_mask": edit_mask_path,
                "robot_reference": reference_path,
                "contact_pairs": pairs_path,
                "human_audit_mask": human_audit_mask_path,
                "review": review_path,
            }.items()
        },
        "paired_supervision": {
            "target_video": str(candidate_path),
            "control_video": str(control_path),
            "reference_image": str(reference_path),
            "regional_edit_mask": str(edit_mask_path),
            "training_scope": "window-specific real-scene geometry supervision; not held-out evidence",
        },
        "limitations": [
            "This supervised candidate is window-specific and does not establish generalization.",
            "The automatic person-region gate proves that composition did not edit the unprotected H3 person region; semantic acceptance of H3 human removal still requires human review.",
            "Source-color similarity is retained as a diagnostic only because robot surfaces can coincidentally match source skin pixels.",
            f"The support-hand bouquet contact uses the older union mask; only {args.expected_stem_id} has persistent instance identity."
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output), **manifest["metrics"], **automatic_gates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
