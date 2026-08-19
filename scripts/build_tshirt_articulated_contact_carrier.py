#!/usr/bin/env python3
"""Build a connected dual-arm contact carrier for T-shirt-folding H3 retakes."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.schema import ACWMActionCondition  # noqa: E402
from phiagent.harness.articulated_camera_rig import (  # noqa: E402
    compile_tshirt_dual_arm_trajectory,
)
from phiagent.harness.cloth_carrier import (  # noqa: E402
    TSHIRT_832X480_CARRIER,
    phase_progress,
    write_carrier_contract,
)
from phiagent.harness.provenance import capture_provenance, write_json_atomic  # noqa: E402
from phiagent.rendering.minimax_h3 import file_sha256  # noqa: E402
from scripts.build_tshirt_length_preserving_carrier import (  # noqa: E402
    _composite,
    _encode_carrier,
    _polygon_mask,
    _rotation_matrix,
    _warp,
)


LOWER_ROBOT_POLYGON = (
    (88, 480),
    (88, 386),
    (160, 365),
    (218, 335),
    (265, 305),
    (307, 273),
    (338, 252),
    (410, 255),
    (449, 270),
    (445, 291),
    (409, 310),
    (371, 336),
    (336, 351),
    (290, 366),
    (253, 386),
    (218, 419),
    (206, 480),
)

UPPER_ROBOT_POLYGON = (
    (510, 90),
    (568, 75),
    (602, 85),
    (624, 104),
    (652, 98),
    (690, 116),
    (700, 160),
    (685, 215),
    (675, 275),
    (667, 355),
    (610, 367),
    (580, 320),
    (570, 250),
    (548, 218),
    (515, 204),
    (510, 183),
    (530, 165),
    (548, 144),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-condition", type=Path, required=True)
    parser.add_argument("--clean-plate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser


def _capsule_mask(cv2, np, shape, first, second, radius):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    a = tuple(int(round(value)) for value in first)
    b = tuple(int(round(value)) for value in second)
    cv2.line(mask, a, b, 255, radius * 2, cv2.LINE_AA)
    cv2.circle(mask, a, radius, 255, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(mask, b, radius, 255, cv2.FILLED, cv2.LINE_AA)
    return mask


def _piece_masks(cv2, np, image, nodes, polygon, radii):
    if len(radii) != len(nodes) - 1:
        raise ValueError("robot piece radius count does not match its links")
    whole = _polygon_mask(cv2, np, image.shape, polygon)
    pieces = []
    for first, second, radius in zip(nodes, nodes[1:], radii):
        capsule = _capsule_mask(cv2, np, image.shape, first, second, radius)
        pieces.append(cv2.bitwise_and(whole, capsule))
    return whole, tuple(pieces)


def _segment_transform(np, source_a, source_b, target_a, target_b):
    source_vector = np.asarray(source_b, dtype=np.float64) - np.asarray(
        source_a, dtype=np.float64
    )
    target_vector = np.asarray(target_b, dtype=np.float64) - np.asarray(
        target_a, dtype=np.float64
    )
    source_length = float(np.linalg.norm(source_vector))
    target_length = float(np.linalg.norm(target_vector))
    if source_length <= 1e-6 or abs(target_length - source_length) > 1e-4:
        raise ValueError("articulated carrier requires fixed nonzero link lengths")
    source_angle = math.atan2(source_vector[1], source_vector[0])
    target_angle = math.atan2(target_vector[1], target_vector[0])
    angle = target_angle - source_angle
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    translation = np.asarray(target_a, dtype=np.float64) - rotation @ np.asarray(
        source_a, dtype=np.float64
    )
    return np.asarray(
        (
            (rotation[0, 0], rotation[0, 1], translation[0]),
            (rotation[1, 0], rotation[1, 1], translation[1]),
        ),
        dtype=np.float32,
    )


def _trajectory_payload(trajectory):
    rigs = {}
    for name, rig in trajectory.rigs.items():
        rigs[name] = {
            "rig_id": rig.rig_id,
            "coordinate_frame": rig.coordinate_frame,
            "reference_nodes_xy": [list(point) for point in rig.reference_nodes_xy],
            "link_lengths_pixels": list(rig.link_lengths_pixels),
            "frames": [
                {
                    "frame_index": frame.frame_index,
                    "nodes_xy": [list(point) for point in frame.nodes_xy],
                    "q_radians": list(frame.q_radians),
                    "qdot_radians_per_second": list(
                        frame.qdot_radians_per_second
                    ),
                    "target_tip_xy": list(frame.target_tip_xy),
                    "contact_entity": frame.contact_entity,
                }
                for frame in trajectory.frames[name]
            ],
        }
    gates = {
        "all_links_connected": trajectory.maximum_link_length_error_pixels < 1e-6,
        "link_lengths_conserved": trajectory.maximum_link_length_error_pixels < 1e-6,
        "joint_motion_continuous": trajectory.maximum_joint_step_radians < 0.20,
        "tip_motion_continuous": trajectory.maximum_tip_step_pixels < 18.0,
        "ik_targets_reached": trajectory.mean_tip_error_pixels < 1e-4,
        "left_contact_precedes_left_sleeve_motion": (
            trajectory.frames["lower_left"][20].contact_entity
            == "viewer_left_sleeve"
        ),
        "right_contact_precedes_right_sleeve_motion": (
            trajectory.frames["upper_right"][60].contact_entity
            == "viewer_right_sleeve"
        ),
    }
    return {
        "schema_version": "1.0.0",
        "method": "fixed-base-planar-FABRIK-camera-carrier",
        "coordinate_frame": trajectory.coordinate_frame,
        "fps": trajectory.fps,
        "frame_count": trajectory.frame_count,
        "rigs": rigs,
        "metrics": {
            "maximum_link_length_error_pixels": (
                trajectory.maximum_link_length_error_pixels
            ),
            "maximum_joint_step_radians": trajectory.maximum_joint_step_radians,
            "maximum_tip_step_pixels": trajectory.maximum_tip_step_pixels,
            "mean_tip_error_pixels": trajectory.mean_tip_error_pixels,
        },
        "gate_results": gates,
        "hard_gates_passed": all(gates.values()),
        "claim_boundary": (
            "Complete q/qdot belongs to two synthetic fixed-base planar camera rigs. "
            "It is an H3 motion condition, not the unidentified real robot asset, "
            "metric camera geometry, force, collision safety, or executable commands."
        ),
    }


def _render_frames(source: Path, clean_plate: Path):
    import cv2
    import numpy as np

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    clean = cv2.imread(str(clean_plate), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (480, 832):
        raise ValueError("carrier source must be a readable 832x480 image")
    if clean is None:
        raise ValueError("clean plate must be a readable image")
    clean = cv2.resize(clean, (832, 480), interpolation=cv2.INTER_LANCZOS4)
    geometry = TSHIRT_832X480_CARRIER
    trajectory = compile_tshirt_dual_arm_trajectory()
    left_mask = _polygon_mask(cv2, np, image.shape, geometry.viewer_left_polygon)
    right_mask = _polygon_mask(cv2, np, image.shape, geometry.viewer_right_polygon)
    body_mask = _polygon_mask(cv2, np, image.shape, geometry.body_polygon)
    lower_rig = trajectory.rigs["lower_left"]
    upper_rig = trajectory.rigs["upper_right"]
    lower_mask, lower_pieces = _piece_masks(
        cv2,
        np,
        image,
        lower_rig.reference_nodes_xy,
        LOWER_ROBOT_POLYGON,
        (52, 46, 38, 46),
    )
    upper_mask, upper_pieces = _piece_masks(
        cv2,
        np,
        image,
        upper_rig.reference_nodes_xy,
        UPPER_ROBOT_POLYGON,
        (62, 44, 46),
    )
    union = np.maximum.reduce(
        (left_mask, right_mask, body_mask, lower_mask, upper_mask)
    )
    union = cv2.GaussianBlur(union, (11, 11), 0)
    background = _composite(np, image, clean, union)
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
        canvas = background.copy()
        if body <= 0.0:
            layer, alpha = _warp(cv2, image, body_mask, upper_matrix, size)
            canvas = _composite(np, canvas, layer, alpha)
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

        for name, rig, pieces in (
            ("lower_left", lower_rig, lower_pieces),
            ("upper_right", upper_rig, upper_pieces),
        ):
            target_nodes = trajectory.frames[name][frame_index].nodes_xy
            for source_a, source_b, target_a, target_b, piece in zip(
                rig.reference_nodes_xy,
                rig.reference_nodes_xy[1:],
                target_nodes,
                target_nodes[1:],
                pieces,
            ):
                matrix = _segment_transform(
                    np, source_a, source_b, target_a, target_b
                )
                layer, alpha = _warp(cv2, image, piece, matrix, size)
                canvas = _composite(np, canvas, layer, alpha)
        frames.append(canvas)
    frames[0] = image.copy()
    return frames, trajectory, union


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    base = args.base_condition.expanduser().resolve()
    clean_plate = args.clean_plate.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"articulated carrier output already exists: {output}")
    if not clean_plate.is_file() or clean_plate.stat().st_size == 0:
        raise ValueError("clean plate does not exist or is empty")
    manifest = json.loads((base / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("base condition manifest must contain one JSON object")
    shutil.copytree(base, output)
    source = output / str(manifest["first_frame"])
    saved_clean = output / "control" / "imagegen-clean-plate.png"
    shutil.copy2(clean_plate, saved_clean)
    frames, trajectory, _ = _render_frames(source, saved_clean)
    carrier = output / "control" / "articulated-contact-cloth-carrier.mp4"
    ffmpeg_command = _encode_carrier(frames, carrier)
    cloth_contract = output / "control" / "carrier-contract.json"
    write_carrier_contract(cloth_contract, TSHIRT_832X480_CARRIER)
    trajectory_path = output / "control" / "articulated-trajectory.json"
    trajectory_payload = _trajectory_payload(trajectory)
    if not trajectory_payload["hard_gates_passed"]:
        raise RuntimeError("articulated carrier trajectory failed a frozen preflight gate")
    write_json_atomic(trajectory_path, trajectory_payload)
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
        "<Video 1> is a connected dual-arm contact-and-cloth motion carrier in the named camera frame. Transfer its fixed-base articulated joint motion, gripper approach, contact timing, coupled left-sleeve and right-sleeve paths, body fold, and final viewer-left transport frame by frame. Improve its compositing from the two pictures while preserving every joint connection and cloth material identity. Never render a detached limb, floating gripper, self-moving cloth, carrier seam, hole, or guide mark.",
    )
    prompt_path = output / str(variant["prompt_file"])
    prompt_path.write_text(prompt)
    manifest.update(
        {
            **capture_provenance(project_root, [sys.executable, *sys.argv], args.seed),
            "status": "articulated_contact_carrier_compiled",
            "honest_status": "NOT STARTED",
            "method": "connected_planar_dual_arm_plus_rigid_sleeves_plus_hash_plan",
            "clean_plate": str(saved_clean.relative_to(output)),
            "clean_plate_sha256": file_sha256(saved_clean),
            "carrier_contract": str(cloth_contract.relative_to(output)),
            "carrier_contract_sha256": file_sha256(cloth_contract),
            "articulated_trajectory": str(trajectory_path.relative_to(output)),
            "articulated_trajectory_sha256": file_sha256(trajectory_path),
            "articulated_trajectory_gates": trajectory_payload["gate_results"],
            "carrier_video": str(carrier.relative_to(output)),
            "carrier_video_sha256": file_sha256(carrier),
            "carrier_ffmpeg_command": ffmpeg_command,
            "claim_boundary": trajectory_payload["claim_boundary"],
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
                "trajectory": str(trajectory_path),
                "trajectory_sha256": file_sha256(trajectory_path),
                "trajectory_metrics": trajectory_payload["metrics"],
                "trajectory_gates": trajectory_payload["gate_results"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
