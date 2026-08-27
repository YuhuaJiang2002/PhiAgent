#!/usr/bin/env python3
"""Fuse a posture-stable RM65 proposal with a camera-ray-aligned proposal.

This is intentionally a refinement stage: monocular RGB fixes an image ray but
does not determine metric depth.  Blending the ray solution with the stable
planar proposal retains the visible motion while regularising the hidden depth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recover_rm65_synchronized_state import _recover_q
from render_realman_rm65_visual_replay import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-state", type=Path, required=True)
    parser.add_argument("--ray-state", type=Path, required=True)
    parser.add_argument("--rm65-urdf", type=Path, required=True)
    parser.add_argument("--ag2f90c-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blend", type=float, required=True)
    parser.add_argument(
        "--target-z-offset",
        type=float,
        default=0.0,
        help="Vertical EEF offset applied after stable/ray fusion and before workspace clipping.",
    )
    parser.add_argument("--left-target-z-offset", type=float, help="Override the common EEF Z offset for the left arm.")
    parser.add_argument("--right-target-z-offset", type=float, help="Override the common EEF Z offset for the right arm.")
    parser.add_argument(
        "--eef-z-keyframes",
        type=Path,
        help="Reviewed frame/left_z/right_z anchors that override monocular EEF height while retaining its XY path.",
    )
    parser.add_argument("--z-range", nargs=2, type=float, default=(0.04, 0.24))
    parser.add_argument("--left-base", nargs=3, type=float, required=True)
    parser.add_argument("--right-base", nargs=3, type=float, required=True)
    parser.add_argument("--left-base-rpy", nargs=3, type=float, required=True)
    parser.add_argument("--right-base-rpy", nargs=3, type=float, required=True)
    parser.add_argument("--left-seed", nargs=6, type=float, required=True)
    parser.add_argument("--right-seed", nargs=6, type=float, required=True)
    parser.add_argument("--rotation-weight", type=float, default=0.02)
    parser.add_argument("--posture-weight", type=float, default=5e-6)
    parser.add_argument("--table-half-size", nargs=2, type=float, default=(0.55, 0.35))
    parser.add_argument("--table-center-y", type=float, default=0.10)
    return parser.parse_args()


def temporal_metrics(q: np.ndarray, fps: float) -> dict[str, float]:
    velocity = np.diff(q, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    return {
        "max_joint_velocity_rad_s": float(np.abs(velocity).max()),
        "max_joint_acceleration_rad_s2": float(np.abs(acceleration).max()),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.blend <= 1.0:
        raise ValueError("--blend must be in [0, 1]")
    stable = np.load(args.stable_state)
    ray = np.load(args.ray_state)
    if stable["left_target_xyz"].shape != ray["left_target_xyz"].shape:
        raise ValueError("candidate state shapes differ")

    targets = {}
    for side in ("left", "right"):
        key = f"{side}_target_xyz"
        targets[side] = stable[key] + args.blend * (ray[key] - stable[key])
        side_offset = getattr(args, f"{side}_target_z_offset")
        targets[side][:, 2] += args.target_z_offset if side_offset is None else side_offset
        targets[side][:, 2] = np.clip(targets[side][:, 2], *args.z_range)
    if args.eef_z_keyframes:
        height_anchors = json.loads(args.eef_z_keyframes.read_text())
        anchor_frames = np.asarray(height_anchors["frames"], dtype=int)
        if anchor_frames[0] != 0 or anchor_frames[-1] != len(targets["left"]) - 1:
            raise ValueError("EEF Z keyframes must include the first and last state frames")
        if np.any(np.diff(anchor_frames) <= 0):
            raise ValueError("EEF Z keyframe frames must be strictly increasing")
        for side in ("left", "right"):
            values = np.asarray(height_anchors[f"{side}_z_m"], dtype=np.float64)
            if values.shape != anchor_frames.shape:
                raise ValueError(f"{side}_z_m must match frames")
            targets[side][:, 2] = np.interp(
                np.arange(len(targets[side])), anchor_frames, values
            )

    model = build_model(
        args.rm65_urdf,
        args.ag2f90c_dir,
        False,
        tuple(args.left_base),
        args.left_base_rpy[2],
        tuple(args.right_base),
        args.right_base_rpy[2],
        args.left_base_rpy[0],
        args.left_base_rpy[1],
        args.right_base_rpy[0],
        args.right_base_rpy[1],
        tuple(args.table_half_size),
        args.table_center_y,
    )
    left_q, left_pos, left_rot = _recover_q(
        model, targets["left"], "left", np.asarray(args.left_base),
        args.rotation_weight, np.asarray(args.left_seed), args.posture_weight,
    )
    right_q, right_pos, right_rot = _recover_q(
        model, targets["right"], "right", np.asarray(args.right_base),
        args.rotation_weight, np.asarray(args.right_seed), args.posture_weight,
    )
    fps = float(stable["fps"])
    output = args.output_dir / "rm65_synchronized_state.npz"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {key: stable[key] for key in stable.files}
    payload.update(
        left_q=left_q,
        right_q=right_q,
        left_target_xyz=targets["left"],
        right_target_xyz=targets["right"],
    )
    np.savez_compressed(output, **payload)
    manifest = {
        "schema_version": "phiagent-rm65-depth-regularized-state/1.0",
        "stable_state": str(args.stable_state),
        "ray_state": str(args.ray_state),
        "blend": args.blend,
        "target_z_offset_m": args.target_z_offset,
        "left_target_z_offset_m": args.left_target_z_offset,
        "right_target_z_offset_m": args.right_target_z_offset,
        "eef_z_keyframes": str(args.eef_z_keyframes) if args.eef_z_keyframes else None,
        "z_range_m": args.z_range,
        "frames": int(len(left_q)),
        "fps": fps,
        "base_poses": {
            "left_xyz_rpy": [*args.left_base, *args.left_base_rpy],
            "right_xyz_rpy": [*args.right_base, *args.right_base_rpy],
        },
        "ik_position_error_m": {
            "left_mean": float(left_pos.mean()), "left_max": float(left_pos.max()),
            "right_mean": float(right_pos.mean()), "right_max": float(right_pos.max()),
        },
        "ik_orientation_error_deg": {
            "left_mean": float(np.degrees(left_rot).mean()), "left_max": float(np.degrees(left_rot).max()),
            "right_mean": float(np.degrees(right_rot).mean()), "right_max": float(np.degrees(right_rot).max()),
        },
        "temporal": {
            "left": temporal_metrics(left_q, fps),
            "right": temporal_metrics(right_q, fps),
        },
        "state_npz": str(output),
        "claim_boundary": "source-conditioned monocular visual replay; depth is regularized, not measured metric ground truth",
    }
    (args.output_dir / "state_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
