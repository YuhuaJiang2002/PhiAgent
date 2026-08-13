#!/usr/bin/env python3
"""Render a calibrated DROID wrist-to-exterior depth-splat lower bound."""

from __future__ import annotations

import argparse
import hashlib
import json
import locale
import math
import os
import platform
import shlex
import socket
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_droid_raw_sequence_alignment import (  # noqa: E402
    _git_state,
    _package_versions,
    _strict_process_snapshot,
)
from phiagent.rendering.wan_animate import (  # noqa: E402
    acquire_gpu_lease,
    query_gpus,
    select_gpu,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def nearest_timestamp_index(values: Sequence[int], query: int) -> int:
    if not values:
        raise ValueError("timestamp sequence cannot be empty")
    return min(range(len(values)), key=lambda index: abs(int(values[index]) - int(query)))


def _rotation_xyz(np: Any, values: Sequence[float]) -> Any:
    x, y, z = (float(value) for value in values)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    return np.asarray(
        [
            [cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy],
            [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
            [-sy, cy * sx, cx * cy],
        ],
        dtype=np.float64,
    )


def _transform(np: Any, pose: Sequence[float]) -> Any:
    values = tuple(float(value) for value in pose)
    if len(values) != 6 or any(not math.isfinite(value) for value in values):
        raise ValueError("camera extrinsic pose must contain six finite values")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rotation_xyz(np, values[3:])
    result[:3, 3] = values[:3]
    return result


def _intrinsics(np: Any, payload: dict[str, Any]) -> Any:
    return np.asarray(
        [
            [float(payload["fx"]), 0.0, float(payload["cx"])],
            [0.0, float(payload["fy"]), float(payload["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _depth_splat(
    np: Any,
    source_bgr: Any,
    source_depth: Any,
    source_intrinsics: Any,
    target_intrinsics: Any,
    target_t_source: Any,
    target_shape: tuple[int, int],
) -> tuple[Any, Any, Any]:
    height, width = source_depth.shape
    target_height, target_width = target_shape
    y, x = np.mgrid[0:height, 0:width]
    finite = np.isfinite(source_depth) & (source_depth > 0)
    z = source_depth[finite].astype(np.float64)
    source_points = np.stack(
        [
            (x[finite] - source_intrinsics[0, 2]) * z / source_intrinsics[0, 0],
            (y[finite] - source_intrinsics[1, 2]) * z / source_intrinsics[1, 1],
            z,
            np.ones_like(z),
        ],
        axis=0,
    )
    target_points = target_t_source @ source_points
    target_z = target_points[2]
    in_front = target_z > 1e-6
    target_points = target_points[:, in_front]
    target_z = target_z[in_front]
    colors = source_bgr[finite][in_front]
    u = np.rint(
        target_intrinsics[0, 0] * target_points[0] / target_z
        + target_intrinsics[0, 2]
    ).astype(np.int64)
    v = np.rint(
        target_intrinsics[1, 1] * target_points[1] / target_z
        + target_intrinsics[1, 2]
    ).astype(np.int64)
    inside = (u >= 0) & (u < target_width) & (v >= 0) & (v < target_height)
    u, v, target_z, colors = u[inside], v[inside], target_z[inside], colors[inside]
    flat_index = v * target_width + u
    z_buffer = np.full(target_height * target_width, np.inf, dtype=np.float64)
    np.minimum.at(z_buffer, flat_index, target_z)
    nearest = target_z <= z_buffer[flat_index] + 1e-8
    flat_index = flat_index[nearest]
    target_z = target_z[nearest]
    colors = colors[nearest]
    order = np.argsort(target_z)[::-1]
    image = np.zeros((target_height * target_width, 3), dtype=np.uint8)
    image[flat_index[order]] = colors[order]
    mask = np.zeros(target_height * target_width, dtype=np.uint8)
    mask[flat_index] = 255
    depth = np.full(target_height * target_width, np.nan, dtype=np.float32)
    depth[flat_index] = target_z.astype(np.float32)
    return (
        image.reshape(target_height, target_width, 3),
        mask.reshape(target_height, target_width),
        depth.reshape(target_height, target_width),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--raw-alignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=1024)
    parser.add_argument("--target-role", choices=("exterior_1", "exterior_2"), default="exterior_1")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--git-commit")
    parser.add_argument("--git-branch")
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest_path = args.download_manifest.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    alignment_path = args.raw_alignment.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite DROID depth warp: {output}")
    for path in (manifest_path, calibration_path, alignment_path):
        if not path.is_file():
            raise ValueError(f"required DROID input is missing: {path}")
    download = json.loads(manifest_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    alignment = json.loads(alignment_path.read_text())
    if (
        download.get("status") != "WORKING"
        or calibration.get("intrinsics_distortion_verified") is not True
        or calibration.get("depth_lineage_verified") is not True
    ):
        raise ValueError("DROID raw download or SVO calibration is not verified")
    episode_index = download.get("episode_index")
    alignment_episodes = alignment.get("episodes")
    alignment_matches = (
        [
            episode
            for episode in alignment_episodes
            if isinstance(episode, dict)
            and episode.get("episode_index") == episode_index
        ]
        if isinstance(alignment_episodes, list)
        else []
    )
    if (
        alignment.get("accepted") is not True
        or alignment.get("raw_to_lerobot_chain_verified") is not True
        or len(alignment_matches) != 1
    ):
        raise ValueError("held-out raw-to-LeRobot alignment is not verified")
    aligned_episode = alignment_matches[0]
    if args.target_role not in aligned_episode.get("cameras", {}):
        raise ValueError("raw alignment lacks the requested target camera")
    root = manifest_path.parent / "data"
    roles = {
        "wrist": str(download["metadata"]["wrist_cam_serial"]),
        args.target_role: str(
            download["metadata"][
                "ext1_cam_serial" if args.target_role == "exterior_1" else "ext2_cam_serial"
            ]
        ),
    }
    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    lease_path, lease = acquire_gpu_lease(selected.physical_index)
    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    uuid_inventory, strict_processes, selected_processes = (
        _strict_process_snapshot(selected.physical_index)
    )
    if selected_processes:
        lease.close()
        raise RuntimeError(
            f"physical GPU {selected.physical_index} has compute processes: "
            f"{selected_processes}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    os.environ["PHIAGENT_PHYSICAL_GPU_INDEX"] = str(selected.physical_index)
    os.environ["LC_ALL"] = "C"
    os.environ["LANG"] = "C"
    locale.setlocale(locale.LC_ALL, "C")

    import cv2
    import h5py
    import numpy as np
    import pyzed.sl as sl

    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    _write_json(
        output / "config.json",
        {
            "schema_version": "1.0.0",
            "status": "STARTED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "download_manifest": str(manifest_path),
            "download_manifest_sha256": _sha256(manifest_path),
            "calibration": str(calibration_path),
            "calibration_sha256": _sha256(calibration_path),
            "raw_alignment": str(alignment_path),
            "raw_alignment_sha256": _sha256(alignment_path),
            "target_role": args.target_role,
            "selected_physical_gpu": asdict(selected),
            "gpu_lease": str(lease_path),
            "gpu_inventory_raw": inventory,
            "gpu_processes_raw": processes,
            "gpu_uuid_inventory_raw": uuid_inventory,
            "strict_gpu_processes_raw": strict_processes,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "locale": "C",
            "seed": args.seed,
            "packages": _package_versions(),
        },
    )
    _write_json(
        output / "git-state.json",
        _git_state(args.git_commit, args.git_branch),
    )
    cameras = {}
    for role, serial in roles.items():
        path = root / f"recordings/SVO/{serial}.svo"
        input_type = sl.InputType()
        input_type.set_from_svo_file(str(path))
        init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.coordinate_units = sl.UNIT.METER
        camera = sl.Camera()
        status = camera.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED failed to open {role} SVO: {status}")
        cameras[role] = camera
    source_camera = cameras["wrist"]
    target_camera = cameras[args.target_role]
    source_payload = calibration["cameras"]["wrist"]
    target_payload = calibration["cameras"][args.target_role]
    source_k = _intrinsics(np, source_payload["left_camera"])
    target_k = _intrinsics(np, target_payload["left_camera"])
    sample_rows = []
    with h5py.File(root / "trajectory.h5", "r") as handle:
        source_times = handle[
            f"observation/timestamp/cameras/{roles['wrist']}_estimated_capture"
        ][...]
        target_times = handle[
            f"observation/timestamp/cameras/{roles[args.target_role]}_estimated_capture"
        ][...]
        source_extrinsics = handle[
            f"observation/camera_extrinsics/{roles['wrist']}_left"
        ][...]
        target_extrinsics = handle[
            f"observation/camera_extrinsics/{roles[args.target_role]}_left"
        ][...]
        depth_archive = np.load(source_payload["depth_artifact"])
        for source_sample in source_payload["depth_samples"]:
            frame_index = int(source_sample["frame_index"])
            timestamp_ns = int(source_sample["timestamp_ns"])
            timestamp_ms = round(timestamp_ns / 1_000_000)
            source_h5 = nearest_timestamp_index(source_times.tolist(), timestamp_ms)
            target_h5 = nearest_timestamp_index(target_times.tolist(), timestamp_ms)
            target_frame = min(
                frame_index, int(target_camera.get_svo_number_of_frames()) - 1
            )
            source_camera.set_svo_position(frame_index)
            target_camera.set_svo_position(target_frame)
            if source_camera.grab() != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"failed to grab wrist frame {frame_index}")
            if target_camera.grab() != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"failed to grab target frame {target_frame}")
            source_mat, target_mat = sl.Mat(), sl.Mat()
            source_camera.retrieve_image(source_mat, sl.VIEW.LEFT)
            target_camera.retrieve_image(target_mat, sl.VIEW.LEFT)
            source_bgra = np.asarray(source_mat.get_data()).copy()
            target_bgra = np.asarray(target_mat.get_data()).copy()
            source_bgr = cv2.cvtColor(source_bgra, cv2.COLOR_BGRA2BGR)
            target_bgr = cv2.cvtColor(target_bgra, cv2.COLOR_BGRA2BGR)
            source_depth = depth_archive[
                f"depth_frame_{frame_index:06d}_m"
            ].astype(np.float32)
            base_t_source = _transform(np, source_extrinsics[source_h5])
            base_t_target = _transform(np, target_extrinsics[target_h5])
            target_t_source = np.linalg.inv(base_t_target) @ base_t_source
            warp, mask, warp_depth = _depth_splat(
                np,
                source_bgr,
                source_depth,
                source_k,
                target_k,
                target_t_source,
                target_bgr.shape[:2],
            )
            valid = mask > 0
            if not valid.any():
                raise ValueError(f"depth warp produced no valid pixels at {frame_index}")
            difference = (
                warp[valid].astype(np.float32) - target_bgr[valid].astype(np.float32)
            )
            mse = float(np.mean(difference**2))
            mae = float(np.mean(np.abs(difference)) / 255.0)
            psnr = 10.0 * math.log10(255.0**2 / max(mse, 1e-12))
            stem = f"frame-{frame_index:06d}"
            cv2.imwrite(str(output / f"{stem}-source.png"), source_bgr)
            cv2.imwrite(str(output / f"{stem}-target.png"), target_bgr)
            cv2.imwrite(str(output / f"{stem}-warp.png"), warp)
            cv2.imwrite(str(output / f"{stem}-mask.png"), mask)
            np.savez_compressed(output / f"{stem}-warp-depth.npz", depth_m=warp_depth)
            sample_rows.append(
                {
                    "source_frame_index": frame_index,
                    "target_frame_index": target_frame,
                    "svo_timestamp_ns": timestamp_ns,
                    "source_h5_index": source_h5,
                    "target_h5_index": target_h5,
                    "source_h5_time_ms": int(source_times[source_h5]),
                    "target_h5_time_ms": int(target_times[target_h5]),
                    "cross_camera_offset_ms": int(
                        target_times[target_h5] - source_times[source_h5]
                    ),
                    "coverage": float(np.mean(valid)),
                    "visible_mae_0_1": mae,
                    "visible_psnr_db": psnr,
                    "target_t_source": target_t_source.tolist(),
                    "artifacts": {
                        name: {
                            "path": str(output / f"{stem}-{name}.{extension}"),
                            "sha256": _sha256(
                                output / f"{stem}-{name}.{extension}"
                            ),
                        }
                        for name, extension in (
                            ("source", "png"),
                            ("target", "png"),
                            ("warp", "png"),
                            ("mask", "png"),
                            ("warp-depth", "npz"),
                        )
                    },
                }
            )
    for camera in cameras.values():
        camera.close()
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "PARTIAL",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "selected_physical_gpu": asdict(selected),
        "episode_id": download.get("episode_id", download["metadata"].get("uuid")),
        "episode_index": episode_index,
        "source_role": "wrist",
        "target_role": args.target_role,
        "target_lerobot_stream": aligned_episode["cameras"][args.target_role][
            "lerobot_stream"
        ],
        "exterior_assignment": download.get("exterior_assignment"),
        "raw_alignment": str(alignment_path),
        "raw_alignment_sha256": _sha256(alignment_path),
        "coordinate_chain": (
            "target_camera_T_wrist_camera(t) = "
            "inverse(robot_base_T_target_camera) @ robot_base_T_wrist_camera(t)"
        ),
        "samples": sample_rows,
        "mean_coverage": float(
            sum(row["coverage"] for row in sample_rows) / len(sample_rows)
        ),
        "mean_visible_mae_0_1": float(
            sum(row["visible_mae_0_1"] for row in sample_rows) / len(sample_rows)
        ),
        "mean_visible_psnr_db": float(
            sum(row["visible_psnr_db"] for row in sample_rows) / len(sample_rows)
        ),
        "claim_boundary": (
            "Calibrated depth splatting is a visible-surface geometric lower bound for "
            "a lineage-mapped held-out real DROID episode. It does not hallucinate "
            "occlusions, establish Strict-W SOTA, or resolve dataset rights."
        ),
        "missing_requirements": [
            "verified raw DROID dataset license",
            "public raw tree for held-out episode 60",
            "learned disocclusion and held-out novel-view quality",
        ],
    }
    _write_json(output / "evaluation.json", result)
    (output / "render.log").write_text(
        f"samples={len(sample_rows)} coverage={result['mean_coverage']:.6f} "
        f"psnr={result['mean_visible_psnr_db']:.6f}\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    lease.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
