#!/usr/bin/env python3
"""Audit raw DROID action, calibration, and timestamp evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shlex
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.view_generation.readiness import extrinsic_variation  # noqa: E402


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _dataset_paths(group: Any, prefix: str = "") -> list[str]:
    import h5py

    paths = []
    for name, value in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(value, h5py.Dataset):
            paths.append(path)
        else:
            paths.extend(_dataset_paths(value, path))
    return paths


def main() -> int:
    args = _parser().parse_args()
    manifest_path = args.download_manifest.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite raw DROID audit: {output}")
    if not manifest_path.is_file():
        raise ValueError(f"raw DROID download manifest is missing: {manifest_path}")
    download = json.loads(manifest_path.read_text())
    if not isinstance(download, dict) or download.get("status") != "WORKING":
        raise ValueError("raw DROID download manifest is not complete")
    root = manifest_path.parent / "data"
    trajectory = root / "trajectory.h5"
    if not trajectory.is_file():
        raise ValueError(f"raw DROID trajectory is missing: {trajectory}")

    import h5py
    import numpy as np

    metadata = download["metadata"]
    serials = {
        "wrist": str(metadata["wrist_cam_serial"]),
        "exterior_1": str(metadata["ext1_cam_serial"]),
        "exterior_2": str(metadata["ext2_cam_serial"]),
    }
    with h5py.File(trajectory, "r") as handle:
        dataset_paths = sorted(_dataset_paths(handle))
        lengths = {
            path: int(handle[path].shape[0])
            for path in dataset_paths
            if handle[path].shape
        }
        expected_length = int(metadata["trajectory_length"])
        mismatched = {
            path: length for path, length in lengths.items() if length != expected_length
        }
        finite_failures = []
        for path in dataset_paths:
            dataset = handle[path]
            if dataset.dtype.kind not in "fc" or dataset.size == 0:
                continue
            if not bool(np.isfinite(dataset[...]).all()):
                finite_failures.append(path)
        extrinsics = {}
        timestamps = {}
        for role, serial in serials.items():
            path = f"observation/camera_extrinsics/{serial}_left"
            if path not in handle:
                raise ValueError(f"raw DROID lacks left-camera extrinsics for {role}/{serial}")
            values = np.asarray(handle[path][...], dtype=np.float64)
            extrinsics[role] = {
                "dataset": path,
                "shape": list(values.shape),
                "semantics": "robot_base_T_camera encoded as xyz + Euler-XYZ radians",
                **extrinsic_variation(values.tolist()),
            }
            time_path = (
                f"observation/timestamp/cameras/{serial}_estimated_capture"
            )
            if time_path not in handle:
                raise ValueError(f"raw DROID lacks estimated capture time for {role}")
            time_values = np.asarray(handle[time_path][...], dtype=np.int64)
            timestamps[role] = {
                "dataset": time_path,
                "strictly_increasing": bool(np.all(np.diff(time_values) > 0)),
                "first": int(time_values[0]),
                "last": int(time_values[-1]),
            }
        wrist_offset_path = (
            f"observation/camera_extrinsics/{serials['wrist']}_left_gripper_offset"
        )
        if wrist_offset_path not in handle:
            raise ValueError("raw DROID lacks wrist camera-to-gripper offset")
        wrist_offset = extrinsic_variation(handle[wrist_offset_path][...].tolist())
        reference_time = np.asarray(
            handle[timestamps["wrist"]["dataset"]][...], dtype=np.int64
        )
        camera_offsets = {}
        for role in ("exterior_1", "exterior_2"):
            values = np.asarray(handle[timestamps[role]["dataset"]][...], dtype=np.int64)
            difference = values - reference_time
            camera_offsets[role] = {
                "median_raw_units": float(np.median(difference)),
                "maximum_absolute_raw_units": int(np.max(np.abs(difference))),
                "units": "dataset timestamp units; conversion not assumed",
            }
        action_contract = {
            "commanded_cartesian": list(handle["action/cartesian_position"].shape),
            "commanded_joint": list(handle["action/joint_position"].shape),
            "measured_cartesian": list(
                handle["observation/robot_state/cartesian_position"].shape
            ),
            "measured_joint": list(
                handle["observation/robot_state/joint_positions"].shape
            ),
            "gripper": list(handle["observation/robot_state/gripper_position"].shape),
            "coordinate_frame": "robot_base:panda-295341-1325686",
            "cartesian_representation": "xyz + Euler-XYZ radians",
        }
        has_intrinsics = "observation/camera_intrinsics" in handle
    if mismatched or finite_failures:
        raise ValueError(
            f"raw DROID HDF5 failed alignment/finite checks: "
            f"mismatched={mismatched}, nonfinite={finite_failures}"
        )
    missing = [
        "verified raw DROID dataset license",
        "mapping to held-out LeRobot episodes 21/60/77",
        "SVO-derived intrinsics and distortion",
        "SVO-derived depth with verified units",
    ]
    if not has_intrinsics:
        missing.append("trajectory.h5 camera intrinsics")
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "BLOCKED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "download_manifest": str(manifest_path),
        "download_manifest_sha256": _sha256(manifest_path),
        "trajectory": str(trajectory),
        "trajectory_sha256": _sha256(trajectory),
        "episode_id": download["episode_id"],
        "trajectory_length": int(metadata["trajectory_length"]),
        "dataset_count": len(dataset_paths),
        "all_dataset_lengths_match": True,
        "all_numeric_datasets_finite": True,
        "camera_serials": serials,
        "extrinsics": extrinsics,
        "wrist_gripper_offset_variation": wrist_offset,
        "timestamps": timestamps,
        "camera_time_offsets": camera_offsets,
        "action_contract": action_contract,
        "has_hdf5_intrinsics": has_intrinsics,
        "missing_requirements": sorted(set(missing)),
        "strict_w_ready": False,
        "labeling_ready_scope": (
            "Frame-aligned measured/commanded robot-base action labels for this "
            "unmapped internal smoke episode; no public training or benchmark claim."
        ),
        "claim_boundary": (
            "The raw schema validates frame-explicit robot-base telemetry and camera "
            "extrinsics. Rights, held-out mapping, intrinsics, distortion, and depth "
            "remain blocked."
        ),
    }
    if any(not math.isfinite(float(value)) for value in wrist_offset.values()):
        raise ValueError("wrist offset audit produced non-finite evidence")
    _write_json(output / "audit.json", result)
    (output / "audit.log").write_text(
        f"datasets={len(dataset_paths)} length={metadata['trajectory_length']} "
        f"missing={len(result['missing_requirements'])}\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

