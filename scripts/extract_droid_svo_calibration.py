#!/usr/bin/env python3
"""Extract pinned DROID SVO intrinsics, distortion, timestamps, and depth samples."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import socket
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402


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


def sample_indices(frame_count: int) -> tuple[int, ...]:
    if frame_count <= 0:
        raise ValueError("SVO frame count must be positive")
    # ZED reports an EOF sentinel position in get_svo_number_of_frames().
    return tuple(sorted({0, frame_count // 2, max(0, frame_count - 2)}))


def resolve_episode_identity(
    download: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, object]:
    episode_id = download.get("episode_id", metadata.get("uuid"))
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("DROID manifest has no episode_id or metadata UUID")
    episode_index = download.get("episode_index")
    if episode_index is not None and (
        not isinstance(episode_index, int) or episode_index < 0
    ):
        raise ValueError("DROID episode_index must be a non-negative integer")
    identity: dict[str, object] = {
        "episode_id": episode_id,
        "lineage_mapped_to_lerobot": episode_index is not None,
    }
    if episode_index is not None:
        identity.update(
            {
                "episode_index": episode_index,
                "raw_gcs_prefix": download.get("raw_gcs_prefix"),
                "exterior_assignment": download.get("exterior_assignment"),
                "sequence_payload_sha256": download.get(
                    "sequence_payload_sha256"
                ),
            }
        )
    return identity


def _packages() -> dict[str, str | None]:
    result = {}
    for name in ("numpy", "pyzed", "Cython"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _camera_payload(camera: Any) -> dict[str, object]:
    return {
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "distortion": [float(value) for value in camera.disto],
        "horizontal_fov_deg": float(camera.h_fov),
        "vertical_fov_deg": float(camera.v_fov),
        "diagonal_fov_deg": float(camera.d_fov),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=1024)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest_path = args.download_manifest.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite DROID SVO audit: {output}")
    if not manifest_path.is_file():
        raise ValueError(f"raw DROID manifest is missing: {manifest_path}")
    download = json.loads(manifest_path.read_text())
    if not isinstance(download, dict) or download.get("status") != "WORKING":
        raise ValueError("raw DROID download is not complete")
    metadata = download["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("raw DROID metadata must contain an object")
    identity = resolve_episode_identity(download, metadata)
    roles = {
        "wrist": str(metadata["wrist_cam_serial"]),
        "exterior_1": str(metadata["ext1_cam_serial"]),
        "exterior_2": str(metadata["ext2_cam_serial"]),
    }
    root = manifest_path.parent / "data"
    svo_paths = {
        role: root / f"recordings/SVO/{serial}.svo"
        for role, serial in roles.items()
    }
    for role, path in svo_paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{role} SVO is missing or empty: {path}")
    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    os.environ["PHIAGENT_PHYSICAL_GPU_INDEX"] = str(selected.physical_index)

    import numpy as np
    import pyzed.sl as sl

    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    config = {
        "schema_version": "1.0.0",
        "status": "STARTED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "download_manifest": str(manifest_path),
        "download_manifest_sha256": _sha256(manifest_path),
        "selected_physical_gpu": asdict(selected),
        "gpu_inventory_raw": inventory,
        "gpu_processes_raw": processes,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "coordinate_units": "meter",
        "depth_mode": "ULTRA",
        "packages": _packages(),
    }
    _write_json(output / "config.json", config)
    cameras = {}
    for role, path in svo_paths.items():
        input_type = sl.InputType()
        input_type.set_from_svo_file(str(path))
        init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_units = sl.UNIT.METER
        camera = sl.Camera()
        status = camera.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED failed to open {role} SVO: {status}")
        info = camera.get_camera_information()
        configuration = info.camera_configuration
        calibration = configuration.calibration_parameters
        frame_count = int(camera.get_svo_number_of_frames())
        indices = sample_indices(frame_count)
        depth = sl.Mat()
        samples = []
        arrays = {}
        for index in indices:
            camera.set_svo_position(index)
            grab_status = camera.grab()
            if grab_status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(
                    f"ZED failed to grab {role} frame {index}: {grab_status}"
                )
            camera.retrieve_measure(depth, sl.MEASURE.DEPTH)
            values = np.asarray(depth.get_data(), dtype=np.float32).copy()
            finite = np.isfinite(values) & (values > 0)
            arrays[f"depth_frame_{index:06d}_m"] = values
            samples.append(
                {
                    "frame_index": index,
                    "timestamp_ns": int(
                        camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                    ),
                    "finite_fraction": float(np.mean(finite)),
                    "minimum_m": float(np.min(values[finite])) if finite.any() else None,
                    "median_m": float(np.median(values[finite])) if finite.any() else None,
                    "maximum_m": float(np.max(values[finite])) if finite.any() else None,
                }
            )
        depth_path = output / f"{role}-depth-samples.npz"
        np.savez_compressed(depth_path, **arrays)
        cameras[role] = {
            "serial": roles[role],
            "svo": str(path),
            "svo_sha256": _sha256(path),
            "reported_serial": str(info.serial_number),
            "camera_model": str(info.camera_model),
            "firmware_version": int(configuration.firmware_version),
            "fps": float(configuration.fps),
            "resolution": {
                "width": int(configuration.resolution.width),
                "height": int(configuration.resolution.height),
            },
            "left_camera": _camera_payload(calibration.left_cam),
            "right_camera": _camera_payload(calibration.right_cam),
            "stereo_translation_m": [
                float(value) for value in calibration.stereo_transform.get_translation().get()
            ],
            "frame_count": frame_count,
            "sample_indices": list(indices),
            "depth_samples": samples,
            "depth_artifact": str(depth_path),
            "depth_artifact_sha256": _sha256(depth_path),
        }
        camera.close()
    if any(
        role_payload["reported_serial"] != role_payload["serial"]
        for role_payload in cameras.values()
    ):
        raise ValueError("SVO-reported camera serial does not match DROID metadata")
    lineage_mapped = bool(identity["lineage_mapped_to_lerobot"])
    missing_requirements = ["verified raw DROID dataset license"]
    if lineage_mapped:
        missing_requirements.append(
            "verified raw-HDF5-to-LeRobot video/action timestamp offset"
        )
        claim_boundary = (
            "SVO-derived calibration and depth are verified for a lineage-mapped "
            "held-out LeRobot episode. Rights and raw-HDF5 timestamp alignment "
            "remain separate gates; this result does not establish generation quality."
        )
    else:
        missing_requirements.append("mapping to held-out LeRobot episodes 21/60/77")
        claim_boundary = (
            "SVO-derived calibration and depth are verified for one unmapped internal "
            "schema-smoke episode. The result is not claim-eligible until rights and "
            "held-out lineage are resolved."
        )
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "BLOCKED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "selected_physical_gpu": asdict(selected),
        **identity,
        "cameras": cameras,
        "intrinsics_distortion_verified": True,
        "depth_lineage_verified": True,
        "depth_units": "meter",
        "strict_w_ready": False,
        "missing_requirements": missing_requirements,
        "claim_boundary": claim_boundary,
    }
    for payload in cameras.values():
        for sample in payload["depth_samples"]:
            fraction = float(sample["finite_fraction"])
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError("invalid SVO depth finite fraction")
    _write_json(output / "calibration.json", result)
    (output / "extraction.log").write_text(
        f"extracted {len(cameras)} cameras on physical GPU {selected.physical_index}\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
