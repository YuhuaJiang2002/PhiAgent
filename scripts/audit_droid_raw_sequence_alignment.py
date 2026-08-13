#!/usr/bin/env python3
"""Verify raw DROID SVO/HDF5 alignment to lineage-mapped SequenceExamples."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import locale
import os
import platform
import shlex
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_droid_sequence_lineage import (  # noqa: E402
    _bytes_feature,
    _dhash64,
    _pixel_psnr_db,
    percentile_nearest_rank,
)
from phiagent.rendering.wan_animate import (  # noqa: E402
    acquire_gpu_lease,
    query_gpus,
    select_gpu,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_ROLES = {
    "wrist": ("wrist_cam_serial", "wrist_image_left"),
    "exterior_1": ("ext1_cam_serial", "exterior_image_1_left"),
    "exterior_2": ("ext2_cam_serial", "exterior_image_2_left"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        nargs=2,
        action="append",
        required=True,
        metavar=("RAW_MANIFEST", "SEQUENCE_PAYLOAD"),
    )
    parser.add_argument("--sequence-lineage-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=1024)
    parser.add_argument("--minimum-p05-psnr-db", type=float, default=25.0)
    parser.add_argument("--maximum-p95-dhash-hamming", type=int, default=8)
    parser.add_argument(
        "--minimum-aligned-adjacent-psnr-gap-db",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--maximum-centered-timestamp-p95-ms",
        type=float,
        default=5.0,
    )
    parser.add_argument("--maximum-terminal-row-gap-ms", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--git-commit")
    parser.add_argument("--git-branch")
    return parser


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


def _git_state(
    commit_override: str | None,
    branch_override: str | None,
) -> dict[str, object]:
    if (commit_override is None) != (branch_override is None):
        raise ValueError("git-commit and git-branch must be provided together")
    if commit_override is not None:
        if len(commit_override) != 40 or any(
            character not in "0123456789abcdef" for character in commit_override
        ):
            raise ValueError("git-commit must be a lowercase 40-character SHA-1")
        return {
            "commit": commit_override,
            "branch": branch_override,
            "dirty": None,
            "status_porcelain": None,
            "resolution": "explicit source-worktree snapshot",
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
        }

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = run("status", "--porcelain=v1")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "resolution": "local Git worktree",
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
    }


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in (
        "numpy",
        "opencv-python",
        "h5py",
        "tensorflow",
        "protobuf",
        "pyzed",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def timestamp_alignment_metrics(
    svo_timestamps_ns: list[int],
    hdf_capture_timestamps_ms: list[int],
) -> dict[str, float]:
    if not svo_timestamps_ns:
        raise ValueError("SVO timestamp sequence is empty")
    if len(hdf_capture_timestamps_ms) != len(svo_timestamps_ns) + 1:
        raise ValueError("HDF5 timestamps must contain exactly one terminal row")
    if any(
        right <= left
        for left, right in zip(svo_timestamps_ns, svo_timestamps_ns[1:])
    ):
        raise ValueError("SVO timestamps must be strictly increasing")
    if any(
        right <= left
        for left, right in zip(
            hdf_capture_timestamps_ms,
            hdf_capture_timestamps_ms[1:],
        )
    ):
        raise ValueError("HDF5 capture timestamps must be strictly increasing")
    offsets_ms = [
        (svo - hdf * 1_000_000) / 1_000_000
        for svo, hdf in zip(
            svo_timestamps_ns,
            hdf_capture_timestamps_ms[:-1],
        )
    ]
    offset_median = median(offsets_ms)
    centered = [abs(value - offset_median) for value in offsets_ms]
    terminal_gap_ms = (
        hdf_capture_timestamps_ms[-1] * 1_000_000 - svo_timestamps_ns[-1]
    ) / 1_000_000
    return {
        "svo_minus_hdf_offset_median_ms": offset_median,
        "centered_absolute_residual_p95_ms": percentile_nearest_rank(
            centered, 0.95
        ),
        "centered_absolute_residual_max_ms": max(centered),
        "svo_period_median_ms": median(
            [
                (right - left) / 1_000_000
                for left, right in zip(
                    svo_timestamps_ns,
                    svo_timestamps_ns[1:],
                )
            ]
        ),
        "hdf_period_median_ms": median(
            [
                float(right - left)
                for left, right in zip(
                    hdf_capture_timestamps_ms[:-1],
                    hdf_capture_timestamps_ms[1:-1],
                )
            ]
        ),
        "terminal_hdf_row_after_last_svo_ms": terminal_gap_ms,
    }


def selected_gpu_processes(
    inventory_lines: list[str],
    process_lines: list[str],
    selected_index: int,
) -> list[str]:
    index_to_uuid = {}
    for line in inventory_lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise ValueError(f"unexpected GPU UUID inventory line: {line}")
        index_to_uuid[int(fields[0])] = fields[1]
    if selected_index not in index_to_uuid:
        raise ValueError(f"selected GPU {selected_index} has no UUID inventory row")
    selected_uuid = index_to_uuid[selected_index]
    return [
        line
        for line in process_lines
        if line.strip() and line.split(",", maxsplit=1)[0].strip() == selected_uuid
    ]


def _strict_process_snapshot(selected_index: int) -> tuple[str, str, list[str]]:
    uuid_inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    process_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if process_result.returncode != 0:
        raise RuntimeError(
            "could not validate GPU compute processes: "
            f"{process_result.stderr.strip()}"
        )
    processes = process_result.stdout
    selected_processes = selected_gpu_processes(
        uuid_inventory.strip().splitlines(),
        processes.strip().splitlines(),
        selected_index,
    )
    return uuid_inventory, processes, selected_processes


def _decode_sequence_frames(
    cv2: Any,
    np: Any,
    encoded_frames: list[bytes],
) -> list[Any]:
    if not encoded_frames:
        raise ValueError("SequenceExample image feature is empty")
    result = []
    for index, encoded in enumerate(encoded_frames):
        frame = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is None:
            raise ValueError(f"could not decode SequenceExample JPEG {index}")
        result.append(frame)
    shapes = {tuple(frame.shape) for frame in result}
    if len(shapes) != 1:
        raise ValueError(f"SequenceExample frames have inconsistent shapes: {shapes}")
    return result


def _pixel_alignment_metrics(
    cv2: Any,
    np: Any,
    raw_frames: list[Any],
    sequence_frames: list[Any],
) -> dict[str, object]:
    if len(raw_frames) != len(sequence_frames):
        raise ValueError("raw SVO and SequenceExample frame counts differ")
    target_size = (
        int(sequence_frames[0].shape[1]),
        int(sequence_frames[0].shape[0]),
    )
    resized_raw = [
        cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        for frame in raw_frames
    ]
    aligned_psnr = [
        _pixel_psnr_db(np, raw, sequence)
        for raw, sequence in zip(resized_raw, sequence_frames)
    ]
    aligned_dhash = [
        float(
            (
                _dhash64(cv2, np, raw)
                ^ _dhash64(cv2, np, sequence)
            ).bit_count()
        )
        for raw, sequence in zip(resized_raw, sequence_frames)
    ]
    adjacent_forward = [
        _pixel_psnr_db(np, raw, sequence)
        for raw, sequence in zip(resized_raw[:-1], sequence_frames[1:])
    ]
    adjacent_backward = [
        _pixel_psnr_db(np, raw, sequence)
        for raw, sequence in zip(resized_raw[1:], sequence_frames[:-1])
    ]
    best_adjacent_median = max(
        median(adjacent_forward),
        median(adjacent_backward),
    )
    sample_indices = {0, len(resized_raw) // 2, len(resized_raw) - 1}
    sampled_hashes = [
        {
            "frame_index": index,
            "raw_resized_pixel_sha256": hashlib.sha256(
                resized_raw[index].tobytes()
            ).hexdigest(),
            "sequence_decoded_pixel_sha256": hashlib.sha256(
                sequence_frames[index].tobytes()
            ).hexdigest(),
        }
        for index in sorted(sample_indices)
    ]
    aligned_median = median(aligned_psnr)
    return {
        "frames_compared": len(raw_frames),
        "psnr_db_p05": percentile_nearest_rank(aligned_psnr, 0.05),
        "psnr_db_median": aligned_median,
        "psnr_db_min": min(aligned_psnr),
        "dhash_hamming_p95": percentile_nearest_rank(aligned_dhash, 0.95),
        "dhash_hamming_max": max(aligned_dhash),
        "best_adjacent_shift_psnr_db_median": best_adjacent_median,
        "aligned_minus_best_adjacent_psnr_gap_db": (
            aligned_median - best_adjacent_median
        ),
        "sampled_decoded_hashes": sampled_hashes,
    }


def _raw_file_record(
    raw_manifest: dict[str, Any],
    relative_path: str,
) -> dict[str, Any]:
    files = raw_manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("raw DROID manifest has no files list")
    matches = [
        record
        for record in files
        if isinstance(record, dict) and record.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(f"raw manifest has no unique record for {relative_path}")
    return matches[0]


def _lineage_episode(
    lineage_manifest: dict[str, Any],
    episode_index: int,
) -> dict[str, Any]:
    episodes = lineage_manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("lineage manifest has no episodes list")
    matches = [
        row
        for row in episodes
        if isinstance(row, dict) and row.get("episode_index") == episode_index
    ]
    if len(matches) != 1:
        raise ValueError(f"lineage manifest has no unique episode {episode_index}")
    return matches[0]


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    lineage_path = args.sequence_lineage_manifest.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite DROID alignment audit: {output}")
    if not lineage_path.is_file():
        raise ValueError(f"sequence lineage manifest is missing: {lineage_path}")
    if (
        args.minimum_free_gpu_mib <= 0
        or args.minimum_p05_psnr_db <= 0
        or args.maximum_p95_dhash_hamming < 0
        or args.minimum_aligned_adjacent_psnr_gap_db <= 0
        or args.maximum_centered_timestamp_p95_ms <= 0
        or args.maximum_terminal_row_gap_ms <= 0
    ):
        raise ValueError("alignment thresholds are invalid")
    cases = [
        (
            Path(raw).expanduser().resolve(),
            Path(sequence).expanduser().resolve(),
        )
        for raw, sequence in args.case
    ]
    if len(set(cases)) != len(cases):
        raise ValueError("DROID alignment cases must be unique")
    for raw_manifest, sequence_payload in cases:
        if not raw_manifest.is_file() or not sequence_payload.is_file():
            raise ValueError("raw manifest and SequenceExample payload must exist")

    lineage = json.loads(lineage_path.read_text())
    if not isinstance(lineage, dict) or lineage.get("accepted") is not True:
        raise ValueError("SequenceExample-to-LeRobot lineage must be accepted")
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
    from tensorflow.train import SequenceExample

    output.mkdir(parents=True)
    (output / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n"
    )
    _write_json(
        output / "config.json",
        {
            "schema_version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cases": [
                {
                    "raw_manifest": str(raw),
                    "raw_manifest_sha256": _sha256(raw),
                    "sequence_payload": str(sequence),
                    "sequence_payload_sha256": _sha256(sequence),
                }
                for raw, sequence in cases
            ],
            "sequence_lineage_manifest": str(lineage_path),
            "sequence_lineage_manifest_sha256": _sha256(lineage_path),
            "selected_physical_gpu": asdict(selected),
            "gpu_lease": str(lease_path),
            "gpu_inventory_raw": inventory,
            "gpu_processes_raw": processes,
            "gpu_uuid_inventory_raw": uuid_inventory,
            "strict_gpu_processes_raw": strict_processes,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "locale": "C",
            "thresholds": {
                "minimum_p05_psnr_db": args.minimum_p05_psnr_db,
                "maximum_p95_dhash_hamming": args.maximum_p95_dhash_hamming,
                "minimum_aligned_adjacent_psnr_gap_db": (
                    args.minimum_aligned_adjacent_psnr_gap_db
                ),
                "maximum_centered_timestamp_p95_ms": (
                    args.maximum_centered_timestamp_p95_ms
                ),
                "maximum_terminal_row_gap_ms": args.maximum_terminal_row_gap_ms,
            },
            "seed": args.seed,
            "seed_use": "recorded for reproducibility; audit is deterministic",
            "packages": _package_versions(),
        },
    )
    _write_json(
        output / "git-state.json",
        _git_state(args.git_commit, args.git_branch),
    )

    episode_results = []
    all_gates = []
    for raw_manifest_path, sequence_payload in cases:
        raw_manifest = json.loads(raw_manifest_path.read_text())
        if not isinstance(raw_manifest, dict) or raw_manifest.get("status") != "WORKING":
            raise ValueError("raw DROID case must have WORKING download status")
        episode_index = raw_manifest.get("episode_index")
        metadata = raw_manifest.get("metadata")
        if not isinstance(episode_index, int) or not isinstance(metadata, dict):
            raise ValueError("raw DROID manifest lacks episode index or metadata")
        lineage_episode = _lineage_episode(lineage, episode_index)
        payload_sha256 = _sha256(sequence_payload)
        if payload_sha256 != raw_manifest.get("sequence_payload_sha256"):
            raise ValueError("SequenceExample payload does not match raw lineage manifest")
        if payload_sha256 != lineage_episode.get("payload_sha256"):
            raise ValueError("SequenceExample payload does not match accepted lineage")
        if (
            raw_manifest.get("exterior_assignment")
            != lineage_episode.get("camera_assignment", {}).get("selected")
        ):
            raise ValueError("raw and LeRobot exterior camera assignments disagree")

        example = SequenceExample.FromString(sequence_payload.read_bytes())
        raw_root = raw_manifest_path.parent / "data"
        hdf_path = raw_root / "trajectory.h5"
        hdf_record = _raw_file_record(raw_manifest, "trajectory.h5")
        if (
            not hdf_path.is_file()
            or _sha256(hdf_path) != hdf_record.get("sha256")
        ):
            raise ValueError("raw HDF5 artifact hash mismatch")
        camera_results = {}
        with h5py.File(hdf_path, "r") as hdf:
            hdf_rows = int(hdf["action/cartesian_position"].shape[0])
            for role, (serial_key, sequence_name) in CAMERA_ROLES.items():
                serial = str(metadata[serial_key])
                relative_svo = f"recordings/SVO/{serial}.svo"
                svo_path = raw_root / relative_svo
                svo_record = _raw_file_record(raw_manifest, relative_svo)
                if (
                    not svo_path.is_file()
                    or _sha256(svo_path) != svo_record.get("sha256")
                ):
                    raise ValueError(f"raw SVO artifact hash mismatch: {role}")
                encoded_frames = _bytes_feature(
                    example,
                    f"steps/observation/{sequence_name}",
                )
                sequence_frames = _decode_sequence_frames(
                    cv2,
                    np,
                    encoded_frames,
                )

                input_type = sl.InputType()
                input_type.set_from_svo_file(str(svo_path))
                init = sl.InitParameters(
                    input_t=input_type,
                    svo_real_time_mode=False,
                )
                init.depth_mode = sl.DEPTH_MODE.NONE
                camera = sl.Camera()
                status = camera.open(init)
                if status != sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError(f"ZED failed to open {role} SVO: {status}")
                reported_frames = int(camera.get_svo_number_of_frames())
                raw_frames = []
                svo_timestamps_ns = []
                image = sl.Mat()
                for frame_index in range(reported_frames):
                    grab_status = camera.grab()
                    if grab_status != sl.ERROR_CODE.SUCCESS:
                        camera.close()
                        raise RuntimeError(
                            f"ZED failed to grab {role} frame {frame_index}: "
                            f"{grab_status}"
                        )
                    camera.retrieve_image(image, sl.VIEW.LEFT)
                    values = np.asarray(image.get_data())
                    if values.ndim != 3 or values.shape[2] < 3:
                        camera.close()
                        raise ValueError(f"unexpected ZED image shape: {values.shape}")
                    raw_frames.append(values[:, :, :3].copy())
                    svo_timestamps_ns.append(
                        int(
                            camera.get_timestamp(
                                sl.TIME_REFERENCE.IMAGE
                            ).get_nanoseconds()
                        )
                    )
                camera.close()
                timestamp_path = (
                    f"observation/timestamp/cameras/{serial}_estimated_capture"
                )
                if timestamp_path not in hdf:
                    raise ValueError(f"HDF5 lacks camera timestamp: {timestamp_path}")
                hdf_timestamps_ms = [
                    int(value) for value in np.asarray(hdf[timestamp_path])
                ]
                pixels = _pixel_alignment_metrics(
                    cv2,
                    np,
                    raw_frames,
                    sequence_frames,
                )
                timestamps = timestamp_alignment_metrics(
                    svo_timestamps_ns,
                    hdf_timestamps_ms,
                )
                gates = {
                    "svo_sequence_frame_count_match": (
                        reported_frames == len(sequence_frames)
                    ),
                    "hdf_has_one_terminal_row": (
                        hdf_rows == len(sequence_frames) + 1
                        and len(hdf_timestamps_ms) == hdf_rows
                    ),
                    "pixel_psnr": (
                        pixels["psnr_db_p05"] >= args.minimum_p05_psnr_db
                    ),
                    "pixel_dhash": (
                        pixels["dhash_hamming_p95"]
                        <= args.maximum_p95_dhash_hamming
                    ),
                    "same_index_beats_adjacent": (
                        pixels["aligned_minus_best_adjacent_psnr_gap_db"]
                        >= args.minimum_aligned_adjacent_psnr_gap_db
                    ),
                    "timestamp_constant_offset": (
                        timestamps["centered_absolute_residual_p95_ms"]
                        <= args.maximum_centered_timestamp_p95_ms
                    ),
                    "terminal_row_after_video": (
                        0
                        < timestamps["terminal_hdf_row_after_last_svo_ms"]
                        <= args.maximum_terminal_row_gap_ms
                    ),
                }
                all_gates.extend(gates.values())
                camera_results[role] = {
                    "serial": serial,
                    "svo": str(svo_path),
                    "svo_sha256": svo_record["sha256"],
                    "sequence_feature": sequence_name,
                    "lerobot_stream": lineage_episode["camera_assignment"][
                        "sequence_to_lerobot_stream"
                    ][sequence_name],
                    "reported_svo_frames": reported_frames,
                    "hdf_rows": hdf_rows,
                    "pixel_alignment": pixels,
                    "timestamp_alignment": timestamps,
                    "gates": gates,
                }
        episode_results.append(
            {
                "episode_index": episode_index,
                "episode_id": metadata.get("uuid"),
                "raw_manifest": str(raw_manifest_path),
                "raw_manifest_sha256": _sha256(raw_manifest_path),
                "sequence_payload": str(sequence_payload),
                "sequence_payload_sha256": payload_sha256,
                "raw_hdf5": str(hdf_path),
                "raw_hdf5_sha256": hdf_record["sha256"],
                "exterior_assignment": raw_manifest["exterior_assignment"],
                "cameras": camera_results,
            }
        )

    accepted = all(all_gates)
    completed_at = datetime.now(timezone.utc).isoformat()
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING" if accepted else "PARTIAL",
        "honest_status": "BLOCKED",
        "accepted": accepted,
        "completed_at": completed_at,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "selected_physical_gpu": asdict(selected),
        "sequence_lineage_manifest": str(lineage_path),
        "sequence_lineage_manifest_sha256": _sha256(lineage_path),
        "episodes": episode_results,
        "episode_count": len(episode_results),
        "camera_count": sum(
            len(episode["cameras"]) for episode in episode_results
        ),
        "raw_hdf5_to_sequence_timing_verified": accepted,
        "raw_svo_to_sequence_pixels_verified": accepted,
        "raw_to_lerobot_chain_verified": accepted,
        "missing_requirements": [
            "verified raw DROID dataset license",
            "public raw tree for held-out episode 60",
        ],
        "claim_boundary": (
            "This audit establishes timestamp and pixel lineage from raw SVO/HDF5 "
            "through accepted SequenceExample-to-LeRobot mappings for episodes "
            "21/77. It does not establish novel-view generation quality, physical "
            "counterfactual correctness, or raw-data training rights."
        ),
    }
    _write_json(output / "alignment.json", result)
    (output / "audit.log").write_text(
        f"{completed_at} accepted={accepted} episodes={len(episode_results)} "
        f"cameras={result['camera_count']}\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "accepted": accepted,
                "episode_count": len(episode_results),
                "camera_count": result["camera_count"],
            },
            sort_keys=True,
        )
    )
    lease.close()
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
