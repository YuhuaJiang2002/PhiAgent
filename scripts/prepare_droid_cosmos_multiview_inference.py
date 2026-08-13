#!/usr/bin/env python3
"""Prepare disclosed DROID conditions for Cosmos robot multiview inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CAMERA_FILE_ROLES = ("head", "hand_0", "hand_1")
SOURCE_TO_ROLE = {
    "exterior_1": "head",
    "wrist": "hand_0",
    "exterior_2": "hand_1",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--cosmos-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument(
        "--split",
        choices=("legacy_dev", "validation", "final_holdout"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-output-frames", type=int, default=17)
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


def select_record(contract: dict[str, Any], episode: int, split: str) -> dict[str, Any]:
    matches = [
        row
        for row in contract.get("records", [])
        if int(row.get("episode_index", -1)) == episode and row.get("split") == split
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {split} record for episode {episode}, got {len(matches)}")
    record = matches[0]
    if record.get("training_use") is not False:
        raise ValueError("inference record must not be a training sample")
    return record


def repeat_first_calibration_row(source: Path, destination: Path, rows: int) -> None:
    """Freeze a disclosed proxy calibration for all requested latent frames."""
    if rows <= 0:
        raise ValueError("calibration row count must be positive")
    lines = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"proxy calibration is empty: {source}")
    destination.write_text("\n".join([lines[0]] * rows) + "\n")


def main() -> int:
    args = _parser().parse_args()
    if args.num_output_frames <= 0 or (args.num_output_frames - 1) % 4:
        raise ValueError("num-output-frames must be positive and satisfy 4n+1")
    contract_path = args.dataset_contract.expanduser().resolve()
    cosmos_root = args.cosmos_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not contract_path.is_file():
        raise ValueError(f"dataset contract is missing: {contract_path}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite inference assets: {output}")
    proxy_root = cosmos_root / "assets" / "robot_multiview-agibot" / "cameras"
    if not proxy_root.is_dir():
        raise ValueError(f"Cosmos proxy calibration directory is missing: {proxy_root}")

    contract = json.loads(contract_path.read_text())
    record = select_record(contract, args.episode, args.split)
    sample_name = record["sample_id"]
    images_dir = output / "input_images"
    cameras_dir = output / "cameras"
    images_dir.mkdir(parents=True)
    cameras_dir.mkdir()
    dataset_root = contract_path.parent

    real_conditions: dict[str, Any] = {}
    withheld_targets: dict[str, Any] = {}
    for source_name, role in SOURCE_TO_ROLE.items():
        camera = record["cameras"][source_name]
        anchor = (dataset_root / camera["real_first_frame_condition"]).resolve()
        target = (dataset_root / camera["video"]).resolve()
        if not anchor.is_file() or _sha256(anchor) != camera["real_first_frame_condition_sha256"]:
            raise ValueError(f"real condition hash mismatch for {source_name}")
        if not target.is_file() or _sha256(target) != camera["video_sha256"]:
            raise ValueError(f"withheld target hash mismatch for {source_name}")
        image_path = images_dir / f"{sample_name}_{role}.png"
        shutil.copy2(anchor, image_path)
        real_conditions[source_name] = {
            "label": "REAL CONDITION",
            "cosmos_role": role,
            "coordinate_frame": camera["coordinate_frame"],
            "path": str(image_path),
            "sha256": _sha256(image_path),
        }
        withheld_targets[source_name] = {
            "label": "WITHHELD REAL TARGET — EVALUATION ONLY",
            "passed_to_model": False,
            "path": str(target),
            "sha256": _sha256(target),
        }

    latent_frames = args.num_output_frames // 4 + 1
    proxy_files: dict[str, Any] = {}
    for role in CAMERA_FILE_ROLES:
        for kind in ("extrinsic", "intrinsic"):
            source = proxy_root / f"0_{kind}_{role}.txt"
            destination = cameras_dir / f"{sample_name}_{kind}_{role}.txt"
            if not source.is_file():
                raise ValueError(f"proxy calibration file is missing: {source}")
            repeat_first_calibration_row(source, destination, latent_frames)
            proxy_files[f"{kind}_{role}"] = {
                "path": str(destination),
                "sha256": _sha256(destination),
                "rows": latent_frames,
            }

    prompt = (
        f"Synchronized three-camera DROID robot manipulation: {record['task_text_real_condition']}. "
        "Preserve the exact robot, gripper, manipulated object, table, and background in every view. "
        "Generate temporally synchronized continuations with the same physical action."
    )
    input_json = output / f"{sample_name}.json"
    _write_json(
        input_json,
        {"name": sample_name, "input_name": sample_name, "prompt": prompt},
    )
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "phiagent_prepare_droid_cosmos_robot_multiview",
        "model": "nvidia/Cosmos-Predict2.5-2B/robot/multiview-agibot",
        "episode_index": args.episode,
        "split": args.split,
        "sample_name": sample_name,
        "input_json": str(input_json),
        "prompt": {"label": "REAL CONDITION", "value": prompt},
        "real_conditions": real_conditions,
        "our_generated_video": {
            "label": "OUR GENERATED VIDEO",
            "definition": "all three camera continuations after the disclosed first frames",
            "expected_frames_per_view": args.num_output_frames,
        },
        "withheld_targets": withheld_targets,
        "camera_metadata": {
            "status": "PROXY — NOT DROID GROUND TRUTH",
            "source": "first row of official Cosmos AgiBot example 0, repeated over time",
            "reason": "the LeRobot DROID-100 conversion does not expose intrinsics/extrinsics",
            "files": proxy_files,
        },
        "leakage_guard": {
            "real_future_target_passed_to_model": False,
            "conditions_limited_to_three_real_first_frames_and_task_text": True,
        },
        "dataset_contract": {"path": str(contract_path), "sha256": _sha256(contract_path)},
    }
    _write_json(output / "inference-manifest.json", manifest)
    print(json.dumps({"output": str(output), "input_json": str(input_json)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
