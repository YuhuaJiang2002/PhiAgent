#!/usr/bin/env python3
"""Prepare leakage-safe Cosmos3 I2V SFT data from DROID 2x2 composites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--materialization", choices=("symlink", "copy"), default="symlink"
    )
    parser.add_argument("--git-commit")
    parser.add_argument("--git-branch")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"{label} is missing or empty: {resolved}")
    return resolved


def validate_contract(contract: dict[str, Any]) -> None:
    leakage = contract.get("leakage_checks", {})
    if leakage.get("final_holdout_used_for_training") is not False:
        raise ValueError("source contract does not isolate final holdout from training")
    video = contract.get("video_contract", {})
    expected = {"width": 768, "height": 432, "fps": 16, "frames": 97}
    actual = {key: int(video.get(key, -1)) for key in expected}
    if actual != expected:
        raise ValueError(f"unexpected DROID composite video contract: {actual}")
    records = contract.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("source contract contains no records")
    training_episodes = {
        int(record["episode_index"])
        for record in records
        if record.get("split") == "train"
    }
    heldout_episodes = {
        int(record["episode_index"])
        for record in records
        if record.get("split") in {"validation", "final_holdout"}
    }
    overlap = sorted(training_episodes & heldout_episodes)
    if overlap:
        raise ValueError(f"episode leakage across train and heldout splits: {overlap}")


def structured_caption(record: dict[str, Any]) -> dict[str, Any]:
    task = " ".join(str(record["raw_task_text"]).strip().split()).rstrip(".")
    if not task:
        task = "perform the manipulation task"
    dense = str(record["prompt"]).strip()
    return {
        "subjects": [
            {
                "description": (
                    "One consistent DROID robot and the same manipulated object, shown "
                    "synchronously from two exterior cameras and one wrist camera"
                ),
                "action": task,
                "state_changes": (
                    "Only the robot joints and manipulated object move; robot identity, "
                    "object identity, table, fixtures, and background remain unchanged"
                ),
            }
        ],
        "background_setting": (
            "The exact real DROID workspace visible in the condition frame, preserved "
            "without camera motion, scene cuts, new objects, or removed objects"
        ),
        "cinematography": {
            "camera_motion": "Static and synchronized in all three active views",
            "framing": (
                "Fixed 2x2 layout: exterior camera 1 top-left, exterior camera 2 "
                "top-right, wrist first-person camera bottom-left, inactive black "
                "tile bottom-right"
            ),
            "camera_angle": "Two fixed third-person angles and one fixed first-person angle",
            "focus": "Sharp robot, gripper, and manipulated object in every active view",
        },
        "actions": [{"time": "0:00-0:06", "description": task}],
        "temporal_caption": dense,
        "style_medium": "Photoreal live-action robot manipulation footage",
        "resolution": {"W": 768, "H": 432},
        "aspect_ratio": "16,9",
        "duration": "6.0625s",
        "fps": 16,
    }


def sft_record(record: dict[str, Any], video_relative_path: str) -> dict[str, Any]:
    caption_json = structured_caption(record)
    return {
        "uuid": str(record["sample_id"]),
        "duration": 97 / 16,
        "width": 768,
        "height": 432,
        "vision_path": video_relative_path,
        "t2w_windows": [
            {
                "start_frame": 0,
                "end_frame": 96,
                "temporal_interval": 1,
                "caption_json": caption_json,
                "caption": caption_json["temporal_caption"],
            }
        ],
    }


def _materialize(source: Path, destination: Path, method: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if method == "copy":
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source)


def main() -> int:
    args = _parser().parse_args()
    contract_path = _require_file(args.composite_contract, "composite dataset contract")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Cosmos3 SFT dataset: {output}")
    output.mkdir(parents=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract(contract)
    source_root = contract_path.parent

    source_hashes: dict[str, dict[str, str]] = {}
    split_counts = {"train": 0, "validation": 0}
    train_rows: list[dict[str, Any]] = []
    validation_samples: list[dict[str, Any]] = []
    for record in contract["records"]:
        split = str(record["split"])
        if split not in {"train", "validation"}:
            continue
        sample_id = str(record["sample_id"])
        video = _require_file(
            source_root / record["real_multiview_target_video"],
            f"{sample_id} real multiview video",
        )
        condition = _require_file(
            source_root / record["real_composite_first_frame_condition"],
            f"{sample_id} real first-frame condition",
        )
        if _sha256(video) != record["real_multiview_target_video_sha256"]:
            raise ValueError(f"target-video hash mismatch: {sample_id}")
        if _sha256(condition) != record["real_composite_first_frame_condition_sha256"]:
            raise ValueError(f"condition-image hash mismatch: {sample_id}")
        source_hashes[sample_id] = {
            "video": record["real_multiview_target_video_sha256"],
            "condition": record["real_composite_first_frame_condition_sha256"],
        }
        caption_json = structured_caption(record)
        if split == "train":
            destination = output / "train/videos" / f"{sample_id}.mp4"
            _materialize(video, destination, args.materialization)
            train_rows.append(sft_record(record, f"videos/{sample_id}.mp4"))
        else:
            condition_destination = output / "val/images" / f"{sample_id}.png"
            target_destination = output / "val/targets" / f"{sample_id}.mp4"
            _materialize(condition, condition_destination, args.materialization)
            _materialize(video, target_destination, args.materialization)
            prompt_path = output / "val/prompts" / f"{sample_id}.json.txt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(
                json.dumps(caption_json, sort_keys=True) + "\n", encoding="utf-8"
            )
            inference_spec = {
                "name": sample_id,
                "model_mode": "image2video",
                "prompt": json.dumps(caption_json, sort_keys=True),
                "vision_path": f"../images/{sample_id}.png",
                "resolution": "480",
                "aspect_ratio": "16,9",
                "num_frames": 93,
                "fps": 16,
                "num_steps": 35,
                "guidance": 6.0,
                "shift": 10.0,
                "seed": int(contract.get("seed", 20260812)),
                "enable_sound": False,
            }
            spec_path = output / "val/inference_prompt_i2v" / f"{sample_id}.json"
            _write_json(spec_path, inference_spec)
            validation_samples.append(
                {
                    "sample_id": sample_id,
                    "episode_index": int(record["episode_index"]),
                    "condition": str(condition_destination.relative_to(output)),
                    "condition_label": "REAL CONDITION",
                    "prompt": str(prompt_path.relative_to(output)),
                    "prompt_label": record["task_text_condition_kind"],
                    "withheld_target": str(target_destination.relative_to(output)),
                    "withheld_target_label": "WITHHELD REAL TARGET — EVALUATION ONLY",
                    "inference_spec": str(spec_path.relative_to(output)),
                    "generated_continuation_label": "OUR GENERATED VIDEO",
                }
            )
        split_counts[split] += 1

    train_jsonl = output / "train/video_dataset_file.jsonl"
    train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with train_jsonl.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING",
        "method": "cosmos3_nano_droid_multiview_i2v_sft_dataset",
        "source_contract": str(contract_path),
        "source_contract_sha256": _sha256(contract_path),
        "materialization": args.materialization,
        "split_counts": split_counts,
        "training": {
            "jsonl": str(train_jsonl.relative_to(output)),
            "jsonl_sha256": _sha256(train_jsonl),
            "conditioning_distribution_required": {"i2v_first_frame": 1.0},
            "target_frames": "real synchronized continuation frames 2-97",
        },
        "validation_samples": validation_samples,
        "exclusions": {
            "legacy_dev": "excluded from Cosmos3 training and validation selection",
            "final_holdout": "excluded from training and validation; remains untouched",
        },
        "leakage_checks": {
            "episode_disjoint_train_validation": True,
            "final_holdout_used_for_training": False,
            "final_holdout_used_for_checkpoint_selection": False,
            "validation_future_frames_are_model_inputs": False,
        },
        "source_hashes": source_hashes,
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "git": {
            "commit": args.git_commit or "unresolved",
            "branch": args.git_branch,
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": os.getcwd(),
    }
    _write_json(output / "dataset-contract.json", summary)
    (output / "command.txt").write_text(
        summary["command_shell"] + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "split_counts": split_counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
