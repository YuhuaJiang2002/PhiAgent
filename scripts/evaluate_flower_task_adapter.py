#!/usr/bin/env python3
"""Compare zero-shot and task-LoRA outputs on one held-out flower-contact clip."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--zero-shot", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--zero-metadata", type=Path, required=True)
    parser.add_argument("--adapted-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-radius", type=int, default=22)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(cv2: Any, path: Path) -> list[Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video contains no decoded frames: {path}")
    return frames


def _similarity(np: Any, first: Any, second: Any, mask: Any | None = None) -> float:
    delta = np.abs(first.astype(np.float32) - second.astype(np.float32))
    if mask is not None:
        delta = delta[mask]
    return math.exp(-float(delta.mean()) / 32.0)


def _motion_similarity(np: Any, target: list[Any], candidate: list[Any], masks: list[Any]) -> float:
    values = []
    for index in range(1, len(target)):
        target_motion = np.abs(
            target[index].astype(np.float32) - target[index - 1].astype(np.float32)
        )
        candidate_motion = np.abs(
            candidate[index].astype(np.float32) - candidate[index - 1].astype(np.float32)
        )
        union = masks[index] | masks[index - 1]
        residual = np.abs(target_motion[union] - candidate_motion[union])
        values.append(math.exp(-float(residual.mean()) / 32.0))
    return sum(values) / len(values)


def _contact_observation(row: dict[str, Any]) -> tuple[float, float, bool]:
    if "contact_xy" in row and "contact_active" in row:
        x, y = row["contact_xy"]
        return float(x), float(y), bool(row["contact_active"])
    if "right_hand_xy" in row and "right_contact_required" in row:
        x, y = row["right_hand_xy"]
        return float(x), float(y), bool(row["right_contact_required"])
    raise ValueError("trajectory row lacks a supported contact observation")


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return payload


def _validate_evaluation_lineage(
    paths: dict[str, Path],
) -> dict[str, object]:
    hashes = {name: _sha256(path) for name, path in paths.items()}
    validation = _json_object(paths["dataset_validation"])
    if validation.get("passed") is not True:
        raise ValueError("dataset validation did not pass")
    matching_records = [
        row
        for row in validation.get("clip_records", ())
        if row.get("split") == "validation"
        and row.get("target_sha256") == hashes["target"]
        and row.get("control_sha256") == hashes["control"]
        and row.get("trajectory_sha256") == hashes["trajectory"]
        and row.get("reference_sha256") == hashes["reference"]
    ]
    if len(matching_records) != 1:
        raise ValueError(
            "target, control, trajectory, and reference do not match one held-out record"
        )
    manifest = _json_object(paths["frozen_manifest"])
    manifest_assets = manifest.get("assets", ())

    def manifest_matches(name: str, kind: str, split: str) -> bool:
        return any(
            row.get("sha256") == hashes[name]
            and row.get("kind") == kind
            and row.get("split") == split
            for row in manifest_assets
        )

    manifest_gates = {
        "target_is_validation_asset": manifest_matches(
            "target",
            "target_video",
            "validation",
        ),
        "control_is_validation_asset": manifest_matches(
            "control",
            "vace_control_video",
            "validation",
        ),
        "reference_is_training_identity_asset": manifest_matches(
            "reference",
            "vace_reference_image",
            "train",
        ),
    }
    if not all(manifest_gates.values()):
        raise ValueError("held-out files do not match the frozen dataset manifest")

    zero_metadata = _json_object(paths["zero_metadata"])
    adapted_metadata = _json_object(paths["adapted_metadata"])
    metadata_by_arm = {
        "zero": (zero_metadata, "zero_shot"),
        "adapted": (adapted_metadata, "adapted"),
    }
    for arm, (metadata, output_name) in metadata_by_arm.items():
        if metadata.get("status") != "completed":
            raise ValueError(f"{arm} inference metadata is not completed")
        if metadata.get("output_sha256") != hashes[output_name]:
            raise ValueError(f"{arm} output hash does not match metadata")
        if Path(str(metadata.get("output", ""))).resolve() != paths[output_name]:
            raise ValueError(f"{arm} output path does not match metadata")
        inputs = metadata.get("inputs", {})
        if (
            inputs.get("control_sha256") != hashes["control"]
            or inputs.get("reference_sha256") != hashes["reference"]
        ):
            raise ValueError(f"{arm} conditioning hashes do not match held-out inputs")
    if zero_metadata["inputs"].get("lora_sha256") is not None:
        raise ValueError("zero-shot metadata unexpectedly contains a LoRA")
    if adapted_metadata["inputs"].get("lora_sha256") != hashes["adapter"]:
        raise ValueError("adapted metadata does not bind the supplied adapter")
    comparable_config_keys = (
        "checkpoint_dir",
        "control_video",
        "reference_image",
        "denoising_strength",
        "prompt",
        "gpu",
        "minimum_free_gpu_mib",
        "seed",
        "height",
        "width",
        "num_frames",
        "fps",
        "steps",
    )
    zero_config = zero_metadata.get("config", {})
    adapted_config = adapted_metadata.get("config", {})
    config_gates = {
        key: zero_config.get(key) == adapted_config.get(key)
        for key in comparable_config_keys
    }
    checkpoint_gates = (
        zero_metadata.get("inputs", {}).get("checkpoint_files")
        == adapted_metadata.get("inputs", {}).get("checkpoint_files")
    )
    source_git_matches = (
        zero_metadata.get("git", {}).get("source_git_head")
        == adapted_metadata.get("git", {}).get("source_git_head")
        and zero_metadata.get("git", {}).get("source_git_status_sha256")
        == adapted_metadata.get("git", {}).get("source_git_status_sha256")
    )
    if (
        not all(config_gates.values())
        or not checkpoint_gates
        or not source_git_matches
    ):
        raise ValueError("zero-shot and adapted inference configurations are not matched")
    return {
        "passed": True,
        "heldout_record": matching_records[0],
        "manifest_gates": manifest_gates,
        "matched_config_gates": config_gates,
        "checkpoint_files_match": checkpoint_gates,
        "source_git_matches": source_git_matches,
        "hashes": hashes,
    }


def main() -> int:
    args = _parser().parse_args()
    paths = {
        "target": args.target.expanduser().resolve(),
        "zero_shot": args.zero_shot.expanduser().resolve(),
        "adapted": args.adapted.expanduser().resolve(),
        "trajectory": args.trajectory.expanduser().resolve(),
        "control": args.control.expanduser().resolve(),
        "reference": args.reference.expanduser().resolve(),
        "adapter": args.adapter.expanduser().resolve(),
        "dataset_validation": args.dataset_validation.expanduser().resolve(),
        "frozen_manifest": args.frozen_manifest.expanduser().resolve(),
        "zero_metadata": args.zero_metadata.expanduser().resolve(),
        "adapted_metadata": args.adapted_metadata.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{name} does not exist or is empty: {path}")
    if args.contact_radius <= 0:
        raise ValueError("contact radius must be positive")
    lineage = _validate_evaluation_lineage(paths)

    import cv2
    import numpy as np

    target = _decode(cv2, paths["target"])
    zero_shot = _decode(cv2, paths["zero_shot"])
    adapted = _decode(cv2, paths["adapted"])
    trajectory = json.loads(paths["trajectory"].read_text())
    rows = trajectory["frames"]
    if not len(target) == len(zero_shot) == len(adapted) == len(rows):
        raise RuntimeError("target, candidates, and trajectory must have equal frame counts")
    shape = target[0].shape
    if any(frame.shape != shape for sequence in (target, zero_shot, adapted) for frame in sequence):
        raise RuntimeError("all validation frames must have the same shape")
    height, width = shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    contact_masks = []
    contact_frames = []
    for index, row in enumerate(rows):
        x, y, required = _contact_observation(row)
        contact_masks.append((xx - x) ** 2 + (yy - y) ** 2 <= args.contact_radius**2)
        if required:
            contact_frames.append(index)
    if not contact_frames:
        raise ValueError("held-out trajectory contains no required contact frames")
    zero_contact = sum(
        _similarity(np, target[index], zero_shot[index], contact_masks[index])
        for index in contact_frames
    ) / len(contact_frames)
    adapted_contact = sum(
        _similarity(np, target[index], adapted[index], contact_masks[index])
        for index in contact_frames
    ) / len(contact_frames)
    zero_global = sum(
        _similarity(np, target_frame, candidate)
        for target_frame, candidate in zip(target, zero_shot)
    ) / len(target)
    adapted_global = sum(
        _similarity(np, target_frame, candidate)
        for target_frame, candidate in zip(target, adapted)
    ) / len(target)
    zero_motion = _motion_similarity(np, target, zero_shot, contact_masks)
    adapted_motion = _motion_similarity(np, target, adapted, contact_masks)
    distinctness = sum(
        float(
            np.mean(
                np.abs(
                    zero_frame.astype(np.float32) - adapted_frame.astype(np.float32)
                )
            )
        )
        for zero_frame, adapted_frame in zip(zero_shot, adapted)
    ) / len(target)
    metrics = {
        "global_similarity": {"zero_shot": zero_global, "adapted": adapted_global},
        "contact_roi_similarity": {
            "zero_shot": zero_contact,
            "adapted": adapted_contact,
        },
        "contact_motion_similarity": {
            "zero_shot": zero_motion,
            "adapted": adapted_motion,
        },
        "adapted_minus_zero_contact": adapted_contact - zero_contact,
        "adapted_minus_zero_motion": adapted_motion - zero_motion,
        "adapted_zero_mean_absolute_difference": distinctness,
    }
    gates = {
        "adapted_output_is_distinct": distinctness >= 0.5,
        "contact_roi_not_regressed": adapted_contact >= zero_contact,
        "contact_motion_not_regressed": adapted_motion >= zero_motion,
    }
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "status": "PARTIAL",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "lineage": lineage,
        "frame_count": len(target),
        "contact_frames": contact_frames,
        "contact_radius_pixels": args.contact_radius,
        "coordinate_frame": trajectory.get("coordinate_frames", {}).get(
            "control",
            "camera:synthetic_pixels",
        ),
        "metrics": metrics,
        "gates": gates,
        "all_proxy_gates_pass": all(gates.values()),
        "limitations": [
            "This held-out clip is synthetic and cannot establish real-video quality.",
            "Pixel similarity near a known contact point is not flower-instance or physics proof.",
            "The smoke LoRA has seen only 12 short procedural training clips.",
        ],
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
