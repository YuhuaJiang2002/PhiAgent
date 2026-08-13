#!/usr/bin/env python3
"""Combine interleaved DA3 samples and render calibrated virtual RGB-D views."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.perception.model_derived_rgbd import (  # noqa: E402
    ModelDerivedRGBDContract,
    audit_model_derived_rgbd,
    depth_splat_rgbd,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON manifest must contain an object: {path}")
    return payload


def _git_state() -> dict[str, Any]:
    result = {}
    for name, command in (
        ("head", ["git", "rev-parse", "HEAD"]),
        ("status", ["git", "status", "--short"]),
    ):
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        result[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-samples", type=Path, required=True)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--offset-samples", type=Path, required=True)
    parser.add_argument("--offset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--virtual-baseline-m", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def _cycle_metrics(
    np: Any,
    source_rgb: Any,
    source_depth: Any,
    source_confidence: Any,
    intrinsics: Any,
    target_from_source: Any,
) -> tuple[dict[str, Any], float, float]:
    forward = depth_splat_rgbd(
        np,
        source_rgb=source_rgb,
        source_depth_m=source_depth,
        source_confidence=source_confidence,
        intrinsics_px=intrinsics,
        target_camera_from_source_camera=target_from_source,
    )
    backward = depth_splat_rgbd(
        np,
        source_rgb=forward["rgb"],
        source_depth_m=forward["depth_m"],
        source_confidence=forward["confidence"],
        intrinsics_px=intrinsics,
        target_camera_from_source_camera=np.linalg.inv(target_from_source),
    )
    valid = backward["valid_mask"] & np.isfinite(source_depth) & (source_depth > 0)
    if not bool(np.any(valid)):
        raise ValueError("virtual camera cycle produced no valid source pixels")
    relative = np.abs(backward["depth_m"][valid] - source_depth[valid]) / np.maximum(
        source_depth[valid], 1e-6
    )
    color_error = np.abs(
        backward["rgb"][valid].astype(np.float32)
        - source_rgb[valid].astype(np.float32)
    )
    return forward, float(np.percentile(relative, 95)), float(np.mean(color_error) / 255.0)


def _write_contact_sheet(
    cv2: Any,
    np: Any,
    path: Path,
    source: Any,
    target: Any,
    valid: Any,
) -> None:
    slots = np.linspace(0, len(source) - 1, min(7, len(source))).round().astype(int)
    rows = []
    for index in slots:
        mask_rgb = np.repeat((valid[index, :, :, None] * 255).astype(np.uint8), 3, axis=2)
        row = np.concatenate((source[index], target[index], mask_rgb), axis=1)
        cv2.putText(
            row,
            f"source | virtual calibrated pose | visible-surface mask   slot={index}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rows.append(row)
    sheet = np.concatenate(rows, axis=0)
    if not cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("failed to write virtual-view contact sheet")


def main() -> int:
    args = _parser().parse_args()
    warnings.simplefilter("error", RuntimeWarning)
    if not (0 < args.virtual_baseline_m <= 0.25):
        raise ValueError("virtual camera baseline must lie in (0, 0.25] metres")
    paths = {
        "primary_samples": args.primary_samples.expanduser().resolve(),
        "primary_manifest": args.primary_manifest.expanduser().resolve(),
        "offset_samples": args.offset_samples.expanduser().resolve(),
        "offset_manifest": args.offset_manifest.expanduser().resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise FileNotFoundError(f"model RGB-D inputs are missing: {missing}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite model RGB-D experiment: {output}")
    output.mkdir(parents=True)
    command = [sys.executable, *sys.argv]
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    config = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "virtual_baseline_m": args.virtual_baseline_m,
        "seed": args.seed,
        "numeric_warnings_policy": "RuntimeWarning=error",
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    import cv2
    import numpy as np

    started = time.perf_counter()
    manifests = [_load_json(paths["primary_manifest"]), _load_json(paths["offset_manifest"])]
    samples = [
        np.load(paths["primary_samples"], allow_pickle=False),
        np.load(paths["offset_samples"], allow_pickle=False),
    ]
    source_hashes = {manifest["input"]["sha256"] for manifest in manifests}
    model_keys = ("name", "revision", "checkpoint_sha256")
    model_signatures = {
        tuple(str(manifest["model"][key]) for key in model_keys)
        for manifest in manifests
    }
    if len(source_hashes) != 1 or len(model_signatures) != 1:
        raise ValueError("interleaved RGB-D runs must bind the same source and model")
    for index, (manifest, sample, sample_name) in enumerate(
        zip(manifests, samples, ("primary_samples", "offset_samples"), strict=True)
    ):
        if manifest.get("model", {}).get("evidence_class") != "foundation_model_estimate":
            raise ValueError("model-derived RGB-D cannot be relabeled as calibrated geometry")
        if manifest["outputs"]["samples"]["sha256"] != config["inputs"][sample_name]["sha256"]:
            raise ValueError(f"DA3 sample hash mismatch for group {index}")
        if len(sample["source_frame_indices"]) != int(manifest["sampling"]["sampled_frames"]):
            raise ValueError(f"DA3 sample count mismatch for group {index}")
    frame_sets = [set(row["source_frame_indices"].tolist()) for row in samples]
    if frame_sets[0] & frame_sets[1]:
        raise ValueError("model RGB-D time lattices must be disjoint")

    arrays = {}
    array_keys = (
        "processed_images_rgb",
        "depth_m",
        "confidence",
        "intrinsics_px",
        "camera_from_world",
        "world_from_camera",
    )
    unsorted_frames = np.concatenate([row["source_frame_indices"] for row in samples])
    unsorted_groups = np.concatenate(
        [np.full(len(row["source_frame_indices"]), index, dtype=np.int16) for index, row in enumerate(samples)]
    )
    order = np.argsort(unsorted_frames)
    arrays["source_frame_indices"] = unsorted_frames[order].astype(np.int32)
    arrays["source_group_indices"] = unsorted_groups[order]
    for key in array_keys:
        arrays[key] = np.concatenate([row[key] for row in samples], axis=0)[order]
    arrays["source_group_world_frames"] = np.asarray(
        ["world:da3_primary_learned_metric", "world:da3_offset_learned_metric"]
    )
    arrays["camera_frame"] = np.asarray("camera:da3_processed_pixels")
    arrays["timeline"] = np.asarray("frame:source_video")
    combined_path = output / "interleaved-model-rgbd.npz"
    np.savez_compressed(combined_path, **arrays)

    offset = samples[1]
    target_rgbs = []
    target_depths = []
    target_confidences = []
    target_valid = []
    target_transforms = []
    cycle_depth_p95 = []
    cycle_rgb_mae = []
    for index in range(len(offset["source_frame_indices"])):
        target_from_source = np.eye(4, dtype=np.float64)
        direction = -1.0 if index % 2 == 0 else 1.0
        target_from_source[0, 3] = direction * args.virtual_baseline_m
        forward, depth_error, rgb_error = _cycle_metrics(
            np,
            offset["processed_images_rgb"][index],
            offset["depth_m"][index].astype(np.float32),
            offset["confidence"][index].astype(np.float32),
            offset["intrinsics_px"][index],
            target_from_source,
        )
        target_rgbs.append(forward["rgb"])
        target_depths.append(forward["depth_m"])
        target_confidences.append(forward["confidence"])
        target_valid.append(forward["valid_mask"])
        target_transforms.append(target_from_source)
        cycle_depth_p95.append(depth_error)
        cycle_rgb_mae.append(rgb_error)
    target_rgbs_array = np.asarray(target_rgbs, dtype=np.uint8)
    target_depths_array = np.asarray(target_depths, dtype=np.float32)
    target_confidences_array = np.asarray(target_confidences, dtype=np.float32)
    target_valid_array = np.asarray(target_valid, dtype=bool)
    target_transforms_array = np.asarray(target_transforms, dtype=np.float32)
    coverage = np.mean(target_valid_array, axis=(1, 2))
    virtual_path = output / "calibrated-virtual-view-rgbd.npz"
    np.savez_compressed(
        virtual_path,
        source_frame_indices=offset["source_frame_indices"].astype(np.int32),
        target_images_rgb=target_rgbs_array,
        target_depth_m=target_depths_array.astype(np.float16),
        target_confidence=target_confidences_array.astype(np.float16),
        target_valid_mask=target_valid_array,
        intrinsics_px=offset["intrinsics_px"].astype(np.float32),
        target_camera_from_source_camera=target_transforms_array,
        source_camera_from_target_camera=np.linalg.inv(target_transforms_array).astype(np.float32),
        source_camera_frame=np.asarray("camera:da3_offset_processed"),
        target_camera_frame=np.asarray("camera:virtual_constructed_4cm"),
        timeline=np.asarray("frame:source_video"),
        evidence_class=np.asarray("foundation_model_estimate"),
    )
    contact_sheet = output / "virtual-view-contact-sheet.png"
    _write_contact_sheet(
        cv2,
        np,
        contact_sheet,
        offset["processed_images_rgb"],
        target_rgbs_array,
        target_valid_array,
    )
    preview = output / "virtual-view-preview.mp4"
    height, width = target_rgbs_array.shape[1:3]
    writer = cv2.VideoWriter(
        str(preview), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (width * 2, height)
    )
    if not writer.isOpened():
        raise RuntimeError("failed to open virtual-view preview writer")
    try:
        for source_rgb, target_rgb in zip(
            offset["processed_images_rgb"], target_rgbs_array, strict=True
        ):
            pair = np.concatenate((source_rgb, target_rgb), axis=1)
            writer.write(cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    contract = ModelDerivedRGBDContract(
        source_video_sha256=next(iter(source_hashes)),
        timeline="frame:source_video",
        fps=float(manifests[0]["input"]["fps"]),
        model_name=manifests[0]["model"]["name"],
        model_revision=manifests[0]["model"]["revision"],
        checkpoint_sha256=manifests[0]["model"]["checkpoint_sha256"],
        source_group_frames=tuple(str(value) for value in arrays["source_group_world_frames"]),
        virtual_camera_frame="camera:virtual_constructed_4cm",
    )
    group_medians = np.asarray(
        [float(np.median(row["depth_m"].astype(np.float32))) for row in samples]
    )
    audit = audit_model_derived_rgbd(
        np,
        contract=contract,
        source_frame_indices=arrays["source_frame_indices"],
        source_group_indices=arrays["source_group_indices"],
        depth_m=arrays["depth_m"],
        virtual_view_coverage=coverage,
        cycle_depth_relative_error_p95=np.asarray(cycle_depth_p95),
        group_median_depth_m=group_medians,
    )
    elapsed = time.perf_counter() - started
    outputs = {
        "interleaved_rgbd": {"path": str(combined_path), "sha256": _sha256(combined_path)},
        "virtual_view_rgbd": {"path": str(virtual_path), "sha256": _sha256(virtual_path)},
        "contact_sheet": {"path": str(contact_sheet), "sha256": _sha256(contact_sheet)},
        "preview": {"path": str(preview), "sha256": _sha256(preview)},
    }
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if audit["proposal_passed"] else "PARTIAL",
        "physical_status": "PARTIAL",
        "passed": bool(audit["proposal_passed"]),
        "physical_calibration_passed": False,
        "source_video_sha256": contract.source_video_sha256,
        "evidence_class": "foundation_model_estimate",
        "command": command,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "opencv-python")
        },
        "seed": args.seed,
        "numeric_warnings_policy": "RuntimeWarning=error",
        "git": _git_state(),
        "model": manifests[0]["model"],
        "coordinate_contract": {
            "timeline": contract.timeline,
            "camera": "camera:da3_processed_pixels",
            "group_world_frames": list(contract.source_group_frames),
            "virtual_camera": contract.virtual_camera_frame,
            "virtual_extrinsics": "target_camera_from_source_camera, exact by construction",
            "depth_unit": "learned metres; no independent scale authority",
        },
        "audit": audit,
        "virtual_view": {
            "baseline_m": args.virtual_baseline_m,
            "samples": int(len(target_rgbs_array)),
            "mean_coverage": float(np.mean(coverage)),
            "coverage_p05": float(np.percentile(coverage, 5)),
            "cycle_depth_relative_error_p95_max": float(np.max(cycle_depth_p95)),
            "cycle_rgb_mae_0_1_mean": float(np.mean(cycle_rgb_mae)),
            "new_occluded_surface_fraction": 0.0,
        },
        "inputs": config["inputs"],
        "outputs": outputs,
        "performance": {
            "combine_render_audit_wall_seconds": elapsed,
            "virtual_views_per_second": len(target_rgbs_array) / max(elapsed, 1e-12),
        },
        "claim_boundary": (
            "These are new model-derived synchronized RGB-D samples and exact-pose virtual "
            "views of already visible surfaces. They are useful proposals/training data, not "
            "an independent sensor, absolute-scale calibration, hidden-surface observation, "
            "full-q measurement, or force measurement."
        ),
    }
    report_path = output / "model-derived-rgbd-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "run.log").write_text(
        f"status={report['status']} samples={audit['metrics']['samples']} "
        f"coverage={audit['metrics']['mean_virtual_view_coverage']:.6f} "
        f"wall_seconds={elapsed:.6f}\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if audit["proposal_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
