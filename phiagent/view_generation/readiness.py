"""Fail-closed readiness checks for calibrated robot novel-view generation."""

from __future__ import annotations

from collections.abc import Mapping
import math
from collections.abc import Sequence
from typing import Any


REQUIRED_STREAMS = (
    "observation.images.wrist_image_left",
    "observation.images.exterior_image_1_left",
)


def extrinsic_variation(
    values: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Measure translation and Euler range of frame-explicit 6-D extrinsics."""

    rows = tuple(tuple(float(value) for value in row) for row in values)
    if not rows or any(len(row) != 6 for row in rows):
        raise ValueError("extrinsics must be a non-empty sequence of 6-D rows")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("extrinsics must contain only finite values")
    ranges = [
        max(row[channel] for row in rows) - min(row[channel] for row in rows)
        for channel in range(6)
    ]
    return {
        "translation_max_range_m": max(ranges[:3]),
        "euler_max_range_rad": max(ranges[3:]),
    }


def audit_droid_novel_view_readiness(
    dataset_info: Mapping[str, Any],
    raw_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Report whether DROID inputs can support a calibrated Strict-W benchmark."""

    missing = []
    features = dataset_info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("DROID dataset info requires a features object")
    for stream in REQUIRED_STREAMS:
        feature = features.get(stream)
        if not isinstance(feature, Mapping) or feature.get("dtype") != "video":
            missing.append(f"video stream {stream}")
    for field in ("observation.state", "action", "timestamp", "episode_index"):
        if field not in features:
            missing.append(f"frame-aligned field {field}")
    state = features.get("observation.state")
    action = features.get("action")
    for label, feature in (("state", state), ("action", action)):
        names = feature.get("names") if isinstance(feature, Mapping) else None
        motors = names.get("motors") if isinstance(names, Mapping) else None
        if (
            not isinstance(motors, list)
            or not motors
            or all(str(name).startswith("motor_") for name in motors)
        ):
            missing.append(f"semantic {label} channel convention and coordinate frame")
    if raw_contract is None:
        missing.extend(
            (
                "raw DROID rights review",
                "camera serial and calibration hashes",
                "wrist and exterior intrinsics/distortion",
                "time-varying world_T_wrist_camera",
                "world_T_exterior_camera",
                "video-to-robot timestamp offset evidence",
                "stereo/depth lineage for visible-surface geometry",
            )
        )
    else:
        if raw_contract.get("rights_reviewed") is not True:
            missing.append("raw DROID rights review")
        cameras = raw_contract.get("cameras")
        if not isinstance(cameras, Mapping):
            missing.append("camera calibration records")
        else:
            for stream in REQUIRED_STREAMS:
                camera = cameras.get(stream)
                if not isinstance(camera, Mapping):
                    missing.append(f"calibration for {stream}")
                    continue
                frame = str(camera.get("coordinate_frame", ""))
                if not frame.startswith("camera:"):
                    missing.append(f"explicit camera frame for {stream}")
                if not isinstance(camera.get("intrinsics"), Mapping):
                    missing.append(f"intrinsics for {stream}")
                extrinsics = camera.get("extrinsics")
                if not isinstance(extrinsics, Mapping):
                    missing.append(f"extrinsics for {stream}")
                elif stream.endswith("wrist_image_left") and extrinsics.get("mode") != "per_frame":
                    missing.append("per-frame wrist-camera extrinsics")
        if raw_contract.get("timestamp_alignment_verified") is not True:
            missing.append("video-to-robot timestamp offset evidence")
        if raw_contract.get("depth_lineage_verified") is not True:
            missing.append("stereo/depth lineage for visible-surface geometry")
    unique_missing = sorted(set(missing))
    ready = not unique_missing
    return {
        "schema_version": "1.0.0",
        "ready": ready,
        "status": "WORKING" if ready else "BLOCKED",
        "lane": "Strict-W: wrist RGB + task + calibrated target camera; no target pixels",
        "missing_requirements": unique_missing,
        "claim_boundary": (
            "Readiness validates information and calibration availability only. It does "
            "not establish novel-view quality, action preservation, or SOTA."
        ),
    }
