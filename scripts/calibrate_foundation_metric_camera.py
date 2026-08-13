#!/usr/bin/env python3
"""Bridge learned DA3 depth to sparse independent metric observations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.perception.metric_camera_calibration import (  # noqa: E402
    MetricDepthCalibrationContract,
    calibrate_metric_camera_sequence,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return payload


def _git_state() -> dict[str, object]:
    state: dict[str, object] = {}
    for name, command in (
        ("head", ("git", "rev-parse", "HEAD")),
        ("status", ("git", "status", "--short")),
    ):
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        state[name] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--da3-samples", type=Path, required=True)
    parser.add_argument("--da3-manifest", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/foundation_metric_camera_calibration_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _contract(config: dict[str, Any], source_sha256: str) -> MetricDepthCalibrationContract:
    return MetricDepthCalibrationContract(
        camera_frame=str(config["camera_frame"]),
        world_frame=str(config["world_frame"]),
        timeline=str(config["timeline"]),
        source_video_sha256=source_sha256,
        minimum_anchors=int(config["minimum_anchors"]),
        minimum_independent_groups=int(config["minimum_independent_groups"]),
        maximum_anchor_relative_error_p95=float(
            config["maximum_anchor_relative_error_p95"]
        ),
        maximum_group_holdout_relative_error_p95=float(
            config["maximum_group_holdout_relative_error_p95"]
        ),
        maximum_scale_standard_deviation_fraction=float(
            config["maximum_scale_standard_deviation_fraction"]
        ),
        minimum_robust_inlier_fraction=float(
            config["minimum_robust_inlier_fraction"]
        ),
        maximum_unscaled_camera_motion_m=float(
            config["maximum_unscaled_camera_motion_m"]
        ),
        maximum_exact_asset_reprojection_rmse_px=float(
            config["maximum_exact_asset_reprojection_rmse_px"]
        ),
        bootstrap_samples=int(config["bootstrap_samples"]),
        allowed_exact_asset_sha256=tuple(
            str(value) for value in config.get("allowed_exact_asset_sha256", ())
        ),
    )


def main() -> int:
    args = _parser().parse_args()
    paths = {
        "da3_samples": args.da3_samples.expanduser().resolve(),
        "da3_manifest": args.da3_manifest.expanduser().resolve(),
        "observations": args.observations.expanduser().resolve(),
        "config": args.config.expanduser().resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required calibration inputs are missing: {missing}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {output_dir}")
    output_dir.mkdir(parents=True)

    import numpy as np

    started = time.perf_counter()
    manifest = _json(paths["da3_manifest"])
    declared_samples_sha256 = (
        manifest.get("outputs", {}).get("samples", {}).get("sha256")
    )
    actual_samples_sha256 = _sha256(paths["da3_samples"])
    if declared_samples_sha256 != actual_samples_sha256:
        raise ValueError(
            "DA3 sample hash does not match the supplied manifest output binding"
        )
    observations_payload = _json(paths["observations"])
    config = _json(paths["config"])
    source_sha256 = str(manifest["input"]["sha256"])
    observation_source_sha256 = str(observations_payload.get("source_video_sha256", ""))
    observations = observations_payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("observations must be a list")
    samples = np.load(paths["da3_samples"], allow_pickle=False)

    source_bound = observation_source_sha256 == source_sha256
    if not source_bound or not observations:
        calibration: dict[str, object] = {
            "anchors_total": len(observations),
            "anchors_admissible": 0,
            "independent_group_ids": [],
            "gates": {
                "source_video_hash_bound": source_bound,
                "anchor_pixels_and_frames_valid": False,
                "anchor_evidence_independent_of_foundation_model": False,
                "minimum_anchor_count": False,
                "minimum_independent_groups": False,
            },
            "passed": False,
            "reasons": (
                ["source_video_hash_bound"] if not source_bound else []
            )
            + (["missing_independent_metric_observations"] if not observations else []),
        }
    else:
        calibration = calibrate_metric_camera_sequence(
            np,
            contract=_contract(config, source_sha256),
            frame_indices=samples["source_frame_indices"],
            intrinsics_px=samples["intrinsics_px"],
            world_from_camera=samples["world_from_camera"],
            predicted_depth_m=samples["depth_m"],
            depth_confidence=samples["confidence"],
            anchor_frame_indices=np.asarray(
                [row["frame_index"] for row in observations], dtype=np.int64
            ),
            anchor_xy_px=np.asarray(
                [[row["pixel_x"], row["pixel_y"]] for row in observations],
                dtype=np.float64,
            ),
            anchor_metric_depth_m=np.asarray(
                [row["metric_depth_m"] for row in observations], dtype=np.float64
            ),
            anchor_metric_depth_std_m=np.asarray(
                [row["standard_deviation_m"] for row in observations],
                dtype=np.float64,
            ),
            anchor_group_ids=[str(row["group_id"]) for row in observations],
            anchor_evidence_classes=[
                str(row["evidence_class"]) for row in observations
            ],
            anchor_complete_q=np.asarray(
                [bool(row.get("complete_q", False)) for row in observations]
            ),
            anchor_asset_sha256=[
                str(row.get("asset_sha256", "")) for row in observations
            ],
            anchor_reprojection_rmse_px=np.asarray(
                [float(row.get("reprojection_rmse_px", "nan")) for row in observations]
            ),
            seed=args.seed,
        )

    calibrated_samples_path = output_dir / "calibrated-camera-samples.npz"
    array_keys = (
        "calibrated_depth_m",
        "calibrated_world_from_camera",
        "intrinsics_px",
        "confidence",
    )
    if bool(calibration.get("passed")):
        np.savez_compressed(
            calibrated_samples_path,
            source_frame_indices=samples["source_frame_indices"],
            depth_m=calibration.pop("calibrated_depth_m"),
            world_from_camera=calibration.pop("calibrated_world_from_camera"),
            intrinsics_px=calibration.pop("intrinsics_px"),
            confidence=calibration.pop("confidence"),
            camera_frame=np.asarray(config["camera_frame"]),
            world_frame=np.asarray(config["world_frame"]),
            timeline=np.asarray(config["timeline"]),
            evidence_class=np.asarray("calibrated_geometry"),
        )
    else:
        for key in array_keys:
            calibration.pop(key, None)

    elapsed = time.perf_counter() - started
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if calibration.get("passed") else "PARTIAL",
        "passed": bool(calibration.get("passed")),
        "honest_scope": (
            "WORKING means sparse independent metric observations calibrated the "
            "DA3 camera proposal under the frozen contract; it does not promote the "
            "robot, stems, forces, or final video."
        ),
        "calibration": calibration,
        "source_video_sha256": source_sha256,
        "observation_source_video_sha256": observation_source_sha256,
        "command": [sys.executable, *sys.argv],
        "seed": args.seed,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {"numpy": importlib.metadata.version("numpy")},
        "git": _git_state(),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "outputs": (
            {
                "calibrated_samples": {
                    "path": str(calibrated_samples_path),
                    "sha256": _sha256(calibrated_samples_path),
                }
            }
            if calibrated_samples_path.is_file()
            else {"calibrated_samples": None}
        ),
        "performance": {"calibration_wall_seconds": elapsed},
        "non_negotiable_rules": [
            "Learned depth or language/object-size priors cannot be calibration anchors.",
            "Exact-asset anchors require a frozen asset SHA, complete q, and accepted render reprojection.",
            "Missing observations produce PARTIAL and no calibrated sample artifact.",
        ],
    }
    report_path = output_dir / "calibration-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
