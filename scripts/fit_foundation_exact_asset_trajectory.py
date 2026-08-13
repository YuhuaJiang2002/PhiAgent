#!/usr/bin/env python3
"""Validate foundation proposals through exact-asset held-out rendering.

The heavyweight keypoint and renderer adapters remain optional. They exchange a
NumPy evidence bundle with this fail-closed compiler, so importing phiagent does
not require a simulator, CUDA, PyTorch, or a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.perception.exact_asset_trajectory import (  # noqa: E402
    ExactAssetTrajectoryContract,
    validate_exact_asset_trajectory,
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


def _joint_schema(asset_paths: dict[str, Path]) -> tuple[tuple[str, ...], tuple[tuple[float, float], ...], dict[str, int]]:
    names: list[str] = []
    limits: list[tuple[float, float]] = []
    counts: dict[str, int] = {}
    for asset_name, path in asset_paths.items():
        rows = []
        for joint in ET.parse(path).getroot().findall(".//joint"):
            name = joint.get("name")
            bounds = joint.get("range")
            if name and bounds:
                values = tuple(float(value) for value in bounds.split())
                if len(values) != 2:
                    raise ValueError(f"joint {name!r} requires a two-value range")
                rows.append((name, (values[0], values[1])))
        counts[asset_name] = len(rows)
        names.extend(name for name, _ in rows)
        limits.extend(limit for _, limit in rows)
    if len(set(names)) != len(names):
        raise ValueError("joint names must be unique across combined exact assets")
    return tuple(names), tuple(limits), counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--da3-manifest", type=Path, required=True)
    parser.add_argument("--g1-model", type=Path, required=True)
    parser.add_argument("--sharpa-left-model", type=Path, required=True)
    parser.add_argument("--sharpa-right-model", type=Path, required=True)
    parser.add_argument("--evidence-bundle", type=Path)
    parser.add_argument("--observability-audit", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/foundation_exact_asset_trajectory_v2.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def _missing_validation(reason: str) -> dict[str, object]:
    gate_names = (
        "source_video_hash_bound",
        "exact_asset_hashes_match_registry",
        "complete_finite_q",
        "joint_limits_passed",
        "joint_velocity_passed",
        "proper_camera_from_robot_base_se3",
        "visible_keypoint_coverage",
        "heldout_frame_count",
        "heldout_group_count",
        "heldout_reprojection_bounded",
        "heldout_silhouette_iou_bounded",
        "full_q_posterior_observable",
        "metric_base_translation_observable",
        "selected_asset_beats_alternatives",
        "dense_full_source_timeline",
        "joint_schema_matches_exact_assets",
        "joint_state_evidence_has_physical_authority",
        "proposal_and_renderer_provenance_named",
    )
    return {
        "gates": {name: False for name in gate_names},
        "proposal_passed": False,
        "passed": False,
        "reasons": [reason],
    }


def main() -> int:
    args = _parser().parse_args()
    paths = {
        "video": args.video.expanduser().resolve(),
        "da3_manifest": args.da3_manifest.expanduser().resolve(),
        "g1_model": args.g1_model.expanduser().resolve(),
        "sharpa_left_model": args.sharpa_left_model.expanduser().resolve(),
        "sharpa_right_model": args.sharpa_right_model.expanduser().resolve(),
        "config": args.config.expanduser().resolve(),
    }
    optional_paths = {
        "evidence_bundle": args.evidence_bundle.expanduser().resolve()
        if args.evidence_bundle
        else None,
        "observability_audit": args.observability_audit.expanduser().resolve()
        if args.observability_audit
        else None,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    missing.extend(
        name
        for name, path in optional_paths.items()
        if path is not None and not path.is_file()
    )
    if missing:
        raise FileNotFoundError(f"required trajectory inputs are missing: {missing}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {output_dir}")
    output_dir.mkdir(parents=True)

    import numpy as np

    started = time.perf_counter()
    config = _json(paths["config"])
    manifest = _json(paths["da3_manifest"])
    audit = (
        _json(optional_paths["observability_audit"])
        if optional_paths["observability_audit"]
        else None
    )
    source_sha256 = _sha256(paths["video"])
    manifest_source_sha256 = str(manifest["input"]["sha256"])
    asset_paths = {
        name: paths[name]
        for name in ("g1_model", "sharpa_left_model", "sharpa_right_model")
    }
    joint_names, joint_limits, joint_counts = _joint_schema(asset_paths)
    expected_counts = {
        name: int(value)
        for name, value in config["expected_joint_counts"].items()
        if name != "total"
    }
    expected_total = int(config["expected_joint_counts"]["total"])
    count_gate = joint_counts == expected_counts and len(joint_names) == expected_total
    actual_asset_sha256 = {name: _sha256(path) for name, path in asset_paths.items()}
    registry_asset_sha256 = {
        str(name): str(value)
        for name, value in config["expected_asset_sha256"].items()
    }
    asset_file_gate = actual_asset_sha256 == registry_asset_sha256
    source_file_gate = source_sha256 == manifest_source_sha256
    evidence_path = optional_paths["evidence_bundle"]
    trajectory_path = output_dir / "robot-trajectory.npz"

    if evidence_path is None:
        validation = _missing_validation(
            "missing_foundation_keypoint_and_exact_render_bundle"
        )
        bundle_metadata: dict[str, object] | None = None
    else:
        bundle = np.load(evidence_path, allow_pickle=False)
        required_fields = (
            "source_video_sha256",
            "joint_state_evidence_class",
            "proposal_model_name",
            "proposal_model_revision",
            "renderer_name",
            "renderer_revision",
            "renderer_asset_names",
            "renderer_asset_sha256",
            "joint_names",
            "source_frame_indices",
            "joint_positions_rad",
            "camera_from_robot_base",
            "observed_keypoints_px",
            "rendered_keypoints_px",
            "keypoint_confidence",
            "fit_frame_mask",
            "heldout_group_ids",
            "silhouette_iou",
            "joint_standard_deviation_rad",
            "base_translation_standard_deviation_m",
            "alternative_asset_reprojection_rmse_px",
        )
        absent = [name for name in required_fields if name not in bundle.files]
        if absent:
            raise ValueError(f"evidence bundle is missing required fields: {absent}")
        renderer_hashes = {
            str(name): str(digest)
            for name, digest in zip(
                bundle["renderer_asset_names"],
                bundle["renderer_asset_sha256"],
                strict=True,
            )
        }
        bundle_joint_names = tuple(str(value) for value in bundle["joint_names"])
        joint_state_evidence = str(bundle["joint_state_evidence_class"])
        evidence_authority = joint_state_evidence in {
            str(value)
            for value in config.get(
                "joint_state_evidence_allowlist",
                (
                    "sensor_measurement",
                    "calibrated_geometry",
                    "physics_solver_estimate",
                ),
            )
        }
        named_provenance = all(
            str(bundle[name]).strip()
            for name in (
                "proposal_model_name",
                "proposal_model_revision",
                "renderer_name",
                "renderer_revision",
            )
        )
        thresholds = config["thresholds"]
        contract = ExactAssetTrajectoryContract(
            embodiment_id=str(config["embodiment_id"]),
            camera_frame=str(config["camera_frame"]),
            robot_base_frame=str(config["robot_base_frame"]),
            timeline=str(config["timeline"]),
            source_video_sha256=source_sha256,
            fps=float(manifest["input"]["fps"]),
            joint_names=joint_names,
            joint_limits_rad=joint_limits,
            asset_sha256=renderer_hashes,
            expected_asset_sha256=actual_asset_sha256,
            minimum_visible_keypoints_per_frame=int(
                thresholds["minimum_visible_keypoints_per_frame"]
            ),
            minimum_heldout_frames=int(thresholds["minimum_heldout_frames"]),
            minimum_heldout_groups=int(thresholds["minimum_heldout_groups"]),
            maximum_reprojection_rmse_px_p95=float(
                thresholds["maximum_reprojection_rmse_px_p95"]
            ),
            minimum_silhouette_iou_p05=float(
                thresholds["minimum_silhouette_iou_p05"]
            ),
            maximum_joint_standard_deviation_rad=float(
                thresholds["maximum_joint_standard_deviation_rad"]
            ),
            maximum_base_translation_standard_deviation_m=float(
                thresholds["maximum_base_translation_standard_deviation_m"]
            ),
            minimum_alternative_asset_error_margin_px_p05=float(
                thresholds["minimum_alternative_asset_error_margin_px_p05"]
            ),
            maximum_joint_velocity_rad_s=float(
                thresholds["maximum_joint_velocity_rad_s"]
            ),
        )
        frames = bundle["source_frame_indices"]
        validation = validate_exact_asset_trajectory(
            np,
            contract=contract,
            evidence_source_video_sha256=str(bundle["source_video_sha256"]),
            frame_indices=frames,
            joint_positions_rad=bundle["joint_positions_rad"],
            camera_from_robot_base=bundle["camera_from_robot_base"],
            observed_keypoints_px=bundle["observed_keypoints_px"],
            rendered_keypoints_px=bundle["rendered_keypoints_px"],
            keypoint_confidence=bundle["keypoint_confidence"],
            fit_frame_mask=bundle["fit_frame_mask"],
            heldout_group_ids=tuple(str(value) for value in bundle["heldout_group_ids"]),
            silhouette_iou=bundle["silhouette_iou"],
            joint_standard_deviation_rad=bundle["joint_standard_deviation_rad"],
            base_translation_standard_deviation_m=bundle[
                "base_translation_standard_deviation_m"
            ],
            alternative_asset_reprojection_rmse_px=bundle[
                "alternative_asset_reprojection_rmse_px"
            ],
            joint_velocities_rad_s=(
                bundle["joint_velocities_rad_s"]
                if "joint_velocities_rad_s" in bundle.files
                else None
            ),
        )
        dense_timeline = bool(
            len(frames) == int(manifest["input"]["frames"])
            and np.array_equal(
                frames, np.arange(int(manifest["input"]["frames"]), dtype=np.int64)
            )
        )
        schema_matches = bundle_joint_names == joint_names
        validation["gates"].update(
            {
                "dense_full_source_timeline": dense_timeline,
                "joint_schema_matches_exact_assets": schema_matches,
                "asset_files_match_registry": asset_file_gate,
                "video_file_matches_manifest": source_file_gate,
                "asset_joint_counts_match_registry": count_gate,
                "joint_state_evidence_has_physical_authority": evidence_authority,
                "proposal_and_renderer_provenance_named": named_provenance,
            }
        )
        validation["passed"] = all(validation["gates"].values())
        validation["reasons"] = [
            name for name, passed in validation["gates"].items() if not passed
        ]
        bundle_metadata = {
            "renderer_asset_sha256": renderer_hashes,
            "joint_schema_matches_exact_assets": schema_matches,
            "joint_state_evidence_class": joint_state_evidence,
            "proposal_model": {
                "name": str(bundle["proposal_model_name"]),
                "revision": str(bundle["proposal_model_revision"]),
            },
            "renderer": {
                "name": str(bundle["renderer_name"]),
                "revision": str(bundle["renderer_revision"]),
            },
        }
        if validation["passed"]:
            observed = bundle["observed_keypoints_px"]
            rendered = bundle["rendered_keypoints_px"]
            confidence = bundle["keypoint_confidence"]
            visible = (
                np.all(np.isfinite(observed), axis=2)
                & np.all(np.isfinite(rendered), axis=2)
                & np.isfinite(confidence)
                & (confidence >= 0.2)
            )
            reprojection = np.empty(len(frames), dtype=np.float64)
            for index in range(len(frames)):
                error = rendered[index, visible[index]] - observed[index, visible[index]]
                reprojection[index] = float(
                    np.sqrt(np.mean(np.sum(np.square(error), axis=1)))
                )
            if "joint_velocities_rad_s" in bundle.files:
                velocity = bundle["joint_velocities_rad_s"]
            else:
                velocity = np.zeros_like(bundle["joint_positions_rad"])
                velocity[1:] = (
                    np.diff(bundle["joint_positions_rad"], axis=0)
                    * float(manifest["input"]["fps"])
                )
                velocity[0] = velocity[1]
            np.savez_compressed(
                trajectory_path,
                embodiment_id=np.asarray(config["embodiment_id"]),
                robot_base_frame=np.asarray(config["robot_base_frame"]),
                camera_frame=np.asarray(config["camera_frame"]),
                timeline=np.asarray(config["timeline"]),
                source_video_sha256=np.asarray(source_sha256),
                source_frame_indices=frames,
                joint_names=np.asarray(joint_names),
                joint_limits_rad=np.asarray(joint_limits, dtype=np.float64),
                joint_positions_rad=bundle["joint_positions_rad"],
                joint_velocities_rad_s=velocity,
                camera_from_robot_base=bundle["camera_from_robot_base"],
                reprojection_rmse_px=reprojection,
                trajectory_evidence=np.asarray(joint_state_evidence),
            )

    elapsed = time.perf_counter() - started
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if validation["passed"] else "PARTIAL",
        "passed": bool(validation["passed"]),
        "honest_scope": (
            "WORKING means a dense full-q trajectory passed exact-asset held-out "
            "rendering and observability gates. It does not promote camera, stems, "
            "forces, or final-video quality."
        ),
        "validation": validation,
        "observability_audit": audit,
        "asset_joint_counts": joint_counts,
        "joint_count": len(joint_names),
        "asset_file_sha256": actual_asset_sha256,
        "asset_registry_sha256": registry_asset_sha256,
        "preflight_gates": {
            "asset_files_match_registry": asset_file_gate,
            "asset_joint_counts_match_registry": count_gate,
            "video_file_matches_manifest": source_file_gate,
        },
        "evidence_bundle": bundle_metadata,
        "source_video_sha256": source_sha256,
        "command": [sys.executable, *sys.argv],
        "seed": args.seed,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {"numpy": importlib.metadata.version("numpy")},
        "git": _git_state(),
        "inputs": {
            **{
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in paths.items()
            },
            **{
                name: ({"path": str(path), "sha256": _sha256(path)} if path else None)
                for name, path in optional_paths.items()
            },
        },
        "outputs": {
            "robot_trajectory": (
                {"path": str(trajectory_path), "sha256": _sha256(trajectory_path)}
                if trajectory_path.is_file()
                else None
            )
        },
        "performance": {"compile_and_validate_wall_seconds": elapsed},
        "non_negotiable_rules": config["immutable_rules"],
    }
    report_path = output_dir / "exact-asset-trajectory-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
