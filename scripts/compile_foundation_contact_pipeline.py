#!/usr/bin/env python3
"""Compile foundation-model proposals into a fail-closed physical pipeline report."""

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
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.perception.foundation_contact import (  # noqa: E402
    ContactForceContract,
    EvidenceClass,
    MetricCameraContract,
    RobotTrajectoryContract,
    StemCenterlineContract,
    decide_foundation_contact_status,
    model_registry,
    validate_contact_force_sequence,
    validate_metric_camera_sequence,
    validate_robot_trajectory,
    validate_stem_centerlines,
)


KNOWN_ASSET_HASHES = {
    "g1_model": "3c2616550a31f33e84d3c80b8e913ac5618c8888019b0c9490dae93493e647f3",
    "sharpa_left_model": "3cbeb46259d4ba63cbdb83085255d1a8f8031c51e0101a6622f6e7e81a64dc11",
    "sharpa_right_model": "43d9cb63d724889b69574a5e0981aee4a2f30d825c85f3098988e3a7a3bb9980",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    result = {}
    for label, command in (
        ("head", ["git", "rev-parse", "HEAD"]),
        ("status", ["git", "status", "--short"]),
    ):
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
        )
        result[label] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return result


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return raw


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--da3-samples", type=Path)
    parser.add_argument("--da3-manifest", type=Path)
    parser.add_argument("--context-scale-report", type=Path)
    parser.add_argument("--metric-camera-samples", type=Path)
    parser.add_argument("--metric-camera-report", type=Path)
    parser.add_argument("--g1-model", type=Path, required=True)
    parser.add_argument("--sharpa-left-model", type=Path, required=True)
    parser.add_argument("--sharpa-right-model", type=Path, required=True)
    parser.add_argument("--stem-centerlines", type=Path)
    parser.add_argument("--robot-trajectory", type=Path)
    parser.add_argument("--robot-fit-report", type=Path)
    parser.add_argument("--contact-forces", type=Path)
    parser.add_argument("--bundle-manifest", type=Path)
    parser.add_argument("--camera-calibration-report", type=Path)
    parser.add_argument("--calibrated-camera-samples", type=Path)
    parser.add_argument("--model-derived-rgbd-report", type=Path)
    parser.add_argument("--generated-observation-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def _camera_report(
    np: Any,
    samples: Any,
    manifest: dict[str, Any],
    context: dict[str, Any],
    *,
    calibration_report: dict[str, Any] | None = None,
    calibration_report_sha256: str | None = None,
    calibrated_samples: Any | None = None,
    calibrated_samples_sha256: str | None = None,
    original_samples_sha256: str,
) -> dict[str, object]:
    if not context.get("passed"):
        uncertainty = None
    else:
        uncertainty = float(context["summary"]["context_scale_variation_fraction_p95"])
    processed_height = int(manifest["sampling"]["processed_height"])
    processed_width = int(manifest["sampling"]["processed_width"])
    calibration_output = (
        calibration_report.get("outputs", {}).get("calibrated_samples")
        if calibration_report
        else None
    )
    calibration_checks = {
        "report_passed": bool(calibration_report and calibration_report.get("passed")),
        "source_video_hash_matches": bool(
            calibration_report
            and calibration_report.get("source_video_sha256")
            == manifest["input"]["sha256"]
        ),
        "source_da3_samples_hash_matches": bool(
            calibration_report
            and calibration_report.get("inputs", {})
            .get("da3_samples", {})
            .get("sha256")
            == original_samples_sha256
        ),
        "calibrated_samples_supplied": calibrated_samples is not None,
        "calibrated_samples_hash_matches": bool(
            isinstance(calibration_output, dict)
            and calibrated_samples_sha256
            and calibration_output.get("sha256")
            == calibrated_samples_sha256
        ),
    }
    bridge_bound = all(calibration_checks.values())
    selected = calibrated_samples if bridge_bound else samples
    calibration = (
        calibration_report.get("calibration", {}) if calibration_report else {}
    )
    contract = MetricCameraContract(
        camera_frame=(
            str(selected["camera_frame"])
            if bridge_bound and "camera_frame" in selected.files
            else "camera:da3_processed_metric"
        ),
        world_frame=(
            str(selected["world_frame"])
            if bridge_bound and "world_frame" in selected.files
            else "world:da3_learned_metric"
        ),
        timeline="frame:source_video",
        fps=float(manifest["input"]["fps"]),
        image_width=processed_width,
        image_height=processed_height,
        intrinsics_evidence=(
            EvidenceClass.CALIBRATED_GEOMETRY
            if bridge_bound
            else EvidenceClass.FOUNDATION_MODEL_ESTIMATE
        ),
        depth_evidence=(
            EvidenceClass.CALIBRATED_GEOMETRY
            if bridge_bound
            else EvidenceClass.FOUNDATION_MODEL_ESTIMATE
        ),
        metric_scale_source=(
            "sparse independent metric anchors + robust affine inverse-depth bridge"
            if bridge_bound
            else "DA3 Nested metric submodel"
        ),
        learned_context_scale_variation_fraction=uncertainty,
        absolute_scale_standard_deviation_fraction=(
            float(calibration["scale_standard_deviation_fraction"])
            if bridge_bound
            else None
        ),
        independent_calibration_groups=(
            len(calibration.get("independent_group_ids", [])) if bridge_bound else 0
        ),
        calibration_report_sha256=(
            calibration_report_sha256 if bridge_bound else None
        ),
    )
    report = validate_metric_camera_sequence(
        np,
        contract=contract,
        frame_indices=selected["source_frame_indices"],
        intrinsics_px=selected["intrinsics_px"],
        world_from_camera=selected["world_from_camera"],
        depth_m=selected["depth_m"],
        depth_confidence=selected["confidence"],
    )
    proposal_samples_hash_bound = (
        manifest.get("outputs", {}).get("samples", {}).get("sha256")
        == original_samples_sha256
    )
    report["gates"]["da3_samples_hash_matches_manifest"] = (
        proposal_samples_hash_bound
    )
    report["proposal_passed"] = bool(report["proposal_passed"]) and (
        proposal_samples_hash_bound
    )
    report["passed"] = all(report["gates"].values())
    report["evidence_class"] = (
        "calibrated_geometry" if bridge_bound else "foundation_model_estimate"
    )
    report["scale_scope"] = (
        "independent sparse metric anchors calibrated absolute scale"
        if bridge_bound
        else "context sensitivity bounded; absolute common-mode scale bias not calibrated"
    )
    report["calibration_bridge"] = {
        "requested": calibration_report is not None or calibrated_samples is not None,
        "bound": bridge_bound,
        "checks": calibration_checks,
        "report_status": (
            calibration_report.get("status") if calibration_report else None
        ),
    }
    return report


def _direct_camera_report(
    np: Any,
    samples: Any,
    report: dict[str, Any],
    *,
    report_sha256: str,
    samples_sha256: str,
) -> dict[str, object]:
    """Validate calibrated or sensor RGB-D without routing it through DA3."""

    evidence = EvidenceClass(str(report["depth_evidence"]))
    checks = {
        "report_passed": report.get("passed") is True,
        "samples_hash_matches": report.get("samples_sha256") == samples_sha256,
        "source_video_hash_named": len(str(report.get("source_video_sha256", ""))) == 64,
        "source_video_hash_matches_samples": (
            "source_video_sha256" in samples.files
            and str(samples["source_video_sha256"])
            == str(report.get("source_video_sha256", ""))
        ),
        "bundle_id_matches_samples": (
            "bundle_id" in samples.files
            and str(samples["bundle_id"]) == str(report.get("bundle_id", ""))
            and bool(str(report.get("bundle_id", "")).strip())
        ),
        "fps_matches_samples": (
            "fps" in samples.files
            and abs(float(samples["fps"]) - float(report["fps"])) <= 1e-9
        ),
        "coordinate_frames_match": (
            str(samples["camera_frame"]) == str(report["camera_frame"])
            and str(samples["world_frame"]) == str(report["world_frame"])
            and str(samples["timeline"]) == str(report["timeline"])
        ),
    }
    contract = MetricCameraContract(
        camera_frame=str(report["camera_frame"]),
        world_frame=str(report["world_frame"]),
        timeline=str(report["timeline"]),
        fps=float(report["fps"]),
        image_width=int(report["image_width"]),
        image_height=int(report["image_height"]),
        intrinsics_evidence=EvidenceClass(str(report["intrinsics_evidence"])),
        depth_evidence=evidence,
        metric_scale_source=str(report["metric_scale_source"]),
        absolute_scale_standard_deviation_fraction=(
            float(report["absolute_scale_standard_deviation_fraction"])
            if report.get("absolute_scale_standard_deviation_fraction") is not None
            else None
        ),
        independent_calibration_groups=int(report.get("independent_calibration_groups", 0)),
        calibration_report_sha256=report_sha256,
    )
    result = validate_metric_camera_sequence(
        np,
        contract=contract,
        frame_indices=samples["source_frame_indices"],
        intrinsics_px=samples["intrinsics_px"],
        world_from_camera=samples["world_from_camera"],
        depth_m=samples["depth_m"],
        depth_confidence=samples["confidence"],
    )
    result["gates"]["direct_artifact_provenance_bound"] = all(checks.values())
    result["passed"] = all(result["gates"].values())
    result["evidence_class"] = evidence.value
    result["scale_scope"] = str(report.get("scale_scope", "direct metric RGB-D"))
    result["direct_metric_camera"] = {
        "bound": all(checks.values()),
        "checks": checks,
        "report_sha256": report_sha256,
        "samples_sha256": samples_sha256,
    }
    return result


def _model_derived_rgbd_diagnostic(
    report: dict[str, Any],
    *,
    report_sha256: str,
    source_video_sha256: str,
) -> dict[str, object]:
    """Bind model RGB-D utility evidence without upgrading camera authority."""

    audit = report.get("audit", {})
    outputs = report.get("outputs", {})
    output_hashes_bound = bool(outputs) and all(
        isinstance(row, dict)
        and Path(str(row.get("path", ""))).expanduser().is_file()
        and _sha256(Path(str(row["path"])).expanduser().resolve()) == row.get("sha256")
        for row in outputs.values()
    )
    checks = {
        "report_proposal_passed": report.get("passed") is True,
        "source_video_hash_matches": (
            report.get("source_video_sha256") == source_video_sha256
        ),
        "evidence_is_foundation_model": (
            report.get("evidence_class") == "foundation_model_estimate"
        ),
        "physical_calibration_remains_false": (
            report.get("physical_calibration_passed") is False
        ),
        "zero_independent_physical_groups": (
            audit.get("independent_physical_groups") == 0
        ),
        "same_video_reason_preserved": (
            audit.get("reason")
            == "model_derived_same_video_not_independent_calibration"
        ),
        "output_artifact_hashes_bound": output_hashes_bound,
    }
    return {
        "bound": all(checks.values()),
        "checks": checks,
        "report_sha256": report_sha256,
        "proposal_passed": audit.get("proposal_passed") is True,
        "physical_calibration_passed": False,
        "independent_physical_groups": 0,
        "metrics": audit.get("metrics"),
        "reason": "model_derived_same_video_not_independent_calibration",
    }


def _generated_observation_diagnostic(
    report: dict[str, Any],
    *,
    report_sha256: str,
    source_video_sha256: str,
) -> dict[str, object]:
    """Bind VLM/generative hypotheses without adding a physical stage pass."""

    audit = report.get("audit", {})
    inputs = report.get("inputs", [])
    input_hashes_bound = bool(inputs) and all(
        isinstance(row, dict)
        and Path(str(row.get("path", ""))).expanduser().is_file()
        and _sha256(Path(str(row["path"])).expanduser().resolve()) == row.get("sha256")
        for row in inputs
    )
    checks = {
        "ensemble_audit_passed": audit.get("passed") is True,
        "source_video_hash_matches": (
            audit.get("source_video_sha256") == source_video_sha256
        ),
        "evidence_is_foundation_model": (
            audit.get("evidence_class") == "foundation_model_estimate"
        ),
        "physical_gate_eligibility_remains_false": (
            audit.get("physical_gate_eligible") is False
        ),
        "zero_independent_physical_groups": (
            audit.get("independent_physical_groups") == 0
        ),
        "at_least_two_models": len(audit.get("models", ())) >= 2,
        "input_artifact_hashes_bound": input_hashes_bound,
    }
    return {
        "bound": all(checks.values()),
        "checks": checks,
        "report_sha256": report_sha256,
        "models": audit.get("models", []),
        "common_frames": audit.get("common_frames", 0),
        "categorical_agreement_fraction": audit.get(
            "categorical_agreement_fraction"
        ),
        "review_frames": audit.get("review_frames", []),
        "physical_gate_eligible": False,
        "independent_physical_groups": 0,
        "scope": "failure mining and active capture targeting only",
    }


def _robot_report(
    np: Any,
    path: Path | None,
    assets: dict[str, dict[str, Any]],
    fps: float,
    *,
    fit_report: dict[str, Any] | None = None,
) -> dict[str, object]:
    fit_validation = fit_report.get("validation", {}) if fit_report else None
    if path is None:
        return {
            "passed": False,
            "reasons": ["missing_full_generalized_coordinate_trajectory"]
            + (
                [str(value) for value in fit_validation.get("reasons", ())]
                if isinstance(fit_validation, dict)
                else []
            ),
            "exact_asset_registry_passed": all(row["hash_matches_registry"] for row in assets.values()),
            "note": "Wrist targets or pixels are not a complete URDF joint trajectory.",
            "analysis_by_synthesis": fit_validation,
        }
    payload = np.load(path, allow_pickle=False)
    contract = RobotTrajectoryContract(
        embodiment_id=str(payload["embodiment_id"]),
        robot_base_frame=str(payload["robot_base_frame"]),
        timeline=str(payload["timeline"]),
        fps=fps,
        joint_names=tuple(str(value) for value in payload["joint_names"]),
        joint_limits_rad=tuple(
            (float(value[0]), float(value[1])) for value in payload["joint_limits_rad"]
        ),
        asset_sha256={name: str(row["sha256"]) for name, row in assets.items()},
        trajectory_evidence=EvidenceClass(str(payload["trajectory_evidence"])),
    )
    result = validate_robot_trajectory(
        np,
        contract=contract,
        frame_indices=payload["source_frame_indices"],
        joint_positions_rad=payload["joint_positions_rad"],
        joint_velocities_rad_s=(
            payload["joint_velocities_rad_s"]
            if "joint_velocities_rad_s" in payload.files
            else None
        ),
        reprojection_rmse_px=(
            payload["reprojection_rmse_px"]
            if "reprojection_rmse_px" in payload.files
            else None
        ),
    )
    result["gates"]["exact_asset_registry_passed"] = bool(assets) and all(
        bool(row.get("hash_matches_registry")) for row in assets.values()
    )
    result["passed"] = all(result["gates"].values())
    if fit_report is not None:
        declared_output = fit_report.get("outputs", {}).get("robot_trajectory")
        report_bound = bool(
            isinstance(declared_output, dict)
            and declared_output.get("sha256") == _sha256(path)
            and fit_report.get("passed") is True
            and isinstance(fit_validation, dict)
            and fit_validation.get("passed") is True
        )
        result["gates"]["exact_asset_analysis_by_synthesis_bound"] = report_bound
        result["passed"] = all(result["gates"].values())
        result["analysis_by_synthesis"] = fit_validation
    return result


def _stem_report(np: Any, path: Path | None) -> dict[str, object]:
    if path is None:
        return {
            "passed": False,
            "reasons": ["missing_persistent_metric_per_stem_centerlines"],
        }
    payload = np.load(path, allow_pickle=False)
    contract = StemCenterlineContract(
        instance_ids=tuple(str(value) for value in payload["instance_ids"]),
        coordinate_frame=str(payload["coordinate_frame"]),
        timeline="frame:source_video",
        nodes_per_stem=int(payload["centerlines_world_m"].shape[2]),
        geometry_evidence=EvidenceClass(str(payload["evidence_class"])),
    )
    return validate_stem_centerlines(
        np,
        contract=contract,
        frame_indices=payload["source_frame_indices"],
        centerlines_m=payload["centerlines_world_m"],
        confidence=payload["confidence"],
    )


def _force_report(np: Any, path: Path | None) -> dict[str, object]:
    if path is None:
        return {
            "passed": False,
            "reasons": ["missing_sensor_or_physics_solver_contact_forces"],
            "note": "Visual depth/feature confidence is forbidden as force evidence.",
        }
    payload = np.load(path, allow_pickle=False)
    contract = ContactForceContract(
        coordinate_frame=str(payload["coordinate_frame"]),
        timeline=str(payload["timeline"]),
        instance_ids=tuple(str(value) for value in payload["instance_ids"]),
        force_evidence=EvidenceClass(str(payload["force_evidence"])),
        source_name=str(payload["source_name"]),
    )
    return validate_contact_force_sequence(
        np,
        contract=contract,
        forces_n=payload["forces_n"],
        solver_residual_n=payload["solver_residual_n"],
        covariance_n2=(payload["covariance_n2"] if "covariance_n2" in payload.files else None),
    )


def _bundle_lineage_report(
    np: Any,
    *,
    camera_samples: Any,
    camera_report: dict[str, Any],
    camera_samples_path: Path,
    camera_report_path: Path,
    robot_path: Path | None,
    stem_path: Path | None,
    force_path: Path | None,
    bundle_manifest_path: Path | None,
) -> dict[str, object]:
    required_paths = {
        "robot_trajectory": robot_path,
        "stem_centerlines": stem_path,
        "contact_forces": force_path,
        "bundle_manifest": bundle_manifest_path,
    }
    missing = [name for name, path in required_paths.items() if path is None]
    if missing:
        return {
            "passed": False,
            "gates": {"complete_bundle_inputs": False},
            "reasons": [f"missing_{name}" for name in missing],
        }
    robot = np.load(robot_path, allow_pickle=False)
    stems = np.load(stem_path, allow_pickle=False)
    forces = np.load(force_path, allow_pickle=False)
    manifest = _load_json(bundle_manifest_path)

    payloads = {
        "camera": camera_samples,
        "robot": robot,
        "stems": stems,
        "forces": forces,
    }
    required_fields = {
        "camera": ("bundle_id", "source_video_sha256", "source_frame_indices", "fps", "timeline"),
        "robot": ("bundle_id", "source_video_sha256", "source_frame_indices", "fps", "timeline"),
        "stems": ("bundle_id", "source_video_sha256", "source_frame_indices", "fps", "timeline"),
        "forces": ("bundle_id", "source_video_sha256", "source_frame_indices", "fps", "timeline"),
    }
    complete_fields = all(
        all(field in payload.files for field in required_fields[name])
        for name, payload in payloads.items()
    )
    if not complete_fields:
        return {
            "passed": False,
            "gates": {
                "complete_bundle_inputs": True,
                "complete_lineage_fields": False,
            },
            "reasons": ["missing_bundle_lineage_fields"],
        }

    bundle_ids = {str(payload["bundle_id"]) for payload in payloads.values()}
    source_hashes = {
        str(payload["source_video_sha256"]) for payload in payloads.values()
    }
    timelines = {str(payload["timeline"]) for payload in payloads.values()}
    fps_values = [float(payload["fps"]) for payload in payloads.values()]
    base_frames = {
        str(camera_samples["world_frame"]),
        str(robot["robot_base_frame"]),
        str(stems["coordinate_frame"]),
        str(forces["coordinate_frame"]),
    }
    frame_indices_match = all(
        np.array_equal(
            camera_samples["source_frame_indices"],
            payload["source_frame_indices"],
        )
        for payload in (robot, stems, forces)
    )
    instance_ids_match = np.array_equal(
        stems["instance_ids"],
        forces["instance_ids"],
    )
    manifest_artifacts = manifest.get("artifacts", {})
    artifact_paths = {
        "metric_camera_samples": camera_samples_path,
        "metric_camera_report": camera_report_path,
        "robot_trajectory": robot_path,
        "stem_centerlines": stem_path,
        "contact_forces": force_path,
    }
    manifest_hashes_match = all(
        isinstance(manifest_artifacts.get(name), dict)
        and manifest_artifacts[name].get("sha256") == _sha256(path)
        for name, path in artifact_paths.items()
    )
    manifest_identity_matches = (
        len(bundle_ids) == 1
        and manifest.get("bundle_id") in bundle_ids
        and len(source_hashes) == 1
        and manifest.get("source_video_sha256") in source_hashes
        and camera_report.get("bundle_id") in bundle_ids
        and camera_report.get("source_video_sha256") in source_hashes
    )
    gates = {
        "complete_bundle_inputs": True,
        "complete_lineage_fields": complete_fields,
        "common_bundle_id": len(bundle_ids) == 1,
        "common_source_video_sha256": len(source_hashes) == 1,
        "common_timeline": len(timelines) == 1,
        "common_fps": max(fps_values) - min(fps_values) <= 1e-9,
        "identical_source_frame_indices": frame_indices_match,
        "common_metric_robot_base_frame": len(base_frames) == 1,
        "stem_force_instance_ids_match": instance_ids_match,
        "bundle_manifest_identity_matches": manifest_identity_matches,
        "bundle_manifest_artifact_hashes_match": manifest_hashes_match,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "bundle_ids": sorted(bundle_ids),
        "source_video_sha256": sorted(source_hashes),
        "timelines": sorted(timelines),
        "fps": fps_values,
        "coordinate_frames": sorted(base_frames),
        "frames": len(camera_samples["source_frame_indices"]),
        "instance_ids": [str(value) for value in stems["instance_ids"]],
        "bundle_manifest_sha256": _sha256(bundle_manifest_path),
    }


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite experiment directory: {output_dir}")
    output_dir.mkdir(parents=True)
    assets_required = {
        "g1_model": args.g1_model.expanduser().resolve(),
        "sharpa_left_model": args.sharpa_left_model.expanduser().resolve(),
        "sharpa_right_model": args.sharpa_right_model.expanduser().resolve(),
    }
    foundation_arguments = {
        "da3_samples": args.da3_samples,
        "da3_manifest": args.da3_manifest,
        "context_scale_report": args.context_scale_report,
    }
    direct_arguments = {
        "metric_camera_samples": args.metric_camera_samples,
        "metric_camera_report": args.metric_camera_report,
    }
    foundation_requested = any(value is not None for value in foundation_arguments.values())
    direct_requested = any(value is not None for value in direct_arguments.values())
    if foundation_requested == direct_requested:
        raise ValueError(
            "supply exactly one complete camera bundle: DA3 proposal inputs or direct metric camera inputs"
        )
    selected_camera_arguments = (
        direct_arguments if direct_requested else foundation_arguments
    )
    missing_camera_arguments = [
        name for name, value in selected_camera_arguments.items() if value is None
    ]
    if missing_camera_arguments:
        raise ValueError(
            f"selected camera bundle is incomplete: {missing_camera_arguments}"
        )
    required = {
        **assets_required,
        **{
            name: value.expanduser().resolve()
            for name, value in selected_camera_arguments.items()
            if value is not None
        },
    }
    if any(not path.is_file() for path in required.values()):
        missing = [name for name, path in required.items() if not path.is_file()]
        raise ValueError(f"required pipeline artifacts are missing: {missing}")
    optional = {
        "stem_centerlines": args.stem_centerlines.expanduser().resolve()
        if args.stem_centerlines
        else None,
        "robot_trajectory": args.robot_trajectory.expanduser().resolve()
        if args.robot_trajectory
        else None,
        "robot_fit_report": args.robot_fit_report.expanduser().resolve()
        if args.robot_fit_report
        else None,
        "contact_forces": args.contact_forces.expanduser().resolve()
        if args.contact_forces
        else None,
        "bundle_manifest": args.bundle_manifest.expanduser().resolve()
        if args.bundle_manifest
        else None,
        "camera_calibration_report": args.camera_calibration_report.expanduser().resolve()
        if args.camera_calibration_report
        else None,
        "calibrated_camera_samples": args.calibrated_camera_samples.expanduser().resolve()
        if args.calibrated_camera_samples
        else None,
        "model_derived_rgbd_report": args.model_derived_rgbd_report.expanduser().resolve()
        if args.model_derived_rgbd_report
        else None,
        "generated_observation_audit": args.generated_observation_audit.expanduser().resolve()
        if args.generated_observation_audit
        else None,
    }
    if any(path is not None and not path.is_file() for path in optional.values()):
        missing = [name for name, path in optional.items() if path is not None and not path.is_file()]
        raise ValueError(f"optional pipeline artifact path does not exist: {missing}")

    import numpy as np

    started = time.perf_counter()
    if direct_requested:
        direct_report_path = required["metric_camera_report"]
        direct_samples_path = required["metric_camera_samples"]
        direct_report = _load_json(direct_report_path)
        direct_samples = np.load(direct_samples_path, allow_pickle=False)
        da3_manifest = None
        context_report = None
        samples = None
        fps = float(direct_report["fps"])
    else:
        da3_manifest = _load_json(required["da3_manifest"])
        context_report = _load_json(required["context_scale_report"])
        samples = np.load(required["da3_samples"], allow_pickle=False)
        direct_report = None
        direct_samples = None
        fps = float(da3_manifest["input"]["fps"])
    calibration_report_path = optional["camera_calibration_report"]
    calibrated_samples_path = optional["calibrated_camera_samples"]
    calibration_report = (
        _load_json(calibration_report_path) if calibration_report_path else None
    )
    calibrated_samples = (
        np.load(calibrated_samples_path, allow_pickle=False)
        if calibrated_samples_path
        else None
    )
    assets = {}
    for name in ("g1_model", "sharpa_left_model", "sharpa_right_model"):
        path = required[name]
        digest = _sha256(path)
        assets[name] = {
            "path": str(path),
            "sha256": digest,
            "expected_sha256": KNOWN_ASSET_HASHES[name],
            "hash_matches_registry": digest == KNOWN_ASSET_HASHES[name],
            "evidence_class": "exact_asset",
        }
    if direct_requested:
        camera = _direct_camera_report(
            np,
            direct_samples,
            direct_report,
            report_sha256=_sha256(required["metric_camera_report"]),
            samples_sha256=_sha256(required["metric_camera_samples"]),
        )
    else:
        camera = _camera_report(
            np,
            samples,
            da3_manifest,
            context_report,
            calibration_report=calibration_report,
            calibration_report_sha256=(
                _sha256(calibration_report_path) if calibration_report_path else None
            ),
            calibrated_samples=calibrated_samples,
            calibrated_samples_sha256=(
                _sha256(calibrated_samples_path) if calibrated_samples_path else None
            ),
            original_samples_sha256=_sha256(required["da3_samples"]),
        )
    model_rgbd_report_path = optional["model_derived_rgbd_report"]
    if model_rgbd_report_path is not None:
        camera["model_derived_rgbd"] = _model_derived_rgbd_diagnostic(
            _load_json(model_rgbd_report_path),
            report_sha256=_sha256(model_rgbd_report_path),
            source_video_sha256=(
                str(direct_report["source_video_sha256"])
                if direct_requested
                else str(da3_manifest["input"]["sha256"])
            ),
        )
    generated_observation_path = optional["generated_observation_audit"]
    diagnostics = {}
    if generated_observation_path is not None:
        diagnostics["generated_observation_ensemble"] = (
            _generated_observation_diagnostic(
                _load_json(generated_observation_path),
                report_sha256=_sha256(generated_observation_path),
                source_video_sha256=(
                    str(direct_report["source_video_sha256"])
                    if direct_requested
                    else str(da3_manifest["input"]["sha256"])
                ),
            )
        )
    robot_fit_report_path = optional["robot_fit_report"]
    robot = _robot_report(
        np,
        optional["robot_trajectory"],
        assets,
        fps,
        fit_report=(
            _load_json(robot_fit_report_path) if robot_fit_report_path else None
        ),
    )
    stems = _stem_report(np, optional["stem_centerlines"])
    forces = _force_report(np, optional["contact_forces"])
    stages = {
        "metric_camera": camera,
        "robot_trajectory": robot,
        "stem_centerlines": stems,
        "contact_forces": forces,
    }
    decision = decide_foundation_contact_status(stages)
    if direct_requested:
        lineage_camera_samples = direct_samples
        lineage_camera_report = direct_report
        lineage_camera_samples_path = required["metric_camera_samples"]
        lineage_camera_report_path = required["metric_camera_report"]
    elif camera.get("calibration_bridge", {}).get("bound"):
        if (
            calibrated_samples is None
            or calibrated_samples_path is None
            or calibration_report is None
            or calibration_report_path is None
        ):
            raise RuntimeError(
                "bound camera calibration lacks its selected samples or report"
            )
        lineage_camera_samples = calibrated_samples
        lineage_camera_report = calibration_report
        lineage_camera_samples_path = calibrated_samples_path
        lineage_camera_report_path = calibration_report_path
    else:
        lineage_camera_samples = samples
        lineage_camera_report = da3_manifest
        lineage_camera_samples_path = required["da3_samples"]
        lineage_camera_report_path = required["da3_manifest"]
    lineage = _bundle_lineage_report(
        np,
        camera_samples=lineage_camera_samples,
        camera_report=lineage_camera_report,
        camera_samples_path=lineage_camera_samples_path,
        camera_report_path=lineage_camera_report_path,
        robot_path=optional["robot_trajectory"],
        stem_path=optional["stem_centerlines"],
        force_path=optional["contact_forces"],
        bundle_manifest_path=optional["bundle_manifest"],
    )
    decision["gates"]["bundle_lineage"] = bool(lineage["passed"])
    if not lineage["passed"]:
        decision["status"] = "PARTIAL"
        decision["missing_or_rejected_stages"].append("bundle_lineage")
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **decision,
        "honest_scope": (
            (
                "Direct calibrated or simulated RGB-D, exact assets, and physical validators "
                "decide acceptance. No learned camera proposal is used."
            )
            if direct_requested
            else (
                "Foundation models propose metric/dynamic observations; exact assets and physical "
                "validators decide acceptance. No missing stage is imputed."
            )
        ),
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy",)
        },
        "seed": args.seed,
        "git": _git_state(),
        "models": (
            [] if direct_requested else [row.to_dict() for row in model_registry()]
        ),
        "robot_assets": assets,
        "inputs": {
            **{
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in required.items()
            },
            **{
                name: ({"path": str(path), "sha256": _sha256(path)} if path else None)
                for name, path in optional.items()
            },
        },
        "stages": stages,
        "bundle_lineage": lineage,
        "diagnostics": diagnostics,
        "performance": {"compile_and_validate_wall_seconds": elapsed},
        "non_negotiable_rules": [
            "A foundation-model estimate cannot be relabeled as a sensor measurement.",
            "A same-video virtual view has exact constructed extrinsics but zero independent physical calibration groups.",
            "Robot identity proposals select a hashed exact asset; they never synthesize a replacement URDF when an exact asset exists.",
            "Force evidence must be tactile/force-torque sensing or a physics solver residual with propagated uncertainty.",
            "WORKING requires metric camera, full q trajectory, persistent metric centerlines, and forces to all pass.",
            "WORKING also requires one hash-bound producer bundle with identical source, frames, FPS, timeline, coordinate frame, and stem IDs.",
        ],
    }
    path = output_dir / "pipeline-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if decision["status"] == "WORKING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
