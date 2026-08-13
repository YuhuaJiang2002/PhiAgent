#!/usr/bin/env python3
"""Jointly optimize persistent flower stems without inventing metric evidence."""

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

from phiagent.perception.multistem_rod_optimizer import (  # noqa: E402
    MultiStemRodContract,
    optimize_multistem_rod_trajectories,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--instance-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--source-video-sha256")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--timeline", default="frame:source_video")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser


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
    result = {}
    for name, command in (
        ("head", ("git", "rev-parse", "HEAD")),
        ("status", ("git", "status", "--short")),
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


def _metric_binding(
    proposal: Any,
    calibration_path: Path | None,
    *,
    source_video_sha256: str,
    coordinate_frame: str,
    geometry_evidence: str,
) -> dict[str, object]:
    if calibration_path is None:
        return {
            "verified": False,
            "independent_groups": 0,
            "checks": {"calibration_supplied": False},
        }
    report = _json(calibration_path)
    calibration = report.get("calibration", {})
    groups = report.get("independent_calibration_groups")
    if groups is None:
        groups = len(calibration.get("independent_group_ids", ()))
    report_world_frame = report.get("world_frame")
    if report_world_frame is None:
        report_world_frame = report.get("config", {}).get("world_frame")
    checks = {
        "calibration_supplied": True,
        "calibration_passed": report.get("passed") is True,
        "source_hash_matches": (
            source_video_sha256
            == str(report.get("source_video_sha256", ""))
        ),
        "alignment_hash_matches": (
            "metric_alignment_report_sha256" in proposal.files
            and str(proposal["metric_alignment_report_sha256"])
            == _sha256(calibration_path)
        ),
        "world_frame_matches": (
            report_world_frame is not None
            and coordinate_frame == str(report_world_frame)
        ),
        "proposal_declares_calibrated_geometry": (
            geometry_evidence == "calibrated_geometry"
        ),
        "independent_groups": int(groups) >= 2,
    }
    return {
        "verified": all(checks.values()),
        "independent_groups": int(groups),
        "checks": checks,
        "report_sha256": _sha256(calibration_path),
    }


def main() -> int:
    args = _parser().parse_args()
    proposal_path = args.proposal.expanduser().resolve()
    instance_path = args.instance_spec.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    calibration_path = (
        args.calibration_report.expanduser().resolve()
        if args.calibration_report
        else None
    )
    for name, path in (
        ("proposal", proposal_path),
        ("instance spec", instance_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing: {path}")
    if calibration_path is not None and not calibration_path.is_file():
        raise FileNotFoundError(
            f"calibration report is missing: {calibration_path}"
        )
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite experiment directory: {output_dir}"
        )
    if args.iterations <= 0:
        raise ValueError("optimization iterations must be positive")
    output_dir.mkdir(parents=True)

    import numpy as np

    np.random.seed(args.seed)
    proposal = np.load(proposal_path, allow_pickle=False)
    required = {
        "confidence",
        "instance_ids",
        "source_frame_indices",
        "coordinate_frame",
    }
    missing = sorted(required - set(proposal.files))
    if missing:
        raise ValueError(f"proposal NPZ lacks required fields: {missing}")
    centerline_key = (
        "centerline_proposals"
        if "centerline_proposals" in proposal.files
        else "centerlines_world_m"
        if "centerlines_world_m" in proposal.files
        else None
    )
    if centerline_key is None:
        raise ValueError(
            "proposal NPZ requires centerline_proposals or centerlines_world_m"
        )
    evidence_key = (
        "geometry_evidence"
        if "geometry_evidence" in proposal.files
        else "evidence_class"
        if "evidence_class" in proposal.files
        else None
    )
    if evidence_key is None:
        raise ValueError(
            "proposal NPZ requires geometry_evidence or evidence_class"
        )
    source_video_sha256 = (
        str(proposal["source_video_sha256"])
        if "source_video_sha256" in proposal.files
        else str(args.source_video_sha256 or "")
    )
    if (
        len(source_video_sha256) != 64
        or any(value not in "0123456789abcdef" for value in source_video_sha256)
    ):
        raise ValueError(
            "proposal lacks source SHA-256; supply --source-video-sha256"
        )
    fps = (
        float(proposal["fps"])
        if "fps" in proposal.files
        else float(args.fps) if args.fps is not None else 0.0
    )
    if fps <= 0:
        raise ValueError("proposal lacks FPS; supply --fps")
    timeline = (
        str(proposal["timeline"])
        if "timeline" in proposal.files
        else str(args.timeline)
    )
    observations = proposal[centerline_key]
    confidence = proposal["confidence"]
    instance_ids = tuple(str(value) for value in proposal["instance_ids"])
    instance_spec = _json(instance_path)
    rows = instance_spec.get("instances")
    if not isinstance(rows, list) or not rows:
        raise ValueError("instance spec requires a nonempty instances list")
    ids_from_spec = tuple(str(row["instance_id"]) for row in rows)
    if ids_from_spec != instance_ids:
        raise ValueError(
            "proposal instance order must exactly match the immutable instance spec"
        )
    root_nodes = tuple(int(row["root_node"]) for row in rows)
    root_modes = tuple(str(row["root_mode"]) for row in rows)
    coordinate_frame = str(proposal["coordinate_frame"])
    geometry_evidence = str(proposal[evidence_key])
    metric = _metric_binding(
        proposal,
        calibration_path,
        source_video_sha256=source_video_sha256,
        coordinate_frame=coordinate_frame,
        geometry_evidence=geometry_evidence,
    )
    contract = MultiStemRodContract(
        instance_ids=instance_ids,
        coordinate_frame=coordinate_frame,
        timeline=timeline,
        fps=fps,
        nodes_per_stem=int(observations.shape[2]),
        root_nodes=root_nodes,
        root_modes=root_modes,
        geometry_evidence=geometry_evidence,
        metric_scale_verified=bool(metric["verified"]),
        independent_calibration_groups=int(metric["independent_groups"]),
    )
    started = time.perf_counter()
    optimized = optimize_multistem_rod_trajectories(
        np,
        observations=observations,
        confidence=confidence,
        proposal_sigma=(
            proposal["position_sigma"]
            if "position_sigma" in proposal.files
            else proposal["position_sigma_m"]
            if "position_sigma_m" in proposal.files
            else None
        ),
        contract=contract,
        iterations=args.iterations,
    )
    elapsed = time.perf_counter() - started
    state_path = output_dir / "multistem-state.npz"
    np.savez_compressed(
        state_path,
        source_frame_indices=proposal["source_frame_indices"],
        source_video_sha256=np.asarray(source_video_sha256),
        instance_ids=np.asarray(instance_ids),
        coordinate_frame=np.asarray(contract.coordinate_frame),
        timeline=np.asarray(contract.timeline),
        fps=np.asarray(contract.fps),
        centerlines=optimized["centerlines"],
        velocity=optimized["velocity"],
        position_covariance=optimized["position_covariance"],
        visible=optimized["visible"],
        reference_segment_lengths=optimized["reference_segment_lengths"],
        geometry_evidence=np.asarray(contract.geometry_evidence),
        metric_scale_verified=np.asarray(contract.metric_scale_verified),
    )
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            "numpy": importlib.metadata.version("numpy"),
        },
        "seed": args.seed,
        "git": _git_state(),
        "inputs": {
            "proposal": {
                "path": str(proposal_path),
                "sha256": _sha256(proposal_path),
            },
            "instance_spec": {
                "path": str(instance_path),
                "sha256": _sha256(instance_path),
            },
            "calibration_report": (
                {
                    "path": str(calibration_path),
                    "sha256": _sha256(calibration_path),
                }
                if calibration_path
                else None
            ),
        },
        "metric_binding": metric,
        "optimization": {
            **optimized["report"],
            "iterations": args.iterations,
            "wall_seconds": elapsed,
        },
        "output": {
            "path": str(state_path),
            "sha256": _sha256(state_path),
            "bytes": state_path.stat().st_size,
        },
        "honest_status": optimized["report"]["status"],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if optimized["report"]["promotion_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
