#!/usr/bin/env python3
"""Run a complete, fail-closed architecture tournament for contact video.

This is intentionally an architecture mutation loop, not a parameter search.
It evaluates two complete families over two chronological probes, preserves
every failed gate, and emits structural repairs without promoting untested code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.agent.contact_dynamics_evolution import (  # noqa: E402
    ArchitectureAssessment,
    ArchitectureEvolutionContract,
    derive_structural_repairs,
    select_architecture,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATES = (
    "articulated_metric_hand",
    "metric_force_closure",
    "causal_stem_motion",
    "persistent_instance_identity",
    "adversarial_audit",
    "human_high_resolution_review",
)
ARCHITECTURES = (
    "pixel-source-lock-v6",
    "metric-articulated-rod-residual-v1",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--frame-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--human-review",
        choices=("accepted", "rejected", "pending"),
        required=True,
        help="Non-overridable full-resolution review outcome.",
    )
    parser.add_argument("--human-review-source", required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _git(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        ["git", *command],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": ["git", *command],
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _maximum_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def assess_visual_group(
    rows: list[dict[str, object]],
    *,
    hand_motion_floor: float,
    stem_motion_floor: float,
    maximum_response_lag_frames: int,
    maximum_frozen_run_frames: int,
) -> dict[str, object]:
    """Measure response while the visible hand is moving near flower support."""

    ordered = sorted(rows, key=lambda item: int(item["frame"]))
    driven = [
        bool(row["measurement_valid"])
        and bool(row["projected_contact"])
        and float(row["hand_motion_p90"]) > hand_motion_floor
        for row in ordered
    ]
    responded = []
    for index, is_driven in enumerate(driven):
        end = min(len(ordered), index + maximum_response_lag_frames + 1)
        responded.append(
            is_driven
            and any(
                float(ordered[candidate]["flower_motion_p90"]) > stem_motion_floor
                for candidate in range(index, end)
            )
        )
    frozen = [
        is_driven and not did_respond
        for is_driven, did_respond in zip(driven, responded, strict=True)
    ]
    driver_frames = sum(driven)
    frozen_frames = sum(frozen)
    maximum_frozen_run = _maximum_true_run(frozen)
    passed = (
        driver_frames > 0
        and frozen_frames == 0
        and maximum_frozen_run <= maximum_frozen_run_frames
    )
    return {
        "frame_start": int(ordered[0]["frame"]),
        "frame_end": int(ordered[-1]["frame"]),
        "driver_frames": driver_frames,
        "frozen_driver_frames": frozen_frames,
        "maximum_frozen_run_frames": maximum_frozen_run,
        "passed": passed,
        "evidence_scope": "2-D projected-contact visual proxy only",
    }


def _assessment(
    *,
    group_id: str,
    architecture_id: str,
    gates: dict[str, bool],
    evidence_path: Path,
    utility: float,
    cost_units: float,
) -> ArchitectureAssessment:
    return ArchitectureAssessment(
        group_id=group_id,
        architecture_id=architecture_id,
        hard_gates=tuple((name, bool(gates[name])) for name in GATES),
        utility=utility,
        cost_units=cost_units,
        evidence_path=str(evidence_path),
    )


def main() -> int:
    args = _parser().parse_args()
    started = time.perf_counter()
    audit_path = args.audit_report.expanduser().resolve()
    metrics_path = args.frame_metrics.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {output_dir}")
    for path in (audit_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True)
    evidence_dir = output_dir / "group-evidence"
    evidence_dir.mkdir()

    audit = json.loads(audit_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    if not isinstance(audit, dict) or not isinstance(metrics, list) or len(metrics) < 2:
        raise ValueError("audit and frame metrics have invalid schemas")
    video = audit["video"]
    expected_frames = int(video["frames"])
    if len(metrics) != expected_frames:
        raise ValueError("frame metric count does not match audited video")
    split = expected_frames // 2
    group_rows = {
        f"same-scene-frames-0-{split - 1}": metrics[:split],
        f"same-scene-frames-{split}-{expected_frames - 1}": metrics[split:],
    }
    floors = audit["motion_measurement_floors"]
    maximum_frozen = int(audit["config"]["maximum_frozen_run_frames"])
    maximum_response_lag = int(audit["config"]["maximum_response_lag_frames"])
    human_pass = args.human_review == "accepted"
    assessments = []
    group_reports = {}
    for group_id, rows in group_rows.items():
        visual = assess_visual_group(
            rows,
            hand_motion_floor=float(floors["hand_motion"]),
            stem_motion_floor=float(floors["stem_motion"]),
            maximum_response_lag_frames=maximum_response_lag,
            maximum_frozen_run_frames=maximum_frozen,
        )
        group_report = {
            "group_id": group_id,
            "scope": "chronological probe within one real scene; not independent-scene generalization",
            "visual_causal_motion": visual,
            "human_high_resolution_review": {
                "outcome": args.human_review,
                "source": args.human_review_source,
                "passed": human_pass,
            },
        }
        evidence_path = evidence_dir / f"{group_id}.json"
        _write_json(evidence_path, group_report)
        group_reports[group_id] = group_report

        pixel_gates = {
            "articulated_metric_hand": bool(audit["gates"]["articulated_metric_hand"]),
            "metric_force_closure": bool(audit["gates"]["metric_force_closure"]),
            "causal_stem_motion": bool(visual["passed"]),
            "persistent_instance_identity": bool(
                audit["gates"]["instance_identity_available_for_all_interactions"]
            ),
            "adversarial_audit": bool(audit["gates"]["adversarial_audit"]),
            "human_high_resolution_review": human_pass,
        }
        assessments.append(
            _assessment(
                group_id=group_id,
                architecture_id=ARCHITECTURES[0],
                gates=pixel_gates,
                evidence_path=evidence_path,
                utility=1.0 - float(visual["frozen_driver_frames"]) / max(
                    int(visual["driver_frames"]), 1
                ),
                cost_units=float(audit["performance"]["wall_seconds"]),
            )
        )

        # The new state-space family has executable contracts and adversarial
        # CPU tests, but no metric reconstruction/render for this real clip.
        # Missing candidate evidence remains false; design intent is not a pass.
        state_gates = {
            "articulated_metric_hand": False,
            "metric_force_closure": False,
            "causal_stem_motion": False,
            "persistent_instance_identity": False,
            "adversarial_audit": True,
            "human_high_resolution_review": False,
        }
        assessments.append(
            _assessment(
                group_id=group_id,
                architecture_id=ARCHITECTURES[1],
                gates=state_gates,
                evidence_path=evidence_path,
                utility=0.0,
                cost_units=0.0,
            )
        )

    contract = ArchitectureEvolutionContract(
        required_gates=GATES,
        required_groups=tuple(group_rows),
        architecture_ids=ARCHITECTURES,
        maximum_cost_units=60.0,
    )
    selection = select_architecture(assessments, contract)
    repairs = derive_structural_repairs(selection)
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "mode": "complete_architecture_tournament_not_hyperparameter_search",
        "seed": args.seed,
        "contract": {
            "required_gates": list(GATES),
            "required_groups": list(group_rows),
            "architectures": list(ARCHITECTURES),
            "maximum_cost_units": contract.maximum_cost_units,
        },
        "group_independence_limit": (
            "The two probes are chronological halves of one scene. They satisfy complete "
            "same-video coverage but cannot establish cross-scene generalization."
        ),
        "selection": selection,
        "structural_repairs": repairs,
        "next_architecture": ARCHITECTURES[1],
        "next_architecture_status": (
            "implemented state/evaluator core; not promoted because no real metric reconstruction "
            "or rendered candidate exists"
        ),
        "assessments": [
            {
                "group_id": row.group_id,
                "architecture_id": row.architecture_id,
                "hard_gates": dict(row.hard_gates),
                "utility": row.utility,
                "cost_units": row.cost_units,
                "evidence_path": row.evidence_path,
            }
            for row in assessments
        ],
        "inputs": {
            "audit_report": {"path": str(audit_path), "sha256": _sha256(audit_path)},
            "frame_metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
        },
        "command": sys.argv,
        "git": {"head": _git(["rev-parse", "HEAD"]), "status": _git(["status", "--short"])},
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {"pytest": importlib.metadata.version("pytest")},
        "performance": {"wall_seconds": elapsed},
        "limitations": [
            "No calibrated depth or measured/simulated contact force exists for this monocular clip.",
            "No real metric-articulated-rod rendered candidate exists to compare against the pixel baseline.",
            "Two halves of one scene are not two independent scenes or embodiments.",
        ],
    }
    _write_json(output_dir / "evolution-result.json", result)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["output_dir"] = str(output_dir)
    _write_json(output_dir / "config.json", config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
