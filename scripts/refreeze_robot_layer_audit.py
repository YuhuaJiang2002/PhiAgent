#!/usr/bin/env python3
"""Reapply immutable audit limits to an already decoded candidate report."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_robot_layer_long_video import _sha256, _summary  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-audit-report", type=Path, required=True)
    parser.add_argument("--input-frame-metrics", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--frozen-limits-report", type=Path, required=True)
    parser.add_argument("--frozen-limits-candidate", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _candidate(report: dict, name: str) -> dict:
    result = next(
        (row for row in report.get("candidates", []) if row.get("name") == name),
        None,
    )
    if result is None:
        raise ValueError(f"audit report has no candidate {name!r}")
    return result


def main() -> int:
    args = _parser().parse_args()
    import numpy as np

    paths = {
        "input_audit": args.input_audit_report.expanduser().resolve(),
        "input_frame_metrics": args.input_frame_metrics.expanduser().resolve(),
        "frozen_audit": args.frozen_limits_report.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{name}: {path}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)

    started = time.perf_counter()
    report = json.loads(paths["input_audit"].read_text())
    source_candidate = _candidate(report, args.candidate_name)
    rows = json.loads(paths["input_frame_metrics"].read_text())
    config = report["config"]
    expected_frames = int(config["expected_frames"])
    if len(rows) != expected_frames or [row["frame"] for row in rows] != list(
        range(expected_frames)
    ):
        raise ValueError("frame metrics do not match the immutable audit timeline")

    frozen_report = json.loads(paths["frozen_audit"].read_text())
    frozen_candidate = _candidate(frozen_report, args.frozen_limits_candidate)
    frozen_limits = {
        key: float(value)
        for key, value in frozen_candidate["summary"][
            "limits_fit_only_on_anchor_frames"
        ].items()
    }
    namespace = argparse.Namespace(
        anchor_start=int(config["anchor_start"]),
        anchor_end_exclusive=int(config["anchor_end_exclusive"]),
        late_start=int(config["late_start"]),
        allowed_late_violation_fraction=float(
            config["allowed_late_violation_fraction"]
        ),
        required_contact_recall=float(config["required_contact_recall"]),
        persistent_grasp_start=int(config["persistent_grasp_start"]),
        persistent_grasp_end_exclusive=int(
            config["persistent_grasp_end_exclusive"]
        ),
        required_persistent_grasp_recall=float(
            config["required_persistent_grasp_recall"]
        ),
        maximum_source_occlusion_gap=int(
            config["maximum_source_occlusion_gap"]
        ),
        minimum_occlusion_bridge_coverage=float(
            config["minimum_occlusion_bridge_coverage"]
        ),
    )
    refrozen_summary = _summary(
        np,
        rows,
        namespace,
        frozen_limits=frozen_limits,
    )

    frame_output = output / f"{args.candidate_name}-frame-metrics.json"
    shutil.copy2(paths["input_frame_metrics"], frame_output)
    source_candidate["summary"] = refrozen_summary
    source_candidate["frame_metrics_path"] = str(frame_output)
    source_candidate["frame_metrics_sha256"] = _sha256(frame_output)
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["method"]["threshold_fit"] = (
        "reused verbatim from immutable prior audit candidate "
        f"{args.frozen_limits_candidate}; late frames never fit their own thresholds"
    )
    report["config"]["frozen_limits_report"] = str(paths["frozen_audit"])
    report["config"]["frozen_limits_candidate"] = args.frozen_limits_candidate
    report["config"]["refreeze_only"] = True
    report["inputs"]["frozen_limits_source"] = {
        "path": str(paths["frozen_audit"]),
        "sha256": _sha256(paths["frozen_audit"]),
        "candidate": args.frozen_limits_candidate,
        "limits": frozen_limits,
    }
    report["refreeze_provenance"] = {
        "input_audit_report": {
            "path": str(paths["input_audit"]),
            "sha256": _sha256(paths["input_audit"]),
        },
        "input_frame_metrics": {
            "path": str(paths["input_frame_metrics"]),
            "sha256": _sha256(paths["input_frame_metrics"]),
        },
        "candidate_video_sha256_unchanged": source_candidate["sha256"],
        "reused_without_redecode": [
            "candidate video hash",
            "per-frame measurements",
            "adversarial attack measurements",
            "persistent-grasp measurements",
        ],
        "recomputed": ["section summaries", "hard-gate decisions"],
        "wall_seconds": time.perf_counter() - started,
    }
    report["wall_seconds"] = report["refreeze_provenance"]["wall_seconds"]
    report_path = output / "audit-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "candidate_sha256": source_candidate["sha256"],
                "image_space_contract_pass": refrozen_summary[
                    "image_space_contract_pass"
                ],
                "failed_gates": [
                    name
                    for name, passed in refrozen_summary["gates"].items()
                    if not passed
                ],
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
