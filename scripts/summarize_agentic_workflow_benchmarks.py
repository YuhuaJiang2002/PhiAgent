#!/usr/bin/env python3
"""Compile matched agentic-workflow benchmarks from retained evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_TRACE = PROJECT_ROOT / (
    "outputs/robot-person-replacement/agentic-runs/20260809T145711Z-5917329d/trace.json"
)
DEFAULT_POSE_RIG = PROJECT_ROOT / (
    "outputs/minimax-h3-flower-validation/"
    "20260810T021100Z-nf4-ref2va-r3/control-pose-rig-v7-v2/evolution.json"
)
DEFAULT_H3 = PROJECT_ROOT / (
    "outputs/minimax-h3-flower-validation/"
    "20260810T021100Z-nf4-ref2va-r3/h3-evaluation-v2/evolution.json"
)
DEFAULT_STRUCTURE_TRACE = PROJECT_ROOT / (
    "outputs/robot-person-replacement/pose-rig-runs/"
    "20260810T160000Z-smoothed-holding-bouquet-v19-accepted/trace.json"
)
DEFAULT_ACWM = PROJECT_ROOT / "demo/showcase/oscar-acwm-evaluation.json"
DEFAULT_REPAIR_POLICY = PROJECT_ROOT / (
    "outputs/flower-repair-policy/20260811T034347Z-636354ef/metrics.json"
)
DEFAULT_LEDGER = PROJECT_ROOT / "experiences/ledger.jsonl"

ROBOT_METRICS = (
    "background_lock",
    "object_lock",
    "subject_replacement",
    "robot_identity",
    "motion_preservation",
    "temporal_consistency",
    "epl_minimum",
    "mean_score",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"benchmark evidence is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"benchmark evidence must be one JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_rows(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metrics: tuple[str, ...],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name in metrics:
        before = float(baseline[name])
        after = float(candidate[name])
        delta = after - before
        rows.append(
            {
                "metric": name,
                "baseline": before,
                "agentic": after,
                "absolute_gain": delta,
                "relative_gain": delta / before if before != 0 else float("nan"),
            }
        )
    return rows


def _ledger_record(ledger: Path, record_id: str) -> dict[str, Any]:
    for line_number, line in enumerate(ledger.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid ledger JSON at {ledger}:{line_number}") from error
        if value.get("record_id") == record_id:
            return value
    raise ValueError(f"required ledger evidence is missing: {record_id}")


def build_report(
    *,
    robot_trace_path: Path,
    pose_rig_path: Path,
    h3_path: Path,
    structure_trace_path: Path,
    acwm_path: Path,
    repair_policy_path: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    robot_trace = _load_json(robot_trace_path)
    pose_rig = _load_json(pose_rig_path)
    h3 = _load_json(h3_path)
    structure_trace = _load_json(structure_trace_path)
    acwm = _load_json(acwm_path)
    repair_policy = _load_json(repair_policy_path)
    h3_rejection = _ledger_record(ledger_path, "2026-08-10.acwm-bowl-minimax-h3-user-rejected")

    baseline_round = robot_trace["rounds"][0]
    best_round_index = int(robot_trace["best_round"])
    best_round = next(row for row in robot_trace["rounds"] if int(row["round"]) == best_round_index)
    thresholds = {key: float(value) for key, value in robot_trace["thresholds"].items()}
    baseline_scores = baseline_round["scorecard"]
    best_scores = best_round["scorecard"]

    baseline_gate_count = sum(
        float(baseline_scores[name]) >= threshold for name, threshold in thresholds.items()
    )
    agentic_gate_count = sum(
        float(best_scores[name]) >= threshold for name, threshold in thresholds.items()
    )

    pose_scores = pose_rig["best_scorecard"]
    h3_scores = h3["best_scorecard"]
    short_window_metrics = (
        "motion_preservation",
        "epl_minimum",
        "temporal_consistency",
        "mean_score",
    )

    structure_acceptance = structure_trace["acceptance"]
    structure_gate_count = len(structure_acceptance)
    if not all(value is True for value in structure_acceptance.values()):
        raise ValueError("the declared v19 structure fallback no longer passes every gate")

    variants = {row["case_id"]: row for row in acwm["variants"]}
    original_oscar_successes = sum(row["status"] == "ACCEPTED" for row in variants.values())
    evolved_cases = [variants["lift-up"], acwm["articulated_carry"]]
    evolved_successes = sum(row["status"] == "ACCEPTED" for row in evolved_cases)
    numeric_metrics = (
        "action_adherence",
        "embodiment_consistency",
        "object_interaction",
        "temporal_consistency",
        "background_consistency",
    )
    accepted_scores = {
        row.get("case_id", "carry-right-lift-arc"): {
            key: row["scores"][key] for key in numeric_metrics
        }
        for row in evolved_cases
    }

    source_paths = (
        robot_trace_path,
        pose_rig_path,
        h3_path,
        structure_trace_path,
        acwm_path,
        repair_policy_path,
    )
    sources = [
        {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(path),
        }
        for path in source_paths
    ]

    return {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "retrospective aggregation of retained matched evidence",
        "robot_embodiment_replacement": {
            "single_pass_vs_agentic_safe_selection": {
                "source_frames": 660,
                "baseline_round": int(baseline_round["round"]),
                "agentic_round": best_round_index,
                "metrics": _metric_rows(baseline_scores, best_scores, ROBOT_METRICS),
                "baseline_hard_gates_passed": baseline_gate_count,
                "agentic_hard_gates_passed": agentic_gate_count,
                "hard_gates_total": len(thresholds),
                "accepted": bool(robot_trace["acceptance"]["thresholds_passed"]),
            },
            "pose_rig_vs_h3_agentic_short_window": {
                "source_frames": 124,
                "duration_seconds": 124 / 24,
                "metrics": _metric_rows(pose_scores, h3_scores, short_window_metrics),
                "accepted": False,
                "failed_gates": {
                    "robot_identity": {
                        "measured": float(h3_scores["robot_identity"]),
                        "required": 0.72,
                    },
                    "motion_preservation": {
                        "measured": float(h3_scores["motion_preservation"]),
                        "required": 0.72,
                    },
                    "epl_minimum": {
                        "measured": float(h3_scores["epl_minimum"]),
                        "required": 0.62,
                    },
                },
            },
            "current_fail_closed_2d_delivery": {
                "source_frames": int(structure_trace["measurement"]["decoded_frames"]),
                "hard_gates_passed": structure_gate_count,
                "hard_gates_total": structure_gate_count,
                "honest_status": structure_trace["honest_status"],
                "scope_limit": "deterministic 2-D structure fallback; not learned, physical, or real-robot execution",
            },
        },
        "acwm": {
            "human_gated_success": [
                {
                    "method": "MiniMax-H3 negative baseline",
                    "accepted": 0,
                    "total": 3,
                    "rate": 0.0,
                    "evidence_record": h3_rejection["record_id"],
                },
                {
                    "method": "single-pass OSCAR batch after posthoc review",
                    "accepted": original_oscar_successes,
                    "total": 3,
                    "rate": original_oscar_successes / 3,
                },
                {
                    "method": "agentic OSCAR condition evolution",
                    "accepted": evolved_successes,
                    "total": 3,
                    "rate": evolved_successes / 3,
                },
            ],
            "absolute_gain_vs_h3_baseline": evolved_successes / 3,
            "absolute_gain_vs_single_pass_oscar": (evolved_successes - original_oscar_successes)
            / 3,
            "accepted_case_scores": accepted_scores,
            "numeric_gate": float(acwm["acceptance"]["required_threshold_per_numeric_gate"]),
            "repair_router": {
                "held_action_selection_accuracy": repair_policy["held_action_selection_accuracy"],
                "guarded_non_regression_rate": repair_policy["guarded_non_regression_rate"],
                "mean_gain_over_raw": repair_policy["mean_gain_over_raw"],
                "candidate_evaluations_before": repair_policy[
                    "full_schedule_candidate_evaluations"
                ],
                "candidate_evaluations_after": repair_policy[
                    "mean_learned_route_candidate_evaluations"
                ],
                "candidate_evaluation_reduction_fraction": repair_policy[
                    "candidate_evaluation_reduction_fraction"
                ],
                "selection_cases": repair_policy["selection_cases"],
            },
            "scope_limit": "one real scene and three selected image-space action types; not robot-base control or physical execution",
        },
        "reporting_rules": [
            "Absolute score differences are reported as percentage-point gains on [0,1] metrics.",
            "Scores from different evaluators are not averaged together.",
            "Human review can veto automatic proxy scores.",
            "The sample counts are too small for a generalization or statistical-significance claim.",
        ],
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "ledger_evidence": {
            "path": str(ledger_path.relative_to(PROJECT_ROOT)),
            "record_id": h3_rejection["record_id"],
            "record_sha256": hashlib.sha256(
                json.dumps(h3_rejection, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "sources": sources,
    }


def render_markdown(report: dict[str, Any]) -> str:
    robot = report["robot_embodiment_replacement"]
    lines = [
        "# Agentic workflow benchmark tables",
        "",
        "## Robot embodiment replacement: single pass versus safe agentic selection",
        "",
        "| Metric | Single pass | Agent-selected | Absolute gain |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in robot["single_pass_vs_agentic_safe_selection"]["metrics"]:
        lines.append(
            f"| `{row['metric']}` | {row['baseline']:.4f} | "
            f"{row['agentic']:.4f} | {row['absolute_gain']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Robot embodiment replacement: matched short-window renderer comparison",
            "",
            "| Metric | 2-D pose rig | H3 + agentic repair | Absolute gain |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in robot["pose_rig_vs_h3_agentic_short_window"]["metrics"]:
        lines.append(
            f"| `{row['metric']}` | {row['baseline']:.4f} | "
            f"{row['agentic']:.4f} | {row['absolute_gain']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## AC-WM: human-gated action success",
            "",
            "| Method | Accepted / total | Rate |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in report["acwm"]["human_gated_success"]:
        lines.append(
            f"| {row['method']} | {row['accepted']}/{row['total']} | {100 * row['rate']:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def _git_state() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"available": True, "head": head, "status": status}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-trace", type=Path, default=DEFAULT_ROBOT_TRACE)
    parser.add_argument("--pose-rig", type=Path, default=DEFAULT_POSE_RIG)
    parser.add_argument("--h3", type=Path, default=DEFAULT_H3)
    parser.add_argument("--structure-trace", type=Path, default=DEFAULT_STRUCTURE_TRACE)
    parser.add_argument("--acwm", type=Path, default=DEFAULT_ACWM)
    parser.add_argument("--repair-policy", type=Path, default=DEFAULT_REPAIR_POLICY)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"benchmark output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    evidence_paths = {
        "robot_trace_path": args.robot_trace.expanduser().resolve(),
        "pose_rig_path": args.pose_rig.expanduser().resolve(),
        "h3_path": args.h3.expanduser().resolve(),
        "structure_trace_path": args.structure_trace.expanduser().resolve(),
        "acwm_path": args.acwm.expanduser().resolve(),
        "repair_policy_path": args.repair_policy.expanduser().resolve(),
        "ledger_path": args.ledger.expanduser().resolve(),
    }
    report = build_report(**evidence_paths)
    report["command"] = sys.argv
    report["git"] = _git_state()

    (output_dir / "benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / "tables.md").write_text(render_markdown(report))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "benchmark": str(output_dir / "benchmark.json"),
                "tables": str(output_dir / "tables.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
