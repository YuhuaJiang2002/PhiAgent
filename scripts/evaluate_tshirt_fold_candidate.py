#!/usr/bin/env python3
"""Evaluate one T-shirt folding proposal with fail-closed visual hard gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.evaluation.tshirt_fold_video import (  # noqa: E402
    extract_and_score_tshirt_fold_video,
    load_tshirt_fold_tracking_contract,
    write_tshirt_fold_evidence,
)
from phiagent.harness.task_reasoning import (  # noqa: E402
    TSHIRT_FOLD_TASK,
    TaskReasoningPlan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--first-frame", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tracking-contract", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def main() -> int:
    args = _parser().parse_args()
    plan_payload = json.loads(args.plan.expanduser().resolve().read_text())
    if not isinstance(plan_payload, dict):
        raise ValueError("task reasoning plan must contain one JSON object")
    plan = TaskReasoningPlan.from_dict(plan_payload)
    if plan.task_type != TSHIRT_FOLD_TASK:
        raise ValueError("T-shirt evaluator requires a T-shirt folding reasoning plan")
    contract = load_tshirt_fold_tracking_contract(args.tracking_contract)
    if contract.coordinate_frame != plan.coordinate_frame:
        raise ValueError("tracking contract and reasoning plan coordinate frames differ")
    score = extract_and_score_tshirt_fold_video(
        args.candidate,
        args.first_frame,
        contract=contract,
    )
    evidence_path = args.metadata.expanduser().resolve().with_suffix(
        ".tshirt-hard-gates.json"
    )
    write_tshirt_fold_evidence(evidence_path, score)
    gate_fraction = sum(score.gate_results.values()) / len(score.gate_results)
    maximum_sleeve_deviation = max(
        item.maximum_relative_deviation for item in score.sleeve_scores.values()
    )
    sleeve_limit = contract.thresholds.sleeve.maximum_relative_deviation
    embodiment_score = _clamp(1.0 - maximum_sleeve_deviation / max(sleeve_limit, 1e-6))
    temporal_score = _clamp(
        1.0
        - score.maximum_material_step_pixels
        / max(contract.thresholds.maximum_material_step_pixels * 2.0, 1e-6)
    )
    payload = {
        "evaluator": "tshirt-fold-camera-material-gates-v1",
        "action_adherence": gate_fraction,
        "embodiment_consistency": embodiment_score,
        "object_interaction": gate_fraction,
        "temporal_consistency": temporal_score,
        "background_consistency": score.background_score,
        "hard_gates_passed": score.hard_gates_passed,
        "human_review_passed": None,
        "diagnoses": [f"hard_gate:{name}" for name in score.failed_gates],
        "evidence": str(evidence_path),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
