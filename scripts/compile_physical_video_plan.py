#!/usr/bin/env python3
"""Compile a typed manipulation request into a frozen plan and scaling policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.harness.provenance import (  # noqa: E402
    capture_provenance,
    write_json_atomic,
)
from phiagent.harness.task_reasoning import (  # noqa: E402
    OPTICAL_MODULE_TASK,
    TSHIRT_FOLD_TASK,
    PhysicalTaskReasoningPlugin,
    TaskReasoningRequest,
    TshirtFoldReasoningPlugin,
)
from phiagent.harness.test_time_scaling import (  # noqa: E402
    ScalingRound,
    TestTimeScalingPolicy,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser


def _plugin(task_type: str):
    if task_type == TSHIRT_FOLD_TASK:
        return TshirtFoldReasoningPlugin()
    if task_type == OPTICAL_MODULE_TASK:
        return PhysicalTaskReasoningPlugin()
    raise ValueError(f"unsupported physical task type: {task_type}")


def main() -> int:
    args = _parser().parse_args()
    request_path = args.request.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"physical video plan output already exists: {output_dir}")
    payload = json.loads(request_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("physical video request must contain one JSON object")
    request = TaskReasoningRequest.from_dict(payload)
    plan = _plugin(request.task_type).analyze(request)
    policy = TestTimeScalingPolicy(
        rounds=(
            ScalingRound(4, 24, 0, "diverse-seed material-consistency search"),
            ScalingRound(4, 32, 100_000, "hard-gate-specific prompt repair"),
            ScalingRound(2, 40, 200_000, "high-step confirmation without gate relaxation"),
        ),
        seed_stride=1009,
        maximum_candidates=10,
    )
    output_dir.mkdir(parents=True)
    request_copy = output_dir / "request.json"
    plan_path = output_dir / "task-reasoning-plan.json"
    policy_path = output_dir / "test-time-scaling.json"
    manifest_path = output_dir / "compile-manifest.json"
    write_json_atomic(request_copy, request.to_dict())
    write_json_atomic(plan_path, plan.to_dict())
    write_json_atomic(policy_path, policy.to_dict())
    write_json_atomic(
        manifest_path,
        {
            **capture_provenance(
                Path(__file__).resolve().parents[1],
                [sys.executable, *sys.argv],
                args.seed,
            ),
            "status": "compiled",
            "honest_status": "NOT STARTED",
            "method": "hash_bound_physical_task_language_plan",
            "request": str(request_copy),
            "task_reasoning_plan": str(plan_path),
            "plan_sha256": plan.plan_sha256,
            "test_time_scaling_policy": str(policy_path),
            "claim_boundary": plan.claim_boundary,
        },
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "plan": str(plan_path),
                "plan_sha256": plan.plan_sha256,
                "policy": str(policy_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
