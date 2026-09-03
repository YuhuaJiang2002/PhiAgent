"""Pre-registered real-robot schedules that never invoke hardware control."""

from __future__ import annotations

import json
import random
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from phiagent.benchmark.artifacts import sha256_file
from phiagent.benchmark.hardware import HardwareAdapterManifest
from phiagent.benchmark.metrics import BenchmarkPolicy, evaluate_submission, simulation_pass
from phiagent.benchmark.schema import BenchmarkSuite, Submission


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def create_real_trial_plan(
    *,
    suite_path: Path,
    submission_path: Path,
    policy_path: Path,
    protocol_path: Path,
    adapter_path: Path,
    session_ids: tuple[str, ...],
    output_dir: Path,
    random_seed: int,
) -> dict[str, Any]:
    """Create blinded schedules and coordinator mappings without moving a robot."""

    target = output_dir.expanduser().resolve()
    if target.exists():
        raise ValueError(f"real trial plan directory already exists: {target}")
    suite = BenchmarkSuite.from_json(suite_path.expanduser().resolve())
    submission = Submission.from_json(submission_path.expanduser().resolve())
    policy = BenchmarkPolicy.from_json(policy_path.expanduser().resolve())
    protocol = _read(protocol_path)
    adapter = HardwareAdapterManifest.from_json(adapter_path.expanduser().resolve())
    if protocol.get("protocol_id") is None:
        raise ValueError("real protocol lacks protocol_id")
    design = protocol.get("design")
    if not isinstance(design, Mapping):
        raise ValueError("real protocol lacks design")
    minimum_trials = int(design["minimum_trials_per_case_and_embodiment"])
    minimum_sessions = int(design["minimum_independent_sessions"])
    if policy.minimum_real_trials != minimum_trials:
        raise ValueError("policy and real protocol disagree on minimum trials")
    if policy.minimum_real_sessions != minimum_sessions:
        raise ValueError("policy and real protocol disagree on minimum sessions")
    if len(session_ids) < minimum_sessions or len(set(session_ids)) != len(session_ids):
        raise ValueError("real plan requires unique session IDs meeting the protocol minimum")
    if any(not session.strip() for session in session_ids):
        raise ValueError("real session IDs cannot be empty")

    report = evaluate_submission(suite, submission, policy)
    record_by_id = {record.case_id: record for record in submission.records}
    eligible_cases = [
        case
        for case in suite.cases
        if "l5_real" in case.required_dimensions
        and simulation_pass(record_by_id[case.case_id], policy)
    ]
    compatibility = [adapter.compatibility(case) for case in eligible_cases]
    incompatible = [item["case_id"] for item in compatibility if not item["compatible"]]
    if incompatible:
        raise ValueError(f"hardware adapter is incompatible with cases: {incompatible}")

    coordinator_rows: list[dict[str, Any]] = []
    for case in eligible_cases:
        timeout_s = case.annotation.get("real_timeout_s", protocol.get("default_timeout_s"))
        if timeout_s is None or float(timeout_s) <= 0:
            raise ValueError(f"case {case.case_id} lacks a positive real timeout")
        for trial_index in range(minimum_trials):
            coordinator_rows.append(
                {
                    "blind_code": secrets.token_hex(12),
                    "case_id": case.case_id,
                    "method": submission.method,
                    "trial_index": trial_index,
                    "session_id": session_ids[trial_index % len(session_ids)],
                    "timeout_s": float(timeout_s),
                }
            )
    generator = random.Random(random_seed)
    generator.shuffle(coordinator_rows)
    for schedule_index, row in enumerate(coordinator_rows):
        row["schedule_index"] = schedule_index

    operator_rows = [
        {
            key: row[key]
            for key in (
                "schedule_index",
                "blind_code",
                "case_id",
                "trial_index",
                "session_id",
                "timeout_s",
            )
        }
        for row in coordinator_rows
    ]
    target.mkdir(parents=True)
    operator_path = target / "operator-schedule.json"
    coordinator_path = target / "coordinator-mapping.json"
    _write(
        operator_path,
        {
            "schema_version": "0.2.0",
            "protocol_id": protocol["protocol_id"],
            "method_hidden": True,
            "rows": operator_rows,
        },
    )
    _write(
        coordinator_path,
        {
            "schema_version": "0.2.0",
            "access": "coordinator_only_do_not_share_with_blind_reviewer",
            "rows": coordinator_rows,
        },
    )
    summary = {
        "schema_version": "0.2.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "suite": suite.name,
        "method": submission.method,
        "adapter": adapter.adapter_name,
        "eligible_case_count": len(eligible_cases),
        "planned_trial_count": len(coordinator_rows),
        "session_count": len(session_ids),
        "random_seed": random_seed,
        "inputs": {
            "suite_sha256": sha256_file(suite_path.expanduser().resolve()),
            "submission_sha256": sha256_file(submission_path.expanduser().resolve()),
            "policy_sha256": sha256_file(policy_path.expanduser().resolve()),
            "protocol_sha256": sha256_file(protocol_path.expanduser().resolve()),
            "adapter_sha256": sha256_file(adapter_path.expanduser().resolve()),
        },
        "schedule_hashes": {
            "operator_schedule_sha256": sha256_file(operator_path),
            "coordinator_mapping_sha256": sha256_file(coordinator_path),
        },
        "hardware_control_invoked": False,
        "adapter_execution_enabled": adapter.execution_enabled,
        "evidence_only": adapter.evidence_only,
        "authorization_ready": adapter.execution_enabled and not adapter.evidence_only,
        "status": (
            "planned"
            if adapter.execution_enabled and not adapter.evidence_only
            else "blocked_pending_site_authorization"
        ),
        "l4_sim_pass_count": report["real_audit"]["sim_pass_count"],
        "claim_boundary": (
            "This plan randomizes and blinds eligible trials. It does not invoke a robot; "
            "each trial still requires a hash-bound pre-registration bundle and site approval."
        ),
    }
    _write(target / "summary.json", summary)
    return summary
