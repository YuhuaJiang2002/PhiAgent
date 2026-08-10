import json
from pathlib import Path

import pytest

from phiagent.learning.experience import (
    ExperienceRecord,
    append_experience,
    load_experiences,
    read_status_inventory,
    summarize_experiences,
)


def _record(base_id: str, **overrides: object) -> ExperienceRecord:
    data: dict[str, object] = {
        "record_id": base_id,
        "recorded_at": "2026-08-10T12:00:00+08:00",
        "status": "PARTIAL",
        "scope": "unit fixture",
        "summary": "A measured result is incomplete.",
        "evidence": ["run/result.json"],
        "lessons": ["A partial result must retain its failed gate."],
        "limitations": ["The held-out threshold failed."],
        "next_actions": ["Run the bounded follow-up."],
    }
    data.update(overrides)
    return ExperienceRecord.from_dict(data)


def test_append_load_and_supersede_experience(tmp_path: Path) -> None:
    ledger = tmp_path / "experiences" / "ledger.jsonl"
    first = _record("trial.001")
    second = _record(
        "trial.002",
        status="WORKING",
        summary="The held-out threshold now passes.",
        evidence=["run-2/result.json", "run-2/config.json"],
        lessons=["The bounded change passed the held-out gate."],
        limitations=[],
        next_actions=[],
        supersedes=["trial.001"],
    )
    append_experience(ledger, first)
    append_experience(ledger, second)

    records = load_experiences(ledger)
    assert records == (first, second)
    assert summarize_experiences(records) == {
        "schema_version": "1.0.0",
        "history_records": 2,
        "active_records": 1,
        "history_by_status": {
            "WORKING": 1,
            "PARTIAL": 1,
            "BLOCKED": 0,
            "NOT STARTED": 0,
        },
        "active_by_status": {
            "WORKING": 1,
            "PARTIAL": 0,
            "BLOCKED": 0,
            "NOT STARTED": 0,
        },
        "active_record_ids": ["trial.002"],
    }


def test_append_rejects_duplicate_and_unknown_history(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    append_experience(ledger, _record("trial.001"))
    with pytest.raises(ValueError, match="already exists"):
        append_experience(ledger, _record("trial.001"))
    with pytest.raises(ValueError, match="must reference existing"):
        append_experience(ledger, _record("trial.002", supersedes=["trial.999"]))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"recorded_at": "2026-08-10T12:00:00"}, "timezone"),
        ({"status": "WORKING", "evidence": []}, "measured evidence"),
        ({"status": "PARTIAL", "limitations": []}, "explicit limitations"),
        ({"status": "BLOCKED", "lessons": []}, "at least one lesson"),
        ({"status": "NOT STARTED", "next_actions": []}, "next action"),
        ({"record_id": "Bad ID"}, "record_id"),
    ],
)
def test_record_rejects_unverifiable_claims(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _record("trial.001", **overrides)


def test_load_reports_corrupt_line(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps(_record("trial.001").to_dict()) + "\n{bad json}\n")
    with pytest.raises(ValueError, match=r"ledger.jsonl:2"):
        load_experiences(ledger)


def test_status_inventory_counts_only_top_level_status_entries(tmp_path: Path) -> None:
    status = tmp_path / "STATUS.md"
    status.write_text(
        "# Status\n\n"
        "## WORKING\n\n- accepted\n  - nested evidence\n\n"
        "## PARTIAL\n\n- partial one\n- partial two\n\n"
        "## BLOCKED\n\n- blocker\n\n"
        "## NOT STARTED\n\n- future\n\n"
        "## Notes\n\n- not a status entry\n"
    )
    inventory = read_status_inventory(status)
    assert inventory.counts == {
        "WORKING": 1,
        "PARTIAL": 2,
        "BLOCKED": 1,
        "NOT STARTED": 1,
    }
    assert inventory.total == 5
