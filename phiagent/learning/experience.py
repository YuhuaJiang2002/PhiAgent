"""Append-only, evidence-backed experience records for PhiAgent.

This module deliberately uses only the Python standard library so importing
``phiagent`` never pulls in model, simulator, GPU, or SkillHone dependencies.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # pragma: no cover - Windows fallback is exercised by code inspection.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


SCHEMA_VERSION = "1.0.0"
STATUSES = ("WORKING", "PARTIAL", "BLOCKED", "NOT STARTED")
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")


def _clean_strings(values: Iterable[object], field: str) -> tuple[str, ...]:
    cleaned = tuple(str(value).strip() for value in values)
    if any(not value for value in cleaned):
        raise ValueError(f"{field} must not contain empty values")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{field} must not contain duplicates")
    return cleaned


@dataclass(frozen=True)
class ExperienceRecord:
    """One immutable decision-history item.

    A later correction must append a new record and name the old record in
    ``supersedes``. Existing JSONL lines are never edited in place.
    """

    record_id: str
    recorded_at: str
    status: str
    scope: str
    summary: str
    evidence: tuple[str, ...]
    lessons: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    run_dir: str | None = None
    supersedes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", self.record_id.strip())
        object.__setattr__(self, "recorded_at", self.recorded_at.strip())
        object.__setattr__(self, "status", self.status.strip().upper())
        object.__setattr__(self, "scope", self.scope.strip())
        object.__setattr__(self, "summary", self.summary.strip())
        object.__setattr__(self, "evidence", _clean_strings(self.evidence, "evidence"))
        object.__setattr__(self, "lessons", _clean_strings(self.lessons, "lessons"))
        object.__setattr__(
            self, "limitations", _clean_strings(self.limitations, "limitations")
        )
        object.__setattr__(
            self, "next_actions", _clean_strings(self.next_actions, "next_actions")
        )
        object.__setattr__(
            self, "supersedes", _clean_strings(self.supersedes, "supersedes")
        )
        object.__setattr__(self, "tags", _clean_strings(self.tags, "tags"))
        if self.run_dir is not None:
            run_dir = self.run_dir.strip()
            object.__setattr__(self, "run_dir", run_dir or None)
        self.validate()

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        if not _ID_PATTERN.fullmatch(self.record_id):
            raise ValueError(
                "record_id must be 3-128 lowercase letters, digits, dots, "
                "underscores, or hyphens"
            )
        try:
            timestamp = datetime.fromisoformat(self.recorded_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("recorded_at must be an ISO-8601 timestamp") from error
        if timestamp.tzinfo is None:
            raise ValueError("recorded_at must include a timezone")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        if not self.scope:
            raise ValueError("scope is required")
        if not self.summary:
            raise ValueError("summary is required")
        if self.status in {"WORKING", "PARTIAL"} and not self.evidence:
            raise ValueError(f"{self.status} records require measured evidence")
        if self.status != "NOT STARTED" and not self.lessons:
            raise ValueError(f"{self.status} records require at least one lesson")
        if self.status in {"PARTIAL", "BLOCKED"} and not self.limitations:
            raise ValueError(f"{self.status} records require explicit limitations")
        if self.status in {"PARTIAL", "BLOCKED", "NOT STARTED"} and not self.next_actions:
            raise ValueError(f"{self.status} records require a next action")
        if self.record_id in self.supersedes:
            raise ValueError("a record cannot supersede itself")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperienceRecord:
        allowed = {
            "schema_version",
            "record_id",
            "recorded_at",
            "status",
            "scope",
            "summary",
            "evidence",
            "lessons",
            "limitations",
            "next_actions",
            "run_dir",
            "supersedes",
            "tags",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown experience fields: {', '.join(unknown)}")
        required = {
            "record_id",
            "recorded_at",
            "status",
            "scope",
            "summary",
            "evidence",
            "lessons",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"missing experience fields: {', '.join(missing)}")
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            record_id=str(data["record_id"]),
            recorded_at=str(data["recorded_at"]),
            status=str(data["status"]),
            scope=str(data["scope"]),
            summary=str(data["summary"]),
            evidence=tuple(data["evidence"]),
            lessons=tuple(data["lessons"]),
            limitations=tuple(data.get("limitations", ())),
            next_actions=tuple(data.get("next_actions", ())),
            run_dir=None if data.get("run_dir") is None else str(data["run_dir"]),
            supersedes=tuple(data.get("supersedes", ())),
            tags=tuple(data.get("tags", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "recorded_at": self.recorded_at,
            "status": self.status,
            "scope": self.scope,
            "summary": self.summary,
            "evidence": list(self.evidence),
            "lessons": list(self.lessons),
            "limitations": list(self.limitations),
            "next_actions": list(self.next_actions),
            "supersedes": list(self.supersedes),
            "tags": list(self.tags),
        }
        if self.run_dir is not None:
            data["run_dir"] = self.run_dir
        return data


@dataclass(frozen=True)
class StatusInventory:
    """Counts of every top-level milestone in ``docs/STATUS.md``."""

    counts: Mapping[str, int]
    source: Path

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def read_status_inventory(path: Path) -> StatusInventory:
    """Count all top-level entries under the four honest status headings."""

    counts: Counter[str] = Counter()
    section: str | None = None
    for raw_line in path.read_text().splitlines():
        if raw_line.startswith("## "):
            heading = raw_line[3:].strip().upper()
            section = heading if heading in STATUSES else None
            continue
        if section is not None and raw_line.startswith("- "):
            counts[section] += 1
    return StatusInventory(
        counts={status: counts[status] for status in STATUSES}, source=path
    )


def load_experiences(path: Path) -> tuple[ExperienceRecord, ...]:
    """Load and validate an append-only JSONL ledger."""

    if not path.exists():
        return ()
    records: list[ExperienceRecord] = []
    known_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
            if not isinstance(raw, dict):
                raise ValueError("record must be a JSON object")
            record = ExperienceRecord.from_dict(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid experience at {path}:{line_number}: {error}") from error
        if record.record_id in known_ids:
            raise ValueError(f"duplicate record_id {record.record_id!r} at {path}:{line_number}")
        unknown_superseded = sorted(set(record.supersedes) - known_ids)
        if unknown_superseded:
            raise ValueError(
                f"record {record.record_id!r} supersedes unknown or future ids: "
                f"{', '.join(unknown_superseded)}"
            )
        known_ids.add(record.record_id)
        records.append(record)
    return tuple(records)


def append_experience(path: Path, record: ExperienceRecord) -> None:
    """Append one record while rejecting duplicates and broken history links."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        if fcntl is not None:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.seek(0)
            existing: list[ExperienceRecord] = []
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                try:
                    raw = json.loads(raw_line)
                    if not isinstance(raw, dict):
                        raise ValueError("record must be a JSON object")
                    existing.append(ExperienceRecord.from_dict(raw))
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid experience at {path}:{line_number}: {error}"
                    ) from error
            existing_ids = {item.record_id for item in existing}
            if record.record_id in existing_ids:
                raise ValueError(f"record_id already exists: {record.record_id}")
            unknown_superseded = sorted(set(record.supersedes) - existing_ids)
            if unknown_superseded:
                raise ValueError(
                    "supersedes must reference existing records: "
                    + ", ".join(unknown_superseded)
                )
            stream.seek(0, os.SEEK_END)
            payload = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def summarize_experiences(records: Iterable[ExperienceRecord]) -> dict[str, Any]:
    """Return stable counts for the full history and currently active records."""

    history = tuple(records)
    ids = [record.record_id for record in history]
    if len(ids) != len(set(ids)):
        raise ValueError("experience record_ids must be unique")
    superseded = {record_id for record in history for record_id in record.supersedes}
    active = tuple(record for record in history if record.record_id not in superseded)
    return {
        "schema_version": SCHEMA_VERSION,
        "history_records": len(history),
        "active_records": len(active),
        "history_by_status": {
            status: sum(record.status == status for record in history) for status in STATUSES
        },
        "active_by_status": {
            status: sum(record.status == status for record in active) for status in STATUSES
        },
        "active_record_ids": [record.record_id for record in active],
    }
