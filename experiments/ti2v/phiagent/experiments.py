"""Dependency-free experiment directory reservation and timestamp validation."""

from __future__ import annotations

import json
import os
import re
import socket
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phiagent.data_engine.provenance import capture_provenance, write_json_atomic

DIRECTORY_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
_DIRECTORY_PATTERN = re.compile(
    r"^(?P<timestamp>\d{8}T\d{6}Z)-(?P<label>[A-Za-z0-9][A-Za-z0-9._-]*)$"
)
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TIMESTAMP_KEYS = {
    "created_at",
    "created_at_utc",
    "directory_timestamp_utc",
    "finished_at_utc",
    "frozen_at_utc",
    "reserved_at_utc",
    "started_at_utc",
}
_AUDIT_FILES = (
    "reservation.json",
    "config.json",
    "manifest.json",
    "provenance.json",
    "command.json",
    "results.json",
)


@dataclass(frozen=True)
class ExperimentReservation:
    path: Path
    timestamp: datetime

    @property
    def directory_timestamp(self) -> str:
        return self.timestamp.strftime(DIRECTORY_TIMESTAMP_FORMAT)

    @property
    def timestamp_utc(self) -> str:
        return _format_utc(self.timestamp)


@dataclass(frozen=True)
class TimestampAudit:
    path: Path
    status: str
    directory_timestamp_utc: str | None
    earliest_recorded_at_utc: str | None
    observations: tuple[dict[str, str], ...]
    issues: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.status == "VALID"

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "status": self.status,
            "valid": self.valid,
            "directory_timestamp_utc": self.directory_timestamp_utc,
            "earliest_recorded_at_utc": self.earliest_recorded_at_utc,
            "observations": list(self.observations),
            "issues": list(self.issues),
        }


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("experiment clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc).replace(microsecond=0)


def _format_utc(value: datetime) -> str:
    return _normalize_utc(value).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a UTC timezone")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError("timestamp must be UTC, not a local timezone offset")
    return parsed.astimezone(timezone.utc)


def _validate_label(label: str) -> None:
    if not _LABEL_PATTERN.fullmatch(label):
        raise ValueError(
            "experiment label must start with an ASCII letter or digit and contain "
            "only ASCII letters, digits, '.', '_' or '-'"
        )


def reserve_experiment_directory(
    root: Path,
    label: str | None = None,
    *,
    clock: Callable[[], datetime] | None = None,
) -> ExperimentReservation:
    """Create a new directory whose name and reservation use one UTC clock read."""

    resolved_label = label or uuid.uuid4().hex[:8]
    _validate_label(resolved_label)
    now = _normalize_utc((clock or (lambda: datetime.now(timezone.utc)))())
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{now.strftime(DIRECTORY_TIMESTAMP_FORMAT)}-{resolved_label}"
    path.mkdir(exist_ok=False)
    reservation = ExperimentReservation(path=path, timestamp=now)
    write_json_atomic(
        path / "reservation.json",
        {
            "schema_version": "1.0.0",
            "status": "RESERVED",
            "directory_name": path.name,
            "directory_timestamp_utc": reservation.timestamp_utc,
            "reserved_at_utc": reservation.timestamp_utc,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        },
    )
    return reservation


def create_frozen_experiment_directory(
    root: Path,
    label: str,
    config: Mapping[str, Any],
    *,
    repo_root: Path,
    command: Sequence[str] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> ExperimentReservation:
    """Reserve an experiment and atomically persist a matching frozen config."""

    protected = {"directory_timestamp_utc", "frozen_at_utc"}
    conflicts = sorted(protected.intersection(config))
    if conflicts:
        raise ValueError("config must not provide clock-owned fields: " + ", ".join(conflicts))
    if "seed" not in config:
        raise ValueError("config must explicitly record an integer seed")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("config seed must be an integer")
    status = config.get("status", "FROZEN_BEFORE_FIRST_RUN")
    if status != "FROZEN_BEFORE_FIRST_RUN":
        raise ValueError("initial experiment status must be FROZEN_BEFORE_FIRST_RUN")

    reservation = reserve_experiment_directory(root, label, clock=clock)
    frozen_config = dict(config)
    frozen_config["status"] = status
    frozen_config["directory_timestamp_utc"] = reservation.timestamp_utc
    frozen_config["frozen_at_utc"] = reservation.timestamp_utc
    write_json_atomic(reservation.path / "config.json", frozen_config)
    write_json_atomic(
        reservation.path / "provenance.json",
        capture_provenance(
            repo_root,
            tuple(command or sys.argv),
            seed=seed,
        ),
    )
    return reservation


def _directory_timestamp(path: Path) -> datetime:
    match = _DIRECTORY_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"experiment directory must match YYYYMMDDTHHMMSSZ-label: {path.name}")
    return datetime.strptime(match.group("timestamp"), DIRECTORY_TIMESTAMP_FORMAT).replace(
        tzinfo=timezone.utc
    )


def audit_experiment_timestamps(path: Path, *, require_reservation: bool = False) -> TimestampAudit:
    """Validate that an experiment directory is not dated after its own records."""

    issues: list[str] = []
    observations: list[dict[str, str]] = []
    try:
        directory_timestamp = _directory_timestamp(path)
        directory_timestamp_utc = _format_utc(directory_timestamp)
    except ValueError as exc:
        directory_timestamp = None
        directory_timestamp_utc = None
        issues.append(str(exc))

    exclusion = path / "EXCLUDED_TIMESTAMP_INVALID.json"
    if exclusion.is_file():
        try:
            marker = json.loads(exclusion.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"invalid exclusion marker: {exc}")
        else:
            if marker.get("status") != "EXCLUDED_TIMESTAMP_INVALID":
                issues.append("exclusion marker has an unexpected status")
            return TimestampAudit(
                path=path,
                status="EXCLUDED",
                directory_timestamp_utc=directory_timestamp_utc,
                earliest_recorded_at_utc=None,
                observations=tuple(observations),
                issues=tuple(issues or (str(marker.get("reason", "excluded")),)),
            )

    reservation_path = path / "reservation.json"
    if require_reservation and not reservation_path.is_file():
        issues.append("reservation.json is required but missing")

    recorded: list[tuple[datetime, str, str]] = []
    for filename in _AUDIT_FILES:
        artifact = path / filename
        if not artifact.is_file():
            continue
        try:
            payload = json.loads(artifact.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"{filename} is not valid JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            issues.append(f"{filename} must contain a JSON object")
            continue
        for key in sorted(_TIMESTAMP_KEYS.intersection(payload)):
            value = payload[key]
            if not isinstance(value, str):
                issues.append(f"{filename}:{key} must be a string")
                continue
            try:
                parsed = _parse_utc(value)
            except ValueError as exc:
                issues.append(f"{filename}:{key}: {exc}")
                continue
            normalized = _format_utc(parsed)
            observations.append({"file": filename, "field": key, "timestamp_utc": normalized})
            if key != "directory_timestamp_utc":
                recorded.append((parsed, filename, key))
            elif directory_timestamp is not None and parsed != directory_timestamp:
                issues.append(
                    f"{filename}:{key}={normalized} does not match directory "
                    f"timestamp {directory_timestamp_utc}"
                )

    if not recorded:
        issues.append("no recorded UTC lifecycle timestamp found")
        earliest = None
    else:
        earliest, earliest_file, earliest_key = min(recorded)
        if directory_timestamp is not None and directory_timestamp > earliest:
            issues.append(
                f"directory timestamp {directory_timestamp_utc} is after "
                f"{earliest_file}:{earliest_key}={_format_utc(earliest)}"
            )

    if reservation_path.is_file() and directory_timestamp is not None:
        reservation_observations = [
            item
            for item in observations
            if item["file"] == "reservation.json"
            and item["field"] in {"directory_timestamp_utc", "reserved_at_utc"}
        ]
        if len(reservation_observations) != 2:
            issues.append(
                "reservation.json must record directory_timestamp_utc and reserved_at_utc"
            )
        elif any(
            item["timestamp_utc"] != directory_timestamp_utc for item in reservation_observations
        ):
            issues.append("reservation timestamps do not match the directory timestamp")

    return TimestampAudit(
        path=path,
        status="INVALID" if issues else "VALID",
        directory_timestamp_utc=directory_timestamp_utc,
        earliest_recorded_at_utc=_format_utc(earliest) if earliest else None,
        observations=tuple(observations),
        issues=tuple(issues),
    )
