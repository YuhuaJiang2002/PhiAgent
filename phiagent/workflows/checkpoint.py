"""Dependency-free, JSON-only checkpoint stores for PhiAgent workflows."""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol


_THREAD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class CheckpointError(RuntimeError):
    """Raised when checkpoint state is invalid or unavailable."""


class CheckpointStore(Protocol):
    """Minimal persistence interface consumed by :class:`CompiledGraph`."""

    def save(
        self,
        thread_id: str,
        *,
        state: Mapping[str, Any],
        next_node: str,
        status: str,
        step: int,
        interrupt: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def load_latest(self, thread_id: str) -> dict[str, Any] | None: ...

    def history(self, thread_id: str) -> list[dict[str, Any]]: ...


def validate_thread_id(thread_id: str) -> str:
    if not _THREAD_ID.fullmatch(thread_id):
        raise ValueError(
            "thread_id must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-' (maximum 128 characters)"
        )
    return thread_id


def json_clone(value: Any) -> Any:
    """Validate that ``value`` is stable JSON and return a detached clone."""

    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"workflow state must be finite JSON: {exc}") from exc
    return json.loads(payload)


def state_sha256(state: Mapping[str, Any]) -> str:
    payload = json.dumps(
        state,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _checkpoint_payload(
    *,
    checkpoint_id: int,
    parent_id: int | None,
    thread_id: str,
    state: Mapping[str, Any],
    next_node: str,
    status: str,
    step: int,
    interrupt: Mapping[str, Any] | None,
    error: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    detached_state = json_clone(dict(state))
    return json_clone(
        {
            "schema_version": "1.0.0",
            "checkpoint_id": checkpoint_id,
            "parent_id": parent_id,
            "thread_id": thread_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "step": step,
            "next_node": next_node,
            "state_sha256": state_sha256(detached_state),
            "state": detached_state,
            "interrupt": interrupt,
            "error": error,
            "metadata": dict(metadata or {}),
        }
    )


@dataclass
class MemoryCheckpointer:
    """Process-local checkpointer intended for tests and ephemeral examples."""

    _threads: dict[str, list[dict[str, Any]]]

    def __init__(self) -> None:
        self._threads = {}

    def save(
        self,
        thread_id: str,
        *,
        state: Mapping[str, Any],
        next_node: str,
        status: str,
        step: int,
        interrupt: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        validate_thread_id(thread_id)
        rows = self._threads.setdefault(thread_id, [])
        payload = _checkpoint_payload(
            checkpoint_id=len(rows),
            parent_id=len(rows) - 1 if rows else None,
            thread_id=thread_id,
            state=state,
            next_node=next_node,
            status=status,
            step=step,
            interrupt=interrupt,
            error=error,
            metadata=metadata,
        )
        rows.append(payload)
        return deepcopy(payload)

    def load_latest(self, thread_id: str) -> dict[str, Any] | None:
        validate_thread_id(thread_id)
        rows = self._threads.get(thread_id, [])
        return deepcopy(rows[-1]) if rows else None

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        validate_thread_id(thread_id)
        return deepcopy(self._threads.get(thread_id, []))


class JsonFileCheckpointer:
    """Append-only JSON checkpoints with an atomically replaced latest pointer."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def _thread_dir(self, thread_id: str) -> Path:
        return self.root / validate_thread_id(thread_id)

    def save(
        self,
        thread_id: str,
        *,
        state: Mapping[str, Any],
        next_node: str,
        status: str,
        step: int,
        interrupt: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        thread_dir = self._thread_dir(thread_id)
        checkpoint_dir = thread_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        previous = self.load_latest(thread_id)
        checkpoint_id = int(previous["checkpoint_id"]) + 1 if previous else 0
        payload = _checkpoint_payload(
            checkpoint_id=checkpoint_id,
            parent_id=int(previous["checkpoint_id"]) if previous else None,
            thread_id=thread_id,
            state=state,
            next_node=next_node,
            status=status,
            step=step,
            interrupt=interrupt,
            error=error,
            metadata=metadata,
        )
        checkpoint_path = checkpoint_dir / f"{checkpoint_id:08d}.json"
        self._atomic_json(checkpoint_path, payload)
        self._atomic_json(thread_dir / "latest.json", payload)
        return payload

    def load_latest(self, thread_id: str) -> dict[str, Any] | None:
        path = self._thread_dir(thread_id) / "latest.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("thread_id") != thread_id:
            raise CheckpointError(f"checkpoint thread mismatch in {path}")
        if state_sha256(payload["state"]) != payload.get("state_sha256"):
            raise CheckpointError(f"checkpoint state hash mismatch in {path}")
        return payload

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        directory = self._thread_dir(thread_id) / "checkpoints"
        if not directory.exists():
            return []
        rows = []
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if state_sha256(payload["state"]) != payload.get("state_sha256"):
                raise CheckpointError(f"checkpoint state hash mismatch in {path}")
            rows.append(payload)
        return rows

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
