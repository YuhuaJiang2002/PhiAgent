"""File-backed manager for recoverable, multi-worker campaign transitions."""

from __future__ import annotations

import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:  # pragma: no cover - production workers are Linux; fallback fails closed.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from phiagent.data_engine.planner import CampaignPlan, JobSpec
from phiagent.data_engine.provenance import utc_now, write_json_atomic
from phiagent.data_engine.state import AuditReport, CampaignState, JobStatus


class CampaignController:
    """Serialize state transitions while keeping jobs executor-agnostic."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.expanduser().resolve()
        self.plan_path = self.run_dir / "plan.json"
        self.state_path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "logs" / "events.jsonl"
        self.lock_path = self.run_dir / ".state.lock"
        for path in (self.plan_path, self.state_path, self.events_path):
            if not path.is_file():
                raise FileNotFoundError(f"campaign run is missing {path.relative_to(self.run_dir)}")

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        if fcntl is None:
            raise RuntimeError("multi-worker campaign control requires POSIX file locking")
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _load(self) -> tuple[CampaignPlan, CampaignState]:
        plan_payload = json.loads(self.plan_path.read_text())
        state_payload = json.loads(self.state_path.read_text())
        plan = CampaignPlan.from_dict(plan_payload)
        state = CampaignState.from_dict(state_payload)
        if state.campaign_id != plan.campaign_id or state.plan_sha256 != plan.plan_sha256:
            raise ValueError("campaign state is not bound to the immutable plan")
        plan_jobs = {item.job_id for item in plan.jobs}
        state_jobs = {item.job_id for item in state.jobs}
        if state_jobs != plan_jobs:
            raise ValueError("campaign state job set does not match the immutable plan")
        return plan, state

    def _commit(self, state: CampaignState, event: dict[str, object]) -> None:
        write_json_atomic(self.state_path, state.to_dict())
        payload = {
            "at": utc_now(),
            "campaign_id": state.campaign_id,
            "plan_sha256": state.plan_sha256,
            "state_revision": state.revision,
            **event,
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def status(self) -> dict[str, object]:
        with self._exclusive():
            plan, state = self._load()
        counts = Counter(item.status.value for item in state.jobs)
        return {
            "campaign_id": state.campaign_id,
            "plan_sha256": state.plan_sha256,
            "state_revision": state.revision,
            "jobs": len(state.jobs),
            "counts": {status.value: counts[status.value] for status in JobStatus},
            "claim_scope": plan.claim_scope,
            "run_dir": str(self.run_dir),
        }

    def claim(
        self,
        worker_id: str,
        *,
        job_id: str | None = None,
        retry_rejected: bool = False,
    ) -> JobSpec | None:
        with self._exclusive():
            plan, state = self._load()
            jobs_by_id = {item.job_id: item for item in plan.jobs}
            if job_id is not None:
                job = jobs_by_id.get(job_id)
                if job is None:
                    raise ValueError(f"unknown job {job_id!r}")
                candidates = (job,)
            else:
                allowed = {JobStatus.PENDING}
                if retry_rejected:
                    allowed.add(JobStatus.REJECTED)
                available_ids = {
                    item.job_id for item in state.jobs if item.status in allowed
                }
                candidates = tuple(
                    item for item in plan.jobs if item.job_id in available_ids
                )
            if not candidates:
                return None
            job = candidates[0]
            updated = state.claim(job.job_id, worker_id)
            self._commit(
                updated,
                {"event": "job_claimed", "job_id": job.job_id, "worker_id": worker_id},
            )
            return job

    def submit(
        self,
        job_id: str,
        worker_id: str,
        artifact_manifest_uri: str,
        artifact_manifest_sha256: str,
    ) -> None:
        with self._exclusive():
            _, state = self._load()
            updated = state.submit(
                job_id,
                worker_id,
                artifact_manifest_uri,
                artifact_manifest_sha256,
            )
            self._commit(
                updated,
                {
                    "event": "job_submitted",
                    "job_id": job_id,
                    "worker_id": worker_id,
                    "artifact_manifest_uri": artifact_manifest_uri,
                    "artifact_manifest_sha256": artifact_manifest_sha256,
                },
            )

    def requeue(self, job_id: str, reason: str) -> None:
        with self._exclusive():
            _, state = self._load()
            current = next((item for item in state.jobs if item.job_id == job_id), None)
            if current is None:
                raise ValueError(f"unknown job {job_id!r}")
            updated = state.requeue(job_id, reason)
            self._commit(
                updated,
                {
                    "event": "job_requeued",
                    "job_id": job_id,
                    "previous_worker_id": current.worker_id,
                    "reason": reason.strip(),
                },
            )

    def audit(self, report: AuditReport) -> JobStatus:
        with self._exclusive():
            plan, state = self._load()
            job = next((item for item in plan.jobs if item.job_id == report.job_id), None)
            if job is None:
                raise ValueError(f"unknown job {report.job_id!r}")
            updated = state.audit(job, report)
            result = next(item for item in updated.jobs if item.job_id == report.job_id)
            self._commit(
                updated,
                {
                    "event": "job_audited",
                    "job_id": report.job_id,
                    "auditor": report.auditor,
                    "decision": result.status.value,
                    "diagnoses": list(result.diagnoses),
                },
            )
            return result.status
