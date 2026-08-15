"""Audited state transitions for long-running data-engine jobs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Mapping

from phiagent.data_engine.planner import CampaignPlan, JobSpec


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AUDIT_PENDING = "audit_pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class EvidenceRef:
    uri: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("evidence uri must be non-empty")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("evidence sha256 must be lowercase hexadecimal")


@dataclass(frozen=True)
class AuditReport:
    job_id: str
    auditor: str
    independent: bool
    gates: Mapping[str, bool]
    evidence: tuple[EvidenceRef, ...]
    diagnostic_mean_score: float | None = None

    def __post_init__(self) -> None:
        if not self.job_id.strip() or not self.auditor.strip():
            raise ValueError("audit job_id and auditor must be non-empty")
        if type(self.independent) is not bool:
            raise ValueError("audit independent must be a boolean")
        if not self.gates or any(not key.strip() for key in self.gates):
            raise ValueError("audit gates must be non-empty")
        if any(type(value) is not bool for value in self.gates.values()):
            raise ValueError("audit gate values must be booleans")
        if not self.evidence:
            raise ValueError("audit requires hash-bound evidence")
        if self.diagnostic_mean_score is not None and (
            not math.isfinite(self.diagnostic_mean_score)
            or not 0 <= self.diagnostic_mean_score <= 1
        ):
            raise ValueError("diagnostic_mean_score must be finite and in [0, 1]")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AuditReport:
        return cls(
            job_id=str(payload["job_id"]),
            auditor=str(payload["auditor"]),
            independent=payload["independent"],
            gates={str(key): value for key, value in payload["gates"].items()},
            evidence=tuple(EvidenceRef(**item) for item in payload["evidence"]),
            diagnostic_mean_score=(
                float(payload["diagnostic_mean_score"])
                if payload.get("diagnostic_mean_score") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class JobState:
    job_id: str
    status: JobStatus
    attempts: int = 0
    worker_id: str | None = None
    artifact_manifest_uri: str | None = None
    artifact_manifest_sha256: str | None = None
    audit_history: tuple[dict[str, object], ...] = ()
    diagnoses: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JobState:
        return cls(
            job_id=str(payload["job_id"]),
            status=JobStatus(str(payload["status"])),
            attempts=int(payload.get("attempts", 0)),
            worker_id=(
                str(payload["worker_id"])
                if payload.get("worker_id") is not None
                else None
            ),
            artifact_manifest_uri=(
                str(payload["artifact_manifest_uri"])
                if payload.get("artifact_manifest_uri") is not None
                else None
            ),
            artifact_manifest_sha256=(
                str(payload["artifact_manifest_sha256"])
                if payload.get("artifact_manifest_sha256") is not None
                else None
            ),
            audit_history=tuple(dict(item) for item in payload.get("audit_history", ())),
            diagnoses=tuple(str(item) for item in payload.get("diagnoses", ())),
        )


@dataclass(frozen=True)
class CampaignState:
    campaign_id: str
    plan_sha256: str
    revision: int
    jobs: tuple[JobState, ...]

    @classmethod
    def from_plan(cls, plan: CampaignPlan) -> CampaignState:
        return cls(
            campaign_id=plan.campaign_id,
            plan_sha256=plan.plan_sha256,
            revision=0,
            jobs=tuple(
                JobState(job_id=job.job_id, status=JobStatus.PENDING) for job in plan.jobs
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "revision": self.revision,
            "jobs": [item.to_dict() for item in self.jobs],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CampaignState:
        return cls(
            campaign_id=str(payload["campaign_id"]),
            plan_sha256=str(payload["plan_sha256"]),
            revision=int(payload["revision"]),
            jobs=tuple(JobState.from_dict(item) for item in payload["jobs"]),
        )

    def _replace_job(self, updated: JobState) -> CampaignState:
        jobs = tuple(updated if item.job_id == updated.job_id else item for item in self.jobs)
        if jobs == self.jobs:
            raise ValueError(f"unknown job {updated.job_id!r}")
        return replace(self, revision=self.revision + 1, jobs=jobs)

    def claim(self, job_id: str, worker_id: str) -> CampaignState:
        current = next((item for item in self.jobs if item.job_id == job_id), None)
        if current is None:
            raise ValueError(f"unknown job {job_id!r}")
        if current.status not in {JobStatus.PENDING, JobStatus.REJECTED}:
            raise ValueError(f"job {job_id!r} cannot be claimed from {current.status.value}")
        if not worker_id.strip():
            raise ValueError("worker_id must be non-empty")
        return self._replace_job(
            replace(
                current,
                status=JobStatus.RUNNING,
                attempts=current.attempts + 1,
                worker_id=worker_id,
                artifact_manifest_uri=None,
                artifact_manifest_sha256=None,
                diagnoses=(),
            )
        )

    def submit(
        self,
        job_id: str,
        worker_id: str,
        artifact_manifest_uri: str,
        artifact_manifest_sha256: str,
    ) -> CampaignState:
        current = next((item for item in self.jobs if item.job_id == job_id), None)
        if current is None:
            raise ValueError(f"unknown job {job_id!r}")
        if current.status is not JobStatus.RUNNING or current.worker_id != worker_id:
            raise ValueError("only the active worker may submit a running job")
        if not artifact_manifest_uri.strip():
            raise ValueError("artifact_manifest_uri must be non-empty")
        EvidenceRef(artifact_manifest_uri, artifact_manifest_sha256)
        return self._replace_job(
            replace(
                current,
                status=JobStatus.AUDIT_PENDING,
                artifact_manifest_uri=artifact_manifest_uri,
                artifact_manifest_sha256=artifact_manifest_sha256,
            )
        )

    def requeue(self, job_id: str, reason: str) -> CampaignState:
        current = next((item for item in self.jobs if item.job_id == job_id), None)
        if current is None:
            raise ValueError(f"unknown job {job_id!r}")
        if current.status not in {
            JobStatus.RUNNING,
            JobStatus.AUDIT_PENDING,
            JobStatus.BLOCKED,
        }:
            raise ValueError(f"job {job_id!r} cannot be requeued from {current.status.value}")
        clean_reason = reason.strip()
        if not clean_reason:
            raise ValueError("requeue reason must be non-empty")
        return self._replace_job(
            replace(
                current,
                status=JobStatus.REJECTED,
                worker_id=None,
                diagnoses=current.diagnoses + (f"requeued:{clean_reason}",),
            )
        )

    def audit(self, job: JobSpec, report: AuditReport) -> CampaignState:
        current = next((item for item in self.jobs if item.job_id == job.job_id), None)
        if current is None or report.job_id != job.job_id:
            raise ValueError("audit job does not match campaign plan")
        if current.status is not JobStatus.AUDIT_PENDING:
            raise ValueError("job must be audit_pending before audit")
        missing = set(job.required_gates) - set(report.gates)
        failed = tuple(gate for gate in job.required_gates if report.gates.get(gate) is not True)
        if report.auditor == current.worker_id:
            diagnoses = ("executor_self_audit",)
            status = JobStatus.REJECTED
        elif not report.independent:
            diagnoses = ("audit_not_independent",)
            status = JobStatus.REJECTED
        elif missing:
            diagnoses = tuple(f"missing_gate:{gate}" for gate in sorted(missing))
            status = JobStatus.REJECTED
        elif failed:
            diagnoses = tuple(f"hard_gate_failed:{gate}" for gate in failed)
            status = JobStatus.REJECTED
        else:
            diagnoses = ()
            status = JobStatus.ACCEPTED
        history_item: dict[str, object] = {
            "auditor": report.auditor,
            "independent": report.independent,
            "gates": dict(report.gates),
            "evidence": [asdict(item) for item in report.evidence],
            "diagnostic_mean_score": report.diagnostic_mean_score,
            "decision": status.value,
        }
        return self._replace_job(
            replace(
                current,
                status=status,
                worker_id=None,
                audit_history=current.audit_history + (history_item,),
                diagnoses=diagnoses,
            )
        )
