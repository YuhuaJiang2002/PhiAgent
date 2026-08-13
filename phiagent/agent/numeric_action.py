"""Agent and durable job manager for exact numeric action video generation."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import socket
import subprocess
import sys
import threading
import traceback
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from phiagent.acwm.adapters import (
    ACWMRenderRequest,
    ACWMRenderResult,
    ACWMVideoRenderer,
)
from phiagent.acwm.numeric import (
    BWM_ACTION_FPS,
    BWM_ACTION_FRAMES,
    CompiledNumericAction,
    NumericActionStatistics,
    compile_bwm_eef_payload,
    numeric_action_channel_specs,
)
from phiagent.acwm.robotwin import BWM_EEF_CHANNELS
from phiagent.acwm.schema import ACWMActionCondition, ACWMCase, ActionRepresentation
from phiagent.learning.experience import ExperienceRecord, append_experience

_JOB_ID = re.compile(r"[0-9a-f]{32}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"{label} does not exist or is empty: {resolved}")
    return resolved


def _captured(command: list[str], *, cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _default_provenance(project_root: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "command": sys.argv,
        "package_versions": sorted(
            f"{name}=={distribution.version}"
            for distribution in distributions()
            if (name := distribution.metadata.get("Name"))
        ),
    }
    for key, command in {
        "git_head": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--short"],
    }.items():
        try:
            result[key] = _captured(command, cwd=project_root)
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            result[key] = f"unavailable: {type(exc).__name__}: {exc}"
    return result


@dataclass(frozen=True)
class NumericActionVideoRequest:
    case_id: str
    first_frame: Path
    source_video: Path
    action: ACWMActionCondition
    prompt: str
    experiment_root: Path
    seed: int = 20260812
    num_inference_steps: int = 20
    guidance_scale: float = 6.0

    def __post_init__(self) -> None:
        if self.action.representation is not ActionRepresentation.EEF_ABSOLUTE:
            raise ValueError("numeric action video generation currently requires eef_absolute")
        numeric_action_channel_specs(self.action.channels)
        if len(self.action.values) != BWM_ACTION_FRAMES:
            raise ValueError(f"numeric BWM actions require exactly {BWM_ACTION_FRAMES} frames")
        if not math.isfinite(self.action.fps) or self.action.fps <= 0:
            raise ValueError("numeric BWM action sample rate must be positive")
        if self.seed < 0 or self.num_inference_steps <= 0 or self.guidance_scale <= 0:
            raise ValueError("generation seed, steps, and guidance must be positive")


@dataclass(frozen=True)
class NumericActionVideoOutcome:
    status: str
    experiment_dir: Path
    output: Path
    metadata: Path
    action: Path
    manifest: Path


class NumericActionVideoAgent:
    """Generate one BWM video while keeping acceptance explicitly deferred."""

    def __init__(
        self,
        renderer: ACWMVideoRenderer,
        *,
        project_root: Path | None = None,
        provenance_provider: Callable[[Path], Mapping[str, object]] | None = None,
    ) -> None:
        self.renderer = renderer
        self.project_root = (
            project_root or Path(__file__).resolve().parents[2]
        ).expanduser().resolve()
        self.provenance_provider = provenance_provider or _default_provenance

    @staticmethod
    def _new_experiment(root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        experiment = root / f"{stamp}-{uuid.uuid4().hex[:8]}"
        experiment.mkdir()
        return experiment

    def run(self, request: NumericActionVideoRequest) -> NumericActionVideoOutcome:
        case = ACWMCase(
            case_id=request.case_id,
            first_frame=_require_file(request.first_frame, "numeric action first frame"),
            source_video=_require_file(request.source_video, "numeric action source video"),
            action=request.action,
            prompt=request.prompt,
        )
        support = self.renderer.supports(case)
        if not support.supported:
            raise ValueError(
                f"{self.renderer.name} rejected the numeric action: {', '.join(support.reasons)}"
            )

        experiment = self._new_experiment(request.experiment_root.expanduser().resolve())
        action_path = experiment / "input" / "action.json"
        request.action.to_json(action_path)
        output = experiment / "prediction.mp4"
        manifest_path = experiment / "manifest.json"
        manifest: dict[str, object] = {
            "schema_version": "1.0.0",
            "status": "running",
            "honest_status": "PARTIAL",
            "method": "exact_numeric_action_conditioned_video_generation",
            "backend": self.renderer.name,
            "case_id": request.case_id,
            "provenance": dict(self.provenance_provider(self.project_root)),
            "input": {
                "first_frame": str(case.first_frame.resolve()),
                "first_frame_sha256": _sha256(case.first_frame),
                "source_video": str(case.source_video.resolve()),
                "source_video_sha256": _sha256(case.source_video),
                "action": str(action_path),
                "action_sha256": _sha256(action_path),
                "coordinate_frame": request.action.coordinate_frame,
                "representation": request.action.representation.value,
                "frames": len(request.action.values),
                "action_sample_hz": request.action.fps,
                "channels": list(request.action.channels),
                "prompt": request.prompt,
            },
            "generation": {
                "seed": request.seed,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale_requested": request.guidance_scale,
            },
            "acceptance": {
                "status": "pending_evaluation_and_human_review",
                "generated_video_is_not_accepted_execution_evidence": True,
            },
        }
        _write_json(manifest_path, manifest)
        try:
            results = self.renderer.render_batch(
                (
                    ACWMRenderRequest(
                        case=case,
                        output=output,
                        experiment_dir=experiment,
                        seed=request.seed,
                        num_inference_steps=request.num_inference_steps,
                        guidance_scale=request.guidance_scale,
                    ),
                )
            )
            if len(results) != 1:
                raise RuntimeError("numeric action renderer returned the wrong result count")
            result: ACWMRenderResult = results[0]
            rendered = _require_file(result.output, "generated numeric-action video")
            metadata = _require_file(result.metadata, "generated numeric-action metadata")
            manifest.update(
                {
                    "status": "generated_pending_review",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "output": {
                        "video": str(rendered),
                        "video_sha256": _sha256(rendered),
                        "metadata": str(metadata),
                        "metadata_sha256": _sha256(metadata),
                    },
                }
            )
            _write_json(manifest_path, manifest)
            return NumericActionVideoOutcome(
                status="generated_pending_review",
                experiment_dir=experiment,
                output=rendered,
                metadata=metadata,
                action=action_path,
                manifest=manifest_path,
            )
        except Exception as exc:
            manifest.update(
                {
                    "status": "failed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            _write_json(manifest_path, manifest)
            raise


@dataclass(frozen=True)
class NumericActionScene:
    first_frame: Path
    source_video: Path
    coordinate_frame: str
    default_condition: ACWMActionCondition | None = None
    action_channels: tuple[str, ...] = BWM_EEF_CHANNELS
    action_sample_hz: float = BWM_ACTION_FPS
    action_stats: NumericActionStatistics | None = None
    initial_state_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "first_frame", _require_file(self.first_frame, "first frame"))
        object.__setattr__(
            self, "source_video", _require_file(self.source_video, "source video")
        )
        if not self.coordinate_frame.startswith("robot_base:"):
            raise ValueError("numeric action scene requires a named robot_base frame")
        numeric_action_channel_specs(self.action_channels)
        if not math.isfinite(self.action_sample_hz) or self.action_sample_hz <= 0:
            raise ValueError("numeric action sample rate must be finite and positive")
        if (
            isinstance(self.initial_state_tolerance, bool)
            or not isinstance(self.initial_state_tolerance, (int, float))
            or not math.isfinite(self.initial_state_tolerance)
            or self.initial_state_tolerance < 0
        ):
            raise ValueError("initial_state_tolerance must be finite and non-negative")
        if self.default_condition is not None:
            if self.default_condition.coordinate_frame != self.coordinate_frame:
                raise ValueError("default condition frame does not match the configured scene")
            if self.default_condition.channels != self.action_channels:
                raise ValueError("default condition channels do not match the configured scene")
            if abs(self.default_condition.fps - self.action_sample_hz) > 1e-6:
                raise ValueError("default condition sample rate does not match the scene")
            NumericActionVideoRequest(
                case_id="default-condition-check",
                first_frame=self.first_frame,
                source_video=self.source_video,
                action=self.default_condition,
                prompt="Validate the configured default numeric action.",
                experiment_root=Path("."),
            )
        if self.action_stats is not None:
            if self.action_stats.coordinate_frame != self.coordinate_frame:
                raise ValueError("action statistics frame does not match the configured scene")
            if self.action_stats.channels != self.action_channels:
                raise ValueError("action statistics channels do not match the configured scene")

    def validate_initial_state(self, action: ACWMActionCondition) -> None:
        if self.default_condition is None:
            return
        errors = tuple(
            abs(actual - expected)
            for actual, expected in zip(
                action.values[0], self.default_condition.values[0]
            )
        )
        if max(errors) > self.initial_state_tolerance:
            channel = max(range(len(errors)), key=errors.__getitem__)
            raise ValueError(
                "numeric action frame 0 does not match the configured first-frame "
                f"state at {self.action_channels[channel]}: error {errors[channel]:.9g} "
                f"> tolerance {self.initial_state_tolerance:.9g}"
            )


class NumericActionJobManager:
    """Persist, execute, and publish numeric action jobs without fake fallbacks."""

    def __init__(
        self,
        agent: NumericActionVideoAgent,
        *,
        scene: NumericActionScene,
        jobs_root: Path,
        experiment_root: Path,
        ledger_path: Path,
        seed: int = 20260812,
        num_inference_steps: int = 20,
        guidance_scale: float = 6.0,
        maximum_queued_jobs: int = 8,
    ) -> None:
        if maximum_queued_jobs < 1:
            raise ValueError("maximum_queued_jobs must be positive")
        if seed < 0 or num_inference_steps <= 0 or guidance_scale <= 0:
            raise ValueError("generation seed, steps, and guidance must be positive")
        self.agent = agent
        self.scene = scene
        self.jobs_root = jobs_root.expanduser().resolve()
        self.experiment_root = experiment_root.expanduser().resolve()
        self.ledger_path = ledger_path.expanduser().resolve()
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.maximum_queued_jobs = maximum_queued_jobs
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.experiment_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="numeric-acwm")
        self._closed = False

    def capabilities(self) -> dict[str, object]:
        default = self.scene.default_condition
        specs = numeric_action_channel_specs(self.scene.action_channels)
        stats = self.scene.action_stats
        return {
            "schema_version": "1.0.0",
            "backend": self.agent.renderer.name,
            "submission_enabled": not self._closed,
            "action_contract": {
                "representation": ActionRepresentation.EEF_ABSOLUTE.value,
                "coordinate_frame": self.scene.coordinate_frame,
                "frames": BWM_ACTION_FRAMES,
                "action_sample_hz": self.scene.action_sample_hz,
                "duration_s": (BWM_ACTION_FRAMES - 1) / self.scene.action_sample_hz,
                "output_fps": getattr(
                    getattr(self.agent.renderer, "config", None), "output_fps", 24
                ),
                "interpolation": (
                    "piecewise_linear_position_slerp_quaternion"
                    if any(spec.quantity == "quaternion" for spec in specs)
                    else "channel_wise_piecewise_linear"
                ),
                "channels": [
                    {
                        **spec.to_dict(),
                        "training_min": stats.minimum[index] if stats else None,
                        "training_max": stats.maximum[index] if stats else None,
                        "training_p01": stats.p01[index] if stats else None,
                        "training_p99": stats.p99[index] if stats else None,
                    }
                    for index, spec in enumerate(specs)
                ],
                "default_start_values": list(default.values[0]) if default else None,
                "default_end_values": list(default.values[-1]) if default else None,
                "initial_state_locked": default is not None,
                "initial_state_tolerance": self.scene.initial_state_tolerance,
                "action_statistics": str(stats.path) if stats else None,
            },
            "acceptance_boundary": (
                "A completed job is a generated prediction pending action-adherence "
                "evaluation and human review; it is not robot-execution evidence."
            ),
        }

    def submit(self, payload: Mapping[str, Any]) -> dict[str, object]:
        with self._lock:
            if self._closed:
                raise RuntimeError("numeric action job manager is closed")
            active = sum(
                job["status"] in {"queued", "running"} for job in self._jobs.values()
            )
            if active >= self.maximum_queued_jobs:
                raise RuntimeError("numeric action queue is full")

        declared_frame = payload.get("coordinate_frame")
        if not isinstance(declared_frame, str) or not declared_frame.strip():
            raise ValueError("numeric action request requires coordinate_frame")
        if declared_frame != self.scene.coordinate_frame:
            raise ValueError(
                f"numeric action coordinate_frame is {declared_frame!r}, "
                f"but this scene requires {self.scene.coordinate_frame!r}"
            )
        action_payload = {
            key: value for key, value in payload.items() if key != "coordinate_frame"
        }
        job_id = uuid.uuid4().hex
        compiled: CompiledNumericAction = compile_bwm_eef_payload(
            action_payload,
            label=f"numeric-{job_id[:12]}",
            coordinate_frame=self.scene.coordinate_frame,
            channels=self.scene.action_channels,
            fps=self.scene.action_sample_hz,
        )
        self.scene.validate_initial_state(compiled.condition)
        statistics_summary = (
            self.scene.action_stats.validate(compiled.condition)
            if self.scene.action_stats is not None
            else None
        )
        created_at = datetime.now(timezone.utc).isoformat()
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        action_path = job_dir / "action.json"
        compiled.condition.to_json(action_path)
        request_path = job_dir / "request.json"
        _write_json(
            request_path,
            {
                "schema_version": "1.0.0",
                "instruction": compiled.condition.instruction,
                "prompt": compiled.prompt,
                "source_mode": compiled.source_mode,
                "summary": compiled.summary,
                "action_statistics": statistics_summary,
                "action": str(action_path),
                "action_sha256": _sha256(action_path),
            },
        )
        job = {
            "schema_version": "1.0.0",
            "job_id": job_id,
            "status": "queued",
            "detail": "The exact 14-D action contract is queued for BWM generation.",
            "created_at": created_at,
            "updated_at": created_at,
            "request_path": str(request_path),
            "action_path": str(action_path),
            "action_summary": compiled.summary,
            "action_statistics": statistics_summary,
            "video_url": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._persist(job)
            self._futures[job_id] = self._executor.submit(self._execute, job_id)
        return self.get(job_id) or {}

    def _persist(self, job: Mapping[str, Any]) -> None:
        _write_json(self.jobs_root / str(job["job_id"]) / "job.json", dict(job))

    def _update(self, job_id: str, **values: object) -> dict[str, Any]:
        with self._lock:
            job = self._jobs[job_id]
            job.update(values)
            job["updated_at"] = datetime.now(timezone.utc).isoformat()
            snapshot = dict(job)
            self._persist(snapshot)
        return snapshot

    def _execute(self, job_id: str) -> None:
        job = self._update(
            job_id,
            status="running",
            detail="BWM is consuming the exact frame-aligned 14-D EEF action.",
        )
        request_payload = json.loads(Path(job["request_path"]).read_text())
        action = ACWMActionCondition.from_json(Path(job["action_path"]))
        try:
            outcome = self.agent.run(
                NumericActionVideoRequest(
                    case_id=action.label,
                    first_frame=self.scene.first_frame,
                    source_video=self.scene.source_video,
                    action=action,
                    prompt=str(request_payload["prompt"]),
                    experiment_root=self.experiment_root,
                    seed=self.seed,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                )
            )
            append_experience(
                self.ledger_path,
                ExperienceRecord(
                    record_id=(
                        f"{datetime.now(timezone.utc):%Y-%m-%d}."
                        f"numeric-action-video-{job_id[:16]}"
                    ),
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    status="PARTIAL",
                    scope="exact numeric BWM action-conditioned video job",
                    summary=(
                        "BWM returned a non-empty MP4 for one explicit 57-frame, "
                        "14-D robot-base EEF action."
                    ),
                    evidence=(
                        f"Action contract: {outcome.action}",
                        f"Generation manifest: {outcome.manifest}",
                        f"Generated video: {outcome.output}",
                    ),
                    lessons=(
                        "Exact numeric model input and visual acceptance must remain separate.",
                    ),
                    limitations=(
                        "The generated prediction has not passed action-adherence or human review.",
                    ),
                    next_actions=(
                        "Run pose/action evaluation and record an explicit human review before acceptance.",
                    ),
                    run_dir=str(outcome.experiment_dir),
                    tags=("acwm", "bwm", "numeric-action", "generated-video"),
                ),
            )
            self._update(
                job_id,
                status="generated_pending_review",
                detail=(
                    "Video generation completed. The result is available for playback but "
                    "remains pending action-adherence evaluation and human review."
                ),
                completed_at=datetime.now(timezone.utc).isoformat(),
                experiment_dir=str(outcome.experiment_dir),
                manifest_path=str(outcome.manifest),
                output_path=str(outcome.output),
                metadata_path=str(outcome.metadata),
                video_url=f"/api/numeric-jobs/{job_id}/video",
            )
        except Exception as exc:
            failure = self._update(
                job_id,
                status="failed",
                detail=f"Numeric action generation failed: {type(exc).__name__}: {exc}",
                completed_at=datetime.now(timezone.utc).isoformat(),
                error={"type": type(exc).__name__, "message": str(exc)},
                traceback=traceback.format_exc(),
            )
            try:
                append_experience(
                    self.ledger_path,
                    ExperienceRecord(
                        record_id=(
                            f"{datetime.now(timezone.utc):%Y-%m-%d}."
                            f"numeric-action-video-{job_id[:16]}-failed"
                        ),
                        recorded_at=datetime.now(timezone.utc).isoformat(),
                        status="BLOCKED",
                        scope="exact numeric BWM action-conditioned video job",
                        summary="The submitted numeric action job produced no publishable result.",
                        evidence=(f"Job record: {self.jobs_root / job_id / 'job.json'}",),
                        lessons=(
                            "Generation failures must stay visible instead of falling back to a preset video.",
                        ),
                        limitations=(str(exc)[:500],),
                        next_actions=(
                            "Inspect the persisted job and backend logs, then retry in a new job.",
                        ),
                        run_dir=failure.get("experiment_dir"),
                        tags=("acwm", "bwm", "numeric-action", "failed"),
                    ),
                )
            except (OSError, ValueError) as ledger_exc:
                self._update(
                    job_id,
                    ledger_error=f"{type(ledger_exc).__name__}: {ledger_exc}",
                )
            raise

    def _load_job(self, job_id: str) -> dict[str, Any] | None:
        if not _JOB_ID.fullmatch(job_id):
            return None
        with self._lock:
            current = self._jobs.get(job_id)
            if current is not None:
                return dict(current)
        path = self.jobs_root / job_id / "job.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or payload.get("job_id") != job_id:
            raise ValueError(f"invalid persisted numeric action job: {path}")
        return payload

    def get(self, job_id: str) -> dict[str, object] | None:
        job = self._load_job(job_id)
        if job is None:
            return None
        public_keys = {
            "schema_version",
            "job_id",
            "status",
            "detail",
            "created_at",
            "updated_at",
            "completed_at",
            "action_summary",
            "video_url",
            "error",
        }
        return {key: value for key, value in job.items() if key in public_keys}

    def action_payload(self, job_id: str) -> dict[str, object] | None:
        job = self._load_job(job_id)
        if job is None:
            return None
        payload = json.loads(Path(job["action_path"]).read_text())
        if not isinstance(payload, dict):
            raise ValueError("persisted numeric action is not a JSON object")
        return payload

    def video_path(self, job_id: str) -> Path | None:
        job = self._load_job(job_id)
        if job is None or job.get("status") != "generated_pending_review":
            return None
        value = job.get("output_path")
        if not isinstance(value, str):
            raise ValueError("completed numeric action job is missing output_path")
        output = _require_file(Path(value), "numeric action job video")
        if not output.is_relative_to(self.experiment_root):
            raise ValueError("numeric action output lies outside the experiment root")
        return output

    def wait(self, job_id: str, *, timeout: float = 30.0) -> dict[str, object]:
        with self._lock:
            future = self._futures.get(job_id)
        if future is None:
            raise ValueError(f"unknown in-memory numeric action job: {job_id}")
        future.result(timeout=timeout)
        result = self.get(job_id)
        if result is None:
            raise RuntimeError(f"numeric action job disappeared: {job_id}")
        return result

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)
