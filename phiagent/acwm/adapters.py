"""Optional external adapters for released action-conditioned world models.

No model framework is imported here.  GPU frameworks and checkpoints live in
isolated external checkouts and are reached only through the explicit runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from phiagent.acwm.schema import ACWMCase, ActionRepresentation
from phiagent.rendering.minimax_h3 import (
    DIFFSYNTH_H3_COMMIT,
    MINIMAX_H3_MODELSCOPE_ID,
    MINIMAX_H3_NF4_MODEL_ID,
    MINIMAX_H3_NF4_REVISION,
    MINIMAX_H3_NF4_SHA256,
    MINIMAX_H3_PROCESSOR_REVISION,
    MINIMAX_H3_PROCESSOR_SHA256,
    file_sha256,
    h3_checkpoint_files,
)
from phiagent.rendering.wan_animate import PreflightError, query_gpus, select_gpu

OSCAR_REPOSITORY_COMMIT = "4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb"
OSCAR_MODEL_REVISION = "c9781ffa7dd8556d862d7d9f338a2ea008a58ca6"
OSCAR_COSMOS_REASON_REVISION = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"
OSCAR_WAN_VAE_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
BWM_REPOSITORY_COMMIT = "44acfd1b06f35f365f02f7bb2fc5da6beafcd6bc"
BWM_MODEL_REVISION = "738a8d3c008e637b8b1b18d5e98a82f6de9c04aa"
BWM_MODEL_SHA256 = "75f863b9474d6e74934db45bb85728fef0adece3d123c667b78349bdade9c7f3"
BWM_MODEL_BYTES = 10_051_484_872
BWM_BASE_MODEL_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
KINEMA4D_REPOSITORY_COMMIT = "716e80249376cb2843af41188a832d56a2d8d78d"
KINEMA4D_MODEL_REVISION = "0c52ee34ee464e9a568e84945e431f62106c4270"
FLOWWAM_REPOSITORY_COMMIT = "f06fa46042e97738c6619c868f1097be6749d48d"
FLOWWAM_MODEL_REVISION = "1e68f76cecfb2caa973abfb24fca92cbc5312a6e"
FLOWWAM_MODEL_SHA256 = "e211e32b6b79b293f7dec1a70794a69c3c1bf922483c06aef3c5f6d5c3be96c4"
FLOWWAM_MODEL_BYTES = 10_137_267_208
FLOWWAM_TOKENIZER_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_head(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise PreflightError(f"cannot inspect external repository {repo}: {exc}") from exc


def _source_revision(repository: Path) -> str:
    """Read a Git HEAD or an exact-revision marker from a pinned source archive."""

    if (repository / ".git").is_dir():
        return _git_head(repository)
    marker = repository / ".phiagent-source-revision"
    if marker.is_file():
        return marker.read_text().strip()
    raise PreflightError(
        f"external repository lacks both .git and .phiagent-source-revision: {repository}"
    )


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise PreflightError(f"missing {label}: {resolved}")
    return resolved


def _require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise PreflightError(f"missing {label}: {resolved}")
    return resolved


def _executable_path(path: Path) -> Path:
    """Return an absolute executable path without dereferencing venv symlinks."""

    return Path(os.path.abspath(path.expanduser()))


def _nvidia_library_paths(python: Path) -> tuple[str, ...]:
    """Discover wheel-provided CUDA libraries visible to an isolated interpreter."""

    script = (
        "from pathlib import Path; import os, sys; "
        "roots=[]; "
        "[roots.extend(str(p.resolve()) for p in (Path(x)/'nvidia').glob('*/lib') "
        "if p.is_dir()) for x in sys.path if x]; "
        "print(os.pathsep.join(dict.fromkeys(roots)))"
    )
    completed = subprocess.run(
        [str(_executable_path(python)), "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return tuple(path for path in completed.stdout.strip().split(os.pathsep) if path)


@dataclass(frozen=True)
class BackendSupport:
    backend: str
    supported: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ACWMRenderRequest:
    case: ACWMCase
    output: Path
    experiment_dir: Path
    seed: int = 42
    num_inference_steps: int = 35
    guidance_scale: float = 6.0

    def __post_init__(self) -> None:
        if self.output.suffix.lower() != ".mp4":
            raise ValueError("AC-WM output must be an .mp4 file")
        if self.seed < 0 or self.num_inference_steps <= 0 or self.guidance_scale <= 0:
            raise ValueError("seed, inference steps, and guidance must be positive")


@dataclass(frozen=True)
class ACWMRenderResult:
    backend: str
    case_id: str
    output: Path
    metadata: Path
    experiment_dir: Path


class ACWMVideoRenderer(Protocol):
    name: str

    def supports(self, case: ACWMCase) -> BackendSupport:
        """Report whether the case has the backend's native action inputs."""

    def render_batch(self, requests: Sequence[ACWMRenderRequest]) -> tuple[ACWMRenderResult, ...]:
        """Render one or more cases while allowing a single model load."""


def _pair_backend_results(
    requests: Sequence[ACWMRenderRequest],
    result_payload: list[dict[str, Any]],
) -> tuple[tuple[ACWMRenderRequest, dict[str, Any]], ...]:
    """Pair ordered backend results without collapsing same-case candidates."""

    if len(result_payload) != len(requests):
        raise RuntimeError("backend returned the wrong number of ordered results")
    pairs = []
    for request, item in zip(requests, result_payload):
        if str(item.get("case_id")) != request.case.case_id:
            raise RuntimeError(
                "backend result order mismatch: "
                f"expected {request.case.case_id}, got {item.get('case_id')}"
            )
        pairs.append((request, item))
    return tuple(pairs)


@dataclass(frozen=True)
class OSCARConfig:
    repository: Path
    checkpoint_dir: Path
    cosmos_reason_path: Path | None = None
    wan_vae_path: Path | None = None
    offline: bool = False
    python_executable: Path | None = None
    gpu_index: int | None = None
    minimum_free_gpu_mib: int = 24 * 1024
    height: int = 480
    width: int = 640
    num_frames: int = 81
    fps: float = 15.0

    @property
    def python(self) -> Path:
        return self.python_executable or self.repository / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class BWMConfig:
    repository: Path
    base_model_dir: Path
    checkpoint_path: Path
    action_stats: Path
    python_executable: Path | None = None
    config_path: Path | None = None
    gpu_index: int | None = None
    minimum_free_gpu_mib: int = 32 * 1024
    output_fps: int = 24

    @property
    def python(self) -> Path:
        return self.python_executable or self.repository / ".venv" / "bin" / "python"

    def __post_init__(self) -> None:
        if self.output_fps <= 0:
            raise ValueError("BWM output FPS must be positive")

    @property
    def resolved_config(self) -> Path:
        return self.config_path or self.repository / "configs" / "infer" / "infer.yaml"


@dataclass(frozen=True)
class Kinema4DConfig:
    repository: Path
    base_transformer: Path
    lora_path: Path
    dataset_root: Path
    episode_list: Path
    python_executable: Path | None = None
    gpu_index: int | None = None
    minimum_free_gpu_mib: int = 72 * 1024
    mode: str = "xyzrgb"

    @property
    def python(self) -> Path:
        return self.python_executable or self.repository / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class MiniMaxH3Config:
    repository: Path
    model_base_path: Path
    python_executable: Path | None = None
    gpu_index: int | None = None
    minimum_free_gpu_mib: int = 54 * 1024
    model_variant: str = "ref2va-nf4"
    height: int = 480
    width: int = 832
    num_frames: int = 124
    fps: float = 24.0
    steps: int = 20
    reference_image_short_edge: int = 768
    reference_video_short_edge: int = 480
    vram_reserve_gib: float = 8.0

    def __post_init__(self) -> None:
        if self.model_variant != "ref2va-nf4":
            raise ValueError("the action-intent harness requires the H3 Ref2VA partition")
        if self.width <= 0 or self.height <= 0 or self.width % 32 or self.height % 32:
            raise ValueError("H3 width and height must be positive multiples of 32")
        if self.fps != 24.0:
            raise ValueError("the released MiniMax-H3 pipeline requires 24 FPS")
        if self.num_frames < 5 or (self.num_frames - 5) % 17:
            raise ValueError("H3 num_frames must satisfy num_frames = 17n + 5")
        if self.steps <= 0:
            raise ValueError("H3 inference steps must be positive")
        if self.reference_image_short_edge <= 0 or self.reference_video_short_edge <= 0:
            raise ValueError("H3 reference short edges must be positive")
        if self.vram_reserve_gib <= 0:
            raise ValueError("H3 VRAM reserve must be positive")

    @property
    def python(self) -> Path:
        return self.python_executable or self.repository / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class FlowWAMConfig:
    repository: Path
    base_model_root: Path
    checkpoint_path: Path
    python_executable: Path | None = None
    gpu_index: int | None = None
    minimum_free_gpu_mib: int = 60 * 1024
    output_fps: int = 24

    @property
    def python(self) -> Path:
        return self.python_executable or self.repository / ".venv" / "bin" / "python"

    def __post_init__(self) -> None:
        if self.output_fps <= 0:
            raise ValueError("FlowWAM output FPS must be positive")


class _ExternalBatchAdapter:
    name: str
    expected_commit: str
    expected_model_revision: str

    def __init__(self, *, project_root: Path | None = None) -> None:
        self.project_root = (
            (project_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
        )

    @property
    def repository(self) -> Path:
        raise NotImplementedError

    @property
    def python(self) -> Path:
        raise NotImplementedError

    @property
    def gpu_index(self) -> int | None:
        raise NotImplementedError

    @property
    def minimum_free_gpu_mib(self) -> int:
        raise NotImplementedError

    def _model_preflight(self) -> dict[str, Any]:
        raise NotImplementedError

    def _runner_arguments(self) -> list[str]:
        raise NotImplementedError

    def _extra_environment(self) -> dict[str, str]:
        return {}

    def preflight(self, *, select_cuda_device: bool = True) -> dict[str, Any]:
        repo = _require_dir(self.repository, f"{self.name} repository")
        _require_file(self.python, f"{self.name} Python")
        head = _source_revision(repo)
        if head != self.expected_commit:
            raise PreflightError(f"{self.name} checkout is {head}, expected {self.expected_commit}")
        report: dict[str, Any] = {
            "backend": self.name,
            "repository": str(repo),
            "repository_commit": head,
            "model_revision_expected": self.expected_model_revision,
            "python": str(_executable_path(self.python)),
            "model": self._model_preflight(),
            "nvidia_library_paths": list(_nvidia_library_paths(self.python)),
        }
        if select_cuda_device:
            gpus, inventory, processes = query_gpus()
            selected = select_gpu(gpus, self.gpu_index, self.minimum_free_gpu_mib)
            report.update(
                {
                    "gpu_inventory": [asdict(gpu) for gpu in gpus],
                    "gpu_inventory_raw": inventory,
                    "gpu_processes_raw": processes,
                    "selected_gpu": asdict(selected),
                }
            )
        return report

    def render_batch(self, requests: Sequence[ACWMRenderRequest]) -> tuple[ACWMRenderResult, ...]:
        if not requests:
            raise ValueError("AC-WM render batch cannot be empty")
        unsupported = [self.supports(request.case) for request in requests]
        failed = [report for report in unsupported if not report.supported]
        if failed:
            details = "; ".join(
                f"{report.backend}: {', '.join(report.reasons)}" for report in failed
            )
            raise ValueError(f"backend input contract rejected the batch: {details}")
        experiment_dirs = {request.experiment_dir.expanduser().resolve() for request in requests}
        if len(experiment_dirs) != 1:
            raise ValueError("one backend batch must share a single experiment directory")
        experiment = next(iter(experiment_dirs)) / "backend-runs" / self.name
        if experiment.exists():
            raise FileExistsError(f"AC-WM backend run already exists: {experiment}")
        experiment.mkdir(parents=True)
        preflight = self.preflight()
        selected_gpu = preflight["selected_gpu"]
        batch_path = experiment / "requests.json"
        result_path = experiment / "results.json"
        payload: list[dict[str, Any]] = []
        for request in requests:
            condition_path = experiment / "conditions" / f"{request.case.case_id}.json"
            request.case.action.to_json(condition_path)
            output = request.output.expanduser().resolve()
            if output.exists():
                raise FileExistsError(f"AC-WM output already exists: {output}")
            payload.append(
                {
                    "case_id": request.case.case_id,
                    "first_frame": str(request.case.first_frame.expanduser().resolve()),
                    "source_video": str(request.case.source_video.expanduser().resolve()),
                    "condition": str(condition_path),
                    "prompt": request.case.prompt,
                    "auxiliary_inputs": {
                        key: str(path.expanduser().resolve())
                        for key, path in request.case.auxiliary_inputs
                    },
                    "output": str(output),
                    "seed": request.seed,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                }
            )
        _write_json(batch_path, payload)
        command = [
            str(_executable_path(self.python)),
            str(self.project_root / "scripts" / "run_acwm_backend.py"),
            self.name,
            "--request-manifest",
            str(batch_path),
            "--result-manifest",
            str(result_path),
            *self._runner_arguments(),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["physical_index"])
        nvidia_library_paths = [str(path) for path in preflight["nvidia_library_paths"]]
        if environment.get("LD_LIBRARY_PATH"):
            nvidia_library_paths.append(environment["LD_LIBRARY_PATH"])
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(nvidia_library_paths)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(self.repository.expanduser().resolve()), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        environment.update(self._extra_environment())
        log_path = experiment / "inference.log"
        started_at = datetime.now(timezone.utc).isoformat()
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=self.repository.expanduser().resolve(),
                env=environment,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        run_manifest = experiment / "manifest.json"
        _write_json(
            run_manifest,
            {
                "schema_version": "1.0.0",
                "status": "completed" if completed.returncode == 0 else "failed",
                "backend": self.name,
                "created_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "command": command,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
                "preflight": preflight,
                "request_manifest": str(batch_path),
                "request_manifest_sha256": _sha256(batch_path),
                "result_manifest": str(result_path),
                "log": str(log_path),
                "returncode": completed.returncode,
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{self.name} inference failed with exit {completed.returncode}; inspect {log_path}"
            )
        result_payload = json.loads(_require_file(result_path, "backend result").read_text())
        if not isinstance(result_payload, list) or len(result_payload) != len(requests):
            raise RuntimeError(f"{self.name} returned an invalid result manifest")
        results: list[ACWMRenderResult] = []
        for request, item in _pair_backend_results(requests, result_payload):
            output = _require_file(Path(str(item["output"])), "generated AC-WM video")
            metadata = _require_file(Path(str(item["metadata"])), "generated video metadata")
            results.append(
                ACWMRenderResult(
                    backend=self.name,
                    case_id=request.case.case_id,
                    output=output,
                    metadata=metadata,
                    experiment_dir=experiment,
                )
            )
        return tuple(results)


class OSCARRenderer(_ExternalBatchAdapter):
    name = "oscar"
    expected_commit = OSCAR_REPOSITORY_COMMIT
    expected_model_revision = OSCAR_MODEL_REVISION

    def __init__(self, config: OSCARConfig, *, project_root: Path | None = None) -> None:
        super().__init__(project_root=project_root)
        self.config = config

    @property
    def repository(self) -> Path:
        return self.config.repository

    @property
    def python(self) -> Path:
        return self.config.python

    @property
    def gpu_index(self) -> int | None:
        return self.config.gpu_index

    @property
    def minimum_free_gpu_mib(self) -> int:
        return self.config.minimum_free_gpu_mib

    def supports(self, case: ACWMCase) -> BackendSupport:
        reasons: list[str] = []
        if case.action.representation is not ActionRepresentation.KINEMATIC_SKELETON_2D:
            reasons.append("requires kinematic_skeleton_2d")
        if case.action.visual_condition is None:
            reasons.append("requires a skeleton video")
        return BackendSupport(self.name, not reasons, tuple(reasons))

    def _model_preflight(self) -> dict[str, Any]:
        checkpoint = _require_dir(self.config.checkpoint_dir, "OSCAR checkpoint")
        _require_dir(checkpoint / "model", "OSCAR distributed checkpoint")
        marker = _require_file(checkpoint / ".phiagent-model-revision", "OSCAR revision marker")
        revision = marker.read_text().strip()
        if revision != self.expected_model_revision:
            raise PreflightError(
                f"OSCAR model revision is {revision}, expected {self.expected_model_revision}"
            )
        runtime: dict[str, Any] = {}
        if self.config.cosmos_reason_path is not None:
            cosmos = _require_dir(
                self.config.cosmos_reason_path,
                "OSCAR Cosmos-Reason1-7B checkpoint",
            )
            marker = _require_file(
                cosmos / ".phiagent-model-revision",
                "OSCAR Cosmos-Reason1-7B revision marker",
            )
            cosmos_revision = marker.read_text().strip()
            if cosmos_revision != OSCAR_COSMOS_REASON_REVISION:
                raise PreflightError(
                    "OSCAR Cosmos-Reason1-7B revision is "
                    f"{cosmos_revision}, expected {OSCAR_COSMOS_REASON_REVISION}"
                )
            runtime["cosmos_reason"] = str(cosmos)
            runtime["cosmos_reason_revision"] = cosmos_revision
        if self.config.wan_vae_path is not None:
            vae = _require_file(self.config.wan_vae_path, "OSCAR Wan2.1 VAE")
            marker = _require_file(
                vae.parent / ".phiagent-wan-vae-revision",
                "OSCAR Wan2.1 VAE revision marker",
            )
            vae_revision = marker.read_text().strip()
            if vae_revision != OSCAR_WAN_VAE_REVISION:
                raise PreflightError(
                    f"OSCAR Wan2.1 VAE revision is {vae_revision}, "
                    f"expected {OSCAR_WAN_VAE_REVISION}"
                )
            runtime["wan_vae"] = str(vae)
            runtime["wan_vae_revision"] = vae_revision
        return {
            "checkpoint": str(checkpoint),
            "revision": revision,
            "runtime_dependencies": runtime,
            "offline": self.config.offline,
        }

    def _extra_environment(self) -> dict[str, str]:
        environment: dict[str, str] = {}
        if self.config.cosmos_reason_path is not None:
            environment["COSMOS_REASON_PATH"] = str(
                self.config.cosmos_reason_path.expanduser().resolve()
            )
        if self.config.wan_vae_path is not None:
            environment["WAN_VAE_PATH"] = str(self.config.wan_vae_path.expanduser().resolve())
        if self.config.offline:
            environment["HF_HUB_OFFLINE"] = "1"
            environment["TRANSFORMERS_OFFLINE"] = "1"
        return environment

    def _runner_arguments(self) -> list[str]:
        return [
            "--repository",
            str(self.repository.expanduser().resolve()),
            "--checkpoint",
            str(self.config.checkpoint_dir.expanduser().resolve()),
            "--height",
            str(self.config.height),
            "--width",
            str(self.config.width),
            "--num-frames",
            str(self.config.num_frames),
            "--fps",
            str(self.config.fps),
        ]


class MiniMaxH3Renderer(_ExternalBatchAdapter):
    """Use H3 Ref2VA as a proposal renderer around explicit camera controls."""

    name = "minimax-h3"
    expected_commit = DIFFSYNTH_H3_COMMIT
    expected_model_revision = MINIMAX_H3_NF4_MODEL_ID

    def __init__(
        self,
        config: MiniMaxH3Config,
        *,
        project_root: Path | None = None,
    ) -> None:
        super().__init__(project_root=project_root)
        self.config = config

    @property
    def repository(self) -> Path:
        return self.config.repository

    @property
    def python(self) -> Path:
        return self.config.python

    @property
    def gpu_index(self) -> int | None:
        return self.config.gpu_index

    @property
    def minimum_free_gpu_mib(self) -> int:
        return self.config.minimum_free_gpu_mib

    def supports(self, case: ACWMCase) -> BackendSupport:
        reasons: list[str] = []
        if case.action.representation is not ActionRepresentation.CAMERA_PIXEL_CONTROL_VIDEO:
            reasons.append("requires camera_pixel_control_video")
        if case.action.visual_condition is None:
            reasons.append("requires a frame-explicit action-control video")
        if len(case.action.timestamps_s) != self.config.num_frames:
            reasons.append(f"requires exactly {self.config.num_frames} action frames")
        else:
            try:
                fps = case.action.fps
            except ValueError as exc:
                reasons.append(str(exc))
            else:
                if not math.isclose(fps, self.config.fps, abs_tol=1e-4):
                    reasons.append(f"requires {self.config.fps:g} FPS action timestamps")
        if "embodiment_reference" not in case.assets:
            reasons.append("requires an embodiment_reference auxiliary image")
        return BackendSupport(self.name, not reasons, tuple(reasons))

    def _model_preflight(self) -> dict[str, Any]:
        model_base = _require_dir(self.config.model_base_path, "DiffSynth model base")
        model_root = model_base / MINIMAX_H3_NF4_MODEL_ID
        model_revision = _require_file(
            model_root / ".phiagent-model-revision",
            "MiniMax-H3 NF4 revision marker",
        ).read_text().strip()
        if model_revision != MINIMAX_H3_NF4_REVISION:
            raise PreflightError(
                f"MiniMax-H3 NF4 revision is {model_revision}, "
                f"expected {MINIMAX_H3_NF4_REVISION}"
            )
        processor_root = model_base / MINIMAX_H3_MODELSCOPE_ID / "Ref2VA" / "processor"
        processor_revision = _require_file(
            processor_root / ".phiagent-model-revision",
            "MiniMax-H3 processor revision marker",
        ).read_text().strip()
        if processor_revision != MINIMAX_H3_PROCESSOR_REVISION:
            raise PreflightError(
                f"MiniMax-H3 processor revision is {processor_revision}, "
                f"expected {MINIMAX_H3_PROCESSOR_REVISION}"
            )
        expected_hashes = {
            **{model_root / name: digest for name, digest in MINIMAX_H3_NF4_SHA256.items()},
            **{
                processor_root / name: digest
                for name, digest in MINIMAX_H3_PROCESSOR_SHA256.items()
            },
        }
        records = []
        for path in h3_checkpoint_files(
            model_base,
            model_variant=self.config.model_variant,
        ):
            resolved = _require_file(path, "MiniMax-H3 checkpoint")
            actual = file_sha256(resolved)
            expected = expected_hashes.get(resolved)
            if expected is None or actual != expected:
                raise PreflightError(
                    f"MiniMax-H3 checkpoint hash mismatch for {resolved}: "
                    f"{actual} != {expected}"
                )
            records.append(
                {
                    "path": str(resolved),
                    "bytes": resolved.stat().st_size,
                    "sha256": actual,
                }
            )
        return {
            "weights": MINIMAX_H3_NF4_MODEL_ID,
            "weights_revision": model_revision,
            "processor": MINIMAX_H3_MODELSCOPE_ID,
            "processor_revision": processor_revision,
            "variant": self.config.model_variant,
            "files": records,
        }

    def _extra_environment(self) -> dict[str, str]:
        return {
            "DIFFSYNTH_MODEL_BASE_PATH": str(
                self.config.model_base_path.expanduser().resolve()
            ),
            "DIFFSYNTH_SKIP_DOWNLOAD": "True",
            "HF_HUB_OFFLINE": "1",
            "MODELSCOPE_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }

    def _runner_arguments(self) -> list[str]:
        return [
            "--repository",
            str(self.repository.expanduser().resolve()),
            "--checkpoint",
            str(self.config.model_base_path.expanduser().resolve()),
            "--height",
            str(self.config.height),
            "--width",
            str(self.config.width),
            "--num-frames",
            str(self.config.num_frames),
            "--fps",
            str(self.config.fps),
            "--h3-model-variant",
            self.config.model_variant,
            "--h3-steps",
            str(self.config.steps),
            "--h3-reference-image-short-edge",
            str(self.config.reference_image_short_edge),
            "--h3-reference-video-short-edge",
            str(self.config.reference_video_short_edge),
            "--h3-vram-reserve-gib",
            str(self.config.vram_reserve_gib),
        ]


class BWMRenderer(_ExternalBatchAdapter):
    name = "bwm"
    expected_commit = BWM_REPOSITORY_COMMIT
    expected_model_revision = BWM_MODEL_REVISION

    def __init__(self, config: BWMConfig, *, project_root: Path | None = None) -> None:
        super().__init__(project_root=project_root)
        self.config = config

    @property
    def repository(self) -> Path:
        return self.config.repository

    @property
    def python(self) -> Path:
        return self.config.python

    @property
    def gpu_index(self) -> int | None:
        return self.config.gpu_index

    @property
    def minimum_free_gpu_mib(self) -> int:
        return self.config.minimum_free_gpu_mib

    def supports(self, case: ACWMCase) -> BackendSupport:
        allowed = {
            ActionRepresentation.EEF_ABSOLUTE,
            ActionRepresentation.EEF_DELTA,
            ActionRepresentation.JOINT_ABSOLUTE,
            ActionRepresentation.JOINT_DELTA,
        }
        reasons: list[str] = []
        if case.action.representation not in allowed:
            reasons.append("requires a robot-base EEF or joint action sequence")
        if len(case.action.channels) != 14:
            reasons.append("public BWM checkpoint requires exactly 14 action channels")
        return BackendSupport(self.name, not reasons, tuple(reasons))

    def _model_preflight(self) -> dict[str, Any]:
        base = _require_dir(self.config.base_model_dir, "BWM Wan2.2 base model")
        base_marker = _require_file(
            base / ".phiagent-model-revision", "BWM Wan2.2 base-model revision marker"
        )
        base_revision = base_marker.read_text().strip()
        if base_revision != BWM_BASE_MODEL_REVISION:
            raise PreflightError(
                f"BWM Wan2.2 base-model revision is {base_revision}, "
                f"expected {BWM_BASE_MODEL_REVISION}"
            )
        checkpoint = _require_file(self.config.checkpoint_path, "BWM checkpoint")
        stats = _require_file(self.config.action_stats, "BWM action statistics")
        config = _require_file(self.config.resolved_config, "BWM inference config")
        marker = _require_file(
            checkpoint.parent / ".phiagent-model-revision", "BWM revision marker"
        )
        verification_path = _require_file(
            checkpoint.parent / ".phiagent-model-verification.json",
            "BWM checkpoint verification",
        )
        verification = json.loads(verification_path.read_text())
        if (
            verification.get("sha256") != BWM_MODEL_SHA256
            or int(verification.get("bytes", -1)) != BWM_MODEL_BYTES
            or checkpoint.stat().st_size != BWM_MODEL_BYTES
        ):
            raise PreflightError("BWM checkpoint verification manifest is invalid")
        revision = marker.read_text().strip()
        if revision != self.expected_model_revision:
            raise PreflightError(
                f"BWM model revision is {revision}, expected {self.expected_model_revision}"
            )
        return {
            "base_model": str(base),
            "base_model_revision": base_revision,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": BWM_MODEL_SHA256,
            "action_stats": str(stats),
            "config": str(config),
            "revision": revision,
        }

    def _runner_arguments(self) -> list[str]:
        return [
            "--repository",
            str(self.repository.expanduser().resolve()),
            "--base-model",
            str(self.config.base_model_dir.expanduser().resolve()),
            "--checkpoint",
            str(self.config.checkpoint_path.expanduser().resolve()),
            "--action-stats",
            str(self.config.action_stats.expanduser().resolve()),
            "--config",
            str(self.config.resolved_config.expanduser().resolve()),
            "--fps",
            str(self.config.output_fps),
        ]


class Kinema4DRenderer(_ExternalBatchAdapter):
    name = "kinema4d"
    expected_commit = KINEMA4D_REPOSITORY_COMMIT
    expected_model_revision = KINEMA4D_MODEL_REVISION

    def __init__(self, config: Kinema4DConfig, *, project_root: Path | None = None) -> None:
        super().__init__(project_root=project_root)
        self.config = config

    @property
    def repository(self) -> Path:
        return self.config.repository

    @property
    def python(self) -> Path:
        return self.config.python

    @property
    def gpu_index(self) -> int | None:
        return self.config.gpu_index

    @property
    def minimum_free_gpu_mib(self) -> int:
        return self.config.minimum_free_gpu_mib

    def supports(self, case: ACWMCase) -> BackendSupport:
        reasons: list[str] = []
        if case.action.representation is not ActionRepresentation.ROBOT_POINTMAP:
            reasons.append("requires a robot_pointmap action condition")
        for key in ("robot_urdf", "camera_calibration"):
            if key not in case.assets:
                reasons.append(f"requires {key}")
        return BackendSupport(self.name, not reasons, tuple(reasons))

    def _model_preflight(self) -> dict[str, Any]:
        transformer = _require_dir(self.config.base_transformer, "Kinema4D base transformer")
        lora = _require_dir(self.config.lora_path, "Kinema4D checkpoint")
        dataset = _require_dir(self.config.dataset_root, "Kinema4D prepared dataset")
        episodes = _require_file(self.config.episode_list, "Kinema4D episode list")
        marker = _require_file(lora / ".phiagent-model-revision", "Kinema4D revision marker")
        revision = marker.read_text().strip()
        if revision != self.expected_model_revision:
            raise PreflightError(
                f"Kinema4D model revision is {revision}, expected {self.expected_model_revision}"
            )
        return {
            "base_transformer": str(transformer),
            "lora": str(lora),
            "dataset_root": str(dataset),
            "episode_list": str(episodes),
            "revision": revision,
        }

    def _runner_arguments(self) -> list[str]:
        return [
            "--repository",
            str(self.repository.expanduser().resolve()),
            "--base-model",
            str(self.config.base_transformer.expanduser().resolve()),
            "--checkpoint",
            str(self.config.lora_path.expanduser().resolve()),
            "--dataset-root",
            str(self.config.dataset_root.expanduser().resolve()),
            "--episode-list",
            str(self.config.episode_list.expanduser().resolve()),
            "--kinema-mode",
            self.config.mode,
        ]


class FlowWAMRenderer(_ExternalBatchAdapter):
    """Dense robot-flow renderer for geometry-grounded action conditioning."""

    name = "flowwam"
    expected_commit = FLOWWAM_REPOSITORY_COMMIT
    expected_model_revision = FLOWWAM_MODEL_REVISION

    def __init__(self, config: FlowWAMConfig, *, project_root: Path | None = None) -> None:
        super().__init__(project_root=project_root)
        self.config = config

    @property
    def repository(self) -> Path:
        return self.config.repository

    @property
    def python(self) -> Path:
        return self.config.python

    @property
    def gpu_index(self) -> int | None:
        return self.config.gpu_index

    @property
    def minimum_free_gpu_mib(self) -> int:
        return self.config.minimum_free_gpu_mib

    def supports(self, case: ACWMCase) -> BackendSupport:
        reasons: list[str] = []
        if case.action.representation is not ActionRepresentation.ROBOT_FLOW:
            reasons.append("requires a camera-frame robot_flow condition")
        if case.action.visual_condition is None:
            reasons.append("requires an encoded robot-only optical-flow video")
        for key in ("robot_urdf", "camera_calibration", "flow_provenance"):
            if key not in case.assets:
                reasons.append(f"requires {key}")
        return BackendSupport(self.name, not reasons, tuple(reasons))

    def _model_preflight(self) -> dict[str, Any]:
        base = _require_dir(self.config.base_model_root, "FlowWAM Wan model root")
        base_manifest_path = _require_file(
            base / ".phiagent-flowwam-base-revisions.json",
            "FlowWAM base-model revision manifest",
        )
        base_manifest = json.loads(base_manifest_path.read_text())
        if (
            base_manifest.get("wan22_ti2v_revision") != BWM_BASE_MODEL_REVISION
            or base_manifest.get("wan21_tokenizer_revision")
            != FLOWWAM_TOKENIZER_REVISION
        ):
            raise PreflightError("FlowWAM base-model revision manifest is invalid")
        checkpoint = _require_file(self.config.checkpoint_path, "FlowWAM checkpoint")
        marker = _require_file(
            checkpoint.parent / ".phiagent-model-revision",
            "FlowWAM model revision marker",
        )
        verification_path = _require_file(
            checkpoint.parent / ".phiagent-model-verification.json",
            "FlowWAM checkpoint verification",
        )
        verification = json.loads(verification_path.read_text())
        if (
            verification.get("sha256") != FLOWWAM_MODEL_SHA256
            or int(verification.get("bytes", -1)) != FLOWWAM_MODEL_BYTES
            or checkpoint.stat().st_size != FLOWWAM_MODEL_BYTES
        ):
            raise PreflightError("FlowWAM checkpoint verification manifest is invalid")
        revision = marker.read_text().strip()
        if revision != self.expected_model_revision:
            raise PreflightError(
                f"FlowWAM model revision is {revision}, expected {self.expected_model_revision}"
            )
        return {
            "base_model_root": str(base),
            "base_model_revisions": base_manifest,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": FLOWWAM_MODEL_SHA256,
            "revision": revision,
            "stage": "worldarena_stage1_without_seedvr_refiner",
        }

    def _runner_arguments(self) -> list[str]:
        return [
            "--repository",
            str(self.repository.expanduser().resolve()),
            "--base-model",
            str(self.config.base_model_root.expanduser().resolve()),
            "--checkpoint",
            str(self.config.checkpoint_path.expanduser().resolve()),
            "--fps",
            str(self.config.output_fps),
        ]
