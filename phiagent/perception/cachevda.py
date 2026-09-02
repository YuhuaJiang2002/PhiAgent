"""Lightweight subprocess adapter for CacheVDA-B-FP16 relative video depth.

The model framework remains in its own external virtual environment. Importing
this module does not import NumPy, PyTorch, OpenCV, xFormers, or FFmpeg bindings.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phiagent.rendering.wan_animate import (
    PreflightError,
    acquire_gpu_lease,
    query_gpus,
    select_gpu,
)


CACHEVDA_BASE_COMMIT = "4f5ae23172ba60fd7bc11ef671cca678842c7072"
CACHEVDA_CHECKPOINT_SHA256 = (
    "775e578e8f9431ec0496514aa466bd0a1f67c28d0f518267809f35a43c04329b"
)
CACHEVDA_CHECKPOINT_BYTES = 458_247_082
CACHEVDA_CORE_SHA256 = {
    "experiments/feature_cache/benchmark_dino_feature_cache.py": (
        "3236031e5f25b8f450213e732fb8e6ebea860b576f4d123f7d0794b6cccc4b13"
    ),
    "experiments/feature_cache/run_full_video_feature_cache.py": (
        "320ac3dcee045c4b548c73d2c52deb568a968a614c51376065331a4bed3bc630"
    ),
    "experiments/feature_cache/run_e2e_optimized.py": (
        "2ba4ce690a1eec04f8dfcbcde653984eadf499e36e550e23954f08c8c6b8b71a"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise PreflightError(f"missing {label}: {resolved}")
    return resolved


def _require_executable(path: Path, label: str) -> Path:
    """Validate an executable path without dereferencing a virtualenv symlink."""

    absolute = Path(os.path.abspath(path.expanduser()))
    if not absolute.is_file() or not os.access(absolute, os.X_OK):
        raise PreflightError(f"missing {label}: {absolute}")
    return absolute


def _git_state(repository: Path) -> dict[str, str]:
    state: dict[str, str] = {}
    for name, command in {
        "head": ["git", "rev-parse", "HEAD"],
        "status": ["git", "status", "--short"],
    }.items():
        completed = subprocess.run(
            command,
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        state[name] = completed.stdout.strip() if completed.returncode == 0 else "unavailable"
    return state


def validate_cachevda_timing(
    payload: object,
    *,
    expected_max_frames: int,
) -> dict[str, Any]:
    """Validate CacheVDA's stable timing-JSON success contract."""

    if not isinstance(payload, dict):
        raise RuntimeError("CacheVDA timing output must be a JSON object")
    if payload.get("status") != "completed":
        raise RuntimeError("CacheVDA timing status is not completed")
    video = payload.get("video")
    if not isinstance(video, dict):
        raise RuntimeError("CacheVDA timing output is missing video metadata")
    frame_count = int(video.get("frame_count", 0))
    if frame_count <= 0:
        raise RuntimeError("CacheVDA reported no output frames")
    if expected_max_frames > 0 and frame_count != expected_max_frames:
        raise RuntimeError(
            f"CacheVDA produced {frame_count} frames, expected {expected_max_frames}"
        )
    inference = payload.get("inference")
    if not isinstance(inference, dict) or int(inference.get("window_count", 0)) <= 0:
        raise RuntimeError("CacheVDA timing output is missing inference windows")
    if float(inference.get("peak_reserved_gib", 0.0)) <= 0:
        raise RuntimeError("CacheVDA timing output is missing CUDA memory evidence")
    return payload


@dataclass(frozen=True)
class CacheVDAConfig:
    repository: Path
    checkpoint: Path | None = None
    python_executable: Path | None = None
    gpu_index: int | None = None
    minimum_free_gpu_mib: int = 12 * 1024
    input_size: int = 518
    max_res: int = 1280
    warmup_windows: int = 1
    preprocess_workers: int = 8
    encode_batch_size: int = 32
    encoder: str = "h264_nvenc"
    log_every: int = 25

    def __post_init__(self) -> None:
        if self.minimum_free_gpu_mib <= 0:
            raise ValueError("CacheVDA minimum free GPU memory must be positive")
        if self.input_size <= 0 or self.max_res <= 0:
            raise ValueError("CacheVDA input size and maximum resolution must be positive")
        if self.warmup_windows < 0:
            raise ValueError("CacheVDA warmup windows cannot be negative")
        if self.preprocess_workers <= 0 or self.encode_batch_size <= 0:
            raise ValueError("CacheVDA worker and encode batch counts must be positive")
        if self.encoder not in {"h264_nvenc", "libx264"}:
            raise ValueError("CacheVDA encoder must be h264_nvenc or libx264")
        if self.log_every <= 0:
            raise ValueError("CacheVDA log interval must be positive")

    @property
    def python(self) -> Path:
        return self.python_executable or self.repository / ".venv" / "bin" / "python"

    @property
    def model_checkpoint(self) -> Path:
        return self.checkpoint or self.repository / "checkpoints" / "video_depth_anything_vitb.pth"

    @property
    def script(self) -> Path:
        return self.repository / "experiments" / "feature_cache" / "run_e2e_optimized.py"


@dataclass(frozen=True)
class CacheVDARequest:
    input_video: Path
    experiment_dir: Path
    max_frames: int = -1

    def __post_init__(self) -> None:
        if self.max_frames == 0 or self.max_frames < -1:
            raise ValueError("CacheVDA max_frames must be -1 or positive")


@dataclass(frozen=True)
class CacheVDAResult:
    experiment_dir: Path
    visualization_video: Path
    timing_json: Path
    result_json: Path
    frame_count: int
    output_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_dir": str(self.experiment_dir),
            "visualization_video": str(self.visualization_video),
            "timing_json": str(self.timing_json),
            "result_json": str(self.result_json),
            "frame_count": self.frame_count,
            "output_sha256": self.output_sha256,
        }


class CacheVDARunner:
    """Run the pinned CacheVDA CLI in its dedicated external environment."""

    def __init__(self, config: CacheVDAConfig) -> None:
        self.config = config

    def build_command(self, request: CacheVDARequest, output_dir: Path) -> list[str]:
        command = [
            str(self.config.python.expanduser().absolute()),
            str(self.config.script.expanduser().resolve()),
            "--input-video",
            str(request.input_video.expanduser().resolve()),
            "--checkpoint",
            str(self.config.model_checkpoint.expanduser().resolve()),
            "--output-dir",
            str(output_dir.expanduser().resolve()),
            "--input-size",
            str(self.config.input_size),
            "--max-res",
            str(self.config.max_res),
            "--max-frames",
            str(request.max_frames),
            "--warmup-windows",
            str(self.config.warmup_windows),
            "--preprocess-workers",
            str(self.config.preprocess_workers),
            "--encode-batch-size",
            str(self.config.encode_batch_size),
            "--encoder",
            self.config.encoder,
            "--log-every",
            str(self.config.log_every),
        ]
        return command

    def preflight(self) -> dict[str, Any]:
        repository = self.config.repository.expanduser().resolve()
        if not repository.is_dir():
            raise PreflightError(f"missing CacheVDA repository: {repository}")
        python = _require_executable(self.config.python, "CacheVDA Python")
        checkpoint = _require_file(self.config.model_checkpoint, "CacheVDA checkpoint")
        script = _require_file(self.config.script, "CacheVDA optimized CLI")
        git = _git_state(repository)
        if git["head"] != CACHEVDA_BASE_COMMIT:
            raise PreflightError(
                f"CacheVDA base commit is {git['head']}, expected {CACHEVDA_BASE_COMMIT}"
            )
        source_hashes = {}
        for relative, expected in CACHEVDA_CORE_SHA256.items():
            path = _require_file(repository / relative, f"CacheVDA source {relative}")
            observed = _sha256(path)
            if observed != expected:
                raise PreflightError(
                    f"CacheVDA source hash mismatch for {relative}: {observed} != {expected}"
                )
            source_hashes[relative] = observed
        if checkpoint.stat().st_size != CACHEVDA_CHECKPOINT_BYTES:
            raise PreflightError("CacheVDA checkpoint byte size mismatch")
        checkpoint_hash = _sha256(checkpoint)
        if checkpoint_hash != CACHEVDA_CHECKPOINT_SHA256:
            raise PreflightError("CacheVDA checkpoint SHA-256 mismatch")

        encoders = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if self.config.encoder not in encoders:
            raise PreflightError(f"FFmpeg does not provide {self.config.encoder}")
        gpus, inventory, processes = query_gpus()
        selected = select_gpu(gpus, self.config.gpu_index, self.config.minimum_free_gpu_mib)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
        probe = (
            "import json, torch, torchvision, xformers; "
            "print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,"
            "'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
            "'autocast_dtype':str(torch.get_autocast_dtype('cuda')),'xformers':xformers.__version__,"
            "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
        )
        runtime_process = subprocess.run(
            [str(python.absolute()), "-c", probe],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        runtime = json.loads(runtime_process.stdout.splitlines()[-1])
        if not runtime["cuda_available"] or runtime["autocast_dtype"] != "torch.float16":
            raise PreflightError(f"CacheVDA CUDA FP16 runtime is unavailable: {runtime}")
        return {
            "schema_version": "1.0.0",
            "repository": str(repository),
            "git": git,
            "base_commit": CACHEVDA_BASE_COMMIT,
            "source_hashes": source_hashes,
            "source_versioning": (
                "base commit plus exact hashes; CacheVDA experiment files are not yet tagged"
            ),
            "python": str(python.absolute()),
            "script": str(script),
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_license": "CC-BY-NC-4.0",
            "encoder": self.config.encoder,
            "runtime": runtime,
            "selected_gpu": asdict(selected),
            "gpu_inventory_raw": inventory,
            "gpu_processes_raw": processes,
        }

    @staticmethod
    def _probe_output_video(path: Path) -> dict[str, Any]:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "stream=codec_type,codec_name,width,height,avg_frame_rate,nb_read_frames",
                "-show_entries",
                "format=duration,size",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
        audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
        if len(video_streams) != 1 or audio_streams:
            raise RuntimeError("CacheVDA output must contain one video stream and no audio")
        stream = video_streams[0]
        if int(stream.get("nb_read_frames", 0)) <= 0:
            raise RuntimeError("CacheVDA output video does not decode")
        return payload

    def run(self, request: CacheVDARequest) -> CacheVDAResult:
        input_video = _require_file(request.input_video, "CacheVDA RGB input video")
        experiment = request.experiment_dir.expanduser().resolve()
        if experiment.exists():
            raise FileExistsError(f"CacheVDA experiment already exists: {experiment}")
        experiment.mkdir(parents=True)
        output_dir = experiment / "outputs"
        output_dir.mkdir()
        preflight = self.preflight()
        selected_gpu = preflight["selected_gpu"]
        command = self.build_command(request, output_dir)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["physical_index"])
        environment["PHIAGENT_PHYSICAL_GPU_INDEX"] = str(selected_gpu["physical_index"])
        config_path = experiment / "config.json"
        _write_json(
            config_path,
            {
                "schema_version": "1.0.0",
                "status": "STARTED",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "input_video": str(input_video),
                "input_sha256": _sha256(input_video),
                "max_frames": request.max_frames,
                "depth_semantics": "relative_affine_ambiguous",
                "visualization_is_not_numeric_depth": True,
                "command": command,
                "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
                "preflight": preflight,
                "phiagent_git": _git_state(Path(__file__).resolve().parents[2]),
            },
        )
        packages = subprocess.run(
            [str(self.config.python.expanduser().absolute()), "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
        (experiment / "packages.txt").write_text(packages.stdout)
        log_path = experiment / "inference.log"
        result_path = experiment / "result.json"
        started = time.monotonic()
        lease_path, lease = acquire_gpu_lease(int(selected_gpu["physical_index"]))
        try:
            leased_gpus, leased_inventory, leased_processes = query_gpus()
            leased_gpu = select_gpu(
                leased_gpus,
                int(selected_gpu["physical_index"]),
                self.config.minimum_free_gpu_mib,
            )
            _write_json(
                experiment / "gpu-lease.json",
                {
                    "physical_gpu": asdict(leased_gpu),
                    "lease": str(lease_path),
                    "inventory_raw": leased_inventory,
                    "processes_raw": leased_processes,
                },
            )
            with log_path.open("w") as log:
                completed = subprocess.run(
                    command,
                    cwd=self.config.repository.expanduser().resolve(),
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
        finally:
            lease.close()
        wall_seconds = time.monotonic() - started
        stem = input_video.stem
        output_video = output_dir / f"{stem}_vis.mp4"
        timing_path = output_dir / f"{stem}_timing.json"
        result_payload: dict[str, Any] = {
            "schema_version": "1.0.0",
            "status": "FAILED",
            "return_code": completed.returncode,
            "wall_seconds": wall_seconds,
            "experiment": str(experiment),
            "log": str(log_path),
            "claim_boundary": (
                "relative monocular depth visualization; not metric depth or RGB-D sensing"
            ),
        }
        if completed.returncode != 0:
            _write_json(result_path, result_payload)
            raise RuntimeError(
                f"CacheVDA exited with {completed.returncode}; inspect {log_path}"
            )
        timing_payload = validate_cachevda_timing(
            json.loads(_require_file(timing_path, "CacheVDA timing JSON").read_text()),
            expected_max_frames=request.max_frames,
        )
        output_video = _require_file(output_video, "CacheVDA visualization video")
        probe = self._probe_output_video(output_video)
        decoded_frames = int(probe["streams"][0]["nb_read_frames"])
        expected_frames = int(timing_payload["video"]["frame_count"])
        if decoded_frames != expected_frames:
            raise RuntimeError(
                f"CacheVDA output decodes {decoded_frames} frames, expected {expected_frames}"
            )
        output_hash = _sha256(output_video)
        result_payload.update(
            {
                "status": "WORKING",
                "visualization_video": str(output_video),
                "visualization_sha256": output_hash,
                "timing_json": str(timing_path),
                "decoded_video": probe,
                "frame_count": expected_frames,
                "relative_depth_finite": True,
                "numeric_depth_saved": False,
            }
        )
        _write_json(result_path, result_payload)
        return CacheVDAResult(
            experiment_dir=experiment,
            visualization_video=output_video,
            timing_json=timing_path,
            result_json=result_path,
            frame_count=expected_frames,
            output_sha256=output_hash,
        )


__all__ = [
    "CACHEVDA_BASE_COMMIT",
    "CACHEVDA_CHECKPOINT_SHA256",
    "CACHEVDA_CORE_SHA256",
    "CacheVDAConfig",
    "CacheVDARequest",
    "CacheVDAResult",
    "CacheVDARunner",
    "validate_cachevda_timing",
]
