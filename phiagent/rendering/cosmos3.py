"""Cosmos 3 video-transfer adapter for physically verified trajectory controls."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from phiagent.rendering.base import (
    TrajectoryConditionedRenderRequest,
    TrajectoryConditionedRenderResult,
)
from phiagent.rendering.wan_animate import (
    PreflightError,
    _assert_files,
    _probe_visual_media,
    _run_capture,
    _validate_input,
    _write_json,
    query_gpus,
    select_gpu,
)

COSMOS3_FRAMEWORK_COMMIT = "4155d61d14b14e05a8cafe2bd796d090fcb5f145"
COSMOS3_MODEL_ID = "nvidia/Cosmos3-Nano"
COSMOS3_MODEL_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
COSMOS3_ASPECT_RATIOS = {"16,9", "4,3", "1,1", "3,4", "9,16"}
WAN22_TI2V_MODEL_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"


@dataclass(frozen=True)
class Cosmos3Config:
    """Configuration for pinned Cosmos 3 Nano edge-controlled transfer."""

    framework_repo: Path
    checkpoint_dir: Path
    python_executable: Path | None = None
    gpu_index: int | None = None
    minimum_free_gpu_mib: int = 60 * 1024
    resolution: int = 480
    fps: int = 30
    num_steps: int = 50
    guidance: float = 3.0
    control_guidance: float = 1.5
    edge_threshold: str = "medium"
    guardrails: bool = True
    vision_only: bool = True
    use_torch_compile: bool = False
    hf_home: Path | None = None
    offline: bool = True

    def validate(self) -> None:
        if self.minimum_free_gpu_mib <= 0:
            raise ValueError("minimum_free_gpu_mib must be positive")
        if self.resolution not in {256, 480, 720}:
            raise ValueError("resolution must be 256, 480, or 720")
        if self.fps not in {10, 16, 24, 30}:
            raise ValueError("fps must be one of 10, 16, 24, or 30")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.guidance <= 0 or self.control_guidance <= 0:
            raise ValueError("guidance values must be positive")
        if self.edge_threshold not in {"low", "medium", "high"}:
            raise ValueError("edge_threshold must be low, medium, or high")

    @property
    def python(self) -> Path:
        return self.python_executable or self.framework_repo / ".venv" / "bin" / "python"

    @property
    def resolved_hf_home(self) -> Path:
        configured = self.hf_home or Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
        return configured.expanduser().resolve()

    @property
    def wan_vae_path(self) -> Path:
        return (
            self.resolved_hf_home
            / "hub"
            / "models--Wan-AI--Wan2.2-TI2V-5B"
            / "snapshots"
            / WAN22_TI2V_MODEL_REVISION
            / "Wan2.2_VAE.pth"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aspect_ratio(width: int, height: int) -> str:
    from math import gcd

    divisor = gcd(width, height)
    return f"{width // divisor},{height // divisor}"


def _prompt_metadata(prompt: str) -> dict[str, Any]:
    return {
        "subjects": [{"description": prompt.strip()}],
        "background_setting": "Preserve the verified simulation scene and object identities.",
        "cinematography": {"camera_motion": "Static", "framing": "Match the control video."},
        "style_medium": "Photorealistic robot manipulation video",
    }


def _free_local_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return str(server.getsockname()[1])


def _probe_control_video(path: Path) -> dict[str, Any]:
    if shutil.which("ffprobe") is None:
        raise PreflightError("ffprobe is required to validate the Cosmos control video")
    try:
        payload = json.loads(
            _run_capture(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate,nb_read_frames",
                    "-of",
                    "json",
                    str(path),
                ]
            )
        )
    except (json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        raise PreflightError(f"ffprobe could not inspect control video: {path}") from exc
    streams = payload.get("streams", [])
    if not streams:
        raise PreflightError(f"control video contains no visual stream: {path}")
    stream = streams[0]
    try:
        numerator, denominator = str(stream["avg_frame_rate"]).split("/", maxsplit=1)
        fps = float(numerator) / float(denominator)
        frame_count = int(stream["nb_read_frames"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise PreflightError(f"control video frame count/FPS is unavailable: {path}") from exc
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frame_count": frame_count,
    }


class Cosmos3TrajectoryRenderer:
    """Transform a verified simulation control video with Cosmos 3 Nano."""

    def __init__(self, config: Cosmos3Config) -> None:
        config.validate()
        self.config = config

    def preflight(self, select_cuda_device: bool = True) -> dict[str, Any]:
        """Validate the pinned framework and select an eligible physical GPU."""

        _assert_files(
            [
                self.config.framework_repo / "cosmos_framework" / "scripts" / "inference.py",
                self.config.python,
                self.config.checkpoint_dir / ".phiagent-model-revision",
                self.config.checkpoint_dir / "config.json",
            ],
            "Cosmos 3 framework/checkpoint files",
        )
        if self.config.vision_only:
            _assert_files([self.config.wan_vae_path], "Cosmos 3 vision tokenizer files")
        actual_model_revision = (
            self.config.checkpoint_dir / ".phiagent-model-revision"
        ).read_text().strip()
        if actual_model_revision != COSMOS3_MODEL_REVISION:
            raise PreflightError(
                f"Cosmos3-Nano checkpoint marker is {actual_model_revision!r}, "
                f"expected {COSMOS3_MODEL_REVISION}"
            )
        actual_commit = _run_capture(
            ["git", "rev-parse", "HEAD"], cwd=self.config.framework_repo
        )
        if actual_commit != COSMOS3_FRAMEWORK_COMMIT:
            raise PreflightError(
                f"Cosmos Framework checkout is {actual_commit}, "
                f"expected pinned {COSMOS3_FRAMEWORK_COMMIT}"
            )
        report: dict[str, Any] = {
            "framework_commit_expected": COSMOS3_FRAMEWORK_COMMIT,
            "framework_commit_actual": actual_commit,
            "model_id": COSMOS3_MODEL_ID,
            "model_revision": actual_model_revision,
            "checkpoint_path": str(self.config.checkpoint_dir),
            "python": str(self.config.python),
        }
        if select_cuda_device:
            gpus, inventory, processes = query_gpus()
            selected = select_gpu(gpus, self.config.gpu_index, self.config.minimum_free_gpu_mib)
            report.update(
                {
                    "gpu_inventory": [asdict(gpu) for gpu in gpus],
                    "gpu_inventory_raw": inventory,
                    "gpu_processes_raw": processes,
                    "selected_gpu": asdict(selected),
                }
            )
            report["python_runtime"] = self._probe_runtime(selected.physical_index)
        return report

    def _probe_runtime(self, physical_gpu_index: int) -> dict[str, Any]:
        code = (
            "import json, torch; import cosmos_framework; "
            "print(json.dumps({'torch':torch.__version__,"
            "'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
            "'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
        )
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_index)
        try:
            runtime = json.loads(
                _run_capture(
                    [str(self.config.python), "-c", code],
                    cwd=self.config.framework_repo,
                    env=environment,
                ).splitlines()[-1]
            )
        except (json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            raise PreflightError(f"Cosmos 3 Python environment probe failed: {exc}") from exc
        if not runtime["cuda_available"]:
            raise PreflightError(
                f"Cosmos 3 cannot use physical GPU {physical_gpu_index} "
                f"with {self.config.python}"
            )
        return runtime

    def build_spec(
        self,
        request: TrajectoryConditionedRenderRequest,
        control_video: Path,
        prompt_path: Path,
        name: str = "phiagent_verified_transfer",
    ) -> dict[str, Any]:
        frame_count = len(request.robot_trajectory.timestamps_s)
        if not 5 <= frame_count <= 300:
            raise PreflightError("Cosmos 3 requires between 5 and 300 aligned trajectory frames")
        expected_period = 1.0 / self.config.fps
        periods = [
            current - previous
            for previous, current in zip(
                request.robot_trajectory.timestamps_s,
                request.robot_trajectory.timestamps_s[1:],
            )
        ]
        if any(abs(period - expected_period) > 1e-4 for period in periods):
            raise PreflightError(
                f"trajectory timestamps must be uniformly sampled at {self.config.fps} FPS"
            )
        aspect_ratio = _aspect_ratio(
            request.camera_intrinsics.width, request.camera_intrinsics.height
        )
        if aspect_ratio not in COSMOS3_ASPECT_RATIOS:
            raise PreflightError(
                f"Cosmos 3 does not support aspect ratio {aspect_ratio}; "
                f"expected one of {sorted(COSMOS3_ASPECT_RATIOS)}"
            )
        return {
            "name": name,
            "model_mode": "video2video",
            "resolution": str(self.config.resolution),
            "aspect_ratio": aspect_ratio,
            "num_frames": frame_count,
            "fps": self.config.fps,
            "shift": 10.0,
            "num_steps": self.config.num_steps,
            "seed": request.seed,
            "num_video_frames_per_chunk": frame_count,
            "num_conditional_frames": 1,
            "num_first_chunk_conditional_frames": 0,
            "share_vision_temporal_positions": True,
            "negative_metadata_mode": "none",
            "negative_prompt_keep_metadata": False,
            "guidance": self.config.guidance,
            "control_guidance": self.config.control_guidance,
            "prompt_path": str(prompt_path),
            "vision_path": str(control_video),
            "edge": {
                "weight": 1.0,
                "preset_edge_threshold": self.config.edge_threshold,
            },
            "emphasize_control_in_prompt": True,
        }

    def build_command(self, spec_path: Path, output_dir: Path, seed: int) -> list[str]:
        command = [
            str(self.config.python),
            "-m",
            "cosmos_framework.scripts.inference",
            "--parallelism-preset=latency",
            "-i",
            str(spec_path),
            "-o",
            str(output_dir),
            "--checkpoint-path",
            str(self.config.checkpoint_dir),
            "--video-save-quality",
            "8",
            "--image_size",
            str(self.config.resolution),
            "--seed",
            str(seed),
            "--benchmark",
        ]
        command.extend(
            [
                "--no-use-torch-compile"
                if not self.config.use_torch_compile
                else "--use-torch-compile"
            ]
        )
        if not self.config.guardrails:
            command.append("--no-guardrails")
        return command

    def build_model_config(self) -> dict[str, Any]:
        config_path = self.config.checkpoint_dir / "config.json"
        payload = json.loads(config_path.read_text())
        if self.config.vision_only:
            model_config = payload["model"]["config"]
            model_config["sound_gen"] = False
            model_config["action_gen"] = False
            tokenizer = model_config["vlm_config"]["tokenizer"]
            tokenizer["config_variant"] = "hf"
            tokenizer["pretrained_model_name"] = str(self.config.checkpoint_dir)
        return payload

    @staticmethod
    def _new_experiment_dir(root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment = root / f"{timestamp}-{uuid.uuid4().hex[:8]}"
        experiment.mkdir()
        return experiment

    def _project_provenance(self) -> dict[str, Any]:
        project_root = Path(__file__).resolve().parents[2]
        provenance: dict[str, Any] = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "argv": sys.argv,
        }
        for key, command in {
            "git_commit": ["git", "rev-parse", "HEAD"],
            "git_status": ["git", "status", "--porcelain"],
            "package_versions": [str(self.config.python), "-m", "pip", "freeze"],
        }.items():
            try:
                provenance[key] = _run_capture(command, cwd=project_root)
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                provenance[key] = f"unavailable: {exc}"
        return provenance

    @staticmethod
    def _execute(
        command: Sequence[str], cwd: Path, log_path: Path, env: Mapping[str, str]
    ) -> None:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=dict(env),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
            return_code = process.wait()
        if return_code:
            raise RuntimeError(
                f"command failed with exit code {return_code}; inspect {log_path}: "
                + shlex.join(command)
            )

    def render(
        self, request: TrajectoryConditionedRenderRequest
    ) -> TrajectoryConditionedRenderResult:
        control_video = _validate_input(
            request.control_video, {".mp4", ".mov", ".mkv", ".webm"}, "control video"
        )
        _probe_visual_media(control_video, "control video")
        control_stream = _probe_control_video(control_video)
        expected_frames = len(request.robot_trajectory.timestamps_s)
        if control_stream["frame_count"] != expected_frames:
            raise PreflightError(
                f"control video has {control_stream['frame_count']} frames; "
                f"the trajectory has {expected_frames}"
            )
        if abs(control_stream["fps"] - self.config.fps) > 1e-3:
            raise PreflightError(
                f"control video is {control_stream['fps']:.6g} FPS; "
                f"the renderer is configured for {self.config.fps} FPS"
            )
        verification_record = _validate_input(
            request.verification_record, {".json"}, "verification record"
        )
        for asset in request.scene_assets:
            _validate_input(asset, {asset.suffix.lower()}, "scene asset")
        output = request.output.expanduser().resolve()
        if output.exists() and not request.overwrite:
            raise PreflightError(f"output already exists (pass overwrite=True to replace it): {output}")

        preflight = self.preflight(select_cuda_device=True)
        experiment = self._new_experiment_dir(request.experiment_root.expanduser().resolve())
        inputs_dir = experiment / "inputs"
        outputs_dir = experiment / "cosmos_output"
        inputs_dir.mkdir()
        outputs_dir.mkdir()
        metadata_path = experiment / "metadata.json"
        alignment_path = experiment / "alignment_report.json"
        prompt_path = inputs_dir / "prompt.json"
        spec_path = inputs_dir / "transfer.json"
        model_config_path = inputs_dir / "model_config.json"
        robot_path = inputs_dir / "robot_trajectory.json"
        prompt_path.write_text(json.dumps(_prompt_metadata(request.prompt), indent=2) + "\n")
        request.robot_trajectory.to_json(robot_path)
        object_paths: list[Path] = []
        for trajectory in request.object_trajectories:
            path = inputs_dir / f"object_trajectory_{trajectory.body_name}.json"
            trajectory.to_json(path)
            object_paths.append(path)
        camera_path = inputs_dir / "camera.json"
        _write_json(
            camera_path,
            {
                "intrinsics": request.camera_intrinsics.to_dict(),
                "camera_T_robot_base": request.camera_T_robot_base.to_dict(),
            },
        )
        copied_control = inputs_dir / f"control{control_video.suffix.lower()}"
        shutil.copy2(control_video, copied_control)
        copied_verification = inputs_dir / "verification.json"
        shutil.copy2(verification_record, copied_verification)
        spec = self.build_spec(request, copied_control, prompt_path)
        _write_json(spec_path, spec)
        _write_json(model_config_path, self.build_model_config())
        command = self.build_command(spec_path, outputs_dir, request.seed)
        command.extend(["--config-file", str(model_config_path)])
        generated_output = outputs_dir / spec["name"] / "vision.mp4"
        processed_control = outputs_dir / spec["name"] / "control_edge.mp4"

        selected_gpu = preflight["selected_gpu"]
        framework_bin = str(self.config.python.parent)
        execution_environment = {
            "CUDA_VISIBLE_DEVICES": str(selected_gpu["physical_index"]),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": _free_local_port(),
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
            "PYTHONHASHSEED": str(request.seed),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PATH": framework_bin + os.pathsep + os.environ.get("PATH", ""),
            "UV_DEFAULT_INDEX": "https://pypi.org/simple",
            "HF_HOME": str(self.config.resolved_hf_home),
        }
        if self.config.offline:
            execution_environment["HF_HUB_OFFLINE"] = "1"
        metadata: dict[str, Any] = {
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "renderer_config": asdict(self.config),
            "preflight": preflight,
            "control_stream": control_stream,
            "command": command,
            "execution_environment": execution_environment,
            "provenance": self._project_provenance(),
            "input_hashes": {
                "control_video": _sha256(copied_control),
                "verification_record": _sha256(copied_verification),
                "robot_trajectory": _sha256(robot_path),
                "object_trajectories": {path.name: _sha256(path) for path in object_paths},
                "camera": _sha256(camera_path),
                "scene_assets": {
                    str(asset.resolve()): _sha256(asset.resolve())
                    for asset in request.scene_assets
                },
            },
            "limitations": [
                "Edge control constrains image structure but does not guarantee exact 3D pose.",
                "Generated-video alignment must be measured before visual binding is accepted.",
            ],
        }
        _write_json(metadata_path, metadata)
        _write_json(
            alignment_path,
            {
                "status": "not_evaluated",
                "accepted": False,
                "reason": "generated output has not yet passed per-frame trajectory alignment",
            },
        )

        environment = os.environ.copy()
        environment.update(execution_environment)
        try:
            self._execute(
                command,
                self.config.framework_repo,
                experiment / "cosmos.log",
                environment,
            )
            _assert_files(
                [generated_output, processed_control],
                "Cosmos 3 output and processed control",
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(generated_output, output)
            from phiagent.rendering.alignment import StructuralAlignmentEvaluator

            alignment = StructuralAlignmentEvaluator().evaluate(
                processed_control,
                output,
                alignment_path,
            )
            metadata.update(
                {
                    "status": "succeeded",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "output": str(output),
                    "output_sha256": _sha256(output),
                    "processed_control_sha256": _sha256(processed_control),
                    "structural_alignment": alignment,
                }
            )
            _write_json(metadata_path, metadata)
        except Exception as exc:
            metadata.update(
                {
                    "status": "failed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "error": repr(exc),
                }
            )
            _write_json(metadata_path, metadata)
            raise
        return TrajectoryConditionedRenderResult(
            output=output,
            experiment_dir=experiment,
            metadata=metadata_path,
            alignment_report=alignment_path,
        )
