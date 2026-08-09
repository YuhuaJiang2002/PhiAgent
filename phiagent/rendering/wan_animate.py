"""Native Wan2.2-Animate adapter with strict preflight and provenance capture."""

from __future__ import annotations

import csv
import fcntl
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
from typing import Any, Iterable, Mapping, Sequence, TextIO

from phiagent.evaluation.object_instance import (
    NormalizedROI,
    ObjectTrackerConfig,
    RGBFrames,
    decode_video,
    preserve_source_object,
    route_object_preservation,
)
from phiagent.rendering.base import VisualTransferRequest, VisualTransferResult

WAN22_COMMIT = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
WAN22_MODEL_ID = "Wan-AI/Wan2.2-Animate-14B"
WAN22_MODEL_REVISION = "cb93a225fbaf1ca100f54e79da8f994995b689b3"
WAN22_MODELSCOPE_REVISION = "bdcd76afebe1932ecb69916dd14ca255780f1d30"
SAM2_COMMIT = "0e78a118995e66bb27d78518c4bd9a3e95b4e266"
CUDA_12_4_RUNTIME_VERSIONS = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "torchaudio": "2.6.0",
    "diffusers": "0.36.0",
    "transformers": "4.51.3",
    "peft": "0.17.1",
    "moviepy": "2.2.1",
    "librosa": "0.11.0",
    "accelerate": "1.5.2",
    "onnxruntime-gpu": "1.20.2",
    "flash-attn": "2.7.4.post1",
}
CUDA_12_8_RUNTIME_VERSIONS = {
    **CUDA_12_4_RUNTIME_VERSIONS,
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "torchaudio": "2.7.1",
    "flash-attn": "2.8.3",
}
SUPPORTED_RUNTIME_PROFILES = {
    "cuda12.4-torch2.6": ("12.4", CUDA_12_4_RUNTIME_VERSIONS),
    "cuda12.8-torch2.7-blackwell": ("12.8", CUDA_12_8_RUNTIME_VERSIONS),
}


class PreflightError(RuntimeError):
    """Raised when a render would be unsafe or cannot run successfully."""


@dataclass(frozen=True)
class GPUInfo:
    physical_index: int
    name: str
    total_mib: int
    used_mib: int
    free_mib: int


@dataclass(frozen=True)
class WanAnimateConfig:
    """Configuration for the pinned native Wan2.2-Animate backend."""

    wan_repo: Path
    checkpoint_dir: Path
    sam2_repo: Path | None = None
    python_executable: Path = Path(sys.executable)
    gpu_index: int | None = None
    minimum_free_gpu_mib: int = 60 * 1024
    resolution_width: int = 1280
    resolution_height: int = 720
    fps: int = 30
    frame_num: int = 77
    infer_frames: int = 80
    reference_frames: int = 1
    mode: str = "animation"
    retarget: bool = True
    use_flux: bool = False
    use_relighting_lora: bool = True
    offload_model: bool = True
    t5_cpu: bool = False
    object_roi: tuple[float, float, float, float] | None = None

    def validate(self) -> None:
        if self.frame_num < 1 or (self.frame_num - 1) % 4:
            raise ValueError("frame_num must be positive and satisfy frame_num = 4n + 1")
        if self.infer_frames < 4 or self.infer_frames % 4:
            raise ValueError("infer_frames must be positive and divisible by 4")
        if self.resolution_width <= 0 or self.resolution_height <= 0:
            raise ValueError("resolution dimensions must be positive")
        if self.fps == 0 or self.fps < -1:
            raise ValueError("fps must be -1 (source FPS) or a positive integer")
        if self.reference_frames not in {1, 5}:
            raise ValueError("reference_frames must be 1 or 5, as recommended upstream")
        if self.mode not in {"animation", "replacement"}:
            raise ValueError("mode must be animation or replacement")
        if self.mode == "replacement" and (self.retarget or self.use_flux):
            raise ValueError("replacement mode does not support pose retargeting or FLUX")
        if self.mode == "replacement" and self.object_roi is None:
            raise ValueError("replacement mode requires an object ROI for mask auditing")
        if self.object_roi is not None:
            NormalizedROI(*self.object_roi)
        if self.use_flux and not self.retarget:
            raise ValueError("use_flux requires retarget=True")
        if self.minimum_free_gpu_mib <= 0:
            raise ValueError("minimum_free_gpu_mib must be positive")


def parse_nvidia_smi_csv(output: str) -> list[GPUInfo]:
    """Parse the exact CSV query used by query_gpus."""

    gpus: list[GPUInfo] = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) != 5:
            raise PreflightError(f"unexpected nvidia-smi row with {len(row)} columns: {row!r}")
        try:
            values = [int(field.strip().removesuffix(" MiB")) for field in row[2:]]
            gpus.append(
                GPUInfo(
                    physical_index=int(row[0].strip()),
                    name=row[1].strip(),
                    total_mib=values[0],
                    used_mib=values[1],
                    free_mib=values[2],
                )
            )
        except ValueError as exc:
            raise PreflightError(f"could not parse nvidia-smi row: {row!r}") from exc
    return gpus


def select_gpu(
    gpus: Sequence[GPUInfo], requested_index: int | None, minimum_free_mib: int
) -> GPUInfo:
    """Select the freest eligible physical GPU without changing process state."""

    if requested_index is not None:
        matches = [gpu for gpu in gpus if gpu.physical_index == requested_index]
        if not matches:
            raise PreflightError(f"requested physical GPU {requested_index} was not reported")
        selected = matches[0]
        if selected.free_mib < minimum_free_mib:
            raise PreflightError(
                f"GPU {requested_index} has {selected.free_mib} MiB free; "
                f"at least {minimum_free_mib} MiB is required"
            )
        return selected

    eligible = [gpu for gpu in gpus if gpu.free_mib >= minimum_free_mib]
    if not eligible:
        summary = ", ".join(f"GPU {gpu.physical_index}: {gpu.free_mib} MiB" for gpu in gpus)
        raise PreflightError(
            f"no GPU has the required {minimum_free_mib} MiB free ({summary or 'no GPUs'})"
        )
    return max(eligible, key=lambda gpu: gpu.free_mib)


def acquire_gpu_lease(physical_index: int) -> tuple[Path, TextIO]:
    """Serialize PhiAgent jobs that selected the same physical GPU."""

    lease_path = Path("/tmp") / f"phiagent-gpu-{physical_index}.lock"
    lease = lease_path.open("a+", encoding="utf-8")
    fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
    return lease_path, lease


def _run_capture(
    command: Sequence[str],
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def _select_runtime_profile(runtime: dict[str, Any]) -> str:
    observed_versions = runtime["packages"]
    profile_failures: dict[str, Any] = {}
    for profile_name, (expected_cuda, expected_versions) in SUPPORTED_RUNTIME_PROFILES.items():
        mismatches = {
            name: {"actual": observed_versions.get(name), "expected": expected}
            for name, expected in expected_versions.items()
            if observed_versions.get(name, "").split("+", maxsplit=1)[0] != expected
        }
        if runtime["torch_cuda"] == expected_cuda and not mismatches:
            return profile_name
        profile_failures[profile_name] = {
            "expected_cuda": expected_cuda,
            "actual_cuda": runtime["torch_cuda"],
            "package_mismatches": mismatches,
        }
    raise PreflightError(
        "GPU Python environment does not match a supported runtime profile: "
        f"{profile_failures}"
    )


def _probe_python_runtime(python_executable: Path, physical_gpu_index: int) -> dict[str, Any]:
    """Validate the selected environment and CUDA device in an isolated process."""

    probe = (
        "import importlib.metadata as m, json, librosa, moviepy, sam2, torch; "
        "names=['torch','torchvision','torchaudio','diffusers','transformers','peft',"
        "'moviepy','librosa','accelerate','onnxruntime-gpu','flash-attn']; "
        "result={'packages':{n:m.version(n) for n in names},"
        "'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
        "'cudnn':torch.backends.cudnn.version()}; "
        "result['logical_gpu_name']=torch.cuda.get_device_name(0) "
        "if result['cuda_available'] else None; print(json.dumps(result))"
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_index)
    try:
        output = _run_capture(
            [str(python_executable), "-c", probe],
            env=environment,
        )
        runtime = json.loads(output.splitlines()[-1])
    except (json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        details = exc.stdout if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise PreflightError(f"GPU Python environment probe failed: {details}") from exc
    if not runtime["cuda_available"]:
        raise PreflightError(
            f"PyTorch cannot use physical GPU {physical_gpu_index} in {python_executable}"
        )
    runtime["profile"] = _select_runtime_profile(runtime)
    return runtime


def query_gpus() -> tuple[list[GPUInfo], str, str]:
    """Return GPU inventory plus raw inventory and process snapshots."""

    if shutil.which("nvidia-smi") is None:
        raise PreflightError("nvidia-smi is not available; Wan2.2-Animate requires NVIDIA CUDA")
    inventory = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        processes = _run_capture(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader",
            ]
        )
    except subprocess.CalledProcessError as exc:
        processes = exc.stdout or "process query failed"
    return parse_nvidia_smi_csv(inventory), inventory, processes


def _assert_files(paths: Iterable[Path], category: str) -> None:
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise PreflightError(f"missing or empty {category}: " + ", ".join(missing))


def _assert_sharded_onnx(directory: Path) -> None:
    """Validate the upstream external-data ONNX directory layout."""

    graph = directory / "end2end.onnx"
    if not directory.is_dir() or not graph.is_file() or graph.stat().st_size == 0:
        raise PreflightError(f"missing sharded pose ONNX graph: {graph}")
    if sum(path.is_file() for path in directory.iterdir()) < 2:
        raise PreflightError(f"pose ONNX external tensor files are missing: {directory}")


def _validate_input(path: Path, allowed_suffixes: set[str], label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise PreflightError(f"{label} does not exist or is empty: {resolved}")
    if resolved.suffix.lower() not in allowed_suffixes:
        raise PreflightError(
            f"unsupported {label} extension {resolved.suffix!r}; "
            f"expected one of {sorted(allowed_suffixes)}"
        )
    return resolved


def _probe_visual_media(path: Path, label: str) -> dict[str, Any]:
    """Require ffprobe to recognize a non-empty visual stream."""

    if shutil.which("ffprobe") is None:
        raise PreflightError("ffprobe is required to validate visual-transfer inputs")
    try:
        output = _run_capture(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height",
                "-of",
                "json",
                str(path),
            ]
        )
        payload = json.loads(output)
    except (json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        raise PreflightError(f"ffprobe could not decode {label}: {path}") from exc
    streams = payload.get("streams", [])
    if not streams:
        raise PreflightError(f"{label} contains no visual stream: {path}")
    stream = streams[0]
    if int(stream.get("width", 0)) <= 0 or int(stream.get("height", 0)) <= 0:
        raise PreflightError(f"{label} has invalid dimensions: {path}")
    return stream


def _probe_video_frame_count(path: Path) -> int:
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
                "stream=nb_read_frames",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    try:
        frame_count = int(payload["streams"][0]["nb_read_frames"])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise PreflightError(f"could not count video frames: {path}") from exc
    if frame_count <= 0:
        raise PreflightError(f"video contains no frames: {path}")
    return frame_count


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class WanAnimateRenderer:
    """Run official Wan preprocessing and native animation inference."""

    def __init__(self, config: WanAnimateConfig) -> None:
        config.validate()
        self.config = config

    @property
    def preprocess_script(self) -> Path:
        return (
            self.config.wan_repo
            / "wan"
            / "modules"
            / "animate"
            / "preprocess"
            / "preprocess_data.py"
        )

    @property
    def generate_script(self) -> Path:
        return self.config.wan_repo / "generate.py"

    @property
    def sam2_repo(self) -> Path:
        configured = self.config.sam2_repo or self.config.wan_repo.parent / "sam2"
        return configured.expanduser().resolve()

    def _execution_environment(
        self, physical_gpu_index: int | None, seed: int
    ) -> dict[str, str]:
        environment = os.environ.copy()
        if physical_gpu_index is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_index)
        environment["PYTHONHASHSEED"] = str(seed)
        nvidia_library_dirs = sorted(
            self.config.python_executable.parent.parent.glob(
                "lib/python*/site-packages/nvidia/*/lib"
            )
        )
        if nvidia_library_dirs:
            existing_libraries = environment.get("LD_LIBRARY_PATH")
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(
                [*(str(path.resolve()) for path in nvidia_library_dirs)]
                + ([existing_libraries] if existing_libraries else [])
            )
        return environment

    def preflight(self, select_cuda_device: bool = True) -> dict[str, Any]:
        """Validate code/checkpoints and optionally select a currently free GPU."""

        _assert_files([self.preprocess_script, self.generate_script], "Wan2.2 source files")
        required_checkpoints = [
            self.config.checkpoint_dir / ".phiagent-model-revision",
            self.config.checkpoint_dir / "config.json",
            self.config.checkpoint_dir / "Wan2.1_VAE.pth",
            self.config.checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth",
            self.config.checkpoint_dir
            / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            self.config.checkpoint_dir / "diffusion_pytorch_model.safetensors.index.json",
            self.config.checkpoint_dir / "process_checkpoint" / "det" / "yolov10m.onnx",
        ]
        required_checkpoints.extend(
            self.config.checkpoint_dir
            / f"diffusion_pytorch_model-0000{i}-of-00004.safetensors"
            for i in range(1, 5)
        )
        if self.config.use_flux:
            required_checkpoints.append(
                self.config.checkpoint_dir
                / "process_checkpoint"
                / "FLUX.1-Kontext-dev"
                / "model_index.json"
            )
        if self.config.mode == "replacement":
            _assert_files(
                [
                    self.sam2_repo / "sam2_configs" / "__init__.py",
                    self.sam2_repo / "sam2_configs" / "sam2_hiera_l.yaml",
                ],
                "SAM2 config package",
            )
            required_checkpoints.append(
                self.config.checkpoint_dir
                / "process_checkpoint"
                / "sam2"
                / "sam2_hiera_large.pt"
            )
            extension_probe = subprocess.run(
                [
                    str(self.config.python_executable),
                    "-c",
                    "import torch; from sam2 import _C; print(_C.__file__)",
                ],
                env=self._execution_environment(None, 0),
                check=False,
                capture_output=True,
                text=True,
            )
            if extension_probe.returncode != 0:
                raise PreflightError(
                    "SAM2 compiled extension is unavailable; install the pinned "
                    f"checkout into {self.config.python_executable} before replacement "
                    f"inference: {extension_probe.stderr.strip()}"
                )
            try:
                sam2_commit = _run_capture(
                    ["git", "rev-parse", "HEAD"], cwd=self.sam2_repo
                )
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                raise PreflightError(
                    f"could not read pinned SAM2 checkout at {self.sam2_repo}"
                ) from exc
            if sam2_commit != SAM2_COMMIT:
                raise PreflightError(
                    f"SAM2 checkout is {sam2_commit}, expected pinned {SAM2_COMMIT}"
                )
            if self.config.use_relighting_lora:
                required_checkpoints.extend(
                    [
                        self.config.checkpoint_dir / "relighting_lora" / "adapter_config.json",
                        self.config.checkpoint_dir
                        / "relighting_lora"
                        / "adapter_model.safetensors",
                    ]
                )
        _assert_files(required_checkpoints, "Wan2.2 checkpoint files")
        _assert_sharded_onnx(
            self.config.checkpoint_dir
            / "process_checkpoint"
            / "pose2d"
            / "vitpose_h_wholebody.onnx"
        )
        actual_model_revision = (
            self.config.checkpoint_dir / ".phiagent-model-revision"
        ).read_text().strip()
        allowed_model_revisions = {
            WAN22_MODEL_REVISION,
            f"modelscope:{WAN22_MODELSCOPE_REVISION}",
        }
        if actual_model_revision not in allowed_model_revisions:
            raise PreflightError(
                f"checkpoint marker is {actual_model_revision!r}, "
                f"expected one of {sorted(allowed_model_revisions)}"
            )

        report: dict[str, Any] = {
            "wan_commit_expected": WAN22_COMMIT,
            "model_id": WAN22_MODEL_ID,
            "model_revision": actual_model_revision,
            "python": str(self.config.python_executable),
        }
        if self.config.mode == "replacement":
            report["sam2_repo"] = str(self.sam2_repo)
            report["sam2_commit"] = SAM2_COMMIT
        try:
            report["wan_commit_actual"] = _run_capture(
                ["git", "rev-parse", "HEAD"], cwd=self.config.wan_repo
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            report["wan_commit_actual"] = "unknown"
        if report["wan_commit_actual"] != WAN22_COMMIT:
            raise PreflightError(
                f"Wan2.2 checkout is {report['wan_commit_actual']}, expected pinned {WAN22_COMMIT}"
            )

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
            report["python_runtime"] = _probe_python_runtime(
                self.config.python_executable, selected.physical_index
            )
        return report

    def build_preprocess_command(
        self, video: Path, robot_image: Path, preprocess_dir: Path
    ) -> list[str]:
        command = [
            str(self.config.python_executable),
            str(self.preprocess_script),
            "--ckpt_path",
            str(self.config.checkpoint_dir / "process_checkpoint"),
            "--video_path",
            str(video),
            "--refer_path",
            str(robot_image),
            "--save_path",
            str(preprocess_dir),
            "--resolution_area",
            str(self.config.resolution_width),
            str(self.config.resolution_height),
            "--fps",
            str(self.config.fps),
        ]
        if self.config.mode == "replacement":
            command.extend(
                [
                    "--iterations",
                    "3",
                    "--k",
                    "7",
                    "--w_len",
                    "1",
                    "--h_len",
                    "1",
                    "--replace_flag",
                ]
            )
        elif self.config.retarget:
            command.append("--retarget_flag")
        if self.config.use_flux:
            command.append("--use_flux")
        return command

    def build_generate_command(
        self, preprocess_dir: Path, generated_output: Path, prompt: str, seed: int
    ) -> list[str]:
        command = [
            str(self.config.python_executable),
            str(self.generate_script),
            "--task",
            "animate-14B",
            "--ckpt_dir",
            str(self.config.checkpoint_dir),
            "--src_root_path",
            str(preprocess_dir),
            "--refert_num",
            str(self.config.reference_frames),
            "--frame_num",
            str(self.config.frame_num),
            "--infer_frames",
            str(self.config.infer_frames),
            "--base_seed",
            str(seed),
            "--prompt",
            prompt,
            "--offload_model",
            str(self.config.offload_model).lower(),
            "--save_file",
            str(generated_output),
        ]
        if self.config.mode == "replacement":
            command.append("--replace_flag")
            if self.config.use_relighting_lora:
                command.append("--use_relighting_lora")
        if self.config.t5_cpu:
            command.append("--t5_cpu")
        return command

    def _new_experiment_dir(self, root: Path) -> Path:
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
            "package_versions": [str(self.config.python_executable), "-m", "pip", "freeze"],
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

    def render(self, request: VisualTransferRequest) -> VisualTransferResult:
        video = _validate_input(request.video, {".mp4", ".mov", ".mkv", ".webm"}, "video")
        image = _validate_input(
            request.robot_image, {".png", ".jpg", ".jpeg", ".webp"}, "robot image"
        )
        _probe_visual_media(video, "video")
        _probe_visual_media(image, "robot image")
        output = request.output.expanduser().resolve()
        if output.suffix.lower() != ".mp4":
            raise PreflightError(f"output must be an .mp4 file: {output}")
        if output.exists() and not request.overwrite:
            raise PreflightError(f"output already exists (pass --overwrite to replace it): {output}")

        preflight = self.preflight(select_cuda_device=True)
        selected_gpu = preflight["selected_gpu"]
        experiment = self._new_experiment_dir(request.experiment_root.expanduser().resolve())
        preprocess_dir = experiment / "preprocess"
        generated_output = experiment / "wan_output.mp4"
        preserved_output = experiment / "object_preserved_output.mp4"
        preprocess_dir.mkdir()
        metadata_path = experiment / "metadata.json"

        preprocess_command = self.build_preprocess_command(video, image, preprocess_dir)
        generate_command = self.build_generate_command(
            preprocess_dir, generated_output, request.prompt, request.seed
        )
        metadata: dict[str, Any] = {
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": asdict(request),
            "renderer_config": asdict(self.config),
            "preflight": preflight,
            "commands": {"preprocess": preprocess_command, "generate": generate_command},
            "provenance": self._project_provenance(),
            "limitations": [
                "Wan2.2-Animate does not enforce robot kinematics, finger contacts, or physics.",
                "The pinned native WanAnimate generator ignores its accepted --prompt argument.",
                (
                    "Replacement mode preserves source pixels outside its estimated character mask; "
                    "mask errors can retain human pixels or replace the manipulated object."
                    if self.config.mode == "replacement"
                    else "Animation mode regenerates the full frame and can drift in background and object appearance."
                ),
            ],
        }
        _write_json(metadata_path, metadata)

        environment = self._execution_environment(
            selected_gpu["physical_index"], request.seed
        )
        lease_path, lease = acquire_gpu_lease(selected_gpu["physical_index"])
        try:
            current_gpus, _, _ = query_gpus()
            select_gpu(
                current_gpus,
                requested_index=selected_gpu["physical_index"],
                minimum_free_mib=self.config.minimum_free_gpu_mib,
            )
            metadata["gpu_lease"] = str(lease_path)
            _write_json(metadata_path, metadata)
            self._execute(
                preprocess_command,
                self.config.wan_repo,
                experiment / "preprocess.log",
                environment,
            )
            preprocessing_outputs = [
                preprocess_dir / "src_pose.mp4",
                preprocess_dir / "src_face.mp4",
            ]
            if self.config.mode == "replacement":
                preprocessing_outputs.extend(
                    [preprocess_dir / "src_bg.mp4", preprocess_dir / "src_mask.mp4"]
                )
            _assert_files(preprocessing_outputs, "Wan preprocessing outputs")
            self._execute(
                generate_command,
                self.config.wan_repo,
                experiment / "generate.log",
                environment,
            )
            _assert_files([generated_output], "Wan output")
            delivered_output = generated_output
            if self.config.mode == "replacement":
                ffmpeg = shutil.which("ffmpeg")
                if ffmpeg is None:
                    raise PreflightError("ffmpeg is required for object-mask auditing")
                assert self.config.object_roi is not None
                ffmpeg_path = Path(ffmpeg).resolve()
                frame_count = _probe_video_frame_count(generated_output)
                tracker_config = ObjectTrackerConfig(
                    initial_roi=NormalizedROI(*self.config.object_roi)
                )
                source_frames = RGBFrames(
                    decode_video(
                        video,
                        ffmpeg_path,
                        width=self.config.resolution_width,
                        height=self.config.resolution_height,
                        fps=self.config.fps,
                        frame_num=frame_count,
                        pixel_format="rgb24",
                    ),
                    width=self.config.resolution_width,
                    height=self.config.resolution_height,
                )
                candidate_frames = RGBFrames(
                    decode_video(
                        generated_output,
                        ffmpeg_path,
                        width=self.config.resolution_width,
                        height=self.config.resolution_height,
                        fps=self.config.fps,
                        frame_num=frame_count,
                        pixel_format="rgb24",
                    ),
                    width=self.config.resolution_width,
                    height=self.config.resolution_height,
                )
                route = route_object_preservation(
                    source_frames,
                    candidate_frames,
                    tracker_config,
                )
                _write_json(
                    experiment / "object_confidence_route.json",
                    asdict(route),
                )
                if route.repair_applied:
                    coverage = preserve_source_object(
                        source_video=video,
                        candidate_video=generated_output,
                        character_mask_video=preprocess_dir / "src_mask.mp4",
                        output_video=preserved_output,
                        object_mask_video=experiment / "source_object_mask.mp4",
                        report_path=experiment / "object_mask_coverage.json",
                        ffmpeg=ffmpeg_path,
                        width=self.config.resolution_width,
                        height=self.config.resolution_height,
                        fps=self.config.fps,
                        frame_num=frame_count,
                        tracker_config=tracker_config,
                    )
                    delivered_output = preserved_output
                    metadata["object_preservation"] = {
                        "confidence_route": asdict(route),
                        "character_mask_mean_coverage": coverage.mean_fraction,
                        "character_mask_maximum_coverage": coverage.maximum_fraction,
                        "composited": True,
                        "raw_model_output": str(generated_output),
                        "delivered_output": str(delivered_output),
                    }
                else:
                    metadata["object_preservation"] = {
                        "confidence_route": asdict(route),
                        "composited": False,
                        "raw_model_output": str(generated_output),
                        "delivered_output": str(generated_output),
                    }
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(delivered_output, output)
            metadata.update(
                {
                    "status": "succeeded",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "output": str(output),
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
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()
        return VisualTransferResult(
            output=output, experiment_dir=experiment, metadata=metadata_path
        )
