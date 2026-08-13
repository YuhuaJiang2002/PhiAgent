#!/usr/bin/env python3
"""Run a reproducible Cosmos3 DROID multiview image-to-video experiment.

The disclosed 2x2 first frame is the only visual condition.  Real future
frames are deliberately absent from the experiment directory and are reserved
for evaluation.  Heavy Cosmos/PyTorch imports stay in the external process.
"""

from __future__ import annotations

import argparse
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.cosmos3 import (  # noqa: E402
    COSMOS3_FRAMEWORK_COMMIT,
    COSMOS3_MODEL_REVISION,
    Cosmos3Config,
    Cosmos3TrajectoryRenderer,
    WAN22_TI2V_MODEL_REVISION,
)
from scripts.experiment_provenance import package_inventory  # noqa: E402


ASPECT_RATIOS = {"16,9", "4,3", "1,1", "3,4", "9,16"}
MAX_FRAMES = {256: 397, 480: 297, 720: 197}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wan-vae", type=Path, required=True)
    parser.add_argument(
        "--expected-framework-commit", default=COSMOS3_FRAMEWORK_COMMIT
    )
    parser.add_argument("--expected-model-revision", default=COSMOS3_MODEL_REVISION)
    parser.add_argument("--condition-image", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--physical-gpus", type=int, nargs="+", required=True)
    parser.add_argument("--python-executable", type=Path)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=60_000)
    parser.add_argument("--resolution", type=int, choices=(256, 480, 720), default=480)
    parser.add_argument("--aspect-ratio", choices=sorted(ASPECT_RATIOS), default="16,9")
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--fps", type=int, choices=(10, 16, 24, 30), default=16)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--master-port", type=int, default=29631)
    parser.add_argument("--offload-guardrails", action="store_true")
    parser.add_argument("--no-guardrails", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--project-source-revision")
    parser.add_argument("--project-source-branch")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"{label} is missing or empty: {resolved}")
    return resolved


def require_executable(path: Path, label: str) -> Path:
    """Return an absolute executable path without dereferencing a venv symlink."""
    absolute = Path(os.path.abspath(str(path.expanduser())))
    if not absolute.is_file() or not os.access(absolute, os.X_OK):
        raise ValueError(f"{label} is missing or not executable: {absolute}")
    return absolute


def _require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is missing: {resolved}")
    return resolved


def validate_generation_shape(resolution: int, num_frames: int) -> None:
    if resolution not in MAX_FRAMES:
        raise ValueError(f"unsupported resolution: {resolution}")
    if num_frames < 25 or (num_frames - 1) % 4:
        raise ValueError("num-frames must be at least 25 and satisfy 4n+1")
    if num_frames > MAX_FRAMES[resolution]:
        raise ValueError(
            f"num-frames exceeds the {resolution}p maximum of {MAX_FRAMES[resolution]}"
        )


def query_physical_gpus() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            raise RuntimeError(f"unexpected nvidia-smi row: {line}")
        rows.append(
            {
                "physical_index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": int(fields[3]),
                "memory_used_mib": int(fields[4]),
                "memory_free_mib": int(fields[5]),
                "utilization_gpu_percent": int(fields[6]),
            }
        )
    if not rows:
        raise RuntimeError("nvidia-smi returned no physical GPUs")
    return rows


def validate_gpu_selection(
    inventory: Sequence[dict[str, Any]],
    selected: Sequence[int],
    minimum_free_gpu_mib: int,
) -> list[dict[str, Any]]:
    if minimum_free_gpu_mib <= 0:
        raise ValueError("minimum-free-gpu-mib must be positive")
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("physical GPU selection must be non-empty and unique")
    if len(selected) not in (1, 2, 4, 8):
        raise ValueError("Cosmos GPU count must be 1, 2, 4, or 8")
    by_index = {int(row["physical_index"]): row for row in inventory}
    missing = sorted(set(selected) - set(by_index))
    if missing:
        raise ValueError(f"physical GPUs do not exist: {missing}")
    rows = [by_index[index] for index in selected]
    insufficient = [
        (row["physical_index"], row["memory_free_mib"])
        for row in rows
        if int(row["memory_free_mib"]) < minimum_free_gpu_mib
    ]
    if insufficient:
        raise RuntimeError(
            f"selected GPUs below {minimum_free_gpu_mib} MiB free: {insufficient}"
        )
    return rows


def build_i2v_spec(
    *,
    sample_id: str,
    prompt: str,
    condition_image: Path,
    resolution: int,
    aspect_ratio: str,
    num_frames: int,
    fps: int,
    num_steps: int,
    guidance: float,
    shift: float,
    seed: int,
) -> dict[str, Any]:
    validate_generation_shape(resolution, num_frames)
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"unsupported aspect ratio: {aspect_ratio}")
    if not sample_id.strip():
        raise ValueError("sample-id must not be empty")
    if not prompt.strip():
        raise ValueError("prompt file must not be empty")
    if num_steps <= 0 or guidance <= 0 or shift <= 0:
        raise ValueError("sampling parameters must be positive")
    return {
        "name": sample_id,
        "model_mode": "image2video",
        "prompt": prompt.strip(),
        "vision_path": str(condition_image),
        "resolution": str(resolution),
        "aspect_ratio": aspect_ratio,
        "num_frames": num_frames,
        "fps": fps,
        "num_steps": num_steps,
        "guidance": guidance,
        "shift": shift,
        "seed": seed,
        "enable_sound": False,
        "negative_metadata_mode": "none",
        "negative_prompt_keep_metadata": True,
    }


def build_inference_command(
    *,
    python: Path,
    framework_repo: Path,
    spec_path: Path,
    model_config_path: Path,
    checkpoint: Path,
    output_dir: Path,
    seed: int,
    gpu_count: int,
    master_port: int,
    offload_guardrails: bool,
    no_guardrails: bool,
) -> list[str]:
    module = [
        "-m",
        "cosmos_framework.scripts.inference",
        "--parallelism-preset=throughput" if gpu_count > 1 else "--parallelism-preset=latency",
        "-i",
        str(spec_path),
        "-o",
        str(output_dir),
        "--checkpoint-path",
        str(checkpoint),
        "--config-file",
        str(model_config_path),
        "--seed",
        str(seed),
        "--no-use-torch-compile",
    ]
    if no_guardrails:
        module.append("--no-guardrails")
    elif offload_guardrails:
        module.append("--offload-guardrail-models")
    if gpu_count == 1:
        return [str(python), *module]
    torchrun = python.parent / "torchrun"
    if not torchrun.is_file():
        raise ValueError(f"torchrun is missing from the Cosmos environment: {torchrun}")
    return [
        str(torchrun),
        f"--nproc-per-node={gpu_count}",
        f"--master-port={master_port}",
        *module,
    ]


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def project_provenance(
    explicit_revision: str | None = None,
    explicit_branch: str | None = None,
) -> dict[str, Any]:
    try:
        commit = _git_output(PROJECT_ROOT, "rev-parse", "HEAD")
        branch = _git_output(PROJECT_ROOT, "branch", "--show-current")
        status = _git_output(PROJECT_ROOT, "status", "--porcelain")
        git_available = True
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        commit = explicit_revision or "unavailable"
        branch = explicit_branch or "unavailable"
        status = f"Git metadata unavailable in execution copy: {exc}"
        git_available = False
    if explicit_revision:
        commit = explicit_revision
    if explicit_branch:
        branch = explicit_branch
    return {
        "commit": commit,
        "branch": branch,
        "status": status,
        "execution_copy_has_git": git_available,
        "explicit_source_revision": explicit_revision,
        "explicit_source_branch": explicit_branch,
        "launcher_sha256": _sha256(Path(__file__).resolve()),
        "cosmos3_adapter_sha256": _sha256(
            PROJECT_ROOT / "phiagent/rendering/cosmos3.py"
        ),
    }


def _external_package_inventory(python: Path) -> str:
    uv = python.parent / "uv"
    command = (
        [str(uv), "pip", "freeze", "--python", str(python)]
        if uv.is_file()
        else [str(python), "-m", "pip", "freeze", "--all"]
    )
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    packages = sorted(
        (line.strip() for line in completed.stdout.splitlines() if line.strip()),
        key=str.casefold,
    )
    if not packages:
        raise RuntimeError(f"could not inventory Cosmos packages with {python}")
    return "\n".join(packages) + "\n"


def _new_experiment_dir(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = resolved / f"{timestamp}-{uuid.uuid4().hex[:8]}"
    experiment.mkdir()
    return experiment


def _probe_video(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)["streams"][0]


def main() -> int:
    args = _parser().parse_args()
    validate_generation_shape(args.resolution, args.num_frames)
    framework_repo = _require_dir(args.framework_repo, "Cosmos Framework checkout")
    if not (framework_repo / ".git").is_dir():
        raise ValueError(f"Cosmos Framework path is not a Git checkout: {framework_repo}")
    framework_commit = _git_output(framework_repo, "rev-parse", "HEAD")
    if framework_commit != args.expected_framework_commit:
        raise ValueError(
            f"Cosmos Framework is {framework_commit}; "
            f"expected {args.expected_framework_commit}"
        )
    checkpoint = _require_dir(args.checkpoint, "Cosmos3 checkpoint")
    revision_marker = _require_file(
        checkpoint / ".phiagent-model-revision", "checkpoint revision marker"
    )
    model_revision = revision_marker.read_text(encoding="utf-8").strip()
    if model_revision != args.expected_model_revision:
        raise ValueError(
            f"Cosmos3 checkpoint is {model_revision}; "
            f"expected {args.expected_model_revision}"
        )
    _require_file(checkpoint / "config.json", "Cosmos3 checkpoint config")
    wan_vae = _require_file(args.wan_vae, "pinned Wan2.2 VAE")
    wan_revision_marker = _require_file(
        wan_vae.parent / ".phiagent-model-revision", "Wan2.2 VAE revision marker"
    )
    wan_revision = wan_revision_marker.read_text(encoding="utf-8").strip()
    if wan_revision != WAN22_TI2V_MODEL_REVISION:
        raise ValueError(
            f"Wan2.2 VAE is {wan_revision}; expected {WAN22_TI2V_MODEL_REVISION}"
        )
    condition = _require_file(args.condition_image, "real condition image")
    prompt_file = _require_file(args.prompt_file, "real task prompt")
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    python = (
        require_executable(args.python_executable, "Cosmos Python")
        if args.python_executable
        else require_executable(framework_repo / ".venv/bin/python", "Cosmos Python")
    )

    inventory = query_physical_gpus()
    selected = validate_gpu_selection(
        inventory, args.physical_gpus, args.minimum_free_gpu_mib
    )
    experiment = _new_experiment_dir(args.experiment_root)
    inputs = experiment / "inputs"
    outputs = experiment / "cosmos_output"
    inputs.mkdir()
    outputs.mkdir()
    copied_condition = inputs / f"real-condition{condition.suffix.lower()}"
    copied_prompt = inputs / "real-task-condition.txt"
    shutil.copy2(condition, copied_condition)
    shutil.copy2(prompt_file, copied_prompt)
    spec_path = inputs / "i2v.json"
    model_config_path = inputs / "model-config.json"
    spec = build_i2v_spec(
        sample_id=args.sample_id,
        prompt=prompt,
        condition_image=copied_condition,
        resolution=args.resolution,
        aspect_ratio=args.aspect_ratio,
        num_frames=args.num_frames,
        fps=args.fps,
        num_steps=args.num_steps,
        guidance=args.guidance,
        shift=args.shift,
        seed=args.seed,
    )
    _write_json(spec_path, spec)
    renderer = Cosmos3TrajectoryRenderer(
        Cosmos3Config(
            framework_repo=framework_repo,
            checkpoint_dir=checkpoint,
            python_executable=python,
            hf_home=args.hf_home,
            wan_vae_override=wan_vae,
            guardrails=not args.no_guardrails,
            offline=not args.online,
        )
    )
    _write_json(model_config_path, renderer.build_model_config())
    command = build_inference_command(
        python=python,
        framework_repo=framework_repo,
        spec_path=spec_path,
        model_config_path=model_config_path,
        checkpoint=checkpoint,
        output_dir=outputs,
        seed=args.seed,
        gpu_count=len(selected),
        master_port=args.master_port,
        offload_guardrails=args.offload_guardrails,
        no_guardrails=args.no_guardrails,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(row["physical_index"]) for row in selected
            ),
            "PYTHONHASHSEED": str(args.seed),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PATH": str(python.parent) + os.pathsep + environment.get("PATH", ""),
        }
    )
    if args.hf_home:
        environment["HF_HOME"] = str(args.hf_home.expanduser().resolve())
    if not args.online:
        environment["HF_HUB_OFFLINE"] = "1"
    metadata_path = experiment / "metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "preflight" if args.preflight_only else "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "cosmos3_nano_droid_multiview_i2v",
        "labels": {
            "condition_image": "REAL CONDITION",
            "task_text": "REAL CONDITION",
            "output_continuation": "OUR GENERATED VIDEO",
        },
        "leakage_guard": {
            "real_future_frames_passed_to_model": False,
            "visual_inputs": [str(copied_condition)],
        },
        "model": {
            "checkpoint": str(checkpoint),
            "revision": model_revision,
            "framework_repo": str(framework_repo),
            "framework_commit": framework_commit,
            "framework_commit_expected": args.expected_framework_commit,
            "model_revision_expected": args.expected_model_revision,
            "wan_vae": str(wan_vae),
            "wan_vae_revision": wan_revision,
        },
        "sampling": spec,
        "seed": args.seed,
        "command": command,
        "command_shell": shlex.join(command),
        "execution_environment": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONHASHSEED",
                "PYTORCH_CUDA_ALLOC_CONF",
                "HF_HOME",
                "HF_HUB_OFFLINE",
            )
            if key in environment
        },
        "gpu_inventory": inventory,
        "selected_gpus": selected,
        "input_hashes": {
            "condition_image": _sha256(copied_condition),
            "prompt_file": _sha256(copied_prompt),
            "checkpoint_revision_marker": _sha256(revision_marker),
            "wan_vae": _sha256(wan_vae),
            "wan_vae_revision_marker": _sha256(wan_revision_marker),
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "project_git": project_provenance(
            args.project_source_revision, args.project_source_branch
        ),
        "launcher_package_versions": package_inventory(),
        "cosmos_package_versions": _external_package_inventory(python),
    }
    _write_json(metadata_path, metadata)
    (experiment / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    if args.preflight_only:
        print(json.dumps({"experiment": str(experiment), "status": "preflight"}))
        return 0

    log_path = experiment / "cosmos.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=framework_repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"Cosmos3 inference failed with exit code {completed.returncode}; "
                f"inspect {log_path}"
            )
        generated = outputs / args.sample_id / "vision.mp4"
        _require_file(generated, "Cosmos3 generated video")
        probe = _probe_video(generated)
        if int(probe["nb_read_frames"]) != args.num_frames:
            raise RuntimeError(
                f"generated video has {probe['nb_read_frames']} frames; "
                f"expected {args.num_frames}"
            )
        metadata.update(
            {
                "status": "succeeded",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output": str(generated),
                "output_sha256": _sha256(generated),
                "output_probe": probe,
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
    print(json.dumps({"experiment": str(experiment), "output": str(generated)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
