#!/usr/bin/env python3
"""Run one calibrated PhiAgent case through an external Plenoptic runtime.

The source clip and camera bundle must already be normalized to 81 frames, 15 fps,
432x768.  This entry point owns no model code and keeps the Cosmos dependency in
its external environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.plenoptic import (  # noqa: E402
    PLENOPTIC_CONTEXT_PARALLEL,
    build_custom_suite,
    inference_command,
    sha256_file,
    validate_inference_record,
    write_json,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--runtime-root", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--source-video", type=Path, required=True)
    result.add_argument("--original-video", type=Path, required=True)
    result.add_argument("--camera-npz", type=Path, required=True)
    result.add_argument("--camera-provenance", type=Path, required=True)
    result.add_argument("--prompt-file", type=Path, required=True)
    result.add_argument("--case-id", required=True)
    result.add_argument("--seed", type=int, default=20260911)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--physical-gpus", type=int, nargs=4, required=True)
    result.add_argument("--minimum-free-gpu-mib", type=int, default=50_000)
    result.add_argument("--master-port", type=int, default=29682)
    result.add_argument("--python-executable", type=Path)
    result.add_argument("--activation-script", type=Path)
    result.add_argument(
        "--source-git-state",
        type=Path,
        help="JSON snapshot from the source checkout when running a deployed copy",
    )
    result.add_argument("--expected-checkpoint-sha256", required=True)
    result.add_argument("--preflight-only", action="store_true")
    return result


def gpu_inventory() -> list[dict[str, object]]:
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
    rows = []
    for line in completed.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
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
                "utilization_percent": int(fields[6]),
            }
        )
    return rows


def gpu_process_inventory() -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",", 3)]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi compute-app row: {line}")
        rows.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_memory_mib": int(fields[3]),
            }
        )
    return rows


def validate_gpu_selection(
    inventory: list[dict[str, object]],
    selected: list[int],
    minimum_free_mib: int,
    compute_apps: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    if len(selected) != PLENOPTIC_CONTEXT_PARALLEL or len(set(selected)) != len(selected):
        raise ValueError("Plenoptic requires exactly four distinct physical GPUs")
    by_index = {int(row["physical_index"]): row for row in inventory}
    if any(index not in by_index for index in selected):
        raise ValueError("selected physical GPU does not exist")
    rows = [by_index[index] for index in selected]
    insufficient = [
        (row["physical_index"], row["memory_free_mib"], row["utilization_percent"])
        for row in rows
        if int(row["memory_free_mib"]) < minimum_free_mib
        or int(row["utilization_percent"]) > 10
    ]
    if insufficient:
        raise RuntimeError(f"selected GPUs are not safely idle: {insufficient}")
    selected_uuids = {str(row["uuid"]) for row in rows}
    conflicts = [
        row for row in (compute_apps or []) if str(row["gpu_uuid"]) in selected_uuids
    ]
    if conflicts:
        raise RuntimeError(f"selected GPUs have active compute processes: {conflicts}")
    # H20 nodes in this deployment have GPUs 0..3 on NUMA 0 and 4..7 on NUMA 1.
    # Refuse a mixed half even though NVLink connects all cards.
    numa = {0 if index < 4 else 1 for index in selected}
    if len(numa) != 1:
        raise RuntimeError("selected GPUs cross NUMA; choose 0..3 or 4..7")
    return rows


def scheduler_state(runtime: Path) -> dict[str, object]:
    path = runtime / "../OUTPUTS/.runtime/h20-1/fixed-validation-h20-1.json"
    if not path.is_file():
        return {"state_file": str(path), "active": False, "available": False}
    value = json.loads(path.read_text(encoding="utf-8"))
    active = False
    try:
        process = Path("/proc") / str(value["pid"])
        active = (
            value.get("host") == socket.gethostname().split(".")[0]
            and (process / "stat").read_text().split()[21] == value["start_ticks"]
            and "validation_runner.py _" in (process / "cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
        )
    except (KeyError, OSError, IndexError):
        active = False
    return {
        "state_file": str(path),
        "available": True,
        "active": active,
        "phase": value.get("phase"),
        "pid": value.get("pid"),
        "run_id": value.get("run_id"),
    }


def activated_command(activation: Path, command: list[str]) -> list[str]:
    return [
        "bash",
        "-c",
        'source "$1"; shift; exec "$@"',
        "plenoptic-phiagent",
        str(activation),
        *command,
    ]


def run_command(command: list[str], env: dict[str, str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Plenoptic runner failed; inspect {log_path}")


def git_state(root: Path) -> dict[str, object]:
    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
        )

    revision = git("rev-parse", "HEAD")
    status = git("status", "--short", "--untracked-files=all")
    if revision.returncode or status.returncode:
        return {
            "available": False,
            "reason": (revision.stderr or status.stderr).strip()[:500],
        }
    return {
        "available": True,
        "revision": revision.stdout.strip(),
        "status_short": status.stdout.splitlines(),
    }


def runtime_versions(activation: Path, python: Path) -> dict[str, object]:
    program = """
import importlib.metadata as metadata
import json
import platform
packages = {}
for name in ('torch', 'torchvision', 'numpy', 'opencv-python', 'decord', 'transformers'):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({'python': platform.python_version(), 'packages': packages}))
"""
    completed = subprocess.run(
        activated_command(activation, [str(python), "-c", program]),
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return {
            "available": False,
            "reason": completed.stderr.strip()[-500:],
        }
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return {
            "available": False,
            "reason": f"runtime version output was not JSON: {error}",
            "stdout_tail": completed.stdout[-500:],
        }
    if not isinstance(value, dict):
        return {"available": False, "reason": "runtime version output was not an object"}
    return value


def command_string(command: list[str]) -> str:
    return shlex.join(command)


def main() -> None:
    args = parser().parse_args()
    runtime = args.runtime_root.resolve()
    runner = runtime / "tools/infer_custom_validation.py"
    activation = (args.activation_script or runtime / "tools/activate_h20.sh").resolve()
    # Keep the virtual-environment entry point itself.  Resolving this symlink to
    # /usr/bin/python bypasses pyvenv.cfg and silently drops the runtime packages.
    python = (args.python_executable
              or runtime / "../ENVIRONMENTS/plenoptic-h20/bin/python").absolute()
    for path, label in ((runtime, "runtime root"), (runner, "runner"),
                        (activation, "activation script"), (python, "python executable")):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    expected_sha = args.expected_checkpoint_sha256.lower()
    if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise ValueError("expected checkpoint SHA-256 must be 64 lowercase hex characters")
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    camera_provenance = json.loads(args.camera_provenance.read_text(encoding="utf-8"))
    output = args.output.resolve()
    try:
        relative_output = output.relative_to((runtime / '../OUTPUTS').resolve())
        if not relative_output.parts:
            raise ValueError('A run subdirectory is required')
    except ValueError as exc:
        raise ValueError("output must be a run directory under the runtime's ../OUTPUTS") from exc
    if output.exists():
        raise FileExistsError(f"output must be a fresh directory: {output}")
    suite = build_custom_suite(
        runtime_root=runtime,
        case_id=args.case_id,
        source_video=args.source_video,
        original_video=args.original_video,
        camera_npz=args.camera_npz,
        prompt=prompt,
        seed=args.seed,
        camera_provenance=camera_provenance,
    )
    scheduled = scheduler_state(runtime)
    if scheduled["active"]:
        raise RuntimeError(f"backend validation scheduler is active: {scheduled}")
    inventory = gpu_inventory()
    compute_apps = gpu_process_inventory()
    selected = validate_gpu_selection(
        inventory, args.physical_gpus, args.minimum_free_gpu_mib, compute_apps
    )
    actual_checkpoint_sha = sha256_file(checkpoint)
    if actual_checkpoint_sha != expected_sha:
        raise ValueError(
            "checkpoint SHA-256 differs from --expected-checkpoint-sha256: "
            f"{actual_checkpoint_sha}"
        )
    output.mkdir(parents=True)
    suite_path = output / "suite.json"
    write_json(suite_path, suite)
    environment = runtime_versions(activation, python)
    source_git = (
        json.loads(args.source_git_state.read_text(encoding="utf-8"))
        if args.source_git_state
        else git_state(PROJECT_ROOT)
    )
    if not isinstance(source_git, dict):
        raise ValueError("source Git state must be a JSON object")
    provenance = {
        "schema": 1,
        "status": "starting",
        "hostname": socket.gethostname(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_root": str(runtime),
        "launcher_sha256": sha256_file(Path(__file__).resolve()),
        "adapter_module_sha256": sha256_file(
            PROJECT_ROOT / "phiagent/rendering/plenoptic.py"
        ),
        "runner_sha256": sha256_file(runner),
        "checkpoint": str(checkpoint),
        "expected_checkpoint_sha256": expected_sha,
        "actual_checkpoint_sha256": actual_checkpoint_sha,
        "selected_gpus": selected,
        "compute_apps": compute_apps,
        "scheduler": scheduled,
        "source_sha256": sha256_file(args.source_video),
        "camera_sha256": sha256_file(args.camera_npz),
        "invocation": command_string([sys.executable, *sys.argv]),
        "git": source_git,
        "source_git_state_sha256": (
            sha256_file(args.source_git_state) if args.source_git_state else None
        ),
        "runtime_versions": environment,
    }
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(str(index) for index in args.physical_gpus),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PYTHONHASHSEED=str(args.seed),
    )
    preflight_output = output / "preflight"
    preflight = inference_command(
        python=python,
        runner=runner,
        checkpoint=checkpoint,
        suite=suite_path,
        output=preflight_output,
        preflight_only=True,
    )
    activated_preflight = activated_command(activation, preflight)
    provenance["preflight_command"] = command_string(activated_preflight)
    write_json(output / "provenance.json", provenance)
    try:
        run_command(activated_preflight, env, output / "preflight.log")
        if args.preflight_only:
            provenance.update(
                status="preflight_passed",
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            write_json(output / "provenance.json", provenance)
            return
        # Recheck immediately before allocating the model.  The CPU preflight can
        # take long enough for another scheduled job to claim the same devices.
        scheduled = scheduler_state(runtime)
        if scheduled["active"]:
            raise RuntimeError(f"backend validation scheduler became active: {scheduled}")
        compute_apps = gpu_process_inventory()
        selected = validate_gpu_selection(
            gpu_inventory(), args.physical_gpus, args.minimum_free_gpu_mib, compute_apps
        )
        provenance["selected_gpus_before_generation"] = selected
        provenance["compute_apps_before_generation"] = compute_apps
        provenance["scheduler_before_generation"] = scheduled
        generation_output = output / "generation"
        generation_worker = inference_command(
            python=python,
            runner=runner,
            checkpoint=checkpoint,
            suite=suite_path,
            output=generation_output,
        )
        if not 1024 <= args.master_port <= 65535:
            raise ValueError("master-port must be in 1024..65535")
        generation = [
            str(python),
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={PLENOPTIC_CONTEXT_PARALLEL}",
            f"--master_port={args.master_port}",
            *generation_worker[1:],
        ]
        activated_generation = activated_command(activation, generation)
        provenance["generation_command"] = command_string(activated_generation)
        write_json(output / "provenance.json", provenance)
        run_command(activated_generation, env, output / "generation.log")
        case_dir = generation_output / args.case_id
        record_path = case_dir / "inference.json"
        video_path = case_dir / "generated.mp4"
        if not record_path.is_file() or not video_path.is_file():
            raise RuntimeError("Plenoptic runner did not create its declared case outputs")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        validate_inference_record(record, case_id=args.case_id,
                                  checkpoint_sha256=expected_sha)
        provenance.update(
            status="generated",
            finished_at=datetime.now(timezone.utc).isoformat(),
            inference_record=str(record_path),
            output_sha256=sha256_file(video_path),
        )
        write_json(output / "provenance.json", provenance)
        write_json(output / "result.json", {
            "dit": str(video_path),
            "generation": str(record_path),
            "provenance": str(output / "provenance.json"),
        })
    except BaseException as error:
        provenance.update(
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error_type=type(error).__name__,
            error=str(error),
        )
        write_json(output / "provenance.json", provenance)
        raise


if __name__ == "__main__":
    main()
