#!/usr/bin/env python3
"""Validate independent, deterministic RoboTwin resets on one physical GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import types
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256_bytes(payload: Any) -> str:
    return hashlib.sha256(memoryview(payload).tobytes()).hexdigest()


def _max_difference(first: Any, second: Any) -> float:
    def flatten(value: Any) -> list[float]:
        if isinstance(value, (list, tuple)):
            return [item for child in value for item in flatten(child)]
        return [float(value)]

    left = flatten(first)
    right = flatten(second)
    if len(left) != len(right):
        raise ValueError(
            f"reset evidence shape mismatch: {len(left)} != {len(right)} values"
        )
    return max((abs(a - b) for a, b in zip(left, right)), default=0.0)


def _build_task_args(source: Path, output: Path, seed: int) -> tuple[Any, dict[str, Any]]:
    import yaml

    task_name = "adjust_bottle"
    config_name = "demo_clean"
    config_root = source / "env_cfg" / "task_config"
    args = yaml.safe_load((config_root / f"{config_name}.yml").read_text())
    if args.get("data_type", {}).get("pointcloud") is not False:
        raise ValueError("Open3D-free reset preflight requires pointcloud=false")
    # These packages are referenced only inside disabled point-cloud/data-save methods.
    sys.modules.setdefault("open3d", types.ModuleType("open3d"))
    sys.modules.setdefault("h5py", types.ModuleType("h5py"))
    planner_module = types.ModuleType("envs.robot.planner")

    class ResetOnlyPlanner:
        def __init__(self, *unused_args: Any, **unused_kwargs: Any) -> None:
            pass

        @staticmethod
        def plan_grippers(now_val: float, target_val: float) -> dict[str, Any]:
            import numpy as np

            values = np.linspace(now_val, target_val, 200)
            return {
                "num_step": 200,
                "per_step": (target_val - now_val) / 200,
                "result": values,
            }

    planner_module.CuroboPlanner = ResetOnlyPlanner
    planner_module.MplibPlanner = ResetOnlyPlanner
    sys.modules.setdefault("envs.robot.planner", planner_module)
    embodiment_config = yaml.safe_load(
        (config_root / "_embodiment_config.yml").read_text()
    )
    embodiment = args["embodiment"]
    if len(embodiment) != 1:
        raise ValueError("reset preflight requires one symmetric embodiment")
    robot_file = (source / embodiment_config[embodiment[0]]["file_path"]).resolve()
    robot_config = yaml.safe_load((robot_file / "config.yml").read_text())
    args.update(
        {
            "task_name": task_name,
            "task_config": config_name,
            "left_robot_file": str(robot_file),
            "right_robot_file": str(robot_file),
            "left_embodiment_config": robot_config,
            "right_embodiment_config": robot_config,
            "dual_arm_embodied": True,
            "embodiment_name": embodiment[0],
            "save_path": str(output / "unused-data"),
            "save_data": False,
            "need_plan": False,
            "eval_mode": False,
            "now_ep_num": 0,
            "seed": seed,
        }
    )
    module = importlib.import_module(f"envs.{task_name}")
    task_class = getattr(module, task_name)
    return task_class(), args


def _worker(args: argparse.Namespace) -> int:
    source = args.runtime_source.expanduser().resolve()
    overlay = args.overlay.expanduser().resolve()
    worker_output = args.worker_output.expanduser().resolve()
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(overlay))
    os.chdir(source)
    import cv2
    import numpy as np

    task, task_args = _build_task_args(source, worker_output.parent, args.worker_seed)
    task.setup_demo(**task_args)
    observation = task.get_obs()
    head = observation["observation"]["head_camera"]["rgb"]
    qpos = observation["joint_action"]["vector"]
    endpose = [
        *observation["endpose"]["left_endpose"],
        observation["endpose"]["left_gripper"],
        *observation["endpose"]["right_endpose"],
        observation["endpose"]["right_gripper"],
    ]
    bottle_point = np.asarray(task.bottle.get_functional_point(0), dtype=np.float64)
    actors = {}
    for actor in task.scene.get_all_actors():
        pose = actor.get_pose()
        actors[str(actor.get_name())] = {
            "translation_m": np.asarray(pose.p, dtype=np.float64).tolist(),
            "quaternion_wxyz": np.asarray(pose.q, dtype=np.float64).tolist(),
        }
    image_path = worker_output.with_suffix(".png")
    cv2.imwrite(str(image_path), cv2.cvtColor(head, cv2.COLOR_RGB2BGR))
    result = {
        "seed": args.worker_seed,
        "qpose_tag": int(task.qpose_tag),
        "model_id": int(task.model_id),
        "qpos": np.asarray(qpos, dtype=np.float64).tolist(),
        "endpose": np.asarray(endpose, dtype=np.float64).tolist(),
        "bottle_functional_point_m": bottle_point.tolist(),
        "head_rgb_shape": list(head.shape),
        "head_rgb_sha256": _sha256_bytes(head),
        "head_rgb": str(image_path),
        "actors": actors,
        "initial_success": bool(task.check_success()),
    }
    task.close_env(clear_cache=True)
    _write_json(worker_output, result)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--different-seed", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--runtime-source", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker-seed", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        if (
            args.runtime_source is None
            or args.worker_output is None
            or args.worker_seed is None
        ):
            raise ValueError("worker mode requires runtime source, output, and seed")
        return _worker(args)
    if (
        args.runtime_manifest is None
        or args.python is None
        or args.output_dir is None
        or args.gpu is None
    ):
        raise ValueError("parent mode requires runtime, Python, output, and GPU")
    manifest_path = args.runtime_manifest.expanduser().resolve()
    python = args.python.expanduser().resolve()
    overlay = args.overlay.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite RoboTwin reset preflight: {output}")
    if not manifest_path.is_file() or not python.is_file() or not overlay.is_dir():
        raise ValueError("RoboTwin reset runtime inputs are missing")
    runtime = json.loads(manifest_path.read_text())
    source = Path(str(runtime["runtime_source"])).expanduser().resolve()
    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    environment["PHIAGENT_PHYSICAL_GPU_INDEX"] = str(selected.physical_index)
    environment["ASSETS_PATH"] = str(source / "assets")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(overlay), str(source), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    _write_json(
        output / "config.json",
        {
            "schema_version": "1.0.0",
            "status": "STARTED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "runtime_manifest": str(manifest_path),
            "runtime_source": str(source),
            "overlay": str(overlay),
            "python": str(python),
            "selected_physical_gpu": asdict(selected),
            "gpu_inventory_raw": inventory,
            "gpu_processes_raw": processes,
            "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
            "seeds": [args.seed, args.seed, args.different_seed],
            "open3d_stubbed_for_pointcloud_disabled_reset_preflight": True,
            "h5py_stubbed_for_save_data_false_reset_preflight": True,
            "motion_planners_stubbed_for_setup_only_reset_preflight": True,
        },
    )
    runs = []
    for label, seed in (
        ("same-a", args.seed),
        ("same-b", args.seed),
        ("different", args.different_seed),
    ):
        worker_output = output / f"{label}.json"
        command = [
            str(python),
            str(Path(__file__).resolve()),
            "--worker",
            "--runtime-source",
            str(source),
            "--overlay",
            str(overlay),
            "--worker-output",
            str(worker_output),
            "--worker-seed",
            str(seed),
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
        (output / f"{label}.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode != 0 or not worker_output.is_file():
            result = {
                "status": "BLOCKED",
                "failed_run": label,
                "return_code": completed.returncode,
            }
            _write_json(output / "result.json", result)
            return 2
        runs.append(json.loads(worker_output.read_text()))
    first, second, different = runs
    same_metrics = {
        "qpos_max_abs": _max_difference(first["qpos"], second["qpos"]),
        "endpose_max_abs": _max_difference(first["endpose"], second["endpose"]),
        "bottle_max_abs_m": _max_difference(
            first["bottle_functional_point_m"],
            second["bottle_functional_point_m"],
        ),
        "head_rgb_exact": first["head_rgb_sha256"] == second["head_rgb_sha256"],
        "qpose_tag_equal": first["qpose_tag"] == second["qpose_tag"],
        "model_id_equal": first["model_id"] == second["model_id"],
    }
    intervention = {
        "bottle_max_abs_m": _max_difference(
            first["bottle_functional_point_m"],
            different["bottle_functional_point_m"],
        ),
        "qpose_tag_changed": first["qpose_tag"] != different["qpose_tag"],
        "model_id_changed": first["model_id"] != different["model_id"],
        "head_rgb_changed": first["head_rgb_sha256"] != different["head_rgb_sha256"],
    }
    deterministic = (
        same_metrics["qpos_max_abs"] <= 1e-6
        and same_metrics["endpose_max_abs"] <= 1e-6
        and same_metrics["bottle_max_abs_m"] <= 1e-6
        and same_metrics["head_rgb_exact"]
        and same_metrics["qpose_tag_equal"]
        and same_metrics["model_id_equal"]
    )
    changed = (
        intervention["bottle_max_abs_m"] > 1e-4
        or intervention["qpose_tag_changed"]
        or intervention["model_id_changed"]
        or intervention["head_rgb_changed"]
    )
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "WORKING" if deterministic and changed else "BLOCKED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "selected_physical_gpu": asdict(selected),
        "same_seed": args.seed,
        "different_seed": args.different_seed,
        "same_seed_metrics": same_metrics,
        "different_seed_intervention": intervention,
        "deterministic_reset": deterministic,
        "different_seed_changes_scene": changed,
        "runs": runs,
        "claim_boundary": (
            "Reset preflight validates independent deterministic setup only. Expert "
            "trajectory collection and paired A+/A_swap replay remain separate gates."
        ),
    }
    _write_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "WORKING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
