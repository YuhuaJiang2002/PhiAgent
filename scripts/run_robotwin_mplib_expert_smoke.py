#!/usr/bin/env python3
"""Run one explicit MPlib-fallback RoboTwin expert trajectory smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shlex
import shutil
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


FALLBACK_MARKER = "PHIAGENT_MPLIB_FALLBACK_V1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value: value.tolist()
            if hasattr(value, "tolist")
            else value.item(),
        )
        + "\n"
    )
    temporary.replace(path)


def patch_planner_for_mplib(source: Path) -> dict[str, str]:
    planner = source / "envs" / "robot" / "planner.py"
    text = planner.read_text()
    if FALLBACK_MARKER in text:
        raise ValueError("RoboTwin planner is already patched")
    marker = "    traceback.print_exc()\n\n\n# ********************** MplibPlanner"
    if text.count(marker) != 1:
        raise ValueError("pinned RoboTwin planner fallback marker changed")
    replacement = f"""    traceback.print_exc()

    # {FALLBACK_MARKER}: gripper interpolation only; arm motion uses MplibPlanner.
    class CuroboPlanner:
        def __init__(self, *args, **kwargs):
            pass

        def plan_grippers(self, now_val, target_val):
            num_step = 200
            values = np.linspace(now_val, target_val, num_step)
            return {{
                "num_step": num_step,
                "per_step": (target_val - now_val) / num_step,
                "result": values,
            }}

        def update_point_cloud(self, *args, **kwargs):
            pass


# ********************** MplibPlanner"""
    before = _sha256(planner)
    planner.write_text(text.replace(marker, replacement))
    return {"before_sha256": before, "after_sha256": _sha256(planner)}


def _install_mplib_routes() -> None:
    import numpy as np

    from envs.robot.robot import Robot

    def plan_multi(
        robot: Any,
        target_list: list[Any],
        *,
        arm: str,
        last_qpos: Any = None,
        **unused_kwargs: Any,
    ) -> dict[str, Any]:
        entity = robot.left_entity if arm == "left" else robot.right_entity
        planner = (
            robot.left_mplib_planner if arm == "left" else robot.right_mplib_planner
        )
        current = entity.get_qpos() if last_qpos is None else last_qpos
        statuses, positions = [], []
        for target in target_list:
            transformed = robot._trans_from_gripper_to_endlink(target, arm_tag=arm)
            result = planner.plan_path(current, transformed, arms_tag=arm, log=False)
            statuses.append("Success" if result.get("status") == "Success" else "Failure")
            positions.append(result.get("position", np.empty((0, len(current)))))
        return {
            "status": np.asarray(statuses, dtype=object),
            "position": positions,
        }

    def plan_single(
        robot: Any,
        target: Any,
        *,
        arm: str,
        last_qpos: Any = None,
        **unused_kwargs: Any,
    ) -> dict[str, Any]:
        entity = robot.left_entity if arm == "left" else robot.right_entity
        planner = (
            robot.left_mplib_planner if arm == "left" else robot.right_mplib_planner
        )
        current = entity.get_qpos() if last_qpos is None else last_qpos
        transformed = robot._trans_from_gripper_to_endlink(target, arm_tag=arm)
        return planner.plan_path(current, transformed, arms_tag=arm)

    Robot.left_plan_multi_path = lambda self, targets, **kwargs: plan_multi(  # type: ignore[method-assign]
        self, targets, arm="left", **kwargs
    )
    Robot.right_plan_multi_path = lambda self, targets, **kwargs: plan_multi(  # type: ignore[method-assign]
        self, targets, arm="right", **kwargs
    )
    Robot.left_plan_path = lambda self, target, **kwargs: plan_single(  # type: ignore[method-assign]
        self, target, arm="left", **kwargs
    )
    Robot.right_plan_path = lambda self, target, **kwargs: plan_single(  # type: ignore[method-assign]
        self, target, arm="right", **kwargs
    )


def _task_args(source: Path, output: Path, seed: int) -> tuple[Any, dict[str, Any]]:
    import yaml

    config_root = source / "env_cfg" / "task_config"
    args = yaml.safe_load((config_root / "demo_clean.yml").read_text())
    if args.get("data_type", {}).get("pointcloud") is not False:
        raise ValueError("expert smoke requires pointcloud=false")
    sys.modules.setdefault("open3d", types.ModuleType("open3d"))
    sys.modules.setdefault("h5py", types.ModuleType("h5py"))
    embodiment_config = yaml.safe_load(
        (config_root / "_embodiment_config.yml").read_text()
    )
    embodiment = args["embodiment"]
    robot_file = (source / embodiment_config[embodiment[0]]["file_path"]).resolve()
    robot_config = yaml.safe_load((robot_file / "config.yml").read_text())
    robot_config["planner"] = "mplib_RRT"
    args.update(
        {
            "task_name": "adjust_bottle",
            "task_config": "demo_clean",
            "left_robot_file": str(robot_file),
            "right_robot_file": str(robot_file),
            "left_embodiment_config": robot_config,
            "right_embodiment_config": robot_config,
            "dual_arm_embodied": True,
            "embodiment_name": embodiment[0],
            "save_path": str(output / "unused-data"),
            "save_data": False,
            "need_plan": True,
            "eval_mode": False,
            "now_ep_num": 0,
            "seed": seed,
        }
    )
    module = importlib.import_module("envs.adjust_bottle")
    _install_mplib_routes()
    return getattr(module, "adjust_bottle")(), args


def _worker(args: argparse.Namespace) -> int:
    source = args.runtime_source.expanduser().resolve()
    overlay = args.overlay.expanduser().resolve()
    output = args.worker_output.expanduser().resolve()
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(overlay))
    os.chdir(source)
    import cv2
    import numpy as np

    task, task_args = _task_args(source, output.parent, args.seed)
    task.setup_demo(**task_args)
    task.play_once()
    observation = task.get_obs()
    head = observation["observation"]["head_camera"]["rgb"]
    image = output.with_suffix(".png")
    cv2.imwrite(str(image), cv2.cvtColor(head, cv2.COLOR_RGB2BGR))
    contacts = task.scene.get_contacts()
    result = {
        "seed": args.seed,
        "planner": "mplib_RRT_with_gripper_only_curobo_fallback",
        "plan_success": bool(task.plan_success),
        "task_success": bool(task.check_success()),
        "left_joint_path_steps": len(task.left_joint_path),
        "right_joint_path_steps": len(task.right_joint_path),
        "left_joint_path": np.asarray(task.left_joint_path, dtype=object).tolist(),
        "right_joint_path": np.asarray(task.right_joint_path, dtype=object).tolist(),
        "final_qpos": np.asarray(
            observation["joint_action"]["vector"], dtype=np.float64
        ).tolist(),
        "final_endpose": {
            key: np.asarray(value, dtype=np.float64).tolist()
            if hasattr(value, "__len__")
            else float(value)
            for key, value in observation["endpose"].items()
        },
        "bottle_functional_point_m": np.asarray(
            task.bottle.get_functional_point(0), dtype=np.float64
        ).tolist(),
        "contact_count_at_final_observation": len(contacts),
        "head_rgb": str(image),
        "head_rgb_sha256": hashlib.sha256(memoryview(head).tobytes()).hexdigest(),
    }
    task.close_env(clear_cache=True)
    _write_json(output, result)
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
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--runtime-source", type=Path)
    parser.add_argument("--worker-output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        if args.runtime_source is None or args.worker_output is None:
            raise ValueError("worker mode requires runtime source and output")
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
        raise FileExistsError(f"refusing to overwrite RoboTwin expert smoke: {output}")
    runtime = json.loads(manifest_path.read_text())
    source = Path(str(runtime["runtime_source"])).expanduser().resolve()
    if not source.is_dir() or not python.is_file() or not overlay.is_dir():
        raise ValueError("RoboTwin expert runtime inputs are missing")
    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    output.mkdir(parents=True)
    runtime_source = output / "runtime-source"
    shutil.copytree(source, runtime_source, symlinks=True)
    patch = patch_planner_for_mplib(runtime_source)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    environment["PHIAGENT_PHYSICAL_GPU_INDEX"] = str(selected.physical_index)
    environment["ASSETS_PATH"] = str(runtime_source / "assets")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(overlay), str(runtime_source), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    worker_output = output / "expert.json"
    command = [
        str(python),
        str(Path(__file__).resolve()),
        "--worker",
        "--runtime-source",
        str(runtime_source),
        "--overlay",
        str(overlay),
        "--worker-output",
        str(worker_output),
        "--seed",
        str(args.seed),
    ]
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    _write_json(
        output / "config.json",
        {
            "schema_version": "1.0.0",
            "status": "STARTED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "runtime_manifest": str(manifest_path),
            "runtime_source": str(runtime_source),
            "overlay": str(overlay),
            "python": str(python),
            "seed": args.seed,
            "selected_physical_gpu": asdict(selected),
            "gpu_inventory_raw": inventory,
            "gpu_processes_raw": processes,
            "planner_patch": patch,
            "method": "robotwin_mplib_fallback_expert_smoke",
            "mplib_routing_monkeypatch": (
                "Curobo arm planning calls are explicitly routed to MplibPlanner; "
                "constraint_pose is not enforced in this diagnostic."
            ),
        },
    )
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=args.timeout_seconds,
        check=False,
    )
    (output / "expert.log").write_text(completed.stdout + completed.stderr)
    evidence = json.loads(worker_output.read_text()) if worker_output.is_file() else None
    working = (
        completed.returncode == 0
        and isinstance(evidence, dict)
        and evidence.get("plan_success") is True
        and evidence.get("task_success") is True
        and (
            int(evidence.get("left_joint_path_steps", 0)) > 0
            or int(evidence.get("right_joint_path_steps", 0)) > 0
        )
    )
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING" if working else "PARTIAL",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "return_code": completed.returncode,
        "selected_physical_gpu": asdict(selected),
        "evidence": evidence,
        "planner_patch": patch,
        "claim_boundary": (
            "This is a non-official MPlib fallback expert smoke. It validates one "
            "successful executable rollout only and cannot replace the official "
            "Curobo recipe or 20-reset paired counterfactual gate."
        ),
    }
    _write_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if working else 2


if __name__ == "__main__":
    raise SystemExit(main())
