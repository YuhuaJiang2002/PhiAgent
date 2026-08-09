"""Deterministic MuJoCo control-video production and trajectory binding."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phiagent.agent.verifier import AgentVerifier
from phiagent.physical_language.schema import FrameKind, FrameRef
from phiagent.rendering.cosmos3 import _probe_control_video, _sha256
from phiagent.simulation.base import SimulationRequest
from phiagent.simulation.mujoco_backend import MujocoBackend


@dataclass(frozen=True)
class ControlVideoBundle:
    experiment_dir: Path
    control_video: Path
    robot_trajectory: Path
    object_trajectories: tuple[Path, ...]
    verification_record: Path
    simulation_record: Path
    manifest: Path


def _run_capture(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def _new_experiment_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = root / f"{timestamp}-{uuid.uuid4().hex[:8]}"
    experiment.mkdir()
    return experiment


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _provenance() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    provenance: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "argv": sys.argv,
    }
    for key, command in {
        "git_commit": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--porcelain"],
        "package_versions": [sys.executable, "-m", "pip", "freeze"],
    }.items():
        try:
            provenance[key] = _run_capture(command, project_root)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            provenance[key] = f"unavailable: {exc}"
    return provenance


def produce_mujoco_control_bundle(
    request: SimulationRequest,
    *,
    camera: str | int | None,
    experiment_root: Path,
    robot_base_name: str,
) -> ControlVideoBundle:
    """Replay, verify, render, and persist one trajectory-bound control bundle."""

    if request.render_output is not None:
        raise ValueError("control bundle owns its unique render_output path")
    experiment = _new_experiment_dir(experiment_root.expanduser().resolve())
    control_video = experiment / "control.mp4"
    resampled = request.trajectory.resample(request.render_fps)
    render_request = SimulationRequest(
        model_xml=request.model_xml,
        trajectory=resampled,
        object_body_names=request.object_body_names,
        required_contact_pairs=request.required_contact_pairs,
        forbidden_contact_pairs=request.forbidden_contact_pairs,
        object_position_goals=request.object_position_goals,
        render_output=control_video,
        render_width=request.render_width,
        render_height=request.render_height,
        render_fps=request.render_fps,
    )
    result = MujocoBackend(camera=camera).simulate(render_request)
    verification = AgentVerifier().verify(result)
    simulation_path = experiment / "simulation.json"
    verification_path = experiment / "verification.json"
    result.to_json(simulation_path)
    _write_json(verification_path, verification.to_dict())
    if not verification.accepted:
        manifest_path = experiment / "manifest.json"
        hashes = {
            "model_xml": _sha256(request.model_xml.resolve()),
            "verification": _sha256(verification_path),
            "simulation": _sha256(simulation_path),
        }
        if control_video.is_file():
            hashes["control_video"] = _sha256(control_video)
        _write_json(
            manifest_path,
            {
                "status": "rejected_control_bundle",
                "camera": camera,
                "robot_base_frame": f"robot_base:{robot_base_name}",
                "hashes": hashes,
                "provenance": _provenance(),
            },
        )
        raise RuntimeError(
            f"control-video simulation was not accepted; inspect {verification_path}"
        )

    stream = _probe_control_video(control_video)
    if stream["frame_count"] != len(resampled.timestamps_s):
        raise RuntimeError(
            f"MuJoCo rendered {stream['frame_count']} frames for "
            f"{len(resampled.timestamps_s)} trajectory samples"
        )
    if abs(stream["fps"] - request.render_fps) > 1e-3:
        raise RuntimeError(
            f"MuJoCo rendered at {stream['fps']:.6g} FPS, expected {request.render_fps}"
        )

    robot_path = experiment / "robot_trajectory.json"
    resampled.to_json(robot_path)
    robot_base = FrameRef(FrameKind.ROBOT_BASE, robot_base_name)
    object_paths: list[Path] = []
    for body_name in request.object_body_names:
        trajectory = result.rigid_body_trajectory(
            body_name,
            resampled.timestamps_s,
            robot_base,
        )
        path = experiment / f"object_trajectory_{body_name}.json"
        trajectory.to_json(path)
        object_paths.append(path)

    manifest_path = experiment / "manifest.json"
    _write_json(
        manifest_path,
        {
            "status": "accepted_control_bundle",
            "camera": camera,
            "robot_base_frame": robot_base.key,
            "stream": stream,
            "source_model": str(request.model_xml.resolve()),
            "hashes": {
                "model_xml": _sha256(request.model_xml.resolve()),
                "control_video": _sha256(control_video),
                "robot_trajectory": _sha256(robot_path),
                "object_trajectories": {
                    path.name: _sha256(path) for path in object_paths
                },
                "verification": _sha256(verification_path),
                "simulation": _sha256(simulation_path),
            },
            "provenance": _provenance(),
        },
    )
    return ControlVideoBundle(
        experiment_dir=experiment,
        control_video=control_video,
        robot_trajectory=robot_path,
        object_trajectories=tuple(object_paths),
        verification_record=verification_path,
        simulation_record=simulation_path,
        manifest=manifest_path,
    )
