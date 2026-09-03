"""Optional adapters for H2R judges, RoboWM-Bench, MuJoCo, and real evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phiagent.acwm.real_robot import RealRobotTrialEvidence
from phiagent.benchmark.schema import BenchmarkCase, RealEvidence, SimulationEvidence
from phiagent.simulation.base import SimulationResult


ROBOWM_TASKS = {
    "pick": "Franka-pick",
    "put_on_plate": "Franka-put_on_plate",
    "discard_trash": "Franka-discard_trash",
    "put_in_drawer": "Franka-put_in_drawer",
    "press_button": "Franka-press_button",
    "close_drawer": "Franka-close_drawer",
    "pull_and_push": "Franka-pull_and_push",
}


@dataclass(frozen=True)
class HarnessEvalWAdapter:
    """Pinned external adapter for visual-world evidence trees.

    HarnessEval-W is intentionally supplementary: its visually inferred physics
    and causality scores cannot set PhiAgent's L4 or L5 gates.
    """

    checkout: Path
    expected_revision: str

    def preflight(self) -> dict[str, Any]:
        checkout = self.checkout.expanduser().resolve()
        if not (checkout / "pyproject.toml").is_file():
            raise ValueError(f"HarnessEval-W checkout is incomplete: {checkout}")
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        )
        revision = completed.stdout.strip()
        if revision != self.expected_revision:
            raise ValueError(
                f"HarnessEval-W revision mismatch: expected {self.expected_revision}, got {revision}"
            )
        executable = shutil.which("harnesseval")
        return {
            "checkout": str(checkout),
            "revision": revision,
            "executable": executable,
            "status": "ready" if executable else "checkout_ready_cli_missing",
            "claim_boundary": (
                "Observation, transition, and persistence evidence only; this adapter "
                "cannot satisfy metric action, simulation, or real-robot gates."
            ),
        }

    def command(
        self,
        *,
        results: Path,
        model_id: str,
        run_root: Path,
        manifest: Path,
        plan_root: Path,
    ) -> list[str]:
        state = self.preflight()
        executable = state["executable"]
        if not executable:
            raise ValueError("HarnessEval-W CLI is not installed in the active environment")
        return [
            str(executable),
            "eval",
            "--results",
            str(results.expanduser().resolve()),
            "--model-id",
            model_id,
            "--run-root",
            str(run_root.expanduser().resolve()),
            "--manifest",
            str(manifest.expanduser().resolve()),
            "--plan-root",
            str(plan_root.expanduser().resolve()),
        ]


def h2r_judge_packet(case: BenchmarkCase) -> dict[str, Any]:
    """Return the public-evidence contract without leaking it into generation prompts."""

    if not case.annotation:
        raise ValueError("H2R judge packet requires a structured case annotation")
    return {
        "protocol": "phiagent_h2r_reproduction_arxiv_2608.13049v1",
        "case_id": case.case_id,
        "source_uri": case.source_uri,
        "generated_evidence_frames": 25,
        "source_evidence_frames_for_contact": 25,
        "rubric": {
            "0": "absent or contradicted evidence",
            "1": "weak evidence",
            "2": "partial completion",
            "3": "mostly correct with minor defect or ambiguity",
            "4": "clear, complete, and stable evidence",
        },
        "target": case.target.to_dict(),
        "annotation": case.annotation,
        "required_output": {
            "goal_scores": "one integer 0..4 per goal predicate ID",
            "action_scores": "one integer 0..4 per action event ID",
            "source_grounded": "boolean hard gate",
            "contact_scores": "one integer 0..4 per applicable contact dimension",
            "embodiment_hard_failure": "boolean hard gate",
            "embodiment_scores": "five integer 0..4 morphology scores",
            "evidence_frame_indices": "0-based indices into each 25-frame sequence",
            "rationale": "short per-metric evidence statements",
        },
    }


@dataclass(frozen=True)
class RoboWMBenchAdapter:
    checkout: Path
    expected_revision: str

    def verify_frozen_episode(
        self,
        *,
        trajectory_root: Path,
        episode_index: int,
        episode_sha256: str,
        pose_sha256: str,
    ) -> dict[str, Any]:
        """Fail closed when a pinned upstream trajectory input has changed."""

        root = trajectory_root.expanduser().resolve()
        candidates = (root / f"episode_{episode_index:06d}.json", root / f"{episode_index}.json")
        episode = next((path for path in candidates if path.is_file()), None)
        if episode is None:
            raise ValueError(f"frozen RoboWM episode {episode_index} is missing from {root}")
        pose = root / "pose.jsonl"
        if not pose.is_file():
            raise ValueError(f"frozen RoboWM pose file is missing: {pose}")

        def digest(path: Path) -> str:
            value = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(chunk)
            return value.hexdigest()

        actual_episode = digest(episode)
        actual_pose = digest(pose)
        if actual_episode != episode_sha256 or actual_pose != pose_sha256:
            raise ValueError("frozen RoboWM episode or pose hash mismatch")
        return {
            "episode": str(episode),
            "episode_sha256": actual_episode,
            "pose": str(pose),
            "pose_sha256": actual_pose,
            "status": "verified",
        }

    def preflight(self) -> dict[str, Any]:
        checkout = self.checkout.expanduser().resolve()
        script = checkout / "scripts" / "robot" / "eval_franka.py"
        if not script.is_file():
            raise ValueError(f"RoboWM-Bench evaluator not found: {script}")
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        )
        revision = completed.stdout.strip()
        if revision != self.expected_revision:
            raise ValueError(
                f"RoboWM-Bench revision mismatch: expected {self.expected_revision}, got {revision}"
            )
        return {
            "checkout": str(checkout),
            "revision": revision,
            "script": str(script),
            "status": "ready",
        }

    def command(
        self,
        *,
        task: str,
        trajectory_root: Path,
        output_root: Path,
        episode_index: int,
        device: str = "cpu",
    ) -> list[str]:
        self.preflight()
        if task not in ROBOWM_TASKS:
            raise ValueError(f"unsupported RoboWM-Bench task alias: {task}")
        if episode_index < 0:
            raise ValueError("episode_index cannot be negative")
        return [
            sys.executable,
            str(self.checkout.resolve() / "scripts" / "robot" / "eval_franka.py"),
            "--task",
            ROBOWM_TASKS[task],
            "--json_root",
            str(trajectory_root.expanduser().resolve()),
            "--output_root",
            str(output_root.expanduser().resolve()),
            "--episode_index",
            str(episode_index),
            "--device",
            device,
            "--headless",
            "--enable_cameras",
            "--part_scores",
        ]

    def runtime_preflight(self) -> dict[str, Any]:
        """Inspect the optional Isaac Lab runtime without importing it."""

        source = self.preflight()
        modules = {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "isaacsim", "isaaclab")
        }
        nvidia_smi = shutil.which("nvidia-smi")
        gpu_names: list[str] = []
        if nvidia_smi is not None:
            completed = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                gpu_names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        checks = {
            "python_3_11": sys.version_info[:2] == (3, 11),
            "torch": modules["torch"],
            "isaacsim": modules["isaacsim"],
            "isaaclab": modules["isaaclab"],
            "nvidia_gpu": bool(gpu_names),
        }
        return {
            **source,
            "status": "ready" if all(checks.values()) else "not_ready",
            "checks": checks,
            "python": sys.version,
            "gpu_names": gpu_names,
            "claim_boundary": "Preflight only; no Isaac episode was executed.",
        }

    def parse_task_outcome_log(self, text: str) -> SimulationEvidence:
        """Import upstream outcome only; do not fabricate a full physical gate."""

        attempts = re.findall(r"Total attempts:\s*(\d+)", text)
        successes = re.findall(r"Successes:\s*(\d+)", text)
        if not attempts or not successes:
            raise ValueError("RoboWM-Bench log lacks final attempt/success counters")
        attempt_count = int(attempts[-1])
        success_count = int(successes[-1])
        if attempt_count != 1 or success_count not in {0, 1}:
            raise ValueError("per-case import requires exactly one upstream replay")
        task_success = success_count == 1
        return SimulationEvidence(
            backend="robowm_bench_isaaclab_task_outcome",
            attempted=True,
            physical_gate_complete=False,
            physically_valid=False,
            task_success=task_success,
            stage_success_rate=float(task_success),
            contact_success_rate=0.0,
            ik_success_rate=0.0,
            joint_limit_violation_rate=0.0,
            velocity_violation_rate=0.0,
            collision_rate=0.0,
            singularity_rate=0.0,
            source_revision=self.expected_revision,
        )


def simulation_evidence_from_phiagent(
    result: SimulationResult,
    *,
    ik_success_rate: float,
    velocity_violation_rate: float,
    singularity_rate: float,
    stage_success_rate: float,
    contact_success_rate: float,
    source_revision: str,
) -> SimulationEvidence:
    """Normalize a PhiAgent physics result after explicit kinematic checks."""

    metric_rates = {
        "ik_success_rate": ik_success_rate,
        "velocity_violation_rate": velocity_violation_rate,
        "singularity_rate": singularity_rate,
        "stage_success_rate": stage_success_rate,
        "contact_success_rate": contact_success_rate,
    }
    if any(not 0.0 <= float(value) <= 1.0 for value in metric_rates.values()):
        raise ValueError("normalized simulation rates must be in [0, 1]")
    return SimulationEvidence(
        backend=result.backend,
        attempted=True,
        physical_gate_complete=True,
        physically_valid=result.physically_valid,
        task_success=result.task_success is True,
        stage_success_rate=float(stage_success_rate),
        contact_success_rate=float(contact_success_rate),
        ik_success_rate=float(ik_success_rate),
        joint_limit_violation_rate=float(bool(result.joint_limit_violations)),
        velocity_violation_rate=float(velocity_violation_rate),
        collision_rate=float(bool(result.collision_events)),
        singularity_rate=float(singularity_rate),
        source_revision=source_revision,
    )


def real_evidence_from_recorded_trial(
    trial: RealRobotTrialEvidence,
    *,
    adapter_name: str,
    session_id: str,
    protocol_id: str,
    trial_index: int,
    reviewer_id_hash: str,
    pre_registered: bool,
    method_blind_code: str | None = None,
    eligibility_checks: dict[str, bool] | None = None,
) -> RealEvidence:
    """Normalize a validated, hash-bound real trial; never commands hardware."""

    trial.validate_contents()
    safety = json.loads(trial.safety_log.read_text())
    outcome = json.loads(trial.outcome.read_text())
    collision_count = int(safety["collision_count"])
    emergency_stop = bool(safety["emergency_stop_triggered"])
    force_violation = bool(safety.get("force_limit_violation", False))
    safety_violation = collision_count > 0 or emergency_stop or force_violation
    file_evidence = trial.to_evidence()["files"]
    return RealEvidence(
        protocol_id=protocol_id,
        pre_registered=pre_registered,
        trial_index=trial_index,
        reviewer_id_hash=reviewer_id_hash,
        artifact_hashes={name: item["sha256"] for name, item in file_evidence.items()},
        adapter=adapter_name,
        robot_id=trial.robot_id,
        hardware_serial=trial.hardware_serial,
        session_id=session_id,
        attempted=True,
        task_success=bool(outcome["task_success"]),
        stage_success_rate=float(outcome["stage_success_rate"]),
        safety_violation=safety_violation,
        human_intervention=bool(outcome.get("human_intervention", False)),
        collision_count=collision_count,
        emergency_stop=emergency_stop,
        force_limit_violation=force_violation,
        blind_review=bool(outcome["blind_review"]),
        trial_id=trial.trial_id,
        method_blind_code=method_blind_code,
        eligibility_checks=dict(eligibility_checks or {}),
    )
