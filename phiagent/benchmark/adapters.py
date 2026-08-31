"""Optional adapters for H2R judges, RoboWM-Bench, MuJoCo, and real evidence."""

from __future__ import annotations

import json
import re
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
            "--part_scores",
        ]

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
    stage_success_rate: float,
) -> RealEvidence:
    """Normalize a validated, hash-bound real trial; never commands hardware."""

    trial.validate_contents()
    safety = json.loads(trial.safety_log.read_text())
    outcome = json.loads(trial.outcome.read_text())
    collision_count = int(safety["collision_count"])
    emergency_stop = bool(safety["emergency_stop_triggered"])
    force_violation = bool(safety.get("force_limit_violation", False))
    safety_violation = collision_count > 0 or emergency_stop or force_violation
    return RealEvidence(
        adapter=adapter_name,
        robot_id=trial.robot_id,
        hardware_serial=trial.hardware_serial,
        session_id=session_id,
        attempted=True,
        task_success=bool(outcome["task_success"]),
        stage_success_rate=float(stage_success_rate),
        safety_violation=safety_violation,
        human_intervention=bool(outcome.get("human_intervention", False)),
        collision_count=collision_count,
        emergency_stop=emergency_stop,
        force_limit_violation=force_violation,
        blind_review=bool(outcome["blind_review"]),
    )
