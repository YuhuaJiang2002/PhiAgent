"""Evidence contract for a genuine AC-WM real-robot demonstration."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from phiagent.acwm.robotwin import BWM_EEF_CHANNELS
from phiagent.acwm.schema import ACWMActionCondition


def _file(root: Path, value: object, label: str) -> Path:
    path = Path(str(value)).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"missing or empty {label}: {resolved}")
    return resolved


def _time(value: object, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RealRobotTrialEvidence:
    trial_id: str
    robot_id: str
    hardware_serial: str
    action: Path
    calibration: Path
    initial_state_video: Path
    prediction_video: Path
    execution_video: Path
    telemetry_csv: Path
    safety_log: Path
    outcome: Path
    prediction_created_at: datetime
    execution_started_at: datetime

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, root: Path) -> "RealRobotTrialEvidence":
        for key in ("trial_id", "robot_id", "hardware_serial"):
            if not str(payload.get(key, "")).strip():
                raise ValueError(f"real-robot trial requires {key}")
        prediction_created = _time(payload.get("prediction_created_at"), "prediction_created_at")
        execution_started = _time(payload.get("execution_started_at"), "execution_started_at")
        if prediction_created > execution_started:
            raise ValueError("prediction must be generated before physical execution starts")
        trial = cls(
            trial_id=str(payload["trial_id"]),
            robot_id=str(payload["robot_id"]),
            hardware_serial=str(payload["hardware_serial"]),
            action=_file(root, payload.get("action"), "action condition"),
            calibration=_file(root, payload.get("calibration"), "calibration"),
            initial_state_video=_file(
                root, payload.get("initial_state_video"), "initial-state video"
            ),
            prediction_video=_file(root, payload.get("prediction_video"), "prediction video"),
            execution_video=_file(root, payload.get("execution_video"), "execution video"),
            telemetry_csv=_file(root, payload.get("telemetry_csv"), "telemetry CSV"),
            safety_log=_file(root, payload.get("safety_log"), "safety log"),
            outcome=_file(root, payload.get("outcome"), "outcome review"),
            prediction_created_at=prediction_created,
            execution_started_at=execution_started,
        )
        trial.validate_contents()
        return trial

    def validate_contents(self) -> None:
        action = ACWMActionCondition.from_json(self.action)
        if not action.coordinate_frame.startswith(f"robot_base:{self.robot_id}"):
            raise ValueError("action coordinate frame does not identify the physical robot")
        if tuple(action.channels) != BWM_EEF_CHANNELS:
            raise ValueError("real-robot action channels do not match the reviewed BWM EEF contract")
        calibration = json.loads(self.calibration.read_text())
        if calibration.get("robot_id") != self.robot_id:
            raise ValueError("calibration robot_id does not match the trial")
        if not calibration.get("camera_frame") or not calibration.get("robot_base_frame"):
            raise ValueError("calibration must name camera and robot-base frames")
        with self.telemetry_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"timestamp_s", *BWM_EEF_CHANNELS}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"telemetry lacks required columns: {sorted(missing)}")
            rows = list(reader)
        if len(rows) < 2:
            raise ValueError("telemetry requires at least two synchronized samples")
        safety = json.loads(self.safety_log.read_text())
        required_safety = {"preflight_passed", "collision_count", "emergency_stop_triggered"}
        if not required_safety <= set(safety):
            raise ValueError("safety log is missing required fields")
        if safety["preflight_passed"] is not True:
            raise ValueError("physical safety preflight did not pass")
        outcome = json.loads(self.outcome.read_text())
        if outcome.get("blind_review") is not True or not isinstance(
            outcome.get("task_success"), bool
        ):
            raise ValueError("outcome must contain a blind boolean task-success review")

    def to_evidence(self) -> dict[str, Any]:
        safety = json.loads(self.safety_log.read_text())
        outcome = json.loads(self.outcome.read_text())
        files = {
            key: value
            for key, value in {
                "action": self.action,
                "calibration": self.calibration,
                "initial_state_video": self.initial_state_video,
                "prediction_video": self.prediction_video,
                "execution_video": self.execution_video,
                "telemetry_csv": self.telemetry_csv,
                "safety_log": self.safety_log,
                "outcome": self.outcome,
            }.items()
        }
        return {
            "trial_id": self.trial_id,
            "robot_id": self.robot_id,
            "hardware_serial": self.hardware_serial,
            "prediction_created_at": self.prediction_created_at.isoformat(),
            "execution_started_at": self.execution_started_at.isoformat(),
            "task_success": outcome["task_success"],
            "safety_violation_free": (
                int(safety["collision_count"]) == 0
                and safety["emergency_stop_triggered"] is False
            ),
            "files": {
                key: {"path": str(path), "sha256": _sha256(path)}
                for key, path in files.items()
            },
        }


def compile_real_robot_demo(
    trials: tuple[RealRobotTrialEvidence, ...], *, minimum_trials: int = 3
) -> dict[str, Any]:
    if len(trials) < minimum_trials:
        raise ValueError(f"real-robot demo requires at least {minimum_trials} repeated trials")
    identifiers = [trial.trial_id for trial in trials]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("real-robot trial identifiers must be unique")
    evidence = [trial.to_evidence() for trial in trials]
    return {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "trial_count": len(trials),
        "task_success_rate": sum(item["task_success"] for item in evidence) / len(evidence),
        "safety_violation_free_rate": sum(
            item["safety_violation_free"] for item in evidence
        )
        / len(evidence),
        "trials": evidence,
        "claim_boundary": (
            "This is a real-robot demo record. SOTA promotion separately requires at least "
            "20 paired trials and dominance over every declared baseline."
        ),
    }
