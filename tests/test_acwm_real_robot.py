import csv
import hashlib
import json
from pathlib import Path

import pytest

from phiagent.acwm.real_robot import RealRobotTrialEvidence, compile_real_robot_demo
from phiagent.acwm.robotwin import BWM_EEF_CHANNELS
from phiagent.acwm.schema import ACWMActionCondition, ActionRepresentation


def _write_trial(root: Path, trial_id: str) -> Path:
    root.mkdir()
    action = ACWMActionCondition(
        label="lift-up",
        instruction="Lift the object",
        timeline="grasp; lift; hold",
        representation=ActionRepresentation.EEF_ABSOLUTE,
        coordinate_frame="robot_base:arm-a/tool0",
        timestamps_s=(0.0, 0.1),
        channels=BWM_EEF_CHANNELS,
        values=(tuple([0.0] * 14), tuple([0.1] * 14)),
    )
    action.to_json(root / "action.json")
    (root / "calibration.json").write_text(
        json.dumps(
            {
                "robot_id": "arm-a",
                "camera_frame": "camera:overhead",
                "robot_base_frame": "robot_base:arm-a",
            }
        )
    )
    for name in ("initial.mp4", "prediction.mp4", "execution.mp4"):
        (root / name).write_bytes(b"video-evidence")
    with (root / "telemetry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("timestamp_s", *BWM_EEF_CHANNELS))
        writer.writeheader()
        for timestamp in (0.0, 0.1):
            writer.writerow({"timestamp_s": timestamp, **dict.fromkeys(BWM_EEF_CHANNELS, 0.0)})
    (root / "safety.json").write_text(
        json.dumps(
            {
                "preflight_passed": True,
                "collision_count": 0,
                "emergency_stop_triggered": False,
                "force_limit_violation": False,
            }
        )
    )
    (root / "outcome.json").write_text(
        json.dumps(
            {
                "blind_review": True,
                "task_success": True,
                "human_intervention": False,
                "stage_success_rate": 1.0,
            }
        )
    )
    def sha256(name: str) -> str:
        return hashlib.sha256((root / name).read_bytes()).hexdigest()

    (root / "pre-registration.json").write_text(
        json.dumps(
            {
                "trial_id": trial_id,
                "case_id": "smoke-case",
                "protocol_id": "phiagent-real-robot-blind-v0.1",
                "robot_id": "arm-a",
                "hardware_serial": "serial-001",
                "adapter_name": "smoke-recorded-adapter",
                "trial_index": 0,
                "registered_at": "2026-08-11T10:00:30+08:00",
                "artifact_hashes": {
                    "action": sha256("action.json"),
                    "calibration": sha256("calibration.json"),
                    "initial_state_video": sha256("initial.mp4"),
                    "prediction_video": sha256("prediction.mp4"),
                },
            }
        )
    )
    descriptor = {
        "trial_id": trial_id,
        "robot_id": "arm-a",
        "hardware_serial": "serial-001",
        "pre_registered_case_manifest": "pre-registration.json",
        "action": "action.json",
        "calibration": "calibration.json",
        "initial_state_video": "initial.mp4",
        "prediction_video": "prediction.mp4",
        "execution_video": "execution.mp4",
        "telemetry_csv": "telemetry.csv",
        "safety_log": "safety.json",
        "outcome": "outcome.json",
        "prediction_created_at": "2026-08-11T10:00:00+08:00",
        "execution_started_at": "2026-08-11T10:01:00+08:00",
    }
    path = root / "trial.json"
    path.write_text(json.dumps(descriptor))
    return path


def test_real_robot_evidence_requires_prediction_before_execution(tmp_path: Path) -> None:
    descriptor = {
        "trial_id": "trial-1",
        "robot_id": "arm-a",
        "hardware_serial": "serial-001",
        "prediction_created_at": "2026-08-11T10:02:00+08:00",
        "execution_started_at": "2026-08-11T10:01:00+08:00",
    }

    with pytest.raises(ValueError, match="before physical execution"):
        RealRobotTrialEvidence.from_dict(descriptor, root=tmp_path)


def test_real_robot_demo_hashes_repeated_physical_trials(tmp_path: Path) -> None:
    trials = []
    for index in range(3):
        descriptor = _write_trial(tmp_path / f"trial-{index}", f"trial-{index}")
        trials.append(
            RealRobotTrialEvidence.from_dict(
                json.loads(descriptor.read_text()), root=descriptor.parent
            )
        )

    result = compile_real_robot_demo(tuple(trials))

    assert result["status"] == "WORKING"
    assert result["trial_count"] == 3
    assert result["task_success_rate"] == 1.0
    assert result["safety_violation_free_rate"] == 1.0
    assert "pre_registered_case_manifest" in result["trials"][0]["files"]
    assert len(result["trials"][0]["files"]["execution_video"]["sha256"]) == 64


def test_real_robot_rejects_artifact_changed_after_registration(tmp_path: Path) -> None:
    descriptor = _write_trial(tmp_path / "trial-tampered", "trial-tampered")
    (descriptor.parent / "prediction.mp4").write_bytes(b"changed-after-registration")

    with pytest.raises(ValueError, match="changed before trial collection"):
        RealRobotTrialEvidence.from_dict(
            json.loads(descriptor.read_text()), root=descriptor.parent
        )
