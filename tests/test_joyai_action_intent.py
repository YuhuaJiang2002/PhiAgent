from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phiagent.world_model.joyai_action_intent import (
    VISUAL_HARD_GATES,
    build_candidate_audit_template,
    candidate_plan,
    compile_action_prompt,
    evaluate_candidate_audit,
    load_action_intent_config,
    select_visual_candidate,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _config_dict(source: Path, reference: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "task_id": "pick-place-v1",
        "instruction": "Pick up the red block and place it in the blue tray.",
        "embodiment": "the exact silver two-finger robot in the reference image",
        "source_video": str(source),
        "source_sha256": _digest(source.read_bytes()),
        "reference_image": str(reference),
        "reference_sha256": _digest(reference.read_bytes()),
        "coordinate_frame": "camera:test_64x48",
        "width": 64,
        "height": 48,
        "fps": 8,
        "model_frame_count": 17,
        "deliverable_frame_count": 16,
        "motion_authority": "source_demonstration",
        "objects": [
            {
                "object_id": "red-block",
                "description": "one rigid red block",
                "persistence_invariants": ["Keep one block with stable identity."],
            }
        ],
        "phases": [
            {
                "phase_id": "approach-block",
                "action": "approach",
                "start_frame": 0,
                "end_frame": 3,
                "actor": "robot_gripper",
                "object_id": "red-block",
                "target": "red block",
                "contact_state": "approaching",
                "instruction": "Move toward the block without early motion.",
            },
            {
                "phase_id": "grasp-block",
                "action": "grasp",
                "start_frame": 4,
                "end_frame": 7,
                "actor": "robot_gripper",
                "object_id": "red-block",
                "target": "red block",
                "contact_state": "held",
                "instruction": "Close on the same block.",
            },
            {
                "phase_id": "transport-block",
                "action": "transport",
                "start_frame": 8,
                "end_frame": 11,
                "actor": "robot_gripper",
                "object_id": "red-block",
                "target": "blue tray",
                "contact_state": "held",
                "instruction": "Move the held block with no lag.",
            },
            {
                "phase_id": "release-block",
                "action": "release",
                "start_frame": 12,
                "end_frame": 15,
                "actor": "robot_gripper",
                "object_id": "red-block",
                "target": "blue tray",
                "contact_state": "supported",
                "instruction": "Release the block into the tray.",
            },
        ],
        "scene_invariants": ["Preserve the camera and background."],
        "joyai": {
            "seeds": [3, 7],
            "num_inference_steps": 2,
            "output_quality": 95,
            "minimum_phase_confidence": 0.75,
        },
    }


@pytest.fixture
def config(tmp_path: Path):
    source = tmp_path / "source.mkv"
    reference = tmp_path / "reference.png"
    source.write_bytes(b"source-video")
    reference.write_bytes(b"reference-image")
    config_path = tmp_path / "intent.json"
    config_path.write_text(json.dumps(_config_dict(source, reference)))
    return load_action_intent_config(config_path, project_root=tmp_path)


def _audit(config, candidate_id: str, confidences=(0.9, 0.9, 0.9, 0.9)):
    candidate_path = config.source_video.parent / f"{candidate_id}.mp4"
    candidate_path.write_bytes(candidate_id.encode())
    return {
        "schema_version": "1.0.0",
        "candidate_id": candidate_id,
        "candidate_path": str(candidate_path),
        "candidate_sha256": _digest(candidate_path.read_bytes()),
        "observer": "independent-video-action-observer",
        "observer_version": "v1",
        "independent_of_renderer": True,
        "phase_observations": [
            {
                "phase_id": phase.phase_id,
                "intended_action": phase.action,
                "observed_action": phase.action,
                "confidence": confidence,
                "evidence_frames": [phase.start_frame, phase.end_frame],
            }
            for phase, confidence in zip(config.phases, confidences)
        ],
        "hard_gates": {name: True for name in VISUAL_HARD_GATES},
        "human_native_resolution_veto": False,
        "physical_gates": {
            "metric_camera": False,
            "exact_robot_q": False,
            "persistent_metric_object_geometry": False,
            "sensor_or_solver_force": False,
        },
    }


def test_config_requires_contiguous_frame_explicit_phases(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    reference = tmp_path / "reference.png"
    source.write_bytes(b"source")
    reference.write_bytes(b"reference")
    raw = _config_dict(source, reference)
    raw["phases"][1]["start_frame"] = 5  # type: ignore[index]
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="contiguous"):
        load_action_intent_config(config_path, project_root=tmp_path)


def test_config_rejects_prompt_only_motion_authority(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    reference = tmp_path / "reference.png"
    source.write_bytes(b"source")
    reference.write_bytes(b"reference")
    raw = _config_dict(source, reference)
    raw["motion_authority"] = "prompt_only"
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="source_demonstration"):
        load_action_intent_config(config_path, project_root=tmp_path)


def test_compiled_prompt_preserves_motion_and_physical_boundary(config) -> None:
    prompt = compile_action_prompt(config)
    assert "GLOBAL ACTION INTENT" in prompt
    assert "frames 4-7" in prompt
    assert "input demonstration is authoritative" in prompt
    assert "not force, metric contact, calibration, or robot-control evidence" in prompt
    assert "Frames 16-16 are cloned protocol tail padding" in prompt


def test_candidate_plan_is_seeded_and_uses_one_causal_full_stream(config, tmp_path: Path) -> None:
    plans = candidate_plan(
        config,
        output_dir=tmp_path / "run",
        prompt_file=tmp_path / "prompt.txt",
        client_script=tmp_path / "client.py",
        python_executable="python-test",
        server_url="ws://example.test/ws",
    )
    assert [plan["candidate_id"] for plan in plans] == ["seed-3", "seed-7"]
    first = plans[0]["command"]
    assert first.count("--input-video") == 1
    assert first[first.index("--expected-frames") + 1] == "17"
    assert first[first.index("--seed") + 1] == "3"
    assert "--throughput-mode" in first


def test_incomplete_audit_template_fails_closed(config) -> None:
    template = build_candidate_audit_template(config, "seed-3")
    with pytest.raises(ValueError, match="absolute"):
        evaluate_candidate_audit(config, template)


def test_audit_hash_must_match_the_candidate_on_disk(config) -> None:
    audit = _audit(config, "seed-3")
    Path(audit["candidate_path"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="audit hash mismatch"):
        evaluate_candidate_audit(config, audit)


def test_inverse_action_evidence_is_bound_to_phase_intervals(config) -> None:
    audit = _audit(config, "seed-3")
    audit["phase_observations"][0]["evidence_frames"] = [4]  # type: ignore[index]
    with pytest.raises(ValueError, match="outside the frozen interval"):
        evaluate_candidate_audit(config, audit)


def test_high_score_cannot_override_a_failed_visual_gate(config) -> None:
    pretty_but_invalid = _audit(config, "seed-3", (1.0, 1.0, 1.0, 1.0))
    pretty_but_invalid["hard_gates"]["object_identity"] = False  # type: ignore[index]
    valid = _audit(config, "seed-7", (0.8, 0.8, 0.8, 0.8))
    decision = select_visual_candidate(config, [pretty_but_invalid, valid])
    assert decision["selected_candidate"] == "seed-7"
    assert not decision["rollback"]
    assert decision["physical_promotable"] is False


def test_no_candidate_rolls_back_when_any_phase_is_not_recovered(config) -> None:
    audit = _audit(config, "seed-3")
    audit["phase_observations"][2]["observed_action"] = "hold"  # type: ignore[index]
    decision = select_visual_candidate(config, [audit])
    assert decision["selected_candidate"] is None
    assert decision["rollback"]
    assert decision["reason"] == "hard_gate_failed"


def test_input_hashes_are_verified(config) -> None:
    assert config.verify_inputs()["source_video"]["sha256"] == config.source_sha256
    config.source_video.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="hash mismatch"):
        config.verify_inputs()
