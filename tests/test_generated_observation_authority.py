from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from phiagent.perception.generated_observation_authority import (
    ObservationSource,
    ObservationSourceKind,
    audit_observation_sources,
    audit_vlm_ensemble,
    validate_vlm_observation_report,
)
from scripts.compile_foundation_contact_pipeline import (
    _generated_observation_diagnostic,
)


def _vlm_report(name: str, *, flower_motion: str = "moving") -> dict[str, object]:
    observations = []
    for frame, timestamp in ((0, 0.0), (24, 1.0)):
        observations.append(
            {
                "frame_index": frame,
                "timestamp_s": timestamp,
                "left_hand_visibility": "partial",
                "right_hand_visibility": "clear",
                "finger_integrity": "ambiguous" if frame else "normal",
                "left_contact": "unknown",
                "right_contact": "grasping",
                "flower_motion": flower_motion,
                "camera_motion": "static",
                "evidence_note": "visible overlap only",
            }
        )
    return {
        "source_video_sha256": "a" * 64,
        "model": {"name": name},
        "evidence_class": "foundation_model_estimate",
        "independent_physical_groups": 0,
        "physical_gate_eligible": False,
        "observations": observations,
        "non_observable": {
            "metric_depth": True,
            "absolute_camera_scale": True,
            "full_q_qdot": True,
            "contact_force": True,
            "force_closure": True,
        },
    }


def test_model_completion_never_becomes_a_physical_group() -> None:
    source = ObservationSource(
        source_id="qwen-vlm",
        kind=ObservationSourceKind.VLM_INFERENCE,
        acquisition_group_id=None,
        source_sha256="a" * 64,
        timeline="frame:source_video",
        coordinate_frame="camera:source_pixels",
        synchronized=True,
        physically_captured=False,
        generated_from_source_sha256="b" * 64,
    )
    audit = audit_observation_sources([source])
    assert audit["independent_physical_group_count"] == 0
    assert not any(audit["authority"].values())
    assert audit["required_new_sources"]["metric_camera"]


def test_real_synchronized_sources_contribute_only_their_own_authority() -> None:
    rgbd = ObservationSource(
        source_id="rgbd-1",
        kind=ObservationSourceKind.RGBD_SENSOR,
        acquisition_group_id="capture-a",
        source_sha256="a" * 64,
        timeline="ptp:robot-clock",
        coordinate_frame="camera:rgbd_optical",
        synchronized=True,
        physically_captured=True,
        metric_calibration_passed=True,
    )
    telemetry = ObservationSource(
        source_id="telemetry-1",
        kind=ObservationSourceKind.ROBOT_TELEMETRY,
        acquisition_group_id="capture-a",
        source_sha256="b" * 64,
        timeline="ptp:robot-clock",
        coordinate_frame="robot:g1_base",
        synchronized=True,
        physically_captured=True,
        exact_asset_bound=True,
    )
    audit = audit_observation_sources([rgbd, telemetry])
    assert audit["independent_physical_group_count"] == 1
    assert audit["authority"] == {
        "metric_camera": True,
        "robot_trajectory": True,
        "contact_forces": False,
    }


def test_vlm_report_fails_closed_on_claimed_metric_observation() -> None:
    report = _vlm_report("qwen3-vl-4b")
    report["non_observable"]["metric_depth"] = False
    with pytest.raises(ValueError, match="unmeasured physical"):
        validate_vlm_observation_report(report)


def test_vlm_ensemble_surfaces_disagreement_and_static_contact() -> None:
    first = _vlm_report("qwen3-vl-4b")
    second = _vlm_report("qwen3-vl-8b", flower_motion="static")
    audit = audit_vlm_ensemble([first, second])
    assert audit["independent_physical_groups"] == 0
    assert audit["categorical_agreement_fraction"] < 1.0
    assert len(audit["review_frames"]) == 2
    assert all(row["contact_with_static_flower"] for row in audit["review_frames"])


def test_vlm_ensemble_rejects_source_hash_mismatch() -> None:
    first = _vlm_report("qwen3-vl-4b")
    second = deepcopy(_vlm_report("qwen3-vl-8b"))
    second["source_video_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="different source"):
        audit_vlm_ensemble([first, second])


def test_compiler_binds_vlm_ensemble_without_upgrading_authority(tmp_path) -> None:
    first = tmp_path / "qwen4b.json"
    second = tmp_path / "qwen8b.json"
    first.write_text("{}")
    second.write_text("{}")
    report = {
        "status": "PARTIAL",
        "inputs": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (first, second)
        ],
        "audit": {
            "passed": True,
            "source_video_sha256": "a" * 64,
            "evidence_class": "foundation_model_estimate",
            "physical_gate_eligible": False,
            "independent_physical_groups": 0,
            "models": ["Qwen3-VL-4B", "Qwen3-VL-8B"],
            "common_frames": 14,
            "categorical_agreement_fraction": 0.25,
            "review_frames": [{"frame_index": 10}],
        },
    }
    result = _generated_observation_diagnostic(
        report,
        report_sha256="c" * 64,
        source_video_sha256="a" * 64,
    )
    assert result["bound"] is True
    assert result["independent_physical_groups"] == 0
    assert result["physical_gate_eligible"] is False
