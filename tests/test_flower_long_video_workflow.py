import hashlib
import json

from phiagent.workflows import GraphStatus
from phiagent.workflows.flower import FLOWER_VISUAL_GATES, build_flower_long_video_workflow


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, *, topology_pass=True, late_contact_recall=1.0):
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"synthetic-video-fixture")
    manifest = {
        "claim_scope": "perceptually plausible synthetic display data",
        "physical_evidence": False,
        "coordinate_frames": {
            "source": "camera:source_native_1280x720",
            "timeline": "absolute_frame_index:full_source_660",
        },
        "limitations": ["no physical claims"],
        "metrics": {
            "frames": 660,
            "video_seconds": 27.5,
            "compositor_wall_seconds": 10.0,
            "frame_transition_delta_p95": 2.0,
            "route_boundary_transition_deltas": {"479": 8.0, "559": 3.0},
            "source_flower_dynamic_frame_fraction": 0.98,
            "postencode_lossless_lock_audit": {
                "decoded_frames": 660,
                "flower_exact_fraction": 1.0,
                "native_background_exact_fraction": 0.995,
            },
        },
        "outputs": {
            "review_video": {"path": str(video), "sha256": _sha256(video)},
        },
    }
    manifest_path = _write_json(tmp_path / "manifest.json", manifest)
    audit = {
        "adversarial_audit_pass": True,
        "candidates": [
            {
                "adversarial": {
                    "all_attacks_detected": True,
                    "sampled_frames": 585,
                    "gates": {
                        "color_attack_detected": True,
                        "contact_attack_detected": True,
                        "structure_ghost_attack_detected": True,
                        "topology_attack_detected": True,
                    },
                },
                "summary": {
                    "sections": {
                        "at_or_after_20_seconds": {
                            "projected_contact_recall": late_contact_recall,
                            "metrics": {
                                "hand_replacement_coverage": {"violation_fraction": 0.2}
                            },
                        }
                    }
                },
                "wall_seconds": 3.0,
            }
        ],
    }
    audit_path = _write_json(tmp_path / "audit.json", audit)
    review = {
        "reviewer": "test",
        "utility": 0.9,
        "decision": "PASS_FOR_SYNTHETIC_DISPLAY_SCOPE" if topology_pass else "REJECT",
        "gates": {
            "human_residue_absent": True,
            "canonical_hand_topology_locked": topology_pass,
            "intermittent_hand_smear_absent": True,
            "long_term_robot_identity_stable": True,
        },
        "high_resolution_review_pass": topology_pass,
        "notes": [],
    }
    review_path = _write_json(tmp_path / "review.json", review)
    candidate_id = "fixture-candidate"
    promotion = {
        "status": "DISPLAY_READY" if topology_pass else "PARTIAL",
        "selected_candidate": candidate_id if topology_pass else None,
        "physical_evidence": False,
        "inputs": [
            {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            {"path": str(audit_path), "sha256": _sha256(audit_path)},
            {"path": str(review_path), "sha256": _sha256(review_path)},
        ],
    }
    promotion_path = _write_json(tmp_path / "promotion.json", promotion)
    return {
        "workflow": "flower-long-video",
        "candidate_id": candidate_id,
        "workspace_root": str(tmp_path),
        "claim_scope": "perceptually plausible synthetic display data",
        "evidence": {
            "manifest": str(manifest_path),
            "adversarial_audit": str(audit_path),
            "human_review": str(review_path),
            "promotion": str(promotion_path),
        },
    }


def test_flower_workflow_accepts_only_visual_scope(tmp_path):
    app = build_flower_long_video_workflow()
    result = app.invoke(_fixture(tmp_path), config={"thread_id": "flower-pass"})

    assert result.status is GraphStatus.COMPLETED
    assert result.state["workflow_outcome"]["status"] == "DISPLAY_READY"
    assert result.state["workflow_outcome"]["physical_evidence"] is False
    assert set(result.state["quality_gates"]) == set(FLOWER_VISUAL_GATES)
    assert len(result.state["risk_windows"]) == 1
    assert result.state["physical_promotion"]["eligible"] is False


def test_flower_workflow_plans_architecture_change_without_threshold_tuning(tmp_path):
    app = build_flower_long_video_workflow()
    result = app.invoke(_fixture(tmp_path, topology_pass=False), config={"thread_id": "flower-fail"})

    assert result.state["workflow_outcome"]["status"] == "PARTIAL"
    assert "canonical_hand_topology_locked" in result.state["workflow_outcome"][
        "failed_hard_visual_gates"
    ]
    plan = result.state["next_quality_iteration"]
    assert plan["threshold_changes_allowed"] is False
    assert any(
        action["architecture"] == "persistent_embodiment_state_with_canonical_topology"
        for action in plan["actions"]
    )


def test_flower_workflow_rejects_missing_visible_contact_without_physical_claim(tmp_path):
    app = build_flower_long_video_workflow()
    result = app.invoke(
        _fixture(tmp_path, late_contact_recall=0.5), config={"thread_id": "flower-contact-fail"}
    )

    assert result.state["workflow_outcome"]["status"] == "PARTIAL"
    assert "late_projected_contact_visible" in result.state["workflow_outcome"][
        "failed_hard_visual_gates"
    ]
    assert result.state["physical_promotion"]["eligible"] is False
    assert any(
        action["architecture"] == "contact_phase_object_motion_coupling"
        for action in result.state["next_quality_iteration"]["actions"]
    )
