from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "promote_perceptual_video_demo.py"
SPEC = importlib.util.spec_from_file_location("promote_perceptual_video_demo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _inputs(*, attack_pass: bool = True, smear_absent: bool = True):
    manifest = {
        "metrics": {
            "video_seconds": 27.5,
            "frames": 660,
            "compositor_wall_seconds": 50.0,
            "source_flower_dynamic_frame_fraction": 0.98,
            "postencode_lossless_lock_audit": {
                "decoded_frames": 660,
                "native_background_exact_fraction": 0.995,
                "flower_exact_fraction": 1.0,
            },
        }
    }
    audit = {
        "adversarial_audit_pass": attack_pass,
        "candidates": [{"summary": {"gates": {"late_skin_like_fraction": True}}}],
    }
    human = {
        "utility": 1.0,
        "high_resolution_review_pass": True,
        "gates": {
            "human_residue_absent": True,
            "canonical_hand_topology_locked": True,
            "intermittent_hand_smear_absent": smear_absent,
            "long_term_robot_identity_stable": True,
        },
    }
    return manifest, audit, human


def test_build_candidate_passes_only_complete_display_contract() -> None:
    manifest, audit, human = _inputs()
    row = MODULE.build_candidate(
        candidate_id="v1",
        manifest=manifest,
        audit=audit,
        human=human,
        evidence_path="manifest.json",
    )
    assert row.passed is True


def test_attack_or_human_veto_cannot_be_hidden_by_utility() -> None:
    manifest, audit, human = _inputs(attack_pass=False, smear_absent=False)
    human["utility"] = 1_000_000.0
    row = MODULE.build_candidate(
        candidate_id="v1",
        manifest=manifest,
        audit=audit,
        human=human,
        evidence_path="manifest.json",
    )
    assert row.passed is False
