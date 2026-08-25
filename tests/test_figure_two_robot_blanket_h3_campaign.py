import json
from pathlib import Path

import pytest

from scripts import evaluate_figure_two_robot_blanket_h3_candidate as evaluator
from scripts import prepare_figure_two_robot_blanket_h3_campaign as preparer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_blanket_h3_visual_gates_are_fail_closed_and_unique() -> None:
    config = json.loads(
        (
            PROJECT_ROOT
            / "configs/physical_video/figure_two_robot_blanket_fold_photorealistic_v2.json"
        ).read_text(encoding="utf-8")
    )
    gate_ids = config["visual_hard_gates"]

    assert len(gate_ids) == len(set(gate_ids))
    assert evaluator.AUTO_GATE_IDS < set(gate_ids)
    assert evaluator.CONCLUSIVE_AUTO_GATE_IDS < evaluator.AUTO_GATE_IDS
    assert {
        "single_uninterrupted_shot_no_cut_crossfade_or_teleport",
        "fixed_camera_background_and_lighting",
    }.isdisjoint(evaluator.CONCLUSIVE_AUTO_GATE_IDS)
    assert config["decision_policy"]["visual_acceptance"].startswith(
        "All visual_hard_gates must be PASS"
    )
    assert "recorded_real_robot_execution" in config["physical_promotion_gates"]


def test_invalid_gate_state_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid gate state"):
        evaluator._gate("example", "PENDING")


def test_preparer_freezes_hash_bound_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "campaign"
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_figure_two_robot_blanket_h3_campaign.py", "--output-dir", str(output)],
    )

    assert preparer.main() == 0

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    spec = json.loads(
        (output / "inputs/generation-spec.json").read_text(encoding="utf-8")
    )
    prompt = (output / "inputs/generation-prompt.txt").read_text(encoding="utf-8")
    assert manifest["candidate_budget"] == 4
    assert spec["model"]["revision"] == evaluator.EXPECTED_REVISION
    assert spec["cases"][0]["harness_version"] == preparer.HARNESS_VERSION
    assert "Plan SHA-256:" in prompt
    assert "missing evidence rejects" in prompt
    assert preparer._sha256(output / "inputs/initial-frame.png") == manifest["hashes"][
        "initial_frame"
    ]
    assert spec["compute"]["num_gpus"] == 4
    assert spec["compute"]["physical_gpu_indices"] == [0, 1, 2, 3]
    assert spec["compute"]["ulysses_degree"] == 4
