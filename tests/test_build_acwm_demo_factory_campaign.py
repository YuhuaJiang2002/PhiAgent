from __future__ import annotations

import json
import sys
from hashlib import sha256

import pytest

from scripts import build_acwm_demo_factory_campaign, run_demo_factory_batch


def _case(tmp_path, name: str, group: str, source: str) -> object:
    root = tmp_path / name
    root.mkdir()
    condition = root / "condition.json"
    condition.write_text("{}\n")
    reviews = root / "reviews"
    reviews.mkdir()
    case = root / "case.json"
    case.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "episode_id": f"{group}-{name}",
                "group_id": group,
                "case_id": f"action-{name}",
                "domain": "agentic-robot-demo-video",
                "seed": 7,
                "license_id": "test-only",
                "source_uri": source,
                "source_sha256": sha256(source.encode()).hexdigest(),
                "condition_manifest_sha256": sha256(condition.read_bytes()).hexdigest(),
                "action_coordinate_frame": "camera:test_pixels",
                "generator": {"id": "fake-acwm", "revision": "revision-one"},
                "evaluator": {"id": "fake-evaluator", "revision": "revision-one"},
                "runner_command": [
                    sys.executable,
                    "scripts/run_agentic_acwm.py",
                    "--condition-manifest",
                    str(condition),
                    "--human-review-dir",
                    str(reviews),
                ],
            }
        )
    )
    return case


def test_builder_emits_valid_complete_tournament_campaign(tmp_path, monkeypatch, capsys) -> None:
    first = _case(tmp_path, "one", "scene-one", "test://scene-one")
    second = _case(tmp_path, "two", "scene-two", "test://scene-two")
    output = tmp_path / "campaign.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_acwm_demo_factory_campaign.py",
            "--campaign-id",
            "two-scene-bootstrap",
            "--case-manifest",
            str(first),
            "--case-manifest",
            str(second),
            "--output",
            str(output),
            "--physical-gpu-index",
            "4",
        ],
    )

    assert build_acwm_demo_factory_campaign.main() == 0
    report = json.loads(capsys.readouterr().out)
    _, contract, recipes, cases, execution = run_demo_factory_batch._load_campaign(output)

    assert report["cases"] == 2
    assert contract.domain == "agentic-robot-demo-video"
    assert tuple(recipes) == contract.recipe_order
    assert len(cases) == 2
    assert execution["device"] == "gpu"
    assert execution["collect_all_recipes"] is True
    assert execution["physical_gpu_index"] == 4


def test_builder_rejects_non_independent_scene_groups(tmp_path, monkeypatch) -> None:
    first = _case(tmp_path, "one", "scene-one", "test://shared")
    second = _case(tmp_path, "two", "scene-one", "test://shared")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_acwm_demo_factory_campaign.py",
            "--campaign-id",
            "invalid-bootstrap",
            "--case-manifest",
            str(first),
            "--case-manifest",
            str(second),
            "--output",
            str(tmp_path / "campaign.json"),
        ],
    )

    with pytest.raises(ValueError, match="two independent"):
        build_acwm_demo_factory_campaign.main()
