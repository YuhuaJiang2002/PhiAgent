from __future__ import annotations

import json
import sys

from scripts import run_demo_factory_batch


def test_cpu_batch_collects_guarded_training_records(tmp_path, monkeypatch) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, pathlib, sys\n"
        "recipe=json.load(open(sys.argv[1]))\n"
        "root=pathlib.Path(sys.argv[2]); video=root/'candidate.mp4'\n"
        "video.write_bytes(('video-'+recipe['recipe_id']).encode())\n"
        "score={'raw':0.5,'generic':0.6,'targeted':0.9}[recipe['recipe_id']]\n"
        "print(json.dumps({'video':str(video),'metrics':{'motion':score,'object':score},"
        "'human_review_passed':True,'cost_units':1.0,'diagnoses':[]}))\n"
    )
    case_manifest = tmp_path / "case.json"
    case_manifest.write_text("{}\n")
    campaign = tmp_path / "campaign.json"
    command = [sys.executable, str(worker), "{recipe_manifest}", "{attempt_dir}"]
    campaign.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "campaign_id": "cpu-smoke",
                "contract": {
                    "domain": "robot-demo",
                    "baseline_recipe_id": "raw",
                    "recipe_order": ["raw", "generic", "targeted"],
                    "context_fields": ["motion", "object"],
                    "metric_weights": {"motion": 1.0, "object": 1.0},
                    "hard_thresholds": {"motion": 0.75, "object": 0.75},
                    "non_regression_tolerances": {"motion": 0.01, "object": 0.01},
                    "cost_budget_units": 3.0,
                    "cost_weight": 0.01,
                    "rejection_penalty": 2.0,
                    "human_review_required": True,
                },
                "recipes": [
                    {"recipe_id": recipe, "command": command, "parameters": {}, "estimated_cost_units": 1.0}
                    for recipe in ("raw", "generic", "targeted")
                ],
                "cases": [
                    {
                        "episode_id": "episode-one",
                        "group_id": "scene-one",
                        "domain": "robot-demo",
                        "manifest": str(case_manifest),
                        "seed": 42,
                    }
                ],
                "execution": {
                    "device": "cpu",
                    "maximum_attempts_per_episode": 3,
                    "collect_all_recipes": True,
                },
            }
        )
    )
    output = tmp_path / "outputs"
    monkeypatch.setattr(run_demo_factory_batch, "_git_state", lambda: {"head": "test"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_demo_factory_batch.py",
            "--campaign",
            str(campaign),
            "--experiment-root",
            str(output),
        ],
    )

    assert run_demo_factory_batch.main() == 0
    experiment = next(output.iterdir())
    manifest = json.loads((experiment / "manifest.json").read_text())
    records = [json.loads(line) for line in (experiment / "episodes.jsonl").read_text().splitlines()]
    accepted = json.loads((experiment / "accepted-video-index.json").read_text())

    assert manifest["status"] == "accepted"
    assert manifest["gpu"]["used"] is False
    assert [row["recipe"]["recipe_id"] for row in records] == ["raw", "generic", "targeted"]
    assert accepted["videos"][0]["recipe_id"] == "targeted"
