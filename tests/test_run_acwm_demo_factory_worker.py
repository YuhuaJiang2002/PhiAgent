from __future__ import annotations

import json
import sys
from hashlib import sha256

from scripts import run_acwm_demo_factory_worker


def test_acwm_worker_normalizes_native_trace(tmp_path, monkeypatch, capsys) -> None:
    fake_runner = tmp_path / "fake_acwm.py"
    fake_runner.write_text(
        "import json, pathlib, sys\n"
        "def value(flag): return sys.argv[sys.argv.index(flag)+1]\n"
        "root=pathlib.Path(value('--experiment-root')); run=root/'run'; run.mkdir(parents=True)\n"
        "video=run/'candidate.mp4'; video.write_bytes(b'real-acwm-candidate')\n"
        "case=value('--case'); trace=run/'trace.json'\n"
        "score={'action_adherence':0.91,'embodiment_consistency':0.92,"
        "'object_interaction':0.93,'temporal_consistency':0.94,"
        "'background_consistency':0.95,'human_review_passed':True,'diagnoses':[]}\n"
        "trace.write_text(json.dumps({'status':'accepted','best_candidate_indices':[0],"
        "'candidates':[{'candidate_index':0,'proposal':{'case_id':case},"
        "'result':{'output':str(video)},'scorecard':score}]}))\n"
        "print(json.dumps({'status':'accepted','trace':str(trace)}))\n"
    )
    condition = tmp_path / "condition.json"
    condition.write_text("{}\n")
    review_dir = tmp_path / "human-review"
    review_dir.mkdir()
    case = tmp_path / "case.json"
    case.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "episode_id": "scene-one-slide-right",
                "group_id": "scene-one",
                "case_id": "slide-right",
                "seed": 42,
                "license_id": "test-license",
                "source_uri": "test://source",
                "source_sha256": "0" * 64,
                "condition_manifest_sha256": sha256(condition.read_bytes()).hexdigest(),
                "action_coordinate_frame": "camera:test_pixels",
                "generator": {"id": "fake-acwm", "revision": "test-revision"},
                "evaluator": {"id": "fake-evaluator", "revision": "test-revision"},
                "runner_command": [
                    sys.executable,
                    str(fake_runner),
                    "--condition-manifest",
                    str(condition),
                    "--human-review-dir",
                    str(review_dir),
                ],
            }
        )
    )
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "recipe_id": "raw-generator",
                "parameters": {"prompt_suffix": "Keep one robot.", "cost_units": 1.25},
            }
        )
    )
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    monkeypatch.setenv("PHIAGENT_PHYSICAL_GPU", "4")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_acwm_demo_factory_worker.py",
            "--case-manifest",
            str(case),
            "--recipe-manifest",
            str(recipe),
            "--attempt-dir",
            str(attempt),
            "--seed",
            "42",
        ],
    )

    assert run_acwm_demo_factory_worker.main() == 0
    result = json.loads(capsys.readouterr().out)
    evidence = json.loads((attempt / "acwm-factory-adapter.json").read_text())

    assert result["metrics"]["action_adherence"] == 0.91
    assert result["human_review_passed"] is True
    assert result["cost_units"] == 1.25
    assert evidence["physical_gpu"] == 4
    assert evidence["trace_status"] == "accepted"
    assert evidence["action_coordinate_frame"] == "camera:test_pixels"
