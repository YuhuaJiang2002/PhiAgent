from __future__ import annotations

import json
import sys

from scripts import train_ego_bottle_repair_policy


def _score(value: float) -> dict[str, float]:
    return {
        "background_lock": value,
        "object_lock": value,
        "subject_replacement": value,
        "robot_identity": value,
        "motion_preservation": value,
        "temporal_consistency": value,
        "epl_minimum": value,
    }


def test_train_ego_bottle_router_runs_held_action_folds(
    tmp_path, monkeypatch
) -> None:
    evolutions = []
    for window in ("window-00", "window-01"):
        for action in ("pour-bottle", "shake-bottle", "handover-bottle"):
            path = (
                tmp_path
                / window
                / "variants"
                / action
                / "agent-evaluation-probe"
                / "evolution.json"
            )
            path.parent.mkdir(parents=True)
            payload = {
                "action_label": action,
                "rounds": [
                    {
                        "repair": {
                            "name": "raw-h3",
                            "support_dilation_pixels": 0,
                            "alpha_blur_sigma": 0.0,
                        },
                        "scorecard": _score(0.6),
                    },
                    {
                        "repair": {
                            "name": "tight-control-support-lock",
                            "support_dilation_pixels": 0,
                            "alpha_blur_sigma": 3.0,
                        },
                        "scorecard": _score(0.7),
                    },
                    {
                        "repair": {
                            "name": "soft-control-support-lock",
                            "support_dilation_pixels": 0,
                            "alpha_blur_sigma": 7.0,
                        },
                        "scorecard": _score(0.8),
                    },
                ],
            }
            path.write_text(json.dumps(payload))
            evolutions.append(path)

    root = tmp_path / "training"
    arguments = ["train_ego_bottle_repair_policy.py"]
    for path in evolutions:
        arguments.extend(("--evolution", str(path)))
    arguments.extend(("--experiment-root", str(root)))
    monkeypatch.setattr(sys, "argv", arguments)

    assert train_ego_bottle_repair_policy.main() == 0
    experiment = next(root.iterdir())
    manifest = json.loads((experiment / "manifest.json").read_text())
    checkpoint = json.loads((experiment / "policy.json").read_text())

    assert manifest["status"] == "completed"
    assert manifest["metrics"]["selection_accuracy"] == 1.0
    assert manifest["metrics"]["selected_non_regression_rate"] == 1.0
    assert checkpoint["method"] == "ego_bottle_repair_ridge_utility_ranker"
