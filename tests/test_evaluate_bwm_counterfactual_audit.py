from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluate_bwm_counterfactual_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_bwm_counterfactual_audit", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_limitations_report_dynamic_trial_and_seed_counts() -> None:
    records = [
        {
            "independent_unit_id": f"wipe_table/physical-episode-{index}",
            "trial_id": f"wipe_table/episode_{index}::action-swap",
            "seed": 42,
        }
        for index in range(20)
    ]

    limitations = _module().audit_limitations(records)

    assert limitations[0] == (
        "The suite contains 20 independent source episodes across 1 task: wipe_table."
    )
    assert limitations[1] == (
        "The suite uses one inference seed (42); seed sensitivity is not estimated."
    )


def test_limitations_describe_multiple_seeds_as_repeated_measurements() -> None:
    records = [
        {
            "independent_unit_id": "wipe_table/physical-episode-0",
            "trial_id": "wipe_table/episode_0::action-swap",
            "seed": seed,
        }
        for seed in (42, 314159)
    ]

    limitations = _module().audit_limitations(records)

    assert "averaged before bootstrap" in limitations[1]
