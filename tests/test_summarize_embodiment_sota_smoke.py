from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "summarize_embodiment_sota_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("summarize_embodiment_sota_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_summary_stops_seed_expansion_when_every_case_fails() -> None:
    config = {
        "cases": [{"case": case} for case in (1, 2, 3)],
        "thresholds": {
            "motion_preservation": 0.75,
            "target_identity": 0.8,
            "object_consistency": 0.75,
            "temporal_consistency": 0.75,
        },
        "seed_expansion_requires_case_passes": 1,
    }
    result = {
        "motion_preservation": 0.9,
        "target_identity": 0.9,
        "object_consistency": 0.5,
        "temporal_consistency": 0.5,
    }

    summary = _module().summarize_smoke(
        config,
        {case: dict(result) for case in (1, 2, 3)},
    )

    assert summary["case_passes"] == 0
    assert summary["all_gates_pass_rate"] == 0.0
    assert summary["expand_to_remaining_seeds"] is False
    assert summary["decision"] == "stop_raw_wan_animate2_seed_expansion"
