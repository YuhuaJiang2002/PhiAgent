from __future__ import annotations

from scripts.summarize_robotwin_reset_inventory import stable_seed_records


def test_stable_seed_records_excludes_blocked_or_nondeterministic_runs() -> None:
    records = [
        {
            "status": "WORKING",
            "same_seed": 2,
            "deterministic_reset": True,
            "different_seed_changes_scene": True,
        },
        {
            "status": "BLOCKED",
            "same_seed": 4,
            "deterministic_reset": True,
            "different_seed_changes_scene": True,
        },
        {
            "status": "WORKING",
            "same_seed": 6,
            "deterministic_reset": False,
            "different_seed_changes_scene": True,
        },
    ]

    assert sorted(stable_seed_records(records)) == [2]
