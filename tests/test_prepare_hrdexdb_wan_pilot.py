from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_hrdexdb_wan_pilot.py"
    )
    spec = importlib.util.spec_from_file_location("prepare_hrdexdb_wan_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wan_pilot_uses_train_reference_and_validation_objects() -> None:
    manifest = {
        "split": {
            "train": ["apple", "banana"],
            "validation": ["beige_brush"],
            "test": ["cactus"],
        }
    }

    assert _module().evaluation_objects(manifest) == ("beige_brush",)
