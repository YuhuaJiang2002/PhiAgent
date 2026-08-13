from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_robotwin_runtime_view.py"
    )
    spec = importlib.util.spec_from_file_location("build_robotwin_runtime_view", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_robotwin_runtime_view_revisions_are_pinned() -> None:
    module = _module()

    assert module.ROBOTWIN_COMMIT == "266f3aadf505a4f7fe9af0faa41a20f5f47cd123"
    assert module.ASSET_REVISION == "c15cc97be71e35244b6605d2d84c187f8565cc4d"
