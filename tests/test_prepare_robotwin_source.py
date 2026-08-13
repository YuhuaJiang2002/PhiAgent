from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_robotwin_source.py"
    spec = importlib.util.spec_from_file_location("prepare_robotwin_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_robotwin_source_revisions_are_exactly_pinned() -> None:
    module = _module()

    assert module.ROBOTWIN_COMMIT == "266f3aadf505a4f7fe9af0faa41a20f5f47cd123"
    assert module.XPOLICYLAB_COMMIT == "c37109c500be67d0dea6b36bf7337bbd26e763cd"
    assert set(module.ARCHIVES) == {"robotwin", "xpolicylab"}
