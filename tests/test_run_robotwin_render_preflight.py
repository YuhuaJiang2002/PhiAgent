from __future__ import annotations

from pathlib import Path


def test_robotwin_render_preflight_requires_explicit_marker() -> None:
    source = Path("scripts/run_robotwin_render_preflight.py").read_text()

    assert '"Render Well" in log' in source
    assert '"Render Error" not in log' in source
    assert 'environment["CUDA_VISIBLE_DEVICES"]' in source
