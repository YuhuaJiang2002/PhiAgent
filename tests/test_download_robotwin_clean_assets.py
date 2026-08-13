from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "download_robotwin_clean_assets.py"
    )
    spec = importlib.util.spec_from_file_location("download_robotwin_clean_assets", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_asset_bundle_is_pinned_without_background_texture() -> None:
    module = _module()

    assert set(module.FILES) == {"embodiments.zip", "objects.zip"}
    assert sum(size for size, _ in module.FILES.values()) == 3_957_637_862
