from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_embodiment_sota_smoke.py"
    spec = importlib.util.spec_from_file_location("prepare_embodiment_sota_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_cases_require_complete_verified_pairs(tmp_path: Path) -> None:
    assets = []
    for case in (1, 2, 3):
        for role in ("source", "transferred"):
            path = tmp_path / f"case-{case}-{role}.mp4"
            path.write_bytes(f"{case}-{role}".encode())
            assets.append(
                {
                    "case": case,
                    "role": role,
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )

    cases = _module().reference_cases(tmp_path, {"assets": assets})

    assert [case["case"] for case in cases] == [1, 2, 3]


def test_reference_cases_reject_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "case.mp4"
    path.write_bytes(b"content")
    manifest = {
        "assets": [
            {
                "case": case,
                "role": role,
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": "0" * 64,
            }
            for case in (1, 2, 3)
            for role in ("source", "transferred")
        ]
    }

    with pytest.raises(ValueError, match="hash mismatch"):
        _module().reference_cases(tmp_path, manifest)
