from __future__ import annotations

import base64
import hashlib
import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_droid_raw_smoke.py"
    spec = importlib.util.spec_from_file_location("download_droid_raw_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_droid_smoke_md5_uses_gcs_base64_encoding(tmp_path: Path) -> None:
    path = tmp_path / "artifact"
    path.write_bytes(b"raw-droid")
    expected = base64.b64encode(hashlib.md5(b"raw-droid").digest()).decode("ascii")

    assert _module().md5_base64(path) == expected


def test_droid_smoke_files_have_exact_expected_total() -> None:
    module = _module()

    assert len(module.FILES) == 5
    assert sum(size for size, _ in module.FILES.values()) == 46_338_367
