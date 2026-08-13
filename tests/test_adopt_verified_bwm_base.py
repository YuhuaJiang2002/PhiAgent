from pathlib import Path

import pytest

from scripts import adopt_verified_bwm_base


def test_verify_source_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        adopt_verified_bwm_base.verify_source(tmp_path)


def test_verify_source_rejects_wrong_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adopt_verified_bwm_base, "EXPECTED_SHA256", {"weight.bin": "0" * 64})
    (tmp_path / "weight.bin").write_bytes(b"not-the-reviewed-weight")

    with pytest.raises(ValueError, match="hash mismatch"):
        adopt_verified_bwm_base.verify_source(tmp_path)
