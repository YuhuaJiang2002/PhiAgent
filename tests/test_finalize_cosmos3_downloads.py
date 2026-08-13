from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.finalize_cosmos3_downloads import verify_vae, wait_for_markers


def test_wait_for_markers_returns_when_all_are_present(tmp_path: Path) -> None:
    markers = [tmp_path / "nano.completed", tmp_path / "vae.completed"]
    for marker in markers:
        marker.touch()
    assert wait_for_markers(markers, 0.01, 0.1) >= 0


def test_wait_for_markers_times_out_without_false_success(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError, match="did not appear"):
        wait_for_markers([tmp_path / "missing"], 0.001, 0.005)


def test_verify_vae_binds_revision_size_and_sha256(tmp_path: Path) -> None:
    checkpoint = tmp_path / "vae"
    checkpoint.mkdir()
    revision = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
    payload = b"pinned-vae"
    (checkpoint / ".phiagent-model-revision").write_text(revision + "\n")
    (checkpoint / "Wan2.2_VAE.pth").write_bytes(payload)
    report = verify_vae(
        checkpoint,
        revision,
        "Wan2.2_VAE.pth",
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )
    assert report["status"] == "WORKING"
    assert report["size_bytes"] == len(payload)


def test_verify_vae_rejects_wrong_digest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "vae"
    checkpoint.mkdir()
    (checkpoint / ".phiagent-model-revision").write_text("rev\n")
    (checkpoint / "Wan2.2_VAE.pth").write_bytes(b"payload")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_vae(checkpoint, "rev", "Wan2.2_VAE.pth", 7, "0" * 64)
