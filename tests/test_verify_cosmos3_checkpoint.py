from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from scripts.verify_cosmos3_checkpoint import (
    commit_verified_completion,
    verify_checkpoint,
)


REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"


def _write_safetensors(path: Path, payload: bytes = b"data") -> int:
    header = b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return len(payload)


def _fixture(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / ".phiagent-model-revision").write_text(REVISION + "\n")
    (checkpoint / "config.json").write_text('{"model":"nano"}\n')
    first = _write_safetensors(checkpoint / "transformer/model-00001.safetensors")
    second = _write_safetensors(checkpoint / "transformer/model-00002.safetensors")
    index = {
        "metadata": {"total_size": first + second},
        "weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"},
    }
    (checkpoint / "transformer/model.safetensors.index.json").write_text(
        json.dumps(index)
    )
    return checkpoint


def test_verifier_binds_revision_index_sizes_and_headers(tmp_path: Path) -> None:
    report = verify_checkpoint(_fixture(tmp_path), REVISION, ["config.json"])
    assert report["status"] == "WORKING"
    assert report["revision"] == REVISION
    assert len(report["indexes"][0]["weights"]) == 2
    assert report["indexes"][0]["expected_total_size_bytes"] == report["indexes"][0]["actual_total_size_bytes"]
    assert report["indexes"][0]["actual_total_file_size_bytes"] > report["indexes"][0]["actual_total_size_bytes"]
    assert report["indexes"][0]["weights"][0]["tensor_count"] == 1
    assert report["required_files"][0]["sha256"]


def test_verifier_rejects_revision_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="revision mismatch"):
        verify_checkpoint(_fixture(tmp_path), "wrong", ["config.json"])


def test_precompletion_verifier_can_gate_before_revision_marker(tmp_path: Path) -> None:
    checkpoint = _fixture(tmp_path)
    (checkpoint / ".phiagent-model-revision").unlink()
    with pytest.raises(ValueError, match="marker is missing"):
        verify_checkpoint(checkpoint, REVISION, ["config.json"])
    report = verify_checkpoint(
        checkpoint,
        REVISION,
        ["config.json"],
        allow_missing_revision_marker=True,
    )
    assert report["status"] == "WORKING"
    assert report["revision_source"].endswith("completion marker pending")


def test_verifier_rejects_missing_or_truncated_weight(tmp_path: Path) -> None:
    checkpoint = _fixture(tmp_path)
    weight = checkpoint / "transformer/model-00002.safetensors"
    weight.unlink()
    with pytest.raises(ValueError, match="missing or empty"):
        verify_checkpoint(checkpoint, REVISION, ["config.json"])

    weight.write_bytes(b"short")
    index_path = checkpoint / "transformer/model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="shorter than its header"):
        verify_checkpoint(checkpoint, REVISION, ["config.json"])


def test_verifier_rejects_header_offset_beyond_payload(tmp_path: Path) -> None:
    checkpoint = _fixture(tmp_path)
    weight = checkpoint / "transformer/model-00002.safetensors"
    header = b'{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    weight.write_bytes(struct.pack("<Q", len(header)) + header + b"data")
    with pytest.raises(ValueError, match="data boundary mismatch"):
        verify_checkpoint(checkpoint, REVISION, ["config.json"])


def test_verified_completion_atomically_commits_revision_and_report_hash(
    tmp_path: Path,
) -> None:
    checkpoint = _fixture(tmp_path)
    (checkpoint / ".phiagent-model-revision").unlink()
    report = verify_checkpoint(
        checkpoint,
        REVISION,
        ["config.json"],
        allow_missing_revision_marker=True,
    )
    report_path = tmp_path / "verification/result.json"
    report_path.parent.mkdir()
    report_path.write_text(json.dumps(report))
    completion = tmp_path / "download.completed"
    commit_verified_completion(checkpoint, REVISION, report_path, completion)
    assert (checkpoint / ".phiagent-model-revision").read_text().strip() == REVISION
    marker = json.loads(completion.read_text())
    assert marker["revision"] == REVISION
    assert len(marker["verification_report_sha256"]) == 64
