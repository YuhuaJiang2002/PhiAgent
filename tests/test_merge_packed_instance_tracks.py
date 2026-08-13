from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import merge_packed_instance_tracks as merger


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(path: Path, masks: np.ndarray) -> None:
    path.mkdir()
    packed_path = path / "tracks.npz"
    packed = np.packbits(masks.reshape(2, 3, -1), axis=2, bitorder="little")
    np.savez_compressed(
        packed_path,
        masks_packed=packed,
        instance_ids=np.asarray(["left", "right"]),
        object_ids=np.asarray([101, 102], dtype=np.int32),
        source_frame_indices=np.asarray([0, 1, 2], dtype=np.int32),
        height=np.asarray(3, dtype=np.int32),
        width=np.asarray(4, dtype=np.int32),
        bitorder=np.asarray("little"),
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "coordinate_frame": "camera:source_video_pixels",
                "outputs": {"packed_masks": {"sha256": _sha256(packed_path)}},
            }
        )
    )


def test_merge_selects_each_instance_without_mask_mixing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_masks = np.zeros((2, 3, 3, 4), dtype=bool)
    first_masks[0, :, 0, 0] = True
    first_masks[1, :, 0, 1] = True
    second_masks = np.zeros_like(first_masks)
    second_masks[0, :, 1, 0] = True
    second_masks[1, :, 2, 3] = True
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "merged"
    _write_run(first, first_masks)
    _write_run(second, second_masks)

    monkeypatch.setattr(merger, "_git_state", lambda _root: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_packed_instance_tracks.py",
            "--track",
            f"{first}::left",
            "--track",
            f"{second}::right",
            "--output-dir",
            str(output),
        ],
    )

    assert merger.main() == 0
    with np.load(output / "merged-instance-tracks-packed.npz", allow_pickle=False) as packed:
        assert packed["instance_ids"].astype(str).tolist() == ["left", "right"]
        assert np.array_equal(
            packed["masks_packed"][0],
            np.packbits(first_masks[0].reshape(3, -1), axis=1, bitorder="little"),
        )
        assert np.array_equal(
            packed["masks_packed"][1],
            np.packbits(second_masks[1].reshape(3, -1), axis=1, bitorder="little"),
        )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "PARTIAL"
    assert [row["instance_id"] for row in manifest["selection"]] == ["left", "right"]
