from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import slice_packed_instance_tracks as slicer


def _write_track(path: Path) -> np.ndarray:
    masks = np.arange(2 * 5 * 4, dtype=np.uint8).reshape(2, 5, 4)
    np.savez_compressed(
        path,
        masks_packed=masks,
        instance_ids=np.asarray(["flower-01", "flower-02"]),
        object_ids=np.asarray([1, 2], dtype=np.int32),
        source_frame_indices=np.arange(10, 15, dtype=np.int32),
        height=np.asarray(4, dtype=np.int32),
        width=np.asarray(8, dtype=np.int32),
        bitorder=np.asarray("little"),
    )
    return masks


def test_slice_preserves_selected_masks_and_reindexes_video_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.npz"
    masks = _write_track(source)
    output = tmp_path / "output"
    monkeypatch.setattr(slicer, "_git_state", lambda _root: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slice_packed_instance_tracks.py", "--input-track", str(source),
            "--start-frame", "1", "--end-frame", "4", "--output-dir", str(output),
        ],
    )

    assert slicer.main() == 0
    with np.load(output / "sliced-instance-tracks-packed.npz", allow_pickle=False) as result:
        assert np.array_equal(result["masks_packed"], masks[:, 1:4])
        assert result["source_frame_indices"].tolist() == [0, 1, 2]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["slice"]["parent_source_frame_indices"] == [11, 12, 13]


def test_slice_rejects_out_of_bounds_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.npz"
    _write_track(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slice_packed_instance_tracks.py", "--input-track", str(source),
            "--start-frame", "3", "--end-frame", "6", "--output-dir", str(tmp_path / "out"),
        ],
    )
    with pytest.raises(ValueError, match="exceeds"):
        slicer.main()
