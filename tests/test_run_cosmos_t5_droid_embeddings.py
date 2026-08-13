from __future__ import annotations

import pickle

import numpy as np
import pytest

from scripts.run_cosmos_t5_droid_embeddings import _embedding_manifest


def _dataset(tmp_path, *, width: int = 1024):
    for folder in ("videos", "metas", "t5_xxl"):
        (tmp_path / folder).mkdir()
    (tmp_path / "videos/sample.mp4").write_bytes(b"video")
    (tmp_path / "metas/sample.txt").write_text("close the drawer")
    with (tmp_path / "t5_xxl/sample.pickle").open("wb") as handle:
        pickle.dump([np.ones((7, width), dtype=np.float16)], handle)
    return tmp_path


def test_embedding_manifest_requires_official_shape(tmp_path) -> None:
    manifest = _embedding_manifest(_dataset(tmp_path))
    assert manifest[0]["sample_id"] == "sample"
    assert manifest[0]["shape"] == [7, 1024]
    assert manifest[0]["dtype"] == "float16"


def test_embedding_manifest_rejects_wrong_width(tmp_path) -> None:
    with pytest.raises(ValueError, match="invalid T5 embedding contract"):
        _embedding_manifest(_dataset(tmp_path, width=768))
