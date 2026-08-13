from __future__ import annotations

import importlib.util
import struct
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "download_droid_mapping_shards.py"
    )
    spec = importlib.util.spec_from_file_location("download_droid_mapping_shards", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mapping_targets_are_pinned_to_expected_shards() -> None:
    module = _module()

    assert module.TARGETS[21] == (
        "r2d2_faceblur-train.tfrecord-00007-of-00031",
        2,
    )
    assert sum(size for size, _ in module.FILES.values()) == 136_080_147


def test_tfrecord_reader_selects_requested_record(tmp_path: Path) -> None:
    path = tmp_path / "records"
    with path.open("wb") as handle:
        for payload in (b"first", b"second"):
            handle.write(struct.pack("<Q", len(payload)))
            handle.write(b"\0" * 4)
            handle.write(payload)
            handle.write(b"\0" * 4)

    assert _module().tfrecord_at(path, 1) == b"second"


def test_parallel_shards_use_distinct_partial_paths(tmp_path: Path) -> None:
    module = _module()

    first = module.partial_path(tmp_path / "record-00007-of-00031")
    second = module.partial_path(tmp_path / "record-00023-of-00031")

    assert first != second
    assert first.name == "record-00007-of-00031.partial"
