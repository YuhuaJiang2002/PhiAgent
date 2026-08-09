from __future__ import annotations

import hashlib
import json

import pytest

from phiagent.rendering.phizero_reference import (
    PHIZERO_HAND_TRANSFER_ASSETS,
    PHIZERO_SITE_REVISION,
    PhiZeroReferenceAsset,
    prepare_reference_assets,
)


def _local_asset(tmp_path, content: bytes = b"reference-video") -> PhiZeroReferenceAsset:
    source = tmp_path / "source.mp4"
    source.write_bytes(content)
    return PhiZeroReferenceAsset(
        name="hand2dex_test_source.mp4",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        role="source",
        case=1,
        url=source.as_uri(),
    )


def test_official_reference_set_pins_three_source_transfer_pairs() -> None:
    assert len(PHIZERO_HAND_TRANSFER_ASSETS) == 6
    assert {(asset.case, asset.role) for asset in PHIZERO_HAND_TRANSFER_ASSETS} == {
        (1, "source"),
        (1, "transferred"),
        (2, "source"),
        (2, "transferred"),
        (3, "source"),
        (3, "transferred"),
    }
    assert all(PHIZERO_SITE_REVISION in asset.url for asset in PHIZERO_HAND_TRANSFER_ASSETS)


def test_prepare_reference_assets_downloads_and_records_manifest(tmp_path) -> None:
    asset = _local_asset(tmp_path)
    output = tmp_path / "output"

    manifest_path = prepare_reference_assets(output, (asset,))

    assert (output / asset.name).read_bytes() == b"reference-video"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["site_revision"] == PHIZERO_SITE_REVISION
    assert manifest["target"].startswith("Figure 8(b)")
    assert manifest["assets"] == [
        {
            "case": 1,
            "duration_s": 3.0,
            "height": 512,
            "name": asset.name,
            "role": "source",
            "sha256": asset.sha256,
            "size_bytes": asset.size_bytes,
            "url": asset.url,
            "width": 896,
        }
    ]


def test_prepare_reference_assets_rejects_existing_mismatch(tmp_path) -> None:
    asset = _local_asset(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / asset.name).write_bytes(b"wrong")

    with pytest.raises(ValueError, match="has size"):
        prepare_reference_assets(output, (asset,))
