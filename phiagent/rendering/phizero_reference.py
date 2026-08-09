"""Pinned public reference assets for the PhiZero hand-to-dexterous-hand demo."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

PHIZERO_PAPER_URL = "https://arxiv.org/abs/2607.28624"
PHIZERO_CODE_URL = "https://github.com/yaoyao-jpg/PhiZero"
PHIZERO_CODE_REVISION = "6bc7428f2ad5282e0c1a7b122465957b6abb1edc"
PHIZERO_SITE_URL = "https://phi-zero.github.io/"
PHIZERO_SITE_REVISION = "72fc49fb17b56fab6f7407239b38bdedf7c76546"
_ASSET_ROOT = (
    "https://raw.githubusercontent.com/Phi-Zero/Phi-Zero.github.io/"
    f"{PHIZERO_SITE_REVISION}/assets/videos/phizero"
)


@dataclass(frozen=True)
class PhiZeroReferenceAsset:
    name: str
    sha256: str
    size_bytes: int
    role: str
    case: int
    url: str
    duration_s: float = 3.0
    width: int = 896
    height: int = 512

    def __post_init__(self) -> None:
        if self.role not in {"source", "transferred"}:
            raise ValueError("reference asset role must be source or transferred")
        if self.case <= 0 or self.size_bytes <= 0:
            raise ValueError("reference asset case and size must be positive")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("reference asset sha256 must be lowercase hexadecimal")


def _asset(case: int, role: str, sha256: str, size_bytes: int) -> PhiZeroReferenceAsset:
    name = f"hand2dex_{case}_{role}.mp4"
    return PhiZeroReferenceAsset(
        name=name,
        sha256=sha256,
        size_bytes=size_bytes,
        role=role,
        case=case,
        url=f"{_ASSET_ROOT}/{name}",
    )


PHIZERO_HAND_TRANSFER_ASSETS = (
    _asset(1, "source", "d863b91c4f160d0b634a73a6b996b5362359245910e7402197b78a14e5ad9a03", 176874),
    _asset(
        1,
        "transferred",
        "014d3615ba0448e6500cbe76f63801603943373d7e47ae7562a1af75666dcd39",
        245007,
    ),
    _asset(2, "source", "3245e585ad2351f1340b02ab37e211866960b89a4d24e4b3a78f5c041b064098", 290600),
    _asset(
        2,
        "transferred",
        "2fc8e9c3d48ed987fabc3e90bcc47cb2432433f8137346a0a3a641088c663327",
        435964,
    ),
    _asset(3, "source", "ad33958a7e291aa2edde71f4437de2959bff2950e740a98e0ee87ee2c787a8c0", 185851),
    _asset(
        3,
        "transferred",
        "af57f9606c7d27696df573ccd086eee4e9f3481512e93d09f2a534ea28ccc834",
        256956,
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_asset(path: Path, asset: PhiZeroReferenceAsset) -> None:
    if path.stat().st_size != asset.size_bytes:
        raise ValueError(
            f"{asset.name} has size {path.stat().st_size}, expected {asset.size_bytes}"
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256 != asset.sha256:
        raise ValueError(
            f"{asset.name} has sha256 {actual_sha256}, expected {asset.sha256}"
        )


def prepare_reference_assets(
    destination: Path,
    assets: Iterable[PhiZeroReferenceAsset] = PHIZERO_HAND_TRANSFER_ASSETS,
) -> Path:
    """Download missing assets, reject mismatches, and write a pinned manifest."""

    destination.mkdir(parents=True, exist_ok=True)
    materialized_assets = tuple(assets)
    if not materialized_assets:
        raise ValueError("at least one PhiZero reference asset is required")
    if len({asset.name for asset in materialized_assets}) != len(materialized_assets):
        raise ValueError("PhiZero reference asset names must be unique")

    for asset in materialized_assets:
        output = destination / asset.name
        if output.exists():
            _verify_asset(output, asset)
            continue

        temporary = output.with_suffix(output.suffix + ".part")
        if temporary.exists():
            raise FileExistsError(f"incomplete download already exists: {temporary}")
        request = urllib.request.Request(asset.url, headers={"User-Agent": "PhiAgent-0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("xb") as handle:
                shutil.copyfileobj(response, handle)
        try:
            _verify_asset(temporary, asset)
        except (OSError, ValueError):
            temporary.unlink(missing_ok=True)
            raise
        temporary.replace(output)

    manifest = {
        "schema_version": "1.0.0",
        "paper_url": PHIZERO_PAPER_URL,
        "code_url": PHIZERO_CODE_URL,
        "code_revision": PHIZERO_CODE_REVISION,
        "site_url": PHIZERO_SITE_URL,
        "site_revision": PHIZERO_SITE_REVISION,
        "target": "Figure 8(b): Human Hand to Sharpa Dexterous Hand Transfer",
        "assets": [asdict(asset) for asset in materialized_assets],
    }
    manifest_path = destination / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(manifest_path)
    return manifest_path
