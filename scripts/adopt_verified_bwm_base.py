#!/usr/bin/env python3
"""Adopt an existing Wan2.2 base cache only after exact file verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.adapters import BWM_BASE_MODEL_REVISION

EXPECTED_SHA256 = {
    "diffusion_pytorch_model-00001-of-00003.safetensors": "720b06c4ade5e87c1246bba8ac95b664c638749cd9b102cf84d823bb44c026a1",
    "diffusion_pytorch_model-00002-of-00003.safetensors": "09ec5ef720d8396f6cfa51fbdcbdb2327e37722afd6e89fd38f1e7e5e782c283",
    "diffusion_pytorch_model-00003-of-00003.safetensors": "6306f7894c345de9093ad588771c2abfaeb668a81f7a6d9a918bd26ba3568e49",
    "diffusion_pytorch_model.safetensors.index.json": "bfa2337f1163e195d24151a72298daf34a620543898109be47e414c8daa5b3fe",
    "models_t5_umt5-xxl-enc-bf16.pth": "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
    "Wan2.2_VAE.pth": "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36",
    "config.json": "d1fea36899d00c2501b836c13ad65af56e2f9529ba622e50886d3f5c3e6c02bc",
    "configuration.json": "a6b66993e9da0feaba8d42d06b21ad9cfaf7d8b591f32fd639ae35b7f5d2d859",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(source: Path) -> dict[str, dict[str, object]]:
    evidence: dict[str, dict[str, object]] = {}
    for name, expected in EXPECTED_SHA256.items():
        path = source / name
        if not path.is_file():
            raise ValueError(f"base-model cache is missing {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"base-model hash mismatch for {name}: {actual} != {expected}")
        evidence[name] = {"bytes": path.stat().st_size, "sha256": actual}
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"verified model destination already exists: {destination}")
    evidence = verify_source(source)
    destination.mkdir(parents=True)
    for name in EXPECTED_SHA256:
        os.symlink(source / name, destination / name)
    (destination / ".phiagent-model-revision").write_text(BWM_BASE_MODEL_REVISION + "\n")
    manifest = {
        "schema_version": "1.0.0",
        "model": "Wan-AI/Wan2.2-TI2V-5B",
        "revision": BWM_BASE_MODEL_REVISION,
        "source": str(source),
        "destination": str(destination),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "files": evidence,
        "storage": "absolute symlinks to byte-verified source files",
    }
    (destination / ".phiagent-verification.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
