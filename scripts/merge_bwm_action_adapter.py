#!/usr/bin/env python3
"""Overlay a trained BWM action encoder onto an audited full checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.expanduser().resolve()
    adapter = args.adapter.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    for path in (base, adapter):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"checkpoint is missing: {path}")
    for path in (output, manifest):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")

    from safetensors.torch import load_file, save_file

    base_state = load_file(str(base), device="cpu")
    adapter_state = load_file(str(adapter), device="cpu")
    prefix = "pipe.action_encoder."
    invalid = sorted(key for key in adapter_state if not key.startswith(prefix))
    if invalid:
        raise ValueError(f"adapter contains non-action tensors: {invalid[:5]}")
    if not adapter_state:
        raise ValueError("adapter contains no tensors")
    nonfinite = [
        key for key, tensor in adapter_state.items() if not bool(tensor.isfinite().all())
    ]
    if nonfinite:
        raise ValueError(f"adapter contains non-finite tensors: {nonfinite}")
    replaced = sorted(set(base_state).intersection(adapter_state))
    added = sorted(set(adapter_state).difference(base_state))
    base_state.update(adapter_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        base_state,
        str(output),
        metadata={
            "format": "pt",
            "phiagent_merge": "BWM full checkpoint plus action adapter",
        },
    )
    payload = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base": {"path": str(base), "bytes": base.stat().st_size, "sha256": _sha256(base)},
        "adapter": {
            "path": str(adapter),
            "bytes": adapter.stat().st_size,
            "sha256": _sha256(adapter),
            "tensor_count": len(adapter_state),
            "parameter_count": sum(tensor.numel() for tensor in adapter_state.values()),
        },
        "merge": {"replaced_keys": replaced, "added_keys": added},
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
            "tensor_count": len(base_state),
        },
        "claim_boundary": "This is a deterministic weight overlay, not evaluation evidence.",
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
