#!/usr/bin/env python3
"""Relocate a frozen adaptation manifest and revalidate every asset hash."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.data.adaptation import AdaptationManifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.manifest.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    payload = json.loads(source.read_text())
    experiment_id = str(payload["experiment_id"])
    for asset in payload["assets"]:
        original = Path(str(asset["path"]))
        try:
            relative = Path(*original.parts[original.parts.index(experiment_id) + 1 :])
        except ValueError as exc:
            raise ValueError(
                f"asset path does not contain experiment_id {experiment_id!r}: {original}"
            ) from exc
        asset["path"] = str(relative)
    manifest = AdaptationManifest.from_spec(payload, dataset_root)
    output = args.output.expanduser().resolve()
    manifest.write_json(output)
    print(f"RELOCATED_MANIFEST={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
