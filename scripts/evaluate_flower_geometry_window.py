#!/usr/bin/env python3
"""Promote a real flower geometry window only after automatic and human gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUTOMATIC_GATES = (
    "complete_human_removal_proxy",
    "two_robot_hands_visible_proxy",
    "active_stem_contact_proxy",
    "support_bouquet_contact_proxy",
    "active_flower_identity_exact_before_encode",
)
SEMANTIC_GATES = (
    "complete_human_removal",
    "two_robot_hands_visible",
    "clear_stem_contact",
    "active_flower_identity_preserved",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--human-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scope",
        default="real 17-frame critical flower geometry window only",
    )
    parser.add_argument("--pass-decision", default="ALLOW_RELIGHTING_WINDOW")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    automatic = manifest.get("automatic_gates")
    if not isinstance(automatic, dict):
        raise ValueError("manifest must contain automatic_gates")
    if set(automatic) != set(AUTOMATIC_GATES):
        raise ValueError("manifest automatic gate names do not match the geometry contract")
    for name in AUTOMATIC_GATES:
        if type(automatic[name]) is not bool:  # noqa: E721 - require JSON boolean
            raise ValueError(f"automatic gate {name} must be a JSON boolean")
    indices = manifest.get("source_frame_indices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(index) is not int for index in indices)  # noqa: E721
        or indices != sorted(set(indices))
    ):
        raise ValueError("source_frame_indices must be unique ordered integers")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or "geometry_candidate" not in outputs:
        raise ValueError("manifest must declare a geometry_candidate output")
    return manifest


def validate_review(
    review: Any, *, expected_frames: list[int], candidate_sha256: str
) -> dict[str, Any]:
    if not isinstance(review, dict) or not str(review.get("reviewer", "")).strip():
        raise ValueError("human review must have a non-empty reviewer")
    review = dict(review)
    reviewed_frames = review.get("reviewed_source_frames")
    reviewed_range = review.get("reviewed_source_frame_range")
    if reviewed_frames is None and reviewed_range is not None:
        if (
            not isinstance(reviewed_range, list)
            or len(reviewed_range) != 3
            or any(type(value) is not int for value in reviewed_range)  # noqa: E721
            or reviewed_range[0] < 0
            or reviewed_range[1] <= reviewed_range[0]
            or reviewed_range[2] <= 0
        ):
            raise ValueError(
                "reviewed_source_frame_range must be [start, end_exclusive, step]"
            )
        reviewed_frames = list(range(*reviewed_range))
        review["reviewed_source_frames"] = reviewed_frames
    if reviewed_frames != expected_frames:
        raise ValueError("human review must cover the manifest source frames exactly")
    if review.get("candidate_sha256") != candidate_sha256:
        raise ValueError("human review candidate hash does not match the geometry candidate")
    gates = review.get("semantic_gates")
    if not isinstance(gates, dict) or set(gates) != set(SEMANTIC_GATES):
        raise ValueError("human review semantic gate names do not match the contract")
    for name in SEMANTIC_GATES:
        if type(gates[name]) is not bool:  # noqa: E721 - require JSON boolean
            raise ValueError(f"semantic gate {name} must be a JSON boolean")
    return review


def verify_outputs(manifest: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for name, row in manifest["outputs"].items():
        if not isinstance(row, dict) or not str(row.get("path", "")):
            raise ValueError(f"manifest output {name} has no path")
        expected = str(row.get("sha256", ""))
        if len(expected) != 64:
            raise ValueError(f"manifest output {name} has no valid SHA-256")
        local_path = artifact_dir / Path(row["path"]).name
        if not local_path.is_file() or local_path.stat().st_size == 0:
            raise ValueError(f"artifact is missing or empty: {local_path}")
        actual = _sha256(local_path)
        if actual != expected:
            raise ValueError(f"artifact hash mismatch for {name}: {local_path}")
        verified[name] = {
            "path": str(local_path),
            "sha256": actual,
            "bytes": local_path.stat().st_size,
        }
    return verified


def main() -> int:
    args = _parser().parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    review_path = args.human_review.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    for name, path in {"manifest": manifest_path, "human_review": review_path}.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{name} is missing or empty: {path}")
    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    verified_outputs = verify_outputs(manifest, artifact_dir)
    candidate_sha256 = verified_outputs["geometry_candidate"]["sha256"]
    review = validate_review(
        json.loads(review_path.read_text()),
        expected_frames=manifest["source_frame_indices"],
        candidate_sha256=candidate_sha256,
    )
    automatic_pass = all(manifest["automatic_gates"].values())
    semantic_pass = all(review["semantic_gates"].values())
    gate_pass = automatic_pass and semantic_pass
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "status": "WORKING" if gate_pass else "PARTIAL",
        "scope": args.scope,
        "decision": args.pass_decision if gate_pass else "BLOCK_RELIGHTING",
        "geometry_gate_pass": gate_pass,
        "source_frame_indices": manifest["source_frame_indices"],
        "automatic_gates": manifest["automatic_gates"],
        "semantic_gates": review["semantic_gates"],
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "human_review": {"path": str(review_path), "sha256": _sha256(review_path)},
        },
        "verified_outputs": verified_outputs,
        "human_review": review,
        "limitations": [
            f"Acceptance is scoped to the {len(manifest['source_frame_indices'])} reviewed source frames and does not establish full-film quality.",
            "Only the active stem track has persistent instance identity; support contact uses the bouquet union.",
            "Relighting must be evaluated for geometry regression before any full-film expansion.",
        ],
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
