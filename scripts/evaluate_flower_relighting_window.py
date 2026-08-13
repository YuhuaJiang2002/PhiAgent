#!/usr/bin/env python3
"""Require geometry, relighting, and human gates before phased expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUTOMATIC_GATES = (
    "all_expected_frames_decoded",
    "flowers_exact_before_encode",
    "prompted_hands_exact_before_encode",
    "protected_interaction_exact_before_encode",
    "outside_robot_exact_before_encode",
    "lora_illumination_signal_nontrivial",
    "lora_illumination_agreement_improved",
    "relighting_residual_temporally_bounded",
)
SEMANTIC_GATES = (
    "complete_human_removal",
    "two_robot_hands_visible",
    "clear_stem_contact",
    "active_flower_identity_preserved",
    "relighting_artifact_free",
    "visible_temporal_flicker_absent",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--geometry-evaluation", type=Path, required=True)
    parser.add_argument("--human-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accepted-geometry-decisions",
        nargs="+",
        default=["ALLOW_RELIGHTING_WINDOW"],
    )
    parser.add_argument(
        "--scope",
        default="confidence-routed relighting on the accepted 17-frame geometry window",
    )
    parser.add_argument("--pass-decision", default="ALLOW_PHASED_FULL_EXPANSION")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("relighting manifest must be a JSON object")
    automatic = manifest.get("automatic_gates")
    if isinstance(automatic, dict) and "all_17_frames_decoded" in automatic:
        automatic = dict(automatic)
        automatic["all_expected_frames_decoded"] = automatic.pop(
            "all_17_frames_decoded"
        )
        manifest = dict(manifest)
        manifest["automatic_gates"] = automatic
    if not isinstance(automatic, dict) or set(automatic) != set(AUTOMATIC_GATES):
        raise ValueError("relighting automatic gate names do not match the contract")
    for name in AUTOMATIC_GATES:
        if type(automatic[name]) is not bool:  # noqa: E721
            raise ValueError(f"automatic gate {name} must be a JSON boolean")
    indices = manifest.get("source_frame_indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError("relighting manifest must contain source_frame_indices")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or "candidate" not in outputs:
        raise ValueError("relighting manifest must declare the lossless candidate")
    return manifest


def validate_geometry_evaluation(
    evaluation: Any, *, accepted_decisions: list[str] | None = None
) -> dict[str, Any]:
    if not isinstance(evaluation, dict):
        raise ValueError("geometry evaluation must be a JSON object")
    accepted = accepted_decisions or ["ALLOW_RELIGHTING_WINDOW"]
    if evaluation.get("decision") not in accepted:
        raise ValueError("geometry evaluation does not allow relighting")
    if evaluation.get("geometry_gate_pass") is not True:
        raise ValueError("geometry evaluation gate is not explicitly true")
    return evaluation


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
        raise ValueError("human review must cover the relighting frames exactly")
    if review.get("candidate_sha256") != candidate_sha256:
        raise ValueError("human review candidate hash does not match")
    semantic = review.get("semantic_gates")
    if not isinstance(semantic, dict) or set(semantic) != set(SEMANTIC_GATES):
        raise ValueError("relighting semantic gate names do not match the contract")
    for name in SEMANTIC_GATES:
        if type(semantic[name]) is not bool:  # noqa: E721
            raise ValueError(f"semantic gate {name} must be a JSON boolean")
    return review


def verify_outputs(manifest: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    verified = {}
    for name, row in manifest["outputs"].items():
        path = artifact_dir / Path(row["path"]).name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"relighting artifact is missing or empty: {path}")
        digest = _sha256(path)
        if digest != row.get("sha256"):
            raise ValueError(f"relighting artifact hash mismatch: {name}")
        verified[name] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    return verified


def main() -> int:
    args = _parser().parse_args()
    paths = {
        "manifest": args.manifest.expanduser().resolve(),
        "geometry_evaluation": args.geometry_evaluation.expanduser().resolve(),
        "human_review": args.human_review.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{name} is missing or empty: {path}")
    manifest = validate_manifest(json.loads(paths["manifest"].read_text()))
    geometry = validate_geometry_evaluation(
        json.loads(paths["geometry_evaluation"].read_text()),
        accepted_decisions=args.accepted_geometry_decisions,
    )
    verified = verify_outputs(manifest, args.artifact_dir.expanduser().resolve())
    review = validate_review(
        json.loads(paths["human_review"].read_text()),
        expected_frames=manifest["source_frame_indices"],
        candidate_sha256=verified["candidate"]["sha256"],
    )
    passed = (
        all(manifest["automatic_gates"].values())
        and all(review["semantic_gates"].values())
        and geometry["geometry_gate_pass"] is True
    )
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "status": "WORKING" if passed else "PARTIAL",
        "scope": args.scope,
        "decision": args.pass_decision if passed else "BLOCK_FULL_EXPANSION",
        "relighting_gate_pass": passed,
        "source_frame_indices": manifest["source_frame_indices"],
        "automatic_gates": manifest["automatic_gates"],
        "semantic_gates": review["semantic_gates"],
        "metrics": manifest["metrics"],
        "verified_outputs": verified,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "human_review": review,
        "limitations": [
            "The direct full-frame LoRA generation was rejected; the accepted candidate routes only bounded luminance into the robot-safe matte.",
            "The lighting change is deliberately subtle and is not a 3D illumination reconstruction.",
            "Full-film work must proceed phase by phase with new flower/contact evidence rather than extrapolating this one instance track."
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
