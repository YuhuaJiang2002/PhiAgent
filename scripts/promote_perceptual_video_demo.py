#!/usr/bin/env python3
"""Finalize a perceptual long-video demo without making physical claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.agent.perceptual_video_harness import (  # noqa: E402
    PERCEPTUAL_DEMO_GATES,
    PerceptualCandidate,
    select_display_candidate,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_candidate(
    *,
    candidate_id: str,
    manifest: dict[str, Any],
    audit: dict[str, Any],
    human: dict[str, Any],
    evidence_path: str,
) -> PerceptualCandidate:
    metrics = manifest["metrics"]
    audit_candidate = audit["candidates"][0]
    late_gates = audit_candidate["summary"]["gates"]
    human_gates = human["gates"]
    postencode = metrics["postencode_lossless_lock_audit"]
    gates = {
        "duration_at_least_20_seconds": float(metrics["video_seconds"]) >= 20.0,
        "full_video_decodes": int(postencode["decoded_frames"]) == int(metrics["frames"]),
        "native_background_locked": float(postencode["native_background_exact_fraction"]) >= 0.99,
        "flower_pixels_locked": float(postencode["flower_exact_fraction"]) == 1.0,
        "flower_response_not_frozen": float(metrics["source_flower_dynamic_frame_fraction"])
        >= 0.95,
        "human_residue_absent": bool(late_gates["late_skin_like_fraction"])
        and bool(human_gates["human_residue_absent"]),
        "canonical_hand_topology_locked": bool(human_gates["canonical_hand_topology_locked"]),
        "intermittent_hand_smear_absent": bool(human_gates["intermittent_hand_smear_absent"]),
        "long_term_robot_identity_stable": bool(human_gates["long_term_robot_identity_stable"]),
        "adversarial_attacks_detected": bool(audit["adversarial_audit_pass"]),
        "high_resolution_review_pass": bool(human["high_resolution_review_pass"]),
    }
    return PerceptualCandidate(
        candidate_id=candidate_id,
        gates=tuple((name, gates[name]) for name in PERCEPTUAL_DEMO_GATES),
        utility=float(human["utility"]),
        wall_seconds=float(metrics["compositor_wall_seconds"]),
        evidence_path=evidence_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--human-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.manifest.resolve(), args.audit.resolve(), args.human_review.resolve()]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    candidate = build_candidate(
        candidate_id=args.candidate_id,
        manifest=_load(paths[0]),
        audit=_load(paths[1]),
        human=_load(paths[2]),
        evidence_path=str(paths[0]),
    )
    decision = select_display_candidate([candidate])
    decision["inputs"] = [{"path": str(path), "sha256": _sha256(path)} for path in paths]
    args.output.resolve().write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if decision["status"] == "DISPLAY_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
