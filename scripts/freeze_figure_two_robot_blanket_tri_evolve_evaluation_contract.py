#!/usr/bin/env python3
"""Freeze candidate-independent E2 blanket evaluation ROIs and review frames."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parser().parse_args()
    campaign = args.campaign_dir.expanduser().resolve()
    inputs = campaign / "inputs"
    output = inputs / "evaluation-contract.json"
    digest_path = inputs / "evaluation-contract.sha256"
    freeze_record = campaign / "provenance/evaluation-contract-freeze.json"
    if output.exists() or digest_path.exists() or freeze_record.exists():
        raise FileExistsError("refusing to reuse an evaluation-contract output")
    config_path = inputs / "campaign-config.json"
    spec_path = inputs / "generation-spec.json"
    manifest_path = campaign / "manifest.json"
    config = _read_json(config_path)
    spec = _read_json(spec_path)
    manifest = _read_json(manifest_path)
    if (
        _sha256(spec_path) != manifest["hashes"]["generation_spec_file"]
        or _sha256(inputs / "initial-frame.png") != config["initial_frame"]["sha256"]
        or spec["challenge"]["initial_frame_sha256"]
        != config["initial_frame"]["sha256"]
    ):
        raise ValueError("campaign hashes differ before evaluation-contract freeze")

    contract = {
        "schema_version": "1.0.0",
        "campaign_id": config["campaign_id"],
        "coordinate_frame": config["coordinate_frames"]["camera"],
        "candidate_independent": True,
        "initial_frame_sha256": config["initial_frame"]["sha256"],
        "task_plan_sha256": manifest["hashes"]["task_plan"],
        "generation_spec_file_sha256": manifest["hashes"]["generation_spec_file"],
        "automatic_thresholds": config["automatic_thresholds"],
        "background_patches_xywh": {
            "upper_center_room": [330, 0, 360, 230],
            "bed_window": [270, 170, 300, 190]
        },
        "terminal_flow_window": {
            "start_frame": 180,
            "frame_count": 12
        },
        "native_review_frame_indices": [
            0,
            10,
            24,
            34,
            52,
            67,
            77,
            96,
            110,
            125,
            149,
            173,
            191
        ],
        "semantic_evidence_policy": (
            "Object identity, assignment, contact causality, phase order, local cloth "
            "continuity, collisions, and target-zone success remain UNAVAILABLE until "
            "a candidate-SHA-bound native-resolution full-video review records PASS."
        ),
        "decision_policy": config["decision_policy"],
        "claim_boundary": config["claim_boundary"]
    }
    contract_sha256 = _canonical_sha256(contract)
    _write_json(output, contract)
    digest_path.write_text(contract_sha256 + "\n", encoding="utf-8")
    git = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    _write_json(
        freeze_record,
        {
            "schema_version": "1.0.0",
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_contract": str(output),
            "evaluation_contract_canonical_sha256": contract_sha256,
            "evaluation_contract_file_sha256": _sha256(output),
            "candidate_outputs_inspected_before_freeze": False,
            "git_status": git.stdout,
            "claim_boundary": config["claim_boundary"]
        },
    )
    print(
        json.dumps(
            {
                "evaluation_contract": str(output),
                "canonical_sha256": contract_sha256,
                "file_sha256": _sha256(output),
                "freeze_record": str(freeze_record)
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
