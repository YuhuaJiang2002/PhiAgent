#!/usr/bin/env python3
"""Apply the full-video topology and non-regression gate to one H3 RSI round."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import socket
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.training.h3_identity_rsi import (  # noqa: E402
    H3IdentityRound,
    IdentityMetrics,
    IdentityPromotionContract,
    TopologyReviewEvidence,
    choose_next_round,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_metrics(path: Path, role: str) -> tuple[IdentityMetrics, str | None]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"identity metrics must be a JSON object: {path}")
    video_sha256 = None
    if role in payload and isinstance(payload[role], dict):
        role_payload = payload[role]
        metrics_payload = role_payload.get("metrics")
        inputs = payload.get("inputs")
        if isinstance(inputs, dict) and isinstance(inputs.get(role), dict):
            video_sha256 = str(inputs[role].get("sha256", "")) or None
    else:
        metrics_payload = payload
    if not isinstance(metrics_payload, dict):
        raise ValueError(f"{role} metrics are missing from {path}")
    return IdentityMetrics.from_dict(metrics_payload), video_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--baseline-topology-evidence", type=Path, required=True)
    parser.add_argument("--topology-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--round-name", default="r0-smoke-r8")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--dataset-repeat", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--allow-rejected", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = {
        "candidate_video": args.candidate_video.expanduser().resolve(),
        "baseline_metrics": args.baseline_metrics.expanduser().resolve(),
        "candidate_metrics": args.candidate_metrics.expanduser().resolve(),
        "baseline_topology_evidence": args.baseline_topology_evidence.expanduser().resolve(),
        "topology_evidence": args.topology_evidence.expanduser().resolve(),
    }
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"RSI assessment output already exists: {output_dir}")
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"RSI assessment input is missing or empty: {path}")
    round_ = H3IdentityRound(
        name=args.round_name,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        dataset_repeat=args.dataset_repeat,
        num_epochs=args.num_epochs,
    )
    baseline, baseline_video_sha256 = _load_metrics(paths["baseline_metrics"], "baseline")
    candidate, candidate_video_sha256 = _load_metrics(
        paths["candidate_metrics"], "candidate"
    )
    baseline_topology = TopologyReviewEvidence.load(paths["baseline_topology_evidence"])
    topology = TopologyReviewEvidence.load(paths["topology_evidence"])
    candidate_sha256 = _sha256(paths["candidate_video"])
    if topology.video_sha256 != candidate_sha256:
        raise ValueError("topology evidence digest does not match the candidate video")
    if baseline_video_sha256 and baseline_topology.video_sha256 != baseline_video_sha256:
        raise ValueError("baseline topology evidence does not match baseline metrics")
    if candidate_video_sha256 and topology.video_sha256 != candidate_video_sha256:
        raise ValueError("candidate topology evidence does not match candidate metrics")
    contract = IdentityPromotionContract()
    baseline = replace(
        baseline,
        topology_integrity=baseline_topology.passing_fraction(
            contract.minimum_topology_review_confidence
        ),
    )
    candidate = replace(
        candidate,
        topology_integrity=topology.passing_fraction(
            contract.minimum_topology_review_confidence
        ),
    )
    assessment = contract.assess(baseline, candidate, topology)
    next_round = choose_next_round(((round_, assessment),))
    if assessment.passed:
        evolution_decision = "ACCEPT"
    elif next_round is not None:
        evolution_decision = "TRAIN_NEXT_REVIEWED_ROUND"
    else:
        evolution_decision = "REQUIRE_STRUCTURAL_BACKBONE"
    output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "1.0.0",
        "method": "bounded_h3_identity_topology_rsi_promotion",
        "status": "accepted" if assessment.passed else "rejected",
        "honest_status": "WORKING" if assessment.passed else "PARTIAL",
        "evolution_decision": evolution_decision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "git_head": subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "round": asdict(round_),
        "contract": asdict(contract),
        "baseline_metrics": asdict(baseline),
        "candidate_metrics": asdict(candidate),
        "assessment": assessment.to_dict(),
        "next_round": asdict(next_round) if next_round is not None else None,
        "limitations": [
            "Topology evidence establishes visible full-frame 2-D anatomy, not 3-D dynamics.",
            "Promotion is specific to the measured held-out video and does not prove universal behavior.",
        ],
    }
    _write_json(output_dir / "assessment.json", manifest)
    print(json.dumps({"output": str(output_dir), **manifest}, indent=2))
    return 0 if assessment.passed or args.allow_rejected else 2


if __name__ == "__main__":
    raise SystemExit(main())
