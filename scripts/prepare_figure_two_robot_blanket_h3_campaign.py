#!/usr/bin/env python3
"""Freeze a hash-bound MiniMax-H3 campaign for two robots folding one quilt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/physical_video/figure_two_robot_blanket_fold_photorealistic_v2.json"
)
HARNESS_VERSION = "figure-two-robot-blanket-fold-photorealistic-v2"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_capture(*args: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "argv": ["git", *args],
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _prompt(config: dict[str, Any], plan_sha256: str) -> str:
    phase_lines = []
    for phase in config["phase_plan"]:
        phase_lines.append(
            "- "
            f"{float(phase['start_seconds']):.3f}-{float(phase['end_seconds']):.3f}s "
            f"[{phase['phase_id']}]: {phase['instruction']}"
        )
    gate_lines = [f"- {gate_id}" for gate_id in config["visual_hard_gates"]]
    return "\n".join(
        [
            "[photorealistic full-body dual-humanoid manipulation video]",
            "<Picture 1> is the exact camera, room, table, quilt, lighting, two "
            "full-body humanoid identities, articulated arms, and initial pose.",
            "<Video 1> is the exact same static scene and camera identity. It contains "
            "no target motion; synthesize only the causal phase plan below.",
            "",
            "Generate one natural, uninterrupted real-camera-looking shot. Keep exactly "
            "two persistent humanoid robots, four anatomically consistent robot arms and "
            "hands, and exactly one persistent slate-blue quilt. Preserve the quilt's "
            "stitched grid, bound edges, four material corners, thickness, color, weave, "
            "and connected topology. Motion must be slow, contact-driven, gravity- and "
            "support-consistent. Preserve realistic metal, fabric, shadows, lens behavior, "
            "and occlusions. Never render CGI styling, extra limbs, human hands, text, "
            "markers, guides, watermarks, cuts, crossfades, teleportation, morphing, "
            "duplication, material growth, or interpenetration.",
            "",
            "[Hash-bound task plan]",
            f"Plan SHA-256: {plan_sha256}",
            "Normalized task: the two robots establish four visible corner contacts; "
            "cooperatively half-fold the near side; settle and regrasp; synchronously "
            "gatefold the left and right thirds; square, release, withdraw, and hold one "
            "compact quilt.",
            *phase_lines,
            "",
            "Non-negotiable visual hard gates; FAIL or missing evidence rejects:",
            *gate_lines,
            "",
            "Use camera-space viewer-left and viewer-right consistently. No score, seed "
            "count, or visual preference may override one failed gate. This output is a "
            "generated RGB proposal, not evidence of calibrated geometry, force, safety, "
            "joint feasibility, or recorded real-robot execution.",
        ]
    )


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse campaign directory: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != "1.0.0":
        raise ValueError("campaign config requires one schema_version 1.0.0 object")
    if config.get("campaign_id") != HARNESS_VERSION:
        raise ValueError("campaign_id differs from the pinned runner harness version")

    source = (PROJECT_ROOT / config["initial_frame"]["path"]).resolve()
    if not source.is_file() or _sha256(source) != config["initial_frame"]["sha256"]:
        raise ValueError("initial frame is missing or differs from its frozen hash")
    if len(set(config["generation"]["seeds"])) != len(config["generation"]["seeds"]):
        raise ValueError("generation seeds must be unique")

    output.mkdir(parents=True)
    inputs = output / "inputs"
    provenance = output / "provenance"
    inputs.mkdir()
    provenance.mkdir()
    frozen_config = inputs / "campaign-config.json"
    frozen_frame = inputs / "initial-frame.png"
    shutil.copy2(config_path, frozen_config)
    shutil.copy2(source, frozen_frame)

    task_plan = {
        "schema_version": "1.0.0",
        "campaign_id": HARNESS_VERSION,
        "coordinate_frames": config["coordinate_frames"],
        "initial_frame_sha256": config["initial_frame"]["sha256"],
        "phase_plan": config["phase_plan"],
        "visual_hard_gates": config["visual_hard_gates"],
        "physical_promotion_gates": config["physical_promotion_gates"],
        "decision_policy": config["decision_policy"],
        "claim_boundary": config["claim_boundary"],
    }
    task_plan_sha256 = _canonical_sha256(task_plan)
    prompt = _prompt(config, task_plan_sha256)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    challenge = {
        "schema_version": "1.0.0",
        "challenge_id": "figure-blanket-flat-001",
        "coordinate_frame": config["coordinate_frames"]["camera"],
        "width": config["initial_frame"]["width"],
        "height": config["initial_frame"]["height"],
        "fps": config["generation"]["fps"],
        "duration_seconds": config["generation"]["target"]["duration_seconds"],
        "initial_frame_path": str(frozen_frame),
        "initial_frame_sha256": config["initial_frame"]["sha256"],
        "background_id": "residential-laundry-bedroom-fixed-v2",
        "camera_id": "figure-blanket-fixed-camera-v2",
        "robot_embodiment_id": "two-full-body-silver-humanoids-v2",
        "quilt_identity_id": "slate-blue-grid-quilt-v2",
        "task_objective": config["task"],
    }
    challenge_sha256 = _canonical_sha256(challenge)
    challenge["challenge_sha256"] = challenge_sha256
    case = {
        "label": HARNESS_VERSION,
        "harness_version": HARNESS_VERSION,
        "harness_sha256": task_plan_sha256,
        "task_plan_sha256": task_plan_sha256,
        "challenge_sha256": challenge_sha256,
        "strategy": "synchronized_lengthwise_then_gatefold",
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
        "seeds": config["generation"]["seeds"],
    }
    spec = {
        "schema_version": "1.0.0",
        "status": "READY",
        "cases": [case],
        "challenge": challenge,
        "model": {
            **config["model"],
            "checkpoint_root": (
                "/opt/phiagent/checkpoints/minimax-h3/"
                f"revision_{config['model']['revision']}"
            ),
            "runtime": {
                **config["model"]["runtime"],
                "performance_mode": "speed",
                "cache": None,
            },
        },
        "generation": {
            key: value
            for key, value in config["generation"].items()
            if key != "seeds"
        },
        "compute": {
            "host": "runtime-selected-h200",
            "minimum_free_mib": 120000,
            "num_gpus": 4,
            "physical_gpu_indices": [0, 1, 2, 3],
            "require_idle": True,
            "required_gpu_name": "NVIDIA H200",
            "ulysses_degree": 4,
            "use_fsdp_inference": True,
        },
        "selection_policy": "all_visual_hard_gates_then_native_user_review",
        "claim_boundary": config["claim_boundary"],
    }
    spec["spec_sha256"] = _canonical_sha256(spec)

    _write_json(inputs / "task-plan.json", task_plan)
    (inputs / "task-plan.sha256").write_text(task_plan_sha256 + "\n", encoding="utf-8")
    (inputs / "generation-prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (inputs / "generation-prompt.sha256").write_text(prompt_sha256 + "\n", encoding="utf-8")
    _write_json(inputs / "generation-spec.json", spec)
    _write_json(provenance / "git-state.json", {
        "head": _git_capture("rev-parse", "HEAD"),
        "status": _git_capture("status", "--short"),
        "diff_stat": _git_capture("diff", "--stat"),
    })
    (provenance / "packages.txt").write_text(
        "\n".join(
            sorted(
                f"{item.metadata['Name']}=={item.version}"
                for item in importlib.metadata.distributions()
                if item.metadata.get("Name")
            )
        ) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0.0",
        "status": "READY_FOR_GPU_GENERATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "argv": [sys.executable, *sys.argv],
        "campaign_id": HARNESS_VERSION,
        "candidate_budget": len(config["generation"]["seeds"]),
        "seeds": config["generation"]["seeds"],
        "hashes": {
            "source_config": _sha256(config_path),
            "frozen_config": _sha256(frozen_config),
            "initial_frame": _sha256(frozen_frame),
            "task_plan": task_plan_sha256,
            "generation_prompt": prompt_sha256,
            "generation_spec_file": _sha256(inputs / "generation-spec.json"),
        },
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
