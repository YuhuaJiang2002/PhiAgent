#!/usr/bin/env python3
"""Freeze the E2 dual-humanoid blanket Tri-Evolve generation campaign."""

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
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.harness.blanket_tri_evolve import (  # noqa: E402
    BlanketDifficulty,
    build_failure_repair_directive,
    canonical_sha256,
    compare_difficulty,
)


HARNESS_VERSION = "figure-two-robot-blanket-tri-evolve-e2-v1"
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/physical_video/figure_two_robot_blanket_fold_tri_evolve_e2_v1.json"
)


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


def _difficulty(payload: dict[str, Any]) -> BlanketDifficulty:
    return BlanketDifficulty(
        environment_id=str(payload["environment_id"]),
        level=str(payload["level"]),
        coordinate_frame=str(payload["coordinate_frame"]),
        obstacle_count=int(payload["obstacle_count"]),
        quilt_rotation_degrees=float(payload["quilt_rotation_degrees"]),
        off_table_overhang_fraction=float(payload["off_table_overhang_fraction"]),
        self_occlusion_fraction=float(payload["self_occlusion_fraction"]),
        required_action_stages=int(payload["required_action_stages"]),
        required_regrasps=int(payload["required_regrasps"]),
        terminal_transport_fraction=float(payload["terminal_transport_fraction"]),
    )


def _prompt(
    config: dict[str, Any],
    *,
    task_plan_sha256: str,
    strategy: dict[str, Any],
    failure_repair: str,
) -> str:
    phase_lines = [
        "- "
        f"{float(phase['start_seconds']):.3f}-{float(phase['end_seconds']):.3f}s "
        f"[{phase['phase_id']}]: {phase['instruction']}"
        for phase in config["phase_plan"]
    ]
    gate_lines = [f"- {gate_id}" for gate_id in config["visual_hard_gates"]]
    return "\n".join(
        [
            "[photorealistic dual-humanoid E2 multi-object manipulation video]",
            "<Picture 1> is the exact fixed camera, room, table, two persistent "
            "humanoid identities, diagonal slate-blue quilt, two cream pillows, "
            "lighting, and initial no-contact pose.",
            "<Video 1> is the exact same static scene. It contains no target motion; "
            "synthesize only the causal task below.",
            "",
            "Generate one uninterrupted documentary-real shot. Keep exactly two "
            "persistent robots with four anatomically consistent arms and hands, "
            "exactly one connected stitched quilt, and exactly two persistent pillows. "
            "Preserve object identity, fabric thickness, bound edges, stitching, metal, "
            "shadows, lens behavior, and background. Motion must follow visible contact, "
            "support, gravity, and the declared order. Never add humans, limbs, robots, "
            "pillows, quilts, text, guides, cuts, crossfades, teleportation, morphing, "
            "material growth, or interpenetration.",
            "",
            "[Hash-bound Tri-Evolve plan]",
            f"Plan SHA-256: {task_plan_sha256}",
            f"Strategy: {strategy['label']} / {strategy['strategy']}",
            f"Strategy directive: {strategy['directive']}",
            failure_repair,
            "",
            "Task order: assigned dual pillow removal; pillow settle; quilt untuck, "
            "overhang recovery and square canonicalization; verification hold; "
            "cooperative lengthwise fold; settle and bilateral regrasp; cooperative "
            "crosswise fold; dual-contact bundle transport; terminal hold.",
            *phase_lines,
            "",
            "Non-negotiable hard gates; FAIL or missing evidence rejects:",
            *gate_lines,
            "",
            "Viewer-left and viewer-right are camera-frame relations only. The exact "
            "first frame is restored by a predeclared deterministic boundary adapter "
            "after generation; the raw proposal remains preserved. No gate or threshold "
            "is relaxed. Generated RGB is not real-robot execution evidence.",
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
    if not isinstance(config, dict) or config.get("campaign_id") != HARNESS_VERSION:
        raise ValueError("campaign config differs from the pinned harness version")
    if config.get("schema_version") != "1.0.0":
        raise ValueError("campaign config requires schema_version 1.0.0")

    source = (PROJECT_ROOT / config["initial_frame"]["path"]).resolve()
    if not source.is_file() or _sha256(source) != config["initial_frame"]["sha256"]:
        raise ValueError("initial frame is missing or differs from its frozen hash")
    strategies = config["tri_evolve"]["strategies"]
    labels = [str(item["label"]) for item in strategies]
    seeds = [int(seed) for item in strategies for seed in item["seeds"]]
    if len(set(labels)) != len(labels) or len(set(seeds)) != len(seeds):
        raise ValueError("strategy labels and generation seeds must be unique")

    incumbent = _difficulty(config["difficulty_evolution"]["incumbent"])
    challenger = _difficulty(config["difficulty_evolution"]["challenger"])
    difficulty_comparison = compare_difficulty(incumbent, challenger)
    if not difficulty_comparison["passed"]:
        raise ValueError("challenger does not monotonically increase frozen difficulty")
    failure_repair = build_failure_repair_directive(
        config["tri_evolve"]["incumbent_failed_gate_ids"]
    )

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
        "difficulty": {
            "incumbent": incumbent.to_dict(),
            "challenger": challenger.to_dict(),
            "comparison": difficulty_comparison,
        },
        "tri_evolve": {
            key: value
            for key, value in config["tri_evolve"].items()
            if key != "strategies"
        },
        "phase_plan": config["phase_plan"],
        "visual_hard_gates": config["visual_hard_gates"],
        "physical_promotion_gates": config["physical_promotion_gates"],
        "automatic_thresholds": config["automatic_thresholds"],
        "decision_policy": config["decision_policy"],
        "claim_boundary": config["claim_boundary"],
    }
    task_plan_sha256 = canonical_sha256(task_plan)
    challenge = {
        "schema_version": "1.0.0",
        "challenge_id": "figure-blanket-obstacle-diagonal-e2-001",
        "coordinate_frame": config["coordinate_frames"]["camera"],
        "width": config["initial_frame"]["width"],
        "height": config["initial_frame"]["height"],
        "fps": config["generation"]["fps"],
        "duration_seconds": config["generation"]["target"]["duration_seconds"],
        "initial_frame_path": str(frozen_frame),
        "initial_frame_sha256": config["initial_frame"]["sha256"],
        "background_id": "residential-laundry-bedroom-fixed-v2",
        "camera_id": "figure-blanket-fixed-camera-v2",
        "robot_embodiment_id": "two-silver-humanoids-v2",
        "quilt_identity_id": "slate-blue-grid-quilt-v2",
        "obstacle_identity_ids": ["cream-pillow-left-v1", "cream-pillow-right-v1"],
        "task_objective": config["task"],
    }
    challenge_sha256 = canonical_sha256(challenge)
    challenge["challenge_sha256"] = challenge_sha256

    cases = []
    for strategy in strategies:
        prompt = _prompt(
            config,
            task_plan_sha256=task_plan_sha256,
            strategy=strategy,
            failure_repair=failure_repair,
        )
        cases.append(
            {
                "label": strategy["label"],
                "harness_version": HARNESS_VERSION,
                "harness_sha256": task_plan_sha256,
                "task_plan_sha256": task_plan_sha256,
                "challenge_sha256": challenge_sha256,
                "strategy": strategy["strategy"],
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "seeds": strategy["seeds"],
            }
        )
        (inputs / f"prompt-{strategy['label']}.txt").write_text(
            prompt + "\n", encoding="utf-8"
        )

    spec = {
        "schema_version": "1.0.0",
        "status": "READY",
        "cases": cases,
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
        "generation": config["generation"],
        "compute": {
            "host": "runtime-selected-h200",
            "minimum_free_mib": 120000,
            "num_gpus": 2,
            "physical_gpu_indices": [0, 1],
            "require_idle": True,
            "required_gpu_name": "NVIDIA H200",
            "ulysses_degree": 2,
            "use_fsdp_inference": True,
        },
        "selection_policy": "all_hard_gates_then_candidate_sha_bound_native_review",
        "claim_boundary": config["claim_boundary"],
    }
    spec["spec_sha256"] = canonical_sha256(spec)

    _write_json(inputs / "task-plan.json", task_plan)
    (inputs / "task-plan.sha256").write_text(
        task_plan_sha256 + "\n", encoding="utf-8"
    )
    _write_json(inputs / "generation-spec.json", spec)
    _write_json(
        provenance / "git-state.json",
        {
            "head": _git_capture("rev-parse", "HEAD"),
            "status": _git_capture("status", "--short"),
            "diff_stat": _git_capture("diff", "--stat"),
        },
    )
    (provenance / "packages.txt").write_text(
        "\n".join(
            sorted(
                f"{item.metadata['Name']}=={item.version}"
                for item in importlib.metadata.distributions()
                if item.metadata.get("Name")
            )
        )
        + "\n",
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
        "candidate_budget": len(seeds),
        "strategies": labels,
        "seeds": seeds,
        "difficulty_comparison": difficulty_comparison,
        "hashes": {
            "source_config": _sha256(config_path),
            "frozen_config": _sha256(frozen_config),
            "initial_frame": _sha256(frozen_frame),
            "task_plan": task_plan_sha256,
            "generation_spec_file": _sha256(inputs / "generation-spec.json"),
        },
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
