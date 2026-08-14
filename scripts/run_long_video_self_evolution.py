#!/usr/bin/env python3
"""Run a resumable, failure-directed long-video harness evolution loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.agent.long_video_self_evolution import (
    FrozenVisualContract,
    evaluate_specialty_contract,
    evaluate_visual_iteration,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"JSON input must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _git_state() -> dict[str, Any]:
    state = {}
    for name, command in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
    }.items():
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True
        )
        state[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    return state


def _gpu_snapshot(requested: tuple[int, ...]) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for raw in completed.stdout.splitlines():
        index, uuid, name, total, used, utilization = [part.strip() for part in raw.split(",", 5)]
        rows.append(
            {
                "physical_index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "utilization_percent": int(utilization),
            }
        )
    available = {row["physical_index"] for row in rows}
    missing = set(requested) - available
    if missing:
        raise ValueError(f"requested physical GPUs are absent: {sorted(missing)}")
    selected = [row for row in rows if row["physical_index"] in requested]
    if any(row["memory_used_mib"] > 4096 for row in selected):
        raise RuntimeError(f"selected GPUs are not idle enough: {selected}")
    return {
        "command": command,
        "all": rows,
        "selected": selected,
        "execution_mode": "gpu" if selected else "cpu_gpu_disabled",
    }


def _run(command: list[str], *, env: dict[str, str], log_path: Path) -> int:
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    log_path.write_text(
        json.dumps(
            {
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return completed.returncode


def _resolve(config_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/usr/bin/ffprobe"))
    parser.add_argument(
        "--physical-gpus",
        default="none",
        help="Comma-separated physical GPU indices, or 'none' for a CPU-only repair loop.",
    )
    parser.add_argument("--maximum-cycles", type=int, default=0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load(config_path)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {output}")
    output.mkdir(parents=True)
    iterations_dir = output / "iterations"
    iterations_dir.mkdir()
    events_path = output / "events.jsonl"
    status_path = output / "status.json"
    requested_gpus = (
        ()
        if args.physical_gpus.strip().lower() == "none"
        else tuple(
            int(item.strip()) for item in args.physical_gpus.split(",") if item.strip()
        )
    )
    if not requested_gpus and args.physical_gpus.strip().lower() != "none":
        raise ValueError("GPU selection must be 'none' or a comma-separated index list")
    gpu = _gpu_snapshot(requested_gpus)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in requested_gpus)
    env["PYTHONUNBUFFERED"] = "1"

    path_keys = (
        "incumbent_video", "source_video", "robot_limb_masks", "person_masks",
        "flower_masks", "pose_limb_masks", "reference_image", "frozen_limits_report",
    )
    paths = {key: _resolve(config_path.parent, str(config[key])) for key in path_keys}
    for key in ("flower_observation_manifest",):
        if key in config:
            paths[key] = _resolve(config_path.parent, str(config[key]))
    paths["specialty_flower_masks"] = _resolve(
        config_path.parent,
        str(config.get("specialty_flower_masks", config["flower_masks"])),
    )
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{name}: {path}")
    contract = FrozenVisualContract.from_dict(config["frozen_contract"])
    strategies = list(config["strategies"])
    if not strategies:
        raise ValueError("at least one repair strategy is required")
    maximum_cycles = args.maximum_cycles or len(strategies)
    maximum_cycles = min(maximum_cycles, len(strategies))
    review_encoder = config.get(
        "review_encoder",
        {
            "preset": "veryfast",
            "crf": 8,
            "chroma_qp_offset": -12,
            "pixel_format": "yuv444p",
        },
    )
    workers = int(config.get("workers", 8))
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "claim_scope": "perceptually plausible synthetic long-video data",
        "physical_evidence": False,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "command": [sys.executable, *sys.argv],
        "git": _git_state(),
        "gpu": gpu,
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "seed": int(config["seed"]),
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "frozen_contract": config["frozen_contract"],
        "threshold_mutation_allowed": False,
        "iterations": [],
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(status_path, {"status": "RUNNING", "iteration": -1})

    automatic_winner = None
    for index, strategy in enumerate(strategies[:maximum_cycles]):
        iteration = iterations_dir / f"iter-{index:02d}-{strategy['id']}"
        iteration.mkdir()
        repair_dir = iteration / "repair"
        specialty_dir = iteration / "specialty-audit"
        persistent_dir = iteration / "persistent-grasp-preflight"
        full_dir = iteration / "full-audit"
        repair_command = [
            str(args.python), "scripts/stabilize_robot_chroma_state.py",
            "--candidate-video", str(paths["incumbent_video"]),
            "--source-video", str(paths["source_video"]),
            "--person-masks", str(paths["person_masks"]),
            "--flower-masks", str(paths["flower_masks"]),
            "--flower-mask-contract", str(
                config.get(
                    "flower_mask_contract",
                    "tracked_front_layer_with_human_negatives",
                )
            ),
            "--visibility-flower-masks", str(paths["specialty_flower_masks"]),
            "--pose-limb-masks", str(paths["pose_limb_masks"]),
            "--robot-limb-masks", str(paths["robot_limb_masks"]),
            "--output-dir", str(repair_dir),
            "--ffmpeg", str(args.ffmpeg),
            "--expected-frames", str(config["expected_frames"]),
            "--fps", str(config["fps"]),
            "--kernel-sizes", str(strategy["kernel_sizes"]),
            "--strength", str(strategy["strength"]),
            "--maximum-chroma-delta", str(strategy["maximum_chroma_delta"]),
            "--saturation-scale", str(strategy.get("saturation_scale", 1.0)),
            "--minimum-chroma-tv-reduction", str(strategy["minimum_chroma_tv_reduction"]),
            "--artifact-policy", "review_only",
            "--review-preset", str(review_encoder["preset"]),
            "--review-crf", str(review_encoder["crf"]),
            "--review-chroma-qp-offset", str(review_encoder["chroma_qp_offset"]),
            "--review-pixel-format", str(review_encoder.get("pixel_format", "yuv420p")),
            "--workers", str(workers),
            "--seed", str(int(config["seed"]) + index),
        ]
        repair_rc = _run(repair_command, env=env, log_path=iteration / "repair-command.json")
        candidate = repair_dir / "robot-chroma-state.mp4"
        repair_manifest_path = repair_dir / "manifest.json"
        if not repair_manifest_path.is_file() or not candidate.is_file():
            row = {
                "index": index, "strategy": strategy, "status": "REJECTED",
                "reason": "repair_command_failed", "returncode": repair_rc,
            }
            manifest["iterations"].append(row)
            _append_event(events_path, row)
            _write_json(status_path, row)
            continue
        repair_manifest = _load(repair_manifest_path)
        if repair_rc != 0 or not all(repair_manifest["gates"].values()):
            row = {
                "index": index,
                "strategy": strategy,
                "status": "REJECTED",
                "reason": "repair_internal_gate_failed",
                "returncode": repair_rc,
                "failed_gates": sorted(
                    name for name, passed in repair_manifest["gates"].items()
                    if not passed
                ),
                "candidate": {"path": str(candidate), "sha256": _sha256(candidate)},
                "repair_manifest": str(repair_manifest_path),
            }
            manifest["iterations"].append(row)
            _append_event(events_path, row)
            _write_json(status_path, row)
            _write_json(output / "manifest.json", manifest)
            continue

        specialty_command = [
            str(args.python), "scripts/audit_right_arm_flower_occlusion.py",
            "--source-video", str(paths["source_video"]),
            "--incumbent-video", str(paths["incumbent_video"]),
            "--challenger-video", str(candidate),
            "--robot-limb-masks", str(paths["robot_limb_masks"]),
            "--person-masks", str(paths["person_masks"]),
            "--flower-masks", str(paths["specialty_flower_masks"]),
            "--pose-limb-masks", str(paths["pose_limb_masks"]),
            # Keep this observer byte-for-byte comparable with the report that
            # froze its absolute non-regression limits.  The expanded tracked
            # front layer is evaluated separately by the full/native audit.
            "--flower-mask-contract", "resolved_visibility",
            "--output-dir", str(specialty_dir),
            "--expected-frames", str(config["expected_frames"]),
            "--fps", str(config["fps"]),
            "--seed", str(int(config["seed"]) + index),
        ]
        specialty_rc = _run(
            specialty_command, env=env, log_path=iteration / "specialty-command.json"
        )
        specialty_report_path = specialty_dir / "report.json"
        # The specialty tool returns 2 when its informational, strictly-better
        # comparison against the incumbent is not conjunctively true (equal
        # zero high-flicker counts are enough).  Promotion uses the separately
        # frozen absolute contract below, so only a missing report is a command
        # failure here.
        if not specialty_report_path.is_file():
            row = {
                "index": index,
                "strategy": strategy,
                "status": "REJECTED",
                "reason": "specialty_audit_command_failed",
                "repair_returncode": repair_rc,
                "specialty_returncode": specialty_rc,
            }
            manifest["iterations"].append(row)
            _append_event(events_path, row)
            _write_json(status_path, row)
            _write_json(output / "manifest.json", manifest)
            continue
        specialty_report = _load(specialty_report_path)
        specialty_decision = evaluate_specialty_contract(
            repair_manifest=repair_manifest,
            specialty_report=specialty_report,
            contract=contract,
        )
        if not specialty_decision["automatic_pass"]:
            row = {
                "index": index,
                "strategy": strategy,
                "status": "REJECTED",
                "reason": "specialty_frozen_contract_failed",
                **specialty_decision,
                "candidate": {"path": str(candidate), "sha256": _sha256(candidate)},
                "repair_manifest": str(repair_manifest_path),
                "specialty_report": str(specialty_report_path),
                "returncodes": {"repair": repair_rc, "specialty": specialty_rc},
            }
            _write_json(iteration / "decision.json", row)
            manifest["iterations"].append(row)
            _append_event(events_path, row)
            _write_json(status_path, row)
            _write_json(output / "manifest.json", manifest)
            continue
        persistent_command = [
            str(args.python), "scripts/audit_persistent_grasp_preflight.py",
            "--source-video", str(paths["source_video"]),
            "--candidate-video", str(candidate),
            "--flower-masks", str(paths["flower_masks"]),
            "--limb-masks", str(paths["pose_limb_masks"]),
            "--output-dir", str(persistent_dir),
            "--ffmpeg", str(args.ffmpeg),
            "--expected-frames", str(config["expected_frames"]),
            "--fps", str(config["fps"]),
            "--start", "497", "--end-exclusive", "644",
            "--replacement-threshold", "12", "--contact-radius", "3",
            "--maximum-source-occlusion-gap", "24",
            "--minimum-bridge-coverage", "0.80", "--required-recall", "1.0",
            "--mask-frame-name", "camera:source_aligned_832x480",
            "--mask-source-width", "1280", "--mask-source-height", "720",
            "--mask-scaled-width", "854", "--mask-scaled-height", "480",
            "--mask-crop-left", "11", "--mask-crop-top", "0",
            "--target-frame-name", "camera:source_native_1280x720",
            "--target-scaled-width", "1280", "--target-scaled-height", "720",
            "--target-crop-left", "0", "--target-crop-top", "0",
            "--target-width", "1280", "--target-height", "720",
        ]
        persistent_rc = _run(
            persistent_command,
            env=env,
            log_path=iteration / "persistent-grasp-preflight-command.json",
        )
        persistent_report_path = persistent_dir / "report.json"
        if not persistent_report_path.is_file():
            row = {
                "index": index,
                "strategy": strategy,
                "status": "REJECTED",
                "reason": "persistent_grasp_preflight_command_failed",
                "returncodes": {
                    "repair": repair_rc,
                    "specialty": specialty_rc,
                    "persistent_preflight": persistent_rc,
                },
            }
            manifest["iterations"].append(row)
            _append_event(events_path, row)
            _write_json(status_path, row)
            _write_json(output / "manifest.json", manifest)
            continue
        persistent_report = _load(persistent_report_path)
        if not bool(persistent_report.get("automatic_pass")):
            row = {
                "index": index,
                "strategy": strategy,
                "status": "REJECTED",
                "reason": "persistent_grasp_preflight_failed",
                "candidate": {"path": str(candidate), "sha256": _sha256(candidate)},
                "repair_manifest": str(repair_manifest_path),
                "specialty_report": str(specialty_report_path),
                "persistent_preflight_report": str(persistent_report_path),
                "failed_frames": persistent_report.get("failed_frames", []),
                "returncodes": {
                    "repair": repair_rc,
                    "specialty": specialty_rc,
                    "persistent_preflight": persistent_rc,
                },
            }
            _write_json(iteration / "decision.json", row)
            manifest["iterations"].append(row)
            _append_event(events_path, row)
            _write_json(status_path, row)
            _write_json(output / "manifest.json", manifest)
            continue
        full_command = [
            str(args.python), "scripts/audit_robot_layer_long_video.py",
            "--source-video", str(paths["source_video"]),
            "--candidate", f"challenger={candidate}",
            "--reference-image", str(paths["reference_image"]),
            "--person-masks", str(paths["person_masks"]),
            "--flower-masks", str(paths["flower_masks"]),
            "--flower-mask-contract", str(
                config.get(
                    "flower_mask_contract",
                    "tracked_front_layer_with_human_negatives",
                )
            ),
            "--limb-masks", str(paths["pose_limb_masks"]),
            "--output-dir", str(full_dir),
            "--ffmpeg", str(args.ffmpeg), "--ffprobe", str(args.ffprobe),
            "--expected-frames", str(config["expected_frames"]),
            "--fps", str(config["fps"]),
            "--reference-frame", "276", "--anchor-start", "259",
            "--anchor-end-exclusive", "297", "--late-start", "480",
            "--person-dilation", "10", "--skin-negative-dilation", "2",
            "--person-core-negative-erosion", "2", "--replacement-threshold", "12",
            "--contact-radius", "3", "--allowed-late-violation-fraction", "0.10",
            "--required-contact-recall", "0.95", "--persistent-grasp-start", "497",
            "--persistent-grasp-end-exclusive", "644", "--maximum-source-occlusion-gap", "24",
            "--minimum-occlusion-bridge-coverage", "0.80",
            "--required-persistent-grasp-recall", "1.0", "--adversarial-stride", "12",
            "--frozen-limits-report", str(paths["frozen_limits_report"]),
            "--frozen-limits-candidate", str(config["frozen_limits_candidate"]),
            "--seed", str(int(config["seed"]) + index),
            "--mask-frame-name", "camera:source_aligned_832x480",
            "--mask-source-width", "1280", "--mask-source-height", "720",
            "--mask-scaled-width", "854", "--mask-scaled-height", "480",
            "--mask-crop-left", "11", "--mask-crop-top", "0",
            "--target-frame-name", "camera:source_native_1280x720",
            "--target-scaled-width", "1280", "--target-scaled-height", "720",
            "--target-crop-left", "0", "--target-crop-top", "0",
            "--target-width", "1280", "--target-height", "720",
        ]
        full_rc = _run(full_command, env=env, log_path=iteration / "full-command.json")
        full_report_path = full_dir / "audit-report.json"
        if not full_report_path.is_file():
            row = {
                "index": index, "strategy": strategy, "status": "REJECTED",
                "reason": "audit_command_failed", "repair_returncode": repair_rc,
                "specialty_returncode": specialty_rc, "full_returncode": full_rc,
            }
        else:
            decision = evaluate_visual_iteration(
                repair_manifest=repair_manifest,
                specialty_report=specialty_report,
                full_report=_load(full_report_path),
                contract=contract,
            )
            row = {
                "index": index,
                "strategy": strategy,
                **decision,
                "candidate": {"path": str(candidate), "sha256": _sha256(candidate)},
                "repair_manifest": str(repair_manifest_path),
                "specialty_report": str(specialty_report_path),
                "persistent_preflight_report": str(persistent_report_path),
                "full_report": str(full_report_path),
                "returncodes": {
                    "repair": repair_rc,
                    "specialty": specialty_rc,
                    "persistent_preflight": persistent_rc,
                    "full": full_rc,
                },
            }
            _write_json(iteration / "decision.json", row)
            if decision["automatic_pass"]:
                automatic_winner = row
        manifest["iterations"].append(row)
        _append_event(events_path, row)
        _write_json(status_path, row)
        _write_json(output / "manifest.json", manifest)
        if automatic_winner is not None:
            break

    if automatic_winner is None:
        manifest["status"] = "PARTIAL"
        manifest["decision"] = "NO_CHALLENGER_PASSED_FROZEN_CONTRACT"
        manifest["next_action"] = "architecture_change_required"
        _write_json(status_path, {
            "status": manifest["status"], "decision": manifest["decision"],
            "iterations": len(manifest["iterations"]),
        })
    else:
        manifest["status"] = "AWAITING_HIGH_RESOLUTION_REVIEW"
        manifest["decision"] = "AUTOMATIC_GATES_PASS_HUMAN_VETO_PENDING"
        manifest["automatic_winner"] = automatic_winner
        review_bundle = output / "high-resolution-review"
        review_bundle.mkdir()
        review_pattern = review_bundle / "timeline-page-%02d.jpg"
        review_command = [
            str(args.ffmpeg), "-y", "-v", "error",
            "-i", str(automatic_winner["candidate"]["path"]),
            "-vf", "fps=2,scale=320:180:flags=lanczos,tile=4x4",
            "-vsync", "vfr", "-q:v", "2", str(review_pattern),
        ]
        review_rc = _run(
            review_command, env=env, log_path=output / "review-bundle-command.json"
        )
        review_pages = sorted(review_bundle.glob("timeline-page-*.jpg"))
        if review_rc != 0 or not review_pages:
            raise RuntimeError("failed to build the high-resolution review bundle")
        review_request = {
            "schema_version": "1.0.0",
            "status": "PENDING",
            "candidate": automatic_winner["candidate"],
            "full_report": automatic_winner["full_report"],
            "timeline_pages": [
                {"path": str(path), "sha256": _sha256(path)} for path in review_pages
            ],
            "required_review": [
                "full 27.5-second timeline at native resolution",
                "right-arm flicker and hand topology",
                "hand-flower z-order and projected contact",
                "flower motion after 20 seconds",
                "colour bleeding or plastic chroma flattening",
            ],
            "physical_claims_disallowed": True,
        }
        _write_json(output / "review-request.json", review_request)
        _write_json(status_path, {
            "status": manifest["status"],
            "candidate": automatic_winner["candidate"],
            "review_request": str(output / "review-request.json"),
        })
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(output / "manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "output_dir": str(output),
        "automatic_winner": automatic_winner,
    }, indent=2, sort_keys=True))
    return 0 if automatic_winner is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
