#!/usr/bin/env python3
"""Compile, execute, or select a JoyAI action-intent visual demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.world_model.joyai_action_intent import (  # noqa: E402
    build_candidate_audit_template,
    candidate_plan,
    compile_action_prompt,
    load_action_intent_config,
    select_visual_candidate,
)
from phiagent.world_model.joyai_sc3 import validate_server_manifest  # noqa: E402


CLIENT_SCRIPT = PROJECT_ROOT / "scripts" / "run_joyai_video_edit_client.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("compile", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--server-url", default="ws://127.0.0.1:18080/ws")
        if name == "run":
            command.add_argument(
                "--candidate-limit",
                type=int,
                help="Run only the first N frozen seeds for a bounded smoke test.",
            )
            command.add_argument(
                "--server-manifest",
                type=Path,
                required=True,
                help=(
                    "WORKING joyai_server_ready manifest from the pinned two-GPU JoyAI launcher."
                ),
            )

    select = subparsers.add_parser("select")
    select.add_argument("--config", type=Path, required=True)
    select.add_argument("--audit-report", type=Path, action="append", required=True)
    select.add_argument("--output", type=Path, required=True)
    return parser


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, check=False)

    head = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--short", "--untracked-files=all")
    worktree = run("diff", "--binary", "--no-ext-diff")
    cached = run("diff", "--cached", "--binary", "--no-ext-diff")
    failures = [
        completed.stderr.decode("utf-8", errors="replace").strip()
        for completed in (head, branch, status, worktree, cached)
        if completed.returncode
    ]
    if failures:
        raise RuntimeError("could not capture Git state: " + "; ".join(failures))
    return {
        "head": head.stdout.decode().strip(),
        "branch": branch.stdout.decode().strip(),
        "status_short": status.stdout.decode("utf-8", errors="replace").splitlines(),
        "worktree_diff": {
            "bytes": len(worktree.stdout),
            "sha256": _sha256_bytes(worktree.stdout),
        },
        "cached_diff": {
            "bytes": len(cached.stdout),
            "sha256": _sha256_bytes(cached.stdout),
        },
    }


def _package_state(output: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    path = output / "packages.txt"
    path.write_text(completed.stdout, encoding="utf-8")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "pip_freeze": str(path),
        "pip_freeze_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }


def _probe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required for the action-intent input preflight")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"ffprobe failed: {completed.stderr.strip()}")
    raw = json.loads(completed.stdout)
    streams = raw.get("streams", [])
    if len(streams) != 1:
        raise ValueError("action-intent source must contain exactly one video stream")
    stream = streams[0]
    rate_text = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    rate = float(Fraction(rate_text))
    duration = float(raw.get("format", {}).get("duration", 0.0))
    inferred_frames = round(duration * rate) if duration > 0 else None
    return {
        "command": command,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": rate,
        "duration_seconds": duration,
        "container_frame_count": (
            None if stream.get("nb_frames") in (None, "N/A") else int(stream["nb_frames"])
        ),
        "duration_inferred_frame_count": inferred_frames,
    }


def _compile(
    *, config_path: Path, output: Path, server_url: str
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    config_path = config_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {output}")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = load_action_intent_config(config_path, project_root=PROJECT_ROOT)
    inputs = config.verify_inputs()
    video_probe = _probe_video(config.source_video)
    if (video_probe["width"], video_probe["height"]) != (config.width, config.height):
        raise ValueError(
            "source dimensions do not match the action-intent camera contract: "
            f"{video_probe['width']}x{video_probe['height']} != "
            f"{config.width}x{config.height}"
        )
    if abs(float(video_probe["fps"]) - config.fps) > 1e-6:
        raise ValueError("source FPS does not match the frozen action-intent timeline")
    observed_frames = video_probe["container_frame_count"]
    if observed_frames is None:
        observed_frames = video_probe["duration_inferred_frame_count"]
    if observed_frames != config.model_frame_count:
        raise ValueError(
            f"source has {observed_frames} frames; expected {config.model_frame_count}"
        )

    output.mkdir(parents=True)
    prompt = compile_action_prompt(config)
    prompt_path = output / "compiled-action-prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    snapshot_path = output / "intent-config.snapshot.json"
    _write_json(snapshot_path, config.to_manifest())

    implementation_paths = (
        PROJECT_ROOT / "phiagent" / "world_model" / "joyai_action_intent.py",
        Path(__file__).resolve(),
        CLIENT_SCRIPT,
        config_path,
    )
    implementation_sources = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in implementation_paths
    ]

    plans = candidate_plan(
        config,
        output_dir=output,
        prompt_file=prompt_path,
        client_script=CLIENT_SCRIPT,
        python_executable=sys.executable,
        server_url=server_url,
    )
    _write_json(output / "candidate-plan.json", list(plans))
    commands_path = output / "commands.sh"
    commands_path.write_text(
        "\n".join(shlex.join(plan["command"]) for plan in plans) + "\n",
        encoding="utf-8",
    )
    audit_dir = output / "audit-templates"
    audit_dir.mkdir()
    for plan in plans:
        _write_json(
            audit_dir / f"{plan['candidate_id']}.json",
            build_candidate_audit_template(config, str(plan["candidate_id"])),
        )

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "stage": "joyai_action_intent_harness",
        "status": "PARTIAL",
        "implementation_status": "WORKING",
        "execution_status": "NOT_STARTED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "command": [sys.executable, *sys.argv],
        "config_source": str(config_path),
        "config_snapshot": str(snapshot_path),
        "config": config.to_manifest(),
        "implementation_sources": implementation_sources,
        "inputs": inputs,
        "source_probe": video_probe,
        "compiled_prompt": {
            "path": str(prompt_path),
            "bytes": len(prompt.encode("utf-8")),
            "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        },
        "candidate_plan": str(output / "candidate-plan.json"),
        "candidate_count": len(plans),
        "git": _git_state(),
        "packages": _package_state(output),
        "gpu_contract": {
            "client_is_gpu_entrypoint": False,
            "inference_authority": "official_joyai_websocket_server",
            "require_server_gpu_manifest": True,
            "note": (
                "The server launcher must select a physical GPU, set "
                "CUDA_VISIBLE_DEVICES, and save its GPU inventory with the run."
            ),
        },
        "selection_contract": {
            "best_of_n": True,
            "independent_inverse_action_audit_required": True,
            "visual_hard_gates": [
                "action_consistency",
                "source_motion_preservation",
                "object_identity",
                "embodiment_identity",
                "temporal_continuity",
                "native_resolution_human_veto",
            ],
            "aggregate_score_may_override_hard_gate": False,
        },
        "claim_boundary": (
            "The harness can produce and select a perceptually plausible real-scene "
            "action rendering. It is not SC3-Eval, numerical action conditioning, "
            "metric calibration, contact force evidence, or executable robot control."
        ),
        "physical_evidence": False,
        "physical_promotion": "REJECTED",
        "physical_reason": "proposal_not_physical_calibration",
    }
    _write_json(output / "manifest.json", manifest)
    return manifest, plans


def _execute(output: Path, plans: tuple[dict[str, Any], ...], limit: int | None) -> int:
    if limit is not None and limit <= 0:
        raise ValueError("candidate_limit must be positive")
    selected_plans = plans if limit is None else plans[:limit]
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    results = []
    for plan in selected_plans:
        command = [str(item) for item in plan["command"]]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        log_path = logs / f"{plan['candidate_id']}.log"
        log_path.write_text(
            "$ " + shlex.join(command) + "\n" + completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        results.append(
            {
                "candidate_id": plan["candidate_id"],
                "returncode": completed.returncode,
                "log": str(log_path),
            }
        )
        if completed.returncode:
            break

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_completed = len(results) == len(selected_plans) and all(
        result["returncode"] == 0 for result in results
    )
    manifest["execution_status"] = "CANDIDATES_READY" if all_completed else "FAILED"
    manifest["execution_completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["execution_results"] = results
    manifest["selection_status"] = "AWAITING_INDEPENDENT_AUDIT"
    _write_json(manifest_path, manifest)
    return 0 if all_completed else 1


def _select(args: argparse.Namespace) -> int:
    config = load_action_intent_config(
        args.config.expanduser().resolve(), project_root=PROJECT_ROOT
    )
    audits = []
    sources = []
    for path in args.audit_report:
        resolved = path.expanduser().resolve()
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"audit report must contain one JSON object: {resolved}")
        audits.append(raw)
        sources.append(str(resolved))
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.expanduser().resolve()),
        "audit_reports": sources,
        **select_visual_candidate(config, audits),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selection report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["selected_candidate"] is not None else 2


def main() -> int:
    args = _parser().parse_args()
    if args.command == "select":
        return _select(args)
    output = args.output_dir.expanduser().resolve()
    server_record = (
        validate_server_manifest(args.server_manifest) if args.command == "run" else None
    )
    manifest, plans = _compile(config_path=args.config, output=output, server_url=args.server_url)
    if args.command == "run":
        manifest["gpu_contract"]["server_manifest"] = server_record
        _write_json(output / "manifest.json", manifest)
        return _execute(output, plans, args.candidate_limit)
    print(
        json.dumps(
            {
                "experiment": str(output),
                "status": manifest["status"],
                "execution_status": manifest["execution_status"],
                "candidate_count": len(plans),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
