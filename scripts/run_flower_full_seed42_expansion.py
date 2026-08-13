#!/usr/bin/env python3
"""Generate overlapping seed-42 Wan replacement windows around a frozen anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--condition-video", type=Path, required=True)
    parser.add_argument("--anchor-candidate", type=Path, required=True)
    parser.add_argument("--anchor-gate-report", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_info(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    row = json.loads(completed.stdout)["streams"][0]
    return {
        "width": int(row["width"]),
        "height": int(row["height"]),
        "fps": row["r_frame_rate"],
        "frames": int(row["nb_frames"]),
    }


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {"head": run("rev-parse", "HEAD"), "status_porcelain": run("status", "--porcelain")}


def _extract_window(
    source: Path,
    condition: Path,
    output: Path,
    *,
    start: int,
    frames: int,
    width: int,
    height: int,
    fps: int,
) -> tuple[Path, Path, list[list[str]]]:
    inputs = output / "inputs"
    inputs.mkdir(parents=True)
    window = inputs / f"source-{start:04d}-{start + frames - 1:04d}.mp4"
    target = inputs / f"robot-condition-{start:04d}.png"
    window_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        (
            f"trim=start_frame={start}:end_frame={start + frames},"
            f"scale={width}:{height}:flags=lanczos,setpts=PTS-STARTPTS"
        ),
        "-frames:v",
        str(frames),
        "-r",
        str(fps),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(window),
    ]
    target_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(condition),
        "-vf",
        f"select=eq(n\\,{start}),scale={width}:{height}:flags=lanczos",
        "-frames:v",
        "1",
        str(target),
    ]
    subprocess.run(window_command, check=True)
    subprocess.run(target_command, check=True)
    if _video_info(window)["frames"] != frames:
        raise RuntimeError(f"window {start} did not contain {frames} frames")
    return window, target, [window_command, target_command]


def main() -> int:
    args = _parser().parse_args()
    paths = {
        "config": args.config.expanduser().resolve(),
        "source_video": args.source_video.expanduser().resolve(),
        "condition_video": args.condition_video.expanduser().resolve(),
        "anchor_candidate": args.anchor_candidate.expanduser().resolve(),
        "anchor_gate_report": args.anchor_gate_report.expanduser().resolve(),
        "project_root": args.project_root.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {name}: {path}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    config = json.loads(paths["config"].read_text())
    if config.get("schema_version") != "1.0.0":
        raise ValueError("config must use schema_version 1.0.0")
    starts = config["starts"]
    if starts != sorted(set(starts)):
        raise ValueError("window starts must be unique and increasing")
    frame_count = int(config["source_frames"])
    window_frames = int(config["window_frames"])
    if starts[0] != 0 or starts[-1] + window_frames != frame_count:
        raise ValueError("window plan must cover the first and final source frame")
    anchor_start = int(config["frozen_anchor"]["start_frame_inclusive"])
    if anchor_start not in starts:
        raise ValueError("frozen anchor must be one planned window")
    if _sha256(paths["anchor_candidate"]) != config["frozen_anchor"]["candidate_sha256"]:
        raise ValueError("frozen anchor candidate hash mismatch")
    gate = json.loads(paths["anchor_gate_report"].read_text())
    if gate.get("status") != "WORKING" or gate.get("decision") != "ALLOW_FULL_FILM_EXPANSION":
        raise ValueError("anchor strict gate does not allow expansion")
    source_info = _video_info(paths["source_video"])
    condition_info = _video_info(paths["condition_video"])
    if source_info["frames"] != frame_count or condition_info["frames"] != frame_count:
        raise ValueError("source and condition videos must match source_frames")

    gpus, inventory, processes = query_gpus()
    selected = [
        select_gpu(gpus, int(index), int(config["minimum_free_gpu_gib"]) * 1024)
        for index in config["gpu_physical_indices"]
    ]
    output.mkdir(parents=True)
    provenance = output / "provenance"
    provenance.mkdir()
    frozen_source = provenance / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), frozen_source)
    shutil.copy2(paths["config"], provenance / paths["config"].name)
    (output / "prompt.txt").write_text(config["prompt"] + "\n")
    preflight = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "source": {"path": str(paths["source_video"]), "sha256": _sha256(paths["source_video"]), "info": source_info},
        "condition": {"path": str(paths["condition_video"]), "sha256": _sha256(paths["condition_video"]), "info": condition_info},
        "anchor_candidate": {"path": str(paths["anchor_candidate"]), "sha256": _sha256(paths["anchor_candidate"])},
        "anchor_gate_report": {"path": str(paths["anchor_gate_report"]), "sha256": _sha256(paths["anchor_gate_report"])},
        "gpu_inventory_raw": inventory,
        "gpu_processes_raw": processes,
        "selected_gpus": [row.__dict__ for row in selected],
        "git": _git_state(paths["project_root"]),
        "execution_source": {"path": str(frozen_source), "sha256": _sha256(frozen_source)},
    }
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")

    window_rows = []
    for index, start in enumerate(starts):
        window_output = output / f"window-{index:02d}-{start:04d}"
        window_output.mkdir()
        window, target, extraction_commands = _extract_window(
            paths["source_video"],
            paths["condition_video"],
            window_output,
            start=start,
            frames=window_frames,
            width=int(config["output_width"]),
            height=int(config["output_height"]),
            fps=int(config["fps"]),
        )
        row = {
            "index": index,
            "start_frame": start,
            "end_frame_inclusive": start + window_frames - 1,
            "window_dir": str(window_output),
            "source_window": str(window),
            "source_window_sha256": _sha256(window),
            "target_image": str(target),
            "target_image_sha256": _sha256(target),
            "extraction_commands": extraction_commands,
            "status": "FROZEN_ANCHOR" if start == anchor_start else "PENDING",
        }
        if start == anchor_start:
            anchor_dir = window_output / "frozen-anchor"
            anchor_dir.mkdir()
            candidate_link = anchor_dir / "candidate.mp4"
            gate_link = anchor_dir / "gate-report.json"
            candidate_link.symlink_to(paths["anchor_candidate"])
            gate_link.symlink_to(paths["anchor_gate_report"])
            row["candidate"] = str(candidate_link)
            row["candidate_sha256"] = _sha256(paths["anchor_candidate"])
            row["gate_report"] = str(gate_link)
        window_rows.append(row)
    (output / "window-plan.json").write_text(json.dumps(window_rows, indent=2, sort_keys=True) + "\n")

    pending = [row for row in window_rows if row["status"] == "PENDING"]
    assignments = {row.physical_index: [] for row in selected}
    for index, row in enumerate(pending):
        assignments[selected[index % len(selected)].physical_index].append(row)

    def worker(physical_gpu: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for row in rows:
            window_dir = Path(row["window_dir"])
            log_path = window_dir / "controller.log"
            command = [
                str(paths["project_root"] / ".venv-gpu/bin/python"),
                str(paths["project_root"] / "scripts/run_agentic_phizero_proxy.py"),
                "--source-video",
                row["source_window"],
                "--reference-video",
                row["source_window"],
                "--target-image",
                row["target_image"],
                "--seed",
                str(config["seed"]),
                "--prompt",
                config["prompt"],
                "--evaluator",
                str(paths["project_root"] / "scripts/local_video_evaluator.py"),
                "--wan-repo",
                str(paths["project_root"] / "external/Wan2.2"),
                "--checkpoint-dir",
                str(paths["project_root"] / "checkpoints/Wan2.2-Animate-14B"),
                "--mode",
                "replacement",
                "--experiment-root",
                str(window_dir / "agent"),
                "--maximum-rounds",
                str(config["maximum_rounds_per_new_window"]),
                "--gpu",
                str(physical_gpu),
                "--minimum-free-gpu-gib",
                str(config["minimum_free_gpu_gib"]),
                "--width",
                str(config["output_width"]),
                "--height",
                str(config["output_height"]),
                "--fps",
                str(config["fps"]),
                "--frame-num",
                str(window_frames),
                "--infer-frames",
                str(config["infer_frames"]),
                "--reference-frames",
                str(config["reference_frames"]),
                "--object-roi",
                *[str(value) for value in config["object_roi_xywh_normalized"]],
            ]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
            started = datetime.now(timezone.utc).isoformat()
            with log_path.open("w") as log:
                completed = subprocess.run(
                    command,
                    cwd=paths["project_root"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            result = {
                **row,
                "physical_gpu": physical_gpu,
                "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                "command": command,
                "started_at": started,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "returncode": completed.returncode,
                "controller_log": str(log_path),
                "status": "SUCCEEDED" if completed.returncode == 0 else "FAILED",
            }
            if completed.returncode == 0:
                traces = list((window_dir / "agent").glob("*/trace.json"))
                candidates = list((window_dir / "agent").glob("*/candidates/000.mp4"))
                if len(traces) != 1 or len(candidates) != 1:
                    result["status"] = "FAILED_OUTPUT_DISCOVERY"
                else:
                    result["trace"] = str(traces[0])
                    result["candidate"] = str(candidates[0])
                    result["candidate_sha256"] = _sha256(candidates[0])
                    evaluations = list((window_dir / "agent").glob("*/candidates/000.evaluation.json"))
                    if len(evaluations) == 1:
                        result["evaluation"] = str(evaluations[0])
            (window_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            results.append(result)
        return results

    generated = []
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        futures = {
            executor.submit(worker, physical_gpu, rows): physical_gpu
            for physical_gpu, rows in assignments.items()
        }
        for future in as_completed(futures):
            generated.extend(future.result())
    generated.sort(key=lambda row: row["start_frame"])
    final_rows = []
    generated_by_start = {row["start_frame"]: row for row in generated}
    for row in window_rows:
        final_rows.append(generated_by_start.get(row["start_frame"], row))
    succeeded = all(row["status"] in {"SUCCEEDED", "FROZEN_ANCHOR"} for row in final_rows)
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL" if succeeded else "BLOCKED",
        "decision": "READY_FOR_WINDOW_REVIEW" if succeeded else "WINDOW_GENERATION_FAILED",
        "method": "wan2.2_replacement_relighting_lora_seed42_overlapping_windows",
        "command": [sys.executable, *sys.argv],
        "seed": config["seed"],
        "coordinate_frame": config["coordinate_frame"],
        "preflight": preflight,
        "windows": final_rows,
        "limitations": [
            "New windows are raw seed-42 candidates and are not accepted until dense human-removal, hand, object, and temporal review completes.",
            "The frozen 110--198 anchor is symlinked and never overwritten by this generator.",
            "The upstream replacement backend accepts prompts but the pinned generator may not use prompt text internally; seed and conditioning remain the operative controls."
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output), "status": manifest["status"], "decision": manifest["decision"]}, indent=2))
    return 0 if succeeded else 2


if __name__ == "__main__":
    raise SystemExit(main())
