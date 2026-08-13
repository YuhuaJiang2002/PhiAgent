#!/usr/bin/env python3
"""Retake only human-residual full-film windows with black face control."""

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
    parser.add_argument("--parent-expansion", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {"head": run("rev-parse", "HEAD"), "status_porcelain": run("status", "--porcelain")}


def _discover_one(root: Path, pattern: str) -> Path | None:
    rows = [path for path in root.glob(pattern) if path.is_file() and path.stat().st_size]
    return rows[0] if len(rows) == 1 else None


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    parent = args.parent_expansion.expanduser().resolve()
    project = args.project_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for label, path in (("config", config_path), ("parent expansion", parent), ("project root", project)):
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "1.0.0":
        raise ValueError("config must use schema_version 1.0.0")
    if config.get("coordinate_frame") != "camera:source_video_pixels":
        raise ValueError("config must name camera:source_video_pixels")
    starts = [int(value) for value in config["retake_starts"]]
    if starts != sorted(set(starts)):
        raise ValueError("retake_starts must be unique and increasing")
    parent_plan = json.loads((parent / "window-plan.json").read_text())
    by_start = {int(row["start_frame"]): row for row in parent_plan}
    missing = sorted(set(starts) - set(by_start))
    if missing:
        raise ValueError(f"retake starts are absent from the parent plan: {missing}")

    gpus, inventory, processes = query_gpus()
    selected = [
        select_gpu(gpus, int(index), int(config["minimum_free_gpu_gib"]) * 1024)
        for index in config["gpu_physical_indices"]
    ]
    output.mkdir(parents=True)
    provenance = output / "provenance"
    provenance.mkdir()
    frozen_script = provenance / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), frozen_script)
    frozen_config = provenance / config_path.name
    shutil.copy2(config_path, frozen_config)
    source_rows = []
    for start in starts:
        parent_row = by_start[start]
        source = Path(parent_row["source_window"]).resolve()
        target = Path(parent_row["target_image"]).resolve()
        if not source.is_file() or not target.is_file():
            raise FileNotFoundError(f"parent inputs are incomplete for start {start}")
        source_rows.append(
            {
                "start_frame": start,
                "end_frame_inclusive": start + int(config["window_frames"]) - 1,
                "source_window": str(source),
                "source_window_sha256": _sha256(source),
                "target_image": str(target),
                "target_image_sha256": _sha256(target),
            }
        )
    preflight = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "gpu_inventory_raw": inventory,
        "gpu_processes_raw": processes,
        "selected_gpus": [row.__dict__ for row in selected],
        "git": _git_state(project),
        "parent_expansion": str(parent),
        "parent_manifest_sha256": _sha256(parent / "manifest.json"),
        "execution_source": {"path": str(frozen_script), "sha256": _sha256(frozen_script)},
        "config": {"path": str(frozen_config), "sha256": _sha256(frozen_config)},
        "windows": source_rows,
    }
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    assignments: dict[int, list[dict[str, Any]]] = {
        row.physical_index: [] for row in selected
    }
    for index, row in enumerate(source_rows):
        assignments[selected[index % len(selected)].physical_index].append(row)

    def worker(gpu: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for row in rows:
            start = int(row["start_frame"])
            window = output / f"window-{start:04d}"
            window.mkdir()
            command = [
                str(project / ".venv-gpu/bin/python"),
                str(project / "scripts/run_agentic_phizero_proxy.py"),
                "--source-video", row["source_window"],
                "--reference-video", row["source_window"],
                "--target-image", row["target_image"],
                "--seed", str(config["seed"]),
                "--prompt", config["prompt"],
                "--evaluator", str(project / "scripts/local_video_evaluator.py"),
                "--wan-repo", str(project / "external/Wan2.2"),
                "--checkpoint-dir", str(project / "checkpoints/Wan2.2-Animate-14B"),
                "--mode", "replacement",
                "--experiment-root", str(window / "agent"),
                "--maximum-rounds", str(config["maximum_rounds_per_window"]),
                "--gpu", str(gpu),
                "--minimum-free-gpu-gib", str(config["minimum_free_gpu_gib"]),
                "--width", str(config["output_width"]),
                "--height", str(config["output_height"]),
                "--fps", str(config["fps"]),
                "--frame-num", str(config["window_frames"]),
                "--infer-frames", str(config["infer_frames"]),
                "--reference-frames", str(config["reference_frames"]),
                "--object-roi", *[str(value) for value in config["object_roi_xywh_normalized"]],
                "--suppress-source-face-control",
            ]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            started = datetime.now(timezone.utc).isoformat()
            log = window / "controller.log"
            with log.open("w") as handle:
                completed = subprocess.run(
                    command,
                    cwd=project,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            candidate = _discover_one(window / "agent", "*/candidates/000.mp4")
            evaluation = _discover_one(window / "agent", "*/candidates/000.evaluation.json")
            trace = _discover_one(window / "agent", "*/trace.json")
            result = {
                **row,
                "physical_gpu": gpu,
                "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                "command": command,
                "started_at": started,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "returncode": completed.returncode,
                "controller_log": str(log),
                "status": "RAW_CANDIDATE_PRESERVED" if candidate else "NO_CANDIDATE",
                "proxy_accepted": completed.returncode == 0,
            }
            if candidate:
                result["candidate"] = str(candidate)
                result["candidate_sha256"] = _sha256(candidate)
            if evaluation:
                result["evaluation"] = str(evaluation)
                result["evaluation_sha256"] = _sha256(evaluation)
            if trace:
                result["trace"] = str(trace)
                result["trace_sha256"] = _sha256(trace)
            (window / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            results.append(result)
        return results

    results = []
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        futures = {
            executor.submit(worker, gpu, rows): gpu
            for gpu, rows in assignments.items()
        }
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda row: row["start_frame"])
    all_preserved = all(row["status"] == "RAW_CANDIDATE_PRESERVED" for row in results)
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "wan2.2_replacement_seed42_black_source_face_control_retakes_v2",
        "status": "PARTIAL" if all_preserved else "BLOCKED",
        "decision": "READY_FOR_DENSE_WINDOW_REVIEW" if all_preserved else "RETAKE_OUTPUT_MISSING",
        "command": [sys.executable, *sys.argv],
        "coordinate_frame": config["coordinate_frame"],
        "seed": config["seed"],
        "preflight": preflight,
        "windows": results,
        "limitations": [
            "Raw candidates are preserved even when the generic proxy rejects them; no retake is accepted without dense semantic review.",
            "Source-face suppression removes one known human-identity control but does not by itself prove complete human removal.",
            "Named flower/contact restoration and temporal repair are downstream stages and have not yet been applied to these retakes.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output), "status": manifest["status"], "decision": manifest["decision"], "windows": [{"start": row["start_frame"], "status": row["status"], "gpu": row["physical_gpu"]} for row in results]}, indent=2))
    return 0 if all_preserved else 2


if __name__ == "__main__":
    raise SystemExit(main())
