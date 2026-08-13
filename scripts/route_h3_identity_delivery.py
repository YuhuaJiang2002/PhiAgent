#!/usr/bin/env python3
"""Route only candidates that pass one matched, task-bound H3 contract."""

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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.agent.h3_identity_routing import (  # noqa: E402
    decide_identity_delivery,
    require_matched_delivery_context,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _probe(ffprobe: Path, video: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream: {video}")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": str(stream["r_frame_rate"]),
        "frames": int(stream["nb_read_frames"]),
    }


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _bound_sha256(payload: Mapping[str, object], input_name: str) -> str:
    inputs = _mapping(payload.get("inputs"), "assessment.inputs")
    entry = _mapping(inputs.get(input_name), f"assessment.inputs.{input_name}")
    value = entry.get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"assessment input {input_name} has no SHA-256")
    return value


def _metric_candidate_sha256(payload: Mapping[str, object]) -> str:
    inputs = _mapping(payload.get("inputs"), "metrics.inputs")
    entry = _mapping(inputs.get("candidate"), "metrics.inputs.candidate")
    value = entry.get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("metrics candidate has no SHA-256")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-assessment", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--fallback-assessment", type=Path, required=True)
    parser.add_argument("--fallback-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-comparison", action="store_true")
    parser.add_argument("--allow-blocked", action="store_true")
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/opt/homebrew/bin/ffprobe"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = {
        "candidate": args.candidate.expanduser().resolve(),
        "candidate_assessment": args.candidate_assessment.expanduser().resolve(),
        "candidate_metrics": args.candidate_metrics.expanduser().resolve(),
        "fallback": args.fallback.expanduser().resolve(),
        "fallback_assessment": args.fallback_assessment.expanduser().resolve(),
        "fallback_metrics": args.fallback_metrics.expanduser().resolve(),
        "ffprobe": args.ffprobe.expanduser().resolve(),
    }
    if args.build_comparison:
        paths["ffmpeg"] = args.ffmpeg.expanduser().resolve()
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{name} does not exist or is empty: {path}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"delivery output already exists: {output_dir}")

    candidate_assessment = json.loads(paths["candidate_assessment"].read_text())
    candidate_metrics = json.loads(paths["candidate_metrics"].read_text())
    fallback_assessment = json.loads(paths["fallback_assessment"].read_text())
    fallback_metrics = json.loads(paths["fallback_metrics"].read_text())
    payloads = {
        "candidate_assessment": candidate_assessment,
        "candidate_metrics": candidate_metrics,
        "fallback_assessment": fallback_assessment,
        "fallback_metrics": fallback_metrics,
    }
    if any(not isinstance(payload, dict) for payload in payloads.values()):
        raise ValueError("assessment and metrics files must contain JSON objects")

    candidate_sha256 = _sha256(paths["candidate"])
    fallback_sha256 = _sha256(paths["fallback"])
    bindings = (
        (
            "candidate assessment",
            _bound_sha256(candidate_assessment, "candidate_video"),
            candidate_sha256,
        ),
        (
            "candidate metrics",
            _metric_candidate_sha256(candidate_metrics),
            candidate_sha256,
        ),
        (
            "fallback assessment",
            _bound_sha256(fallback_assessment, "candidate_video"),
            fallback_sha256,
        ),
        (
            "fallback metrics",
            _metric_candidate_sha256(fallback_metrics),
            fallback_sha256,
        ),
        (
            "candidate assessment metrics",
            _bound_sha256(candidate_assessment, "candidate_metrics"),
            _sha256(paths["candidate_metrics"]),
        ),
        (
            "fallback assessment metrics",
            _bound_sha256(fallback_assessment, "candidate_metrics"),
            _sha256(paths["fallback_metrics"]),
        ),
    )
    for label, recorded, actual in bindings:
        if recorded != actual:
            raise ValueError(f"{label} SHA-256 mismatch: {recorded} != {actual}")
    matched_context = require_matched_delivery_context(
        candidate_metrics, fallback_metrics
    )

    candidate_probe = _probe(paths["ffprobe"], paths["candidate"])
    fallback_probe = _probe(paths["ffprobe"], paths["fallback"])
    if fallback_probe != candidate_probe:
        raise ValueError(
            "fallback must be assessed after exact frame selection and encoding: "
            f"{fallback_probe} != {candidate_probe}"
        )
    decision = decide_identity_delivery(candidate_assessment, fallback_assessment)
    output_dir.mkdir(parents=True)

    output = None
    comparison = None
    comparison_command = None
    if decision.deliverable:
        selected = (
            paths["candidate"] if decision.route == "candidate" else paths["fallback"]
        )
        output = output_dir / "delivered.mp4"
        shutil.copy2(selected, output)
        if args.build_comparison:
            comparison = output_dir / "candidate-left-delivered-right.mp4"
            comparison_command = [
                str(paths["ffmpeg"]),
                "-v",
                "error",
                "-i",
                str(paths["candidate"]),
                "-i",
                str(output),
                "-filter_complex",
                "[0:v][1:v]hstack=inputs=2[out]",
                "-map",
                "[out]",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(comparison),
            ]
            subprocess.run(comparison_command, check=True)

    manifest = {
        "schema_version": "2.0.0",
        "method": "matched_task_fail_closed_h3_identity_delivery_routing",
        "status": "completed" if decision.deliverable else "blocked",
        "honest_status": "WORKING" if decision.deliverable else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "comparison_command": comparison_command,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "decision": asdict(decision),
        "matched_delivery_context_sha256": matched_context,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "candidate_probe": candidate_probe,
        "fallback_probe": fallback_probe,
        "output": (
            {
                "path": str(output),
                "sha256": _sha256(output),
                "probe": _probe(paths["ffprobe"], output),
            }
            if output is not None
            else None
        ),
        "comparison": (
            {
                "path": str(comparison),
                "sha256": _sha256(comparison),
                "layout": "reviewed candidate left; delivered output right",
                "probe": _probe(paths["ffprobe"], comparison),
            }
            if comparison is not None
            else None
        ),
        "limitations": [
            "No video is emitted when both task-bound assessments fail.",
            "Visible 2-D topology and action metrics do not establish physical execution.",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    if decision.deliverable or args.allow_blocked:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
