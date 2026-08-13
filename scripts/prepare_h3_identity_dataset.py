#!/usr/bin/env python3
"""Compile a rights-attributed, immutable MiniMax-H3 identity dataset."""

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
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.training.h3_identity_rsi import (  # noqa: E402
    IdentityDatasetPlan,
    build_diffsynth_metadata,
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


def _git_state(root: Path) -> dict[str, object]:
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "available": status.returncode == 0,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
        "error": status.stderr.strip() if status.returncode else None,
    }


def _run(command: list[str], *, log: Path | None = None) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if log is not None:
        with log.open("a", encoding="utf-8") as handle:
            handle.write("$ " + shlex.join(command) + "\n")
            handle.write(completed.stdout)
            handle.write(completed.stderr)
    return completed.stdout


def _probe(ffprobe: Path, video: Path) -> dict[str, object]:
    payload = json.loads(
        _run(
            [
                str(ffprobe),
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(video),
            ]
        )
    )
    streams = payload.get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(video_streams) != 1 or len(audio_streams) != 1:
        raise RuntimeError(f"prepared clip must contain one video and one audio stream: {video}")
    return {
        "video": video_streams[0],
        "audio": audio_streams[0],
        "format": payload.get("format", {}),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/h3-identity-datasets"))
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    plan_path = args.plan.expanduser().resolve()
    plan = IdentityDatasetPlan.load(plan_path)
    clips = plan.split(args.split)
    if not clips:
        raise ValueError(f"plan {plan.name} has no {args.split} clips")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = (
        args.experiment_dir.expanduser().resolve()
        if args.experiment_dir
        else args.output_root.expanduser().resolve() / f"{stamp}-{uuid4().hex[:8]}"
    )
    experiment.mkdir(parents=True, exist_ok=False)
    manifest_path = experiment / "manifest.json"
    ffmpeg = args.ffmpeg or Path(shutil.which("ffmpeg") or "")
    ffprobe = args.ffprobe or Path(shutil.which("ffprobe") or "")
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "method": "h3_native_identity_rights_attributed_dataset_compilation",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git": _git_state(project_root),
        "seed": 0,
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "split": args.split,
        "dataset_contract": {
            "fps": plan.fps,
            "width": plan.width,
            "height": plan.height,
            "num_frames": plan.num_frames,
            "temporal_grouping": "17n+5",
            "reference_condition": "one explicit source frame per target clip",
        },
        "records": [],
    }
    _write_json(manifest_path, manifest)
    log_path = experiment / "commands.log"
    try:
        for executable in (ffmpeg, ffprobe):
            if not executable.is_file():
                raise ValueError(f"required executable not found: {executable}")
        clip_dir = experiment / "clips"
        reference_dir = experiment / "references"
        clip_dir.mkdir()
        reference_dir.mkdir()
        metadata_inputs = []
        records = []
        for spec in clips:
            source = (project_root / spec.source_video).resolve()
            try:
                source.relative_to(project_root)
            except ValueError as error:
                raise ValueError(f"source escaped the project root: {source}") from error
            if not source.is_file() or source.stat().st_size == 0:
                raise ValueError(f"source video is missing or empty: {source}")
            output = clip_dir / f"{spec.clip_id}.mp4"
            reference = reference_dir / f"{spec.clip_id}.png"
            if spec.source_crop is not None:
                crop_x, crop_y, crop_width, crop_height = spec.source_crop
                crop_filter = f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},"
            else:
                crop_filter = ""
            video_filter = crop_filter + (
                f"fps={plan.fps},"
                f"scale={plan.width}:{plan.height}:force_original_aspect_ratio=increase,"
                f"crop={plan.width}:{plan.height},setsar=1"
            )
            compile_command = [
                str(ffmpeg),
                "-v",
                "error",
                "-ss",
                f"{spec.source_start_seconds:.6f}",
                "-i",
                str(source),
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=32000",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                video_filter,
                "-frames:v",
                str(plan.num_frames),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "14",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-ar",
                "32000",
                "-ac",
                "2",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output),
            ]
            _run(compile_command, log=log_path)
            reference_command = [
                str(ffmpeg),
                "-v",
                "error",
                "-i",
                str(output),
                "-vf",
                f"select=eq(n\\,{spec.reference_frame})",
                "-frames:v",
                "1",
                str(reference),
            ]
            _run(reference_command, log=log_path)
            probe = _probe(ffprobe, output)
            video_stream = probe["video"]
            audio_stream = probe["audio"]
            assert isinstance(video_stream, dict) and isinstance(audio_stream, dict)
            if int(video_stream.get("nb_frames", 0)) != plan.num_frames:
                raise RuntimeError(
                    f"{output} has {video_stream.get('nb_frames')} frames, expected {plan.num_frames}"
                )
            if (int(video_stream["width"]), int(video_stream["height"])) != (
                plan.width,
                plan.height,
            ):
                raise RuntimeError(f"prepared dimensions do not match plan: {output}")
            if int(audio_stream.get("sample_rate", 0)) != 32000:
                raise RuntimeError(f"prepared audio is not 32 kHz: {output}")
            relative_video = output.relative_to(experiment).as_posix()
            relative_reference = reference.relative_to(experiment).as_posix()
            metadata_inputs.append((spec, relative_video, relative_reference))
            records.append(
                {
                    "clip": spec.clip_id,
                    "subject": spec.subject_id,
                    "scene": spec.scene_id,
                    "review_status": spec.review_status,
                    "source": str(source),
                    "source_sha256": _sha256(source),
                    "source_start_seconds": spec.source_start_seconds,
                    "source_crop": spec.source_crop,
                    "license_id": spec.license_id,
                    "source_uri": spec.source_uri,
                    "video": relative_video,
                    "video_sha256": _sha256(output),
                    "reference": relative_reference,
                    "reference_sha256": _sha256(reference),
                    "probe": probe,
                    "commands": [compile_command, reference_command],
                }
            )
        metadata = build_diffsynth_metadata(metadata_inputs)
        metadata_path = experiment / "metadata.json"
        _write_json(metadata_path, metadata)
        shutil.copy2(plan_path, experiment / "plan.json")
        manifest.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "metadata": str(metadata_path),
                "metadata_sha256": _sha256(metadata_path),
                "records": records,
                "accepted_for_training": all(
                    record["review_status"] in {"accepted", "partial"} for record in records
                ),
                "limitations": [
                    "Deterministic rendered identity is supervision for appearance/topology, not evidence of real-robot physics.",
                    "PARTIAL source clips remain labelled and may be excluded by later RSI rounds.",
                ],
            }
        )
        _write_json(manifest_path, manifest)
        print(json.dumps({"dataset": str(experiment), "clips": len(records)}))
        return 0
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
