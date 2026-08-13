#!/usr/bin/env python3
"""Extract source-anchored JoyAI repair windows with complete provenance."""

from __future__ import annotations

import argparse
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
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.joyai_video_edit import (  # noqa: E402
    DEFAULT_FLOWER_WINDOWS,
    JoyAIFlowerEditContract,
    JoyAIWindow,
    sha256_file,
    write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path(shutil.which("ffprobe") or "ffprobe"))
    parser.add_argument(
        "--window",
        type=int,
        nargs=3,
        action="append",
        metavar=("START", "END", "SEAM"),
        help="inclusive full-timeline range; repeat for multiple 1+8n windows",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-isotropic-fit-height-upscale",
        action="store_true",
        help=(
            "for an explicitly low-resolution incumbent, resize isotropically to model height "
            "then center-crop; records the transform and never grants metric authority"
        ),
    )
    return parser


def probe_video(ffprobe: Path, video: Path) -> dict[str, Any]:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,pix_fmt,codec_name",
        "-of",
        "json",
        str(video),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected exactly one video stream in {video}")
    stream = streams[0]
    rate = Fraction(stream["avg_frame_rate"])
    frames = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps_fraction": str(rate),
        "fps": float(rate),
        "frames": frames,
        "codec_name": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "probe_command": command,
    }


def build_extract_command(
    *, ffmpeg: Path, video: Path, output: Path, window: JoyAIWindow, contract: JoyAIFlowerEditContract
) -> list[str]:
    select = f"between(n\\,{window.start_frame}\\,{window.end_frame})"
    transforms = []
    if contract.transform_kind == "isotropic_fit_height_then_center_crop":
        transforms.append(f"scale={contract.resized_width}:{contract.model_height}:flags=lanczos")
    transforms.append(
        f"crop={contract.model_width}:{contract.model_height}:{contract.crop_left}:{contract.crop_top}"
    )
    filtergraph = f"select='{select}'," + ",".join(transforms) + f",setpts=N/({contract.fps}*TB)"
    return [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        filtergraph,
        "-an",
        "-fps_mode",
        "passthrough",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "bgr0",
        str(output),
    ]


def build_endpoint_command(
    *, ffmpeg: Path, video: Path, output: Path, frame: int, contract: JoyAIFlowerEditContract
) -> list[str]:
    transforms = []
    if contract.transform_kind == "isotropic_fit_height_then_center_crop":
        transforms.append(f"scale={contract.resized_width}:{contract.model_height}:flags=lanczos")
    transforms.append(
        f"crop={contract.model_width}:{contract.model_height}:{contract.crop_left}:{contract.crop_top}"
    )
    filtergraph = f"select='eq(n\\,{frame})'," + ",".join(transforms)
    return [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        filtergraph,
        "-frames:v",
        "1",
        str(output),
    ]


def _run(command: Sequence[str], log: Path) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    log.write_text(
        "$ " + shlex.join(command) + "\n" + completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"command returned {completed.returncode}; inspect {log}")


def _git_state(output: Path) -> dict[str, Any]:
    def capture(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    status = capture("status", "--short")
    status_path = output / "git-status.txt"
    status_path.write_text(status + ("\n" if status else ""), encoding="utf-8")
    return {
        "head": capture("rev-parse", "HEAD"),
        "branch": capture("branch", "--show-current"),
        "dirty": bool(status),
        "status_path": str(status_path),
        "status_sha256": sha256_file(status_path),
    }


def _package_state(output: Path) -> dict[str, Any]:
    packages_path = output / "packages.txt"
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False
    )
    packages_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages_path": str(packages_path),
        "packages_sha256": sha256_file(packages_path),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    candidate = args.candidate_video.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    ffprobe = args.ffprobe.expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    if output.exists():
        raise FileExistsError(f"JoyAI preparation experiment already exists: {output}")
    if not ffmpeg.is_file() or not ffprobe.is_file():
        raise FileNotFoundError("ffmpeg and ffprobe must be real executable paths")

    windows = tuple(JoyAIWindow(*row) for row in args.window) if args.window else DEFAULT_FLOWER_WINDOWS
    source = probe_video(ffprobe, candidate)
    if args.allow_isotropic_fit_height_upscale:
        resized_width = round(source["width"] * 720 / source["height"])
        if resized_width < 1248:
            raise ValueError("fit-height resize is too narrow for the 1248-pixel JoyAI crop")
        contract = JoyAIFlowerEditContract(
            windows=windows,
            source_width=source["width"],
            source_height=source["height"],
            crop_left=(resized_width - 1248) // 2,
            transform_kind="isotropic_fit_height_then_center_crop",
            resized_width=resized_width,
            seed=args.seed,
        )
    else:
        contract = JoyAIFlowerEditContract(windows=windows, seed=args.seed)
    contract.validate()
    if not args.allow_isotropic_fit_height_upscale and (
        source["width"], source["height"]
    ) != (contract.source_width, contract.source_height):
        raise ValueError(
            f"source dimensions {source['width']}x{source['height']} do not match explicit "
            f"frame {contract.source_width}x{contract.source_height}"
        )
    if Fraction(source["fps_fraction"]) != Fraction(contract.fps, 1):
        raise ValueError(f"source FPS {source['fps_fraction']} != {contract.fps}")
    if source["frames"] <= max(window.end_frame for window in windows):
        raise ValueError("source video is shorter than the requested full-timeline window")

    output.mkdir(parents=True)
    inputs = output / "inputs"
    logs = output / "logs"
    anchors = output / "anchors"
    inputs.mkdir()
    logs.mkdir()
    anchors.mkdir()
    commands: list[list[str]] = [source["probe_command"]]
    outputs = []
    for index, window in enumerate(windows):
        name = f"window-{index:02d}-frames-{window.start_frame:04d}-{window.end_frame:04d}"
        video_path = inputs / f"{name}-joyai-1248x720-ffv1.mkv"
        command = build_extract_command(
            ffmpeg=ffmpeg, video=candidate, output=video_path, window=window, contract=contract
        )
        _run(command, logs / f"extract-{index:02d}.log")
        commands.append(command)
        window_probe = probe_video(ffprobe, video_path)
        commands.append(window_probe["probe_command"])
        if window_probe["frames"] != window.frame_count:
            raise RuntimeError(
                f"extracted {window_probe['frames']} frames for {name}; expected {window.frame_count}"
            )
        endpoint_rows = []
        for label, frame in (("first", window.start_frame), ("last", window.end_frame)):
            endpoint = anchors / f"{name}-{label}-source-locked.png"
            endpoint_command = build_endpoint_command(
                ffmpeg=ffmpeg, video=candidate, output=endpoint, frame=frame, contract=contract
            )
            _run(endpoint_command, logs / f"endpoint-{index:02d}-{label}.log")
            commands.append(endpoint_command)
            endpoint_rows.append(
                {"label": label, "full_timeline_frame": frame, "path": str(endpoint), "sha256": sha256_file(endpoint)}
            )
        outputs.append(
            {
                "window": {
                    "start_frame": window.start_frame,
                    "end_frame": window.end_frame,
                    "seam_frame": window.seam_frame,
                    "frame_count": window.frame_count,
                },
                "model_input": {
                    "path": str(video_path),
                    "sha256": sha256_file(video_path),
                    "probe": window_probe,
                },
                "locked_endpoints": endpoint_rows,
            }
        )

    write_json(output / "commands.json", commands)
    manifest = {
        **contract.to_manifest(),
        "status": "WORKING",
        "stage": "joyai_input_preparation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "source_video": {
            "path": str(candidate),
            "sha256": sha256_file(candidate),
            "probe": source,
        },
        "outputs": outputs,
        "commands": {"path": str(output / "commands.json"), "sha256": sha256_file(output / "commands.json")},
        "git": _git_state(output),
        "runtime": _package_state(output),
        "acceptance": {
            "all_windows_exact_1_plus_8n": all((row.frame_count - 1) % 8 == 0 for row in windows),
            "integer_crop_no_rescale": contract.transform_kind == "integer_center_crop_no_rescale",
            "explicit_isotropic_upscale": contract.transform_kind == "isotropic_fit_height_then_center_crop",
            "source_endpoints_archived": True,
            "candidate_generated": False,
            "physical_evidence": False,
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    manifest = prepare(_parser().parse_args())
    print(
        json.dumps(
            {
                "experiment": str(Path(manifest["commands"]["path"]).parent),
                "status": manifest["status"],
                "windows": len(manifest["outputs"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
