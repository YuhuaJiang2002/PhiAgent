#!/usr/bin/env python3
"""Package a hash-bound JoyAI SC3 diagnostic comparison without promoting it."""

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
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.joyai_video_edit import sha256_file, write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--confirmation-run-dir",
        type=Path,
        help="Independent repeat run that must select the same seed and pass automatic gates.",
    )
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--intent-label", required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    parser.add_argument(
        "--ffprobe", type=Path, default=Path(shutil.which("ffprobe") or "ffprobe")
    )
    return parser


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {resolved}")
    return resolved


def _score_for_selected_seed(manifest: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("JoyAI run manifest has no selection")
    selected_seed = int(selection["selected_seed"])
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("JoyAI run manifest has no candidate list")
    matches = [
        item
        for item in candidates
        if isinstance(item, Mapping) and int(item.get("seed", -1)) == selected_seed
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("score"), Mapping):
        raise ValueError("selected seed does not map to exactly one score")
    score = dict(matches[0]["score"])
    if score.get("automatic_pass") is not True:
        raise ValueError("refusing to package a candidate that failed automatic gates")
    if score.get("human_review_passed") is False:
        raise ValueError("refusing to package a human-rejected candidate")
    return selected_seed, score


def _probe_video(ffprobe: Path, path: Path) -> dict[str, Any]:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError(f"{path} must contain exactly one video stream")
    stream = streams[0]
    rate = Fraction(str(stream["avg_frame_rate"]))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps_numerator": rate.numerator,
        "fps_denominator": rate.denominator,
        "fps": float(rate),
        "frames": int(stream["nb_read_frames"]),
        "duration_seconds": float(stream["duration"]),
    }


def _font(size: int):
    try:
        from PIL import ImageFont
    except ImportError as exc:
        raise RuntimeError("showcase packaging requires the optional Pillow package") from exc
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size), str(path)
    return ImageFont.load_default(), "Pillow-default"


def _centered_x(draw: Any, text: str, font: Any, width: int, offset: int = 0) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return offset + max(0, (width - (box[2] - box[0])) // 2)


def _build_layout(
    output: Path,
    *,
    intent_label: str,
    seed: int,
    score: Mapping[str, Any],
    automatic_confirmations: int,
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("showcase packaging requires the optional Pillow package") from exc
    canvas = Image.new("RGB", (1280, 720), (12, 15, 20))
    draw = ImageDraw.Draw(canvas)
    title_font, title_font_path = _font(30)
    label_font, label_font_path = _font(24)
    detail_font, detail_font_path = _font(19)
    warning_font, warning_font_path = _font(18)
    draw.text(
        (_centered_x(draw, intent_label, title_font, 1280), 18),
        intent_label,
        font=title_font,
        fill=(245, 247, 250),
    )
    left = "ACTION CARRIER (motion authority)"
    right = f"JOYAI 0811 RESIDUAL RENDER (seed {seed})"
    draw.text(
        (_centered_x(draw, left, label_font, 640), 72),
        left,
        font=label_font,
        fill=(99, 179, 237),
    )
    draw.text(
        (_centered_x(draw, right, label_font, 640, 640), 72),
        right,
        font=label_font,
        fill=(104, 211, 145),
    )
    score_text = (
        f"action {float(score['action_adherence']):.3f}  |  "
        f"object {float(score['object_interaction']):.3f}  |  "
        f"embodiment {float(score['embodiment_consistency']):.3f}  |  "
        f"temporal {float(score['temporal_consistency']):.3f}  |  "
        f"background {float(score['background_consistency']):.3f}"
    )
    draw.text(
        (_centered_x(draw, score_text, detail_font, 1280), 610),
        score_text,
        font=detail_font,
        fill=(225, 230, 236),
    )
    warning = (
        f"PARTIAL - automatic gates {automatic_confirmations}/{automatic_confirmations} - "
        "human review pending - "
        "not physical execution or contact evidence"
    )
    draw.text(
        (_centered_x(draw, warning, warning_font, 1280), 655),
        warning,
        font=warning_font,
        fill=(252, 129, 129),
    )
    canvas.save(output, format="PNG", optimize=False)
    return {
        "title_font": title_font_path,
        "label_font": label_font_path,
        "detail_font": detail_font_path,
        "warning_font": warning_font_path,
    }


def _run(command: list[str], log_path: Path) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path.write_text(
        "$ "
        + shlex.join(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"showcase command failed; inspect {log_path}")


def _git_state() -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, arguments in {
        "head": ("rev-parse", "HEAD"),
        "branch": ("branch", "--show-current"),
        "status": ("status", "--short", "--untracked-files=no"),
    }.items():
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        state[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    for name, arguments in {
        "worktree_diff": ("diff", "--binary", "--no-ext-diff"),
        "cached_diff": ("diff", "--cached", "--binary", "--no-ext-diff"),
    }.items():
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
        state[name] = {
            "returncode": completed.returncode,
            "bytes": len(completed.stdout),
            "sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "stderr": completed.stderr.decode("utf-8", errors="replace").strip(),
        }
    return state


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
        "path": str(path),
        "sha256": sha256_file(path),
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }


def _tool_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"could not inspect {executable}: {completed.stderr.strip()}")
    return completed.stdout.splitlines()[0]


def main() -> int:
    args = _parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    run_manifest = _require_file(run_dir / "manifest.json", "JoyAI run manifest")
    carrier = _require_file(args.carrier, "action carrier")
    ffmpeg = _require_file(args.ffmpeg, "ffmpeg")
    ffprobe = _require_file(args.ffprobe, "ffprobe")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite showcase experiment: {output}")
    output.mkdir(parents=True)

    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("JoyAI run manifest must contain one JSON object")
    seed, score = _score_for_selected_seed(payload)
    confirmation_record = None
    automatic_confirmations = 1
    if args.confirmation_run_dir is not None:
        confirmation_dir = args.confirmation_run_dir.expanduser().resolve()
        confirmation_manifest = _require_file(
            confirmation_dir / "manifest.json", "confirmation run manifest"
        )
        confirmation_payload = json.loads(
            confirmation_manifest.read_text(encoding="utf-8")
        )
        if not isinstance(confirmation_payload, Mapping):
            raise ValueError("confirmation manifest must contain one JSON object")
        confirmation_seed, confirmation_score = _score_for_selected_seed(
            confirmation_payload
        )
        if confirmation_seed != seed:
            raise ValueError(
                f"confirmation selected seed {confirmation_seed}, expected {seed}"
            )
        automatic_confirmations += 1
        confirmation_record = {
            "run_manifest": {
                "path": str(confirmation_manifest),
                "sha256": sha256_file(confirmation_manifest),
            },
            "selected_seed": confirmation_seed,
            "score": confirmation_score,
        }
    candidate = _require_file(
        run_dir / "candidates" / f"seed-{seed}" / "candidate-restored-review.mp4",
        "selected restored candidate",
    )
    metadata_path = _require_file(
        run_dir / "candidates" / f"seed-{seed}" / "candidate-metadata.json",
        "selected candidate metadata",
    )
    evaluation_path = _require_file(
        run_dir / "candidates" / f"seed-{seed}" / "evaluation" / "evaluation.json",
        "selected candidate evaluation",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    observed_hash = sha256_file(candidate)
    expected_hashes = {
        str(metadata["review"]["sha256"]),
        str(evaluation["candidate_sha256"]),
    }
    if expected_hashes != {observed_hash}:
        raise ValueError(
            f"selected candidate hash evidence disagrees: {expected_hashes} vs {observed_hash}"
        )

    carrier_stream = _probe_video(ffprobe, carrier)
    candidate_stream = _probe_video(ffprobe, candidate)
    expected_stream = {
        "width": 640,
        "height": 480,
        "fps_numerator": 15,
        "fps_denominator": 1,
        "fps": 15.0,
        "frames": 81,
        "duration_seconds": 5.4,
    }
    for label, stream in (("carrier", carrier_stream), ("candidate", candidate_stream)):
        for key in ("width", "height", "fps_numerator", "fps_denominator", "frames"):
            if stream[key] != expected_stream[key]:
                raise ValueError(f"{label} stream {key}={stream[key]} != {expected_stream[key]}")

    layout = output / "layout.png"
    fonts = _build_layout(
        layout,
        intent_label=args.intent_label.strip(),
        seed=seed,
        score=score,
        automatic_confirmations=automatic_confirmations,
    )
    video = output / "joyai-sc3-action-carrier-partial.mp4"
    render_command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-framerate",
        "15",
        "-i",
        str(layout),
        "-i",
        str(carrier),
        "-i",
        str(candidate),
        "-filter_complex",
        (
            "[1:v]setpts=PTS-STARTPTS,scale=640:480:flags=lanczos[left];"
            "[2:v]setpts=PTS-STARTPTS,scale=640:480:flags=lanczos[right];"
            "[0:v][left]overlay=0:110:shortest=1[base];"
            "[base][right]overlay=640:110:shortest=1,format=yuv420p[out]"
        ),
        "-map",
        "[out]",
        "-an",
        "-frames:v",
        "81",
        "-r",
        "15",
        "-c:v",
        "libx264",
        "-crf",
        "12",
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
        str(video),
    ]
    _run(render_command, output / "render.log")
    poster = output / "joyai-sc3-action-carrier-partial-poster.jpg"
    poster_command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        "select=eq(n\\,40)",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(poster),
    ]
    _run(poster_command, output / "poster.log")
    output_stream = _probe_video(ffprobe, video)
    if (output_stream["width"], output_stream["height"], output_stream["frames"]) != (
        1280,
        720,
        81,
    ):
        raise RuntimeError("packaged showcase violates the 1280x720, 81-frame contract")
    status = "WORKING" if score.get("human_review_passed") is True else "PARTIAL"
    manifest = {
        "schema_version": "1.0.0",
        "status": status,
        "stage": "joyai_sc3_diagnostic_showcase",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "command": [sys.executable, *sys.argv],
        "git": _git_state(),
        "packages": _package_state(output),
        "source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "tools": {
            "ffmpeg": _tool_version(ffmpeg),
            "ffprobe": _tool_version(ffprobe),
        },
        "intent_label": args.intent_label,
        "selected_seed": seed,
        "score": score,
        "automatic_confirmations": automatic_confirmations,
        "confirmation": confirmation_record,
        "inputs": {
            "run_manifest": {
                "path": str(run_manifest),
                "sha256": sha256_file(run_manifest),
            },
            "carrier": {
                "path": str(carrier),
                "sha256": sha256_file(carrier),
                "stream": carrier_stream,
            },
            "candidate": {
                "path": str(candidate),
                "sha256": observed_hash,
                "stream": candidate_stream,
            },
            "candidate_metadata": {
                "path": str(metadata_path),
                "sha256": sha256_file(metadata_path),
            },
            "evaluation": {
                "path": str(evaluation_path),
                "sha256": sha256_file(evaluation_path),
            },
        },
        "layout": {
            "path": str(layout),
            "sha256": sha256_file(layout),
            "fonts": fonts,
        },
        "outputs": {
            "video": {
                "path": str(video),
                "sha256": sha256_file(video),
                "stream": output_stream,
            },
            "poster": {
                "path": str(poster),
                "sha256": sha256_file(poster),
            },
        },
        "commands": {
            "render": render_command,
            "poster": poster_command,
        },
        "human_review_passed": score.get("human_review_passed"),
        "physical_evidence": False,
        "limitations": [
            "The right panel is a generated visual proposal, not real-robot footage.",
            "Automatic image-space gates do not establish 3-D contact, force, or safety.",
            "The artifact remains PARTIAL until native-resolution human review passes.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": status, "output_dir": str(output), **manifest["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
