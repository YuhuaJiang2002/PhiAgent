#!/usr/bin/env python3
"""Package audited long real-scene videos and task-adaptation evidence for Pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def parse_ffprobe(payload: dict[str, object]) -> dict[str, float | int | str]:
    """Normalize the one-video-stream ffprobe response used by the manifest."""
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise ValueError("ffprobe payload must contain exactly one video stream")
    stream = streams[0]
    fraction = str(stream["avg_frame_rate"])
    numerator, denominator = (int(part) for part in fraction.split("/", maxsplit=1))
    if denominator == 0:
        raise ValueError("video frame-rate denominator cannot be zero")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_frames"]),
        "fps": numerator / denominator,
        "duration_seconds": float(stream["duration"]),
    }


def probe_video(path: Path) -> dict[str, float | int | str]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_ffprobe(json.loads(completed.stdout))


def render_poster(video: Path, destination: Path, timestamp: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(destination),
        ],
        check=True,
    )


def _asset(path: Path, *, role: str, status: str, method: str, limitation: str) -> dict[str, object]:
    return {
        "path": _display_path(path),
        "role": role,
        "status": status,
        "method": method,
        "limitation": limitation,
        "sha256": _sha256(path),
        "video": probe_video(path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--showcase-dir", type=Path, default=Path("demo/showcase"))
    parser.add_argument(
        "--training-evidence-dir",
        type=Path,
        default=Path("outputs/flower-task-adaptation/20260811T171000Z-real-window-rank8-ablation-v2"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    showcase = args.showcase_dir.resolve()
    evidence = args.training_evidence_dir.resolve()
    videos = {
        "shadow_hand_20s": showcase / "five-finger-shadow-arm-background-locked.mp4",
        "flower_source_27s": showcase / "real-flower-arranging.mp4",
        "flower_h3_epl_27s": showcase / "real-flower-arranging-h3-wide-light-shadow-hand-clean-vertical.mp4",
    }
    required = [*videos.values(), evidence / "comparison.mp4", evidence / "evaluation.json", evidence / "storyboard.jpg"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing showcase inputs: {missing}")

    poster_specs = {
        "shadow_hand_20s": ("long-case-shadow-hand-poster.jpg", 10.0),
        "flower_source_27s": ("long-case-flower-source-poster.jpg", 13.0),
        "flower_h3_epl_27s": ("long-case-flower-robot-poster.jpg", 13.0),
    }
    for case_id, (name, timestamp) in poster_specs.items():
        render_poster(videos[case_id], showcase / name, timestamp)

    training_outputs = {
        evidence / "comparison.mp4": showcase / "flower-task-vace-real-window-ablation.mp4",
        evidence / "evaluation.json": showcase / "flower-task-vace-real-window-evaluation.json",
        evidence / "storyboard.jpg": showcase / "flower-task-vace-real-window-poster.jpg",
    }
    for source, destination in training_outputs.items():
        shutil.copy2(source, destination)

    cases = {
        "shadow_hand_20s": _asset(
            videos["shadow_hand_20s"],
            role="long geometric retargeting result",
            status="WORKING",
            method="MediaPipe + Dexpilot + background lock",
            limitation="No manipulated object; not a generative AC-WM result.",
        ),
        "flower_source_27s": _asset(
            videos["flower_source_27s"],
            role="real human observation and evaluation source",
            status="INPUT_ONLY",
            method="unaltered real-scene video",
            limitation="This is not robot execution or generated output.",
        ),
        "flower_h3_epl_27s": _asset(
            videos["flower_h3_epl_27s"],
            role="long real-scene visual robot-transfer negative evidence",
            status="PARTIAL_USER_REJECTED",
            method="H3 + EPL localized rendering, protection and temporal gates",
            limitation="Dense full-timeline review found residual human hands and forearms around frames 135-240 and 405-474, including a large forearm at frame 462.",
        ),
        "flower_vace_real_window": _asset(
            showcase / "flower-task-vace-real-window-ablation.mp4",
            role="matched zero-shot versus task-LoRA critical-window ablation",
            status="PARTIAL",
            method="Wan2.1-VACE-1.3B rank-8 regional LoRA, 96 optimization steps",
            limitation="Seventeen-frame gate failed 0/4 semantic checks; full 27.5-second inference was blocked.",
        ),
    }
    training_evaluation = json.loads((evidence / "evaluation.json").read_text())
    manifest = {
        "schema_version": "1.0.0",
        "evidence_created_at": training_evaluation["created_at"],
        "cases": cases,
        "training_evaluation": {
            "path": _display_path(showcase / "flower-task-vace-real-window-evaluation.json"),
            "sha256": _sha256(showcase / "flower-task-vace-real-window-evaluation.json"),
        },
        "limitations": [
            "Long duration is reported separately from action-condition quality.",
            "Only the OSCAR comparison elsewhere on the page is accepted native AC-WM evidence.",
        ],
    }
    output = showcase / "long-real-scene-cases.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(output), "cases": len(cases)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
