#!/usr/bin/env python3
"""Render a synchronized real-scene versus visual robot-execution comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _probe(path: Path) -> dict[str, float | int]:
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
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = (int(part) for part in stream["avg_frame_rate"].split("/"))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": numerator / denominator,
        "frames": int(stream["nb_frames"]),
        "duration_seconds": float(stream["duration"]),
    }


def _label_image(
    destination: Path,
    *,
    width: int,
    height: int,
    title: str,
    subtitle: str,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    regular = "/System/Library/Fonts/Supplemental/Arial.ttf"
    image = Image.new("RGBA", (width, height), (18, 20, 22, 232))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 9, height), fill=(92, 238, 170, 255))
    draw.text(
        (26, 8),
        title,
        font=ImageFont.truetype(bold, 25),
        fill=(92, 238, 170, 255),
    )
    draw.text(
        (26, 39),
        subtitle,
        font=ImageFont.truetype(regular, 16),
        fill=(225, 229, 232, 255),
    )
    image.save(destination)


def _footer_image(destination: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    regular = "/System/Library/Fonts/Supplemental/Arial.ttf"
    image = Image.new("RGBA", (1664, 50), (12, 14, 16, 238))
    draw = ImageDraw.Draw(image)
    message = (
        "SYNCHRONIZED VISUAL COMPARISON | REAL SCENE + RENDERED ROBOT EFFECTS | "
        "NO PHYSICAL ROBOT CLAIM"
    )
    draw.text(
        (46, 15),
        message,
        font=ImageFont.truetype(regular, 17),
        fill=(196, 202, 207, 255),
    )
    image.save(destination)


def _git_state() -> dict[str, object]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pour", type=Path, required=True)
    parser.add_argument("--shake", type=Path, required=True)
    parser.add_argument("--handover", type=Path, required=True)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("outputs/acwm-real-scene-execution-comparison"),
    )
    parser.add_argument(
        "--publish-video",
        type=Path,
        default=Path("demo/showcase/acwm-real-scene-vs-rendered-robot-execution-10s.mp4"),
    )
    parser.add_argument(
        "--publish-poster",
        type=Path,
        default=Path("demo/showcase/acwm-real-scene-vs-rendered-robot-execution-poster.jpg"),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser


def main() -> int:
    args = _parser().parse_args()
    sources = {
        "real_scene_input": args.source.expanduser().resolve(),
        "rendered_pour": args.pour.expanduser().resolve(),
        "rendered_shake": args.shake.expanduser().resolve(),
        "rendered_handover": args.handover.expanduser().resolve(),
    }
    for path in sources.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"comparison input is missing: {path}")

    video_info = {name: _probe(path) for name, path in sources.items()}
    for name, info in video_info.items():
        if info["frames"] != 240 or abs(info["fps"] - 24.0) > 1e-6:
            raise ValueError(f"{name} must be exactly 240 frames at 24 FPS")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    experiment = args.experiment_root.expanduser().resolve() / timestamp
    experiment.mkdir(parents=True, exist_ok=False)
    output = experiment / "real-scene-vs-rendered-robot-execution.mp4"
    poster = experiment / "poster.jpg"
    labels_dir = experiment / "labels"
    labels_dir.mkdir()

    config = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "STARTED",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "sources": {name: str(path) for name, path in sources.items()},
        "source_sha256": {name: _sha256(path) for name, path in sources.items()},
        "video_info": video_info,
        "layout": "2x2 synchronized, 832x480 per tile, 24 FPS, ten seconds",
        "claim_boundary": "Rendered visual execution effect; not physical robot footage.",
    }
    _write_json(experiment / "config.json", config)
    _write_json(experiment / "git-state.json", _git_state())
    (experiment / "command.txt").write_text(" ".join(sys.argv) + "\n")

    labels = (
        (
            "REAL-SCENE INPUT",
            "recorded human execution | synchronized reference",
        ),
        (
            "RENDERED ROBOT: POUR",
            "visual execution effect | simulated, not physical footage",
        ),
        (
            "RENDERED ROBOT: SHAKE",
            "visual execution effect | simulated, not physical footage",
        ),
        (
            "RENDERED ROBOT: HANDOVER",
            "visual execution effect | simulated, not physical footage",
        ),
    )
    label_paths = []
    for index, (title, subtitle) in enumerate(labels):
        label_path = labels_dir / f"tile-{index}.png"
        _label_image(label_path, width=832, height=68, title=title, subtitle=subtitle)
        label_paths.append(label_path)
    footer = labels_dir / "footer.png"
    _footer_image(footer)

    filter_graph = (
        "[0:v]scale=832:480:force_original_aspect_ratio=increase,"
        "crop=832:480,setsar=1[b0];[b0][4:v]overlay=0:0:format=auto[v0];"
        "[1:v]scale=832:480:force_original_aspect_ratio=increase,"
        "crop=832:480,setsar=1[b1];[b1][5:v]overlay=0:0:format=auto[v1];"
        "[2:v]scale=832:480:force_original_aspect_ratio=increase,"
        "crop=832:480,setsar=1[b2];[b2][6:v]overlay=0:0:format=auto[v2];"
        "[3:v]scale=832:480:force_original_aspect_ratio=increase,"
        "crop=832:480,setsar=1[b3];[b3][7:v]overlay=0:0:format=auto[v3];"
        "[v0][v1]hstack=inputs=2[top];[v2][v3]hstack=inputs=2[bottom];"
        "[top][bottom]vstack=inputs=2[grid];[grid][8:v]overlay=0:H-h:format=auto[outv]"
    )
    command = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for path in sources.values():
        command.extend(("-i", str(path)))
    for path in (*label_paths, footer):
        command.extend(("-i", str(path)))
    command.extend(
        (
            "-filter_complex",
            filter_graph,
            "-map",
            "[outv]",
            "-an",
            "-r",
            "24",
            "-frames:v",
            "240",
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        )
    )
    _write_json(experiment / "render-command.json", command)
    with (experiment / "render.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"ffmpeg comparison render failed with code {completed.returncode}")

    subprocess.run(
        [
            args.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            "5.0",
            "-i",
            str(output),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(poster),
        ],
        check=True,
    )
    publish_video = args.publish_video.expanduser().resolve()
    publish_poster = args.publish_poster.expanduser().resolve()
    publish_video.parent.mkdir(parents=True, exist_ok=True)
    publish_poster.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, publish_video)
    shutil.copy2(poster, publish_poster)

    package_versions = {}
    for package in ("Pillow",):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed-as-distribution"
    output_info = _probe(output)
    manifest = {
        **config,
        "status": "PARTIAL_VISUALIZATION",
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
            **output_info,
        },
        "published": {
            "video": str(publish_video),
            "poster": str(publish_poster),
        },
        "packages": package_versions,
        "limitations": [
            "The robot panels are rendered visual counterfactuals in a recorded real scene.",
            "They are not output from the one-step BWM smoke checkpoint and are not physical robot execution.",
            "Visual differences do not establish causal action accuracy or task success.",
        ],
    }
    _write_json(experiment / "manifest.json", manifest)
    print(json.dumps(manifest["output"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
