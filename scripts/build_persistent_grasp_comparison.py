#!/usr/bin/env python3
"""Build a provenance-carrying real/human versus robot comparison video."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def _probe(ffprobe: Path, path: Path) -> dict[str, float | int]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    numerator, denominator = (
        int(value) for value in str(stream["avg_frame_rate"]).split("/", 1)
    )
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
        "fps": numerator / denominator,
        "duration": float(payload["format"]["duration"]),
    }


def comparison_filter(*, panel_width: int, panel_height: int) -> str:
    """Return the deterministic side-by-side layout filter."""

    return (
        f"[0:v]scale={panel_width}:{panel_height}:flags=lanczos,setsar=1[left];"
        f"[1:v]scale={panel_width}:{panel_height}:flags=lanczos,setsar=1[right];"
        "[left][right]hstack=inputs=2[body];[2:v][body]vstack=inputs=2[out]"
    )


def _render_header(
    path: Path,
    *,
    width: int,
    height: int,
    labels: tuple[str, str] = (
        "REAL HUMAN REFERENCE",
        "PHIAGENT ROBOT RESULT",
    ),
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        import cv2
        import numpy as np

        image = np.full((height, width, 3), (22, 14, 8), dtype=np.uint8)
        positioned_labels = (
            (labels[0], width // 4),
            (labels[1], 3 * width // 4),
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.72
        thickness = 2
        for label, center in positioned_labels:
            (text_width, text_height), _ = cv2.getTextSize(
                label, font, scale, thickness
            )
            cv2.putText(
                image,
                label,
                (center - text_width // 2, (height + text_height) // 2 - 2),
                font,
                scale,
                (248, 242, 235),
                thickness,
                cv2.LINE_AA,
            )
        cv2.line(image, (width // 2, 0), (width // 2, height), (89, 74, 57), 1)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"failed to write comparison header: {path}")
        return

    image = Image.new("RGB", (width, height), (8, 14, 22))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    positioned_labels = (
        (labels[0], width // 4),
        (labels[1], 3 * width // 4),
    )
    for label, center in positioned_labels:
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text((center - text_width // 2, 9), label, fill=(235, 242, 248), font=font)
    draw.line((width // 2, 0, width // 2, height), fill=(57, 74, 89), width=1)
    image.save(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--robot-video", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument(
        "--audit-candidate",
        help="candidate name when the audit report compares more than one video",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("ffprobe"))
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--header-height", type=int, default=48)
    parser.add_argument("--poster-frame", type=int, default=572)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--preset", default="medium")
    parser.add_argument(
        "--all-intra",
        action="store_true",
        help="Disable temporal prediction so repaired frames cannot affect frozen neighbours.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if not 0 <= args.crf <= 51:
        raise ValueError("crf must be in [0, 51]")
    output_dir.mkdir(parents=True, exist_ok=False)
    source = args.source_video.expanduser().resolve()
    robot = args.robot_video.expanduser().resolve()
    audit_path = args.audit_report.expanduser().resolve()
    for path in (source, robot, audit_path, args.ffmpeg, args.ffprobe):
        if not path.is_file():
            raise ValueError(f"required input is missing: {path}")
    source_info = _probe(args.ffprobe, source)
    robot_info = _probe(args.ffprobe, robot)
    for name, info in (("source", source_info), ("robot", robot_info)):
        if info["frames"] != args.expected_frames or abs(info["fps"] - args.fps) > 1e-6:
            raise ValueError(f"{name} video does not match the declared timeline")

    audit = json.loads(audit_path.read_text())
    candidates = audit.get("candidates", [])
    if args.audit_candidate is None:
        if len(candidates) != 1:
            raise ValueError(
                "audit report must contain exactly one candidate unless "
                "--audit-candidate is supplied"
            )
        candidate_audit = candidates[0]
    else:
        matches = [
            row for row in candidates if row.get("name") == args.audit_candidate
        ]
        if len(matches) != 1:
            raise ValueError(
                f"audit report has no unique candidate {args.audit_candidate!r}"
            )
        candidate_audit = matches[0]
    summary = candidate_audit["summary"]
    persistent = summary.get("persistent_grasp")
    if not persistent:
        raise ValueError("audit report has no persistent-grasp result")
    failed_persistent_gates = sorted(
        name for name, passed in persistent["gates"].items() if not passed
    )
    failed_image_gates = sorted(
        name for name, passed in summary["gates"].items() if not passed
    )
    failed_adversarial_gates = sorted(
        name
        for name, passed in candidate_audit["adversarial"]["gates"].items()
        if not passed
    )

    header = output_dir / "comparison-header.png"
    video = output_dir / "real-vs-robot-persistent-grasp-27p5s.mp4"
    poster = output_dir / "real-vs-robot-persistent-grasp-poster.jpg"
    _render_header(
        header,
        width=args.panel_width * 2,
        height=args.header_height,
    )
    filter_value = comparison_filter(
        panel_width=args.panel_width,
        panel_height=args.panel_height,
    )
    encode_command = [
        str(args.ffmpeg), "-y", "-v", "error",
        "-i", str(source), "-i", str(robot),
        "-loop", "1", "-framerate", f"{args.fps:.8f}", "-i", str(header),
        "-filter_complex", filter_value,
        "-map", "[out]", "-frames:v", str(args.expected_frames),
        "-r", f"{args.fps:.8f}", "-an", "-c:v", "libx264", "-preset", args.preset,
        "-crf", str(args.crf), "-pix_fmt", "yuv420p",
    ]
    if args.all_intra:
        encode_command.extend(
            ["-x264-params", "keyint=1:min-keyint=1:scenecut=0"]
        )
    encode_command.extend(["-movflags", "+faststart", str(video)])
    subprocess.run(encode_command, check=True)
    poster_command = [
        str(args.ffmpeg), "-y", "-v", "error", "-ss",
        f"{args.poster_frame / args.fps:.8f}", "-i", str(video),
        "-frames:v", "1", "-q:v", "2", str(poster),
    ]
    subprocess.run(poster_command, check=True)
    video_info = _probe(args.ffprobe, video)
    if video_info["frames"] != args.expected_frames:
        raise RuntimeError("comparison output is not the complete timeline")

    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "honest_status": (
            "PARTIAL: the 2-D occlusion-aware persistent-grasp contract "
            + (
                "passes"
                if not failed_persistent_gates
                else "fails gates=" + repr(failed_persistent_gates)
            )
            + "; "
            f"failed image gates={failed_image_gates or 'none'}, failed adversarial "
            f"gates={failed_adversarial_gates or 'none'}; no metric depth or "
            "force-closure claim."
        ),
        "inputs": {
            "source": {"path": _display_path(source), "sha256": _sha256(source), "video": source_info},
            "robot": {"path": _display_path(robot), "sha256": _sha256(robot), "video": robot_info},
            "audit": {"path": _display_path(audit_path), "sha256": _sha256(audit_path)},
        },
        "persistent_grasp": persistent,
        "audit_summary": {
            "candidate": candidate_audit["name"],
            "full_report_sha256": _sha256(audit_path),
            "full_report_wall_seconds": audit["wall_seconds"],
            "candidate_audit_fps": candidate_audit["audit_fps"],
            "image_space_contract_pass": summary["image_space_contract_pass"],
            "gates": summary["gates"],
            "adversarial": candidate_audit["adversarial"],
        },
        "commands": {"encode": encode_command, "poster": poster_command},
        "outputs": {
            "video": {"path": _display_path(video), "sha256": _sha256(video), "video": video_info},
            "poster": {"path": _display_path(poster), "sha256": _sha256(poster)},
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "limitations": [
            "The grasp gate is a camera-frame visual invariant, not 3-D contact evidence.",
            *(
                [
                    "The persistent-grasp contract is not fully satisfied: "
                    + ", ".join(failed_persistent_gates)
                    + f"; visual recall={persistent['visual_grasp_recall']:.6f}."
                ]
                if failed_persistent_gates
                else []
            ),
            *(
                [
                    "Late hand edge energy still violates its frozen lower gate in "
                    f"{summary['sections']['at_or_after_20_seconds']['metrics']['hand_edge_energy_lower_gate']['violation_fraction']:.2%} "
                    "of late frames."
                ]
                if not summary["gates"]["late_hand_edge_energy_lower_gate"]
                else []
            ),
            *(
                [
                    "At least one declared adversarial detector still fails: "
                    + ", ".join(failed_adversarial_gates)
                    + "."
                ]
                if failed_adversarial_gates
                else []
            ),
            "The robot remains a generated visual replacement rather than a verified real-robot execution.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"video": str(video), "poster": str(poster), "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
