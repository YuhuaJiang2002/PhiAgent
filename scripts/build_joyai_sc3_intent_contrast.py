#!/usr/bin/env python3
"""Package two same-first-frame JoyAI rollouts with distinct user intents."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.acwm.schema import ACWMActionCondition  # noqa: E402
from phiagent.rendering.joyai_video_edit import sha256_file, write_json  # noqa: E402
from scripts.build_joyai_sc3_showcase import (  # noqa: E402
    _centered_x,
    _font,
    _git_state,
    _package_state,
    _probe_video,
    _require_file,
    _run,
    _score_for_selected_seed,
    _tool_version,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lift-run-dir", type=Path, required=True)
    parser.add_argument("--carry-run-dir", type=Path, required=True)
    parser.add_argument("--lift-action", type=Path, required=True)
    parser.add_argument("--carry-action", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    parser.add_argument(
        "--ffprobe", type=Path, default=Path(shutil.which("ffprobe") or "ffprobe")
    )
    return parser


def _selected_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = _require_file(run_dir / "manifest.json", "JoyAI run manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("JoyAI run manifest must contain one JSON object")
    seed, score = _score_for_selected_seed(payload)
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
    candidate_hash = sha256_file(candidate)
    if {
        str(metadata["review"]["sha256"]),
        str(evaluation["candidate_sha256"]),
    } != {candidate_hash}:
        raise ValueError(f"candidate hash evidence disagrees for seed {seed}")
    first_frame = payload.get("preflight", {}).get("first_frame")
    if not isinstance(first_frame, Mapping) or not first_frame.get("sha256"):
        raise ValueError("run manifest lacks first-frame evidence")
    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "manifest": payload,
        "seed": seed,
        "score": score,
        "candidate": candidate,
        "candidate_sha256": candidate_hash,
        "metadata_path": metadata_path,
        "evaluation_path": evaluation_path,
        "evaluation": evaluation,
        "first_frame_sha256": str(first_frame["sha256"]),
    }


def _intent_geometry(
    lift_action: ACWMActionCondition,
    carry_action: ACWMActionCondition,
    lift_evaluation: Mapping[str, Any],
    carry_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    if lift_action.coordinate_frame != carry_action.coordinate_frame:
        raise ValueError("intent actions must use the same camera frame")
    if lift_action.channels != carry_action.channels:
        raise ValueError("intent actions must use the same channels")
    x_index = lift_action.channels.index("object_center_x_px")
    y_index = lift_action.channels.index("object_center_y_px")

    def point(row: tuple[float, ...]) -> tuple[float, float]:
        return float(row[x_index]), float(row[y_index])

    lift_start = point(lift_action.values[0])
    carry_start = point(carry_action.values[0])
    if lift_start != carry_start:
        raise ValueError("intent actions must start from the same object center")
    lift_target = point(lift_action.values[-1])
    carry_target = point(carry_action.values[-1])
    expected_separation = math.dist(lift_target, carry_target)
    if expected_separation < 120:
        raise ValueError(
            f"intent targets are insufficiently separated: {expected_separation:.3f}px"
        )

    def observed(evaluation: Mapping[str, Any]) -> tuple[float, float]:
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("evaluation lacks metrics")
        value = metrics.get("observed_terminal_xy")
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("evaluation lacks an observed terminal object center")
        return float(value[0]), float(value[1])

    lift_observed = observed(lift_evaluation)
    carry_observed = observed(carry_evaluation)
    observed_separation = math.dist(lift_observed, carry_observed)
    if observed_separation < 120:
        raise ValueError(
            f"generated intent endpoints are insufficiently separated: "
            f"{observed_separation:.3f}px"
        )
    return {
        "coordinate_frame": lift_action.coordinate_frame,
        "shared_start_xy": list(lift_start),
        "lift_target_xy": list(lift_target),
        "carry_target_xy": list(carry_target),
        "expected_terminal_separation_px": expected_separation,
        "lift_observed_terminal_xy": list(lift_observed),
        "carry_observed_terminal_xy": list(carry_observed),
        "observed_terminal_separation_px": observed_separation,
        "lift_delta_xy": [
            lift_target[0] - lift_start[0],
            lift_target[1] - lift_start[1],
        ],
        "carry_delta_xy": [
            carry_target[0] - carry_start[0],
            carry_target[1] - carry_start[1],
        ],
    }


def _trajectory_overlay(
    output: Path,
    *,
    action: ACWMActionCondition,
    color: tuple[int, int, int, int],
    label: str,
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("intent-contrast packaging requires Pillow") from exc
    x_index = action.channels.index("object_center_x_px")
    y_index = action.channels.index("object_center_y_px")
    points = [
        (round(row[x_index]), round(row[y_index]))
        for row in action.values
    ]
    canvas = Image.new("RGBA", (640, 480), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.line(points, fill=color, width=5, joint="curve")
    start = points[0]
    target = points[-1]
    draw.ellipse(
        (start[0] - 8, start[1] - 8, start[0] + 8, start[1] + 8),
        outline=(255, 255, 255, 230),
        width=3,
    )
    draw.ellipse(
        (target[0] - 13, target[1] - 13, target[0] + 13, target[1] + 13),
        outline=color,
        width=5,
    )
    font, font_path = _font(18)
    draw.text((target[0] + 16, max(4, target[1] - 12)), label, font=font, fill=color)
    canvas.save(output, format="PNG", optimize=False)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "font": font_path,
        "points": len(points),
        "start_xy": list(start),
        "target_xy": list(target),
    }


def _build_layout(
    output: Path,
    *,
    lift: Mapping[str, Any],
    carry: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("intent-contrast packaging requires Pillow") from exc
    canvas = Image.new("RGB", (1280, 720), (12, 15, 20))
    draw = ImageDraw.Draw(canvas)
    title_font, title_path = _font(30)
    label_font, label_path = _font(23)
    detail_font, detail_path = _font(18)
    warning_font, warning_path = _font(17)
    title = "Same real first frame, two distinct user intents"
    draw.text(
        (_centered_x(draw, title, title_font, 1280), 14),
        title,
        font=title_font,
        fill=(245, 247, 250),
    )
    left = f"INTENT A: LIFT UP AND HOLD  (JoyAI seed {lift['seed']})"
    right = f"INTENT B: CARRY RIGHT AND HOLD  (JoyAI seed {carry['seed']})"
    draw.text(
        (_centered_x(draw, left, label_font, 640), 60),
        left,
        font=label_font,
        fill=(99, 179, 237),
    )
    draw.text(
        (_centered_x(draw, right, label_font, 640, 640), 60),
        right,
        font=label_font,
        fill=(246, 173, 85),
    )
    lift_score = lift["score"]
    carry_score = carry["score"]
    lift_detail = (
        f"A: action {float(lift_score['action_adherence']):.3f} | "
        f"object {float(lift_score['object_interaction']):.3f} | "
        f"target ({geometry['lift_target_xy'][0]:.0f}, {geometry['lift_target_xy'][1]:.0f})"
    )
    carry_detail = (
        f"B: action {float(carry_score['action_adherence']):.3f} | "
        f"object {float(carry_score['object_interaction']):.3f} | "
        f"target ({geometry['carry_target_xy'][0]:.0f}, {geometry['carry_target_xy'][1]:.0f})"
    )
    draw.text(
        (_centered_x(draw, lift_detail, detail_font, 640), 594),
        lift_detail,
        font=detail_font,
        fill=(204, 228, 247),
    )
    draw.text(
        (_centered_x(draw, carry_detail, detail_font, 640, 640), 594),
        carry_detail,
        font=detail_font,
        fill=(253, 230, 200),
    )
    separation = (
        f"Requested target separation: "
        f"{float(geometry['expected_terminal_separation_px']):.0f}px  |  "
        f"Observed generated endpoint separation: "
        f"{float(geometry['observed_terminal_separation_px']):.0f}px"
    )
    draw.text(
        (_centered_x(draw, separation, detail_font, 1280), 632),
        separation,
        font=detail_font,
        fill=(229, 231, 235),
    )
    warning = (
        "PARTIAL - generated visual proposals - human review pending - "
        "not physical execution or contact evidence"
    )
    draw.text(
        (_centered_x(draw, warning, warning_font, 1280), 674),
        warning,
        font=warning_font,
        fill=(252, 129, 129),
    )
    canvas.save(output, format="PNG", optimize=False)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "fonts": {
            "title": title_path,
            "label": label_path,
            "detail": detail_path,
            "warning": warning_path,
        },
    }


def main() -> int:
    args = _parser().parse_args()
    lift_dir = args.lift_run_dir.expanduser().resolve()
    carry_dir = args.carry_run_dir.expanduser().resolve()
    lift = _selected_run(lift_dir)
    carry = _selected_run(carry_dir)
    if lift["first_frame_sha256"] != carry["first_frame_sha256"]:
        raise ValueError("intent runs do not share the same real first frame")
    lift_action_path = _require_file(args.lift_action, "lift action condition")
    carry_action_path = _require_file(args.carry_action, "carry action condition")
    lift_action = ACWMActionCondition.from_json(lift_action_path)
    carry_action = ACWMActionCondition.from_json(carry_action_path)
    geometry = _intent_geometry(
        lift_action,
        carry_action,
        lift["evaluation"],
        carry["evaluation"],
    )
    ffmpeg = _require_file(args.ffmpeg, "ffmpeg")
    ffprobe = _require_file(args.ffprobe, "ffprobe")
    lift_stream = _probe_video(ffprobe, lift["candidate"])
    carry_stream = _probe_video(ffprobe, carry["candidate"])
    for label, stream in (("lift", lift_stream), ("carry", carry_stream)):
        if (
            stream["width"],
            stream["height"],
            stream["fps_numerator"],
            stream["fps_denominator"],
            stream["frames"],
        ) != (640, 480, 15, 1, 81):
            raise ValueError(f"{label} candidate violates the shared video contract")

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite intent-contrast run: {output}")
    output.mkdir(parents=True)
    layout_path = output / "layout.png"
    layout = _build_layout(output=layout_path, lift=lift, carry=carry, geometry=geometry)
    lift_overlay_path = output / "lift-trajectory.png"
    carry_overlay_path = output / "carry-trajectory.png"
    lift_overlay = _trajectory_overlay(
        lift_overlay_path,
        action=lift_action,
        color=(66, 153, 225, 190),
        label="LIFT TARGET",
    )
    carry_overlay = _trajectory_overlay(
        carry_overlay_path,
        action=carry_action,
        color=(237, 137, 54, 190),
        label="RIGHT TARGET",
    )
    video = output / "joyai-sc3-two-intents-partial.mp4"
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
        str(layout_path),
        "-i",
        str(lift["candidate"]),
        "-i",
        str(carry["candidate"]),
        "-loop",
        "1",
        "-framerate",
        "15",
        "-i",
        str(lift_overlay_path),
        "-loop",
        "1",
        "-framerate",
        "15",
        "-i",
        str(carry_overlay_path),
        "-filter_complex",
        (
            "[1:v]setpts=PTS-STARTPTS,scale=640:480:flags=lanczos[lift_video];"
            "[2:v]setpts=PTS-STARTPTS,scale=640:480:flags=lanczos[carry_video];"
            "[3:v]format=rgba[lift_path];[4:v]format=rgba[carry_path];"
            "[lift_video][lift_path]overlay=0:0:shortest=1[lift];"
            "[carry_video][carry_path]overlay=0:0:shortest=1[carry];"
            "[0:v][lift]overlay=0:100:shortest=1[base];"
            "[base][carry]overlay=640:100:shortest=1,format=yuv420p[out]"
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
    poster = output / "joyai-sc3-two-intents-partial-poster.jpg"
    poster_command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        "select=eq(n\\,64)",
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
        raise RuntimeError("intent-contrast output violates the 1280x720 contract")
    manifest = {
        "schema_version": "1.0.0",
        "status": "PARTIAL",
        "stage": "joyai_sc3_same_first_frame_intent_contrast",
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
        "shared_first_frame_sha256": lift["first_frame_sha256"],
        "geometry": geometry,
        "intents": {
            "lift_up": {
                "action": str(lift_action_path),
                "action_sha256": sha256_file(lift_action_path),
                "run_manifest": str(lift["manifest_path"]),
                "run_manifest_sha256": sha256_file(lift["manifest_path"]),
                "seed": lift["seed"],
                "score": lift["score"],
                "candidate": str(lift["candidate"]),
                "candidate_sha256": lift["candidate_sha256"],
                "evaluation": str(lift["evaluation_path"]),
                "evaluation_sha256": sha256_file(lift["evaluation_path"]),
                "stream": lift_stream,
            },
            "carry_right": {
                "action": str(carry_action_path),
                "action_sha256": sha256_file(carry_action_path),
                "run_manifest": str(carry["manifest_path"]),
                "run_manifest_sha256": sha256_file(carry["manifest_path"]),
                "seed": carry["seed"],
                "score": carry["score"],
                "candidate": str(carry["candidate"]),
                "candidate_sha256": carry["candidate_sha256"],
                "evaluation": str(carry["evaluation_path"]),
                "evaluation_sha256": sha256_file(carry["evaluation_path"]),
                "stream": carry_stream,
            },
        },
        "layout": layout,
        "trajectory_overlays": {
            "lift_up": lift_overlay,
            "carry_right": carry_overlay,
        },
        "commands": {
            "render": render_command,
            "poster": poster_command,
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
        "human_review_passed": None,
        "physical_evidence": False,
        "limitations": [
            "Both panels are generated visual proposals, not real-robot footage.",
            "Image-space endpoint separation is not metric 3-D action validation.",
            "Native-resolution human review remains pending.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output), **manifest["outputs"], "geometry": geometry}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
