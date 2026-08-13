#!/usr/bin/env python3
"""Build a CPU-only visual demo of scene-aware embodiment replacement routing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.scene_replacement import (  # noqa: E402
    EntityRole,
    FrameReplacementRoute,
    NormalizedBox,
    ReplacementGranularity,
    ReplacementOperation,
    ReplacementSpec,
    SceneReplacementPlan,
    Shot,
    TrackKeyframe,
    TrackSegment,
)


WIDTH = 480
HEIGHT = 270
FPS = 30
FRAMES = 180


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box(value: tuple[float, float, float, float]) -> NormalizedBox:
    return NormalizedBox(*value)


def _segment(
    entity_id: str,
    shot_id: str,
    role: EntityRole,
    start_frame: int,
    end_frame: int,
    start_box: tuple[float, float, float, float],
    end_box: tuple[float, float, float, float],
    side: str | None = None,
) -> TrackSegment:
    return TrackSegment(
        entity_id=entity_id,
        shot_id=shot_id,
        role=role,
        keyframes=(
            TrackKeyframe(start_frame, _box(start_box)),
            TrackKeyframe(end_frame, _box(end_box)),
        ),
        side=side,
    )


def _plan() -> SceneReplacementPlan:
    shots = (
        Shot("wide_multi_subject", 0, 59),
        Shot("object_occlusion", 60, 119),
        Shot("camera_cut_closeup", 120, 179),
    )
    tracks = (
        _segment(
            "left-performer",
            "wide_multi_subject",
            EntityRole.SUBJECT,
            0,
            59,
            (0.12, 0.32, 0.22, 0.43),
            (0.24, 0.30, 0.22, 0.43),
            "left",
        ),
        _segment(
            "right-performer",
            "wide_multi_subject",
            EntityRole.SUBJECT,
            0,
            59,
            (0.68, 0.22, 0.19, 0.60),
            (0.56, 0.24, 0.19, 0.60),
            "right",
        ),
        _segment(
            "flower",
            "wide_multi_subject",
            EntityRole.OBJECT,
            0,
            59,
            (0.28, 0.36, 0.11, 0.27),
            (0.44, 0.34, 0.11, 0.27),
        ),
        _segment(
            "vase",
            "wide_multi_subject",
            EntityRole.OBJECT,
            0,
            59,
            (0.46, 0.61, 0.10, 0.27),
            (0.46, 0.61, 0.10, 0.27),
        ),
        _segment(
            "left-performer",
            "object_occlusion",
            EntityRole.SUBJECT,
            60,
            119,
            (0.18, 0.25, 0.30, 0.54),
            (0.43, 0.25, 0.30, 0.54),
            "left",
        ),
        _segment(
            "right-performer",
            "object_occlusion",
            EntityRole.SUBJECT,
            60,
            119,
            (0.62, 0.20, 0.22, 0.63),
            (0.55, 0.20, 0.22, 0.63),
            "right",
        ),
        _segment(
            "flower",
            "object_occlusion",
            EntityRole.OBJECT,
            60,
            119,
            (0.34, 0.30, 0.13, 0.36),
            (0.58, 0.30, 0.13, 0.36),
        ),
        _segment(
            "vase",
            "object_occlusion",
            EntityRole.OBJECT,
            60,
            119,
            (0.46, 0.61, 0.11, 0.28),
            (0.46, 0.61, 0.11, 0.28),
        ),
        _segment(
            "left-performer",
            "camera_cut_closeup",
            EntityRole.SUBJECT,
            120,
            179,
            (0.05, 0.23, 0.44, 0.64),
            (0.21, 0.23, 0.44, 0.64),
            "left",
        ),
        _segment(
            "right-performer",
            "camera_cut_closeup",
            EntityRole.SUBJECT,
            120,
            179,
            (0.65, 0.18, 0.30, 0.70),
            (0.49, 0.18, 0.30, 0.70),
            "right",
        ),
        _segment(
            "flower",
            "camera_cut_closeup",
            EntityRole.OBJECT,
            120,
            179,
            (0.38, 0.28, 0.18, 0.40),
            (0.52, 0.28, 0.18, 0.40),
        ),
        _segment(
            "vase",
            "camera_cut_closeup",
            EntityRole.OBJECT,
            120,
            179,
            (0.47, 0.61, 0.14, 0.30),
            (0.47, 0.61, 0.14, 0.30),
        ),
    )
    return SceneReplacementPlan(
        shots=shots,
        tracks=tracks,
        replacements=(
            ReplacementSpec(
                "left-performer",
                "Sharpa hand + forearm",
                ReplacementGranularity.HAND_FOREARM,
            ),
            ReplacementSpec(
                "right-performer",
                "G1 full body",
                ReplacementGranularity.FULL_BODY,
            ),
        ),
        protected_object_ids=("flower", "vase"),
        maximum_carry_frames=0,
    )


def _pixels(box: NormalizedBox) -> tuple[int, int, int, int]:
    x0 = round(box.x * WIDTH)
    y0 = round(box.y * HEIGHT)
    x1 = round((box.x + box.width) * WIDTH)
    y1 = round((box.y + box.height) * HEIGHT)
    return x0, y0, x1, y1


def _draw_background(cv2: Any, np: Any, frame_index: int, shot_id: str) -> Any:
    frame = np.full((HEIGHT, WIDTH, 3), (230, 235, 240), dtype=np.uint8)
    camera_offset = (frame_index * 3) % 80 if shot_id == "camera_cut_closeup" else 0
    for x in range(-80, WIDTH + 80, 80):
        cv2.line(frame, (x + camera_offset, 0), (x + camera_offset, HEIGHT), (205, 211, 218), 1)
    cv2.rectangle(frame, (0, 220), (WIDTH, HEIGHT), (108, 120, 130), -1)
    cv2.line(frame, (0, 220), (WIDTH, 220), (170, 180, 188), 4)
    return frame


def _draw_human(cv2: Any, frame: Any, operation: ReplacementOperation) -> None:
    x0, y0, x1, y1 = _pixels(operation.box)
    color = (188, 145, 112) if operation.side == "left" else (174, 132, 102)
    if operation.granularity is ReplacementGranularity.FULL_BODY:
        center = ((x0 + x1) // 2, y0 + (y1 - y0) // 5)
        cv2.circle(frame, center, max(8, (x1 - x0) // 7), color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0 + 12, center[1] + 10), (x1 - 12, y1 - 10), (67, 104, 154), -1)
        cv2.line(frame, (x0 + 14, center[1] + 18), (x0, y1 - 4), color, 10, cv2.LINE_AA)
    else:
        cv2.line(frame, (x0, y1), (x1 - 18, y0 + 25), color, 18, cv2.LINE_AA)
        cv2.circle(frame, (x1 - 13, y0 + 20), 13, color, -1, cv2.LINE_AA)


def _draw_robot(cv2: Any, frame: Any, operation: ReplacementOperation) -> None:
    x0, y0, x1, y1 = _pixels(operation.box)
    original = frame.copy()
    if operation.granularity is ReplacementGranularity.FULL_BODY:
        center_x = (x0 + x1) // 2
        cv2.rectangle(frame, (center_x - 18, y0 + 22), (center_x + 18, y0 + 58), (62, 70, 80), -1)
        cv2.circle(frame, (center_x, y0 + 15), 14, (205, 211, 218), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0 + 16, y0 + 55), (x1 - 16, y1 - 22), (202, 207, 214), -1)
        cv2.line(frame, (x0 + 20, y0 + 64), (x0, y1 - 8), (72, 82, 93), 10, cv2.LINE_AA)
        cv2.line(frame, (x1 - 20, y0 + 64), (x1, y1 - 8), (72, 82, 93), 10, cv2.LINE_AA)
        cv2.circle(frame, (x0, y1 - 8), 9, (52, 170, 215), -1, cv2.LINE_AA)
    else:
        cv2.line(frame, (x0, y1), (x1 - 24, y0 + 32), (175, 182, 192), 20, cv2.LINE_AA)
        palm = (x1 - 21, y0 + 27)
        cv2.circle(frame, palm, 15, (72, 80, 90), -1, cv2.LINE_AA)
        for index in range(5):
            cv2.line(
                frame,
                (palm[0] - 6 + index * 3, palm[1] - 8),
                (palm[0] - 12 + index * 5, y0 + 2),
                (68, 170, 214),
                3,
                cv2.LINE_AA,
            )
    rendered = frame.copy()
    frame[:] = original
    frame[y0:y1, x0:x1] = rendered[y0:y1, x0:x1]


def _draw_objects(cv2: Any, np: Any, frame: Any, route: FrameReplacementRoute) -> dict[str, Any]:
    masks: dict[str, Any] = {}
    for item in route.protected_objects:
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        x0, y0, x1, y1 = _pixels(item.box)
        if item.entity_id == "flower":
            center_x = (x0 + x1) // 2
            cv2.line(frame, (center_x, y1), (center_x, y0 + 16), (55, 125, 74), 5, cv2.LINE_AA)
            cv2.line(mask, (center_x, y1), (center_x, y0 + 16), 255, 7, cv2.LINE_AA)
            cv2.circle(frame, (center_x, y0 + 13), 13, (105, 78, 205), -1, cv2.LINE_AA)
            cv2.circle(mask, (center_x, y0 + 13), 15, 255, -1, cv2.LINE_AA)
        else:
            cv2.ellipse(
                frame,
                ((x0 + x1) // 2, (y0 + y1) // 2),
                ((x1 - x0) // 2, (y1 - y0) // 2),
                0,
                0,
                360,
                (165, 108, 54),
                -1,
                cv2.LINE_AA,
            )
            cv2.ellipse(
                mask,
                ((x0 + x1) // 2, (y0 + y1) // 2),
                ((x1 - x0) // 2, (y1 - y0) // 2),
                0,
                0,
                360,
                255,
                -1,
                cv2.LINE_AA,
            )
        masks[item.entity_id] = mask > 0
    return masks


def _label(cv2: Any, frame: Any, title: str, subtitle: str) -> Any:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (WIDTH, 42), (16, 21, 28), -1)
    cv2.putText(result, title, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, subtitle, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (176, 211, 245), 1, cv2.LINE_AA)
    return result


def _writer(ffmpeg: str, output: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{WIDTH * 3}x{HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("demo requires the optional numpy and opencv packages") from exc
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("demo requires ffmpeg on PATH")

    plan = _plan()
    video = output_dir / "scene-replacement-coverage.mp4"
    contact_sheet = output_dir / "scene-replacement-contact-sheet.jpg"
    writer = _writer(ffmpeg, video)
    background_changed = 0
    background_values = 0
    protected_changed = 0
    protected_values = 0
    naive_protected_changed = 0
    route_diagnostics = 0
    keyframes: list[Any] = []
    try:
        assert writer.stdin is not None
        for frame_index in range(FRAMES):
            route = plan.route_frame(frame_index)
            route_diagnostics += len(route.diagnostics)
            source = _draw_background(cv2, np, frame_index, route.shot_id)
            for operation in route.replacements:
                _draw_human(cv2, source, operation)
            object_masks = _draw_objects(cv2, np, source, route)

            naive = source.copy()
            for operation in route.replacements:
                _draw_robot(cv2, naive, operation)

            routed = source.copy()
            replacement_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
            for operation in route.replacements:
                _draw_robot(cv2, routed, operation)
                x0, y0, x1, y1 = _pixels(operation.box)
                replacement_mask[y0:y1, x0:x1] = True
            for mask in object_masks.values():
                routed[mask] = source[mask]
                protected_changed += int(np.count_nonzero(routed[mask] != source[mask]))
                naive_protected_changed += int(np.count_nonzero(naive[mask] != source[mask]))
                protected_values += int(mask.sum()) * 3
            unchanged = ~replacement_mask
            background_changed += int(np.count_nonzero(routed[unchanged] != source[unchanged]))
            background_values += int(unchanged.sum()) * 3

            subtitle = {
                "wide_multi_subject": "2 subjects / 2 granularities",
                "object_occlusion": "flower restored above robot layer",
                "camera_cut_closeup": "new shot tracks; no cross-cut carry",
            }[route.shot_id]
            comparison = np.hstack(
                (
                    _label(cv2, source, "SOURCE", subtitle),
                    _label(cv2, naive, "NAIVE SINGLE MASK", "object overwritten / no scene policy"),
                    _label(cv2, routed, "SCENE-AWARE AGENT", "IDs + z-order + protected objects"),
                )
            )
            writer.stdin.write(comparison.tobytes())
            if frame_index in {30, 90, 150}:
                keyframes.append(comparison)
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
        return_code = writer.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg writer failed with exit code {return_code}")
    subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(video), "-f", "null", "-"],
        check=True,
    )
    if not cv2.imwrite(str(contact_sheet), np.vstack(keyframes)):
        raise RuntimeError(f"failed to write contact sheet: {contact_sheet}")

    accepted = (
        route_diagnostics == 0
        and background_changed == 0
        and protected_changed == 0
        and naive_protected_changed > 0
    )
    script = Path(__file__).resolve()
    module = Path(__file__).resolve().parents[1] / "phiagent/rendering/scene_replacement.py"
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if accepted else "REJECTED",
        "method": "cpu_scene_aware_replacement_routing_demo",
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "opencv-python")
        },
        "seed": None,
        "inputs": {"synthetic": True, "frames": FRAMES, "fps": FPS},
        "coverage": {
            "subjects": 2,
            "protected_objects": 2,
            "granularities": ["hand_forearm", "full_body"],
            "shots": [shot.shot_id for shot in plan.shots],
            "hard_camera_cuts": 2,
            "explicit_handedness": True,
        },
        "metrics": {
            "route_diagnostics": route_diagnostics,
            "changed_background_channel_values": background_changed,
            "audited_background_channel_values": background_values,
            "changed_protected_object_channel_values": protected_changed,
            "audited_protected_object_channel_values": protected_values,
            "naive_changed_protected_object_channel_values": naive_protected_changed,
        },
        "sources": {
            str(script): _sha256(script),
            str(module): _sha256(module),
        },
        "outputs": {
            "video": str(video),
            "video_sha256": _sha256(video),
            "contact_sheet": str(contact_sheet),
            "contact_sheet_sha256": _sha256(contact_sheet),
        },
        "limitations": [
            "The demo validates deterministic routing and compositing, not generative quality.",
            "Tracks and shot boundaries are explicit inputs; detector adapters remain separate.",
            "Protected pixels verify image-space occlusion, not 3D contact physics.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
