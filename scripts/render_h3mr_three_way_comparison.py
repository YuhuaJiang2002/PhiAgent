#!/usr/bin/env python3
"""Compose an audited Native/Stage-3/Adaptive-VDA three-way demo video.

The renderer consumes two already-audited, frame-aligned comparison videos:
Native-vs-Adaptive and Stage-3-vs-Adaptive. It reuses their RGB/skeleton and
GT-trajectory panels, then draws one metric table from the supplied summaries.
No model inference or candidate selection is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


FPS = 30.0
FRAME_COUNT = 300
FRAME_SIZE = (1920, 1080)
BACKGROUND = (19, 23, 32)
PANEL_BACKGROUND = (31, 36, 48)
PANEL_BORDER = (74, 82, 99)
TEXT = (238, 240, 245)
MUTED = (165, 173, 190)
GREEN = (77, 213, 151)

# Source rectangles follow the frozen two-way renderer's 1920x1080 layout.
SOURCE_RGB_LEFT = (36, 96, 906, 510)
SOURCE_RGB_RIGHT = (978, 96, 906, 510)
SOURCE_TRAJECTORY_LEFT = (36, 632, 430, 354)
SOURCE_TRAJECTORY_RIGHT = (486, 632, 430, 354)

DESTINATION_RGB = (
    (24, 94, 608, 342),
    (656, 94, 608, 342),
    (1288, 94, 608, 342),
)
DESTINATION_TRAJECTORY = (
    (24, 466, 350, 288),
    (390, 466, 350, 288),
    (756, 466, 350, 288),
)
METRICS_RECT = (1122, 466, 774, 442)

METRIC_SPECS = (
    ("pa_mpjpe_mm", "PA (mm)"),
    ("w_mpjpe_mm", "W (mm)"),
    ("wa_mpjpe_mm", "WA (mm)"),
    ("rte_percent", "RTE (%)"),
    ("accel_m_s2", "Accel"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-comparison", type=Path, required=True)
    parser.add_argument("--stage3-comparison", type=Path, required=True)
    parser.add_argument("--native-summary", type=Path, required=True)
    parser.add_argument("--adaptive-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(image: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    x, y, width, height = rect
    cv2.rectangle(image, (x, y), (x + width, y + height), PANEL_BACKGROUND, -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), PANEL_BORDER, 2)


def crop(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = rect
    result = frame[y : y + height, x : x + width]
    if result.shape[:2] != (height, width):
        raise RuntimeError(f"source frame cannot satisfy crop {rect}")
    return result


def paste_scaled(
    canvas: np.ndarray,
    source: np.ndarray,
    destination: tuple[int, int, int, int],
) -> None:
    x, y, width, height = destination
    resized = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
    canvas[y : y + height, x : x + width] = resized
    cv2.rectangle(canvas, (x, y), (x + width, y + height), PANEL_BORDER, 2)


def read_mean(payload: Mapping[str, Any], path: tuple[str, ...]) -> float:
    value: Any = payload
    for key in path:
        value = value[key]
    return float(value["mean"])


def load_metrics(
    native_summary_path: Path,
    adaptive_summary_path: Path,
) -> dict[str, dict[str, float]]:
    native_summary = json.loads(native_summary_path.read_text(encoding="utf-8"))
    adaptive_summary = json.loads(adaptive_summary_path.read_text(encoding="utf-8"))
    metrics: dict[str, dict[str, float]] = {}
    for key, _ in METRIC_SPECS:
        native = read_mean(native_summary, ("metrics", "native_hawor", key))
        stage3 = read_mean(adaptive_summary, ("aggregate", "stage3_bir", key))
        adaptive = read_mean(
            adaptive_summary,
            ("aggregate", "stage3_bir_plus_adaptive_vda", key),
        )
        metrics[key] = {
            "native_hawor": native,
            "stage3_bir": stage3,
            "stage3_bir_plus_adaptive_vda": adaptive,
            "improvement_vs_native_percent": (native - adaptive) / native * 100.0,
            "improvement_vs_stage3_percent": (stage3 - adaptive) / stage3 * 100.0,
        }
    return metrics


def format_gain(value: float) -> str:
    if abs(value) < 0.005:
        return "~0.00%"
    return f"{value:+.2f}%"


def draw_metrics(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    metrics: Mapping[str, Mapping[str, float]],
) -> None:
    draw_panel(canvas, rect)
    x, y, width, height = rect
    put_text(canvas, "H2O Train20 aggregate (lower is better)", (x + 18, y + 34), 0.58, TEXT, 2)
    put_text(
        canvas,
        "Same 20 sequences / 40 hand tracks / 25,352 shared valid frames",
        (x + 18, y + 60),
        0.38,
        MUTED,
    )
    columns = (x + 18, x + 146, x + 270, x + 394, x + 526, x + 650)
    headings = ("Metric", "Native", "Stage-3", "Adaptive", "vs Native", "vs S3")
    for position, heading in zip(columns, headings, strict=True):
        put_text(canvas, heading, (position, y + 94), 0.40, MUTED, 1)
    cv2.line(canvas, (x + 16, y + 108), (x + width - 16, y + 108), PANEL_BORDER, 1)
    row_y = y + 144
    for key, label in METRIC_SPECS:
        values = metrics[key]
        put_text(canvas, label, (columns[0], row_y), 0.43, TEXT, 1)
        put_text(canvas, f"{values['native_hawor']:.3f}", (columns[1], row_y), 0.43, TEXT, 1)
        put_text(canvas, f"{values['stage3_bir']:.3f}", (columns[2], row_y), 0.43, TEXT, 1)
        put_text(
            canvas,
            f"{values['stage3_bir_plus_adaptive_vda']:.3f}",
            (columns[3], row_y),
            0.43,
            GREEN,
            1,
        )
        put_text(
            canvas,
            format_gain(values["improvement_vs_native_percent"]),
            (columns[4], row_y),
            0.43,
            GREEN,
            2,
        )
        put_text(
            canvas,
            format_gain(values["improvement_vs_stage3_percent"]),
            (columns[5], row_y),
            0.43,
            GREEN,
            2,
        )
        row_y += 54
    put_text(
        canvas,
        "Accel unit: m/s^2. Gains compare Adaptive VDA with each fixed baseline.",
        (x + 18, y + height - 20),
        0.37,
        MUTED,
    )


def validate_capture(capture: cv2.VideoCapture, label: str) -> None:
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {label} video")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if (width, height) != FRAME_SIZE or frames != FRAME_COUNT or abs(fps - FPS) > 1e-3:
        raise RuntimeError(
            f"{label} must be {FRAME_SIZE[0]}x{FRAME_SIZE[1]}, "
            f"{FRAME_COUNT} frames at {FPS} FPS; got {width}x{height}, {frames}, {fps}"
        )


def decode_audit(path: Path) -> dict[str, int | float | bool]:
    capture = cv2.VideoCapture(str(path))
    validate_capture(capture, "output")
    decoded = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame.shape[:2] != (FRAME_SIZE[1], FRAME_SIZE[0]):
            raise RuntimeError("decoded output frame has an unexpected shape")
        decoded += 1
    capture.release()
    if decoded != FRAME_COUNT:
        raise RuntimeError(f"decoded {decoded} output frames, expected {FRAME_COUNT}")
    return {
        "passed": True,
        "decoded_frames": decoded,
        "width": FRAME_SIZE[0],
        "height": FRAME_SIZE[1],
        "fps": FPS,
        "duration_seconds": FRAME_COUNT / FPS,
    }


def render(args: argparse.Namespace) -> dict[str, Any]:
    inputs = {
        "native_comparison": args.native_comparison.expanduser().resolve(),
        "stage3_comparison": args.stage3_comparison.expanduser().resolve(),
        "native_summary": args.native_summary.expanduser().resolve(),
        "adaptive_summary": args.adaptive_summary.expanduser().resolve(),
    }
    for label, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    output = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    poster = output.with_suffix(".jpg")
    existing = [str(path) for path in (output, poster, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite outputs: " + ", ".join(existing))
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(inputs["native_summary"], inputs["adaptive_summary"])
    native_capture = cv2.VideoCapture(str(inputs["native_comparison"]))
    stage3_capture = cv2.VideoCapture(str(inputs["stage3_comparison"]))
    validate_capture(native_capture, "Native")
    validate_capture(stage3_capture, "Stage-3")

    temporary_paths: list[Path] = []
    poster_temporary: Path | None = None
    writer: cv2.VideoWriter | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".three-way-source.", suffix=".mp4", dir=output.parent)
        os.close(descriptor)
        intermediate = Path(name)
        intermediate.unlink()
        temporary_paths.append(intermediate)
        writer = cv2.VideoWriter(
            str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), FPS, FRAME_SIZE
        )
        if not writer.isOpened():
            raise RuntimeError("cannot open the intermediate video writer")
        for frame_index in range(FRAME_COUNT):
            native_ok, native_frame = native_capture.read()
            stage3_ok, stage3_frame = stage3_capture.read()
            if not native_ok or not stage3_ok:
                raise RuntimeError(f"input decoding stopped at frame {frame_index}")
            canvas = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), BACKGROUND, dtype=np.uint8)
            put_text(
                canvas,
                "Native HaWoR vs Stage-3-BIR vs Stage-3-BIR + Adaptive VDA",
                (28, 48),
                0.90,
                TEXT,
                2,
            )
            put_text(
                canvas,
                "subject2_ego__h2__3 | fixed first 10 s | three methods, one timeline",
                (30, 76),
                0.46,
                MUTED,
            )
            rgb_sources = (
                crop(native_frame, SOURCE_RGB_LEFT),
                crop(stage3_frame, SOURCE_RGB_LEFT),
                crop(stage3_frame, SOURCE_RGB_RIGHT),
            )
            for source, destination in zip(rgb_sources, DESTINATION_RGB, strict=True):
                paste_scaled(canvas, source, destination)
            trajectory_sources = (
                crop(native_frame, SOURCE_TRAJECTORY_LEFT),
                crop(stage3_frame, SOURCE_TRAJECTORY_LEFT),
                crop(stage3_frame, SOURCE_TRAJECTORY_RIGHT),
            )
            for source, destination in zip(
                trajectory_sources, DESTINATION_TRAJECTORY, strict=True
            ):
                paste_scaled(canvas, source, destination)
            draw_metrics(canvas, METRICS_RECT, metrics)
            draw_panel(canvas, (24, 778, 1082, 130))
            put_text(
                canvas,
                "Adaptive routing: 1 hard sequence uses frozen V1 dense recovery;",
                (46, 818),
                0.54,
                TEXT,
                1,
            )
            put_text(
                canvas,
                "19 ordinary sequences use V2 (11 applied, 8 exact Stage-3 fallbacks).",
                (46, 850),
                0.54,
                TEXT,
                1,
            )
            put_text(
                canvas,
                "Local hand geometry is protected; Adaptive VDA targets depth and world motion.",
                (46, 884),
                0.48,
                GREEN,
                1,
            )
            put_text(
                canvas,
                "GT appears only in trajectory visualization and final aggregate evaluation;",
                (28, 958),
                0.48,
                MUTED,
                1,
            )
            put_text(
                canvas,
                "Adaptive candidates were generated and SHA-frozen before evaluation.",
                (28, 986),
                0.48,
                MUTED,
                1,
            )
            put_text(
                canvas,
                f"Frame {frame_index + 1}/{FRAME_COUNT}",
                (1710, 1028),
                0.48,
                MUTED,
            )
            writer.write(canvas)
            if frame_index == FRAME_COUNT // 2:
                descriptor, poster_name = tempfile.mkstemp(
                    prefix=f".{poster.stem}.", suffix=".jpg", dir=poster.parent
                )
                os.close(descriptor)
                poster_temporary = Path(poster_name)
                if not cv2.imwrite(
                    str(poster_temporary), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]
                ):
                    raise RuntimeError("cannot write poster")
        writer.release()
        writer = None
        native_capture.release()
        stage3_capture.release()

        descriptor, name = tempfile.mkstemp(
            prefix=f".{output.stem}.", suffix=".mp4", dir=output.parent
        )
        os.close(descriptor)
        encoded = Path(name)
        encoded.unlink()
        temporary_paths.append(encoded)
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(intermediate),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(encoded),
            ],
            check=True,
        )
        os.replace(encoded, output)
        if poster_temporary is None:
            raise RuntimeError("poster frame was not generated")
        os.replace(poster_temporary, poster)
        audit = decode_audit(output)
        result = {
            "schema_version": "1.0.0",
            "name": "Native-HaWoR-vs-Stage-3-BIR-vs-Adaptive-VDA",
            "status": "complete",
            "sequence": "subject2_ego__h2__3",
            "selection": "first 300 frames of the pre-frozen ordinary pilot; not GT-selected",
            "methods": [
                "Native HaWoR",
                "Stage-3-BIR",
                "Stage-3-BIR + Adaptive VDA",
            ],
            "candidate_generation_uses_3d_gt": False,
            "gt_usage": "trajectory visualization and final aggregate evaluation only",
            "metrics": metrics,
            "inputs": {
                label: {"path": str(path), "sha256": sha256(path)}
                for label, path in inputs.items()
            },
            "output": {
                "video": str(output),
                "video_sha256": sha256(output),
                "poster": str(poster),
                "poster_sha256": sha256(poster),
            },
            "decode_audit": audit,
        }
        atomic_json(manifest_path, result)
        return result
    finally:
        if writer is not None:
            writer.release()
        native_capture.release()
        stage3_capture.release()
        if poster_temporary is not None:
            poster_temporary.unlink(missing_ok=True)
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def main() -> int:
    result = render(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
