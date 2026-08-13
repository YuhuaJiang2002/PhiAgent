#!/usr/bin/env python3
"""Build deterministic dense review sheets for wide H3 person coverage."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


HIGH_MOTION = [
    236,
    238,
    237,
    239,
    601,
    603,
    248,
    250,
    139,
    141,
    140,
    142,
    602,
    604,
    37,
    39,
    38,
    40,
    192,
    194,
    247,
    249,
    644,
    646,
]
RISK_WINDOWS = [(110, 136), (235, 255), (384, 405), (475, 492)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _collect(cv2: Any, path: Path, indices: set[int]) -> dict[int, Any]:
    capture = cv2.VideoCapture(str(path))
    frames: dict[int, Any] = {}
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in indices:
            frames[index] = frame
        index += 1
    capture.release()
    if index != 660 or set(frames) != indices:
        raise RuntimeError(f"incomplete decode for {path}: {index} frames")
    return frames


def _annotate(cv2: Any, frame: Any, text: str) -> Any:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (7, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def _grid(cv2: Any, np: Any, tiles: list[Any], columns: int) -> Any:
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(tiles[0]))
        rows.append(cv2.hconcat(row))
    return cv2.vconcat(rows)


def _write_timeline(
    cv2: Any,
    np: Any,
    source: dict[int, Any],
    before: dict[int, Any],
    after: dict[int, Any],
    indices: list[int],
    output: Path,
) -> None:
    tiles = []
    for index in indices:
        layers = []
        for label, frames in (("source", source), ("v74", before), ("wide", after)):
            tile = cv2.resize(frames[index], (240, 138), interpolation=cv2.INTER_AREA)
            layers.append(_annotate(cv2, tile, f"{label} {index}"))
        tiles.append(cv2.vconcat(layers))
    cv2.imwrite(str(output), _grid(cv2, np, tiles, 7), [cv2.IMWRITE_JPEG_QUALITY, 94])


def _write_lower_comparison(
    cv2: Any,
    np: Any,
    before: dict[int, Any],
    after: dict[int, Any],
    indices: list[int],
    output: Path,
) -> None:
    tiles = []
    for index in indices:
        pair = []
        for label, frames in (("v74", before), ("wide", after)):
            crop = frames[index][140:480, 330:800]
            crop = cv2.resize(crop, (235, 170), interpolation=cv2.INTER_AREA)
            pair.append(_annotate(cv2, crop, f"{label} {index}"))
        tiles.append(cv2.hconcat(pair))
    cv2.imwrite(str(output), _grid(cv2, np, tiles, 4), [cv2.IMWRITE_JPEG_QUALITY, 95])


def main() -> int:
    args = _parser().parse_args()
    import cv2
    import numpy as np

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline = list(range(0, 660, 24))
    dense = list(range(0, 660, 12))
    risk = [
        index
        for start, end in RISK_WINDOWS
        for index in range(start, end + 1)
    ]
    all_indices = set(timeline + dense + risk + HIGH_MOTION)
    source = _collect(cv2, args.source.expanduser().resolve(), all_indices)
    before = _collect(cv2, args.before.expanduser().resolve(), all_indices)
    after = _collect(cv2, args.after.expanduser().resolve(), all_indices)

    outputs = {
        "timeline": output_dir / "timeline-1fps-source-v74-wide.jpg",
        "dense_lower": output_dir / "lower-workarea-every12-v74-wide.jpg",
        "risk_windows": output_dir / "risk-windows-v74-wide.jpg",
        "high_motion": output_dir / "high-motion-v74-wide.jpg",
    }
    _write_timeline(cv2, np, source, before, after, timeline, outputs["timeline"])
    _write_lower_comparison(cv2, np, before, after, dense, outputs["dense_lower"])
    _write_lower_comparison(cv2, np, before, after, risk, outputs["risk_windows"])
    _write_lower_comparison(
        cv2, np, before, after, HIGH_MOTION, outputs["high_motion"]
    )
    source_copy = output_dir.parent / "provenance" / "execution-sources" / Path(__file__).name
    source_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), source_copy)
    report = {
        "schema_version": "1.0.0",
        "coordinate_frame": "camera:H3_output_pixels",
        "timeline_indices": timeline,
        "dense_indices": dense,
        "risk_windows_inclusive": RISK_WINDOWS,
        "high_motion_indices": HIGH_MOTION,
        "lower_crop_xyxy": [330, 140, 800, 480],
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    (output_dir / "coverage-review.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
