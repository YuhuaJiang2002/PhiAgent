#!/usr/bin/env python3
"""Render fast keyframe probes for skin-constrained residual-arm material."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compose_h3_layered_replacement import (  # noqa: E402
    build_residual_arm_skin_support,
    build_tracked_polygon_alpha,
    build_tracked_robot_arm_material,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--crop", default="430,270,300,150")
    parser.add_argument("--close-width", type=int, default=25)
    parser.add_argument("--close-height", type=int, default=11)
    parser.add_argument("--minimum-area", type=int, default=500)
    parser.add_argument("--dilation", type=int, default=6)
    parser.add_argument("--temporal-radius", type=int, default=2)
    parser.add_argument("--feather-sigma", type=float, default=3.0)
    parser.add_argument("--full-track", action="store_true")
    parser.add_argument("--columns", type=int, default=3)
    return parser


def _read_frame(cv2: Any, capture: Any, index: int) -> Any:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"could not decode frame {index}")
    return frame


def _coherent_flower_core(cv2: Any, np: Any, frame: Any) -> Any:
    hue, saturation, value = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    green = (hue >= 28) & (hue <= 91) & (saturation >= 67) & (value >= 28)
    pink = (hue >= 145) & (hue <= 179) & (saturation >= 115) & (value >= 55)
    candidate = np.logical_or(green, pink)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    result = np.zeros(candidate.shape, dtype=bool)
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= 35:
            result = np.logical_or(result, labels == component)
    return result


def main() -> int:
    args = _parser().parse_args()
    import cv2
    import numpy as np

    payload = json.loads(args.tracks.read_text())
    if payload.get("coordinate_frame") != "camera:H3_output_pixels":
        raise ValueError("tracks must use camera:H3_output_pixels")
    tracks = payload["tracks"]
    indices = [int(value) for value in args.frames.split(",")]
    crop_x, crop_y, crop_width, crop_height = (
        int(value) for value in args.crop.split(",")
    )
    capture = cv2.VideoCapture(str(args.video))
    tiles = []
    for index in indices:
        frame = _read_frame(cv2, capture, index)
        height, width = frame.shape[:2]
        residual_alpha = build_tracked_polygon_alpha(
            cv2,
            np,
            shape=(height, width),
            tracks=tracks,
            frame_index=index,
            feather_sigma=4.0,
        )
        if not args.full_track:
            support = np.zeros((height, width), dtype=bool)
            for neighbor in range(
                max(0, index - args.temporal_radius),
                index + args.temporal_radius + 1,
            ):
                neighbor_frame = _read_frame(cv2, capture, neighbor)
                search = build_tracked_polygon_alpha(
                    cv2,
                    np,
                    shape=(height, width),
                    tracks=tracks,
                    frame_index=neighbor,
                    feather_sigma=0.0,
                )
                support = np.logical_or(
                    support,
                    build_residual_arm_skin_support(
                        cv2,
                        np,
                        frame=neighbor_frame,
                        search_alpha=search,
                        close_width=args.close_width,
                        close_height=args.close_height,
                        minimum_area=args.minimum_area,
                        dilation=args.dilation,
                    ),
                )
            support_alpha = support.astype(np.float32)
            if args.feather_sigma:
                support_alpha = cv2.GaussianBlur(
                    support_alpha, (0, 0), args.feather_sigma
                )
                support_alpha[support] = 1.0
            residual_alpha *= support_alpha
        preserve = _coherent_flower_core(cv2, np, frame)
        residual_alpha[preserve] = 0.0
        material = build_tracked_robot_arm_material(
            cv2,
            np,
            frame=frame,
            tracks=tracks,
            frame_index=index,
            style="silver",
        )
        result = np.rint(
            frame.astype(np.float32) * (1.0 - residual_alpha[..., None])
            + material.astype(np.float32) * residual_alpha[..., None]
        ).astype(np.uint8)
        result[preserve] = frame[preserve]
        tile = result[
            crop_y : crop_y + crop_height,
            crop_x : crop_x + crop_width,
        ]
        tile = cv2.resize(tile, (crop_width * 2, crop_height * 2))
        cv2.putText(
            tile,
            f"frame {index}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    capture.release()
    columns = min(args.columns, len(tiles))
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"could not write probe sheet {args.output}")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
