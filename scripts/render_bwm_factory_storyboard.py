#!/usr/bin/env python3
"""Render an auditable reference/baseline/candidate factory storyboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_bwm_heldout_pair import _decode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--baseline-videos", type=Path, required=True)
    parser.add_argument("--candidate-videos", type=Path, required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-width", type=int, default=192)
    parser.add_argument("--cell-height", type=int, default=144)
    args = parser.parse_args()

    import cv2
    import numpy as np

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite storyboard: {output}")
    metadata = args.metadata.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    baseline_root = args.baseline_videos.expanduser().resolve()
    candidate_root = args.candidate_videos.expanduser().resolve()
    rows = [json.loads(line) for line in metadata.read_text().splitlines() if line]
    if not rows:
        raise ValueError("metadata contains no samples")
    if args.cell_width <= 0 or args.cell_height <= 0:
        raise ValueError("cell dimensions must be positive")
    sources = ("REFERENCE", "OFFICIAL BWM", args.candidate_label.upper())
    frame_labels = ("history", "mid-future", "endpoint")
    margin_left = 180
    header_height = 58
    columns = len(sources) * len(frame_labels)
    canvas = np.zeros(
        (
            header_height + len(rows) * args.cell_height,
            margin_left + columns * args.cell_width,
            3,
        ),
        dtype=np.uint8,
    )
    for source_index, source_label in enumerate(sources):
        x = margin_left + source_index * len(frame_labels) * args.cell_width
        cv2.putText(
            canvas,
            source_label,
            (x + 8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        for frame_index, frame_label in enumerate(frame_labels):
            cv2.putText(
                canvas,
                frame_label,
                (x + frame_index * args.cell_width + 8, 47),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 210, 255),
                1,
                cv2.LINE_AA,
            )
    for row_index, row in enumerate(rows):
        episode_index = int(row["episode_index"])
        video = row["video"]
        if not isinstance(video, dict):
            raise ValueError("video metadata must be an object")
        start = int(video["start_frame"])
        end = int(video["end_frame"])
        reference = _decode(cv2, dataset_root / str(video["data"]))[start : end + 1]
        baseline = _decode(cv2, baseline_root / f"episode{episode_index}.mp4")
        candidate = _decode(cv2, candidate_root / f"episode{episode_index}.mp4")
        if not (len(reference) == len(baseline) == len(candidate)):
            raise ValueError(f"frame-count mismatch for episode {episode_index}")
        history = int(row["history_frames"])
        indices = (history - 1, (history + len(reference) - 1) // 2, len(reference) - 1)
        y = header_height + row_index * args.cell_height
        cv2.putText(
            canvas,
            f"ep{episode_index} {row.get('task', '')}",
            (8, y + args.cell_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        for source_index, frames in enumerate((reference, baseline, candidate)):
            for frame_index, index in enumerate(indices):
                cell = cv2.resize(
                    frames[index],
                    (args.cell_width, args.cell_height),
                    interpolation=cv2.INTER_AREA,
                )
                x = margin_left + (
                    source_index * len(indices) + frame_index
                ) * args.cell_width
                canvas[y : y + args.cell_height, x : x + args.cell_width] = cell
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"failed to write storyboard: {output}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
