#!/usr/bin/env python3
"""Verify that a video edit preserves all pixels and motion outside allowed masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames or fps <= 0:
        raise ValueError(f"invalid video: {path}")
    return frames, fps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--edited", type=Path, required=True)
    parser.add_argument("--original-hand-mask", type=Path, required=True)
    parser.add_argument("--edited-hand-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.original,
        args.edited,
        args.original_hand_mask,
        args.edited_hand_mask,
    ):
        if not path.is_file():
            raise ValueError(f"input does not exist: {path}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    original, original_fps = _read_video(args.original)
    edited, edited_fps = _read_video(args.edited)
    original_masks, original_mask_fps = _read_video(args.original_hand_mask)
    edited_masks, edited_mask_fps = _read_video(args.edited_hand_mask)
    counts = {
        len(original),
        len(edited),
        len(original_masks),
        len(edited_masks),
    }
    if len(counts) != 1:
        raise ValueError(f"video and mask frame counts differ: {sorted(counts)}")
    dimensions = {
        frame.shape[:2]
        for sequence in (original, edited, original_masks, edited_masks)
        for frame in sequence
    }
    if len(dimensions) != 1:
        raise ValueError(f"video and mask dimensions differ: {dimensions}")

    per_frame_mae: list[float] = []
    changed_fractions: list[float] = []
    transition_errors: list[float] = []
    allowed_fractions: list[float] = []
    previous_original = previous_edited = previous_allowed = None
    for source, candidate, source_mask_frame, candidate_mask_frame in zip(
        original, edited, original_masks, edited_masks
    ):
        source_mask = cv2.cvtColor(source_mask_frame, cv2.COLOR_BGR2GRAY) >= 128
        candidate_mask = cv2.cvtColor(candidate_mask_frame, cv2.COLOR_BGR2GRAY) >= 128
        allowed = cv2.dilate(
            (source_mask | candidate_mask).astype(np.uint8),
            np.ones((17, 17), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        outside = ~allowed
        difference = np.abs(source.astype(np.int16) - candidate.astype(np.int16))
        selected = difference[outside]
        per_frame_mae.append(float(selected.mean()) if selected.size else 0.0)
        changed_fractions.append(
            float(np.mean(np.any(difference > 8, axis=2) & outside))
        )
        allowed_fractions.append(float(np.mean(allowed)))
        if previous_original is not None:
            transition_outside = ~(allowed | previous_allowed)
            source_motion = np.abs(
                source.astype(np.int16) - previous_original.astype(np.int16)
            )
            candidate_motion = np.abs(
                candidate.astype(np.int16) - previous_edited.astype(np.int16)
            )
            transition_difference = np.abs(source_motion - candidate_motion)
            selected_transition = transition_difference[transition_outside]
            transition_errors.append(
                float(selected_transition.mean()) / 255.0
                if selected_transition.size
                else 0.0
            )
        previous_original = source
        previous_edited = candidate
        previous_allowed = allowed

    metrics = {
        "frame_count": len(original),
        "original_fps": original_fps,
        "edited_fps": edited_fps,
        "original_mask_fps": original_mask_fps,
        "edited_mask_fps": edited_mask_fps,
        "mean_allowed_fraction": sum(allowed_fractions) / len(allowed_fractions),
        "mean_outside_mask_mae": sum(per_frame_mae) / len(per_frame_mae),
        "maximum_outside_mask_mae": max(per_frame_mae),
        "mean_outside_changed_fraction": sum(changed_fractions)
        / len(changed_fractions),
        "maximum_outside_changed_fraction": max(changed_fractions),
        "mean_outside_motion_error": sum(transition_errors)
        / max(1, len(transition_errors)),
        "maximum_outside_motion_error": max(transition_errors, default=0.0),
    }
    gates = {
        "frame_count_preserved": len(original) == len(edited),
        "fps_preserved": abs(original_fps - edited_fps) <= 0.01,
        "edit_is_localized": metrics["mean_allowed_fraction"] <= 0.25,
        "background_pixel_mean": metrics["mean_outside_mask_mae"] <= 3.0,
        "background_pixel_worst_frame": metrics["maximum_outside_mask_mae"] <= 5.0,
        "background_changed_fraction": metrics[
            "maximum_outside_changed_fraction"
        ]
        <= 0.05,
        "background_motion_mean": metrics["mean_outside_motion_error"] <= 0.015,
        "background_motion_worst_frame": metrics[
            "maximum_outside_motion_error"
        ]
        <= 0.03,
    }
    payload = {
        "schema_version": "1.0.0",
        "method": "localized edit pixel and motion preservation",
        "original": str(args.original.resolve()),
        "original_sha256": _sha256(args.original),
        "edited": str(args.edited.resolve()),
        "edited_sha256": _sha256(args.edited),
        "metrics": metrics,
        "gates": gates,
        "accepted": all(gates.values()),
        "limitations": [
            "Decoded-pixel comparison includes codec round-trip differences.",
            "Action preservation is measured outside the edited hand union mask.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
