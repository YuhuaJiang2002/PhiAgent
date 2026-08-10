#!/usr/bin/env python3
"""Evaluate a matched real-video flower-contact ablation without hiding human review.

The automatic scores are deliberately proxies.  They measure preservation and
motion alignment, while an explicit storyboard review supplies the semantic
checks that pixel metrics cannot establish (complete human replacement, two
robot hands, and believable hand--stem contact).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REVIEW_FIELDS = (
    "human_residue_absent",
    "two_robot_hands_visible",
    "causal_stem_contact_visible",
    "flowers_identity_preserved",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--control-video", type=Path, required=True)
    parser.add_argument("--edit-mask", type=Path, required=True)
    parser.add_argument("--zero-shot", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--human-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-outside-similarity", type=float, default=0.97)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(cv2: Any, path: Path) -> list[Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video contains no decoded frames: {path}")
    return frames


def _masked_similarity(np: Any, first: Any, second: Any, mask: Any) -> float:
    if not bool(mask.any()):
        raise ValueError("similarity mask must contain at least one pixel")
    delta = np.abs(first.astype(np.float32) - second.astype(np.float32))[mask]
    return math.exp(-float(delta.mean()) / 32.0)


def _masked_change(np: Any, source: Any, candidate: Any, mask: Any) -> float:
    if not bool(mask.any()):
        raise ValueError("change mask must contain at least one pixel")
    delta = np.abs(source.astype(np.float32) - candidate.astype(np.float32))[mask]
    return float(delta.mean()) / 255.0


def _motion_alignment(
    np: Any, control: list[Any], candidate: list[Any], masks: list[Any]
) -> float:
    """Return cosine alignment of temporal magnitude inside the edit region."""

    alignments = []
    for index in range(1, len(control)):
        region = masks[index] | masks[index - 1]
        if not bool(region.any()):
            continue
        control_motion = np.mean(
            np.abs(
                control[index].astype(np.float32)
                - control[index - 1].astype(np.float32)
            ),
            axis=2,
        )[region]
        candidate_motion = np.mean(
            np.abs(
                candidate[index].astype(np.float32)
                - candidate[index - 1].astype(np.float32)
            ),
            axis=2,
        )[region]
        denominator = float(
            np.linalg.norm(control_motion) * np.linalg.norm(candidate_motion)
        )
        alignment = (
            float(np.dot(control_motion, candidate_motion)) / denominator
            if denominator > 1e-8
            else 0.0
        )
        alignments.append(alignment)
    if not alignments:
        raise ValueError("motion evaluation has no non-empty temporal mask")
    return sum(alignments) / len(alignments)


def _validate_review(review: Any) -> dict[str, Any]:
    if not isinstance(review, dict) or not str(review.get("reviewer", "")).strip():
        raise ValueError("human review must be an object with a non-empty reviewer")
    candidates = review.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError("human review must contain a candidates object")
    for name in ("zero_shot", "adapted"):
        row = candidates.get(name)
        if not isinstance(row, dict):
            raise ValueError(f"human review is missing candidate {name}")
        for field in REVIEW_FIELDS:
            if type(row.get(field)) is not bool:  # noqa: E721 - bool, not truthiness
                raise ValueError(f"{name}.{field} must be a JSON boolean")
    return review


def main() -> int:
    args = _parser().parse_args()
    if not 0.0 < args.minimum_outside_similarity <= 1.0:
        raise ValueError("minimum outside similarity must be in (0, 1]")
    paths = {
        "input_video": args.input_video.expanduser().resolve(),
        "control_video": args.control_video.expanduser().resolve(),
        "edit_mask": args.edit_mask.expanduser().resolve(),
        "zero_shot": args.zero_shot.expanduser().resolve(),
        "adapted": args.adapted.expanduser().resolve(),
        "human_review": args.human_review.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{name} does not exist or is empty: {path}")

    import cv2
    import numpy as np

    sequences = {
        name: _decode(cv2, path)
        for name, path in paths.items()
        if name != "human_review"
    }
    frame_counts = {name: len(frames) for name, frames in sequences.items()}
    if len(set(frame_counts.values())) != 1:
        raise RuntimeError(f"videos must have equal decoded frame counts: {frame_counts}")
    shapes = {
        frame.shape
        for frames in sequences.values()
        for frame in frames
    }
    if len(shapes) != 1:
        raise RuntimeError(f"all decoded frames must have one shape: {sorted(shapes)}")
    masks = [np.mean(frame, axis=2) >= 127.0 for frame in sequences["edit_mask"]]
    review = _validate_review(json.loads(paths["human_review"].read_text()))

    source = sequences["input_video"]
    control = sequences["control_video"]
    metrics: dict[str, dict[str, float]] = {
        "outside_edit_similarity_to_input": {},
        "inside_edit_mean_absolute_change": {},
        "control_motion_alignment_inside_edit": {},
    }
    for name in ("zero_shot", "adapted"):
        candidate = sequences[name]
        metrics["outside_edit_similarity_to_input"][name] = sum(
            _masked_similarity(np, src, dst, ~mask)
            for src, dst, mask in zip(source, candidate, masks)
        ) / len(source)
        metrics["inside_edit_mean_absolute_change"][name] = sum(
            _masked_change(np, src, dst, mask)
            for src, dst, mask in zip(source, candidate, masks)
        ) / len(source)
        metrics["control_motion_alignment_inside_edit"][name] = _motion_alignment(
            np, control, candidate, masks
        )
    metrics["adapted_vs_zero"] = {
        "mean_absolute_difference": sum(
            float(
                np.mean(
                    np.abs(
                        first.astype(np.float32) - second.astype(np.float32)
                    )
                )
            )
            for first, second in zip(sequences["zero_shot"], sequences["adapted"])
        )
        / len(source),
        "outside_similarity_delta": (
            metrics["outside_edit_similarity_to_input"]["adapted"]
            - metrics["outside_edit_similarity_to_input"]["zero_shot"]
        ),
        "control_motion_alignment_delta": (
            metrics["control_motion_alignment_inside_edit"]["adapted"]
            - metrics["control_motion_alignment_inside_edit"]["zero_shot"]
        ),
    }

    automatic_gates = {
        "adapted_outside_edit_preserved": (
            metrics["outside_edit_similarity_to_input"]["adapted"]
            >= args.minimum_outside_similarity
        ),
        "adapted_outside_not_regressed": (
            metrics["outside_edit_similarity_to_input"]["adapted"]
            >= metrics["outside_edit_similarity_to_input"]["zero_shot"]
        ),
        "adapted_control_motion_not_regressed": (
            metrics["control_motion_alignment_inside_edit"]["adapted"]
            >= metrics["control_motion_alignment_inside_edit"]["zero_shot"]
        ),
    }
    semantic_gates = {
        field: bool(review["candidates"]["adapted"][field])
        for field in REVIEW_FIELDS
    }
    full_expansion_gate_pass = all(automatic_gates.values()) and all(
        semantic_gates.values()
    )
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "status": "PARTIAL",
        "decision": (
            "ALLOW_FULL_EXPANSION"
            if full_expansion_gate_pass
            else "REJECT_FULL_EXPANSION"
        ),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "coordinate_frame": "camera:source_pixels_resampled_448x256",
        "frame_count": len(source),
        "edit_mask_fraction": {
            "minimum": min(float(mask.mean()) for mask in masks),
            "maximum": max(float(mask.mean()) for mask in masks),
        },
        "metrics": metrics,
        "automatic_gates": automatic_gates,
        "semantic_gates": semantic_gates,
        "full_expansion_gate_pass": full_expansion_gate_pass,
        "human_review": review,
        "limitations": [
            "Automatic preservation and motion scores are proxies, not contact physics proof.",
            "The edit mask is derived from union flower evidence, not persistent stem instances.",
            "This 17-frame critical window cannot establish 27.5-second temporal quality.",
        ],
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
