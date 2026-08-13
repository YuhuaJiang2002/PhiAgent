#!/usr/bin/env python3
"""Evaluate zero-shot versus RGB-alpha-contact LoRA on a >=20s holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.object_factored_long_video import rgb_to_opencv_hsv  # noqa: E402
from phiagent.rendering.robot_layer_contract import (  # noqa: E402
    canonical_palette_histogram,
    palette_surprisal,
    replacement_mask,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--edit-mask", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--teacher-target", type=Path, required=True)
    parser.add_argument("--zero-shot", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--replacement-threshold", type=float, default=12.0)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(np: Any, ffmpeg: Path, path: Path, width: int, height: int) -> Any:
    completed = subprocess.run(
        [
            str(ffmpeg), "-v", "error", "-i", str(path), "-vf",
            f"scale={width}:{height}:flags=area", "-an", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-",
        ],
        check=True,
        capture_output=True,
    )
    size = width * height * 3
    if not completed.stdout or len(completed.stdout) % size:
        raise ValueError(f"invalid RGB decode byte count for {path}")
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(-1, height, width, 3)


def _similarity(np: Any, first: Any, second: Any, mask: Any) -> float:
    if not np.any(mask):
        raise ValueError("similarity region is empty")
    difference = np.abs(first.astype(np.float32) - second.astype(np.float32))[mask]
    return math.exp(-float(difference.mean()) / 32.0)


def _edge(np: Any, frame: Any) -> Any:
    luma = (
        0.2126 * frame[..., 0].astype(np.float32)
        + 0.7152 * frame[..., 1].astype(np.float32)
        + 0.0722 * frame[..., 2].astype(np.float32)
    )
    result = np.zeros_like(luma)
    result[:, 1:] += np.abs(luma[:, 1:] - luma[:, :-1])
    result[1:, :] += np.abs(luma[1:, :] - luma[:-1, :])
    return np.clip(result, 0, 255)


def _edge_similarity(np: Any, first: Any, second: Any, mask: Any) -> float:
    difference = np.abs(_edge(np, first) - _edge(np, second))[mask]
    return math.exp(-float(difference.mean()) / 32.0)


def _temporal_similarity(np: Any, teacher: Any, candidate: Any, masks: Any) -> float:
    values = []
    for index in range(1, len(teacher)):
        region = masks[index] | masks[index - 1]
        teacher_delta = teacher[index].astype(np.int16) - teacher[index - 1].astype(np.int16)
        candidate_delta = candidate[index].astype(np.int16) - candidate[index - 1].astype(np.int16)
        residual = np.abs(teacher_delta - candidate_delta)[region]
        values.append(math.exp(-float(residual.mean()) / 32.0))
    return sum(values) / len(values)


def _candidate_metrics(
    np: Any,
    *,
    candidate: Any,
    source: Any,
    teacher: Any,
    masks: Any,
    contact_regions: Any,
    palette: Any,
    replacement_threshold: float,
) -> dict[str, float]:
    contact_indices = [index for index, mask in enumerate(contact_regions) if np.any(mask)]
    if not contact_indices:
        raise ValueError("held-out control contains no contact-marked frames")
    inside = sum(
        _similarity(np, teacher[index], candidate[index], masks[index])
        for index in range(len(candidate))
    ) / len(candidate)
    contact = sum(
        _similarity(np, teacher[index], candidate[index], contact_regions[index])
        for index in contact_indices
    ) / len(contact_indices)
    outside = sum(
        _similarity(np, source[index], candidate[index], ~masks[index])
        for index in range(len(candidate))
    ) / len(candidate)
    topology = sum(
        _edge_similarity(np, teacher[index], candidate[index], masks[index])
        for index in range(len(candidate))
    ) / len(candidate)
    palette_score = sum(
        palette_surprisal(np, candidate[index], masks[index], palette)
        for index in range(len(candidate))
    ) / len(candidate)
    chroma = []
    replacement = []
    for index in range(len(candidate)):
        _, saturation, value = rgb_to_opencv_hsv(np, candidate[index])
        chroma.append(float(np.mean(((saturation >= 110) & (value >= 48))[masks[index]])))
        changed = replacement_mask(
            np, candidate[index], source[index], threshold=replacement_threshold
        )
        replacement.append(float(np.mean(changed[masks[index]])))
    return {
        "inside_teacher_similarity": inside,
        "contact_teacher_similarity": contact,
        "outside_source_similarity": outside,
        "topology_edge_teacher_similarity": topology,
        "temporal_teacher_similarity": _temporal_similarity(np, teacher, candidate, masks),
        "canonical_palette_surprisal": palette_score,
        "high_chroma_fraction": sum(chroma) / len(chroma),
        "replacement_coverage": sum(replacement) / len(replacement),
    }


def adapter_gates(
    zero: dict[str, float], adapted: dict[str, float], teacher: dict[str, float], distinctness: float
) -> dict[str, bool]:
    """Return predeclared relative and absolute held-out gates."""

    return {
        "adapted_output_is_distinct": distinctness >= 0.5,
        "inside_teacher_similarity_not_regressed": adapted["inside_teacher_similarity"] >= zero["inside_teacher_similarity"],
        "contact_teacher_similarity_not_regressed": adapted["contact_teacher_similarity"] >= zero["contact_teacher_similarity"],
        "topology_edge_similarity_not_regressed": adapted["topology_edge_teacher_similarity"] >= zero["topology_edge_teacher_similarity"],
        "temporal_similarity_not_regressed": adapted["temporal_teacher_similarity"] >= zero["temporal_teacher_similarity"],
        "palette_distance_to_teacher_not_regressed": abs(adapted["canonical_palette_surprisal"] - teacher["canonical_palette_surprisal"]) <= abs(zero["canonical_palette_surprisal"] - teacher["canonical_palette_surprisal"]),
        "outside_source_similarity_at_least_0p90": adapted["outside_source_similarity"] >= 0.90,
        "inside_teacher_similarity_at_least_0p25": adapted["inside_teacher_similarity"] >= 0.25,
        "contact_teacher_similarity_at_least_0p25": adapted["contact_teacher_similarity"] >= 0.25,
        "topology_edge_similarity_at_least_0p50": adapted["topology_edge_teacher_similarity"] >= 0.50,
        "temporal_similarity_at_least_0p50": adapted["temporal_teacher_similarity"] >= 0.50,
        "replacement_coverage_at_least_0p70": adapted["replacement_coverage"] >= 0.70,
    }


def main() -> int:
    args = _parser().parse_args()
    import numpy as np
    from PIL import Image

    paths = {
        "source": args.source.resolve(),
        "control": args.control.resolve(),
        "edit_mask": args.edit_mask.resolve(),
        "reference": args.reference.resolve(),
        "teacher_target": args.teacher_target.resolve(),
        "zero_shot": args.zero_shot.resolve(),
        "adapted": args.adapted.resolve(),
        "heldout_manifest": args.heldout_manifest.resolve(),
    }
    sequences = {
        name: _decode(np, args.ffmpeg, path, args.width, args.height)
        for name, path in paths.items()
        if name not in {"reference", "heldout_manifest"}
    }
    counts = {name: len(value) for name, value in sequences.items()}
    if set(counts.values()) != {args.frames}:
        raise ValueError(f"all evaluation videos must have {args.frames} frames: {counts}")
    masks = np.mean(sequences["edit_mask"], axis=3) >= 127
    contact_regions = sequences["control"][..., 2] >= 96
    reference = np.asarray(Image.open(paths["reference"]).convert("RGB").resize((args.width, args.height)))
    palette = canonical_palette_histogram(np, reference, np.any(masks, axis=0), bins=8)
    source = sequences["source"]
    teacher = sequences["teacher_target"]
    results = {}
    for name in ("teacher_target", "zero_shot", "adapted"):
        results[name] = _candidate_metrics(
            np,
            candidate=sequences[name],
            source=source,
            teacher=teacher,
            masks=masks,
            contact_regions=contact_regions,
            palette=palette,
            replacement_threshold=args.replacement_threshold,
        )
    distinctness = float(
        np.mean(
            np.abs(
                sequences["adapted"].astype(np.float32)
                - sequences["zero_shot"].astype(np.float32)
            )
        )
    )
    gates = adapter_gates(
        results["zero_shot"], results["adapted"], results["teacher_target"], distinctness
    )
    heldout = json.loads(paths["heldout_manifest"].read_text())
    output_payload = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if all(gates.values()) else "PARTIAL",
        "scope": "same-scene temporally-held-out >=20s adapter proxy only",
        "source_frame_indices": heldout["source_frame_indices"],
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "metrics": results,
        "adapted_zero_mean_absolute_difference": distinctness,
        "gates": gates,
        "all_heldout_proxy_gates_pass": all(gates.values()),
        "limitations": [
            "The holdout is after 20 seconds but comes from the same scene and robot identity.",
            "Teacher similarity is not a perceptual or physical-contact guarantee.",
            "The alpha/contact control encodes 2D support and cannot validate depth or force.",
            "A full 27.5-second model-only rollout remains disallowed unless this short gate passes.",
        ],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
