#!/usr/bin/env python3
"""Fail closed on systematic adjacent near-duplication in DROID composites."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


MEDIAN_DUPLICATE_FRACTION_MAX = 0.15
P90_DUPLICATE_FRACTION_MAX = 0.30
NEAR_DUPLICATE_MAD_MAX = 0.25


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def near_duplicate_fraction(frame_mads: list[float], threshold: float) -> float:
    if not frame_mads:
        raise ValueError("at least one adjacent-frame MAD is required")
    if threshold < 0:
        raise ValueError("near-duplicate threshold must be non-negative")
    return sum(value <= threshold for value in frame_mads) / len(frame_mads)


def _frame_mads(cv2: Any, np: Any, video: Path) -> list[float]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"could not decode video: {video}")
    previous = None
    mads = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        current = frame.astype(np.float32)
        if previous is not None:
            mads.append(float(np.mean(np.abs(current - previous))))
        previous = current
    capture.release()
    if not mads:
        raise ValueError(f"video has fewer than two decoded frames: {video}")
    return mads


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values or not 0 < percentile <= 1:
        raise ValueError("percentile requires non-empty values and 0 < percentile <= 1")
    ordered = sorted(values)
    index = max(0, int((len(ordered) * percentile) + 0.999999999) - 1)
    return ordered[index]


def main() -> int:
    args = _parser().parse_args()
    contract_path = args.dataset_contract.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not contract_path.is_file():
        raise ValueError("dataset contract must exist")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    contract: dict[str, Any] = json.loads(contract_path.read_text())
    records = [row for row in contract.get("records", []) if row.get("split") == "train"]
    if not records:
        raise ValueError("dataset contract has no training records")

    import cv2
    import numpy as np

    examples = []
    for row in records:
        video = (contract_path.parent / row["real_multiview_target_video"]).resolve()
        frame_mads = _frame_mads(cv2, np, video)
        examples.append(
            {
                "sample_id": row["sample_id"],
                "video": str(video),
                "video_sha256": _sha256(video),
                "decoded_frames": len(frame_mads) + 1,
                "adjacent_frame_mad_minimum_0_255": min(frame_mads),
                "adjacent_frame_mad_median_0_255": median(frame_mads),
                "adjacent_near_duplicate_fraction": near_duplicate_fraction(
                    frame_mads, NEAR_DUPLICATE_MAD_MAX
                ),
            }
        )
    fractions = [row["adjacent_near_duplicate_fraction"] for row in examples]
    aggregate = {
        "median_adjacent_near_duplicate_fraction": median(fractions),
        "p90_adjacent_near_duplicate_fraction": percentile_nearest_rank(fractions, 0.90),
        "minimum_adjacent_near_duplicate_fraction": min(fractions),
        "maximum_adjacent_near_duplicate_fraction": max(fractions),
    }
    gates = {
        "median": aggregate["median_adjacent_near_duplicate_fraction"]
        <= MEDIAN_DUPLICATE_FRACTION_MAX,
        "p90": aggregate["p90_adjacent_near_duplicate_fraction"]
        <= P90_DUPLICATE_FRACTION_MAX,
    }
    accepted = all(gates.values())
    payload = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if accepted else "PARTIAL",
        "accepted": accepted,
        "method": "decoded_pixel_mad_adjacent_near_duplicate_audit",
        "dataset_contract": str(contract_path),
        "dataset_contract_sha256": _sha256(contract_path),
        "thresholds": {
            "near_duplicate_mad_max_0_255": NEAR_DUPLICATE_MAD_MAX,
            "median_max": MEDIAN_DUPLICATE_FRACTION_MAX,
            "p90_max": P90_DUPLICATE_FRACTION_MAX,
        },
        "aggregate": aggregate,
        "gates": gates,
        "examples": examples,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, payload)
    print(json.dumps({"output": str(output), "accepted": accepted, **aggregate}))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
