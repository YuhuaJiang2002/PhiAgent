#!/usr/bin/env python3
"""Audit pixel lineage of wrist-only Cosmos3 DROID SFT sequences."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_droid_view_lora import _decode, _ssim_map  # noqa: E402


WIDTH = 768
HEIGHT = 432
SAMPLED_FUTURE_FRAMES = (1, 24, 48, 72, 96)
LINEAGE_SSIM_MIN = 0.97
WRIST_OVER_EXTERIOR_MARGIN_MIN = 0.05
TILES = {
    "exterior_1": (0, 0),
    "exterior_2": (384, 0),
    "wrist": (0, 216),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unresolved"


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unresolved"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_status": run("status", "--short", "--untracked-files=no"),
    }


def canonical_crop(cv2: Any, frame: Any, view: str) -> Any:
    if view not in TILES:
        raise ValueError(f"unknown DROID camera view: {view}")
    x, y = TILES[view]
    crop = frame[y : y + 216, x : x + 384]
    return cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4)


def mean_ssim(cv2: Any, np: Any, first: Any, second: Any) -> float:
    return float(np.mean(_ssim_map(cv2, np, first, second)))


def lineage_gates(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "condition_to_wrist_ssim": metrics["condition_to_wrist_ssim"]
        >= LINEAGE_SSIM_MIN,
        "derived_frame0_to_wrist_ssim": metrics["derived_frame0_to_wrist_ssim"]
        >= LINEAGE_SSIM_MIN,
        "minimum_future_to_exterior_ssim": metrics["minimum_future_to_exterior_ssim"]
        >= LINEAGE_SSIM_MIN,
        "wrist_over_exterior_margin": metrics["wrist_over_exterior_margin"]
        >= WRIST_OVER_EXTERIOR_MARGIN_MIN,
    }


def main() -> int:
    args = _parser().parse_args()
    contract_path = args.dataset_contract.expanduser().resolve()
    if not contract_path.is_file():
        raise ValueError(f"dataset contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("method") != "cosmos3_nano_droid_wrist_only_to_exterior_i2v_sft_dataset":
        raise ValueError("dataset is not the wrist-only Cosmos3 contract")
    if contract.get("leakage_checks", {}).get("condition_contains_exterior_pixels") is not False:
        raise ValueError("dataset contract does not declare a wrist-only condition")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite wrist lineage audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, *sys.argv]

    import cv2
    import numpy as np

    rows: list[dict[str, Any]] = []
    cached_source_path: Path | None = None
    cached_source = None
    for record in contract.get("records", []):
        source_path = Path(str(record["source_composite"])).expanduser().resolve()
        derived_path = (contract_path.parent / record["target"]).resolve()
        condition_path = (contract_path.parent / record["condition"]).resolve()
        if _sha256(source_path) != record["source_composite_sha256"]:
            raise ValueError(f"source composite hash changed: {source_path}")
        if _sha256(derived_path) != record["target_sha256"]:
            raise ValueError(f"derived target hash changed: {derived_path}")
        if _sha256(condition_path) != record["condition_sha256"]:
            raise ValueError(f"condition hash changed: {condition_path}")
        if cached_source_path != source_path:
            cached_source = _decode(cv2, np, source_path)
            cached_source_path = source_path
        assert cached_source is not None
        derived = _decode(cv2, np, derived_path)
        condition_bgr = cv2.imread(str(condition_path))
        if condition_bgr is None:
            raise ValueError(f"could not decode condition: {condition_path}")
        condition = cv2.cvtColor(condition_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if len(cached_source) != 97 or len(derived) != 97:
            raise ValueError(f"lineage audit requires 97 frames: {record['sample_id']}")
        wrist = canonical_crop(cv2, cached_source[0], "wrist")
        exterior_scores = [
            mean_ssim(cv2, np, condition, canonical_crop(cv2, cached_source[0], view))
            for view in ("exterior_1", "exterior_2")
        ]
        target_view = str(record["target_view"])
        future_scores = [
            mean_ssim(
                cv2,
                np,
                derived[index],
                canonical_crop(cv2, cached_source[index], target_view),
            )
            for index in SAMPLED_FUTURE_FRAMES
        ]
        condition_wrist_ssim = mean_ssim(cv2, np, condition, wrist)
        metrics = {
            "condition_to_wrist_ssim": condition_wrist_ssim,
            "derived_frame0_to_wrist_ssim": mean_ssim(cv2, np, derived[0], wrist),
            "minimum_future_to_exterior_ssim": min(future_scores),
            "mean_future_to_exterior_ssim": float(np.mean(future_scores)),
            "maximum_condition_to_exterior_ssim": max(exterior_scores),
            "wrist_over_exterior_margin": condition_wrist_ssim - max(exterior_scores),
        }
        gates = lineage_gates(metrics)
        rows.append(
            {
                "sample_id": record["sample_id"],
                "source_sample_id": record["source_sample_id"],
                "split": record["split"],
                "target_view": target_view,
                "sampled_future_frames": list(SAMPLED_FUTURE_FRAMES),
                "metrics": metrics,
                "gates": gates,
                "accepted": all(gates.values()),
            }
        )
    if not rows:
        raise ValueError("wrist-only dataset contains no derived records")
    aggregate = {
        "minimum_condition_to_wrist_ssim": min(
            row["metrics"]["condition_to_wrist_ssim"] for row in rows
        ),
        "minimum_derived_frame0_to_wrist_ssim": min(
            row["metrics"]["derived_frame0_to_wrist_ssim"] for row in rows
        ),
        "minimum_future_to_exterior_ssim": min(
            row["metrics"]["minimum_future_to_exterior_ssim"] for row in rows
        ),
        "minimum_wrist_over_exterior_margin": min(
            row["metrics"]["wrist_over_exterior_margin"] for row in rows
        ),
    }
    if any(not math.isfinite(value) for value in aggregate.values()):
        raise ValueError("lineage audit produced a non-finite metric")
    accepted = all(row["accepted"] for row in rows)
    payload = {
        "schema_version": "1.0.0",
        "method": "phiagent_cosmos3_droid_wrist_only_pixel_lineage_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if accepted else "PARTIAL",
        "accepted": accepted,
        "gates": {
            "lineage_ssim_min": LINEAGE_SSIM_MIN,
            "wrist_over_exterior_margin_min": WRIST_OVER_EXTERIOR_MARGIN_MIN,
        },
        "aggregate": aggregate,
        "records": rows,
        "coordinate_frames": {
            "source": "canonical_droid_2x2_composite_pixel_frame",
            "condition": "resized_wrist_camera_pixel_frame",
            "target": "resized_named_exterior_camera_pixel_frame",
            "comparison": "768x432 RGB normalized pixel frame",
        },
        "dataset_contract": str(contract_path),
        "dataset_contract_sha256": _sha256(contract_path),
        "seed": 0,
        "determinism": "no stochastic operations; seed is recorded as zero",
        "command": command,
        "command_shell": shlex.join(command),
        "git": _git_state(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "numpy": _package_version("numpy"),
            "opencv": _package_version("opencv-python", "opencv-python-headless"),
        },
        "limitations": [
            "This verifies pixel lineage and camera-view separation, not model quality.",
            "H.264 encoding makes the lineage comparison similarity-based rather than byte-identical.",
        ],
    }
    _write_json(output, payload)
    (output.parent / "command.txt").write_text(payload["command_shell"] + "\n")
    _write_json(
        output.parent / "config.json",
        {
            "dataset_contract": str(contract_path),
            "dataset_contract_sha256": payload["dataset_contract_sha256"],
            "sampled_future_frames": list(SAMPLED_FUTURE_FRAMES),
            "lineage_ssim_min": LINEAGE_SSIM_MIN,
            "wrist_over_exterior_margin_min": WRIST_OVER_EXTERIOR_MARGIN_MIN,
            "seed": payload["seed"],
        },
    )
    print(json.dumps({"output": str(output), "accepted": accepted, "status": payload["status"]}))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
