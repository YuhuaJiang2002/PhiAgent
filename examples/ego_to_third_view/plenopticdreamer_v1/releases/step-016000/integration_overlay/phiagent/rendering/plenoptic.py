"""Lightweight contracts for an external Plenoptic stage-1 runtime.

This module intentionally does not import torch or the Cosmos runtime.  It binds a
PhiAgent request to the ``opencv_npz`` camera contract consumed by the separately
installed Plenoptic runner.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


PLENOPTIC_FRAMES = 81
PLENOPTIC_HEIGHT = 432
PLENOPTIC_WIDTH = 768
PLENOPTIC_FPS = 15
PLENOPTIC_CONTEXT_PARALLEL = 4
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _runtime_layout(root: Path):
    path = root / 'tools/plenoptic_paths.py'
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location('_phiagent_runtime_layout', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_runtime_path(root: Path, value: Path) -> Path:
    layout = _runtime_layout(root.resolve())
    return layout.rooted(value).resolve() if layout else value.resolve()


def _relative_file(root: Path, value: Path, label: str) -> tuple[Path, str]:
    root = root.resolve()
    path = resolve_runtime_path(root, value)
    allowed = (root, (root/'../DATASETS').resolve(), (root/'../OUTPUTS').resolve())
    if not any(base in path.parents for base in allowed):
        raise ValueError(f"{label} must be inside the runtime's code, DATASETS or OUTPUTS directories: {path}")
    relative = Path(os.path.relpath(path, root)).as_posix()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    return path, relative


def validate_camera_npz(path: Path, *, frames: int = PLENOPTIC_FRAMES) -> dict[str, Any]:
    """Validate source/target camera arrays without making NumPy a package import."""
    import numpy as np

    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) < {"c2w", "intrinsics"}:
            raise ValueError("camera npz requires c2w and intrinsics arrays")
        c2w = saved["c2w"]
        intrinsics = saved["intrinsics"]
    if c2w.shape != (2, frames, 4, 4):
        raise ValueError(f"c2w must have shape (2,{frames},4,4), got {c2w.shape}")
    if intrinsics.shape != (2, frames, 3, 3):
        raise ValueError(
            f"intrinsics must have shape (2,{frames},3,3), got {intrinsics.shape}"
        )
    if not np.isfinite(c2w).all() or not np.isfinite(intrinsics).all():
        raise ValueError("camera arrays contain non-finite values")
    rotations = c2w[..., :3, :3]
    if not np.allclose(c2w[..., 3, :], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("c2w homogeneous rows are invalid")
    if not np.allclose(
        rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=2e-4
    ) or not np.allclose(np.linalg.det(rotations), 1.0, atol=2e-4):
        raise ValueError("c2w rotations are not proper orthonormal matrices")
    if (
        np.any(intrinsics[..., 0, 0] <= 0)
        or np.any(intrinsics[..., 1, 1] <= 0)
        or not np.allclose(intrinsics[..., 2, :], [0, 0, 1], atol=1e-5)
    ):
        raise ValueError("camera intrinsics are invalid")
    baseline = np.linalg.norm(c2w[1, :, :3, 3] - c2w[0, :, :3, 3], axis=-1)
    return {
        "frames": frames,
        "source_target_baseline_min": float(baseline.min()),
        "source_target_baseline_max": float(baseline.max()),
        "sha256": sha256_file(path),
    }


def build_custom_suite(
    *,
    runtime_root: Path,
    case_id: str,
    source_video: Path,
    original_video: Path,
    camera_npz: Path,
    prompt: str,
    seed: int,
    camera_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one source-only case accepted by ``infer_custom_validation.py``."""
    if not _SAFE_ID.fullmatch(case_id):
        raise ValueError("case_id must use lowercase letters, digits, '_' or '-'")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if camera_provenance.get("calibrated") is not True:
        raise ValueError(
            "pipeline integration requires calibrated cameras; approximate geometry "
            "must remain an explicitly qualitative diagnostic"
        )
    root = runtime_root.resolve()
    source, source_rel = _relative_file(root, source_video, "source video")
    original, original_rel = _relative_file(root, original_video, "original video")
    cameras, camera_rel = _relative_file(root, camera_npz, "camera npz")
    camera_report = validate_camera_npz(cameras)
    scene = {
        "dataset": "custom",
        "split": "val",
        "scene_id": f"phiagent/{case_id}",
        "structurally_complete": True,
        "videos": {"source": source_rel},
        "original_video": original_rel,
        "camera_format": "opencv_npz",
        "camera_file": camera_rel,
        "frames": PLENOPTIC_FRAMES,
        "height": PLENOPTIC_HEIGHT,
        "width": PLENOPTIC_WIDTH,
        "fps": PLENOPTIC_FPS,
        "camera_provenance": dict(camera_provenance, camera_npz=camera_report),
    }
    fingerprints = {
        source_rel: sha256_file(source),
        original_rel: sha256_file(original),
        camera_rel: camera_report["sha256"],
    }
    return {
        "schema": 1,
        "name": "phiagent-calibrated-plenoptic-v1",
        "scope": "Calibrated source-only PhiAgent view generation; no target RGB supplied.",
        "cases": [
            {
                "id": case_id,
                "dataset": "custom",
                "scene_id": scene["scene_id"],
                "source_cameras": ["source"],
                "target_camera": "target",
                "seed": seed,
                "prompt": prompt.strip(),
                "qualitative_only": True,
                "has_target_reference": False,
                "scene": scene,
                "input_sha256": fingerprints,
            }
        ],
    }


def inference_command(
    *,
    python: Path,
    runner: Path,
    checkpoint: Path,
    suite: Path,
    output: Path,
    preflight_only: bool = False,
) -> list[str]:
    command = [
        str(python),
        str(runner),
        "--checkpoint",
        str(checkpoint),
        "--suite",
        str(suite),
        "--output",
        str(output),
    ]
    if preflight_only:
        command.append("--check-only")
    return command


def validate_inference_record(
    record: Mapping[str, Any], *, case_id: str, checkpoint_sha256: str
) -> None:
    required = {
        "status": "generated",
        "case_id": case_id,
        "checkpoint_sha256": checkpoint_sha256,
        "context_parallel_size": PLENOPTIC_CONTEXT_PARALLEL,
        "frames": PLENOPTIC_FRAMES,
        "height": PLENOPTIC_HEIGHT,
        "width": PLENOPTIC_WIDTH,
        "target_reference_used_during_generation": False,
        "qualitative_only": True,
        "has_target_reference": False,
    }
    mismatches = {key: (record.get(key), value) for key, value in required.items()
                  if record.get(key) != value}
    if mismatches:
        raise ValueError(f"Plenoptic inference record violates the adapter contract: {mismatches}")
    visible = str(record.get("cuda_visible_devices", "")).split(",")
    if len(visible) != PLENOPTIC_CONTEXT_PARALLEL or len(set(visible)) != len(visible):
        raise ValueError("inference record does not identify four distinct visible GPUs")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
