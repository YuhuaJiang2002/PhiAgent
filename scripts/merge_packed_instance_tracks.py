#!/usr/bin/env python3
"""Merge independently accepted instances from immutable packed track runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--track",
        action="append",
        required=True,
        metavar="RUN_DIR::INSTANCE_ID",
        help="Select one instance from an immutable packed-track run; repeat as needed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-basename", default="merged-instance-tracks-packed.npz")
    parser.add_argument(
        "--coordinate-frame",
        default="camera:source_video_pixels",
        choices=["camera:source_video_pixels"],
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(project_root: Path) -> dict[str, str | None]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args], cwd=project_root, capture_output=True, text=True
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "head": run("rev-parse", "HEAD"),
        "status_porcelain": run("status", "--porcelain"),
    }


def _parse_track(value: str) -> tuple[Path, str]:
    run_text, separator, instance_id = value.partition("::")
    if not separator or not run_text or not instance_id:
        raise ValueError("--track must use RUN_DIR::INSTANCE_ID")
    return Path(run_text).expanduser().resolve(), instance_id


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    if not args.output_basename.endswith(".npz") or Path(args.output_basename).name != args.output_basename:
        raise ValueError("--output-basename must be a plain .npz filename")

    import numpy as np

    selected_masks = []
    instance_ids: list[str] = []
    object_ids: list[int] = []
    common: dict[str, Any] | None = None
    inputs = []
    for raw_track in args.track:
        run_dir, instance_id = _parse_track(raw_track)
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("coordinate_frame") != args.coordinate_frame:
            raise ValueError(f"coordinate-frame mismatch: {run_dir}")
        packed_paths = [path for path in run_dir.glob("*.npz") if path.is_file()]
        if len(packed_paths) != 1:
            raise ValueError(f"{run_dir} must contain exactly one packed NPZ")
        packed_path = packed_paths[0]
        expected_hash = manifest["outputs"]["packed_masks"]["sha256"]
        if _sha256(packed_path) != expected_hash:
            raise ValueError(f"packed-track hash mismatch: {packed_path}")
        with np.load(packed_path, allow_pickle=False) as packed:
            ids = packed["instance_ids"].astype(str).tolist()
            if instance_id not in ids:
                raise KeyError(f"{instance_id} is absent from {packed_path}")
            index = ids.index(instance_id)
            metadata = {
                "source_frame_indices": packed["source_frame_indices"].astype(np.int32),
                "height": int(packed["height"]),
                "width": int(packed["width"]),
                "bitorder": str(packed["bitorder"]),
                "packed_frame_width": int(packed["masks_packed"].shape[2]),
            }
            if common is None:
                common = metadata
            else:
                for key in ("height", "width", "bitorder", "packed_frame_width"):
                    if common[key] != metadata[key]:
                        raise ValueError(f"track metadata mismatch for {key}")
                if not np.array_equal(
                    common["source_frame_indices"], metadata["source_frame_indices"]
                ):
                    raise ValueError("source-frame indices do not match")
            if instance_id in instance_ids:
                raise ValueError(f"duplicate selected instance ID: {instance_id}")
            object_id = int(packed["object_ids"][index])
            if object_id in object_ids:
                raise ValueError(f"duplicate selected object ID: {object_id}")
            instance_ids.append(instance_id)
            object_ids.append(object_id)
            selected_masks.append(packed["masks_packed"][index].copy())
        inputs.append(
            {
                "run_dir": str(run_dir),
                "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
                "packed_masks": {"path": str(packed_path), "sha256": expected_hash},
                "selected_instance_id": instance_id,
                "selected_object_id": object_id,
            }
        )

    assert common is not None
    output.mkdir(parents=True)
    output_path = output / args.output_basename
    np.savez_compressed(
        output_path,
        masks_packed=np.stack(selected_masks, axis=0),
        instance_ids=np.asarray(instance_ids),
        object_ids=np.asarray(object_ids, dtype=np.int32),
        source_frame_indices=common["source_frame_indices"],
        height=np.asarray(common["height"], dtype=np.int32),
        width=np.asarray(common["width"], dtype=np.int32),
        bitorder=np.asarray(common["bitorder"]),
    )
    created_at = datetime.now(timezone.utc).isoformat()
    project_root = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": "1.0.0",
        "created_at": created_at,
        "status": "PARTIAL",
        "method": "select_independently_reviewed_instance_tracks_without_mask_mixing",
        "coordinate_frame": args.coordinate_frame,
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {"numpy": np.__version__},
        "git": _git_state(project_root),
        "inputs": inputs,
        "selection": [
            {"instance_id": instance_id, "object_id": object_id}
            for instance_id, object_id in zip(instance_ids, object_ids)
        ],
        "outputs": {
            "packed_masks": {"path": str(output_path), "sha256": _sha256(output_path)}
        },
        "limitations": [
            "This merge preserves selected masks byte-for-byte and does not make them accepted.",
            "The strict geometry, contact, identity, and semantic evaluator must still pass on the merged run.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output_dir": str(output), "instances": instance_ids}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
