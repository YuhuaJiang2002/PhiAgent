#!/usr/bin/env python3
"""Slice an immutable packed instance-track artifact onto a shorter video window."""

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-track", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-basename", default="sliced-instance-tracks-packed.npz")
    parser.add_argument("--coordinate-frame", default="camera:source_video_pixels")
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

    return {"head": run("rev-parse", "HEAD"), "status_porcelain": run("status", "--porcelain")}


def main() -> int:
    args = _parser().parse_args()
    source = args.input_track.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    if args.start_frame < 0 or args.end_frame <= args.start_frame:
        raise ValueError("frame range must be non-empty and non-negative")
    if not args.output_basename.endswith(".npz") or Path(args.output_basename).name != args.output_basename:
        raise ValueError("--output-basename must be a plain .npz filename")

    import numpy as np

    with np.load(source, allow_pickle=False) as packed:
        required = {
            "masks_packed", "instance_ids", "object_ids", "source_frame_indices",
            "height", "width", "bitorder",
        }
        missing = required - set(packed.files)
        if missing:
            raise ValueError(f"packed track is missing fields: {sorted(missing)}")
        masks = packed["masks_packed"]
        frame_count = int(masks.shape[1])
        if packed["source_frame_indices"].shape != (frame_count,):
            raise ValueError("source_frame_indices do not match packed frame count")
        if args.end_frame > frame_count:
            raise ValueError(f"end frame {args.end_frame} exceeds {frame_count}")
        sliced_masks = masks[:, args.start_frame : args.end_frame].copy()
        instance_ids = packed["instance_ids"].copy()
        object_ids = packed["object_ids"].copy()
        parent_indices = packed["source_frame_indices"][args.start_frame : args.end_frame].copy()
        height = int(packed["height"])
        width = int(packed["width"])
        bitorder = str(packed["bitorder"])

    output.mkdir(parents=True)
    output_path = output / args.output_basename
    np.savez_compressed(
        output_path,
        masks_packed=sliced_masks,
        instance_ids=instance_ids,
        object_ids=object_ids,
        source_frame_indices=np.arange(args.end_frame - args.start_frame, dtype=np.int32),
        height=np.asarray(height, dtype=np.int32),
        width=np.asarray(width, dtype=np.int32),
        bitorder=np.asarray(bitorder),
    )
    project_root = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "method": "byte_exact_temporal_slice_of_real_packed_instance_tracks",
        "coordinate_frame": args.coordinate_frame,
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {"numpy": np.__version__},
        "git": _git_state(project_root),
        "input": {"path": str(source), "sha256": _sha256(source)},
        "slice": {
            "parent_frame_range_half_open": [args.start_frame, args.end_frame],
            "parent_source_frame_indices": parent_indices.astype(int).tolist(),
            "output_source_frame_indices": list(range(args.end_frame - args.start_frame)),
        },
        "instances": [
            {"instance_id": str(instance_id), "object_id": int(object_id)}
            for instance_id, object_id in zip(instance_ids, object_ids)
        ],
        "outputs": {
            "packed_masks": {"path": str(output_path), "sha256": _sha256(output_path)}
        },
        "limitations": [
            "Slicing preserves packed masks exactly but does not independently accept the shorter window.",
            "The shortened video must pass the unchanged strict instance/contact gate.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path), "frames": args.end_frame - args.start_frame}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
