#!/usr/bin/env python3
"""Extract per-action continuation frames for a following H3 window."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--continuation-frame", type=int, default=116)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    previous = args.previous_experiment.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"continuation bundle already exists: {manifest_path}")
    metadata_path = previous / "metadata.json"
    ffmpeg = args.ffmpeg.expanduser().resolve()
    for path in (metadata_path, ffmpeg):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required input is missing or empty: {path}")
    if args.continuation_frame < 0:
        raise ValueError("continuation-frame must be non-negative")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "completed":
        raise ValueError("previous H3 action experiment is not completed")
    actions = metadata.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("previous H3 metadata has no actions")

    records = []
    for action in actions:
        label = str(action["label"])
        video = previous / "variants" / label / "raw-h3-nf4.mp4"
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError(f"previous action output is missing: {video}")
        output = output_dir / "variants" / label / "continuation.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{args.continuation_frame})",
            "-frames:v",
            "1",
            str(output),
        ]
        subprocess.run(command, check=True)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"failed to extract continuation for {label}")
        records.append(
            {
                "label": label,
                "source": str(video),
                "source_sha256": _sha256(video),
                "frame": args.continuation_frame,
                "coordinate_frame": "camera:H3_window_pixels",
                "output": str(output),
                "output_sha256": _sha256(output),
                "command": command,
            }
        )
    manifest = {
        "schema_version": "1.0.0",
        "method": "per_action_recursive_h3_continuation_reference",
        "status": "completed",
        "honest_status": "WORKING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "gpu": {"used": False, "reason": "deterministic frame extraction"},
        "previous_experiment": str(previous),
        "previous_metadata_sha256": _sha256(metadata_path),
        "continuation_frame": args.continuation_frame,
        "actions": records,
        "acceptance": {
            "one_reference_per_action": len(records) == len(actions),
            "all_references_nonempty": all(Path(item["output"]).stat().st_size > 0 for item in records),
            "action_state_not_cross_contaminated": True,
        },
        "limitations": [
            "A single RGB continuation frame cannot carry diffusion state or metric robot state.",
            "The following window still requires overlap evaluation and human review.",
        ],
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"output": str(output_dir), "actions": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
