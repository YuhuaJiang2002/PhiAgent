#!/usr/bin/env python3
"""Restore reviewed named flower instances from aligned source pixels.

The operation is deliberately narrow: source RGB is copied only inside SAM2
instance masks, with a bounded exterior feather.  It does not move a flower,
alter robot geometry, or smooth video frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--track-run", type=Path, required=True)
    parser.add_argument("--instance-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="flower-instances-restored.mp4")
    parser.add_argument("--feather-sigma", type=float, default=1.25)
    parser.add_argument("--crf", type=int, default=16)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_video(cv2: Any, path: Path) -> tuple[list[Any], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video is empty: {path}")
    return frames, fps


def _load_masks(np: Any, track_run: Path, requested: list[str]) -> tuple[Any, dict[str, Any]]:
    manifest_path = track_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("coordinate_frame") != "camera:source_video_pixels":
        raise ValueError("track coordinate frame must be camera:source_video_pixels")
    packed_files = list(track_run.glob("*.npz"))
    if len(packed_files) != 1:
        raise ValueError("track run must contain exactly one NPZ")
    packed_path = packed_files[0]
    expected_hash = manifest["outputs"]["packed_masks"]["sha256"]
    if _sha256(packed_path) != expected_hash:
        raise ValueError("track NPZ hash mismatch")
    data = np.load(packed_path)
    ids = data["instance_ids"].astype(str).tolist()
    missing = sorted(set(requested) - set(ids))
    if missing:
        raise KeyError(f"requested instances are missing: {missing}")
    height = int(data["height"])
    width = int(data["width"])
    indices = data["source_frame_indices"].astype(int).tolist()
    flat = np.unpackbits(data["masks_packed"], axis=2, bitorder=str(data["bitorder"]))
    masks = flat[:, :, : height * width].reshape(len(ids), len(indices), height, width)
    selected = np.stack([masks[ids.index(instance_id)] for instance_id in requested]).astype(bool)
    return selected, {
        "manifest": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "packed_masks": packed_path,
        "packed_masks_sha256": expected_hash,
        "coordinate_frame": manifest["coordinate_frame"],
        "source_frame_indices": indices,
        "height": height,
        "width": width,
    }


def _git_state(root: Path) -> dict[str, str | None]:
    values = {}
    for label, args in {
        "head": ["rev-parse", "HEAD"],
        "status_porcelain": ["status", "--porcelain"],
    }.items():
        completed = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        values[label] = completed.stdout.strip() if completed.returncode == 0 else None
    return values


def main() -> int:
    args = _parser().parse_args()
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith(".mp4"):
        raise ValueError("output-name must be one local .mp4 filename")
    if args.feather_sigma < 0 or not 0 <= args.crf <= 51:
        raise ValueError("invalid feather or CRF")
    paths = {
        "source_video": args.source_video.expanduser().resolve(),
        "candidate_video": args.candidate_video.expanduser().resolve(),
        "track_run": args.track_run.expanduser().resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")

    import cv2
    import numpy as np

    source, source_fps = _read_video(cv2, paths["source_video"])
    candidate, candidate_fps = _read_video(cv2, paths["candidate_video"])
    masks, track = _load_masks(np, paths["track_run"], args.instance_id)
    frame_count = len(source)
    if len(candidate) != frame_count or masks.shape[1] != frame_count:
        raise ValueError("source, candidate, and masks must have the same frame count")
    if abs(source_fps - candidate_fps) > 1e-6:
        raise ValueError("source and candidate FPS must match")
    height, width = source[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in source + candidate):
        raise ValueError("all frames must have identical dimensions")
    if (track["height"], track["width"]) != (height, width):
        raise ValueError("track masks do not match video dimensions")
    if track["source_frame_indices"] != list(range(frame_count)):
        raise ValueError("track indices must be contiguous window-local frames")

    output.mkdir(parents=True)
    provenance = output / "provenance/execution-sources"
    provenance.mkdir(parents=True)
    frozen_source = provenance / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), frozen_source)
    output_video = output / args.output_name
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    command = [
        str(Path(ffmpeg).resolve()), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{source_fps:.12g}", "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output_video),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    restored = []
    rows = []
    for frame_index in range(frame_count):
        core = np.any(masks[:, frame_index], axis=0)
        if args.feather_sigma:
            soft = cv2.GaussianBlur(
                core.astype(np.float32), (0, 0), args.feather_sigma
            )
            alpha = np.maximum(core.astype(np.float32), soft)
        else:
            alpha = core.astype(np.float32)
        support = alpha > 1e-6
        composed = candidate[frame_index].copy()
        blended = (
            alpha[..., None] * source[frame_index].astype(np.float32)
            + (1.0 - alpha[..., None]) * candidate[frame_index].astype(np.float32)
        ).round().clip(0, 255).astype(np.uint8)
        composed[support] = blended[support]
        composed[core] = source[frame_index][core]
        process.stdin.write(composed.tobytes())
        restored.append(composed)
        rows.append(
            {
                "local_frame": frame_index,
                "core_fraction": float(np.mean(core)),
                "support_fraction": float(np.mean(support)),
                "core_exact_source_preencode": bool(
                    np.array_equal(composed[core], source[frame_index][core])
                ),
                "outside_support_exact_candidate_preencode": bool(
                    np.array_equal(composed[~support], candidate[frame_index][~support])
                ),
            }
        )
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg encode failed")

    review_indices = np.unique(np.rint(np.linspace(0, frame_count - 1, 24)).astype(int))
    cells = []
    for index in review_indices:
        frame = restored[int(index)].copy()
        core = np.any(masks[:, int(index)], axis=0)
        contours, _ = cv2.findContours(core.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)
        cv2.putText(frame, f"local {index}", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2)
        cells.append(cv2.resize(frame, (448, 256), interpolation=cv2.INTER_AREA))
    sheet = cv2.vconcat([cv2.hconcat(cells[index:index + 4]) for index in range(0, 24, 4)])
    review_path = output / "review-storyboard.jpg"
    cv2.imwrite(str(review_path), sheet)

    created_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": "1.0.0",
        "created_at": created_at,
        "status": "PARTIAL",
        "decision": "READY_FOR_INDEPENDENT_CANDIDATE_TRACKING_AND_CONTACT_GATE",
        "method": "bounded_exact_source_named_flower_instance_restoration",
        "coordinate_frame": track["coordinate_frame"],
        "instance_ids": args.instance_id,
        "feather_sigma": args.feather_sigma,
        "frames": rows,
        "summary": {
            "frame_count": frame_count,
            "core_fraction_mean": float(np.mean([row["core_fraction"] for row in rows])),
            "support_fraction_max": float(np.max([row["support_fraction"] for row in rows])),
            "all_core_exact_source_preencode": all(row["core_exact_source_preencode"] for row in rows),
            "all_outside_support_exact_candidate_preencode": all(row["outside_support_exact_candidate_preencode"] for row in rows),
        },
        "limitations": [
            "This restores reviewed 2-D blossom pixels; it does not prove a 3-D grasp or force contact.",
            "The result remains PARTIAL until flower identities are re-tracked on the encoded candidate and explicit robot-hand contact gates pass."
        ],
    }
    report_path = output / "restoration-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    project_root = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": "1.0.0",
        "created_at": created_at,
        "status": "PARTIAL",
        "command": [sys.executable, *sys.argv],
        "ffmpeg_command": command,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git": _git_state(project_root),
        "inputs": {
            "source_video": {"path": str(paths["source_video"]), "sha256": _sha256(paths["source_video"])},
            "candidate_video": {"path": str(paths["candidate_video"]), "sha256": _sha256(paths["candidate_video"])},
            "track": {key: str(value) if isinstance(value, Path) else value for key, value in track.items()},
        },
        "execution_source": {"path": str(frozen_source), "sha256": _sha256(frozen_source)},
        "outputs": {
            "video": {"path": str(output_video), "sha256": _sha256(output_video)},
            "report": {"path": str(report_path), "sha256": _sha256(report_path)},
            "review": {"path": str(review_path), "sha256": _sha256(review_path)},
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_video), "sha256": _sha256(output_video), "status": "PARTIAL"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
