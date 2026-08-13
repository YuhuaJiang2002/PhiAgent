#!/usr/bin/env python3
"""Promote one immutable flower-robot candidate into a vertical comparison demo.

This intentionally does not repair or blend candidate frames.  It verifies an
accepted candidate manifest, preserves that candidate for the complete source
timeline, and only performs presentation resizing/stacking for the demo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_ACCEPTANCE_GATES = (
    "background_lock_passed",
    "flower_layer_present",
    "full_clip_decoded",
    "hand_rendered_area_stable",
    "human_review_passed",
    "limb_chain_connected",
    "no_cross_dissolve_passed",
    "temporal_index_exact",
    "zero_full_frame_transition_outliers",
    "zero_person_roi_transition_outliers",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, capture: bool = False) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=capture,
        capture_output=capture,
    )
    return completed.stdout.strip() if capture else ""


def _probe(ffprobe: str, path: Path) -> dict[str, Any]:
    raw = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,"
            "nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    streams = json.loads(raw).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    stream["decoded_frames"] = int(stream.pop("nb_read_frames"))
    return stream


def _route_candidate(manifest: dict[str, Any], candidate: Path) -> dict[str, Any]:
    acceptance = manifest.get("acceptance", {})
    failed = [name for name in REQUIRED_ACCEPTANCE_GATES if acceptance.get(name) is not True]
    recorded_path = Path(manifest.get("outputs", {}).get("video", "")).resolve()
    recorded_hash = manifest.get("outputs", {}).get("video_sha256")
    actual_hash = _sha256(candidate)
    reasons: list[str] = []
    if manifest.get("status") != "accepted":
        reasons.append(f"manifest status is {manifest.get('status')!r}, not 'accepted'")
    if failed:
        reasons.append("failed acceptance gates: " + ", ".join(failed))
    if recorded_path != candidate.resolve():
        reasons.append(f"candidate path differs from accepted immutable path {recorded_path}")
    if recorded_hash != actual_hash:
        reasons.append("candidate SHA-256 differs from accepted immutable hash")
    return {
        "decision": "preserve_candidate_all_frames" if not reasons else "reject_candidate",
        "accepted": not reasons,
        "failed_gates": failed,
        "reasons": reasons,
        "candidate_sha256": actual_hash,
        "recorded_candidate_sha256": recorded_hash,
    }


def _comparison_filter(width: int, row_height: int, *, labels: bool = True) -> str:
    label_texts = ("REAL HUMAN INPUT", "CONFIDENCE-ROUTED ROBOT")
    filters = []
    for index, label in enumerate(label_texts):
        chain = (
            f"[{index}:v]scale={width}:{row_height - 6}:"
            "force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={width}:{row_height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
        if labels:
            chain += (
                ",drawbox=x=0:y=0:w=iw:h=38:color=black@0.70:t=fill,"
                f"drawtext=text='{label}':x=(w-text_w)/2:y=9:"
                "fontsize=22:fontcolor=white"
            )
        filters.append(chain + f"[v{index}]")
    return ";".join(filters) + ";[v0][v1]vstack=inputs=2[out]"


def _ffmpeg_has_filter(ffmpeg: str, name: str) -> bool:
    filters = _run([ffmpeg, "-hide_banner", "-filters"], capture=True)
    return any(
        len(parts) >= 2 and parts[1] == name
        for line in filters.splitlines()
        if (parts := line.split())
    )


def _git_state(project_root: Path) -> dict[str, Any]:
    try:
        return {
            "available": True,
            "head": _run(["git", "rev-parse", "HEAD"], capture=True),
            "status": _run(["git", "status", "--porcelain=v1"], capture=True).splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": str(error)}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--showcase-output", type=Path)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--row-height", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    source = args.source.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    candidate_manifest = args.candidate_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (source, candidate, candidate_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"immutable experiment directory already exists: {output_dir}")
    if args.width <= 0 or args.row_height <= 44:
        raise ValueError("width must be positive and row-height must exceed 44")

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required")

    drawtext_available = _ffmpeg_has_filter(ffmpeg, "drawtext")
    output_dir.mkdir(parents=True)
    logs_dir = output_dir / "logs"
    final_dir = output_dir / "final"
    logs_dir.mkdir()
    final_dir.mkdir()
    command = [sys.executable, *sys.argv]
    config = {
        "schema_version": "1.0.0",
        "method": "EPL confidence routing with immutable full-track promotion",
        "source": str(source),
        "candidate": str(candidate),
        "candidate_manifest": str(candidate_manifest),
        "width": args.width,
        "row_height": args.row_height,
        "seed": args.seed,
        "postprocessing": "presentation-only resize, label, and vertical stack; no candidate repair",
        "labels_embedded": drawtext_available,
    }
    _write_json(output_dir / "config.json", config)
    (output_dir / "command.txt").write_text(" ".join(command) + "\n")
    _write_json(output_dir / "git-state.json", _git_state(project_root))
    _write_json(
        output_dir / "environment.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "ffmpeg": _run([ffmpeg, "-version"], capture=True).splitlines()[0],
            "ffprobe": _run([ffprobe, "-version"], capture=True).splitlines()[0],
            "gpu": {"used": False, "reason": "CPU-only immutable comparison assembly"},
        },
    )

    accepted_manifest = json.loads(candidate_manifest.read_text())
    route = _route_candidate(accepted_manifest, candidate)
    _write_json(output_dir / "route-decision.json", route)
    if not route["accepted"]:
        raise RuntimeError("candidate routing rejected: " + "; ".join(route["reasons"]))

    source_info = _probe(ffprobe, source)
    candidate_info = _probe(ffprobe, candidate)
    alignment_fields = ("width", "height", "r_frame_rate", "decoded_frames")
    alignment = {
        field: source_info[field] == candidate_info[field] for field in alignment_fields
    }
    if not all(alignment.values()):
        raise RuntimeError(
            f"source/candidate alignment mismatch: {source_info} vs {candidate_info}"
        )

    comparison = final_dir / "real-vs-confidence-routed-robot-vertical.mp4"
    filter_graph = _comparison_filter(
        args.width, args.row_height, labels=drawtext_available
    )
    encode_command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-i",
        str(candidate),
        "-filter_complex",
        filter_graph,
        "-map",
        "[out]",
        "-frames:v",
        str(source_info["decoded_frames"]),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "15",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(comparison),
    ]
    _run(encode_command)
    (logs_dir / "encode-command.txt").write_text(" ".join(encode_command) + "\n")

    poster = final_dir / "poster.jpg"
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-ss",
            "13.75",
            "-i",
            str(comparison),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(poster),
        ]
    )
    storyboard = final_dir / "storyboard-12.jpg"
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(comparison),
            "-vf",
            "fps=12/27.5,scale=336:-2,tile=3x4:padding=4:margin=4:color=black",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(storyboard),
        ]
    )

    comparison_info = _probe(ffprobe, comparison)
    automatic_acceptance = {
        "candidate_route_preserves_all_frames": route["decision"]
        == "preserve_candidate_all_frames",
        "source_candidate_alignment_exact": all(alignment.values()),
        "comparison_decoded_full_timeline": comparison_info["decoded_frames"]
        == source_info["decoded_frames"],
        "comparison_dimensions_exact": comparison_info["width"] == args.width
        and comparison_info["height"] == args.row_height * 2,
        "comparison_yuv420p": comparison_info.get("pix_fmt") == "yuv420p",
        "candidate_frames_unmodified_before_presentation": True,
        "no_per_frame_repair": True,
    }
    if not all(automatic_acceptance.values()):
        raise RuntimeError(f"comparison acceptance failed: {automatic_acceptance}")

    manifest = {
        "schema_version": "1.0.0",
        "status": "accepted",
        "honest_status": "WORKING",
        "scope": "full-length synchronized 2D human-versus-robot flower-arranging comparison",
        "method": "confidence-route one accepted immutable track; presentation-only vertical stack",
        "labels_embedded": drawtext_available,
        "route": route,
        "automatic_acceptance": automatic_acceptance,
        "inputs": {
            "source": {
                "path": str(source),
                "sha256": _sha256(source),
                "probe": source_info,
            },
            "candidate": {
                "path": str(candidate),
                "sha256": route["candidate_sha256"],
                "probe": candidate_info,
                "manifest": str(candidate_manifest),
                "manifest_sha256": _sha256(candidate_manifest),
            },
        },
        "output": {
            "comparison": str(comparison),
            "comparison_sha256": _sha256(comparison),
            "probe": comparison_info,
            "poster": str(poster),
            "storyboard": str(storyboard),
        },
        "limitations": [
            "This is a 2D image-space visualization, not calibrated 3D or real-robot execution.",
            "The bouquet is a morphology-locked RGBA object without petal or stem deformation.",
        ],
    }
    _write_json(final_dir / "manifest.json", manifest)
    (logs_dir / "run.log").write_text(
        "route=preserve_candidate_all_frames\n"
        f"source_frames={source_info['decoded_frames']}\n"
        f"candidate_frames={candidate_info['decoded_frames']}\n"
        f"comparison_frames={comparison_info['decoded_frames']}\n"
        "candidate_repair=disabled\n"
    )

    if args.showcase_output is not None:
        showcase = args.showcase_output.expanduser().resolve()
        if showcase.exists():
            raise FileExistsError(f"showcase output already exists: {showcase}")
        showcase.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(comparison, showcase)
        shutil.copy2(poster, showcase.with_name(showcase.stem + "-poster.jpg"))
        showcase_manifest = {
            "schema_version": "1.0.0",
            "source_experiment": str(output_dir),
            "comparison_sha256": _sha256(showcase),
            "route_decision": route["decision"],
            "candidate_sha256": route["candidate_sha256"],
            "layout": ["top: real human input", "bottom: confidence-routed robot"],
            "frames": comparison_info["decoded_frames"],
            "fps": comparison_info["avg_frame_rate"],
            "width": comparison_info["width"],
            "height": comparison_info["height"],
        }
        _write_json(showcase.with_suffix(".json"), showcase_manifest)
        manifest["showcase"] = {
            "video": str(showcase),
            "sha256": showcase_manifest["comparison_sha256"],
        }
        _write_json(final_dir / "manifest.json", manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
