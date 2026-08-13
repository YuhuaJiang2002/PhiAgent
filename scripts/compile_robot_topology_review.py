#!/usr/bin/env python3
"""Compile range-based semantic review into full-frame topology evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.training.h3_identity_rsi import (  # noqa: E402
    KINEMATIC_TOPOLOGY_FRAME_GATES,
    TOPOLOGY_FRAME_GATES,
    TopologyFrameReview,
    TopologyReviewEvidence,
)


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


def _git_state(root: Path) -> dict[str, object]:
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "available": status.returncode == 0,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def _probe(ffprobe: Path, video: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError("topology review requires exactly one video stream")
    stream = streams[0]
    total_frames = int(stream.get("nb_read_frames", 0))
    if total_frames <= 0:
        raise ValueError("ffprobe did not decode a positive video frame count")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": str(stream["r_frame_rate"]),
        "decoded_frames": total_frames,
    }


def _decoded_frame_sha256(ffmpeg: Path, video: Path, total_frames: int) -> tuple[str, ...]:
    completed = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-f",
            "framemd5",
            "-hash",
            "sha256",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    digests = []
    for line in completed.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6 or not re.fullmatch(r"[0-9a-f]{64}", fields[-1]):
            raise ValueError(f"unexpected decoded-frame checksum line: {line!r}")
        digests.append(fields[-1])
    if len(digests) != total_frames:
        raise ValueError(
            f"ffmpeg hashed {len(digests)} decoded frames, expected {total_frames}"
        )
    return tuple(digests)


def _expand_plan(
    plan: dict[str, object], total_frames: int, decoded_frame_sha256: tuple[str, ...]
) -> tuple[TopologyFrameReview, ...]:
    ranges = plan.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("review plan must contain non-empty ranges")
    expanded: list[TopologyFrameReview | None] = [None] * total_frames
    plan_schema_version = str(plan.get("schema_version", "1.0.0"))
    if plan_schema_version not in {"1.0.0", "2.0.0"}:
        raise ValueError(f"unsupported topology review plan schema {plan_schema_version!r}")
    required_gates = (
        (*TOPOLOGY_FRAME_GATES, *KINEMATIC_TOPOLOGY_FRAME_GATES)
        if plan_schema_version == "2.0.0"
        else TOPOLOGY_FRAME_GATES
    )
    for entry in ranges:
        if not isinstance(entry, dict):
            raise ValueError("every topology range must be a JSON object")
        start, end = int(entry["start_frame"]), int(entry["end_frame"])
        if start < 0 or end < start or end >= total_frames:
            raise ValueError(f"invalid topology range [{start}, {end}]")
        missing = [name for name in required_gates if name not in entry]
        if missing:
            raise ValueError(f"topology range [{start}, {end}] is missing gates: {missing}")
        non_boolean = [name for name in required_gates if not isinstance(entry[name], bool)]
        if non_boolean:
            raise ValueError(
                f"topology range [{start}, {end}] gates must be JSON booleans: {non_boolean}"
            )
        confidence = float(entry["confidence"])
        note = str(entry.get("note", ""))
        values = {name: entry[name] for name in required_gates}
        for frame_index in range(start, end + 1):
            if expanded[frame_index] is not None:
                raise ValueError(f"overlapping topology review at frame {frame_index}")
            expanded[frame_index] = TopologyFrameReview(
                frame_index=frame_index,
                confidence=confidence,
                note=note,
                decoded_frame_sha256=decoded_frame_sha256[frame_index],
                **values,
            )
    missing_frames = [index for index, review in enumerate(expanded) if review is None]
    if missing_frames:
        raise ValueError(f"topology review does not cover frames: {missing_frames[:20]}")
    return tuple(review for review in expanded if review is not None)


def _build_demo(
    cv2: Any,
    np: Any,
    input_video: Path,
    output_video: Path,
    reviews: tuple[TopologyFrameReview, ...],
    minimum_confidence: float,
) -> None:
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode topology demo source: {input_video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    temporary = output_video.with_suffix(".temporary.mp4")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open topology demo writer: {temporary}")
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        review = reviews[frame_index]
        passed = review.passed(minimum_confidence)
        failures = list(review.failed_gates())
        if review.confidence < minimum_confidence:
            failures.append("review_confidence")
        color = (55, 185, 55) if passed else (45, 45, 230)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 62), color, thickness=-1)
        frame = cv2.addWeighted(overlay, 0.56, frame, 0.44, 0)
        headline = f"FRAME {frame_index:03d}  {'TOPOLOGY PASS' if passed else 'HARD REJECT'}"
        detail = "all semantic gates" if passed else ", ".join(failures[:3])
        cv2.putText(
            frame,
            headline,
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            detail,
            (12, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_index += 1
    capture.release()
    writer.release()
    if frame_index != len(reviews):
        raise RuntimeError(f"topology demo decoded {frame_index} frames, expected {len(reviews)}")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(temporary),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_video),
        ],
        check=True,
    )
    temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, default=Path("/opt/homebrew/bin/ffprobe"))
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--minimum-confidence", type=float, default=0.95)
    parser.add_argument("--build-demo", action="store_true")
    parser.add_argument("--allow-rejected", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    video = args.video.expanduser().resolve()
    plan_path = args.plan.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"topology output directory already exists: {output_dir}")
    for path in (video, plan_path, args.ffprobe, args.ffmpeg):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required topology input is missing or empty: {path}")
    if not 0.0 <= args.minimum_confidence <= 1.0:
        raise ValueError("minimum confidence must lie in [0, 1]")
    output_dir.mkdir(parents=True)
    plan = json.loads(plan_path.read_text())
    if not isinstance(plan, dict):
        raise ValueError("topology review plan must be a JSON object")
    probe = _probe(args.ffprobe, video)
    decoded_frame_sha256 = _decoded_frame_sha256(
        args.ffmpeg, video, int(probe["decoded_frames"])
    )
    reviews = _expand_plan(plan, int(probe["decoded_frames"]), decoded_frame_sha256)
    evidence = TopologyReviewEvidence(
        video_sha256=_sha256(video),
        total_frames=int(probe["decoded_frames"]),
        reviewer=str(plan["reviewer"]),
        review_method=str(plan["review_method"]),
        frames=reviews,
    )
    passing_fraction = evidence.passing_fraction(args.minimum_confidence)
    failed_frames = evidence.failed_frames(args.minimum_confidence)
    evidence_path = output_dir / "topology-evidence.json"
    _write_json(
        evidence_path,
        {
            "schema_version": evidence.schema_version,
            "video_sha256": evidence.video_sha256,
            "total_frames": evidence.total_frames,
            "reviewer": evidence.reviewer,
            "review_method": evidence.review_method,
            "frames": [
                {
                    "frame_index": frame.frame_index,
                    "decoded_frame_sha256": frame.decoded_frame_sha256,
                    **{name: getattr(frame, name) for name in TOPOLOGY_FRAME_GATES},
                    **{
                        name: getattr(frame, name)
                        for name in KINEMATIC_TOPOLOGY_FRAME_GATES
                    },
                    "confidence": frame.confidence,
                    "note": frame.note,
                }
                for frame in evidence.frames
            ],
        },
    )
    demo_path = output_dir / "topology-review.mp4"
    if args.build_demo:
        import cv2
        import numpy as np

        _build_demo(cv2, np, video, demo_path, reviews, args.minimum_confidence)
    accepted = (
        evidence.coverage_complete()
        and evidence.decoded_frame_digests_complete()
        and evidence.kinematic_detail_complete()
        and passing_fraction == 1.0
    )
    packages = {}
    for name in ("numpy", "opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "schema_version": "1.0.0",
        "method": "full_frame_semantic_robot_topology_review",
        "status": "accepted" if accepted else "rejected",
        "honest_status": "WORKING" if accepted else "PARTIAL",
        "evolution_decision": "ACCEPT" if accepted else "REGENERATE_WORLD_MODEL_CANDIDATE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "seed": None,
        "deterministic": True,
        "git": _git_state(root),
        "packages": packages,
        "inputs": {
            "video": {"path": str(video), "sha256": evidence.video_sha256},
            "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        },
        "probe": probe,
        "contract": {
            "required_coverage": 1.0,
            "required_decoded_frame_digest_coverage": 1.0,
            "required_kinematic_detail_coverage": 1.0,
            "required_passing_fraction": 1.0,
            "minimum_confidence": args.minimum_confidence,
            "gates": list((*TOPOLOGY_FRAME_GATES, *KINEMATIC_TOPOLOGY_FRAME_GATES)),
            "aggregation": "hard_conjunction_over_every_decoded_frame",
        },
        "result": {
            "coverage_complete": evidence.coverage_complete(),
            "decoded_frame_digests_complete": evidence.decoded_frame_digests_complete(),
            "kinematic_detail_complete": evidence.kinematic_detail_complete(),
            "passing_fraction": passing_fraction,
            "passing_frames": evidence.total_frames - len(failed_frames),
            "failed_frames": len(failed_frames),
            "first_failed_frames": list(failed_frames[:40]),
            "failure_histogram": evidence.failure_histogram(args.minimum_confidence),
        },
        "outputs": {
            "evidence": str(evidence_path),
            "evidence_sha256": _sha256(evidence_path),
            "demo": str(demo_path) if demo_path.is_file() else None,
            "demo_sha256": _sha256(demo_path) if demo_path.is_file() else None,
        },
        "limitations": [
            "Semantic topology labels are review evidence, not a learned 3-D reconstruction.",
            "A passing review establishes visible 2-D topology only, not physical joint feasibility.",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output": str(output_dir), **manifest["result"]}, indent=2))
    return 0 if accepted or args.allow_rejected else 2


if __name__ == "__main__":
    raise SystemExit(main())
