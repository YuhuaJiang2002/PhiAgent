#!/usr/bin/env python3
"""Compile a 660-frame bimanual flower/contact contract from real evidence.

The existing SAM2 artifact is a flower *union* track.  This compiler uses it to
locate likely interaction intervals, but deliberately records every inferred
stem as a proxy rather than promoting connected components to flower identity.
The resulting contract can drive phase-local generation while its hard blockers
prevent an unsupported quality claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.agent.flower_task_adaptation import (  # noqa: E402
    BimanualPhase,
    ContactConstraint,
    EvidenceKind,
    FlowerInstanceSpec,
    FlowerTaskContract,
    HandPhase,
    HandSide,
    OcclusionOrder,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--flower-union-masks", type=Path, required=True)
    parser.add_argument("--source-limb-masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/usr/bin/ffprobe"))
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--contact-on-pixels", type=float, default=8.0)
    parser.add_argument("--contact-off-pixels", type=float, default=18.0)
    parser.add_argument("--minimum-contact-frames", type=int, default=6)
    parser.add_argument("--merge-gap-frames", type=int, default=12)
    parser.add_argument("--approach-frames", type=int, default=12)
    parser.add_argument("--grasp-frames", type=int, default=6)
    parser.add_argument("--release-frames", type=int, default=6)
    parser.add_argument("--retract-frames", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


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


def _video_info(ffprobe: Path, video: Path) -> dict[str, int | float]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/", maxsplit=1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": int(numerator) / int(denominator),
        "frames": int(stream["nb_frames"]),
        "duration": float(payload["format"]["duration"]),
    }


def _load_packed(np: Any, path: Path, key: str) -> tuple[Any, int, int]:
    payload = np.load(path)
    height, width = int(payload["height"]), int(payload["width"])
    packed = payload[key]
    unpacked = np.unpackbits(
        packed,
        axis=1,
        bitorder=str(payload["bitorder"]),
    )[:, : height * width]
    return unpacked.reshape(len(packed), height, width).astype(np.uint8), height, width


def _median_filter(np: Any, values: Any, radius: int = 3) -> Any:
    result = values.copy()
    for frame in range(len(values)):
        result[frame] = np.nanmedian(
            values[max(0, frame - radius) : min(len(values), frame + radius + 1)],
            axis=0,
        )
    return result


def _hand_flower_distances(cv2: Any, np: Any, flower_masks: Any, landmarks: Any) -> Any:
    frame_count, height, width = flower_masks.shape
    distances = np.full((frame_count, 2), np.nan, dtype=np.float32)
    hand_landmark_indices = ((4, 6, 8, 10), (5, 7, 9, 11))
    for frame, flower_mask in enumerate(flower_masks):
        distance_map = cv2.distanceTransform(1 - flower_mask, cv2.DIST_L2, 3)
        for side, indices in enumerate(hand_landmark_indices):
            samples = []
            for index in indices:
                x, y = landmarks[frame, index]
                px, py = round(float(x)), round(float(y))
                if math.isfinite(float(x + y)) and 0 <= px < width and 0 <= py < height:
                    samples.append(float(distance_map[py, px]))
            if samples:
                distances[frame, side] = min(samples)
    return _median_filter(np, distances)


def _contact_intervals(
    values: Any,
    *,
    on_threshold: float,
    off_threshold: float,
    minimum_frames: int,
    merge_gap: int,
) -> tuple[tuple[int, int], ...]:
    if not 0 <= on_threshold < off_threshold:
        raise ValueError("contact thresholds must satisfy 0 <= on < off")
    intervals: list[tuple[int, int]] = []
    active = False
    start = 0
    for frame, value in enumerate(values):
        if not active and math.isfinite(float(value)) and value <= on_threshold:
            active, start = True, frame
        elif active and (not math.isfinite(float(value)) or value >= off_threshold):
            if frame - start >= minimum_frames:
                intervals.append((start, frame))
            active = False
    if active and len(values) - start >= minimum_frames:
        intervals.append((start, len(values)))
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return tuple(merged)


def _build_phases(
    frame_count: int,
    right_intervals: tuple[tuple[int, int], ...],
    *,
    approach_frames: int,
    grasp_frames: int,
    release_frames: int,
    retract_frames: int,
) -> tuple[BimanualPhase, ...]:
    right_phase = [HandPhase.OBSERVE] * frame_count
    right_flower: list[str | None] = [None] * frame_count
    for interval_index, (start, end) in enumerate(right_intervals):
        flower_id = f"active-stem-proxy-{interval_index:02d}"
        for frame in range(max(0, start - approach_frames), start):
            if right_phase[frame] is HandPhase.OBSERVE:
                right_phase[frame] = HandPhase.APPROACH
        grasp_end = min(end, start + grasp_frames)
        release_start = max(grasp_end, end - release_frames)
        for frame in range(start, grasp_end):
            right_phase[frame], right_flower[frame] = HandPhase.GRASP, flower_id
        for frame in range(grasp_end, release_start):
            right_phase[frame], right_flower[frame] = HandPhase.MANIPULATE, flower_id
        for frame in range(release_start, end):
            right_phase[frame], right_flower[frame] = HandPhase.RELEASE, flower_id
        for frame in range(end, min(frame_count, end + retract_frames)):
            if right_phase[frame] is HandPhase.OBSERVE:
                right_phase[frame] = HandPhase.RETRACT

    phases: list[BimanualPhase] = []
    start = 0
    current = (HandPhase.HOLD, right_phase[0], "bouquet-main", right_flower[0])
    for frame in range(1, frame_count + 1):
        following = (
            None
            if frame == frame_count
            else (HandPhase.HOLD, right_phase[frame], "bouquet-main", right_flower[frame])
        )
        if following != current:
            phases.append(
                BimanualPhase(
                    phase_id=f"phase-{len(phases):03d}-{current[1].value}",
                    start_frame=start,
                    end_frame_exclusive=frame,
                    left_phase=current[0],
                    right_phase=current[1],
                    left_flower_id=current[2],
                    right_flower_id=current[3],
                )
            )
            start, current = frame, following
    return tuple(phases)


def _build_contacts(
    phases: tuple[BimanualPhase, ...],
    evidence_path: str,
) -> tuple[ContactConstraint, ...]:
    contacts = [
        ContactConstraint(
            "left-bouquet-full-clip-manual-proxy",
            HandSide.LEFT,
            "bouquet-main",
            0,
            phases[-1].end_frame_exclusive,
            HandPhase.HOLD,
            EvidenceKind.MANUAL_REVIEW,
            0.80,
            OcclusionOrder.DEPTH_TRACK_REQUIRED,
            evidence=("manual-review:source-storyboard:left-hand-holds-bouquet", evidence_path),
        )
    ]
    for phase in phases:
        if phase.right_phase not in {
            HandPhase.GRASP,
            HandPhase.MANIPULATE,
            HandPhase.RELEASE,
        }:
            continue
        assert phase.right_flower_id is not None
        contacts.append(
            ContactConstraint(
                f"right-{phase.phase_id}",
                HandSide.RIGHT,
                phase.right_flower_id,
                phase.start_frame,
                phase.end_frame_exclusive,
                phase.right_phase,
                EvidenceKind.UNION_MASK_PROXY,
                0.65,
                OcclusionOrder.DEPTH_TRACK_REQUIRED,
                evidence=(evidence_path,),
            )
        )
    return tuple(contacts)


def _plan_generation_jobs(
    phases: tuple[BimanualPhase, ...],
    contacts: tuple[ContactConstraint, ...],
    frame_count: int,
) -> list[dict[str, object]]:
    output_frames, overlap = 80, 16
    stride = output_frames - overlap
    final_start = frame_count - output_frames
    starts = [0]
    while starts[-1] < final_start:
        candidate = min(starts[-1] + stride, final_start)
        if candidate == starts[-1]:
            break
        starts.append(candidate)
    jobs = []
    for index, start in enumerate(starts):
        end = start + output_frames
        active_phases = [
            phase for phase in phases if phase.start_frame < end and phase.end_frame_exclusive > start
        ]
        active_contacts = [
            contact
            for contact in contacts
            if contact.start_frame < end and contact.end_frame_exclusive > start
        ]
        right_actions = list(dict.fromkeys(phase.right_phase.value for phase in active_phases))
        flower_ids = list(dict.fromkeys(contact.flower_id for contact in active_contacts))
        jobs.append(
            {
                "job_id": f"window-{index:02d}-{start:04d}-{end:04d}",
                "start_frame": start,
                "end_frame_exclusive": end,
                "input_frames": 81,
                "expected_output_frames": output_frames,
                "candidate_policy": "generate_multiple_then_preserve_one_immutable_window_candidate",
                "frame_level_candidate_mixing": False,
                "phase_ids": [phase.phase_id for phase in active_phases],
                "right_hand_actions": right_actions,
                "required_flower_ids": flower_ids,
                "prompt_constraint": (
                    "The left robot hand continuously holds bouquet-main. The right hand follows "
                    f"the ordered states {', '.join(right_actions)} without changing flower identity. "
                    "A flower moves only while held; the gripping hand occludes the stem at contact."
                ),
            }
        )
    return jobs


def _git_state() -> dict[str, object]:
    result: dict[str, object] = {}
    for key, command in {
        "head": ["git", "rev-parse", "--verify", "HEAD"],
        "status": ["git", "--no-pager", "status", "--short"],
    }.items():
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        result[key] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip().splitlines(),
            "stderr": completed.stderr.strip(),
        }
    return result


def _write_audit_video(
    cv2: Any,
    np: Any,
    *,
    source: Path,
    output: Path,
    flower_masks: Any,
    landmarks: Any,
    phases: tuple[BimanualPhase, ...],
    distances: Any,
    fps: float,
    ffmpeg: Path,
) -> None:
    height, width = flower_masks.shape[1:]
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode source video: {source}")
    command = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "14",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert writer.stdin is not None
    phase_index = 0
    kernel = np.ones((3, 3), np.uint8)
    for frame_index in range(len(flower_masks)):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"source decode stopped at frame {frame_index}")
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        mask = flower_masks[frame_index] > 0
        edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
        overlay = frame.copy()
        overlay[mask] = np.rint(0.55 * frame[mask] + 0.45 * np.asarray([40, 190, 40])).astype(
            np.uint8
        )
        overlay[edge] = (20, 255, 20)
        while phases[phase_index].end_frame_exclusive <= frame_index:
            phase_index += 1
        phase = phases[phase_index]
        for side, indices, color in (
            (0, (4, 6, 8, 10), (255, 170, 30)),
            (1, (5, 7, 9, 11), (30, 100, 255)),
        ):
            for index in indices:
                x, y = landmarks[frame_index, index]
                if math.isfinite(float(x + y)):
                    cv2.circle(overlay, (round(float(x)), round(float(y))), 4, color, -1)
            cv2.putText(
                overlay,
                f"{'L' if side == 0 else 'R'} union-distance={distances[frame_index, side]:.1f}px",
                (12, 28 + side * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            overlay,
            f"frame {frame_index:03d}/659  L=hold  R={phase.right_phase.value}",
            (12, height - 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "SAM2 union proxy - NOT single-stem identity evidence",
            (12, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (30, 30, 255),
            2,
            cv2.LINE_AA,
        )
        writer.stdin.write(overlay.tobytes())
    capture.release()
    writer.stdin.close()
    if writer.wait():
        raise RuntimeError("ffmpeg failed to encode contact audit video")


def main() -> int:
    args = _parser().parse_args()
    paths = {
        "source_video": args.source_video.expanduser().resolve(),
        "flower_union_masks": args.flower_union_masks.expanduser().resolve(),
        "source_limb_masks": args.source_limb_masks.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
        "ffprobe": args.ffprobe.expanduser().resolve(),
    }
    if args.adapter_checkpoint is not None:
        paths["adapter_checkpoint"] = args.adapter_checkpoint.expanduser().resolve()
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{name} does not exist or is empty: {path}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {output}")
    output.mkdir(parents=True)

    import cv2
    import numpy as np

    source_info = _video_info(paths["ffprobe"], paths["source_video"])
    flower_masks, height, width = _load_packed(
        np, paths["flower_union_masks"], "packed"
    )
    limb_payload = np.load(paths["source_limb_masks"])
    landmarks = limb_payload["landmarks_xy"]
    if (
        int(source_info["frames"]) != 660
        or abs(float(source_info["fps"]) - 24.0) > 1e-6
        or flower_masks.shape != (660, height, width)
        or landmarks.shape != (660, 12, 2)
    ):
        raise RuntimeError(
            "the full task contract requires aligned 660-frame/24-FPS video, masks, and landmarks"
        )
    distances = _hand_flower_distances(cv2, np, flower_masks, landmarks)
    right_intervals = _contact_intervals(
        distances[:, 1],
        on_threshold=args.contact_on_pixels,
        off_threshold=args.contact_off_pixels,
        minimum_frames=args.minimum_contact_frames,
        merge_gap=args.merge_gap_frames,
    )
    phases = _build_phases(
        660,
        right_intervals,
        approach_frames=args.approach_frames,
        grasp_frames=args.grasp_frames,
        release_frames=args.release_frames,
        retract_frames=args.retract_frames,
    )
    evidence_path = str(paths["flower_union_masks"])
    contacts = _build_contacts(phases, evidence_path)
    observed_frames = int(np.count_nonzero(flower_masks.reshape(660, -1).any(axis=1)))
    instances = [
        FlowerInstanceSpec(
            "bouquet-main",
            observed_frames,
            660,
            EvidenceKind.UNION_MASK_PROXY,
            False,
            evidence=(evidence_path,),
        )
    ]
    for index, (start, end) in enumerate(right_intervals):
        instances.append(
            FlowerInstanceSpec(
                f"active-stem-proxy-{index:02d}",
                end - start,
                660,
                EvidenceKind.UNION_MASK_PROXY,
                False,
                evidence=(evidence_path,),
            )
        )
    adapter_hash = (
        _sha256(paths["adapter_checkpoint"])
        if "adapter_checkpoint" in paths
        else None
    )
    contract = FlowerTaskContract(
        frame_count=660,
        fps=24.0,
        instances=tuple(instances),
        contacts=contacts,
        phases=phases,
        adapter_checkpoint_sha256=adapter_hash,
    )
    jobs = _plan_generation_jobs(phases, contacts, 660)
    contract_path = output / "task-contract.json"
    jobs_path = output / "generation-jobs.json"
    audit_path = output / "contact-instance-audit.mp4"
    storyboard_path = output / "contact-instance-storyboard.jpg"
    _write_json(contract_path, contract.to_dict())
    _write_json(
        jobs_path,
        {
            "schema_version": "1.0.0",
            "frame_count": 660,
            "fps": 24.0,
            "jobs": jobs,
            "selection_policy": "one immutable hard-gate-passing candidate per complete window",
            "frame_level_candidate_mixing": False,
        },
    )
    _write_audit_video(
        cv2,
        np,
        source=paths["source_video"],
        output=audit_path,
        flower_masks=flower_masks,
        landmarks=landmarks,
        phases=phases,
        distances=distances,
        fps=24.0,
        ffmpeg=paths["ffmpeg"],
    )
    subprocess.run(
        [
            str(paths["ffmpeg"]),
            "-y",
            "-v",
            "error",
            "-i",
            str(audit_path),
            "-vf",
            "fps=1,scale=416:-1,tile=4x7",
            "-frames:v",
            "1",
            str(storyboard_path),
        ],
        check=True,
    )
    packages = {}
    for name in ("numpy", "opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "schema_version": "1.0.0",
        "method": "bimanual_flower_phase_instance_contact_contract",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "honest_status": "PARTIAL",
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "seed": args.seed,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "gpu": {"used": False, "reason": "CPU evidence compilation and audit rendering"},
        "git": _git_state(),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "source_info": source_info,
        "contact_inference": {
            "right_intervals": [list(interval) for interval in right_intervals],
            "distance_landmarks": {
                "left": [15, 17, 19, 21],
                "right": [16, 18, 20, 22],
            },
            "distance_coordinate_frame": "camera:source_pixels",
            "thresholds": {
                "on_pixels": args.contact_on_pixels,
                "off_pixels": args.contact_off_pixels,
                "minimum_frames": args.minimum_contact_frames,
                "merge_gap_frames": args.merge_gap_frames,
            },
        },
        "claim_ready": contract.claim_ready,
        "claim_blockers": list(contract.claim_blockers),
        "outputs": {
            "contract": {"path": str(contract_path), "sha256": _sha256(contract_path)},
            "generation_jobs": {"path": str(jobs_path), "sha256": _sha256(jobs_path)},
            "audit_video": {"path": str(audit_path), "sha256": _sha256(audit_path)},
            "storyboard": {"path": str(storyboard_path), "sha256": _sha256(storyboard_path)},
        },
        "limitations": [
            "The pinned SAM2 artifact is one flower-union track, not per-stem instance masks.",
            "Union proximity cannot prove which flower is held or the front/back depth order.",
            "The left-hand full-clip bouquet hold is a visual-review proxy, not force or 3-D contact.",
            "This contract prepares generation and evaluation; it is not a generated candidate video.",
        ],
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output), "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
