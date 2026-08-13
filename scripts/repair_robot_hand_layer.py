#!/usr/bin/env python3
"""Repair audited hand under-coverage with a bounded second robot layer.

Only donor pixels that replace the source where the base candidate does not are
eligible.  Copying is restricted to tracked hand support and excludes the
resolved source-flower layer.  This is a deterministic image-space repair, not
a claim of 3D grasp validity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.object_factored_long_video import (  # noqa: E402
    SourceResizeCrop,
    binary_dilate_square,
    remap_boolean_mask,
    resolve_flower_visibility,
    source_skin_like,
    strict_flower_seed,
)
from phiagent.rendering.robot_layer_contract import (  # noqa: E402
    merge_missing_replacement,
    project_missing_contact,
)


def expanded_repair_frames(
    failures: list[int],
    *,
    total_frames: int,
    padding: int,
    maximum_gap: int,
) -> tuple[int, ...]:
    """Expand nearby failed frames into bounded temporal repair intervals."""

    if total_frames <= 0 or padding < 0 or maximum_gap < 1:
        raise ValueError("invalid repair interval settings")
    selected = sorted(set(failures))
    if any(frame < 0 or frame >= total_frames for frame in selected):
        raise ValueError("repair failure frame lies outside the timeline")
    if not selected:
        return ()
    groups: list[tuple[int, int]] = []
    start = previous = selected[0]
    for frame in selected[1:]:
        if frame - previous > maximum_gap:
            groups.append((start, previous))
            start = frame
        previous = frame
    groups.append((start, previous))
    result = set()
    for start, end in groups:
        result.update(
            range(max(0, start - padding), min(total_frames, end + padding + 1))
        )
    return tuple(sorted(result))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--base-video", type=Path, required=True)
    parser.add_argument("--donor-video", type=Path, required=True)
    parser.add_argument("--person-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument("--limb-masks", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--base-frame-metrics", type=Path, required=True)
    parser.add_argument("--base-name", required=True)
    parser.add_argument(
        "--repair-trigger",
        choices=("hand_coverage", "projected_contact"),
        default="hand_coverage",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--late-start", type=int, default=480)
    parser.add_argument("--replacement-threshold", type=float, default=12.0)
    parser.add_argument("--hand-expansion", type=int, default=1)
    parser.add_argument("--contact-radius", type=int, default=3)
    parser.add_argument("--contact-bridge-steps", type=int, default=6)
    parser.add_argument("--temporal-padding", type=int, default=2)
    parser.add_argument("--maximum-failure-gap", type=int, default=3)
    parser.add_argument("--person-dilation", type=int, default=10)
    parser.add_argument("--skin-negative-dilation", type=int, default=2)
    parser.add_argument("--person-core-negative-erosion", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--mask-frame-name", required=True)
    parser.add_argument("--mask-source-width", type=int, required=True)
    parser.add_argument("--mask-source-height", type=int, required=True)
    parser.add_argument("--mask-scaled-width", type=int, required=True)
    parser.add_argument("--mask-scaled-height", type=int, required=True)
    parser.add_argument("--mask-crop-left", type=int, required=True)
    parser.add_argument("--mask-crop-top", type=int, required=True)
    parser.add_argument("--target-frame-name", required=True)
    parser.add_argument("--target-scaled-width", type=int, required=True)
    parser.add_argument("--target-scaled-height", type=int, required=True)
    parser.add_argument("--target-crop-left", type=int, required=True)
    parser.add_argument("--target-crop-top", type=int, required=True)
    parser.add_argument("--target-width", type=int, required=True)
    parser.add_argument("--target-height", type=int, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_packed(np: Any, path: Path, key: str) -> tuple[Any, int, int, str]:
    payload = np.load(path, allow_pickle=False)
    return payload[key], int(payload["height"]), int(payload["width"]), str(payload["bitorder"])


def _unpack(np: Any, payload: tuple[Any, int, int, str], index: int) -> Any:
    packed, height, width, bitorder = payload
    values = np.unpackbits(packed[index], bitorder=bitorder)[: height * width]
    return values.reshape(height, width).astype(bool)


def _read_frame(process: subprocess.Popen[bytes], frame_bytes: int, label: str, index: int) -> bytes:
    assert process.stdout is not None
    chunks = []
    remaining = frame_bytes
    while remaining:
        chunk = process.stdout.read(remaining)
        if not chunk:
            raise RuntimeError(f"{label} decoder ended before frame {index}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decoder(ffmpeg: Path, path: Path, *, source: bool, target: SourceResizeCrop) -> list[str]:
    command = [str(ffmpeg), "-v", "error", "-i", str(path.resolve())]
    if source:
        command.extend(
            [
                "-vf",
                (
                    f"scale={target.scaled_width}:{target.scaled_height}:flags=area,"
                    f"crop={target.output_width}:{target.output_height}:"
                    f"{target.crop_left}:{target.crop_top}"
                ),
            ]
        )
    return [*command, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


def _writer(ffmpeg: Path, output: Path, width: int, height: int, fps: float, lossless: bool) -> list[str]:
    command = [
        str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "-", "-an",
    ]
    if lossless:
        return [*command, "-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "bgr0", str(output)]
    return [
        *command, "-c:v", "libx264", "-preset", "medium", "-crf", "10",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]


def main() -> int:
    args = _parser().parse_args()
    import numpy as np

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit = json.loads(args.audit_report.read_text())
    candidate = next(
        (item for item in audit["candidates"] if item["name"] == args.base_name),
        None,
    )
    if candidate is None:
        raise ValueError(f"audit has no candidate {args.base_name!r}")
    hand_limit = candidate["summary"]["limits_fit_only_on_anchor_frames"][
        "hand_replacement_coverage"
    ]
    rows = json.loads(args.base_frame_metrics.read_text())
    if args.repair_trigger == "hand_coverage":
        failures = [
            int(row["frame"])
            for row in rows
            if row["frame"] >= args.late_start
            and row["hand_replacement_coverage"] < hand_limit
        ]
    else:
        failures = [
            int(row["frame"])
            for row in rows
            if row["frame"] >= args.late_start
            and row["contact_required"]
            and not row["contact_observed"]
        ]
    active_frames = expanded_repair_frames(
        failures,
        total_frames=args.expected_frames,
        padding=args.temporal_padding,
        maximum_gap=args.maximum_failure_gap,
    )
    active = set(active_frames)
    person_payload = _load_packed(np, args.person_masks, "packed")
    flower_payload = _load_packed(np, args.flower_masks, "packed")
    hand_payload = _load_packed(np, args.limb_masks, "hands_packed")
    _, mask_height, mask_width, _ = person_payload
    mask_frame = SourceResizeCrop(
        args.mask_frame_name,
        args.mask_source_width,
        args.mask_source_height,
        args.mask_scaled_width,
        args.mask_scaled_height,
        args.mask_crop_left,
        args.mask_crop_top,
        mask_width,
        mask_height,
    )
    target = SourceResizeCrop(
        args.target_frame_name,
        args.mask_source_width,
        args.mask_source_height,
        args.target_scaled_width,
        args.target_scaled_height,
        args.target_crop_left,
        args.target_crop_top,
        args.target_width,
        args.target_height,
    )
    mask_frame.validate()
    target.validate()
    decoder_specs = {
        "source": _decoder(args.ffmpeg, args.source_video, source=True, target=target),
        "base": _decoder(args.ffmpeg, args.base_video, source=False, target=target),
        "donor": _decoder(args.ffmpeg, args.donor_video, source=False, target=target),
    }
    decoders = {
        name: subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for name, command in decoder_specs.items()
    }
    lossless_path = output / "robot-hand-union-lossless.mkv"
    review_path = output / "robot-hand-union-27p5s.mp4"
    writer_specs = {
        "lossless": _writer(args.ffmpeg, lossless_path, target.output_width, target.output_height, args.fps, True),
        "review": _writer(args.ffmpeg, review_path, target.output_width, target.output_height, args.fps, False),
    }
    writers = {
        name: subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for name, command in writer_specs.items()
    }
    frame_bytes = target.output_width * target.output_height * 3
    started = time.perf_counter()
    copied_per_frame = []
    contact_projected_per_frame = []
    contact_projection_steps = []
    contact_projection_passed = []
    offsets = []
    for index in range(args.expected_frames):
        frames = {
            name: np.frombuffer(
                _read_frame(process, frame_bytes, name, index), dtype=np.uint8
            ).reshape(target.output_height, target.output_width, 3)
            for name, process in decoders.items()
        }
        result = frames["base"]
        copied = np.zeros((target.output_height, target.output_width), dtype=bool)
        offset = (0.0, 0.0, 0.0)
        if index in active:
            person = remap_boolean_mask(
                np,
                _unpack(np, person_payload, index),
                source_frame=mask_frame,
                target_frame=target,
            )
            tracked_flower = remap_boolean_mask(
                np,
                _unpack(np, flower_payload, index),
                source_frame=mask_frame,
                target_frame=target,
            )
            hands = remap_boolean_mask(
                np,
                _unpack(np, hand_payload, index),
                source_frame=mask_frame,
                target_frame=target,
            )
            support = binary_dilate_square(np, person, args.person_dilation)
            skin = source_skin_like(np, frames["source"]) & (person | hands)
            skin = binary_dilate_square(
                np, skin & support, args.skin_negative_dilation
            )
            flower = resolve_flower_visibility(
                np,
                candidates=tracked_flower | strict_flower_seed(np, frames["source"]),
                edit_support=support,
                source_person=person,
                source_skin_negative=skin,
                person_core_erosion=args.person_core_negative_erosion,
            )
            result, copied, offset = merge_missing_replacement(
                np,
                base_rgb=frames["base"],
                donor_rgb=frames["donor"],
                source_rgb=frames["source"],
                hand_support=hands,
                protected_object=flower,
                replacement_threshold=args.replacement_threshold,
                expansion_radius=args.hand_expansion,
            )
            projected = np.zeros_like(copied)
            projection_steps = 0
            projection_passed = False
            if args.repair_trigger == "projected_contact":
                result, projected, projection_steps, projection_passed = (
                    project_missing_contact(
                        np,
                        candidate_rgb=result,
                        source_rgb=frames["source"],
                        hand_support=hands,
                        protected_object=flower,
                        replacement_threshold=args.replacement_threshold,
                        contact_radius=args.contact_radius,
                        maximum_bridge_steps=args.contact_bridge_steps,
                    )
                )
        else:
            projected = np.zeros_like(copied)
            projection_steps = 0
            projection_passed = False
        for writer in writers.values():
            assert writer.stdin is not None
            writer.stdin.write(result.tobytes())
        copied_per_frame.append(int(np.count_nonzero(copied)))
        contact_projected_per_frame.append(int(np.count_nonzero(projected)))
        contact_projection_steps.append(projection_steps)
        contact_projection_passed.append(projection_passed)
        offsets.append(offset)
    for writer in writers.values():
        assert writer.stdin is not None
        writer.stdin.close()
    process_errors = {}
    for group, processes in (("decoder", decoders), ("writer", writers)):
        for name, process in processes.items():
            if group == "decoder" and process.stdout is not None:
                process.stdout.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            returncode = process.wait()
            process_errors[f"{group}:{name}"] = {
                "returncode": returncode,
                "stderr": stderr,
            }
            if returncode:
                raise RuntimeError(f"{group} {name} failed: {stderr[-2000:]}")
    wall = time.perf_counter() - started
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "method": "source_aware_missing_robot_hand_layer_union_and_contact_projection",
        "scope": "2D hand replacement coverage repair; no 3D contact claim",
        "command": [sys.executable, *sys.argv],
        "coordinate_frames": {
            "mask": mask_frame.to_dict(),
            "target": target.to_dict(),
            "timeline": f"absolute_frame_index:full_source_{args.expected_frames}",
        },
        "audit_source": {
            "path": str(args.audit_report.resolve()),
            "sha256": _sha256(args.audit_report),
            "base_name": args.base_name,
            "hand_replacement_lower_limit": hand_limit,
            "raw_failed_frames": failures,
            "temporally_expanded_repair_frames": list(active_frames),
            "repair_trigger": args.repair_trigger,
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in {
                "source": args.source_video,
                "base": args.base_video,
                "donor": args.donor_video,
                "person_masks": args.person_masks,
                "flower_masks": args.flower_masks,
                "limb_masks": args.limb_masks,
            }.items()
        },
        "outputs": {
            "lossless": {"path": str(lossless_path), "sha256": _sha256(lossless_path)},
            "review": {"path": str(review_path), "sha256": _sha256(review_path)},
        },
        "metrics": {
            "frames": args.expected_frames,
            "video_seconds": args.expected_frames / args.fps,
            "wall_seconds": wall,
            "repair_fps": args.expected_frames / wall,
            "realtime_factor": wall / (args.expected_frames / args.fps),
            "frames_with_copied_pixels": sum(value > 0 for value in copied_per_frame),
            "total_copied_pixels": sum(copied_per_frame),
            "maximum_copied_pixels_per_frame": max(copied_per_frame),
            "frames_with_contact_projection": sum(
                value > 0 for value in contact_projected_per_frame
            ),
            "total_contact_projected_pixels": sum(contact_projected_per_frame),
            "maximum_contact_projected_pixels_per_frame": max(
                contact_projected_per_frame
            ),
            "maximum_contact_projection_steps": max(contact_projection_steps),
            "contact_projection_passed_frames": sum(contact_projection_passed),
            "mean_color_offset_on_active_frames": [
                float(np.mean([offset[channel] for index, offset in enumerate(offsets) if index in active]))
                for channel in range(3)
            ],
        },
        "processes": process_errors,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "seed": args.seed,
        "limitations": [
            "The donor is another generated hypothesis, not ground-truth paired robot footage.",
            "The protected flower union cannot establish persistent per-stem identity.",
            "Projected hand/object adjacency is not force closure or depth-correct grasp evidence.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(review_path), "metrics": manifest["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
