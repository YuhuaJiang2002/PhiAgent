#!/usr/bin/env python3
"""Build quality-gated RGB-alpha-contact VACE distillation clips.

Only a teacher that passes the real long-video image-space contract and its
adversarial attacks can become supervision.  Training clips cover the
pre-20-second interval; validation clips are held entirely at or after 20
seconds.  This prevents the long-horizon tail from leaking into training.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.data.adaptation import (  # noqa: E402
    AdaptationArm,
    AdaptationAsset,
    AdaptationAssetKind,
    AdaptationManifest,
    AdaptationSplit,
    VaceTrainingExample,
    file_sha256,
)
from phiagent.rendering.object_factored_long_video import (  # noqa: E402
    SourceResizeCrop,
    binary_dilate_square,
    remap_boolean_mask,
    resolve_flower_visibility,
    source_skin_like,
    strict_flower_seed,
)
from phiagent.rendering.robot_layer_contract import (  # noqa: E402
    RobotLayerContract,
    make_state_control,
)


UPPER = (
    "palette_surprisal",
    "high_chroma_fraction",
    "skin_like_fraction",
    "spatial_chroma_tv",
)
LOWER = (
    "replacement_coverage",
    "arm_replacement_coverage",
    "hand_replacement_coverage",
    "grid_topology_coverage",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--teacher-video", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--person-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument("--limb-masks", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--teacher-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--train-clips", type=int, default=12)
    parser.add_argument("--validation-clips", type=int, default=4)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--source-frame-step", type=int, default=3)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--late-start", type=int, default=480)
    parser.add_argument("--minimum-window-pass-fraction", type=float, default=0.80)
    parser.add_argument("--contact-radius", type=int, default=3)
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


def _row_passes(row: dict[str, Any], limits: dict[str, float]) -> bool:
    return (
        all(float(row[name]) <= float(limits[name]) for name in UPPER)
        and all(float(row[name]) >= float(limits[name]) for name in LOWER)
        and bool(row["contact_pass"])
    )


def select_curriculum_starts(
    rows: list[dict[str, Any]],
    limits: dict[str, float],
    *,
    train_clips: int,
    validation_clips: int,
    frames: int,
    source_frame_step: int,
    late_start: int,
    minimum_pass_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select diverse train windows and disjoint >=20s validation windows."""

    if frames <= 0 or source_frame_step <= 0 or train_clips <= 0 or validation_clips <= 0:
        raise ValueError("curriculum dimensions and clip counts must be positive")
    by_frame = {int(row["frame"]): row for row in rows}
    total = max(by_frame) + 1
    span = (frames - 1) * source_frame_step

    def candidates(start_min: int, start_max: int) -> list[dict[str, Any]]:
        result = []
        for start in range(start_min, start_max + 1):
            indices = [start + index * source_frame_step for index in range(frames)]
            selected = [by_frame[index] for index in indices if index in by_frame]
            if len(selected) != frames:
                continue
            passed = sum(_row_passes(row, limits) for row in selected)
            fraction = passed / frames
            if fraction < minimum_pass_fraction:
                continue
            contacts = sum(bool(row["contact_required"]) for row in selected)
            result.append(
                {
                    "start": start,
                    "indices": indices,
                    "pass_fraction": fraction,
                    "required_contact_frames": contacts,
                }
            )
        return result

    train_pool = candidates(0, late_start - span - 1)
    validation_pool = candidates(late_start, total - span - 1)

    def diverse(pool: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        if len(pool) < count:
            raise ValueError(
                f"only {len(pool)} quality-gated windows are available for {count} clips"
            )
        low, high = pool[0]["start"], pool[-1]["start"] + 1
        selected = []
        for index in range(count):
            bin_low = low + (high - low) * index / count
            bin_high = low + (high - low) * (index + 1) / count
            inside = [
                item
                for item in pool
                if bin_low <= item["start"] < bin_high
                and item not in selected
            ]
            if not inside:
                inside = [item for item in pool if item not in selected]
            center = (bin_low + bin_high) / 2.0
            selected.append(
                max(
                    inside,
                    key=lambda item: (
                        item["pass_fraction"],
                        min(item["required_contact_frames"], 4),
                        -abs(item["start"] - center),
                    ),
                )
            )
        return selected

    return diverse(train_pool, train_clips), diverse(validation_pool, validation_clips)


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _load_packed(np: Any, path: Path, key: str) -> tuple[Any, int, int, str]:
    payload = np.load(path, allow_pickle=False)
    return payload[key], int(payload["height"]), int(payload["width"]), str(payload["bitorder"])


def _unpack(np: Any, payload: tuple[Any, int, int, str], index: int) -> Any:
    packed, height, width, bitorder = payload
    return np.unpackbits(packed[index], bitorder=bitorder)[: height * width].reshape(
        height, width
    ).astype(bool)


def _read_frame(process: subprocess.Popen[bytes], size: int, label: str, index: int) -> bytes:
    assert process.stdout is not None
    result = bytearray()
    while len(result) < size:
        chunk = process.stdout.read(size - len(result))
        if not chunk:
            raise RuntimeError(f"{label} decoder ended before frame {index}")
        result.extend(chunk)
    return bytes(result)


def _encode(ffmpeg: Path, frames: Any, path: Path, fps: int) -> None:
    process = subprocess.Popen(
        [
            str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{frames.shape[2]}x{frames.shape[1]}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    _, stderr = process.communicate(frames.tobytes())
    if process.returncode:
        raise RuntimeError(f"ffmpeg failed for {path}: {stderr.decode(errors='replace')[-2000:]}")


def _asset(
    asset_id: str,
    path: Path,
    split: AdaptationSplit,
    kind: AdaptationAssetKind,
) -> AdaptationAsset:
    return AdaptationAsset(
        asset_id=asset_id,
        path=str(path.resolve()),
        split=split,
        kind=kind,
        source_uri=f"local://quality-gated-robot-layer/{asset_id}",
        rights_basis="user-authorized local experiment derivative; development-only training",
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
        training_authorized=True,
    )


def main() -> int:
    args = _parser().parse_args()
    import numpy as np
    from PIL import Image

    if args.frames < 9 or (args.frames - 1) % 4:
        raise ValueError("VACE frames must satisfy 4n+1 and be at least 9")
    if args.width % 16 or args.height % 16:
        raise ValueError("VACE dimensions must be divisible by 16")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit = json.loads(args.audit_report.read_text())
    teacher = next(
        (candidate for candidate in audit["candidates"] if candidate["name"] == args.teacher_name),
        None,
    )
    if teacher is None:
        raise ValueError(f"audit has no candidate {args.teacher_name!r}")
    if not teacher["summary"]["image_space_contract_pass"]:
        raise ValueError("refusing to distill a teacher that fails the image-space contract")
    if not teacher["adversarial"]["all_attacks_detected"]:
        raise ValueError("refusing to distill before all adversarial attacks are detected")
    frame_metrics_path = Path(teacher["frame_metrics_path"])
    rows = json.loads(frame_metrics_path.read_text())
    limits = teacher["summary"]["limits_fit_only_on_anchor_frames"]
    train_windows, validation_windows = select_curriculum_starts(
        rows,
        limits,
        train_clips=args.train_clips,
        validation_clips=args.validation_clips,
        frames=args.frames,
        source_frame_step=args.source_frame_step,
        late_start=args.late_start,
        minimum_pass_fraction=args.minimum_window_pass_fraction,
    )
    windows = [
        (AdaptationSplit.TRAIN, item) for item in train_windows
    ] + [(AdaptationSplit.VALIDATION, item) for item in validation_windows]
    requested = {index for _, item in windows for index in item["indices"]}
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
    target_frame = SourceResizeCrop(
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
    contract = RobotLayerContract(
        target_frame.name,
        f"absolute_frame_index:full_source_{args.expected_frames}",
        target_frame.output_width,
        target_frame.output_height,
        args.contact_radius,
    )
    contract.validate()
    source_command = [
        str(args.ffmpeg), "-v", "error", "-i", str(args.source_video.resolve()), "-vf",
        (
            f"scale={target_frame.scaled_width}:{target_frame.scaled_height}:flags=area,"
            f"crop={target_frame.output_width}:{target_frame.output_height}:"
            f"{target_frame.crop_left}:{target_frame.crop_top}"
        ),
        "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    teacher_command = [
        str(args.ffmpeg), "-v", "error", "-i", str(args.teacher_video.resolve()),
        "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    source_decoder = subprocess.Popen(source_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    teacher_decoder = subprocess.Popen(teacher_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = target_frame.output_width * target_frame.output_height * 3
    target_frames: dict[int, Any] = {}
    control_frames: dict[int, Any] = {}
    state_rows = []
    for index in range(args.expected_frames):
        source = np.frombuffer(_read_frame(source_decoder, frame_size, "source", index), dtype=np.uint8).reshape(
            target_frame.output_height, target_frame.output_width, 3
        )
        teacher_rgb = np.frombuffer(_read_frame(teacher_decoder, frame_size, "teacher", index), dtype=np.uint8).reshape(
            target_frame.output_height, target_frame.output_width, 3
        )
        if index not in requested:
            continue
        person = remap_boolean_mask(
            np, _unpack(np, person_payload, index), source_frame=mask_frame, target_frame=target_frame
        )
        flower_track = remap_boolean_mask(
            np, _unpack(np, flower_payload, index), source_frame=mask_frame, target_frame=target_frame
        )
        hands = remap_boolean_mask(
            np, _unpack(np, hand_payload, index), source_frame=mask_frame, target_frame=target_frame
        )
        alpha = binary_dilate_square(np, person, args.person_dilation)
        skin = source_skin_like(np, source) & (person | hands)
        skin = binary_dilate_square(np, skin & alpha, args.skin_negative_dilation)
        flower = resolve_flower_visibility(
            np,
            candidates=flower_track | strict_flower_seed(np, source),
            edit_support=alpha,
            source_person=person,
            source_skin_negative=skin,
            person_core_erosion=args.person_core_negative_erosion,
        )
        control = make_state_control(
            np,
            robot_alpha=alpha,
            hand_mask=hands,
            object_mask=flower,
            contact_radius=args.contact_radius,
        )
        target_frames[index] = np.asarray(
            Image.fromarray(teacher_rgb).resize((args.width, args.height), Image.Resampling.LANCZOS)
        )
        control_frames[index] = np.asarray(
            Image.fromarray(control).resize((args.width, args.height), Image.Resampling.NEAREST)
        )
        state_rows.append(
            {
                "frame": index,
                "alpha_fraction": float(alpha.mean()),
                "object_boundary_pixels": int(np.count_nonzero(control[..., 1])),
                "contact_marker_pixels": int(np.count_nonzero(control[..., 2])),
            }
        )
    for label, process in (("source", source_decoder), ("teacher", teacher_decoder)):
        if process.stdout is not None:
            process.stdout.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        if process.wait():
            raise RuntimeError(f"{label} decoder failed: {stderr[-2000:]}")
    reference_completed = subprocess.run(
        [
            str(args.ffmpeg), "-v", "error", "-i", str(args.reference_image.resolve()), "-vf",
            (
                f"scale={target_frame.scaled_width}:{target_frame.scaled_height}:flags=area,"
                f"crop={target_frame.output_width}:{target_frame.output_height}:"
                f"{target_frame.crop_left}:{target_frame.crop_top},"
                f"scale={args.width}:{args.height}:flags=lanczos"
            ),
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True,
        capture_output=True,
    )
    reference = np.frombuffer(reference_completed.stdout, dtype=np.uint8).reshape(
        args.height, args.width, 3
    )
    reference_dir = output / "train"
    reference_dir.mkdir(parents=True)
    reference_path = reference_dir / "canonical-reference.png"
    Image.fromarray(reference).save(reference_path)
    assets: list[AdaptationAsset] = [
        _asset(
            "train-canonical-reference",
            reference_path,
            AdaptationSplit.TRAIN,
            AdaptationAssetKind.VACE_REFERENCE_IMAGE,
        )
    ]
    examples = []
    validation_records = []
    for clip_index, (split, window) in enumerate(windows):
        clip_dir = output / split.value / f"clip-{clip_index:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        target_path = clip_dir / "target.mp4"
        control_path = clip_dir / "rgb-alpha-contact-control.mp4"
        clip_targets = np.stack([target_frames[index] for index in window["indices"]])
        clip_controls = np.stack([control_frames[index] for index in window["indices"]])
        _encode(args.ffmpeg, clip_targets, target_path, args.fps)
        _encode(args.ffmpeg, clip_controls, control_path, args.fps)
        record = {
            **window,
            "target": str(target_path.resolve()),
            "control": str(control_path.resolve()),
            "reference": str(reference_path.resolve()),
        }
        (clip_dir / "contract.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        if split is AdaptationSplit.TRAIN:
            prefix = f"train-{clip_index:03d}"
            target_asset = _asset(prefix + "-target", target_path, split, AdaptationAssetKind.TARGET_VIDEO)
            control_asset = _asset(prefix + "-control", control_path, split, AdaptationAssetKind.VACE_CONTROL_VIDEO)
            assets.extend((target_asset, control_asset))
            examples.append(
                VaceTrainingExample(
                    example_id=f"robot-layer-contact-{clip_index:03d}",
                    target_video_asset_id=target_asset.asset_id,
                    control_video_asset_id=control_asset.asset_id,
                    reference_image_asset_id="train-canonical-reference",
                    prompt=(
                        "A consistent silver humanoid robot replaces the tracked person and "
                        "uses two articulated hands to arrange flowers; preserve robot material, "
                        "limb topology, visible source flowers, and marked hand-flower contact."
                    ),
                )
            )
        else:
            validation_records.append(record)
    manifest = AdaptationManifest(
        experiment_id=output.name,
        arm=AdaptationArm.VACE_LORA,
        assets=tuple(assets),
        vace_examples=tuple(examples),
        evidence_scope="development_only",
    )
    manifest_path = output / "frozen" / "manifest.json"
    manifest.write_json(manifest_path)
    (output / "validation.json").write_text(
        json.dumps(validation_records, indent=2, sort_keys=True) + "\n"
    )
    (output / "state-contract-frames.json").write_text(
        json.dumps(state_rows, indent=2, sort_keys=True) + "\n"
    )
    packages = {}
    for name in ("numpy", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    provenance = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "method": "quality_gated_real_robot_rgb_alpha_contact_vace_curriculum",
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "contract": contract.to_dict(),
        "teacher_audit": {
            "path": str(args.audit_report.resolve()),
            "sha256": _sha256(args.audit_report),
            "teacher": args.teacher_name,
            "image_space_contract_pass": True,
            "adversarial_audit_pass": True,
        },
        "train_windows": train_windows,
        "validation_windows_at_or_after_20_seconds": validation_windows,
        "manifest_sha256": _sha256(manifest_path),
        "seed": args.seed,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "gpu": {"used": False, "reason": "CPU curriculum construction"},
        "limitations": [
            "The real teacher is generated and comes from one scene and robot identity.",
            "Validation is held after 20 seconds but remains the same source video.",
            "The contact channel encodes 2D adjacency, not force or depth.",
            "Adapter training and an unchanged held-out gate remain required.",
        ],
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"dataset": str(output), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
