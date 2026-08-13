#!/usr/bin/env python3
"""Route only LoRA-derived illumination into geometry-safe robot pixels.

The direct Wan relighting output is treated as an illumination proposal, never
as replacement geometry.  The accepted geometry window remains the immutable
pixel source.  Flowers, prompted hands, the contact band, and all pixels outside
the independently tracked robot matte are preserved exactly before encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-video", type=Path, required=True)
    parser.add_argument("--relighting-proposal-video", type=Path, required=True)
    parser.add_argument("--relighting-metadata", type=Path, required=True)
    parser.add_argument("--robot-masks", type=Path, required=True)
    parser.add_argument("--stem-instances", type=Path, required=True)
    parser.add_argument("--flower-union", type=Path, required=True)
    parser.add_argument("--hand-instances", type=Path, required=True)
    parser.add_argument(
        "--source-frame-range",
        type=int,
        nargs=2,
        metavar=("START", "END_EXCLUSIVE"),
        help="Optional contiguous subset of the aligned stem/hand tracks",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--strength", type=float, default=0.55)
    parser.add_argument("--maximum-lab-light-delta", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(cv2: Any, path: Path) -> list[Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video contains no decoded frames: {path}")
    return frames


def _unpack(np: Any, path: Path, key: str, frames: int, height: int, width: int) -> Any:
    payload = np.load(path)
    packed = payload[key]
    unpacked = np.unpackbits(packed, axis=-1, bitorder=str(payload["bitorder"]))
    return unpacked[..., : height * width].reshape(frames, height, width).astype(bool)


def select_source_frame_positions(
    source_indices: list[int], frame_range: list[int] | None
) -> tuple[list[int], list[int]]:
    if frame_range is None:
        return source_indices, list(range(len(source_indices)))
    start, end = frame_range
    if start < 0 or end <= start:
        raise ValueError("source-frame-range must satisfy 0 <= START < END_EXCLUSIVE")
    positions_by_frame = {frame: position for position, frame in enumerate(source_indices)}
    requested = list(range(start, end))
    missing = [frame for frame in requested if frame not in positions_by_frame]
    if missing:
        raise ValueError(
            f"source-frame-range is not fully represented by the tracks; missing {missing[:8]}"
        )
    return requested, [positions_by_frame[frame] for frame in requested]


def build_safe_relight_mask(
    cv2: Any,
    np: Any,
    robot: Any,
    flowers: Any,
    stem: Any,
    hands: Any,
) -> tuple[Any, Any]:
    protected = flowers | stem | hands
    protected = (
        cv2.dilate(
            protected.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        )
        > 0
    )
    interior = (
        cv2.erode(
            robot.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        > 0
    )
    # Keep an owned copy because some OpenCV comparison arrays expose temporary
    # storage that can be reused by a following NumPy unary operation.
    protected = protected.copy()
    safe = np.logical_and(interior, np.logical_not(protected))
    return safe, protected


def illumination_field(
    cv2: Any,
    np: Any,
    geometry_l: Any,
    proposal_l: Any,
    safe_mask: Any,
    maximum_delta: float,
) -> Any:
    safe = safe_mask.astype(np.float32)
    raw_delta = np.clip(
        proposal_l.astype(np.float32) - geometry_l.astype(np.float32),
        -maximum_delta,
        maximum_delta,
    )
    numerator = cv2.GaussianBlur(raw_delta * safe, (0, 0), 25.0)
    denominator = cv2.GaussianBlur(safe, (0, 0), 25.0)
    field = numerator / np.maximum(denominator, 0.05)
    return np.clip(field, -maximum_delta, maximum_delta)


def _encode(
    ffmpeg: Path,
    np: Any,
    frames: Any,
    path: Path,
    fps: int,
    pix_fmt: str,
    *,
    lossless_rgb: bool = False,
) -> None:
    height, width = frames.shape[1:3]
    command = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
    ]
    if lossless_rgb:
        command.extend(["-c:v", "libx264rgb", "-crf", "0", "-pix_fmt", "bgr24"])
    else:
        command.extend(["-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p"])
    command.append(str(path))
    completed = subprocess.run(command, input=np.ascontiguousarray(frames).tobytes(), check=False)
    if completed.returncode:
        raise RuntimeError(f"ffmpeg failed to encode {path}")


def main() -> int:
    args = _parser().parse_args()
    if args.fps <= 0 or not 0.0 < args.strength <= 1.0:
        raise ValueError("fps must be positive and strength must be in (0, 1]")
    if not 0.0 < args.maximum_lab_light_delta <= 32.0:
        raise ValueError("maximum Lab light delta must be in (0, 32]")
    paths = {
        "geometry_video": args.geometry_video.expanduser().resolve(),
        "relighting_proposal_video": args.relighting_proposal_video.expanduser().resolve(),
        "relighting_metadata": args.relighting_metadata.expanduser().resolve(),
        "robot_masks": args.robot_masks.expanduser().resolve(),
        "stem_instances": args.stem_instances.expanduser().resolve(),
        "flower_union": args.flower_union.expanduser().resolve(),
        "hand_instances": args.hand_instances.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty {name}: {path}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")

    import cv2
    import numpy as np

    geometry = _decode(cv2, paths["geometry_video"])
    proposal = _decode(cv2, paths["relighting_proposal_video"])
    if len(geometry) != len(proposal):
        raise RuntimeError(
            f"geometry/proposal frame mismatch: {len(geometry)} != {len(proposal)}"
        )
    shape = geometry[0].shape
    if any(frame.shape != shape for frame in [*geometry, *proposal]):
        raise RuntimeError("geometry and proposal frames must share one shape")
    height, width = shape[:2]
    stem_payload = np.load(paths["stem_instances"])
    full_indices = [int(value) for value in stem_payload["source_frame_indices"]]
    indices, selected_positions = select_source_frame_positions(
        full_indices, args.source_frame_range
    )
    if len(indices) != len(geometry):
        raise RuntimeError("stem track and videos must contain the same number of frames")
    stems = np.unpackbits(
        stem_payload["masks_packed"], axis=2, bitorder=str(stem_payload["bitorder"])
    )[..., : height * width].reshape(
        1, len(full_indices), height, width
    ).astype(bool)[0, selected_positions]
    hands_payload = np.load(paths["hand_instances"])
    if [int(value) for value in hands_payload["source_frame_indices"]] != full_indices:
        raise RuntimeError("hand and stem tracks must use identical source frames")
    hands = np.unpackbits(
        hands_payload["masks_packed"], axis=2, bitorder=str(hands_payload["bitorder"])
    )[..., : height * width].reshape(
        2, len(full_indices), height, width
    ).astype(bool)[:, selected_positions]
    robot = _unpack(np, paths["robot_masks"], "packed", 660, height, width)[indices]
    flower_union = _unpack(
        np, paths["flower_union"], "packed", 660, height, width
    )[indices]

    safe_masks, protected_masks, geometry_l, proposal_l, fields = [], [], [], [], []
    for index in range(len(indices)):
        safe, protected = build_safe_relight_mask(
            cv2,
            np,
            robot[index],
            flower_union[index],
            stems[index],
            hands[:, index].any(axis=0),
        )
        if int(np.count_nonzero(safe)) < 1000:
            raise RuntimeError(f"safe relighting region is too small at local frame {index}")
        base_lab = cv2.cvtColor(geometry[index], cv2.COLOR_BGR2LAB)
        proposal_lab = cv2.cvtColor(proposal[index], cv2.COLOR_BGR2LAB)
        geometry_l.append(base_lab[..., 0].astype(np.float32))
        proposal_l.append(proposal_lab[..., 0].astype(np.float32))
        fields.append(
            illumination_field(
                cv2,
                np,
                geometry_l[-1],
                proposal_l[-1],
                safe,
                args.maximum_lab_light_delta,
            )
        )
        safe_masks.append(safe)
        protected_masks.append(protected)
    for _ in range(2):
        fields = [
            0.25 * fields[max(0, index - 1)]
            + 0.50 * fields[index]
            + 0.25 * fields[min(len(fields) - 1, index + 1)]
            for index in range(len(fields))
        ]

    composed, mask_frames = [], []
    protected_exact, outside_exact, hand_exact, flower_exact = [], [], [], []
    proposal_error_before, proposal_error_after, changed_fraction, changed_mae = [], [], [], []
    for index, base in enumerate(geometry):
        allowed = np.logical_and(
            robot[index], np.logical_not(protected_masks[index]).copy()
        )
        feather = cv2.GaussianBlur(
            safe_masks[index].astype(np.float32), (0, 0), 3.0
        )
        alpha = np.clip(feather, 0.0, 1.0) * allowed.astype(np.float32)
        lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 0] = np.clip(
            lab[..., 0] + args.strength * fields[index] * alpha, 0.0, 255.0
        )
        transformed = cv2.cvtColor(np.rint(lab).astype(np.uint8), cv2.COLOR_LAB2BGR)
        frame = base.copy()
        frame[allowed] = transformed[allowed]
        composed.append(frame)
        mask_frames.append(np.rint(alpha * 255.0).astype(np.uint8))

        protected_exact.append(float(np.mean(np.all(frame[protected_masks[index]] == base[protected_masks[index]], axis=1))))
        outside = np.logical_not(robot[index]).copy()
        outside_exact.append(float(np.mean(np.all(frame[outside] == base[outside], axis=1))))
        hand_region = hands[:, index].any(axis=0)
        flower_region = np.logical_or(flower_union[index], stems[index])
        hand_exact.append(float(np.mean(np.all(frame[hand_region] == base[hand_region], axis=1))))
        flower_exact.append(float(np.mean(np.all(frame[flower_region] == base[flower_region], axis=1))))
        safe = safe_masks[index]
        output_l = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        proposal_error_before.append(float(np.mean(np.abs(geometry_l[index][safe] - proposal_l[index][safe]))))
        proposal_error_after.append(float(np.mean(np.abs(output_l[safe] - proposal_l[index][safe]))))
        delta = np.mean(np.abs(frame.astype(np.float32) - base.astype(np.float32)), axis=2)
        changed_fraction.append(float(np.mean(delta[allowed] > 0.5)))
        changed_mae.append(float(np.mean(delta[allowed])))

    composed_array = np.stack(composed)
    mask_array = np.stack(mask_frames)
    output.mkdir(parents=True)
    candidate_path = output / "confidence-routed-relighting-lossless.mp4"
    compatibility_path = output / "confidence-routed-relighting.mp4"
    mask_path = output / "safe-relighting-mask.mp4"
    _encode(
        paths["ffmpeg"],
        np,
        composed_array,
        candidate_path,
        args.fps,
        "bgr24",
        lossless_rgb=True,
    )
    _encode(
        paths["ffmpeg"], np, composed_array, compatibility_path, args.fps, "bgr24"
    )
    _encode(paths["ffmpeg"], np, mask_array, mask_path, args.fps, "gray")

    relighting_residuals = [
        frame.astype(np.float32) - base.astype(np.float32)
        for frame, base in zip(composed, geometry)
    ]
    temporal_residual_rows = []
    for index in range(1, len(relighting_residuals)):
        region = safe_masks[index] | safe_masks[index - 1]
        temporal_residual_rows.append(
            {
                "from_source_frame": indices[index - 1],
                "to_source_frame": indices[index],
                "rgb_mae": float(
                    np.mean(
                        np.abs(
                            relighting_residuals[index]
                            - relighting_residuals[index - 1]
                        )[region]
                    )
                ),
            }
        )
    temporal_residual_mae = [
        float(row["rgb_mae"]) for row in temporal_residual_rows
    ]
    maximum_temporal_residual = max(
        temporal_residual_rows, key=lambda row: float(row["rgb_mae"])
    )

    rows = []
    review_indices = set(
        int(value)
        for value in np.rint(
            np.linspace(0, len(composed) - 1, min(5, len(composed)))
        ).astype(np.int32)
    )
    if len(composed) > 2:
        review_indices.update({len(composed) // 2 - 1, len(composed) // 2})
    for index in sorted(review_indices):
        cells = []
        for label, frame in (
            ("GEOMETRY", geometry[index]),
            ("LORA-RAW-REJECTED", proposal[index]),
            ("SAFE-MASK", np.repeat(mask_array[index][..., None], 3, axis=2)),
            ("ROUTED", composed[index]),
        ):
            item = cv2.resize(frame, (416, 240), interpolation=cv2.INTER_AREA)
            cv2.putText(item, f"{label} f={index}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cells.append(item)
        rows.append(cv2.hconcat(cells))
    review_path = output / "confidence-routed-review.jpg"
    cv2.imwrite(str(review_path), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 95])

    mean_before = float(np.mean(proposal_error_before))
    mean_after = float(np.mean(proposal_error_after))
    automatic_gates = {
        "all_expected_frames_decoded": len(composed) == len(indices),
        "flowers_exact_before_encode": min(flower_exact) == 1.0,
        "prompted_hands_exact_before_encode": min(hand_exact) == 1.0,
        "protected_interaction_exact_before_encode": min(protected_exact) == 1.0,
        "outside_robot_exact_before_encode": min(outside_exact) == 1.0,
        "lora_illumination_signal_nontrivial": float(np.mean(changed_mae)) >= 0.25,
        "lora_illumination_agreement_improved": mean_after < mean_before,
        "relighting_residual_temporally_bounded": max(temporal_residual_mae) <= 1.5,
    }
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "method": "confidence_routed_lora_luminance_with_immutable_geometry",
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "seed": args.seed,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "gpu": {"used": False, "reason": "deterministic CPU confidence routing"},
        "coordinate_frame": "camera:H3_output_pixels_832x480",
        "source_frame_indices": indices,
        "config": {
            "strength": args.strength,
            "maximum_lab_light_delta": args.maximum_lab_light_delta,
            "protected_dilation_pixels": 9,
            "robot_interior_erosion_pixels": 4,
            "spatial_sigma_pixels": 25.0,
            "temporal_kernel": [0.25, 0.5, 0.25],
            "temporal_smoothing_passes": 2,
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "metrics": {
            "safe_mask_fraction_min": float(min(mask.mean() for mask in safe_masks)),
            "safe_mask_fraction_max": float(max(mask.mean() for mask in safe_masks)),
            "changed_fraction_mean_in_allowed_region": float(np.mean(changed_fraction)),
            "changed_rgb_mae_mean_in_allowed_region": float(np.mean(changed_mae)),
            "lora_luminance_mae_before": mean_before,
            "lora_luminance_mae_after": mean_after,
            "relighting_residual_temporal_rgb_mae_mean": float(
                np.mean(temporal_residual_mae)
            ),
            "relighting_residual_temporal_rgb_mae_max": max(temporal_residual_mae),
            "relighting_residual_temporal_max_transition": maximum_temporal_residual,
            "relighting_residual_temporal_rows": temporal_residual_rows,
            "flowers_exact_fraction_min_before_encode": min(flower_exact),
            "prompted_hands_exact_fraction_min_before_encode": min(hand_exact),
            "protected_exact_fraction_min_before_encode": min(protected_exact),
            "outside_robot_exact_fraction_min_before_encode": min(outside_exact),
        },
        "automatic_gates": automatic_gates,
        "all_automatic_gates_pass": all(automatic_gates.values()),
        "outputs": {
            "candidate": {"path": str(candidate_path), "sha256": _sha256(candidate_path)},
            "compatibility_candidate": {
                "path": str(compatibility_path),
                "sha256": _sha256(compatibility_path),
            },
            "safe_mask": {"path": str(mask_path), "sha256": _sha256(mask_path)},
            "review": {"path": str(review_path), "sha256": _sha256(review_path)},
        },
        "limitations": [
            "The direct LoRA video is rejected because it regenerates flowers and the table; only its bounded low-frequency luminance signal is routed.",
            "This operation preserves 2D geometry but does not establish physical contact or 3D illumination correctness.",
            f"Acceptance remains scoped to these {len(indices)} source frames until human review; other phases require independent flower/contact evidence."
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output), **manifest["metrics"], **automatic_gates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
