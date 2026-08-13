#!/usr/bin/env python3
"""Audit articulated contact evidence and causal flower motion in a long video.

The visual motion audit is deliberately separate from physical contact.  It can
detect a flower region that is frozen while a nearby hand moves, but it cannot
promote 2-D adjacency to 3-D contact.  Metric contact and force closure fail
closed unless calibrated depth, an articulated joint sequence, and forces are
provided by an upstream reconstruction/simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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

from phiagent.rendering.contact_dynamics import (  # noqa: E402
    ArticulatedHandContract,
    InteractionFrameContract,
    MetricContactContract,
    assess_metric_force_closure,
    causal_motion_audit,
    validate_kinematic_sequence,
)
from phiagent.rendering.object_factored_long_video import (  # noqa: E402
    SourceResizeCrop,
    binary_dilate_square,
    remap_boolean_mask,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--person-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument("--limb-masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--contact-radius", type=int, default=3)
    parser.add_argument("--interaction-radius", type=int, default=16)
    parser.add_argument("--maximum-response-lag-frames", type=int, default=2)
    parser.add_argument("--maximum-frozen-run-frames", type=int, default=2)
    parser.add_argument("--minimum-local-flower-pixels", type=int, default=20)
    parser.add_argument("--active-stem-track", type=Path, action="append", default=[])
    parser.add_argument("--hand-kinematics", type=Path)
    parser.add_argument("--metric-contact", type=Path)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--mask-frame-name", required=True)
    parser.add_argument("--mask-source-width", type=int, required=True)
    parser.add_argument("--mask-source-height", type=int, required=True)
    parser.add_argument("--mask-scaled-width", type=int, required=True)
    parser.add_argument("--mask-scaled-height", type=int, required=True)
    parser.add_argument("--mask-crop-left", type=int, required=True)
    parser.add_argument("--mask-crop-top", type=int, required=True)
    parser.add_argument("--target-frame-name", required=True)
    parser.add_argument("--target-width", type=int, required=True)
    parser.add_argument("--target-height", type=int, required=True)
    parser.add_argument("--target-scaled-width", type=int, required=True)
    parser.add_argument("--target-scaled-height", type=int, required=True)
    parser.add_argument("--target-crop-left", type=int, required=True)
    parser.add_argument("--target-crop-top", type=int, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    result = {}
    for label, command in (
        ("head", ["git", "rev-parse", "HEAD"]),
        ("status", ["git", "status", "--short"]),
    ):
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
        )
        result[label] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return result


def _load_packed(np: Any, path: Path, key: str) -> tuple[Any, dict[str, Any]]:
    payload = np.load(path, allow_pickle=False)
    metadata = {
        "height": int(payload["height"]),
        "width": int(payload["width"]),
        "bitorder": str(payload["bitorder"]),
    }
    return payload[key], metadata


def _unpack(np: Any, packed: Any, metadata: dict[str, Any], index: int) -> Any:
    count = metadata["height"] * metadata["width"]
    return np.unpackbits(
        packed[index], bitorder=metadata["bitorder"]
    )[:count].reshape(metadata["height"], metadata["width"]).astype(bool)


def _mapped(
    np: Any,
    packed: Any,
    metadata: dict[str, Any],
    index: int,
    source_frame: SourceResizeCrop,
    target_frame: SourceResizeCrop,
) -> Any:
    return remap_boolean_mask(
        np,
        _unpack(np, packed, metadata, index),
        source_frame=source_frame,
        target_frame=target_frame,
    )


def _quantile(values: list[float], quantile: float, default: float) -> float:
    if not values:
        return default
    import numpy as np

    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _metric_contact_assessment(np: Any, path: Path | None, frame: InteractionFrameContract) -> dict[str, object]:
    if path is None:
        return assess_metric_force_closure(
            np,
            contact_points_m=None,
            surface_gaps_m=None,
            contact_normals=None,
            contact_forces_n=None,
            object_center_m=None,
            external_force_n=None,
            external_moment_nm=None,
            fingertip_indices=None,
            frame_contract=frame,
            contact_contract=MetricContactContract(),
            depth_source=None,
            force_source=None,
            occlusion_order_known=False,
        )
    raw = json.loads(path.read_text())
    contract = InteractionFrameContract(
        camera_frame=raw["camera_frame"],
        metric_frame=raw["metric_frame"],
        timeline=raw["timeline"],
        fps=float(raw["fps"]),
        fx_pixels=float(raw["intrinsics_pixels"]["fx"]),
        fy_pixels=float(raw["intrinsics_pixels"]["fy"]),
        cx_pixels=float(raw["intrinsics_pixels"]["cx"]),
        cy_pixels=float(raw["intrinsics_pixels"]["cy"]),
        metric_scale_source=raw["metric_scale_source"],
    )
    return assess_metric_force_closure(
        np,
        contact_points_m=raw["contact_points_m"],
        surface_gaps_m=raw["surface_gaps_m"],
        contact_normals=raw["contact_normals"],
        contact_forces_n=raw["contact_forces_n"],
        object_center_m=raw["object_center_m"],
        external_force_n=raw["external_force_n"],
        external_moment_nm=raw["external_moment_nm"],
        fingertip_indices=raw["fingertip_indices"],
        frame_contract=contract,
        contact_contract=MetricContactContract(),
        depth_source=raw.get("depth_source"),
        force_source=raw.get("force_source"),
        occlusion_order_known=bool(raw.get("occlusion_order_known")),
    )


def _kinematic_assessment(np: Any, path: Path | None) -> dict[str, object]:
    if path is None:
        return {
            "passed": False,
            "reasons": ["missing_articulated_metric_joint_sequence"],
            "note": "Pixel hand support cannot establish a fixed joint tree.",
        }
    payload = np.load(path, allow_pickle=False)
    names = tuple(str(value) for value in payload["joint_names"])
    contract = ArticulatedHandContract(
        embodiment_id=str(payload["embodiment_id"]),
        coordinate_frame=str(payload["coordinate_frame"]),
        joint_names=names,
        parent_indices=tuple(int(value) for value in payload["parent_indices"]),
        joint_limits_rad=tuple(
            (float(value[0]), float(value[1])) for value in payload["joint_limits_rad"]
        ),
        fingertip_indices=tuple(int(value) for value in payload["fingertip_indices"]),
        palm_index=int(payload["palm_index"]),
    )
    return validate_kinematic_sequence(
        np,
        joints_xyz_m=payload["joints_xyz_m"],
        joint_angles_rad=payload["joint_angles_rad"],
        contract=contract,
    )


def _active_stem_summary(np: Any, paths: list[Path]) -> list[dict[str, object]]:
    summaries = []
    for path in paths:
        payload = np.load(path, allow_pickle=False)
        masks = payload["masks_packed"]
        indices = payload["source_frame_indices"].astype(int)
        height = int(payload["height"])
        width = int(payload["width"])
        bitorder = str(payload["bitorder"])
        for instance_index, instance_id in enumerate(payload["instance_ids"]):
            centroids = []
            empty = []
            for local_index, frame_index in enumerate(indices):
                mask = np.unpackbits(
                    masks[instance_index, local_index], bitorder=bitorder
                )[: height * width].reshape(height, width).astype(bool)
                ys, xs = np.nonzero(mask)
                if len(xs):
                    centroids.append((int(frame_index), float(xs.mean()), float(ys.mean())))
                else:
                    empty.append(int(frame_index))
            speeds = [
                ((right[1] - left[1]) ** 2 + (right[2] - left[2]) ** 2) ** 0.5
                / max(1, right[0] - left[0])
                for left, right in zip(centroids, centroids[1:])
            ]
            summaries.append(
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "instance_id": str(instance_id),
                    "sampled_frames": len(indices),
                    "source_frame_start": int(indices[0]),
                    "source_frame_end": int(indices[-1]),
                    "empty_frames": empty,
                    "centroid_speed_pixels_per_source_frame_median": (
                        float(np.median(speeds)) if speeds else None
                    ),
                    "centroid_speed_pixels_per_source_frame_p05": (
                        float(np.quantile(speeds, 0.05)) if speeds else None
                    ),
                    "identity_scope": "prompted single-stem visual track, not physical instrumentation",
                }
            )
    return summaries


def _adversarial_audit(np: Any, grasp: Any, hand: Any, stem: Any, floors: dict[str, float], args: argparse.Namespace) -> dict[str, object]:
    erased = stem.copy()
    driver = grasp & (hand > floors["hand_motion"])
    for index in np.flatnonzero(driver):
        erased[int(index) : min(len(erased), int(index) + args.maximum_response_lag_frames + 1)] = 0.0
    response_attack = causal_motion_audit(
        np,
        grasp_active=grasp,
        hand_speed=hand,
        stem_speed=erased,
        hand_motion_floor=floors["hand_motion"],
        stem_motion_floor=floors["stem_motion"],
        maximum_response_lag_frames=args.maximum_response_lag_frames,
        maximum_frozen_run_frames=args.maximum_frozen_run_frames,
    )
    frame = InteractionFrameContract(
        camera_frame=args.target_frame_name,
        metric_frame="camera:metric_unavailable",
        timeline=f"absolute_frame_index:{args.expected_frames}",
        fps=args.fps,
    )
    depth_spoof = _metric_contact_assessment(np, None, frame)
    topology_detected = False
    try:
        ArticulatedHandContract(
            embodiment_id="attacked",
            coordinate_frame="robot_base:attacked",
            joint_names=("root", "finger-a", "finger-b"),
            parent_indices=(-1, 2, 0),
            joint_limits_rad=((-1.0, 1.0),) * 3,
            fingertip_indices=(1, 2),
            palm_index=0,
        ).validate()
    except ValueError:
        topology_detected = True
    gates = {
        "contact_response_erasure_detected": not bool(response_attack["passed"]),
        "missing_depth_force_spoof_rejected": not bool(depth_spoof["passed"]),
        "broken_joint_tree_detected": topology_detected,
    }
    return {
        "gates": gates,
        "all_attacks_detected": all(gates.values()),
        "contact_response_erasure": response_attack,
        "depth_force_spoof": depth_spoof,
    }


def main() -> int:
    args = _parser().parse_args()
    import cv2
    import numpy as np

    if args.expected_frames <= 1 or args.fps <= 0:
        raise ValueError("expected frames and FPS must be positive")
    inputs = {
        "candidate": args.candidate.expanduser().resolve(),
        "person_masks": args.person_masks.expanduser().resolve(),
        "flower_masks": args.flower_masks.expanduser().resolve(),
        "limb_masks": args.limb_masks.expanduser().resolve(),
    }
    optional = {
        "hand_kinematics": args.hand_kinematics.expanduser().resolve()
        if args.hand_kinematics
        else None,
        "metric_contact": args.metric_contact.expanduser().resolve()
        if args.metric_contact
        else None,
    }
    active_stems = [path.expanduser().resolve() for path in args.active_stem_track]
    for name, path in {**inputs, **optional}.items():
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    for path in active_stems:
        if not path.is_file():
            raise FileNotFoundError(f"missing active stem track: {path}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    output.mkdir(parents=True)

    source_frame = SourceResizeCrop(
        name=args.mask_frame_name,
        source_width=args.mask_source_width,
        source_height=args.mask_source_height,
        scaled_width=args.mask_scaled_width,
        scaled_height=args.mask_scaled_height,
        crop_left=args.mask_crop_left,
        crop_top=args.mask_crop_top,
        output_width=int(np.load(inputs["person_masks"], allow_pickle=False)["width"]),
        output_height=int(np.load(inputs["person_masks"], allow_pickle=False)["height"]),
    )
    target_frame = SourceResizeCrop(
        name=args.target_frame_name,
        source_width=args.mask_source_width,
        source_height=args.mask_source_height,
        scaled_width=args.target_scaled_width,
        scaled_height=args.target_scaled_height,
        crop_left=args.target_crop_left,
        crop_top=args.target_crop_top,
        output_width=args.target_width,
        output_height=args.target_height,
    )
    source_frame.validate()
    target_frame.validate()
    person_packed, person_meta = _load_packed(np, inputs["person_masks"], "packed")
    flower_packed, flower_meta = _load_packed(np, inputs["flower_masks"], "packed")
    hand_packed, hand_meta = _load_packed(np, inputs["limb_masks"], "hands_packed")

    capture = cv2.VideoCapture(str(inputs["candidate"]))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {inputs['candidate']}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if (width, height, frames) != (
        target_frame.output_width,
        target_frame.output_height,
        args.expected_frames,
    ):
        raise ValueError(
            f"candidate is {width}x{height}x{frames}, expected "
            f"{target_frame.output_width}x{target_frame.output_height}x{args.expected_frames}"
        )
    if abs(fps - args.fps) > 1e-3:
        raise ValueError(f"candidate FPS is {fps}, expected {args.fps}")

    rows = []
    previous_gray = None
    started = time.perf_counter()
    for index in range(args.expected_frames):
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"candidate decoder stopped at frame {index}")
        person = _mapped(
            np, person_packed, person_meta, index, source_frame, target_frame
        )
        flower = _mapped(
            np, flower_packed, flower_meta, index, source_frame, target_frame
        )
        hands = _mapped(
            np, hand_packed, hand_meta, index, source_frame, target_frame
        )
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if previous_gray is None:
            rows.append(
                {
                    "frame": index,
                    "seconds": index / args.fps,
                    "projected_contact": False,
                    "local_flower_pixels": 0,
                    "hand_motion_p90": 0.0,
                    "flower_motion_p90": 0.0,
                    "background_motion_p95": 0.0,
                    "measurement_valid": False,
                }
            )
            previous_gray = gray
            continue
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.linalg.norm(flow, axis=2)
        projected_contact = bool(
            np.any(binary_dilate_square(np, hands, args.contact_radius) & flower)
        )
        local_flower = flower & binary_dilate_square(
            np, hands, args.interaction_radius
        )
        hand_region = hands & ~flower
        safe_background = ~binary_dilate_square(np, person | flower, args.interaction_radius)
        measurement_valid = bool(
            projected_contact
            and int(np.count_nonzero(local_flower)) >= args.minimum_local_flower_pixels
            and int(np.count_nonzero(hand_region)) >= args.minimum_local_flower_pixels
        )
        rows.append(
            {
                "frame": index,
                "seconds": index / args.fps,
                "projected_contact": projected_contact,
                "local_flower_pixels": int(np.count_nonzero(local_flower)),
                "hand_motion_p90": (
                    float(np.quantile(magnitude[hand_region], 0.90))
                    if np.any(hand_region)
                    else 0.0
                ),
                "flower_motion_p90": (
                    float(np.quantile(magnitude[local_flower], 0.90))
                    if np.any(local_flower)
                    else 0.0
                ),
                "background_motion_p95": (
                    float(np.quantile(magnitude[safe_background], 0.95))
                    if np.any(safe_background)
                    else 0.0
                ),
                "measurement_valid": measurement_valid,
            }
        )
        previous_gray = gray
    capture.release()
    wall_seconds = time.perf_counter() - started

    background = [row["background_motion_p95"] for row in rows[1:]]
    sensor_noise = _quantile(background, 0.95, 0.0)
    floors = {
        "background_flow_p95_across_frames": sensor_noise,
        "stem_motion": max(0.10, 3.0 * sensor_noise),
        "hand_motion": max(0.30, 6.0 * sensor_noise),
        "derivation": (
            "motion floors are derived from the candidate's static-background "
            "Farneback-flow noise; no failing interaction frame fits its own threshold"
        ),
    }
    grasp = np.asarray(
        [bool(row["projected_contact"] and row["measurement_valid"]) for row in rows]
    )
    hand_speed = np.asarray([row["hand_motion_p90"] for row in rows])
    stem_speed = np.asarray([row["flower_motion_p90"] for row in rows])
    causal = causal_motion_audit(
        np,
        grasp_active=grasp,
        hand_speed=hand_speed,
        stem_speed=stem_speed,
        hand_motion_floor=floors["hand_motion"],
        stem_motion_floor=floors["stem_motion"],
        maximum_response_lag_frames=args.maximum_response_lag_frames,
        maximum_frozen_run_frames=args.maximum_frozen_run_frames,
    )
    causal["evidence_scope"] = (
        "projected-contact visual coupling proxy; cannot establish grasp or physical causality"
    )
    frame_contract = InteractionFrameContract(
        camera_frame=args.target_frame_name,
        metric_frame="camera:metric_unavailable",
        timeline=f"absolute_frame_index:full_source_{args.expected_frames}",
        fps=args.fps,
    )
    metric_contact = _metric_contact_assessment(np, optional["metric_contact"], frame_contract)
    kinematics = _kinematic_assessment(np, optional["hand_kinematics"])
    adversarial = _adversarial_audit(
        np, grasp, hand_speed, stem_speed, floors, args
    )

    frame_metrics = output / "frame-metrics.json"
    frame_metrics.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    active_stem_summary = _active_stem_summary(np, active_stems)
    packages = {}
    for package in ("numpy", "opencv-python"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    gates = {
        "articulated_metric_hand": bool(kinematics["passed"]),
        "metric_force_closure": bool(metric_contact["passed"]),
        "causal_flower_motion_proxy": bool(causal["passed"]),
        "instance_identity_available_for_all_interactions": False,
        "adversarial_audit": bool(adversarial["all_attacks_detected"]),
    }
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if all(gates.values()) else "PARTIAL",
        "honest_scope": (
            "Full-video visual dynamics audit plus fail-closed physical-contact preflight; "
            "2-D flow and adjacency are not depth or force evidence"
        ),
        "command": [sys.executable, *sys.argv],
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "git": _git_state(),
        "config": {
            key: [str(item) for item in value]
            if isinstance(value, list)
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        },
        "coordinate_contract": frame_contract.to_dict(),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "optional_inputs": {
            name: (
                {"path": str(path), "sha256": _sha256(path)} if path else None
            )
            for name, path in optional.items()
        },
        "video": {
            "frames": args.expected_frames,
            "fps": args.fps,
            "seconds": args.expected_frames / args.fps,
            "width": width,
            "height": height,
        },
        "motion_measurement_floors": floors,
        "visual_causal_motion": causal,
        "articulated_hand": kinematics,
        "metric_contact": metric_contact,
        "active_stem_tracks": active_stem_summary,
        "gates": gates,
        "adversarial": adversarial,
        "performance": {
            "wall_seconds": wall_seconds,
            "frames_per_second": args.expected_frames / wall_seconds,
            "realtime_factor": wall_seconds / (args.expected_frames / args.fps),
        },
        "outputs": {
            "frame_metrics": {
                "path": str(frame_metrics),
                "sha256": _sha256(frame_metrics),
            }
        },
        "limitations": [
            "The union flower track cannot assign every interaction to one persistent stem.",
            "Optical flow measures visible image motion and may be affected by occlusion or compression.",
            "Physical contact remains rejected without calibrated metric depth and force evidence.",
            "A monocular source cannot by itself prove force closure; a reconstruction plus simulator or RGB-D/tactile capture is required.",
        ],
    }
    report_path = output / "audit-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
