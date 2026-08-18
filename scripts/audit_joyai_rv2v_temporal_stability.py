#!/usr/bin/env python3
"""Compare a JoyAI RV2V challenger with its exact input on real video frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-video", type=Path, required=True)
    parser.add_argument("--challenger-video", type=Path, required=True)
    parser.add_argument("--robot-masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--evaluation-width", type=int, default=320)
    parser.add_argument("--evaluation-height", type=int, default=180)
    parser.add_argument("--foreground-erosion-pixels", type=int, default=2)
    parser.add_argument("--background-dilation-pixels", type=int, default=5)
    parser.add_argument("--minimum-motion-energy", type=float, default=0.5)
    parser.add_argument(
        "--joyai-blend-alpha",
        type=float,
        default=1.0,
        help="Evaluate a uniform source-anchored blend with this JoyAI contribution.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    def capture(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    status = capture("status", "--short")
    return {
        "head": capture("rev-parse", "HEAD"),
        "branch": capture("branch", "--show-current"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _probe(cv2: Any, path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        return {
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frames": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        }
    finally:
        capture.release()


def _load_masks(np: Any, path: Path, expected_frames: int) -> tuple[Any, int, int]:
    payload = np.load(path, allow_pickle=False)
    packed = payload["packed"]
    height = int(payload["height"])
    width = int(payload["width"])
    bitorder = str(payload["bitorder"])
    if packed.shape[0] != expected_frames:
        raise ValueError(f"mask frame count {packed.shape[0]} != {expected_frames}")
    masks = np.unpackbits(packed, axis=1, bitorder=bitorder)[:, : height * width]
    return masks.reshape(expected_frames, height, width).astype(np.uint8), width, height


def _project_mask(cv2: Any, np: Any, mask: Any, width: int, height: int) -> Any:
    if mask.shape != (480, 832):
        raise ValueError(f"expected legacy 832x480 mask, observed {mask.shape[::-1]}")
    canvas = np.zeros((480, 854), dtype=np.uint8)
    canvas[:, 11:843] = mask
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_NEAREST)


def _summary(np: Any, values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _masked_mean(np: Any, values: Any, mask: Any) -> float:
    if not np.any(mask):
        raise ValueError("metric mask is empty")
    return float(np.mean(values[mask]))


def _flow_aligned_previous(cv2: Any, np: Any, previous: Any, current: Any) -> Any:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    current_to_previous = cv2.calcOpticalFlowFarneback(
        current_gray,
        previous_gray,
        None,
        0.5,
        4,
        21,
        4,
        7,
        1.5,
        0,
    )
    grid_x, grid_y = np.meshgrid(
        np.arange(current.shape[1], dtype=np.float32),
        np.arange(current.shape[0], dtype=np.float32),
    )
    return cv2.remap(
        previous,
        grid_x + current_to_previous[..., 0],
        grid_y + current_to_previous[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _laplacian_energy(cv2: Any, np: Any, frame: Any, mask: Any) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    return _masked_mean(np, laplacian, mask)


def main() -> int:
    args = _parser().parse_args()
    import cv2
    import numpy as np

    incumbent = args.incumbent_video.expanduser().resolve()
    challenger = args.challenger_video.expanduser().resolve()
    mask_path = args.robot_masks.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for path in (incumbent, challenger, mask_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    if not 0.0 <= args.joyai_blend_alpha <= 1.0:
        raise ValueError("JoyAI blend alpha must be in [0, 1]")
    output.mkdir(parents=True)

    probes = {"incumbent": _probe(cv2, incumbent), "challenger": _probe(cv2, challenger)}
    for name, probe in probes.items():
        if probe["frames"] != args.expected_frames or abs(probe["fps"] - args.fps) > 0.01:
            raise ValueError(f"{name} violates timeline contract: {probe}")
        if (probe["width"], probe["height"]) != (1280, 720):
            raise ValueError(f"{name} must be 1280x720: {probe}")

    masks, mask_width, mask_height = _load_masks(np, mask_path, args.expected_frames)
    if (mask_width, mask_height) != (832, 480):
        raise ValueError(f"unexpected packed mask frame: {(mask_width, mask_height)}")
    kernel_erode = np.ones(
        (2 * args.foreground_erosion_pixels + 1,) * 2, dtype=np.uint8
    )
    kernel_dilate = np.ones(
        (2 * args.background_dilation_pixels + 1,) * 2, dtype=np.uint8
    )
    captures = {
        "incumbent": cv2.VideoCapture(str(incumbent)),
        "challenger": cv2.VideoCapture(str(challenger)),
    }
    values: dict[str, list[float]] = {
        "incumbent_jitter": [],
        "challenger_jitter": [],
        "incumbent_motion": [],
        "challenger_motion": [],
        "incumbent_sharpness": [],
        "challenger_sharpness": [],
        "background_mae": [],
        "full_frame_mae": [],
    }
    cadence: dict[str, dict[str, list[float]]] = {
        "modulo_3": {str(index): [] for index in range(3)},
        "causal_modulo_8": {str(index): [] for index in range(8)},
    }
    rows = []
    previous: dict[str, Any] | None = None
    started = time.perf_counter()
    try:
        for index in range(args.expected_frames):
            frames: dict[str, Any] = {}
            for name, capture in captures.items():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"{name} decode stopped at frame {index}")
                frames[name] = cv2.resize(
                    frame,
                    (args.evaluation_width, args.evaluation_height),
                    interpolation=cv2.INTER_AREA,
                )
            if args.joyai_blend_alpha < 1.0:
                frames["challenger"] = cv2.addWeighted(
                    frames["challenger"],
                    args.joyai_blend_alpha,
                    frames["incumbent"],
                    1.0 - args.joyai_blend_alpha,
                    0.0,
                )
            robot = _project_mask(
                cv2, np, masks[index], args.evaluation_width, args.evaluation_height
            )
            foreground = cv2.erode(robot, kernel_erode) > 0
            background = cv2.dilate(robot, kernel_dilate) == 0
            difference = np.abs(
                frames["challenger"].astype(np.float32)
                - frames["incumbent"].astype(np.float32)
            )
            background_mae = _masked_mean(np, difference, background)
            full_mae = float(np.mean(difference))
            values["background_mae"].append(background_mae)
            values["full_frame_mae"].append(full_mae)
            incumbent_sharpness = _laplacian_energy(
                cv2, np, frames["incumbent"], foreground
            )
            challenger_sharpness = _laplacian_energy(
                cv2, np, frames["challenger"], foreground
            )
            values["incumbent_sharpness"].append(incumbent_sharpness)
            values["challenger_sharpness"].append(challenger_sharpness)

            row: dict[str, Any] = {
                "frame": index,
                "background_mae": background_mae,
                "full_frame_mae": full_mae,
                "incumbent_sharpness": incumbent_sharpness,
                "challenger_sharpness": challenger_sharpness,
            }
            if previous is not None:
                aligned = {
                    name: _flow_aligned_previous(
                        cv2, np, previous[name], frames["incumbent"]
                    )
                    for name in ("incumbent", "challenger")
                }
                for name in ("incumbent", "challenger"):
                    temporal = np.mean(
                        np.abs(frames[name].astype(np.float32) - aligned[name].astype(np.float32)),
                        axis=2,
                    )
                    raw_motion = np.mean(
                        np.abs(
                            frames[name].astype(np.float32)
                            - previous[name].astype(np.float32)
                        ),
                        axis=2,
                    )
                    jitter = _masked_mean(np, temporal, foreground)
                    motion = _masked_mean(np, raw_motion, foreground)
                    values[f"{name}_jitter"].append(jitter)
                    values[f"{name}_motion"].append(motion)
                    row[f"{name}_jitter"] = jitter
                    row[f"{name}_motion"] = motion
                cadence["modulo_3"][str(index % 3)].append(row["challenger_jitter"])
                cadence["causal_modulo_8"][str((index - 1) % 8)].append(
                    row["challenger_jitter"]
                )
            rows.append(row)
            previous = frames
    finally:
        for capture in captures.values():
            capture.release()

    elapsed = time.perf_counter() - started
    incumbent_jitter = _summary(np, values["incumbent_jitter"])
    challenger_jitter = _summary(np, values["challenger_jitter"])
    incumbent_motion = _summary(np, values["incumbent_motion"])
    challenger_motion = _summary(np, values["challenger_motion"])
    incumbent_sharpness = _summary(np, values["incumbent_sharpness"])
    challenger_sharpness = _summary(np, values["challenger_sharpness"])
    motion_ratio = challenger_motion["mean"] / max(incumbent_motion["mean"], 1e-9)
    sharpness_ratio = challenger_sharpness["mean"] / max(
        incumbent_sharpness["mean"], 1e-9
    )
    reduction = 1.0 - challenger_jitter["mean"] / max(incumbent_jitter["mean"], 1e-9)
    freeze_count = sum(
        value < args.minimum_motion_energy for value in values["challenger_motion"]
    )
    report = {
        "schema_version": "1.0.0",
        "status": "PARTIAL",
        "stage": "joyai_0811_rv2v_temporal_stability_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "git": _git_state(),
        "inputs": {
            "incumbent": {"path": str(incumbent), "sha256": _sha256(incumbent)},
            "challenger": {"path": str(challenger), "sha256": _sha256(challenger)},
            "robot_masks": {"path": str(mask_path), "sha256": _sha256(mask_path)},
        },
        "probes": probes,
        "evaluation_transform": {
            "kind": "uniform_source_anchored_linear_blend",
            "joyai_alpha": args.joyai_blend_alpha,
            "incumbent_alpha": 1.0 - args.joyai_blend_alpha,
            "frame_adaptive": False,
        },
        "metrics": {
            "flow_aligned_robot_jitter": {
                "incumbent": incumbent_jitter,
                "challenger": challenger_jitter,
                "relative_mean_reduction": reduction,
            },
            "robot_motion_energy": {
                "incumbent": incumbent_motion,
                "challenger": challenger_motion,
                "challenger_to_incumbent_ratio": motion_ratio,
                "challenger_low_motion_transition_count": freeze_count,
            },
            "robot_sharpness": {
                "incumbent": incumbent_sharpness,
                "challenger": challenger_sharpness,
                "challenger_to_incumbent_ratio": sharpness_ratio,
            },
            "background_mae_8bit": _summary(np, values["background_mae"]),
            "full_frame_mae_8bit": _summary(np, values["full_frame_mae"]),
            "cadence": {
                group: {phase: _summary(np, phase_values) for phase, phase_values in phases.items()}
                for group, phases in cadence.items()
            },
            "processing_wall_seconds": elapsed,
            "processing_fps": args.expected_frames / elapsed,
        },
        "acceptance": {
            "temporal_mean_improved": reduction > 0.0,
            "temporal_p95_improved": challenger_jitter["p95"] < incumbent_jitter["p95"],
            "motion_retained_80_to_120_percent": 0.8 <= motion_ratio <= 1.2,
            "sharpness_retained_at_least_90_percent": sharpness_ratio >= 0.9,
            "background_mean_mae_at_most_8": _summary(np, values["background_mae"])[
                "mean"
            ]
            <= 8.0,
            "no_near_frozen_transitions": freeze_count == 0,
        },
        "limitations": [
            "The optical-flow and packed-mask metrics are image-space proxies, not physical evidence.",
            "A human native-resolution review remains a veto for robot identity, hands, flowers, and contact.",
        ],
        "physical_evidence": False,
    }
    report["acceptance"]["automatic_gates_pass"] = all(report["acceptance"].values())
    (output / "per-frame-metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "audit-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
