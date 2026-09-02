#!/usr/bin/env python3
"""Replace reviewed RM65 gripper events without modifying arm trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-npz", type=Path, required=True)
    parser.add_argument("--event-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def event_command(
    frame_count: int,
    frames: np.ndarray,
    closed_01: np.ndarray,
    gaussian_sigma_frames: float,
) -> np.ndarray:
    frames = np.asarray(frames, dtype=int)
    closed_01 = np.asarray(closed_01, dtype=np.float64)
    if frames.ndim != 1 or closed_01.shape != frames.shape:
        raise ValueError("event frames and closed_01 values must be matching 1-D arrays")
    if frames[0] != 0 or frames[-1] != frame_count - 1:
        raise ValueError("event frames must include the first and final state frames")
    if np.any(np.diff(frames) <= 0):
        raise ValueError("event frames must be strictly increasing")
    if np.any((closed_01 < 0.0) | (closed_01 > 1.0)):
        raise ValueError("closed_01 values must lie in [0, 1]")
    if gaussian_sigma_frames < 0:
        raise ValueError("gaussian_sigma_frames must be non-negative")
    command = np.interp(np.arange(frame_count), frames, closed_01)
    if gaussian_sigma_frames:
        command = gaussian_filter1d(
            command, sigma=gaussian_sigma_frames, mode="nearest"
        )
    command[frames] = closed_01
    return np.clip(command, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    config = json.loads(args.event_config.read_text())
    with np.load(args.state_npz) as loaded:
        payload = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
    frame_count = len(payload["left_q"])
    manifest: dict[str, object] = {
        "schema_version": "phiagent-rm65-gripper-event-refinement/1.0",
        "input_state": str(args.state_npz),
        "event_config": str(args.event_config),
        "frames": frame_count,
        "sides": {},
    }
    for side in ("left", "right"):
        key = f"{side}_gripper_command"
        if side not in config:
            manifest["sides"][side] = {"mode": "preserve"}
            continue
        specification = config[side]
        original = np.asarray(payload[key], dtype=np.float64)
        command = event_command(
            frame_count,
            np.asarray(specification["frames"]),
            np.asarray(specification["closed_01"]),
            float(specification.get("gaussian_sigma_frames", 0.0)),
        )
        payload[key] = command
        crossings = (np.flatnonzero(np.diff(command >= 0.5)) + 1).tolist()
        manifest["sides"][side] = {
            "mode": "reviewed_events",
            "half_command_crossing_frames": crossings,
            "max_abs_delta_from_input": float(np.abs(command - original).max()),
            "changed_frame_count": int(np.count_nonzero(np.abs(command - original) > 1e-6)),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "rm65_synchronized_state.npz"
    np.savez_compressed(state_path, **payload)
    manifest["output_state"] = str(state_path)
    manifest["claim_boundary"] = (
        "manually reviewed image-space gripper events; not gripper encoder, contact, "
        "force or cloth-attachment ground truth"
    )
    (args.output_dir / "gripper_event_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
