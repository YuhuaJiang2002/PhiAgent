#!/usr/bin/env python3
"""Compose reviewed left/right RM65 trajectories without re-solving accepted sides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-state", type=Path, required=True)
    parser.add_argument("--left-state", type=Path, required=True)
    parser.add_argument("--right-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left-roll-smoothing-window", type=int, default=1)
    parser.add_argument("--right-roll-smoothing-window", type=int, default=1)
    return parser.parse_args()


def triangular_smooth(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if window < 1 or window % 2 == 0:
        raise ValueError("roll smoothing window must be a positive odd integer")
    if window == 1:
        return values.copy(), np.asarray((1.0,), dtype=np.float64)
    radius = window // 2
    kernel = np.concatenate(
        (
            np.arange(1, radius + 2, dtype=np.float64),
            np.arange(radius, 0, -1, dtype=np.float64),
        )
    )
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid"), kernel


def compose_states(
    base: np.lib.npyio.NpzFile,
    left: np.lib.npyio.NpzFile,
    right: np.lib.npyio.NpzFile,
    left_window: int,
    right_window: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    required = {"left_q", "right_q", "left_target_xyz", "right_target_xyz", "fps"}
    for label, state in (("base", base), ("left", left), ("right", right)):
        missing = required.difference(state.files)
        if missing:
            raise ValueError(f"{label} state is missing arrays: {sorted(missing)}")
    frames = len(base["left_q"])
    for label, state in (("left", left), ("right", right)):
        if len(state["left_q"]) != frames or len(state["right_q"]) != frames:
            raise ValueError(f"{label} state frame count differs from the base state")
        if not np.isclose(float(state["fps"]), float(base["fps"])):
            raise ValueError(f"{label} state FPS differs from the base state")

    payload = {key: np.asarray(base[key]).copy() for key in base.files}
    payload["left_q"] = np.asarray(left["left_q"], dtype=np.float64).copy()
    payload["right_q"] = np.asarray(right["right_q"], dtype=np.float64).copy()
    payload["left_target_xyz"] = np.asarray(left["left_target_xyz"]).copy()
    payload["right_target_xyz"] = np.asarray(right["right_target_xyz"]).copy()
    left_raw = payload["left_q"][:, 5].copy()
    right_raw = payload["right_q"][:, 5].copy()
    payload["left_q"][:, 5], left_kernel = triangular_smooth(left_raw, left_window)
    payload["right_q"][:, 5], right_kernel = triangular_smooth(right_raw, right_window)
    manifest = {
        "schema_version": "phiagent-rm65-composed-state/1.0",
        "frames": frames,
        "fps": float(payload["fps"]),
        "roll_smoothing": {
            "left": {
                "window_frames": left_window,
                "kernel": left_kernel.tolist(),
                "max_abs_delta_rad": float(
                    np.abs(payload["left_q"][:, 5] - left_raw).max()
                ),
            },
            "right": {
                "window_frames": right_window,
                "kernel": right_kernel.tolist(),
                "max_abs_delta_rad": float(
                    np.abs(payload["right_q"][:, 5] - right_raw).max()
                ),
            },
        },
    }
    return payload, manifest


def main() -> None:
    args = parse_args()
    with (
        np.load(args.base_state) as base,
        np.load(args.left_state) as left,
        np.load(args.right_state) as right,
    ):
        payload, manifest = compose_states(
            base,
            left,
            right,
            args.left_roll_smoothing_window,
            args.right_roll_smoothing_window,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "rm65_synchronized_state.npz"
    np.savez_compressed(state_path, **payload)
    manifest.update(
        {
            "base_state": str(args.base_state),
            "left_state": str(args.left_state),
            "right_state": str(args.right_state),
            "state_npz": str(state_path),
        }
    )
    manifest_path = args.output_dir / "compose_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
