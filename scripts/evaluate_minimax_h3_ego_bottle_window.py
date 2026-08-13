#!/usr/bin/env python3
"""Evaluate and conservatively background-lock one Ego bottle H3 window."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.training.ego_repair_policy import (  # noqa: E402
    EgoNonRegressionContract,
    EgoRepairPolicy,
)


REPAIR_RECIPES = (
    {"name": "raw-h3", "support_dilation_pixels": 0, "alpha_blur_sigma": 0.0},
    {
        "name": "tight-control-support-lock",
        "support_dilation_pixels": 0,
        "alpha_blur_sigma": 3.0,
    },
    {
        "name": "soft-control-support-lock",
        "support_dilation_pixels": 0,
        "alpha_blur_sigma": 7.0,
    },
    {
        "name": "dilated-control-support-lock",
        "support_dilation_pixels": 12,
        "alpha_blur_sigma": 7.0,
    },
    {
        "name": "wide-soft-control-support-lock",
        "support_dilation_pixels": 20,
        "alpha_blur_sigma": 9.0,
    },
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


def _decode(cv2: Any, path: Path) -> tuple[list[Any], dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    info: dict[str, float | int] = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"decoded no frames from {path}")
    info["frames"] = len(frames)
    return frames, info


def _resize(cv2: Any, frames: list[Any], width: int, height: int) -> list[Any]:
    if frames[0].shape[:2] == (height, width):
        return frames
    return [
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]


def action_support_mask(cv2: Any, np: Any, source: Any, control: Any) -> Any:
    """Return the declared camera-pixel support changed by the action control."""

    difference = np.max(
        np.abs(control.astype(np.float32) - source.astype(np.float32)), axis=2
    )
    # The control adapter deliberately softens the source human hands.  A low
    # threshold turns that broad cleanup halo into action support and lets the
    # generated scene overwrite half the frame.  The stronger threshold and
    # bounded kernels retain the explicit robot/bottle paint while excluding
    # the low-amplitude cleanup field.
    mask = (difference >= 32.0).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    return cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )


def _skin_mask(cv2: Any, frame: Any) -> Any:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(ycrcb, (0, 134, 80), (255, 180, 132))
    return cv2.bitwise_and(
        skin,
        cv2.inRange(hsv, (0, 35, 40), (25, 210, 255)),
    )


def _blue_mask(cv2: Any, frame: Any) -> Any:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (82, 70, 25), (132, 255, 255))


def motion_adherence_score(
    np: Any,
    control: list[Any],
    candidate: list[Any],
    masks: list[Any],
) -> tuple[float, dict[str, float]]:
    """Compare temporal change magnitudes only inside declared action support."""

    reference_motion = []
    candidate_motion = []
    for index in range(1, len(control)):
        active = (masks[index] > 0) | (masks[index - 1] > 0)
        if not np.any(active):
            continue
        reference_delta = np.abs(
            control[index].astype(np.float32) - control[index - 1].astype(np.float32)
        )
        candidate_delta = np.abs(
            candidate[index].astype(np.float32) - candidate[index - 1].astype(np.float32)
        )
        reference_motion.append(float(reference_delta[active].mean()))
        candidate_motion.append(float(candidate_delta[active].mean()))
    if not reference_motion:
        return 0.0, {"reference_motion": 0.0, "candidate_motion": 0.0}
    reference = np.asarray(reference_motion, dtype=np.float64)
    observed = np.asarray(candidate_motion, dtype=np.float64)
    error = float(np.mean(np.abs(reference - observed)))
    scale = float(np.mean(reference)) + 5.0
    score = float(np.exp(-error / scale))
    return score, {
        "reference_motion": float(reference.mean()),
        "candidate_motion": float(observed.mean()),
        "mean_absolute_motion_error": error,
    }


def apply_repair(
    cv2: Any,
    np: Any,
    source: list[Any],
    raw: list[Any],
    masks: list[Any],
    recipe: dict[str, object],
) -> list[Any]:
    """Apply one bounded camera-frame repair recipe."""

    if recipe["name"] == "raw-h3":
        return [frame.copy() for frame in raw]
    dilation = int(recipe["support_dilation_pixels"])
    blur_sigma = float(recipe["alpha_blur_sigma"])
    kernel = (
        None
        if dilation == 0
        else cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1)
        )
    )
    result = []
    for source_frame, raw_frame, mask in zip(source, raw, masks):
        active = mask if kernel is None else cv2.dilate(mask, kernel)
        alpha = (
            active.astype(np.float32) / 255.0
            if blur_sigma == 0
            else cv2.GaussianBlur(active, (0, 0), blur_sigma).astype(np.float32)
            / 255.0
        )
        result.append(
            np.rint(
                raw_frame.astype(np.float32) * alpha[..., None]
                + source_frame.astype(np.float32) * (1.0 - alpha[..., None])
            ).astype(np.uint8)
        )
    return result


def score_candidate(
    cv2: Any,
    np: Any,
    source: list[Any],
    control: list[Any],
    candidate: list[Any],
    masks: list[Any],
) -> tuple[dict[str, float], dict[str, object]]:
    """Score one raw or repaired candidate against the declared Ego control."""

    source_skin = final_skin = 0
    expected_blue = retained_blue = 0
    expected_metal = retained_metal = 0
    background_errors = []
    active_fractions = []
    for source_frame, control_frame, final_frame, mask in zip(
        source, control, candidate, masks
    ):
        active = mask > 0
        active_fractions.append(float(np.mean(active)))
        source_skin_mask = (_skin_mask(cv2, source_frame) > 0) & active
        final_skin_mask = (_skin_mask(cv2, final_frame) > 0) & active
        source_skin += int(source_skin_mask.sum())
        final_skin += int(final_skin_mask.sum())

        blue = (_blue_mask(cv2, control_frame) > 0) & active
        blue_dilated = cv2.dilate(
            blue.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        ) > 0
        final_blue = _blue_mask(cv2, final_frame) > 0
        expected_blue += int(blue.sum())
        retained_blue += int((final_blue & blue_dilated).sum())

        control_hsv = cv2.cvtColor(control_frame, cv2.COLOR_BGR2HSV)
        final_hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        changed = np.max(
            np.abs(
                control_frame.astype(np.float32) - source_frame.astype(np.float32)
            ),
            axis=2,
        ) >= 16.0
        metal = (
            (control_hsv[..., 1] <= 55)
            & (control_hsv[..., 2] >= 145)
            & (control_hsv[..., 2] <= 245)
            & changed
        )
        final_metal = (final_hsv[..., 1] <= 75) & (final_hsv[..., 2] >= 80)
        expected_metal += int(metal.sum())
        retained_metal += int((final_metal & metal).sum())

        safety = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        )
        background = safety == 0
        if np.any(background):
            difference = np.abs(
                final_frame.astype(np.float32) - source_frame.astype(np.float32)
            )
            background_errors.append(float(difference[background].mean()))

    subject_replacement = float(
        np.clip(1.0 - final_skin / max(1, source_skin), 0.0, 1.0)
    )
    object_lock = float(np.clip(retained_blue / max(1, expected_blue), 0.0, 1.0))
    robot_identity = float(
        np.clip(retained_metal / max(1, expected_metal), 0.0, 1.0)
    )
    background_mae = float(np.mean(background_errors))
    background_lock = float(np.exp(-background_mae / 5.0))
    motion_preservation, motion_detail = motion_adherence_score(
        np, control, candidate, masks
    )
    temporal_consistency = float(
        np.exp(
            -abs(
                motion_detail["candidate_motion"]
                - motion_detail["reference_motion"]
            )
            / (motion_detail["reference_motion"] + 8.0)
        )
    )
    scorecard = {
        "background_lock": background_lock,
        "object_lock": object_lock,
        "subject_replacement": subject_replacement,
        "robot_identity": robot_identity,
        "motion_preservation": motion_preservation,
        "temporal_consistency": temporal_consistency,
        "epl_minimum": min(
            object_lock, subject_replacement, robot_identity, motion_preservation
        ),
    }
    metrics: dict[str, object] = {
        "background_mae_outside_support": background_mae,
        "mean_action_support_fraction": float(np.mean(active_fractions)),
        "source_skin_pixels_in_support": source_skin,
        "final_skin_pixels_in_support": final_skin,
        "expected_blue_pixels": expected_blue,
        "retained_blue_pixels": retained_blue,
        "expected_metal_pixels": expected_metal,
        "retained_metal_pixels": retained_metal,
        "motion": motion_detail,
    }
    return scorecard, metrics


def _write_video(ffmpeg: Path, output: Path, frames: list[Any], fps: float) -> None:
    height, width = frames[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt",
            "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "-",
            "-an", "-c:v", "libx264", "-crf", "14", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed to encode {output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--raw-h3", type=Path, required=True)
    parser.add_argument("--robot-reference", type=Path, required=True)
    parser.add_argument("--action-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repair-policy",
        type=Path,
        help="Optional domain-matched Ego repair-router checkpoint.",
    )
    parser.add_argument(
        "--human-review", choices=("pending", "passed", "failed"), default="pending"
    )
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = {
        "source": args.source.expanduser().resolve(),
        "control": args.control.expanduser().resolve(),
        "raw_h3": args.raw_h3.expanduser().resolve(),
        "robot_reference": args.robot_reference.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")
    output_dir = args.output_dir.expanduser().resolve()
    evolution_path = output_dir / "evolution.json"
    if evolution_path.exists():
        raise FileExistsError(f"Ego evaluation already exists: {evolution_path}")

    import cv2
    import numpy as np

    raw, raw_info = _decode(cv2, paths["raw_h3"])
    source, source_info = _decode(cv2, paths["source"])
    control, control_info = _decode(cv2, paths["control"])
    if len({len(raw), len(source), len(control)}) != 1:
        raise ValueError("source, control and H3 window frame counts differ")
    if abs(float(raw_info["fps"]) - float(source_info["fps"])) > 1e-6:
        raise ValueError("source and H3 FPS differ")
    height, width = raw[0].shape[:2]
    source = _resize(cv2, source, width, height)
    control = _resize(cv2, control, width, height)
    masks = [
        action_support_mask(cv2, np, source_frame, control_frame)
        for source_frame, control_frame in zip(source, control)
    ]

    policy_path = (
        args.repair_policy.expanduser().resolve() if args.repair_policy else None
    )
    if policy_path is not None and (
        not policy_path.is_file() or policy_path.stat().st_size == 0
    ):
        raise ValueError(f"repair policy is missing or empty: {policy_path}")
    policy = EgoRepairPolicy.load(policy_path) if policy_path else None
    contract = (
        policy.non_regression_contract if policy else EgoNonRegressionContract()
    )
    rounds = []
    raw_scorecard: dict[str, float] | None = None
    for index, recipe in enumerate(REPAIR_RECIPES):
        candidate = apply_repair(cv2, np, source, raw, masks, dict(recipe))
        scorecard, metrics = score_candidate(
            cv2, np, source, control, candidate, masks
        )
        if index == 0:
            raw_scorecard = scorecard
        assert raw_scorecard is not None
        non_regression = contract.assess(raw_scorecard, scorecard)
        round_path = output_dir / "rounds" / f"{index:02d}-{recipe['name']}.mp4"
        _write_video(paths["ffmpeg"], round_path, candidate, float(raw_info["fps"]))
        rounds.append(
            {
                "index": index,
                "repair": dict(recipe),
                "scorecard": scorecard,
                "metrics": metrics,
                "non_regression": non_regression,
                "output": str(round_path),
                "output_sha256": _sha256(round_path),
            }
        )
        del candidate

    predicted = []
    if policy is None:
        selected = next(
            record
            for record in rounds
            if record["repair"]["name"] == "tight-control-support-lock"
        )
        learned_policy = None
    else:
        repair_records = list(rounds)
        by_name = {
            str(record["repair"]["name"]): record for record in repair_records
        }
        ranked = policy.rank(
            raw_scorecard, [record["repair"] for record in repair_records]
        )
        for recipe, prediction in ranked:
            record = by_name[str(recipe["name"])]
            predicted.append(
                {
                    "repair": recipe["name"],
                    "predicted_constrained_utility": prediction,
                    "non_regression": record["non_regression"],
                }
            )
        selected = next(
            (
                by_name[str(recipe["name"])]
                for recipe, _ in ranked
                if bool(by_name[str(recipe["name"])]["non_regression"]["passed"])
            ),
            rounds[0],
        )
        learned_policy = {
            "checkpoint": str(policy_path),
            "checkpoint_sha256": _sha256(policy_path),
            "method": "held_action_epic_ego_bottle_repair_router",
            "ranked_candidates": predicted,
            "raw_is_ranked_and_remains_the_non_regressing_fallback": True,
        }

    scorecard = selected["scorecard"]
    metrics = selected["metrics"]
    thresholds = {
        "background_lock": 0.95,
        "object_lock": 0.45,
        "subject_replacement": 0.70,
        "robot_identity": 0.55,
        "motion_preservation": 0.50,
        "temporal_consistency": 0.65,
        "epl_minimum": 0.45,
    }
    gates = {key: scorecard[key] >= value for key, value in thresholds.items()}
    gates["human_review"] = args.human_review == "passed"
    accepted = all(gates.values())
    final_path = output_dir / "final-background-locked.mp4"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected["output"], final_path)
    evolution = {
        "schema_version": "1.0.0",
        "method": "epic_ego_bottle_repair_tournament_and_proxy_gates",
        "status": "accepted" if accepted else "rejected",
        "honest_status": "WORKING" if accepted else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu": {"used": False, "reason": "deterministic post-generation evaluation"},
        "action_label": args.action_label,
        "coordinate_frame": "camera:H3_output_pixels aligned to EPIC-KITCHENS frame index",
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in paths.items()
        },
        "video_info": {
            "raw": raw_info,
            "source": source_info,
            "control": control_info,
        },
        "best_round": selected["index"],
        "best_repair": selected["repair"]["name"],
        "best_scorecard": scorecard,
        "best_source_face_replacement": scorecard["subject_replacement"],
        "learned_repair_policy": learned_policy,
        "rounds": rounds,
        "metrics": metrics,
        "thresholds": thresholds,
        "gates": gates,
        "human_review": args.human_review,
        "output": {"path": str(final_path), "sha256": _sha256(final_path)},
        "limitations": [
            "Blue-color, skin-color, metallic-material and motion measurements are image-space proxies, not object tracking, force or contact sensing.",
            "Background locking is bounded to control-derived support and does not train or fine-tune H3.",
            "The optional learned module only ranks four bounded repair recipes and falls back to raw H3 if every repair violates the non-regression contract.",
            "A WORKING claim additionally requires explicit human review and long-window seam review.",
        ],
    }
    _write_json(evolution_path, evolution)
    print(json.dumps({"output": str(output_dir), "status": evolution["status"], "gates": gates}, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
