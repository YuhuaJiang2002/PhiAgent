#!/usr/bin/env python3
"""Repair local temporal jerk while locking accepted flower/contact geometry.

The trained flower repair policy chooses the repair contract.  Pixel repair is
then limited to source-unsupported second-order changes inside the original
character mask.  Named flower pixels, a narrow hand/flower contact band, and
the background remain exact before encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.evaluation.object_instance import NormalizedROI  # noqa: E402
from phiagent.evaluation.video_proxy import evaluate_local_videos, resolve_ffmpeg  # noqa: E402
from phiagent.training.flower_repair_policy import FlowerRepairPolicy  # noqa: E402


@dataclass(frozen=True)
class RepairRound:
    name: str
    excess_threshold: float
    strength: float
    passes: int


DEFAULT_ROUNDS = (
    RepairRound("round-01-conservative", 12.0, 0.70, 1),
    RepairRound("round-02-balanced", 8.0, 0.80, 2),
    RepairRound("round-03-strong", 5.0, 0.90, 3),
    RepairRound("round-04-max-bounded", 3.0, 0.95, 4),
    RepairRound("round-05-max-bounded", 2.0, 1.00, 5),
    RepairRound("round-06-max-bounded", 1.0, 1.00, 8),
    RepairRound("round-07-max-bounded", 0.0, 1.00, 12),
    RepairRound("round-08-max-bounded", 0.0, 1.00, 20),
    RepairRound("round-09-max-bounded", 0.0, 1.00, 30),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--character-mask-video", type=Path, required=True)
    parser.add_argument("--flower-tracks", type=Path, required=True)
    parser.add_argument("--hand-tracks", type=Path, required=True)
    parser.add_argument("--aligned-evaluation", type=Path, required=True)
    parser.add_argument("--strict-gate-report", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--target-image", type=Path, required=True)
    parser.add_argument("--backend-metadata", type=Path, required=True)
    parser.add_argument("--object-roi", type=float, nargs=4, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-dilation", type=int, default=2)
    parser.add_argument("--mask-feather-sigma", type=float, default=1.5)
    parser.add_argument("--flower-dilation", type=int, default=2)
    parser.add_argument("--contact-band", type=int, default=12)
    parser.add_argument("--minimum-temporal", type=float, default=0.75)
    parser.add_argument("--minimum-motion", type=float, default=0.82)
    parser.add_argument("--minimum-recovery-baseline-motion", type=float, default=0.75)
    parser.add_argument("--maximum-motion-regression", type=float, default=0.01)
    parser.add_argument("--maximum-identity-regression", type=float, default=0.01)
    parser.add_argument(
        "--allow-aligned-motion-recovery",
        action="store_true",
        help=(
            "Allow bounded repair only when every strict gate except aligned motion/temporal "
            "already passes; the repaired candidate must still reach --minimum-motion."
        ),
    )
    parser.add_argument(
        "--only-round",
        choices=[repair_round.name for repair_round in DEFAULT_ROUNDS],
        help="Evaluate one named round in a new immutable experiment.",
    )
    parser.add_argument("--ffmpeg", type=Path)
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


def _decode(cv2: Any, path: Path, *, gray: bool = False) -> tuple[list[Any], dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if gray else frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video has no decoded frames: {path}")
    height, width = frames[0].shape[:2]
    return frames, {"frames": len(frames), "width": width, "height": height, "fps": fps}


def _fit_mask_frames(np: Any, masks: Any, frame_count: int) -> Any:
    """Allow only a missing terminal preprocessing mask, duplicated explicitly."""

    if len(masks) == frame_count:
        return masks
    if len(masks) == frame_count - 1:
        return np.concatenate([masks, masks[-1:]], axis=0)
    raise ValueError(f"mask frame count {len(masks)} cannot align to {frame_count} frames")


def repair_eligibility_mode(
    strict_gate: dict[str, Any],
    *,
    baseline_motion: float,
    allow_aligned_motion_recovery: bool,
    minimum_recovery_baseline_motion: float,
) -> str:
    """Return the fail-closed repair mode accepted by the strict report."""

    if strict_gate.get("geometry_all_gates_passed") is True:
        return "strict-geometry-temporal-only"
    if not allow_aligned_motion_recovery:
        raise RuntimeError("strict geometry gate is not accepted; temporal-only repair is forbidden")
    gates = strict_gate.get("gates")
    if not isinstance(gates, dict) or not gates:
        raise RuntimeError("strict report has no auditable gate mapping")
    failed = {key for key, value in gates.items() if value is not True}
    allowed_failures = {"aligned:motion", "aligned:temporal"}
    if "aligned:motion" not in failed or not failed <= allowed_failures:
        raise RuntimeError(
            "aligned-motion recovery requires every non-motion/non-temporal strict gate to pass"
        )
    if baseline_motion < minimum_recovery_baseline_motion:
        raise RuntimeError("aligned-motion recovery baseline is below its bounded floor")
    return "bounded-aligned-motion-and-temporal-recovery"


def _load_packed_union(np: Any, path: Path) -> tuple[Any, list[str]]:
    payload = np.load(path, allow_pickle=False)
    height = int(payload["height"])
    width = int(payload["width"])
    packed = payload["masks_packed"]
    unpacked = np.unpackbits(packed, axis=-1, bitorder=str(payload["bitorder"]))
    masks = unpacked[..., : height * width].reshape(
        packed.shape[0], packed.shape[1], height, width
    ).astype(bool)
    return np.any(masks, axis=0), payload["instance_ids"].astype(str).tolist()


def _dilate_masks(cv2: Any, np: Any, masks: Any, pixels: int) -> Any:
    if pixels == 0:
        return masks.copy()
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
    return np.stack([cv2.dilate(mask.astype(np.uint8), kernel) > 0 for mask in masks])


def build_protection_masks(
    cv2: Any,
    np: Any,
    flower_masks: Any,
    hand_masks: Any,
    *,
    flower_dilation: int,
    contact_band: int,
) -> tuple[Any, Any]:
    """Return hard protection and the auditable hand/flower contact band."""

    flowers = _dilate_masks(cv2, np, flower_masks, flower_dilation)
    flower_contact_neighborhood = _dilate_masks(cv2, np, flower_masks, contact_band)
    contact = flower_contact_neighborhood & hand_masks
    return flowers | contact, contact


def source_unsupported_alpha(
    cv2: Any,
    np: Any,
    candidate: list[Any],
    source: list[Any],
    editable: Any,
    protected: Any,
    *,
    excess_threshold: float,
    dilation_pixels: int,
    feather_sigma: float,
) -> Any:
    """Build per-frame alpha from candidate jerk exceeding source jerk."""

    count, height, width = editable.shape
    result = np.zeros((count, height, width), dtype=np.float32)
    kernel = np.ones((dilation_pixels * 2 + 1, dilation_pixels * 2 + 1), np.uint8)
    for index in range(1, count - 1):
        candidate_jerk = np.abs(
            candidate[index + 1].astype(np.float32)
            - 2.0 * candidate[index].astype(np.float32)
            + candidate[index - 1].astype(np.float32)
        ).mean(axis=2)
        source_jerk = np.abs(
            source[index + 1].astype(np.float32)
            - 2.0 * source[index].astype(np.float32)
            + source[index - 1].astype(np.float32)
        ).mean(axis=2)
        support = (candidate_jerk - source_jerk >= excess_threshold) & editable[index]
        support &= ~protected[index]
        support = cv2.dilate(support.astype(np.uint8), kernel) > 0
        alpha = cv2.GaussianBlur(support.astype(np.float32), (0, 0), feather_sigma)
        alpha[~editable[index] | protected[index]] = 0.0
        result[index] = np.clip(alpha, 0.0, 1.0)
    return result


def apply_local_crossfade_pass(
    np: Any,
    frames: list[Any],
    alpha: Any,
    protected: Any,
    *,
    strength: float,
) -> list[Any]:
    """Move only alpha-supported pixels toward the adjacent-frame midpoint."""

    original = [frame.copy() for frame in frames]
    result = [frame.copy() for frame in frames]
    for index in range(1, len(frames) - 1):
        weight = np.clip(alpha[index] * strength, 0.0, 1.0)[..., None]
        bridge = (
            original[index - 1].astype(np.float32)
            + original[index + 1].astype(np.float32)
        ) * 0.5
        composite = original[index].astype(np.float32) * (1.0 - weight) + bridge * weight
        result[index] = np.clip(np.rint(composite), 0, 255).astype(np.uint8)
        result[index][protected[index]] = original[index][protected[index]]
    return result


def _encode(ffmpeg: Path, frames: list[Any], output: Path, fps: float, *, lossless: bool) -> None:
    height, width = frames[0].shape[:2]
    codec = (
        ["-c:v", "libx264rgb", "-crf", "0", "-pix_fmt", "rgb24"]
        if lossless
        else ["-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p"]
    )
    process = subprocess.Popen(
        [
            str(ffmpeg), "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "-", "-an",
            *codec, "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
    if process.wait():
        raise RuntimeError(f"FFmpeg failed to encode {output}")


def _policy_ranking(policy: FlowerRepairPolicy, baseline: dict[str, float]) -> list[dict[str, Any]]:
    scorecard = {
        "background_lock": 1.0,
        "object_lock": 1.0,
        "subject_replacement": 1.0,
        "robot_identity": baseline["target_identity"],
        "motion_preservation": baseline["motion_preservation"],
        "temporal_consistency": baseline["temporal_consistency"],
        "epl_minimum": min(baseline["motion_preservation"], baseline["temporal_consistency"]),
    }
    recipes = [
        {"name": "no-temporal-repair", "hard_background_lock": 0, "restore_source_flowers": 0, "exclude_source_face_from_flower_restore": 0, "mask_dilation_pixels": 0, "flower_dilation_pixels": 0, "face_box_margin_pixels": 0},
        {"name": "global-noise-smoothing", "hard_background_lock": 0, "restore_source_flowers": 0, "exclude_source_face_from_flower_restore": 0, "mask_dilation_pixels": 3, "flower_dilation_pixels": 0, "face_box_margin_pixels": 0},
        {"name": "epl-local-crossfade", "hard_background_lock": 1, "restore_source_flowers": 1, "exclude_source_face_from_flower_restore": 1, "mask_dilation_pixels": 2, "flower_dilation_pixels": 2, "face_box_margin_pixels": 12},
        {"name": "epl-local-flow", "hard_background_lock": 1, "restore_source_flowers": 1, "exclude_source_face_from_flower_restore": 1, "mask_dilation_pixels": 3, "flower_dilation_pixels": 2, "face_box_margin_pixels": 12},
    ]
    return [
        {"recipe": dict(recipe), "predicted_constrained_utility": utility}
        for recipe, utility in policy.rank(scorecard, recipes)
    ]


def _evaluate(
    *,
    source: Path,
    target_image: Path,
    candidate: Path,
    ffmpeg: Path,
    object_roi: tuple[float, float, float, float],
) -> dict[str, float]:
    metrics = evaluate_local_videos(
        source=source,
        reference=source,
        target_image=target_image,
        candidate=candidate,
        ffmpeg=ffmpeg,
        object_roi=NormalizedROI(*object_roi),
    )
    return {key: float(value) for key, value in asdict(metrics).items()}


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    paths = {
        key: value.expanduser().resolve()
        for key, value in vars(args).items()
        if isinstance(value, Path)
    }
    for label, path in paths.items():
        if label in {"output_dir", "ffmpeg"}:
            continue
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")
    output.mkdir(parents=True)
    (output / "rounds").mkdir()
    (output / "provenance" / "execution-sources").mkdir(parents=True)

    import cv2
    import numpy as np

    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    candidate, candidate_info = _decode(cv2, paths["candidate"])
    source, source_info = _decode(cv2, paths["source"])
    mask_frames, mask_info = _decode(cv2, paths["character_mask_video"], gray=True)
    if len(candidate) != len(source) or candidate_info["width"] != source_info["width"] or candidate_info["height"] != source_info["height"]:
        raise ValueError("candidate and source videos must have identical frame geometry")
    if abs(float(candidate_info["fps"]) - float(source_info["fps"])) > 1e-6:
        raise ValueError("candidate and source FPS differ")
    character = _fit_mask_frames(np, np.stack(mask_frames) >= 127, len(candidate))
    flower_masks, flower_ids = _load_packed_union(np, paths["flower_tracks"])
    hand_masks, hand_ids = _load_packed_union(np, paths["hand_tracks"])
    if flower_masks.shape != character.shape or hand_masks.shape != character.shape:
        raise ValueError("tracked masks do not match video geometry")
    editable = _dilate_masks(cv2, np, character, args.mask_dilation)
    protected, contact = build_protection_masks(
        cv2,
        np,
        flower_masks,
        hand_masks,
        flower_dilation=args.flower_dilation,
        contact_band=args.contact_band,
    )

    baseline_record = json.loads(paths["aligned_evaluation"].read_text())
    baseline = {key: float(value) for key, value in baseline_record["metrics"].items()}
    strict_gate = json.loads(paths["strict_gate_report"].read_text())
    eligibility_mode = repair_eligibility_mode(
        strict_gate,
        baseline_motion=baseline["motion_preservation"],
        allow_aligned_motion_recovery=args.allow_aligned_motion_recovery,
        minimum_recovery_baseline_motion=args.minimum_recovery_baseline_motion,
    )
    ranking = _policy_ranking(FlowerRepairPolicy.load(paths["policy"]), baseline)
    if ranking[0]["recipe"]["name"] != "epl-local-crossfade":
        raise RuntimeError("trained policy did not select epl-local-crossfade")

    rounds = []
    selected: dict[str, Any] | None = None
    recipes = tuple(
        recipe
        for recipe in DEFAULT_ROUNDS
        if args.only_round is None or recipe.name == args.only_round
    )
    for recipe in recipes:
        frames = [frame.copy() for frame in candidate]
        touched = np.zeros_like(character)
        for _pass_index in range(recipe.passes):
            alpha = source_unsupported_alpha(
                cv2,
                np,
                frames,
                source,
                editable,
                protected,
                excess_threshold=recipe.excess_threshold,
                dilation_pixels=args.mask_dilation,
                feather_sigma=args.mask_feather_sigma,
            )
            touched |= alpha > 0
            frames = apply_local_crossfade_pass(
                np, frames, alpha, protected, strength=recipe.strength
            )
        outside_exact = all(
            np.array_equal(candidate[index][~editable[index]], frames[index][~editable[index]])
            for index in range(len(frames))
        )
        protected_exact = all(
            np.array_equal(candidate[index][protected[index]], frames[index][protected[index]])
            for index in range(len(frames))
        )
        round_dir = output / "rounds" / recipe.name
        round_dir.mkdir()
        lossless = round_dir / "candidate-lossless.mp4"
        compatibility = round_dir / "candidate.mp4"
        _encode(ffmpeg, frames, lossless, float(candidate_info["fps"]), lossless=True)
        _encode(ffmpeg, frames, compatibility, float(candidate_info["fps"]), lossless=False)
        metrics = _evaluate(
            source=paths["source"],
            target_image=paths["target_image"],
            candidate=lossless,
            ffmpeg=ffmpeg,
            object_roi=tuple(args.object_roi),
        )
        compatibility_metrics = _evaluate(
            source=paths["source"],
            target_image=paths["target_image"],
            candidate=compatibility,
            ffmpeg=ffmpeg,
            object_roi=tuple(args.object_roi),
        )
        gates = {
            "temporal": metrics["temporal_consistency"] >= args.minimum_temporal,
            "motion_threshold": metrics["motion_preservation"] >= args.minimum_motion,
            "motion_non_regression": metrics["motion_preservation"] >= baseline["motion_preservation"] - args.maximum_motion_regression,
            "identity_non_regression": metrics["target_identity"] >= baseline["target_identity"] - args.maximum_identity_regression,
            "outside_editable_exact_preencode": outside_exact,
            "flower_and_contact_exact_preencode": protected_exact,
        }
        record = {
            "round": asdict(recipe),
            "metrics": metrics,
            "compatibility_metrics": compatibility_metrics,
            "gates": gates,
            "all_gates_pass": all(gates.values()),
            "modified_fraction_mean_preencode": float(touched.mean()),
            "modified_fraction_max_preencode": float(touched.reshape(len(frames), -1).mean(axis=1).max()),
            "lossless": {"path": str(lossless), "sha256": _sha256(lossless)},
            "compatibility": {"path": str(compatibility), "sha256": _sha256(compatibility)},
        }
        _write_json(round_dir / "evaluation.json", record)
        rounds.append(record)
        # Prefer the first (least aggressive) passing round.  A higher proxy
        # score is not permission to alter more robot geometry.
        if record["all_gates_pass"] and selected is None:
            selected = record

    source_copy = output / "provenance" / "execution-sources" / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), source_copy)
    packages = {}
    for package in ("numpy", "opencv-python", "opencv-python-headless"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "trained_policy_routed_source_unsupported_epl_local_crossfade_v1",
        "status": "PARTIAL",
        "decision": "RETRACK_SELECTED_CANDIDATE" if selected else "HOLD_STRICT_WINDOW",
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu": {"used": False, "reason": "CPU pixel routing over GPU-generated and GPU-tracked inputs"},
        "packages": packages,
        "coordinate_frame": "camera:source_video_pixels",
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in paths.items()
            if label not in {"output_dir", "ffmpeg"}
        },
        "input_info": {"candidate": candidate_info, "source": source_info, "character_mask": mask_info},
        "mask_alignment": {"terminal_character_mask_duplicated": mask_info["frames"] == candidate_info["frames"] - 1},
        "eligibility_mode": eligibility_mode,
        "instances": {"flowers": flower_ids, "hands": hand_ids},
        "protection": {
            "flower_dilation_pixels": args.flower_dilation,
            "contact_band_pixels": args.contact_band,
            "flower_fraction_mean": float(flower_masks.mean()),
            "contact_fraction_mean": float(contact.mean()),
            "protected_fraction_mean": float(protected.mean()),
        },
        "baseline_metrics": baseline,
        "trained_policy": {"path": str(paths["policy"]), "ranking": ranking, "selected_recipe": ranking[0]["recipe"]["name"]},
        "thresholds": {
            "minimum_temporal": args.minimum_temporal,
            "minimum_motion": args.minimum_motion,
            "minimum_recovery_baseline_motion": args.minimum_recovery_baseline_motion,
            "maximum_motion_regression": args.maximum_motion_regression,
            "maximum_identity_regression": args.maximum_identity_regression,
        },
        "rounds": rounds,
        "only_round": args.only_round,
        "selected": selected,
        "execution_source": {"path": str(source_copy), "sha256": _sha256(source_copy)},
        "limitations": [
            "The trained policy selects a bounded repair route; it is not itself a pixel generator.",
            "Automatic temporal, motion, and identity scores are deterministic image-space proxies.",
            "A selected candidate remains PARTIAL until independent GPU re-tracking and the unchanged strict object/contact gate pass.",
            "Only named flower heads and their measured hand-contact band are hard protected; untracked stems still require visual review.",
        ],
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output), "decision": manifest["decision"], "rounds": [{"name": row["round"]["name"], "metrics": {key: row["metrics"][key] for key in ("motion_preservation", "target_identity", "temporal_consistency")}, "all_gates_pass": row["all_gates_pass"]} for row in rounds], "selected": selected["round"]["name"] if selected else None}, indent=2))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
