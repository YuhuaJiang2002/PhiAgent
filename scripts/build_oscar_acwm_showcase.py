#!/usr/bin/env python3
"""Build the portable web showcase from reviewed OSCAR AC-WM evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.acwm.adapters import (  # noqa: E402
    OSCAR_MODEL_REVISION,
    OSCAR_REPOSITORY_COMMIT,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _writer(ffmpeg: str, output: Path, *, width: int, height: int, fps: float) -> Any:
    return subprocess.Popen(
        [
            ffmpeg,
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
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def _read_video(cv2: Any, path: Path) -> tuple[list[Any], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames or fps <= 0:
        raise RuntimeError(f"invalid video {path}")
    return frames, fps


def _label(
    cv2: Any,
    frame: Any,
    title: str,
    score: float,
    *,
    status: str = "ACCEPTED",
    color: tuple[int, int, int] = (106, 245, 200),
) -> Any:
    labeled = frame.copy()
    overlay = labeled.copy()
    cv2.rectangle(overlay, (0, 0), (labeled.shape[1], 46), (4, 12, 9), -1)
    cv2.addWeighted(overlay, 0.84, labeled, 0.16, 0, labeled)
    cv2.putText(
        labeled,
        f"{status}  /  {title}",
        (16, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        labeled,
        f"ACTION {score:.2f}",
        (labeled.shape[1] - 138, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 245, 230),
        1,
        cv2.LINE_AA,
    )
    return labeled


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", type=Path, default=Path("outputs/acwm-open-models/20260810T143008Z-ae32011f")
    )
    parser.add_argument(
        "--condition-run",
        type=Path,
        default=Path("outputs/acwm-oscar-conditions/20260810T131648Z-hand2dex2-v2"),
    )
    parser.add_argument(
        "--repair-run",
        type=Path,
        default=Path("outputs/acwm-open-models/20260810T144421Z-efdb8fea"),
    )
    parser.add_argument(
        "--structure-run",
        type=Path,
        default=Path("outputs/acwm-hand-structure/20260810T162000Z-oscar-slide-right-lock-v1"),
    )
    parser.add_argument(
        "--canonical-run",
        type=Path,
        default=Path("outputs/acwm-hand-canonical/20260810T161000Z-oscar-slide-right-sam2-v5"),
    )
    parser.add_argument(
        "--articulated-run",
        type=Path,
        default=Path("outputs/acwm-open-models/20260810T155518Z-06311bc4"),
    )
    parser.add_argument(
        "--articulated-condition-run",
        type=Path,
        default=Path(
            "outputs/acwm-oscar-conditions/20260810T155400Z-hand2dex2-right-lift-arc-v2"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("demo/showcase"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    run = args.run.expanduser().resolve()
    condition_run = args.condition_run.expanduser().resolve()
    repair_run = args.repair_run.expanduser().resolve()
    structure_run = args.structure_run.expanduser().resolve()
    canonical_run = args.canonical_run.expanduser().resolve()
    articulated_run = args.articulated_run.expanduser().resolve()
    articulated_condition_run = args.articulated_condition_run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate_paths = {
        "slide-left": run / "candidates" / "000-slide-left-oscar.mp4",
        "slide-right": run / "candidates" / "001-slide-right-oscar.mp4",
        "lift-up": run / "candidates" / "002-lift-up-oscar.mp4",
    }
    candidate_indices = {
        "slide-left": "000",
        "slide-right": "001",
        "lift-up": "002",
    }
    evaluations = {
        label: json.loads(
            (
                run / "candidates" / "evaluation-v2" / f"{index}-{label}-oscar" / "evaluation.json"
            ).read_text()
        )
        for index, label in (("000", "slide-left"), ("001", "slide-right"), ("002", "lift-up"))
    }
    articulated_video = articulated_run / "candidates" / "000-slide-right-oscar.mp4"
    articulated_evaluation_path = (
        articulated_run
        / "candidates"
        / "evaluation-human-reviewed"
        / "000-slide-right-oscar"
        / "evaluation.json"
    )
    articulated_evaluation = json.loads(articulated_evaluation_path.read_text())
    required = [
        *candidate_paths.values(),
        condition_run / "input" / "real-scene-source.mp4",
        condition_run / "manifest.json",
        repair_run / "trace.json",
        structure_run / "manifest.json",
        structure_run / "output" / "structure-locked.mp4",
        structure_run / "evaluation" / "evaluation.json",
        canonical_run / "manifest.json",
        canonical_run / "canonical-hand-preview.jpg",
        articulated_video,
        articulated_evaluation_path,
        articulated_condition_run / "manifest.json",
        articulated_condition_run / "variants" / "slide-right" / "skeleton-overlay.mp4",
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise ValueError(f"showcase inputs are missing: {missing}")

    names = {
        "slide-left": "oscar-acwm-slide-left-rejected.mp4",
        "slide-right": "oscar-acwm-slide-right-raw.mp4",
        "lift-up": "oscar-acwm-lift-up.mp4",
    }
    for label, source in candidate_paths.items():
        shutil.copy2(source, output / names[label])
    articulated_name = "oscar-acwm-carry-right.mp4"
    shutil.copy2(articulated_video, output / articulated_name)
    structure_video = structure_run / "output" / "structure-locked.mp4"
    structure_name = "oscar-acwm-slide-right-structure-locked.mp4"
    shutil.copy2(structure_video, output / structure_name)
    # Keep the legacy route working, but point it at the native articulated result.
    shutil.copy2(articulated_video, output / "oscar-acwm-slide-right.mp4")
    shutil.copy2(
        structure_run / "review" / "storyboard.jpg",
        output / "oscar-acwm-slide-right-structure-locked-storyboard.jpg",
    )
    shutil.copy2(
        canonical_run / "canonical-hand-preview.jpg",
        output / "oscar-acwm-canonical-hand-preview.jpg",
    )
    shutil.copy2(
        articulated_evaluation_path.parent / "storyboard.jpg",
        output / "oscar-acwm-carry-right-storyboard.jpg",
    )
    shutil.copy2(
        articulated_condition_run / "variants" / "slide-right" / "skeleton-overlay.mp4",
        output / "oscar-acwm-carry-right-condition.mp4",
    )
    shutil.copy2(
        condition_run / "input" / "real-scene-source.mp4",
        output / "oscar-acwm-real-source.mp4",
    )
    for label in candidate_paths:
        shutil.copy2(
            condition_run / "variants" / label / "skeleton-overlay.mp4",
            output / f"oscar-acwm-{label}-condition.mp4",
        )
        shutil.copy2(
            run
            / "candidates"
            / "evaluation-v2"
            / f"{candidate_indices[label]}-{label}-oscar"
            / "storyboard.jpg",
            output / f"oscar-acwm-{label}-storyboard.jpg",
        )

    import cv2
    import numpy as np

    raw_right_frames, raw_right_fps = _read_video(cv2, candidate_paths["slide-right"])
    locked_frames, locked_fps = _read_video(cv2, structure_video)
    right_frames, right_fps = _read_video(cv2, articulated_video)
    lift_frames, lift_fps = _read_video(cv2, candidate_paths["lift-up"])
    if (
        len(raw_right_frames) != len(locked_frames)
        or len(locked_frames) != len(right_frames)
        or len(right_frames) != len(lift_frames)
        or abs(raw_right_fps - locked_fps) > 1e-3
        or abs(locked_fps - right_fps) > 1e-3
        or abs(right_fps - lift_fps) > 1e-3
    ):
        raise ValueError("accepted OSCAR videos are not temporally aligned")
    structure_evaluation = json.loads(
        (structure_run / "evaluation" / "evaluation.json").read_text()
    )
    comparison_path = output / "oscar-acwm-accepted-comparison.mp4"
    writer = _writer(
        shutil.which("ffmpeg") or "ffmpeg",
        comparison_path,
        width=1280,
        height=480,
        fps=right_fps,
    )
    comparison_frames = []
    try:
        for right, lift in zip(right_frames, lift_frames):
            right = cv2.resize(right, (640, 480), interpolation=cv2.INTER_AREA)
            lift = cv2.resize(lift, (640, 480), interpolation=cv2.INTER_AREA)
            comparison = np.hstack(
                (
                    _label(
                        cv2,
                        right,
                        "LIFT-ARC CARRY RIGHT / NATIVE",
                        articulated_evaluation["action_adherence"],
                    ),
                    _label(cv2, lift, "LIFT UP", evaluations["lift-up"]["action_adherence"]),
                )
            )
            comparison_frames.append(comparison)
            assert writer.stdin is not None
            writer.stdin.write(comparison.tobytes())
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
        returncode = writer.wait()
    if returncode:
        raise RuntimeError(f"comparison ffmpeg failed with exit {returncode}")

    repair_comparison_path = output / "oscar-acwm-slide-right-raw-vs-structure-lock.mp4"
    repair_writer = _writer(
        shutil.which("ffmpeg") or "ffmpeg",
        repair_comparison_path,
        width=1280,
        height=480,
        fps=right_fps,
    )
    repair_comparison_frames = []
    try:
        for raw, locked in zip(raw_right_frames, locked_frames):
            raw = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_AREA)
            locked = cv2.resize(locked, (640, 480), interpolation=cv2.INTER_AREA)
            comparison = np.hstack(
                (
                    _label(
                        cv2,
                        raw,
                        "RAW OSCAR",
                        evaluations["slide-right"]["action_adherence"],
                        status="REJECTED / HAND DRIFT",
                        color=(89, 132, 255),
                    ),
                    _label(
                        cv2,
                        locked,
                        "STRUCTURE LOCKED",
                        structure_evaluation["action_adherence"],
                        status="REJECTED / RIGID SHIFT",
                        color=(89, 132, 255),
                    ),
                )
            )
            repair_comparison_frames.append(comparison)
            assert repair_writer.stdin is not None
            repair_writer.stdin.write(comparison.tobytes())
    finally:
        if repair_writer.stdin is not None:
            repair_writer.stdin.close()
        repair_returncode = repair_writer.wait()
    if repair_returncode:
        raise RuntimeError(f"repair comparison ffmpeg failed with exit {repair_returncode}")
    cv2.imwrite(
        str(output / "oscar-acwm-slide-right-raw-vs-structure-lock-poster.jpg"),
        repair_comparison_frames[len(repair_comparison_frames) // 2],
    )
    cv2.imwrite(
        str(output / "oscar-acwm-accepted-comparison-poster.jpg"),
        comparison_frames[len(comparison_frames) // 2],
    )
    social = np.full((630, 1200, 3), (7, 16, 13), dtype=np.uint8)
    preview = cv2.resize(
        comparison_frames[len(comparison_frames) // 2],
        (1200, 450),
        interpolation=cv2.INTER_AREA,
    )
    social[90:540] = preview
    cv2.putText(
        social,
        "PHIAGENT / OPEN AC-WM / REAL-SCENE ACTION CONTROL",
        (28, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (106, 245, 200),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        social,
        "OSCAR-2B  |  SAME REAL FIRST FRAME  |  NATIVE ARTICULATED ACTIONS",
        (28, 592),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (230, 245, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output.parent / "og.png"), social)
    for label, frames in (
        ("slide-left", _read_video(cv2, candidate_paths["slide-left"])[0]),
        ("slide-right", right_frames),
        ("slide-right-raw", raw_right_frames),
        ("carry-right", right_frames),
        ("lift-up", lift_frames),
    ):
        cv2.imwrite(
            str(output / f"oscar-acwm-{label}-poster.jpg"),
            frames[min(len(frames) - 1, 64)],
        )

    repair_trace = json.loads((repair_run / "trace.json").read_text())
    repair_score = repair_trace["candidates"][0]["scorecard"]
    structure_manifest = json.loads((structure_run / "manifest.json").read_text())
    canonical_manifest = json.loads((canonical_run / "manifest.json").read_text())
    articulated_condition_manifest = json.loads(
        (articulated_condition_run / "manifest.json").read_text()
    )
    articulated_variant = articulated_condition_manifest["variants"][0]
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "summary": (
            "Lift-up and lift-arc carry-right are accepted as native articulated "
            "OSCAR generations. The original slide actions and rigid structure-lock "
            "repair remain user-rejected evidence."
        ),
        "model": {
            "name": "OSCAR-2B",
            "repository_commit": OSCAR_REPOSITORY_COMMIT,
            "model_revision": OSCAR_MODEL_REVISION,
        },
        "matched_protocol": {
            "seed": 20260810,
            "num_inference_steps": 35,
            "guidance_scale": 6.0,
            "frames": 81,
            "fps": 15.0,
            "resolution": [640, 480],
            "same_real_first_frame": True,
        },
        "source": {
            "video": "oscar-acwm-real-source.mp4",
            "sha256": _sha256(output / "oscar-acwm-real-source.mp4"),
        },
        "variants": [
            {
                "case_id": label,
                "status": (
                    "USER_REJECTED_STRUCTURE"
                    if label == "slide-right"
                    else (
                        "ACCEPTED"
                        if evaluations[label]["human_review_passed"]
                        and evaluations[label]["action_adherence"] >= 0.75
                        else "REJECTED"
                    )
                ),
                "video": names[label],
                "video_sha256": _sha256(output / names[label]),
                "condition": f"oscar-acwm-{label}-condition.mp4",
                "storyboard": f"oscar-acwm-{label}-storyboard.jpg",
                "scores": {
                    key: evaluations[label][key]
                    for key in (
                        "action_adherence",
                        "embodiment_consistency",
                        "object_interaction",
                        "temporal_consistency",
                        "background_consistency",
                        "human_review_passed",
                    )
                },
                "diagnoses": evaluations[label]["diagnoses"],
            }
            for label in candidate_paths
        ],
        "structure_repair": {
            "case_id": "slide-right",
            "status": "USER_REJECTED_RIGID_TRANSLATION",
            "method": structure_manifest["method"],
            "video": structure_name,
            "video_sha256": _sha256(output / structure_name),
            "raw_vs_repair_video": repair_comparison_path.name,
            "raw_vs_repair_video_sha256": _sha256(repair_comparison_path),
            "canonical_hand_preview": "oscar-acwm-canonical-hand-preview.jpg",
            "canonical_segmentation": {
                "method": canonical_manifest["method"],
                "source_revision": canonical_manifest["source_revision"],
                "checkpoint_sha256": canonical_manifest["checkpoint_sha256"],
                "connected_components": canonical_manifest["selection"]["connected_components"],
            },
            "structure_gates": structure_manifest["acceptance"],
            "scores": {
                key: structure_evaluation[key]
                for key in (
                    "action_adherence",
                    "embodiment_consistency",
                    "object_interaction",
                    "temporal_consistency",
                    "background_consistency",
                    "human_review_passed",
                )
            },
            "limitations": structure_manifest["limitations"],
        },
        "articulated_carry": {
            "case_id": "carry-right-lift-arc",
            "status": "ACCEPTED",
            "method": "native_oscar_articulation_with_reviewed_lift_arc_condition",
            "instruction": articulated_variant["instruction"],
            "timeline": articulated_variant["timeline"],
            "prompt": articulated_variant["prompt"],
            "vertical_motion_template": articulated_variant["vertical_motion_template"],
            "video": articulated_name,
            "video_sha256": _sha256(output / articulated_name),
            "condition": "oscar-acwm-carry-right-condition.mp4",
            "condition_sha256": _sha256(
                output / "oscar-acwm-carry-right-condition.mp4"
            ),
            "storyboard": "oscar-acwm-carry-right-storyboard.jpg",
            "source_run": str(articulated_run),
            "condition_run": str(articulated_condition_run),
            "scores": {
                key: articulated_evaluation[key]
                for key in (
                    "action_adherence",
                    "embodiment_consistency",
                    "object_interaction",
                    "temporal_consistency",
                    "background_consistency",
                    "human_review_passed",
                )
            },
            "metrics": articulated_evaluation["metrics"],
            "human_review": articulated_evaluation["human_review"],
            "diagnoses": articulated_evaluation["diagnoses"],
        },
        "repair": {
            "case_id": "slide-left",
            "change": "terminal wrist camera pixel changed from (233.5, 285.8) to (100, 309)",
            "status": "REJECTED",
            "scores": repair_score,
            "evidence": "Action-condition repair moved the robot across the bowl but did not move the bowl.",
        },
        "acceptance": {
            "required_threshold_per_numeric_gate": 0.75,
            "human_review_required": True,
            "raw_model_accepted_cases": ["lift-up", "carry-right-lift-arc"],
            "workflow_accepted_cases": ["lift-up", "carry-right-lift-arc"],
            "rejected_cases": [
                "slide-left",
                "slide-right-raw",
                "slide-right-structure-locked",
            ],
        },
        "limitations": [
            "This is a camera-pixel skeleton world-model rollout, not a robot-base action or physical execution.",
            "The yellow-object detector and robot-edge metrics are proxies; explicit human review remains mandatory.",
            "OSCAR did not preserve causal leftward bowl contact in this out-of-domain real scene.",
            "Posthoc user review rejected the raw slide-right hand morphology and then rejected the structure-locked repair because its whole hand shifts rigidly.",
            "The accepted rightward action is explicitly lift-then-carry along an image-space arc; it is not evidence for contact-preserving tabletop sliding.",
            "The historical structure repair freezes one 2-D hand topology and does not add finger articulation or 3-D kinematic validity.",
        ],
    }
    _write_json(output / "oscar-acwm-evaluation.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
