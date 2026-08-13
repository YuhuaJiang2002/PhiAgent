#!/usr/bin/env python3
"""Prepare leakage-safe Wan-Animate-2 inputs from the HRDexDB object pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REFERENCE_OBJECT = "apple"
EVALUATION_SPLIT = "validation"
PROMPT = (
    "An Inspire F1 dexterous robot hand performs the demonstrated grasp while "
    "preserving the object, contact sequence, camera, and background."
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


def evaluation_objects(manifest: dict[str, Any]) -> tuple[str, ...]:
    split = manifest.get("split")
    if not isinstance(split, dict):
        raise ValueError("HRDexDB manifest lacks object split")
    train = set(map(str, split.get("train", [])))
    validation = tuple(map(str, split.get(EVALUATION_SPLIT, [])))
    test = set(map(str, split.get("test", [])))
    if (
        REFERENCE_OBJECT not in train
        or not validation
        or train.intersection(validation)
        or test.intersection(validation)
    ):
        raise ValueError("HRDexDB Wan pilot split or reference bank leaks")
    return validation


def _probe(ffprobe: Path, video: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": str(stream["r_frame_rate"]),
        "frames": int(stream["nb_frames"]),
        "duration_seconds": float(payload["format"]["duration"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    parser.add_argument(
        "--ffprobe", type=Path, default=Path(shutil.which("ffprobe") or "ffprobe")
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest_path = args.dataset_manifest.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    ffprobe = args.ffprobe.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite HRDexDB Wan pilot: {output}")
    if not manifest_path.is_file():
        raise ValueError(f"HRDexDB manifest is missing: {manifest_path}")
    if not ffmpeg.is_file() or not ffprobe.is_file():
        raise ValueError("ffmpeg and ffprobe are required")
    manifest = json.loads(manifest_path.read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "WORKING"
        or manifest.get("dataset_revision")
        != "a46347556efd7ed87e70e7e87293b462d7253d6f"
    ):
        raise ValueError("HRDexDB pilot is not a completed pinned dataset")
    objects = evaluation_objects(manifest)
    pairs = manifest["pairs"]
    data = manifest_path.parent / "data"
    reference_pair = pairs[REFERENCE_OBJECT]
    reference_video = (
        data
        / "inspire_f1"
        / REFERENCE_OBJECT
        / str(reference_pair["robot_scene"])
        / "vid"
        / "22641005.mp4"
    )
    if not reference_video.is_file():
        raise ValueError(f"train-object reference video is missing: {reference_video}")
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    target = output / "inspire-f1-train-object-reference.png"
    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(reference_video),
            "-frames:v",
            "1",
            str(target),
        ],
        check=True,
    )
    cases = []
    for object_name in objects:
        pair = pairs[object_name]
        human = (
            data
            / "human"
            / object_name
            / str(pair["human_episode"])
            / "vid"
            / "22641005.mp4"
        )
        robot = (
            data
            / "inspire_f1"
            / object_name
            / str(pair["robot_scene"])
            / "vid"
            / "22641005.mp4"
        )
        for path in (human, robot):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"HRDexDB evaluation video is missing: {path}")
        cases.append(
            {
                "object": object_name,
                "split": EVALUATION_SPLIT,
                "human_episode": int(pair["human_episode"]),
                "robot_scene": int(pair["robot_scene"]),
                "source_video": str(human),
                "source_sha256": _sha256(human),
                "source_probe": _probe(ffprobe, human),
                "paired_robot_reference_video": str(robot),
                "paired_robot_reference_sha256": _sha256(robot),
                "paired_robot_reference_probe": _probe(ffprobe, robot),
                "target_image": str(target),
                "target_image_sha256": _sha256(target),
                "target_reference_object": REFERENCE_OBJECT,
                "target_reference_split": "train",
                "seed": args.seed,
                "prompt": PROMPT,
                "method_label": "wan_animate2_hrdexdb_object_disjoint_raw_baseline",
            }
        )
    cases_path = output / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases)
    )
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "PARTIAL",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "license": manifest["license"],
        "evaluation_split": EVALUATION_SPLIT,
        "reference_object": REFERENCE_OBJECT,
        "reference_object_split": "train",
        "target_image": str(target),
        "target_image_sha256": _sha256(target),
        "cases": cases,
        "cases_jsonl": str(cases_path),
        "cases_jsonl_sha256": _sha256(cases_path),
        "claim_boundary": (
            "Validation-object raw baseline only. Paired robot videos are evaluator "
            "references and never generation inputs; held-out test objects remain sealed."
        ),
    }
    _write_json(output / "manifest.json", result)
    (output / "prepare.log").write_text(
        f"prepared {len(cases)} validation objects with train-object reference\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

