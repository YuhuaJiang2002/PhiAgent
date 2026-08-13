#!/usr/bin/env python3
"""Download a pinned, object-disjoint HRDexDB embodiment-transfer pilot."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import shlex
import shutil
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASET_ID = "HRDexDB/HRDexDB"
DATASET_REVISION = "a46347556efd7ed87e70e7e87293b462d7253d6f"
DATASET_LICENSE = "cc-by-nc-4.0"
CAMERA_ID = "22641005"
SPLIT_OBJECTS = {
    "train": (
        "apple",
        "attached_container",
        "baby_beaker",
        "balloon_whisk",
        "bamboo_basket",
        "bamboo_box",
    ),
    "validation": ("banana", "beige_brush"),
    "test": ("box_pink", "cactus"),
}
ARM_FILES = ("action.npy", "position.npy", "time.npy", "torque.npy", "velocity.npy")
HAND_FILES = (
    "right_commands.npy",
    "right_commands_time.npy",
    "right_joint_states.npy",
    "right_joint_states_time.npy",
    "right_tactile.npy",
    "right_tactile_time.npy",
)
TIMESTAMP_FILES = ("frame_id.npy", "timestamp.npy")
FROZEN_PAIRS = {
    "apple": {"robot_scene": 0, "human_episode": 0, "robot_c2r": True},
    "attached_container": {"robot_scene": 0, "human_episode": 0, "robot_c2r": True},
    "baby_beaker": {"robot_scene": 0, "human_episode": 0, "robot_c2r": True},
    "balloon_whisk": {"robot_scene": 0, "human_episode": 0, "robot_c2r": True},
    "bamboo_basket": {"robot_scene": 0, "human_episode": 0, "robot_c2r": True},
    "bamboo_box": {"robot_scene": 0, "human_episode": 0, "robot_c2r": True},
    "banana": {"robot_scene": 3, "human_episode": 3, "robot_c2r": False},
    "beige_brush": {"robot_scene": 2, "human_episode": 0, "robot_c2r": True},
    "box_pink": {"robot_scene": 4, "human_episode": 4, "robot_c2r": False},
    "cactus": {"robot_scene": 0, "human_episode": 0, "robot_c2r": True},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def pilot_split() -> dict[str, tuple[str, ...]]:
    """Return the frozen object-level split after checking for leakage."""

    seen: set[str] = set()
    for split, objects in SPLIT_OBJECTS.items():
        if not objects or seen.intersection(objects):
            raise ValueError(f"HRDexDB split {split!r} is empty or overlaps")
        seen.update(objects)
    if len(seen) != 10:
        raise ValueError("HRDexDB pilot must contain exactly ten unique objects")
    if seen != set(FROZEN_PAIRS):
        raise ValueError("HRDexDB frozen pair map does not match the object split")
    return dict(SPLIT_OBJECTS)


def validate_grasp_result(
    payload: Any,
    object_name: str,
    *,
    expected_human_episode: int | None = None,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{object_name} grasp_result must contain a JSON object")
    if payload.get("grasp_success") is not True:
        raise ValueError(f"{object_name} is not a successful robot grasp")
    try:
        paired = int(payload.get("human_paired_episode", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{object_name} is not paired with a human episode") from exc
    if paired < 0:
        raise ValueError(f"{object_name} is not paired with a human episode")
    if expected_human_episode is not None and paired != expected_human_episode:
        raise ValueError(
            f"{object_name} paired human episode changed from "
            f"{expected_human_episode} to {paired}"
        )


def _resolve_url(path: str) -> str:
    encoded = urllib.parse.quote(path, safe="/")
    return (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{DATASET_REVISION}/{encoded}"
    )


def _tree_url(path: str) -> str:
    encoded = urllib.parse.quote(path, safe="/")
    return (
        f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/"
        f"{DATASET_REVISION}/{encoded}?expand=false&limit=1000"
    )


def _request_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "PhiAgent/0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _mano_paths(object_name: str, human_episode: int) -> tuple[str, ...]:
    root = f"human/{object_name}/{human_episode}/hand/mano_params"
    payload = _request_json(_tree_url(root))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"HRDexDB returned no MANO parameters for {object_name}")
    paths = sorted(
        str(item["path"])
        for item in payload
        if isinstance(item, dict)
        and item.get("type") == "file"
        and str(item.get("path", "")).endswith(".json")
    )
    if not paths:
        raise ValueError(f"HRDexDB MANO listing is invalid for {object_name}")
    return tuple(paths)


def _explicit_paths(
    object_name: str,
    *,
    robot_scene: int,
    human_episode: int,
    robot_c2r: bool,
) -> tuple[str, ...]:
    human = f"human/{object_name}/{human_episode}"
    robot = f"inspire_f1/{object_name}/{robot_scene}"
    paths = [
        f"{human}/C2R.npy",
        f"{human}/cam_param/ego_calib.json",
        f"{human}/cam_param/extrinsics.json",
        f"{human}/cam_param/intrinsics.json",
        f"{human}/vid/{CAMERA_ID}.mp4",
        f"{robot}/cam_param/extrinsics.json",
        f"{robot}/cam_param/intrinsics.json",
        f"{robot}/grasp_result.json",
        f"{robot}/vid/{CAMERA_ID}.mp4",
        f"object_6d_pose_v2/human/{object_name}_{human_episode}.npz",
        f"object_6d_pose_v2/inspire_f1/{object_name}_{robot_scene}.npz",
        f"assets/mesh_v2/{object_name}/{object_name}.obj",
        f"assets/mesh_v2/{object_name}/{object_name}.obj.diameter.json",
    ]
    if robot_c2r:
        paths.append(f"{robot}/C2R.npy")
    paths.extend(f"{robot}/raw/arm/{name}" for name in ARM_FILES)
    paths.extend(f"{robot}/raw/hand/{name}" for name in HAND_FILES)
    paths.extend(f"{robot}/raw/timestamps/{name}" for name in TIMESTAMP_FILES)
    return tuple(paths)


def _directory_names(path: str) -> set[str]:
    payload = _request_json(_tree_url(path))
    if not isinstance(payload, list):
        raise ValueError(f"HRDexDB returned an invalid directory listing for {path}")
    return {
        str(item["path"]).rsplit("/", maxsplit=1)[-1]
        for item in payload
        if isinstance(item, dict)
    }


def _complete_pair(object_name: str) -> dict[str, int]:
    payload = _request_json(_tree_url(f"inspire_f1/{object_name}"))
    if not isinstance(payload, list):
        raise ValueError(f"HRDexDB returned invalid scenes for {object_name}")
    scenes = sorted(
        int(str(item["path"]).rsplit("/", maxsplit=1)[-1])
        for item in payload
        if isinstance(item, dict)
        and item.get("type") == "directory"
        and str(item.get("path", "")).rsplit("/", maxsplit=1)[-1].isdigit()
    )
    for scene in scenes:
        robot_root = f"inspire_f1/{object_name}/{scene}"
        names = _directory_names(robot_root)
        if not {"raw", "vid", "grasp_result.json"}.issubset(names):
            continue
        grasp = _request_json(_resolve_url(f"{robot_root}/grasp_result.json"))
        if not isinstance(grasp, dict) or grasp.get("grasp_success") is not True:
            continue
        try:
            paired = int(grasp.get("human_paired_episode", -1))
        except (TypeError, ValueError):
            continue
        if paired < 0:
            continue
        try:
            human_video = _directory_names(f"human/{object_name}/{paired}/vid")
            robot_video = _directory_names(f"{robot_root}/vid")
        except urllib.error.HTTPError:
            continue
        if f"{CAMERA_ID}.mp4" not in human_video or f"{CAMERA_ID}.mp4" not in robot_video:
            continue
        return {
            "robot_scene": scene,
            "human_episode": paired,
            "robot_c2r": "C2R.npy" in names,
        }
    raise ValueError(f"no complete successful HRDexDB pair found for {object_name}")


def _download(
    root: Path,
    relative: str,
    *,
    reuse_root: Path | None,
    retries: int,
) -> dict[str, object]:
    destination = root / relative
    if destination.exists():
        raise FileExistsError(f"download destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    reusable = None if reuse_root is None else reuse_root / relative
    if reusable is not None and reusable.is_file() and reusable.stat().st_size > 0:
        try:
            os.link(reusable, destination)
            storage = "hard_link"
        except OSError:
            shutil.copy2(reusable, destination)
            storage = "copy"
        return {
            "path": relative,
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "source_url": _resolve_url(relative),
            "reused": True,
            "reuse_storage": storage,
        }
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(
        _resolve_url(relative), headers={"User-Agent": "PhiAgent/0"}
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with (
                urllib.request.urlopen(request, timeout=300) as response,
                partial.open("wb") as handle,
            ):
                while block := response.read(1024 * 1024):
                    handle.write(block)
            last_error = None
            break
        except Exception as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(2 ** (attempt + 1))
    if last_error is not None:
        raise RuntimeError(
            f"failed to download HRDexDB path {relative} after {retries} attempts: "
            f"{last_error}"
        ) from last_error
    if partial.stat().st_size == 0:
        raise ValueError(f"download is empty: {relative}")
    partial.replace(destination)
    return {
        "path": relative,
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "source_url": _resolve_url(relative),
        "reused": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--reuse-data-root", type=Path)
    parser.add_argument("--without-mano", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite HRDexDB pilot: {output}")
    if args.workers <= 0 or args.retries <= 0:
        raise ValueError("workers and retries must be positive")
    reuse_root = (
        None
        if args.reuse_data_root is None
        else args.reuse_data_root.expanduser().resolve()
    )
    if reuse_root is not None and not reuse_root.is_dir():
        raise ValueError(f"reuse data root is missing: {reuse_root}")
    split = pilot_split()
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    _write_json(
        output / "config.json",
        {
            "schema_version": "1.0.0",
            "status": "STARTED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "license": DATASET_LICENSE,
            "camera_id": CAMERA_ID,
            "split": {name: list(objects) for name, objects in split.items()},
            "workers": args.workers,
            "retries": args.retries,
            "include_mano": not args.without_mano,
            "reuse_data_root": None if reuse_root is None else str(reuse_root),
        },
    )
    data_root = output / "data"
    paths = {"README.md"}
    pairs = {}
    for objects in split.values():
        for object_name in objects:
            pair = dict(FROZEN_PAIRS[object_name])
            pairs[object_name] = pair
            paths.update(_explicit_paths(object_name, **pair))
            if not args.without_mano:
                paths.update(_mano_paths(object_name, pair["human_episode"]))
    records: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _download,
                data_root,
                relative,
                reuse_root=reuse_root,
                retries=args.retries,
            ): relative
            for relative in sorted(paths)
        }
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: str(item["path"]))
    for objects in split.values():
        for object_name in objects:
            pair = pairs[object_name]
            grasp = json.loads(
                (
                    data_root
                    / (
                        f"inspire_f1/{object_name}/"
                        f"{pair['robot_scene']}/grasp_result.json"
                    )
                ).read_text()
            )
            validate_grasp_result(
                grasp,
                object_name,
                expected_human_episode=pair["human_episode"],
            )
    summary = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "PARTIAL",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "license": DATASET_LICENSE,
        "rights_boundary": (
            "Non-commercial research only under CC BY-NC 4.0; attribution, "
            "license link, modification notice, and separate privacy review are required."
        ),
        "camera_id": CAMERA_ID,
        "split": {name: list(objects) for name, objects in split.items()},
        "objects": sum((list(objects) for objects in split.values()), []),
        "pairs": pairs,
        "file_count": len(records),
        "total_bytes": sum(int(item["bytes"]) for item in records),
        "files": records,
        "reused_file_count": sum(bool(item["reused"]) for item in records),
        "claim_boundary": (
            "This download validates an object-disjoint data contract. It does not "
            "establish human-to-robot transfer quality or SOTA."
        ),
    }
    _write_json(output / "manifest.json", summary)
    (output / "download.log").write_text(
        f"downloaded {summary['file_count']} files / {summary['total_bytes']} bytes\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
