#!/usr/bin/env python3
"""Prepare immutable, rights-attributed real-background sources for H3 RSI."""

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
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _run(command: list[str], log: Path) -> subprocess.CompletedProcess[str]:
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + shlex.join(command) + "\n")
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(completed.stdout)
        handle.write(completed.stderr)
    return completed


def _git_state() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0.0":
        raise ValueError("source config must use schema_version 1.0.0")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source config must contain a non-empty sources list")
    ids: set[str] = set()
    for item in sources:
        if not isinstance(item, dict):
            raise ValueError("every source entry must be an object")
        source_id = str(item.get("source_id", ""))
        if not source_id or source_id in ids:
            raise ValueError(f"source_id must be non-empty and unique: {source_id!r}")
        ids.add(source_id)
        if item.get("split") not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split for {source_id}")
        for field in ("landing_url", "creator", "license_id"):
            if not str(item.get(field, "")).strip():
                raise ValueError(f"{source_id} is missing {field}")
        locations = [bool(item.get("direct_url")), bool(item.get("local_path"))]
        if sum(locations) != 1:
            raise ValueError(f"{source_id} requires exactly one direct_url or local_path")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/h3-identity-domain-sources")
    )
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--curl", default=shutil.which("curl"))
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = (
        args.experiment_dir.expanduser().resolve()
        if args.experiment_dir
        else args.output_root.expanduser().resolve() / f"{stamp}-{uuid4().hex[:8]}"
    )
    experiment.mkdir(parents=True, exist_ok=False)
    manifest_path = experiment / "manifest.json"
    log_path = experiment / "commands.log"
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "method": "rights_attributed_real_background_source_preparation",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "seed": 20260811,
        "git": _git_state(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "license_url": config["license_url"],
        "release_policy": config["release_policy"],
        "sources": [],
    }
    _write_json(manifest_path, manifest)
    try:
        executables = {name: Path(value or "") for name, value in {
            "curl": args.curl,
            "ffmpeg": args.ffmpeg,
            "ffprobe": args.ffprobe,
        }.items()}
        for name, path in executables.items():
            if not path.is_file():
                raise ValueError(f"{name} executable is missing: {path}")
        source_dir = experiment / "sources"
        review_dir = experiment / "review"
        source_dir.mkdir()
        review_dir.mkdir()
        records = []
        for raw in config["sources"]:
            assert isinstance(raw, dict)
            source_id = str(raw["source_id"])
            output = source_dir / f"{source_id}.mp4"
            acquisition: dict[str, object]
            if raw.get("direct_url"):
                temporary = output.with_suffix(".download")
                command = [
                    str(executables["curl"]),
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--output",
                    str(temporary),
                    str(raw["direct_url"]),
                ]
                _run(command, log_path)
                temporary.replace(output)
                acquisition = {"kind": "download", "command": command}
            else:
                source = (PROJECT_ROOT / str(raw["local_path"])).resolve()
                source.relative_to(PROJECT_ROOT)
                if not source.is_file() or source.stat().st_size == 0:
                    raise ValueError(f"local source is missing or empty: {source}")
                shutil.copy2(source, output)
                acquisition = {
                    "kind": "local_copy",
                    "source": str(source),
                    "source_sha256": _sha256(source),
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"COPY {source} -> {output}\n")
            probe_command = [
                str(executables["ffprobe"]),
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(output),
            ]
            probe = json.loads(_run(probe_command, log_path).stdout)
            video_streams = [
                stream
                for stream in probe.get("streams", [])
                if stream.get("codec_type") == "video"
            ]
            if len(video_streams) != 1:
                raise RuntimeError(f"{source_id} must contain exactly one video stream")
            storyboard = review_dir / f"{source_id}-storyboard.jpg"
            review_command = [
                str(executables["ffmpeg"]),
                "-v",
                "error",
                "-i",
                str(output),
                "-vf",
                "fps=1/2,scale=320:-1,tile=4x3:padding=4:margin=4",
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(storyboard),
            ]
            _run(review_command, log_path)
            records.append(
                {
                    "source_id": source_id,
                    "split": raw["split"],
                    "landing_url": raw["landing_url"],
                    "creator": raw["creator"],
                    "license_id": raw["license_id"],
                    "direct_url": raw.get("direct_url"),
                    "path": str(output),
                    "bytes": output.stat().st_size,
                    "sha256": _sha256(output),
                    "probe": probe,
                    "storyboard": str(storyboard),
                    "storyboard_sha256": _sha256(storyboard),
                    "acquisition": acquisition,
                }
            )
        shutil.copy2(config_path, experiment / "source-config.json")
        manifest.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "executables": {
                    name: str(path) for name, path in executables.items()
                },
                "sources": records,
                "limitations": [
                    "Pexels media is retained under ignored experiment outputs and is not vendored in the source tree.",
                    "This preparation step establishes provenance and split assignment; it does not make the source frames topology-positive.",
                    "Any adapter distribution still requires a combined upstream-model and training-data legal review.",
                ],
            }
        )
        _write_json(manifest_path, manifest)
        print(json.dumps({"experiment": str(experiment), "sources": len(records)}))
        return 0
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
