#!/usr/bin/env python3
"""Download one pinned raw DROID episode for internal calibration auditing."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import platform
import shlex
import socket
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


EPISODE_ID = "IPRL+5085c3ce+2023-10-07-16h-12m-06s"
GCS_PREFIX = (
    "1.0.1/IPRL/success/2023-10-07/"
    "Sat_Oct__7_16:12:06_2023"
)
FILES = {
    "trajectory.h5": (2_073_542, "YI1ncY2xVoHpohiYImgZkw=="),
    f"metadata_{EPISODE_ID}.json": (1_797, "kFFRFl2mX7KujWvPQzzstQ=="),
    "recordings/SVO/12391924.svo": (14_441_196, "hLqrCiwZpfR/Myc58bvMmg=="),
    "recordings/SVO/27432424.svo": (14_859_433, "ohli71nj3v8u4voQVghazA=="),
    "recordings/SVO/28221883.svo": (14_962_399, "BAuKc2BLHIFI4BEPZI4KJQ=="),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode("ascii")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _url(relative: str) -> str:
    path = urllib.parse.quote(f"{GCS_PREFIX}/{relative}", safe="/")
    return f"https://storage.googleapis.com/gresearch/robotics/droid_raw/{path}"


def _download(root: Path, relative: str) -> dict[str, object]:
    expected_bytes, expected_md5 = FILES[relative]
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"download destination already exists: {destination}")
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(_url(relative), headers={"User-Agent": "PhiAgent/0"})
    try:
        with (
            urllib.request.urlopen(request, timeout=300) as response,
            partial.open("wb") as handle,
        ):
            while block := response.read(1024 * 1024):
                handle.write(block)
    except Exception as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download raw DROID path {relative}: {error}") from error
    if partial.stat().st_size != expected_bytes:
        raise ValueError(
            f"raw DROID byte mismatch for {relative}: "
            f"{partial.stat().st_size} != {expected_bytes}"
        )
    partial.replace(destination)
    actual_md5 = md5_base64(destination)
    if actual_md5 != expected_md5:
        raise ValueError(
            f"raw DROID GCS MD5 mismatch for {relative}: {actual_md5} != {expected_md5}"
        )
    return {
        "path": relative,
        "bytes": expected_bytes,
        "gcs_md5_base64": actual_md5,
        "sha256": _sha256(destination),
        "source_url": _url(relative),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite raw DROID smoke: {output}")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    _write_json(
        output / "config.json",
        {
            "schema_version": "1.0.0",
            "status": "STARTED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "episode_id": EPISODE_ID,
            "gcs_prefix": GCS_PREFIX,
            "workers": args.workers,
            "expected_total_bytes": sum(size for size, _ in FILES.values()),
            "rights_status": "BLOCKED_INTERNAL_TECHNICAL_AUDIT_ONLY",
        },
    )
    data = output / "data"
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_download, data, relative): relative
            for relative in FILES
        }
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: str(item["path"]))
    metadata = json.loads((data / f"metadata_{EPISODE_ID}.json").read_text())
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "BLOCKED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "episode_id": EPISODE_ID,
        "metadata": metadata,
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(int(item["bytes"]) for item in records),
        "rights_boundary": (
            "The official DROID site has no verified dataset license. These files are "
            "retained only for internal technical calibration audit and are not "
            "claim-eligible training or redistribution evidence."
        ),
        "mapping_boundary": (
            "This raw episode has no verified mapping to LeRobot DROID-100 episodes "
            "21, 60, or 77."
        ),
    }
    _write_json(output / "manifest.json", result)
    (output / "download.log").write_text(
        f"downloaded {result['file_count']} files / {result['total_bytes']} bytes\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

