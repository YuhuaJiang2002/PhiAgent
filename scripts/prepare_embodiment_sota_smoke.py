#!/usr/bin/env python3
"""Prepare immutable Hand2Dex inputs for matched embodiment-transfer baselines."""

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


def reference_cases(
    reference_root: Path,
    manifest: dict[str, object],
) -> tuple[dict[str, object], ...]:
    """Validate the three pinned source/transferred pairs."""

    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise ValueError("PhiZero reference manifest requires an assets array")
    indexed = {}
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("PhiZero assets must be objects")
        key = int(asset["case"]), str(asset["role"])
        if key in indexed:
            raise ValueError(f"duplicate PhiZero asset {key}")
        path = reference_root / str(asset["name"])
        if not path.is_file() or path.stat().st_size != int(asset["size_bytes"]):
            raise ValueError(f"PhiZero asset is missing or has wrong size: {path}")
        if _sha256(path) != asset["sha256"]:
            raise ValueError(f"PhiZero asset hash mismatch: {path}")
        indexed[key] = {"path": path, "asset": asset}
    cases = []
    for case in (1, 2, 3):
        try:
            source = indexed[(case, "source")]
            transferred = indexed[(case, "transferred")]
        except KeyError as exc:
            raise ValueError(f"PhiZero case {case} is incomplete") from exc
        cases.append(
            {
                "case": case,
                "source_video": str(source["path"]),
                "source_sha256": source["asset"]["sha256"],
                "reference_video": str(transferred["path"]),
                "reference_sha256": transferred["asset"]["sha256"],
            }
        )
    return tuple(cases)


def _git_state(root: Path, commit: str | None, branch: str | None) -> dict[str, object]:
    if (commit is None) != (branch is None):
        raise ValueError("--git-commit and --git-branch must be supplied together")
    if commit is not None:
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ValueError("--git-commit must be a lowercase 40-character SHA-1")
        return {
            "commit": commit,
            "branch": branch,
            "resolution": "explicit immutable source snapshot",
            "dirty": True,
        }
    return {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "branch": subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "resolution": "local Git checkout",
        "dirty": subprocess.run(
            ["git", "diff", "--quiet"], cwd=root, check=False
        ).returncode
        != 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    parser.add_argument("--seed", type=int, action="append", default=[42])
    parser.add_argument("--git-commit")
    parser.add_argument("--git-branch")
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    reference_root = args.reference_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite embodiment smoke inputs: {output}")
    if not ffmpeg.is_file():
        raise ValueError(f"ffmpeg is missing: {ffmpeg}")
    manifest_path = reference_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"PhiZero reference manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("PhiZero reference manifest must contain an object")
    cases = reference_cases(reference_root, manifest)
    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    command = [sys.executable, *sys.argv]
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    _write_json(
        output / "config.json",
        {
            "schema_version": "1.0.0",
            "status": "STARTED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reference_root": str(reference_root),
            "reference_manifest_sha256": _sha256(manifest_path),
            "seeds": sorted(set(args.seed)),
            "ffmpeg": str(ffmpeg),
        },
    )
    _write_json(
        output / "git-state.json",
        _git_state(project_root, args.git_commit, args.git_branch),
    )
    planned = []
    log_lines = []
    for case in cases:
        target = inputs / f"hand2dex-{case['case']}-official-target-frame.png"
        extraction = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(case["reference_video"]),
            "-frames:v",
            "1",
            str(target),
        ]
        completed = subprocess.run(
            extraction,
            capture_output=True,
            text=True,
            check=False,
        )
        log_lines.append(f"{shlex.join(extraction)}\n{completed.stderr}")
        if completed.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"failed to extract target frame for case {case['case']}")
        for seed in sorted(set(args.seed)):
            planned.append(
                {
                    **case,
                    "target_image": str(target),
                    "target_sha256": _sha256(target),
                    "seed": seed,
                    "prompt": (
                        "A Sharpa Wave dexterous robot hand performs the demonstrated "
                        "object manipulation while preserving the scene and object."
                    ),
                }
            )
    cases_path = output / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in planned))
    (output / "prepare.log").write_text("\n".join(log_lines))
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "PARTIAL",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cases": len(cases),
        "jobs": len(planned),
        "seeds": sorted(set(args.seed)),
        "cases_jsonl": str(cases_path),
        "cases_jsonl_sha256": _sha256(cases_path),
        "reference_manifest": str(manifest_path),
        "reference_manifest_sha256": _sha256(manifest_path),
        "claim_boundary": (
            "These three public Hand2Dex pairs are a harness smoke only. They cannot "
            "establish SOTA, physical contact, or an exact PhiZero reproduction."
        ),
    }
    _write_json(output / "manifest.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

