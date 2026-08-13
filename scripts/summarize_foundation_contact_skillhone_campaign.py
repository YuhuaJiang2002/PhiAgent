#!/usr/bin/env python3
"""Aggregate SkillHone splits into one provenance-complete campaign record."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return payload


def _git_state(path: Path) -> dict[str, object]:
    state: dict[str, object] = {}
    for name, command in (
        ("head", ("git", "rev-parse", "HEAD")),
        ("status", ("git", "status", "--short")),
        ("remote", ("git", "remote", "get-url", "origin")),
    ):
        result = subprocess.run(
            command,
            cwd=path,
            check=False,
            capture_output=True,
            text=True,
        )
        state[name] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    return state


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, object]:
    if not rows:
        raise ValueError("at least one score is required")
    splits = [str(row.get("split", "")) for row in rows]
    if any(not split_name for split_name in splits) or len(set(splits)) != len(splits):
        raise ValueError("score splits must be named and unique")
    total = sum(int(row["n_total"]) for row in rows)
    passed = sum(int(row["n_passed"]) for row in rows)
    strict = sum(int(row.get("n_strict", 0)) for row in rows)
    errors = sum(int(row.get("n_errors", 0)) for row in rows)
    missing = sum(int(row.get("n_no_answer", 0)) for row in rows)
    wall_proxy = sum(
        max(
            (float(trace.get("duration_s", 0.0)) for trace in row.get("traces", ())),
            default=0.0,
        )
        for row in rows
    )
    all_passed = total > 0 and passed == total and strict == total and errors == 0
    return {
        "splits": {
            str(row["split"]): {
                "passed": int(row["n_passed"]),
                "total": int(row["n_total"]),
                "strict": int(row.get("n_strict", 0)),
                "errors": int(row.get("n_errors", 0)),
                "missing_answers": int(row.get("n_no_answer", 0)),
                "score": float(row["score"]),
            }
            for row in rows
        },
        "passed": passed,
        "total": total,
        "strict": strict,
        "errors": errors,
        "missing_answers": missing,
        "strict_pass_rate": passed / total if total else 0.0,
        "measured_split_wall_seconds_proxy": wall_proxy,
        "measured_items_per_second_proxy": total / wall_proxy if wall_proxy else 0.0,
        "all_passed": all_passed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--failed-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--skill-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--skillhone-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def main() -> int:
    args = _parser().parse_args()
    score_paths = [path.expanduser().resolve() for path in args.score]
    failed_dirs = [path.expanduser().resolve() for path in args.failed_run_dir]
    skill_dir = args.skill_dir.expanduser().resolve()
    eval_dir = args.eval_dir.expanduser().resolve()
    skillhone_dir = args.skillhone_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    required_files = score_paths + [skill_dir / "SKILL.md"]
    if any(not path.is_file() for path in required_files):
        raise FileNotFoundError("one or more campaign score/skill files are missing")
    if any(not path.is_dir() for path in (eval_dir, skillhone_dir, *failed_dirs)):
        raise FileNotFoundError("one or more campaign directories are missing")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {output_dir}")
    output_dir.mkdir(parents=True)

    score_rows = [_json(path) for path in score_paths]
    summary = summarize_scores(score_rows)
    settings_path = Path.home() / ".skillhone/settings.json"
    settings_mode = oct(settings_path.stat().st_mode & 0o777) if settings_path.is_file() else None
    failed_artifacts = []
    for directory in failed_dirs:
        failed_artifacts.append(
            {
                "path": str(directory),
                "files": [
                    {
                        "path": str(path),
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in sorted(directory.rglob("*"))
                    if path.is_file()
                ],
            }
        )
    evaluated_commands = []
    eval_script = skillhone_dir / "scripts/eval.py"
    for score_path, row in zip(score_paths, score_rows, strict=True):
        evaluated_commands.append(
            [
                sys.executable,
                str(eval_script),
                "--skill-dir",
                str(skill_dir),
                "--eval-dir",
                str(eval_dir),
                "--split",
                str(row["split"]),
                "--n-probe",
                "0",
                "--output",
                str(score_path),
                "--trace-dir",
                str(score_path.parent / "traces"),
            ]
        )
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if summary["all_passed"] else "PARTIAL",
        "honest_scope": (
            "WORKING means the behavioral skill passed the current private splits; "
            "physical_model_promoted remains false."
        ),
        "physical_model_promoted": False,
        "summary": summary,
        "command": [sys.executable, *sys.argv],
        "evaluated_commands": evaluated_commands,
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "pid": os.getpid(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("requests", "numpy")
        },
        "git": {
            "project": _git_state(PROJECT_ROOT),
            "skill": _git_state(skill_dir),
            "eval": _git_state(eval_dir),
            "skillhone": _git_state(skillhone_dir),
        },
        "inputs": {
            "skill": {
                "path": str(skill_dir / "SKILL.md"),
                "sha256": _sha256(skill_dir / "SKILL.md"),
            },
            "skillhone_runtime": {
                "eval_script": {
                    "path": str(eval_script),
                    "sha256": _sha256(eval_script),
                },
                "evaluation_template": {
                    "path": str(skillhone_dir / "scripts/evaluation/template.py"),
                    "sha256": _sha256(
                        skillhone_dir / "scripts/evaluation/template.py"
                    ),
                },
            },
            "scores": [
                {"path": str(path), "sha256": _sha256(path)} for path in score_paths
            ],
            "splits": {
                str(row["split"]): {
                    "path": str(eval_dir / f"{row['split']}.jsonl"),
                    "sha256": _sha256(eval_dir / f"{row['split']}.jsonl"),
                }
                for row in score_rows
            },
            "settings": (
                {
                    "path": str(settings_path),
                    "sha256": _sha256(settings_path),
                    "mode": settings_mode,
                    "contents_redacted": True,
                }
                if settings_path.is_file()
                else None
            ),
        },
        "failed_runs_retained": failed_artifacts,
    }
    path = output_dir / "campaign-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if summary["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
