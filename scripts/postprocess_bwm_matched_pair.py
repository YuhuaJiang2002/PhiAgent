#!/usr/bin/env python3
"""Discover and evaluate a completed watched BWM baseline/candidate pair."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def find_campaign(root: Path) -> Path:
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "manifest.json").is_file()
        and (path / "videos").is_dir()
    )
    if len(candidates) != 1:
        raise ValueError(f"expected one complete BWM campaign under {root}, found {len(candidates)}")
    return candidates[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watcher-root", type=Path, required=True)
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = _parser().parse_args()
    watcher = args.watcher_root.expanduser().resolve()
    result_path = watcher / "result.json"
    if not result_path.is_file():
        raise ValueError(f"BWM watcher result is missing: {result_path}")
    result = json.loads(result_path.read_text())
    if result.get("status") != "WORKING":
        raise ValueError("BWM watcher did not complete both matched arms")
    baseline = find_campaign(watcher / "runs" / "official-bwm")
    candidate = find_campaign(watcher / "runs" / "promoted-adapter")
    output = args.output_dir.expanduser().resolve()
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "evaluate_bwm_counterfactual_audit.py"),
        "--suite-manifest",
        str(args.suite_manifest.expanduser().resolve()),
        "--run",
        "official-bwm",
        str(args.seed),
        str(baseline),
        "--run",
        "promoted-adapter",
        str(args.seed),
        str(candidate),
        "--output-dir",
        str(output),
        "--candidate-model",
        "promoted-adapter",
        "--baseline-model",
        "official-bwm",
        "--minimum-independent-trials",
        "20",
        "--bootstrap-iterations",
        "10000",
        "--confidence",
        "0.95",
        "--seed",
        "20260812",
        "--git-commit",
        "ac37eb606fffef1b9e7bed9b93d9b64dd4d78fd3",
        "--git-branch",
        "codex/acwm-native-articulation-demo",
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

