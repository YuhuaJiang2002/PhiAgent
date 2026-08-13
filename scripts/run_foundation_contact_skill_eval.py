#!/usr/bin/env python3
"""Run a local-model behavior eval with complete experiment provenance."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.agent.foundation_contact_skill_eval import (  # noqa: E402
    evaluate_items,
    load_items,
    ollama_completion,
    sha256_file,
)


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--split", default="probe")
    parser.add_argument("--n", type=int, default=0, help="0 evaluates the whole split")
    parser.add_argument("--model", default="qwen3:4b-instruct-2507-q4_K_M")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _package_versions() -> list[str]:
    return sorted(
        f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )


def _ollama_model_record(base_url: str, model: str, timeout_s: float) -> dict:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=timeout_s) as response:
        payload = json.load(response)
    for record in payload.get("models", []):
        if record.get("name") == model or record.get("model") == model:
            return record
    return {"name": model, "available": False}


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "behavior-eval.json"
    if manifest_path.exists():
        raise FileExistsError(f"experiment already exists: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    skill_path = args.skill_dir.resolve() / "SKILL.md"
    split_path = args.eval_dir.resolve() / f"{args.split}.jsonl"
    skill_text = skill_path.read_text(encoding="utf-8")
    items = load_items(split_path, limit=args.n)
    completion = ollama_completion(
        base_url=args.ollama_url,
        model=args.model,
        timeout_s=args.timeout,
        seed=args.seed,
        num_predict=args.num_predict,
    )
    evaluation = evaluate_items(
        items,
        skill_text=skill_text,
        completion=completion,
        max_retries=args.retries,
    )

    manifest = {
        "schema_version": 1,
        "evaluation": evaluation,
        "config": {
            "model": args.model,
            "ollama_url": args.ollama_url,
            "split": args.split,
            "n": args.n,
            "timeout_s": args.timeout,
            "seed": args.seed,
            "num_predict": args.num_predict,
            "retries": args.retries,
        },
        "inputs": {
            "skill_path": str(skill_path),
            "skill_sha256": sha256_file(skill_path),
            "split_path": str(split_path),
            "split_sha256": sha256_file(split_path),
        },
        "provenance": {
            "command": [sys.executable, *sys.argv],
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "package_versions": _package_versions(),
            "ollama_model": _ollama_model_record(args.ollama_url, args.model, args.timeout),
            "pid": os.getpid(),
            "project_git_head": _git(PROJECT_ROOT, "rev-parse", "HEAD"),
            "project_git_status": _git(PROJECT_ROOT, "status", "--short"),
            "skill_git_head": _git(args.skill_dir.resolve(), "rev-parse", "HEAD"),
            "eval_git_head": _git(args.eval_dir.resolve(), "rev-parse", "HEAD"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "status": evaluation["status"],
                "score": evaluation["score"],
                "passed": evaluation["passed"],
                "total": evaluation["total"],
                "items_per_second": evaluation["items_per_second"],
                "generation_tokens_per_second": evaluation["generation_tokens_per_second"],
                "physical_model_promoted": False,
            },
            indent=2,
        )
    )
    return 0 if evaluation["status"] == "WORKING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
