#!/usr/bin/env python3
"""Prepare pinned official Wan-Animate-2 source and optional checkpoint."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.wan_animate2 import (  # noqa: E402
    WAN_ANIMATE2_COMMIT,
    WAN_ANIMATE2_MODEL_ID,
    WAN_ANIMATE2_MODEL_REVISION,
    WAN_ANIMATE2_MODELSCOPE_REVISION,
    verify_wan_animate2_checkpoint,
    verify_wan_animate2_source,
)


def _run(command: list[str], cwd: Path | None = None) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, default=Path("external"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument(
        "--source",
        choices=("huggingface", "modelscope"),
        default="huggingface",
    )
    args = parser.parse_args()
    external_root = args.external_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    external_root.mkdir(parents=True, exist_ok=True)
    repo = external_root / "Wan-Animate-2"
    if not repo.exists():
        _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/Wan-Video/Wan-Animate-2.git",
                str(repo),
            ]
        )
    _run(["git", "fetch", "origin", WAN_ANIMATE2_COMMIT], cwd=repo)
    _run(["git", "checkout", "--detach", WAN_ANIMATE2_COMMIT], cwd=repo)
    verify_wan_animate2_source(repo)

    checkpoint = checkpoint_root / "Wan2.2-Animate-2-14B"
    if args.download_model:
        if shutil.disk_usage(checkpoint_root).free < 120 * 1024**3:
            raise SystemExit("at least 120 GiB free is required before model download")
        if args.source == "huggingface":
            cli = shutil.which("hf") or shutil.which("huggingface-cli")
            if cli is None:
                raise SystemExit("Hugging Face CLI is required for --source huggingface")
            command = [
                cli,
                "download",
                WAN_ANIMATE2_MODEL_ID,
                "--revision",
                WAN_ANIMATE2_MODEL_REVISION,
                "--local-dir",
                str(checkpoint),
            ]
            marker = WAN_ANIMATE2_MODEL_REVISION
        else:
            cli = shutil.which("modelscope")
            if cli is None:
                raise SystemExit("ModelScope CLI is required for --source modelscope")
            command = [
                cli,
                "download",
                "--model",
                WAN_ANIMATE2_MODEL_ID,
                "--revision",
                WAN_ANIMATE2_MODELSCOPE_REVISION,
                "--local_dir",
                str(checkpoint),
            ]
            marker = f"modelscope:{WAN_ANIMATE2_MODELSCOPE_REVISION}"
        _run(command)
        (checkpoint / ".phiagent-model-revision").write_text(marker + "\n")
        verify_wan_animate2_checkpoint(checkpoint)
    print(f"WAN_ANIMATE2_REPO={repo}")
    if checkpoint.exists():
        print(f"WAN_ANIMATE2_CHECKPOINT={checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
