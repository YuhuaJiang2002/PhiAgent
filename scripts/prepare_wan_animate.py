#!/usr/bin/env python3
"""Prepare the pinned Wan2.2 source and checkpoint after a storage preflight."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

WAN_COMMIT = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
SAM2_COMMIT = "0e78a118995e66bb27d78518c4bd9a3e95b4e266"
MODEL_ID = "Wan-AI/Wan2.2-Animate-14B"
MODEL_REVISION = "cb93a225fbaf1ca100f54e79da8f994995b689b3"
MODELSCOPE_REVISION = "bdcd76afebe1932ecb69916dd14ca255780f1d30"
MINIMUM_FREE_GIB = 120


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def environment_cli(*names: str) -> str | None:
    """Prefer a CLI installed beside the active Python over a system copy."""

    # Do not resolve the venv's Python symlink: resolving would jump to
    # /usr/bin and accidentally select a stale system-level CLI.
    scripts_dir = Path(sys.executable).parent
    for name in names:
        candidate = scripts_dir / name
        if candidate.is_file():
            return str(candidate)
    for name in names:
        candidate = shutil.which(name)
        if candidate is not None:
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, default=Path("external"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--source",
        choices=("huggingface", "modelscope"),
        default="huggingface",
        help="official checkpoint mirror to use",
    )
    args = parser.parse_args()
    external_root = args.external_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    for root in (external_root, checkpoint_root):
        root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(checkpoint_root).free / 1024**3
    if free_gib < MINIMUM_FREE_GIB:
        raise SystemExit(
            f"refusing checkpoint download: {free_gib:.1f} GiB free, "
            f"{MINIMUM_FREE_GIB} GiB required"
        )
    if shutil.which("git-lfs") is None:
        raise SystemExit("git-lfs is required")

    repo = external_root / "Wan2.2"
    if not repo.exists():
        run(["git", "clone", "https://github.com/Wan-Video/Wan2.2.git", str(repo)])
    run(["git", "fetch", "origin", WAN_COMMIT], cwd=repo)
    run(["git", "checkout", "--detach", WAN_COMMIT], cwd=repo)

    sam2_repo = external_root / "sam2"
    if not sam2_repo.exists():
        run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/facebookresearch/sam2.git",
                str(sam2_repo),
            ]
        )
    run(["git", "fetch", "origin", SAM2_COMMIT], cwd=sam2_repo)
    run(["git", "checkout", "--detach", SAM2_COMMIT], cwd=sam2_repo)
    if not (sam2_repo / "sam2_configs" / "sam2_hiera_l.yaml").is_file():
        raise SystemExit("pinned SAM2 checkout is missing sam2_hiera_l.yaml")

    checkpoint = checkpoint_root / "Wan2.2-Animate-14B"
    if args.source == "huggingface":
        cli = environment_cli("hf", "huggingface-cli")
        if cli is None:
            raise SystemExit("Hugging Face CLI is required for --source huggingface")
        run(
            [
                cli,
                "download",
                MODEL_ID,
                "--revision",
                MODEL_REVISION,
                "--local-dir",
                str(checkpoint),
            ]
        )
        marker = MODEL_REVISION
    else:
        cli = environment_cli("modelscope")
        if cli is None:
            raise SystemExit("modelscope CLI is required for --source modelscope")
        run(
            [
                cli,
                "download",
                MODEL_ID,
                "--revision",
                MODELSCOPE_REVISION,
                "--local-dir",
                str(checkpoint),
            ]
        )
        marker = f"modelscope:{MODELSCOPE_REVISION}"
    (checkpoint / ".phiagent-model-revision").write_text(marker + "\n")
    print(f"WAN22_REPO={repo}")
    print(f"WAN22_CHECKPOINT={checkpoint}")
    print(f"SAM2_REPO={sam2_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
