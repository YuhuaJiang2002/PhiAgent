#!/usr/bin/env python3
"""Generate one isolated articulated asset with the optional ArtiCraft adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from phiagent.assets import (
    ArticraftAssetGenerator,
    ArticraftConfig,
    AssetGenerationRequest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("description")
    parser.add_argument("--articraft-repo", type=Path, default=Path("external/Articraft"))
    parser.add_argument(
        "--articraft-python",
        type=Path,
        default=Path("external/Articraft/.venv/bin/python"),
    )
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/articraft"))
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument(
        "--provider",
        choices=("openai", "gemini", "anthropic", "openrouter"),
        default="openai",
    )
    parser.add_argument("--model")
    args = parser.parse_args()

    generator = ArticraftAssetGenerator(
        ArticraftConfig(
            repo=args.articraft_repo,
            python_executable=args.articraft_python,
        )
    )
    result = generator.generate(
        AssetGenerationRequest(
            description=args.description,
            experiment_root=args.experiment_root,
            reference_image=args.reference_image,
            provider=args.provider,
            model=args.model,
        )
    )
    print(f"ARTIFACT={result.artifact}")
    print(f"METADATA={result.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
