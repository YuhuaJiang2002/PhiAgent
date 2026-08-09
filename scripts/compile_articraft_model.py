#!/usr/bin/env python3
"""Compile an authored ArtiCraft SDK model without calling an LLM provider."""

from __future__ import annotations

import argparse
from pathlib import Path

from phiagent.assets import (
    ArticraftAssetGenerator,
    ArticraftConfig,
    AssetCompilationRequest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", type=Path)
    parser.add_argument("--articraft-repo", type=Path, default=Path("external/Articraft"))
    parser.add_argument(
        "--articraft-python",
        type=Path,
        default=Path("external/Articraft/.venv/bin/python"),
    )
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/articraft-sdk"))
    args = parser.parse_args()

    generator = ArticraftAssetGenerator(
        ArticraftConfig(
            repo=args.articraft_repo,
            python_executable=args.articraft_python,
        )
    )
    result = generator.compile_model(
        AssetCompilationRequest(
            model_file=args.model_file,
            experiment_root=args.experiment_root,
        )
    )
    print(f"ARTIFACT={result.artifact}")
    print(f"METADATA={result.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
