#!/usr/bin/env python3
"""Seed one Python process before executing a reviewed external trainer."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--training-script", type=Path, required=True)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    training_script = args.training_script.expanduser().resolve()
    if not training_script.is_file():
        raise ValueError(f"training script does not exist: {training_script}")
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    # cuBLAS needs this environment contract before torch initializes CUDA.
    # Fail closed on unsupported nondeterministic kernels below: nominally
    # identical RSI rounds must not silently produce different checkpoints.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(args.seed)
    import numpy
    import torch

    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    forwarded = args.training_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    sys.argv = [str(training_script), *forwarded]
    runpy.run_path(str(training_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
