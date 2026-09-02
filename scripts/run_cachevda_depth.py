#!/usr/bin/env python3
"""Run one audited RGB-video to CacheVDA relative-depth visualization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.perception.cachevda import (  # noqa: E402
    CacheVDAConfig,
    CacheVDARequest,
    CacheVDARunner,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=12 * 1024)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-res", type=int, default=1280)
    parser.add_argument("--warmup-windows", type=int, default=1)
    parser.add_argument("--preprocess-workers", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--encoder", choices=("h264_nvenc", "libx264"), default="h264_nvenc")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    config = CacheVDAConfig(
        repository=args.repository,
        checkpoint=args.checkpoint,
        python_executable=args.python,
        gpu_index=args.gpu,
        minimum_free_gpu_mib=args.minimum_free_gpu_mib,
        input_size=args.input_size,
        max_res=args.max_res,
        warmup_windows=args.warmup_windows,
        preprocess_workers=args.preprocess_workers,
        encode_batch_size=args.encode_batch_size,
        encoder=args.encoder,
        log_every=args.log_every,
    )
    result = CacheVDARunner(config).run(
        CacheVDARequest(
            input_video=args.input_video,
            experiment_dir=args.experiment_dir,
            max_frames=args.max_frames,
        )
    )
    print(json.dumps({"status": "WORKING", **result.to_dict()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
