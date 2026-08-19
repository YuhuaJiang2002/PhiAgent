#!/usr/bin/env python3
"""Run the action-carrier JoyAI SC3-inspired rendering demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.world_model.joyai_sc3 import (  # noqa: E402
    JoyAISC3Runner,
    load_config,
    with_runtime_overrides,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Validate the real action carrier, compile the prompt, create the exact "
            "JoyAI input, and save candidate commands without contacting a GPU service."
        ),
    )
    parser.add_argument(
        "--client-python",
        type=Path,
        help="JoyAI client interpreter; its virtualenv symlink is preserved.",
    )
    parser.add_argument(
        "--evaluator-python",
        type=Path,
        help="Interpreter containing the configured inverse visual evaluator dependencies.",
    )
    parser.add_argument(
        "--server-manifest",
        type=Path,
        help=(
            "WORKING joyai_server_ready manifest from "
            "scripts/launch_joyai_video_edit_server.py; required for inference."
        ),
    )
    parser.add_argument("--server-url", help="Override the checked-in WebSocket URL.")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        help="Override only the fresh experiment root; checked-in action/model settings remain frozen.",
    )
    parser.add_argument(
        "--candidate-seed",
        type=int,
        action="append",
        help="Run a subset of the frozen seed set; repeat for multiple seeds.",
    )
    parser.add_argument(
        "--source-git-state",
        type=Path,
        help="Hash-bound Git-state JSON from the source workspace when running a staged bundle.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = with_runtime_overrides(
        load_config(config_path),
        client_python=args.client_python,
        evaluator_python=args.evaluator_python,
        server_manifest=args.server_manifest,
        server_url=args.server_url,
        source_git_state=args.source_git_state,
        experiment_root=args.experiment_root,
        candidate_seeds=args.candidate_seed,
    )
    result = JoyAISC3Runner(
        config,
        config_path=config_path,
        project_root=PROJECT_ROOT,
    ).run(prepare_only=args.prepare_only)
    print(
        json.dumps(
            {
                "status": result["status"],
                "stage": result["stage"],
                "experiment_dir": str(Path(result["packages"]["path"]).parent),
                "model_inference": result.get("model_inference"),
                "selection": result.get("selection"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "WORKING" or args.prepare_only else 2


if __name__ == "__main__":
    raise SystemExit(main())
