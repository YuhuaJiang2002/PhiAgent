#!/usr/bin/env python3
"""Create a reproducible, safe 4x-A800 Sol-Engine MiniMax-H3 A/B bundle.

The script is intentionally a planner: it inspects the requested physical GPUs
before emitting the two Docker launch scripts, but it never silently falls back
to dense attention or declares the resulting videos equivalent.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acceleration.sol_engine import (  # noqa: E402
    SolEngineH3Config,
    plan_h3_ab_experiment,
    write_h3_ab_plan,
    write_h3_quality_evidence_template,
)
from phiagent.rendering.wan_animate import query_gpus  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-indices", type=int, nargs=4, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--duration-seconds", type=float, default=5.166667)
    parser.add_argument("--minimum-free-mib", type=int, default=70 * 1024)
    return parser


def _gpu_evidence(indices: tuple[int, int, int, int], minimum_free_mib: int) -> dict[str, object]:
    gpus, raw_inventory, raw_processes = query_gpus()
    by_index = {gpu.physical_index: gpu for gpu in gpus}
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise ValueError(f"requested physical GPUs are unavailable: {missing}")
    selected = [by_index[index] for index in indices]
    busy = [gpu for gpu in selected if gpu.free_mib < minimum_free_mib]
    if busy:
        state = ", ".join(f"GPU {gpu.physical_index}: {gpu.free_mib} MiB free" for gpu in busy)
        raise ValueError(f"requested GPUs do not meet {minimum_free_mib} MiB free: {state}")
    return {
        "selected": [
            {
                "physical_index": gpu.physical_index,
                "name": gpu.name,
                "total_mib": gpu.total_mib,
                "used_mib": gpu.used_mib,
                "free_mib": gpu.free_mib,
            }
            for gpu in selected
        ],
        "nvidia_smi_inventory": raw_inventory,
        "nvidia_smi_processes": raw_processes,
    }


def _write_launcher(path: Path, *, env: dict[str, str], source: Path) -> None:
    devices = env["CUDA_VISIBLE_DEVICES"]
    assignments = "\n".join(
        f"export {name}={shlex.quote(value)}" for name, value in sorted(env.items())
    )
    # ``docker run`` does not inherit arbitrary host exports.  The external H3
    # entrypoint reads OUT_DIR and all H3_* fields inside the container, so pass
    # the resolved allowlist explicitly rather than relying on host state.
    container_env = "".join(f"  -e {name} \\\n" for name in sorted(env))
    source_mount = shlex.quote(f"{source}:{source}:rw")
    model_view_mount = shlex.quote(f"{env['H3_MODEL_PATH']}:{env['H3_MODEL_PATH']}:ro")
    model_source_mount = shlex.quote(
        f"{env['H3_MODEL_SOURCE_PATH']}:{env['H3_MODEL_SOURCE_PATH']}:ro"
    )
    output_parent = Path(env["OUT_DIR"]).parent
    output_mount = shlex.quote(f"{output_parent}:{output_parent}:rw")
    workdir = shlex.quote(str(source))
    image = shlex.quote(env["H3_CONTAINER_IMAGE"])
    entrypoint = shlex.quote(str(source / "models/minimax_h3/A100/run_minimax_h3_gpu.sh"))
    text = f"""#!/usr/bin/env bash
set -euo pipefail
{assignments}
mkdir -p \"$OUT_DIR\"
exec docker run --rm --gpus '"device={devices}"' \\
  --ipc=host --shm-size=64g \\
{container_env}  -v {source_mount} \\
  -v {model_view_mount} \\
  -v {model_source_mount} \\
  -v {output_mount} \\
  -w {workdir} \\
  {image} \\
  bash {entrypoint}
"""
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _write_h3_model_view(view: Path, source_root: Path) -> None:
    """Expose H3's nested FL2VA partition through SGLang's two path checks.

    This writes only symlinks, never copies model weights.  ``model_index`` is
    required at the root for generic Diffusers discovery, while MiniMax's
    pipeline subsequently requires the named FL2VA partition.
    """

    partition = source_root / "FL2VA"
    index = partition / "model_index.json"
    if not index.is_file():
        raise ValueError(f"H3 source partition is incomplete: {index}")
    view.mkdir(parents=True, exist_ok=False)
    # The generic Diffusers probe additionally validates the component names
    # next to the root index. Mirror the partition's layout with symlinks;
    # each target remains the single original checkpoint on JuiceFS.
    for child in partition.iterdir():
        (view / child.name).symlink_to(child, target_is_directory=child.is_dir())
    (view / "FL2VA").symlink_to(partition, target_is_directory=True)


def main() -> int:
    args = _parser().parse_args()
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required for the pinned Sol-Engine A100 runtime")
    indices = tuple(args.gpu_indices)
    config = SolEngineH3Config(
        source=args.source,
        model_path=args.model_path,
        output_root=args.output_root,
        gpu_indices=indices,
        prompt_file=args.prompt_file,
        seed=args.seed,
        steps=args.steps,
        duration_seconds=args.duration_seconds,
    )
    evidence = _gpu_evidence(indices, args.minimum_free_mib)
    plan = plan_h3_ab_experiment(config)
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    _write_h3_model_view(root / "model_view", config.model_path.expanduser().resolve())
    write_h3_ab_plan(plan, root / "plan.json")
    write_h3_quality_evidence_template(root / "quality_evidence.template.json")
    # Callers commonly redirect each profile's stdout before invoking the
    # launcher.  Materialize these paths as part of the plan so that such
    # redirection cannot prevent either profile from starting.
    plan.dense_output.mkdir(parents=True, exist_ok=False)
    plan.sol_output.mkdir(parents=True, exist_ok=False)
    _write_launcher(root / "run_dense.sh", env=plan.dense_env, source=config.source.resolve())
    _write_launcher(root / "run_sol_fullopt_exact.sh", env=plan.sol_env, source=config.source.resolve())
    (root / "gpu_selection.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "minimum_free_mib": args.minimum_free_mib,
                **evidence,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "planned", "output_root": str(root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
