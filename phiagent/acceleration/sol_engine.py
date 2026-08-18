"""Thin, dependency-free contract for running Sol-Engine MiniMax-H3 A/B tests.

Sol-Attn is an external runtime optimization, not a PhiAgent model feature.  This
module deliberately does not import Torch, SGLang, or Sol-Engine.  It makes an
experiment reproducible and refuses to label a sparse run successful unless its
matched dense control, runtime evidence, and quality/physical gates are present.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SOL_ENGINE_REPOSITORY = "https://github.com/NVlabs/Sana.git"
SOL_ENGINE_REVISION = "6fb7eb11c3435555ec6d6adf0d5572d339d2c6eb"
H3_A100_RUNTIME = "models/minimax_h3/A100"
H3_MODEL_ID = "MiniMaxAI/MiniMax-H3"
H3_MODEL_REVISION = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
H3_MODEL_SUBFOLDER = "FL2VA"


class SolEnginePreflightError(RuntimeError):
    """Raised when a requested external runtime is not the audited source."""


@dataclass(frozen=True)
class SolEngineH3Config:
    """Invariant settings for an apples-to-apples 4x A800 / A100 H3 test."""

    source: Path
    model_path: Path
    output_root: Path
    gpu_indices: tuple[int, int, int, int]
    prompt_file: Path
    seed: int = 0
    steps: int = 50
    duration_seconds: float = 5.166667
    model_revision: str = H3_MODEL_REVISION
    container_image: str = "lmsysorg/sglang:nightly-dev-cu13-20260803-12eadf86"

    def validate(self) -> None:
        if len(self.gpu_indices) != 4 or len(set(self.gpu_indices)) != 4:
            raise ValueError("H3 Sol-Engine A100 profile requires exactly four distinct GPUs")
        if any(index < 0 for index in self.gpu_indices):
            raise ValueError("GPU indices must be non-negative physical indices")
        if self.steps <= 0 or self.duration_seconds <= 0 or self.seed < 0:
            raise ValueError("seed, steps, and duration_seconds must be positive")
        if not self.model_revision:
            raise ValueError("model_revision is required")


@dataclass(frozen=True)
class H3ABPlan:
    """Commands and immutable comparison fields for the dense/Sol pair."""

    dense_env: dict[str, str]
    sol_env: dict[str, str]
    dense_output: Path
    sol_output: Path
    immutable_fields: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "dense_env": self.dense_env,
            "sol_env": self.sol_env,
            "dense_output": str(self.dense_output),
            "sol_output": str(self.sol_output),
            "immutable_fields": self.immutable_fields,
        }


@dataclass(frozen=True)
class H3ABResult:
    """Evidence-based decision; ``accepted`` never means physics is proven."""

    accepted: bool
    speedup: float | None
    reasons: tuple[str, ...]
    dense_seconds: float | None
    sol_seconds: float | None


def _source_revision(source: Path) -> str:
    marker = source / ".phiagent-source-revision"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    if (source / ".git").is_dir():
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True, capture_output=True, check=False
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    return ""


def validate_sol_engine_source(source: Path) -> str:
    """Check the pinned external checkout without importing its heavy runtime."""

    source = source.expanduser().resolve()
    required = (
        source / H3_A100_RUNTIME / "gpu_infer.py",
        source / H3_A100_RUNTIME / "adapter.py",
        source / "techniques" / "sparse_backends" / "sol_attn" / "__init__.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SolEnginePreflightError(f"Sol-Engine source is missing required files: {missing}")
    revision = _source_revision(source)
    if revision != SOL_ENGINE_REVISION:
        raise SolEnginePreflightError(
            f"Sol-Engine revision is {revision or 'unreadable'}, expected {SOL_ENGINE_REVISION}"
        )
    return revision


def plan_h3_ab_experiment(config: SolEngineH3Config) -> H3ABPlan:
    """Build a strict control/optimized pair, with all generation fields locked."""

    config.validate()
    validate_sol_engine_source(config.source)
    model_path = config.model_path.expanduser().resolve()
    prompt_file = config.prompt_file.expanduser().resolve()
    if not model_path.is_dir():
        raise SolEnginePreflightError(f"H3 model path is missing: {model_path}")
    # The pinned MiniMax snapshot stores the Diffusers entry point under
    # ``FL2VA``.  The runtime model view created by the launcher exposes its
    # model_index at the view root (for SGLang's early probe) and the FL2VA
    # partition beneath it (for the H3 pipeline's partition check).
    model_partition = model_path / H3_MODEL_SUBFOLDER
    if not (model_partition / "model_index.json").is_file():
        raise SolEnginePreflightError(
            f"H3 model path lacks {H3_MODEL_SUBFOLDER}/model_index.json: {model_path}"
        )
    if not prompt_file.is_file():
        raise SolEnginePreflightError(f"H3 prompt file is missing: {prompt_file}")
    common = {
        "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in config.gpu_indices),
        "H3_CONTAINER_RUNTIME": "none",
        "H3_MODEL_PATH": str(config.output_root.expanduser().resolve() / "model_view"),
        "H3_MODEL_SOURCE_PATH": str(model_path),
        "H3_MODEL_SUBFOLDER": H3_MODEL_SUBFOLDER,
        "H3_MODEL_REVISION": config.model_revision,
        "H3_PROMPT_FILE": str(prompt_file),
        "H3_MEASURED_NUM_STEPS": str(config.steps),
        "H3_WARMUP_NUM_STEPS": str(config.steps),
        "H3_DURATION_SECONDS": str(config.duration_seconds),
        "H3_SEED": str(config.seed),
        "H3_WARMUP_SEED": str(config.seed + 10_000),
        "H3_CONTAINER_IMAGE": config.container_image,
        "HF_HUB_OFFLINE": "1",
        "SOL_ATTN_STRICT": "1",
    }
    root = config.output_root.expanduser().resolve()
    return H3ABPlan(
        dense_env={**common, "H3_SOL_PROFILE": "dense", "OUT_DIR": str(root / "dense")},
        sol_env={
            **common,
            "H3_SOL_PROFILE": "fullopt_exact",
            "OUT_DIR": str(root / "sol_fullopt_exact"),
        },
        dense_output=root / "dense",
        sol_output=root / "sol_fullopt_exact",
        immutable_fields={
            "source_revision": SOL_ENGINE_REVISION,
            "model_id": H3_MODEL_ID,
            "model_revision": config.model_revision,
            "model_source_path": str(model_path),
            "gpu_indices": list(config.gpu_indices),
            "seed": config.seed,
            "steps": config.steps,
            "duration_seconds": config.duration_seconds,
            "prompt_file": str(prompt_file),
        },
    )


def write_h3_ab_plan(plan: H3ABPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_h3_quality_evidence_template(path: Path) -> None:
    """Write an explicitly failing template for the required matched-video review.

    It is intentional that every gate starts false: a completed generation does
    not become an accepted physical result until an evaluator supplies evidence.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    template = {
        "source": "fill with the matched-video evaluator and review artifact",
        "matched_inputs": False,
        "automated_quality_passed": False,
        "temporal_consistency_passed": False,
        "action_consistency_passed": False,
        "physical_gate_passed": False,
        "human_review_passed": False,
    }
    path.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _inference_seconds(benchmark: Mapping[str, Any]) -> float | None:
    measured = benchmark.get("measured")
    if isinstance(measured, Mapping):
        value = measured.get("inference_time_s")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    value = benchmark.get("inference_time_s")
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def validate_matched_h3_benchmarks(
    dense_benchmark: Mapping[str, Any], sol_benchmark: Mapping[str, Any]
) -> tuple[bool, tuple[str, ...]]:
    """Verify the workload fields that must be equal before video comparison."""

    reasons: list[str] = []
    for section, fields in (
        ("model", ("repo_or_path", "subfolder", "revision", "partition", "dtype")),
        (
            "workload",
            (
                "task",
                "width",
                "height",
                "frames",
                "duration_s",
                "measured_steps",
                "seed",
                "prompt_sha256",
            ),
        ),
        ("topology", ("num_gpus", "tensor_parallel", "ulysses", "fsdp_inference")),
    ):
        left = dense_benchmark.get(section)
        right = sol_benchmark.get(section)
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            reasons.append(f"missing {section} section")
            continue
        for field in fields:
            if left.get(field) != right.get(field):
                reasons.append(f"mismatched {section}.{field}")
    return not reasons, tuple(reasons)


def assess_h3_ab_result(
    dense_benchmark: Mapping[str, Any],
    sol_benchmark: Mapping[str, Any],
    quality_evidence: Mapping[str, Any],
    *,
    minimum_speedup: float = 1.15,
) -> H3ABResult:
    """Accept only a real measured speedup with explicit video/physical gates.

    ``quality_evidence`` must come from an actual matched-video evaluation.  Its
    required booleans intentionally prevent a timing-only run from becoming a
    quality claim.
    """

    if minimum_speedup <= 1:
        raise ValueError("minimum_speedup must be greater than one")
    dense_seconds = _inference_seconds(dense_benchmark)
    sol_seconds = _inference_seconds(sol_benchmark)
    reasons: list[str] = []
    matched_inputs, mismatch_reasons = validate_matched_h3_benchmarks(
        dense_benchmark, sol_benchmark
    )
    if not matched_inputs:
        reasons.append("benchmark workload mismatch: " + ", ".join(mismatch_reasons))
    speedup = None
    if dense_seconds is None or sol_seconds is None:
        reasons.append("missing measured inference_time_s in one or both benchmark files")
    else:
        speedup = dense_seconds / sol_seconds
        if speedup < minimum_speedup:
            reasons.append(f"speedup {speedup:.3f}x is below required {minimum_speedup:.3f}x")
    execution = sol_benchmark.get("execution")
    if not isinstance(execution, Mapping) or not execution.get("measured_sparse_ranks"):
        reasons.append("Sol benchmark lacks evidence that sparse attention ran on measured ranks")
    required_gates = (
        "matched_inputs",
        "automated_quality_passed",
        "temporal_consistency_passed",
        "action_consistency_passed",
        "physical_gate_passed",
        "human_review_passed",
    )
    missing_or_false = [name for name in required_gates if quality_evidence.get(name) is not True]
    if missing_or_false:
        reasons.append("quality/physical evidence missing or failed: " + ", ".join(missing_or_false))
    source = quality_evidence.get("source")
    if not isinstance(source, str) or not source.strip():
        reasons.append("quality evidence must name the evaluation source")
    return H3ABResult(
        accepted=not reasons,
        speedup=speedup,
        reasons=tuple(reasons),
        dense_seconds=dense_seconds,
        sol_seconds=sol_seconds,
    )
