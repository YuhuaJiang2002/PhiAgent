from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from phiagent.acceleration.sol_engine import (
    H3_MODEL_SUBFOLDER,
    H3_MODEL_REVISION,
    SOL_ENGINE_REVISION,
    SolEngineH3Config,
    SolEnginePreflightError,
    assess_h3_ab_result,
    plan_h3_ab_experiment,
    validate_matched_h3_benchmarks,
)
from phiagent.routing.model_router import (
    LLMROUTER_REVISION,
    ModelProfile,
    RouteOutcome,
    RouteRequest,
    build_llmrouter_standard_data,
    build_llmrouter_training_rows,
    route_request,
    validate_llmrouter_source,
)


def _sol_source(root: Path) -> Path:
    source = root / "sol"
    (source / "models/minimax_h3/A100").mkdir(parents=True)
    (source / "techniques/sparse_backends/sol_attn").mkdir(parents=True)
    for relative in (
        "models/minimax_h3/A100/gpu_infer.py",
        "models/minimax_h3/A100/adapter.py",
        "techniques/sparse_backends/sol_attn/__init__.py",
    ):
        (source / relative).write_text("# placeholder\n")
    (source / ".phiagent-source-revision").write_text(SOL_ENGINE_REVISION + "\n")
    return source


def test_h3_plan_locks_dense_and_sol_inputs(tmp_path: Path) -> None:
    source = _sol_source(tmp_path)
    model = tmp_path / "model"
    (model / H3_MODEL_SUBFOLDER).mkdir(parents=True)
    (model / H3_MODEL_SUBFOLDER / "model_index.json").write_text("{}")
    (model / H3_MODEL_SUBFOLDER / "transformer").mkdir()
    prompt = tmp_path / "prompt.json"
    prompt.write_text(json.dumps({"prompt": "a robot sets down a cup"}))
    plan = plan_h3_ab_experiment(
        SolEngineH3Config(
            source=source,
            model_path=model,
            output_root=tmp_path / "run",
            gpu_indices=(0, 1, 2, 3),
            prompt_file=prompt,
            seed=7,
        )
    )
    assert plan.dense_env["H3_SOL_PROFILE"] == "dense"
    assert plan.sol_env["H3_SOL_PROFILE"] == "fullopt_exact"
    for key in ("CUDA_VISIBLE_DEVICES", "H3_MODEL_REVISION", "H3_SEED", "H3_MEASURED_NUM_STEPS"):
        assert plan.dense_env[key] == plan.sol_env[key]
    assert plan.dense_env["H3_MODEL_REVISION"] == H3_MODEL_REVISION
    assert plan.dense_env["H3_MODEL_PATH"] == str(tmp_path / "run" / "model_view")
    assert plan.dense_env["H3_MODEL_SOURCE_PATH"] == str(model)
    assert plan.dense_env["H3_MODEL_SUBFOLDER"] == H3_MODEL_SUBFOLDER


def test_h3_launcher_passes_resolved_environment_into_docker(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/plan_sol_engine_h3_ab.py"
    spec = importlib.util.spec_from_file_location("h3_planner", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "run.sh"
    module._write_launcher(
        output,
        env={
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "OUT_DIR": "/tmp/out",
            "H3_CONTAINER_RUNTIME": "none",
            "H3_MODEL_PATH": "/tmp/model",
            "H3_MODEL_SOURCE_PATH": "/tmp/model-source",
            "H3_CONTAINER_IMAGE": "image",
        },
        source=tmp_path / "source",
    )
    text = output.read_text()
    for name in ("CUDA_VISIBLE_DEVICES", "OUT_DIR", "H3_CONTAINER_RUNTIME", "H3_MODEL_PATH"):
        assert f"-e {name} " in text
    assert "-e H3_MODEL_PATH \\\n+  -v " not in text
    assert "\\\n  \n  -v " not in text


def test_h3_launcher_shell_quotes_paths(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/plan_sol_engine_h3_ab.py"
    spec = importlib.util.spec_from_file_location("h3_planner_quoting", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "run.sh"
    module._write_launcher(
        output,
        env={
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "OUT_DIR": "/tmp/output with spaces",
            "H3_CONTAINER_RUNTIME": "none",
            "H3_MODEL_PATH": "/tmp/model view",
            "H3_MODEL_SOURCE_PATH": "/tmp/model source",
            "H3_CONTAINER_IMAGE": "image",
        },
        source=Path("/tmp/source tree"),
    )
    text = output.read_text()
    assert "export OUT_DIR='/tmp/output with spaces'" in text
    assert "-v '/tmp/source tree:/tmp/source tree:rw'" in text
    assert "-v '/tmp/model view:/tmp/model view:ro'" in text


def test_h3_planner_materializes_profile_output_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/plan_sol_engine_h3_ab.py"
    spec = importlib.util.spec_from_file_location("h3_planner_directories", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = _sol_source(tmp_path)
    model = tmp_path / "model"
    (model / H3_MODEL_SUBFOLDER).mkdir(parents=True)
    (model / H3_MODEL_SUBFOLDER / "model_index.json").write_text("{}")
    (model / H3_MODEL_SUBFOLDER / "transformer").mkdir()
    prompt = tmp_path / "prompt.json"
    prompt.write_text(json.dumps({"prompt": "a robot sets down a cup"}))
    output = tmp_path / "output"
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(module, "_gpu_evidence", lambda *args: {"selected": []})
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "planner",
            "--source", str(source), "--model-path", str(model), "--prompt-file", str(prompt),
            "--output-root", str(output), "--gpu-indices", "0", "1", "2", "3",
        ],
    )
    assert module.main() == 0
    assert (output / "dense").is_dir()
    assert (output / "sol_fullopt_exact").is_dir()
    assert (output / "model_view" / "model_index.json").is_symlink()
    assert (output / "model_view" / H3_MODEL_SUBFOLDER).is_symlink()
    assert (output / "model_view" / "transformer").is_symlink()


def test_quality_evidence_template_fails_closed(tmp_path: Path) -> None:
    from phiagent.acceleration.sol_engine import write_h3_quality_evidence_template

    path = tmp_path / "quality.json"
    write_h3_quality_evidence_template(path)
    evidence = json.loads(path.read_text())
    assert evidence["source"]
    assert all(value is False for key, value in evidence.items() if key != "source")


def test_matched_h3_benchmark_validator_rejects_seed_drift() -> None:
    benchmark = {
        "model": {"repo_or_path": "p", "subfolder": "FL2VA", "revision": "r", "partition": "FL2VA", "dtype": "bfloat16"},
        "workload": {"task": "t2va", "width": 1, "height": 1, "frames": 5, "duration_s": 1.0, "measured_steps": 2, "seed": 0, "prompt_sha256": "x"},
        "topology": {"num_gpus": 4, "tensor_parallel": 1, "ulysses": 4, "fsdp_inference": True},
    }
    matched, reasons = validate_matched_h3_benchmarks(benchmark, benchmark)
    assert matched and not reasons
    drift = {**benchmark, "workload": {**benchmark["workload"], "seed": 1}}
    matched, reasons = validate_matched_h3_benchmarks(benchmark, drift)
    assert not matched
    assert "mismatched workload.seed" in reasons


def test_h3_plan_rejects_unpinned_source(tmp_path: Path) -> None:
    source = _sol_source(tmp_path)
    (source / ".phiagent-source-revision").write_text("wrong\n")
    model = tmp_path / "model"
    (model / H3_MODEL_SUBFOLDER).mkdir(parents=True)
    (model / H3_MODEL_SUBFOLDER / "model_index.json").write_text("{}")
    prompt = tmp_path / "prompt.json"
    prompt.write_text("{}")
    with pytest.raises(SolEnginePreflightError, match="expected"):
        plan_h3_ab_experiment(
            SolEngineH3Config(source, model, tmp_path / "run", (0, 1, 2, 3), prompt)
        )


def test_h3_acceptance_needs_speed_sparse_evidence_and_quality_gates() -> None:
    common = {
        "model": {
            "repo_or_path": "p",
            "subfolder": "FL2VA",
            "revision": "r",
            "partition": "FL2VA",
            "dtype": "bfloat16",
        },
        "workload": {
            "task": "t2va",
            "width": 1,
            "height": 1,
            "frames": 5,
            "duration_s": 1.0,
            "measured_steps": 2,
            "seed": 0,
            "prompt_sha256": "x",
        },
        "topology": {
            "num_gpus": 4,
            "tensor_parallel": 1,
            "ulysses": 4,
            "fsdp_inference": True,
        },
    }
    dense = {**common, "measured": {"inference_time_s": 100.0}}
    sol = {
        **common,
        "measured": {"inference_time_s": 70.0},
        "execution": {"measured_sparse_ranks": [0, 1, 2, 3]},
    }
    gates = {
        "source": "matched-video evaluator",
        "matched_inputs": True,
        "automated_quality_passed": True,
        "temporal_consistency_passed": True,
        "action_consistency_passed": True,
        "physical_gate_passed": True,
        "human_review_passed": True,
    }
    result = assess_h3_ab_result(dense, sol, gates)
    assert result.accepted
    assert result.speedup == pytest.approx(100 / 70)
    rejected = assess_h3_ab_result(dense, sol, {**gates, "human_review_passed": False})
    assert not rejected.accepted
    assert "human_review_passed" in " ".join(rejected.reasons)
    mismatched = {**sol, "workload": {**common["workload"], "seed": 1}}
    rejected = assess_h3_ab_result(dense, mismatched, gates)
    assert not rejected.accepted
    assert "benchmark workload mismatch" in " ".join(rejected.reasons)


def test_router_hard_gates_precede_latency() -> None:
    profiles = (
        ModelProfile("joy_preview", frozenset({"edit"}), 10, 1, physical_gate=False),
        ModelProfile("wan_verified", frozenset({"edit", "action"}), 50, 2, physical_gate=True),
    )
    decision = route_request(
        RouteRequest(frozenset({"edit", "action"}), minimum_quality_tier=2, requires_physical_gate=True),
        profiles,
    )
    assert decision.selected is not None
    assert decision.selected.name == "wan_verified"
    assert decision.rejected["joy_preview"] == "missing_required_capability"


def test_llmrouter_source_validation_uses_pinned_marker(tmp_path: Path) -> None:
    source = tmp_path / "llmrouter"
    (source / "llmrouter/models").mkdir(parents=True)
    (source / "llmrouter/models/meta_router.py").write_text("# placeholder\n")
    (source / ".phiagent-source-revision").write_text(LLMROUTER_REVISION + "\n")
    assert validate_llmrouter_source(source) == LLMROUTER_REVISION


def test_llmrouter_rows_only_label_fully_accepted_candidates() -> None:
    common = {"instruction": "move cup left", "task": "robot_edit"}
    rows = build_llmrouter_training_rows(
        (
            RouteOutcome("r1", "fast_unverified", common, 5.0, True, True, True, False, True),
            RouteOutcome("r1", "slower_verified", common, 20.0, True, True, True, True, True),
            RouteOutcome("r2", "rejected", common, 1.0, False, False, False, False, False),
        )
    )
    assert rows[0]["oracle_profile"] == "slower_verified"
    assert rows[0]["oracle_latency_seconds"] == 20.0
    assert rows[1]["oracle_profile"] is None


def test_llmrouter_standard_export_excludes_unaccepted_labels() -> None:
    accepted = RouteOutcome(
        "accepted", "wan", {"instruction": "move cup left", "task": "robot_edit"},
        1.0, True, True, True, True, True,
    )
    rejected = RouteOutcome(
        "rejected", "fast_preview", {"instruction": "move cup right"},
        0.1, True, True, False, False, False,
    )
    queries, labels, unlabeled = build_llmrouter_standard_data((accepted, rejected))
    assert [item["query_id"] for item in queries] == ["accepted", "rejected"]
    assert labels == ({"query_id": "accepted", "best_model": "wan"},)
    assert unlabeled == ("rejected",)
