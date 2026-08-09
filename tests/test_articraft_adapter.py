from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from phiagent.assets.articraft import (
    ARTICRAFT_COMMIT,
    ArticraftAssetGenerator,
    ArticraftConfig,
    AssetGenerationError,
    _absolute_without_resolving,
)
from phiagent.assets.base import AssetCompilationRequest, AssetGenerationRequest


def _generator(tmp_path: Path) -> ArticraftAssetGenerator:
    repo = tmp_path / "Articraft"
    repo.mkdir()
    return ArticraftAssetGenerator(
        ArticraftConfig(repo=repo, python_executable=Path(sys.executable))
    )


def test_asset_request_rejects_empty_description_and_unknown_provider(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="description"):
        AssetGenerationRequest("", tmp_path)
    with pytest.raises(ValueError, match="provider"):
        AssetGenerationRequest("a cup", tmp_path, provider="unknown")


def test_preflight_rejects_unpinned_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _generator(tmp_path)
    monkeypatch.setattr("phiagent.assets.articraft._run_capture", lambda *args, **kwargs: "wrong")
    with pytest.raises(AssetGenerationError, match="expected pinned"):
        generator.preflight()


def test_runtime_path_does_not_resolve_virtualenv_symlink(tmp_path: Path) -> None:
    runtime = tmp_path / "venv" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.symlink_to(Path(sys.executable))
    assert _absolute_without_resolving(runtime) == runtime.absolute()


def test_generate_persists_validated_usdz_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _generator(tmp_path)
    monkeypatch.setattr(
        generator,
        "preflight",
        lambda: {
            "articraft_commit": ARTICRAFT_COMMIT,
            "mini_articraft_version": "0.test",
            "python_executable": sys.executable,
        },
    )
    monkeypatch.setattr(generator, "_project_provenance", lambda: {"test": True})

    def fake_execute(command: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> None:
        del command, env
        run_dir = cwd / "runs" / "run-1"
        run_dir.mkdir(parents=True)
        artifact = run_dir / "asset.usdz"
        artifact.write_bytes(b"usdz")
        (cwd / "articraft-result.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "run_dir": str(run_dir),
                    "artifact": str(artifact),
                }
            )
        )
        log_path.write_text("generated\n")

    monkeypatch.setattr(generator, "_execute", fake_execute)
    result = generator.generate(
        AssetGenerationRequest(
            description="a graspable bottle with a hinged cap",
            experiment_root=tmp_path / "outputs",
        )
    )

    assert result.artifact_format == "usdz"
    assert result.artifact.read_bytes() == b"usdz"
    metadata = json.loads(result.metadata.read_text())
    assert metadata["status"] == "succeeded"
    assert metadata["preflight"]["articraft_commit"] == ARTICRAFT_COMMIT
    assert "verified handover scene" in metadata["limitations"][0]


def test_generate_records_invalid_output_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _generator(tmp_path)
    monkeypatch.setattr(generator, "preflight", lambda: {"articraft_commit": ARTICRAFT_COMMIT})
    monkeypatch.setattr(generator, "_project_provenance", lambda: {})

    def fake_execute(command: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> None:
        del command, log_path, env
        run_dir = cwd / "runs" / "run-1"
        run_dir.mkdir(parents=True)
        artifact = run_dir / "asset.obj"
        artifact.write_text("not usdz")
        (cwd / "articraft-result.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "run_dir": str(run_dir),
                    "artifact": str(artifact),
                }
            )
        )

    monkeypatch.setattr(generator, "_execute", fake_execute)
    output_root = tmp_path / "outputs"
    with pytest.raises(AssetGenerationError, match="non-empty USDZ"):
        generator.generate(AssetGenerationRequest("a cup", output_root))

    experiments = list(output_root.iterdir())
    assert len(experiments) == 1
    metadata = json.loads((experiments[0] / "metadata.json").read_text())
    assert metadata["status"] == "failed"


def test_compile_model_uses_no_provider_and_records_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _generator(tmp_path)
    monkeypatch.setattr(generator, "preflight", lambda: {"articraft_commit": ARTICRAFT_COMMIT})
    monkeypatch.setattr(generator, "_project_provenance", lambda: {})
    model_file = tmp_path / "model.py"
    model_file.write_text("object_model = object()\n")

    def fake_execute(command: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> None:
        del env
        assert "provider" not in command
        run_dir = cwd / "runs" / "authored-model"
        run_dir.mkdir(parents=True)
        artifact = run_dir / "asset.usdz"
        artifact.write_bytes(b"usdz")
        (cwd / "articraft-result.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "run_dir": str(run_dir),
                    "artifact": str(artifact),
                    "compile_report": {"counts": {"failures": 0, "warnings": 0}},
                }
            )
        )
        log_path.write_text("compiled\n")

    monkeypatch.setattr(generator, "_execute", fake_execute)
    result = generator.compile_model(
        AssetCompilationRequest(model_file=model_file, experiment_root=tmp_path / "outputs")
    )

    metadata = json.loads(result.metadata.read_text())
    assert metadata["backend"] == "mini-articraft-sdk"
    assert metadata["compile_report"]["counts"]["failures"] == 0
