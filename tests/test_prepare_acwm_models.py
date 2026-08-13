from __future__ import annotations

from pathlib import Path

import pytest

from scripts import prepare_acwm_models


def test_prepare_repository_accepts_exact_revision_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = "bwm"
    _, directory, revision = prepare_acwm_models.REPOSITORIES[backend]
    repository = tmp_path / directory
    repository.mkdir()
    (repository / ".phiagent-source-revision").write_text(revision + "\n")
    monkeypatch.setattr(
        prepare_acwm_models,
        "_run",
        lambda *args, **kwargs: pytest.fail("an exact archive must not run Git"),
    )

    assert prepare_acwm_models._prepare_repository(backend, tmp_path) == repository


def test_prepare_repository_rejects_wrong_revision_archive(tmp_path: Path) -> None:
    backend = "bwm"
    _, directory, _ = prepare_acwm_models.REPOSITORIES[backend]
    repository = tmp_path / directory
    repository.mkdir()
    (repository / ".phiagent-source-revision").write_text("wrong\n")

    with pytest.raises(RuntimeError, match="expected revision marker"):
        prepare_acwm_models._prepare_repository(backend, tmp_path)


def test_flowwam_repository_is_pinned() -> None:
    url, directory, revision = prepare_acwm_models.REPOSITORIES["flowwam"]

    assert url.endswith("YixiangChen515/FlowWAM_WorldArena.git")
    assert directory == "FlowWAM_WorldArena"
    assert revision == "f06fa46042e97738c6619c868f1097be6749d48d"
