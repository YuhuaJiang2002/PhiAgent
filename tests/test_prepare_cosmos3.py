from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import prepare_cosmos3


def test_has_commit_uses_exact_commit_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(prepare_cosmos3.subprocess, "run", run)
    assert prepare_cosmos3._has_commit(tmp_path, "abc123")
    assert calls == [["git", "cat-file", "-e", "abc123^{commit}"]]


def test_uv_executable_accepts_supported_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare_cosmos3.shutil, "which", lambda name: "/tools/uv")
    monkeypatch.setattr(
        prepare_cosmos3.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="uv 0.11.3\n"),
    )
    assert prepare_cosmos3._uv_executable() == "/tools/uv"


def test_uv_executable_rejects_old_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare_cosmos3.shutil, "which", lambda name: "/tools/uv")
    monkeypatch.setattr(
        prepare_cosmos3.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="uv 0.8.22\n"),
    )
    with pytest.raises(RuntimeError, match=r"uv>=0\.11\.3"):
        prepare_cosmos3._uv_executable()
