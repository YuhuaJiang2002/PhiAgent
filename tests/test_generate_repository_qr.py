from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_repository_qr.py"


def _load_module():
    spec = spec_from_file_location("generate_repository_qr", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_github_url_normalizes_trailing_slash():
    module = _load_module()
    assert module.validate_github_url("https://github.com/example/project/") == (
        "https://github.com/example/project"
    )


@pytest.mark.parametrize(
    "value",
    [
        "http://github.com/example/project",
        "https://example.com/example/project",
        "https://github.com/example/project/issues",
    ],
)
def test_validate_github_url_rejects_unsafe_or_non_repository_targets(value):
    module = _load_module()
    with pytest.raises(ValueError):
        module.validate_github_url(value)
