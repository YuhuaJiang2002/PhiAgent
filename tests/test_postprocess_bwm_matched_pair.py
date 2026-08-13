from __future__ import annotations

from pathlib import Path

import pytest

from scripts.postprocess_bwm_matched_pair import find_campaign


def test_find_campaign_requires_exactly_one_complete_run(tmp_path: Path) -> None:
    complete = tmp_path / "complete"
    (complete / "videos").mkdir(parents=True)
    (complete / "manifest.json").write_text("{}")

    assert find_campaign(tmp_path) == complete

    second = tmp_path / "second"
    (second / "videos").mkdir(parents=True)
    (second / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="found 2"):
        find_campaign(tmp_path)
