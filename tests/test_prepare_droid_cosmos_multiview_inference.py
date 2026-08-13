from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prepare_droid_cosmos_multiview_inference import (
    repeat_first_calibration_row,
    select_record,
)


def test_select_record_requires_non_training_split() -> None:
    contract = {
        "records": [
            {"episode_index": 54, "split": "final_holdout", "training_use": False},
            {"episode_index": 3, "split": "train", "training_use": True},
        ]
    }
    assert select_record(contract, 54, "final_holdout")["episode_index"] == 54
    with pytest.raises(ValueError, match="must not be a training"):
        select_record(contract, 3, "train")


def test_repeat_first_calibration_row_freezes_proxy(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("1 2 3 4\n5 6 7 8\n")
    repeat_first_calibration_row(source, destination, 3)
    assert destination.read_text() == "1 2 3 4\n1 2 3 4\n1 2 3 4\n"


def test_repeat_first_calibration_row_rejects_empty_source(tmp_path: Path) -> None:
    source = tmp_path / "empty.txt"
    source.write_text("")
    with pytest.raises(ValueError, match="empty"):
        repeat_first_calibration_row(source, tmp_path / "out.txt", 2)
