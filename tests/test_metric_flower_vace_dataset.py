from __future__ import annotations

import pytest

from scripts.export_metric_flower_vace_dataset import selected_clip_starts


def test_metric_flower_vace_split_is_frame_disjoint_and_full_span() -> None:
    split = selected_clip_starts(
        source_frames=660,
        clip_frames=17,
        source_frame_step=3,
    )
    span = 48
    train = {
        frame
        for start in split["train"]
        for frame in range(start, start + span + 1)
    }
    validation = {
        frame
        for start in split["validation"]
        for frame in range(start, start + span + 1)
    }

    assert len(split["train"]) == 12
    assert len(split["validation"]) == 4
    assert not train.intersection(validation)
    assert min(train | validation) == 0
    assert max(train | validation) == 648


def test_metric_flower_vace_split_rejects_changed_contract() -> None:
    with pytest.raises(ValueError, match="frozen v1 split"):
        selected_clip_starts(
            source_frames=659,
            clip_frames=17,
            source_frame_step=3,
        )
