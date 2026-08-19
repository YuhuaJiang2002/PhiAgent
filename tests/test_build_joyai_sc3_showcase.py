from __future__ import annotations

import pytest

from scripts.build_joyai_sc3_showcase import _score_for_selected_seed


def _manifest(*, automatic: bool = True, human: bool | None = None) -> dict:
    return {
        "selection": {"selected_seed": 181},
        "candidates": [
            {
                "seed": 181,
                "score": {
                    "automatic_pass": automatic,
                    "human_review_passed": human,
                },
            }
        ],
    }


def test_partial_automatic_winner_can_be_packaged_for_review() -> None:
    seed, score = _score_for_selected_seed(_manifest())

    assert seed == 181
    assert score["human_review_passed"] is None


def test_failed_or_human_rejected_candidate_cannot_be_packaged() -> None:
    with pytest.raises(ValueError, match="failed automatic"):
        _score_for_selected_seed(_manifest(automatic=False))
    with pytest.raises(ValueError, match="human-rejected"):
        _score_for_selected_seed(_manifest(human=False))
