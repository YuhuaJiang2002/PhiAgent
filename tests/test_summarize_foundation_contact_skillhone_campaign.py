from __future__ import annotations

import pytest

from scripts.summarize_foundation_contact_skillhone_campaign import summarize_scores


def _row(split: str, passed: int, total: int) -> dict[str, object]:
    return {
        "split": split,
        "n_passed": passed,
        "n_total": total,
        "n_strict": passed,
        "n_errors": 0,
        "n_no_answer": total - passed,
        "score": passed / total,
        "traces": [{"duration_s": float(total)}],
    }


def test_skillhone_campaign_requires_every_strict_item() -> None:
    result = summarize_scores(
        [_row("probe", 4, 4), _row("test", 3, 3), _row("adversarial", 7, 7)]
    )

    assert result["passed"] == 14
    assert result["total"] == 14
    assert result["strict"] == 14
    assert result["all_passed"] is True


def test_skillhone_campaign_rejects_missing_answer() -> None:
    result = summarize_scores([_row("adversarial", 6, 7)])

    assert result["all_passed"] is False
    assert result["missing_answers"] == 1


def test_skillhone_campaign_rejects_duplicate_splits() -> None:
    with pytest.raises(ValueError, match="unique"):
        summarize_scores([_row("probe", 1, 1), _row("probe", 1, 1)])
