import pytest

from phiagent.rendering.deghost import (
    MaskedDeghostConfig,
    ObjectGhostRepairConfig,
    build_deghost_filter,
    interval_weight,
)


def test_deghost_filter_targets_character_and_object_masks() -> None:
    expression = build_deghost_filter(7)
    assert "hqdn3d=7" in expression
    assert "[character_mask]maskedmerge" in expression
    assert "[object_mask]maskedmerge[out]" in expression


def test_deghost_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        MaskedDeghostConfig(strength=0)
    with pytest.raises(ValueError, match="CRF"):
        MaskedDeghostConfig(crf=52)


def test_interval_weight_has_soft_boundaries() -> None:
    intervals = ((1.0, 2.0),)
    assert interval_weight(29, 30.0, intervals, 3) == 0
    assert interval_weight(30, 30.0, intervals, 3) == pytest.approx(1 / 3)
    assert interval_weight(32, 30.0, intervals, 3) == 1
    assert interval_weight(60, 30.0, intervals, 3) == pytest.approx(1 / 3)


def test_object_repair_config_rejects_invalid_interval() -> None:
    with pytest.raises(ValueError, match="ranges"):
        ObjectGhostRepairConfig(intervals_s=((2.0, 1.0),))
