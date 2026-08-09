from pathlib import Path

import pytest

from phiagent.evaluation.object_instance import NormalizedROI
from phiagent.rendering.hand_style import (
    GraphiteHandConfig,
    apply_graphite_hand_style,
)


def test_graphite_hand_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        GraphiteHandConfig(opacity=1.1)
    with pytest.raises(ValueError, match="positive"):
        GraphiteHandConfig(opacity=0)


def test_hand_style_rejects_missing_input_before_heavy_import(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="candidate is missing"):
        apply_graphite_hand_style(
            candidate=tmp_path / "missing.mp4",
            hand_mask=tmp_path / "mask.mp4",
            output=tmp_path / "output.mp4",
            object_roi=NormalizedROI(0.1, 0.1, 0.2, 0.2),
            config=GraphiteHandConfig(),
        )
