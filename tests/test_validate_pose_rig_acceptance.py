from __future__ import annotations

from scripts.validate_pose_rig_acceptance import _transition_statistics


def test_transition_statistics_reports_zero_values_over_strict_threshold() -> None:
    import numpy as np

    result = _transition_statistics(np, [1.0, 1.2, 0.8, 3.9], 4.0)

    assert result["outlier_count"] == 0
    assert result["outlier_transition_to_frames"] == []


def test_transition_statistics_reports_destination_frame_indices() -> None:
    import numpy as np

    result = _transition_statistics(np, [1.0, 5.0, 1.0], 4.0)

    assert result["outlier_count"] == 1
    assert result["outlier_transition_to_frames"] == [2]
