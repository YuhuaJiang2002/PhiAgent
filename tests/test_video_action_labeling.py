from __future__ import annotations

import pytest

from phiagent.labeling.video_action import (
    aggregate_video_action_groups,
    eef_state_deltas,
    integrate_eef_deltas,
    video_action_episode_metrics,
)


def test_eef_delta_round_trip_preserves_frame_explicit_states() -> None:
    states = [
        [0.0] * 14,
        [0.1, 0.2, 0.3, 0.0, 0.0, 3.13, 0.5] * 2,
        [0.2, 0.1, 0.4, 0.0, 0.0, -3.13, 0.25] * 2,
    ]

    deltas = eef_state_deltas(states)
    reconstructed = integrate_eef_deltas(states[0], deltas)

    assert reconstructed[1] == pytest.approx(states[1])
    assert reconstructed[2] == pytest.approx(states[2])


def test_video_action_metrics_are_zero_for_exact_prediction() -> None:
    states = [[0.0] * 14, [0.1] * 14, [0.2] * 14]
    deltas = eef_state_deltas(states)

    metrics = video_action_episode_metrics(
        deltas,
        deltas,
        states,
        states,
        channel_scale=[1.0] * 14,
    )

    assert all(value == pytest.approx(0.0, abs=1e-9) for value in metrics.values())


def test_video_action_aggregation_counts_physical_episodes_not_clips() -> None:
    records = [
        {
            "method": method,
            "independent_group_id": group,
            "metrics": {"error": value},
        }
        for method, base in (("candidate", 1.0), ("baseline", 2.0))
        for group, value in (
            ("episode-a", base),
            ("episode-a", base + 0.2),
            ("episode-b", base + 0.4),
        )
    ]

    result = aggregate_video_action_groups(records)

    assert result["candidate"]["raw_clips"] == 3
    assert result["candidate"]["independent_groups"] == 2
    assert result["candidate"]["per_group"]["episode-a"]["error"] == pytest.approx(1.1)
