from __future__ import annotations

import copy

import pytest

from phiagent.acwm.counterfactual import (
    aggregate_counterfactual_records,
    build_action_swap_suite,
    compare_counterfactual_models,
    rebase_absolute_eef_future,
    validate_counterfactual_sources,
)
from phiagent.acwm.worldarena import attach_worldarena_lineage


def _row(index: int) -> dict:
    return {
        "action": {
            "data": f"actions/task/episode_{index}/end-pose.parquet",
            "start_frame": 0,
            "end_frame": 56,
        },
        "coordinate_frame": "robot_base:calibrated",
        "episode_index": index,
        "group_id": f"task-episode_{index}",
        "history_frames": 9,
        "independent_group_id": f"physical-episode-{index}",
        "length": 57,
        "prompt": "wipe the table",
        "source_episode": f"task/episode_{index}",
        "split": "test",
        "video": {
            "data": f"assets/task/episode_{index}/cam_high.mp4",
            "start_frame": 0,
            "end_frame": 56,
        },
    }


def test_action_swap_suite_changes_only_action_and_audit_fields() -> None:
    sources = [_row(2), _row(0), _row(1)]
    original = copy.deepcopy(sources)

    rows, pairs = build_action_swap_suite(sources, episode_index_start=400)

    assert sources == original
    assert len(rows) == 6
    assert len(pairs) == 3
    for factual, swapped in zip(rows[::2], rows[1::2]):
        assert factual["video"] == swapped["video"]
        assert factual["prompt"] == swapped["prompt"]
        assert factual["action"] != swapped["action"]
        assert factual["counterfactual"]["variant"] == "factual"
        assert swapped["counterfactual"]["variant"] == "swapped"
        assert factual["counterfactual"]["paired_episode_index"] == swapped["episode_index"]
        assert swapped["counterfactual"]["paired_episode_index"] == factual["episode_index"]
        assert (
            factual["counterfactual"]["action_source_episode"]
            != swapped["counterfactual"]["action_source_episode"]
        )
    assert [row["episode_index"] for row in rows] == [400, 401, 402, 403, 404, 405]


def test_counterfactual_source_validation_rejects_frame_and_split_leakage() -> None:
    invalid_frame = _row(0)
    invalid_frame["coordinate_frame"] = "camera:pixels"
    with pytest.raises(ValueError, match="robot_base"):
        validate_counterfactual_sources([invalid_frame, _row(1)])

    invalid_split = _row(0)
    invalid_split["split"] = "validation"
    with pytest.raises(ValueError, match="frozen test"):
        validate_counterfactual_sources([invalid_split, _row(1)])


def test_counterfactual_source_validation_rejects_mismatched_contracts() -> None:
    mismatched = _row(1)
    mismatched["history_frames"] = 5
    with pytest.raises(ValueError, match="share coordinate frame"):
        validate_counterfactual_sources([_row(0), mismatched])


def test_action_donor_must_come_from_an_independent_physical_episode() -> None:
    first = _row(0)
    same_group = _row(1)
    different_group = _row(2)
    same_group["independent_group_id"] = first["independent_group_id"]

    rows, pairs = build_action_swap_suite([first, same_group, different_group])

    assert rows[1]["counterfactual"]["action_source_episode"] == "task/episode_2"
    assert pairs[0]["swapped_action_independent_group_id"] == "physical-episode-2"


def test_counterfactual_construction_requires_two_episodes() -> None:
    with pytest.raises(ValueError, match="at least two"):
        build_action_swap_suite([_row(0)])


def test_rebase_absolute_eef_future_preserves_history_and_rebases_motion() -> None:
    source = [[0.0] * 14 for _ in range(4)]
    donor = [[10.0] * 14 for _ in range(4)]
    source[1] = [1.0] * 14
    donor[1] = [10.0] * 14
    donor[2] = [11.0] * 14
    donor[3] = [12.0] * 14
    donor[2][6] = donor[2][13] = 0.25
    donor[3][6] = donor[3][13] = 0.75

    rebased = rebase_absolute_eef_future(source, donor, history_frames=2)

    assert rebased[:2] == tuple(tuple(row) for row in source[:2])
    assert rebased[2][0:3] == (2.0, 2.0, 2.0)
    assert rebased[3][7:10] == (3.0, 3.0, 3.0)
    assert rebased[2][6] == rebased[2][13] == 0.25
    assert rebased[3][6] == rebased[3][13] == 0.75
    assert all(-3.141593 <= rebased[3][channel] <= 3.141593 for channel in (3, 4, 5))


def test_rebase_quaternion_future_preserves_unit_rotations() -> None:
    source = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 2
        for _ in range(4)
    ]
    donor = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 2
        for _ in range(4)
    ]
    source[1][3:7] = [0.0, 0.0, 2**-0.5, 2**-0.5]
    donor[2][3:7] = [2**-0.5, 0.0, 0.0, 2**-0.5]
    donor[3][3:7] = [1.0, 0.0, 0.0, 0.0]
    donor[2][7:10] = [1.0, 2.0, 3.0]

    rebased = rebase_absolute_eef_future(
        source,
        donor,
        history_frames=2,
        rotation_representation="quaternion",
    )

    assert rebased[:2] == tuple(tuple(row) for row in source[:2])
    assert rebased[2][7:10] == (1.0, 2.0, 3.0)
    assert sum(value * value for value in rebased[2][3:7]) == pytest.approx(1.0)
    assert sum(value * value for value in rebased[3][3:7]) == pytest.approx(1.0)
    assert rebased[2][6] != donor[2][6]


def test_action_swap_suite_can_reference_derived_actions() -> None:
    rows, pairs = build_action_swap_suite(
        [_row(0), _row(1)],
        swapped_action_by_source_episode={
            "task/episode_0": "derived/source-0.parquet",
            "task/episode_1": "derived/source-1.parquet",
        },
    )

    assert rows[1]["action"]["data"] == "derived/source-0.parquet"
    assert rows[3]["action"]["data"] == "derived/source-1.parquet"
    assert rows[1]["action"]["start_frame"] == 0
    assert rows[1]["action"]["end_frame"] == 56
    assert pairs[0]["donor_action"].endswith("episode_1/end-pose.parquet")


def _audit_record(
    model: str,
    seed: int,
    trial: str,
    value: float,
    independent_unit: str | None = None,
) -> dict:
    return {
        "model_id": model,
        "seed": seed,
        "trial_id": trial,
        "independent_unit_id": independent_unit or trial,
        "metrics": {
            "factual_future_ssim": value,
            "wrong_action_ssim_margin": value / 10.0,
        },
    }


def test_counterfactual_aggregation_averages_seeds_before_trials() -> None:
    records = [
        _audit_record(model, seed, trial, base + seed / 100.0)
        for model, base in (("base", 0.5), ("candidate", 0.7))
        for trial in ("episode-a", "episode-b")
        for seed in (1, 2)
    ]

    models = aggregate_counterfactual_records(records)

    assert models["candidate"]["independent_trials"] == 2
    assert models["candidate"]["seeds"] == [1, 2]
    assert models["candidate"]["per_trial"]["episode-a"]["factual_future_ssim"] == pytest.approx(
        0.715
    )


def test_counterfactual_comparison_fails_closed_below_episode_minimum() -> None:
    records = [
        _audit_record(model, seed, trial, value)
        for model, value in (("base", 0.5), ("candidate", 0.7))
        for trial in ("episode-a", "episode-b")
        for seed in (1, 2)
    ]

    result = compare_counterfactual_models(
        records,
        candidate_model="candidate",
        baseline_model="base",
        primary_metrics={
            "factual_future_ssim": "higher",
            "wrong_action_ssim_margin": "higher",
        },
        minimum_independent_trials=20,
        bootstrap_iterations=100,
    )

    assert result["all_primary_metric_lower_bounds_positive"] is True
    assert result["decision_eligible"] is False
    assert result["audit_passed"] is False


def test_counterfactual_aggregation_does_not_count_clips_as_independent() -> None:
    records = [
        _audit_record(model, 1, trial, value, independent_unit="physical-episode")
        for model, value in (("base", 0.5), ("candidate", 0.7))
        for trial in ("clip-a", "clip-b")
    ]

    models = aggregate_counterfactual_records(records)

    assert models["candidate"]["raw_trials"] == 2
    assert models["candidate"]["independent_trials"] == 1


def test_worldarena_lineage_groups_clips_from_one_physical_episode() -> None:
    rows = [_row(0), _row(1)]
    del rows[0]["independent_group_id"]
    del rows[1]["independent_group_id"]
    rows[0]["source_episode"] = "task/episode_0"
    rows[1]["source_episode"] = "task/episode_1"
    manifest = {
        "episodes": [
            {
                "task": "task",
                "episode": "episode_0",
                "meta": {
                    "source_episode": "task/raw_episode_7",
                    "source_clip_index": 0,
                },
            },
            {
                "task": "task",
                "episode": "episode_1",
                "meta": {
                    "source_episode": "task/raw_episode_7",
                    "source_clip_index": 1,
                },
            },
        ]
    }

    enriched = attach_worldarena_lineage(rows, manifest)

    assert enriched[0]["independent_group_id"] == "task/raw_episode_7"
    assert enriched[1]["independent_group_id"] == "task/raw_episode_7"
