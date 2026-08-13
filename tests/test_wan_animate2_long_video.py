from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_wan_animate2_long_video import (
    _parse_gpu_pairs,
    _parse_reused_windows,
    _covered_source_frames,
    _long_horizon_continuity,
    _recovered_persistent_generation_metrics,
    _throughput_metrics,
    partition_contiguous,
    plan_selected_windows,
    plan_windows,
)
from scripts.stitch_wan_animate2_long_video import (
    apply_color_offset,
    estimate_background_offset,
    merge_at_best_seam,
    merge_quality_anchor,
    select_seam,
)


def test_plan_windows_covers_660_frames_with_one_padded_tail_frame() -> None:
    windows = plan_windows(660, clip_len=81, overlap=16)

    assert [window.start_frame for window in windows] == [
        0,
        64,
        128,
        192,
        256,
        320,
        384,
        448,
        512,
        580,
    ]
    assert all(window.expected_output_frames == 80 for window in windows)
    assert windows[-1].source_frames == 80
    assert windows[-1].padded_frames == 1
    covered = {
        frame
        for window in windows
        for frame in range(
            window.start_frame,
            window.start_frame + window.expected_output_frames,
        )
    }
    assert covered == set(range(660))


def test_plan_windows_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        plan_windows(660, clip_len=81, overlap=80)


def test_selected_bridge_windows_are_sorted_and_validated() -> None:
    windows = plan_selected_windows(660, [224, 32, 96], clip_len=81)
    assert [window.start_frame for window in windows] == [32, 96, 224]
    assert all(window.expected_output_frames == 80 for window in windows)
    with pytest.raises(ValueError, match="unique"):
        plan_selected_windows(660, [32, 32], clip_len=81)
    with pytest.raises(ValueError, match="outside"):
        plan_selected_windows(660, [581], clip_len=81)


def test_runner_records_fresh_gpu_state_before_each_window() -> None:
    source = Path("scripts/run_wan_animate2_long_video.py").read_text()
    assert "current_gpus, current_inventory, current_processes = query_gpus()" in source
    assert 'window_record["gpu_pre_window"]' in source


def test_persistent_scheduler_balances_contiguous_temporal_chains() -> None:
    items = [{"index": index} for index in range(10)]

    groups = partition_contiguous(items, 3)

    assert [[item["index"] for item in group] for group in groups] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]


def test_gpu_pairs_must_be_disjoint_physical_devices() -> None:
    assert _parse_gpu_pairs(["1,4", "5,6"]) == ((1, 4), (5, 6))
    with pytest.raises(ValueError, match="cannot be shared"):
        _parse_gpu_pairs(["1,4", "4,6"])
    with pytest.raises(ValueError, match="must be PHYSICAL"):
        _parse_gpu_pairs(["1"])


def test_reused_window_results_are_keyed_by_unique_source_frame(tmp_path: Path) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"

    assert _parse_reused_windows([f"0={first}", f"64={second}"]) == {
        0: first.resolve(),
        64: second.resolve(),
    }
    with pytest.raises(ValueError, match="duplicate"):
        _parse_reused_windows([f"0={first}", f"0={second}"])
    with pytest.raises(ValueError, match="START_FRAME"):
        _parse_reused_windows([str(first)])


def test_throughput_reports_useful_speed_and_gpu_cost() -> None:
    metrics = _throughput_metrics(
        source_frames=660,
        fps=24.0,
        generated_frames=800,
        generation_wall_seconds=600.0,
        end_to_end_wall_seconds=660.0,
        batch_gpu_seconds=2_400.0,
    )

    assert metrics["useful_video_seconds"] == pytest.approx(27.5)
    assert metrics["effective_generation_fps"] == pytest.approx(1.1)
    assert metrics["effective_end_to_end_fps"] == pytest.approx(1.0)
    assert metrics["a800_gpu_hours"] == pytest.approx(2 / 3)


def test_recovered_generation_timing_sums_each_completed_gpu_pair() -> None:
    metrics = _recovered_persistent_generation_metrics(
        {
            "batches": [
                {
                    "status": "completed",
                    "returncode": 0,
                    "started_at": "2026-08-12T00:00:00+00:00",
                    "completed_at": "2026-08-12T00:10:00+00:00",
                    "physical_gpu_pair": [{"physical_index": 0}, {"physical_index": 1}],
                },
                {
                    "status": "completed",
                    "returncode": 0,
                    "started_at": "2026-08-12T00:01:00+00:00",
                    "completed_at": "2026-08-12T00:09:00+00:00",
                    "physical_gpu_pair": [{"physical_index": 2}, {"physical_index": 3}],
                },
            ]
        },
        source_frames=660,
        fps=24.0,
        generated_frames=800,
    )

    assert metrics["generation_wall_seconds"] == pytest.approx(600.0)
    assert metrics["effective_generation_fps"] == pytest.approx(1.1)
    assert metrics["a800_gpu_hours"] == pytest.approx(0.6)


def test_selected_window_throughput_counts_only_unique_timeline_coverage() -> None:
    windows = [
        {"start_frame": 0, "expected_output_frames": 80},
        {"start_frame": 64, "expected_output_frames": 80},
        {"start_frame": 224, "expected_output_frames": 80},
    ]

    assert _covered_source_frames(windows) == 224


def test_single_window_continuity_report_has_explicit_empty_overlap_metrics() -> None:
    pytest.importorskip("numpy")
    report = _long_horizon_continuity(
        [
            {
                "index": 0,
                "batch_index": -1,
                "start_frame": 288,
                "expected_output_frames": 80,
                "reference": {"kind": "source_camera_frame"},
            }
        ],
        fps=24.0,
    )

    assert report["covered_source_frames"] == 80
    assert report["covered_video_seconds"] == pytest.approx(80 / 24)
    assert report["is_20s_or_longer"] is False
    assert report["overlaps"] == []
    assert report["mean_best_seam_mad"] is None


def test_persistent_batch_validates_cuda_mapping_and_supports_rolling_anchor() -> None:
    source = Path("scripts/run_wan_animate2_persistent_batch.py").read_text()
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES")' in source
    assert 'temporal_anchor_mode == "rolling"' in source
    assert "continuation-reference.png" in source


def test_background_offset_aligns_stable_region() -> None:
    np = pytest.importorskip("numpy")
    reference = [np.full((8, 12, 3), 100, dtype=np.uint8) for _ in range(6)]
    candidate = [np.full((8, 12, 3), 90, dtype=np.uint8) for _ in range(6)]

    offset = estimate_background_offset(
        np,
        reference=reference,
        reference_start=4,
        candidate=candidate,
        candidate_start=6,
    )
    aligned = apply_color_offset(np, candidate, offset)

    assert offset == pytest.approx((10.0, 10.0, 10.0))
    assert int(aligned[0].min()) == 100
    assert int(aligned[0].max()) == 100


def test_seam_search_selects_lowest_transition() -> None:
    np = pytest.importorskip("numpy")
    current = [np.full((6, 8, 3), value, dtype=np.uint8) for value in range(10)]
    following = [
        np.full((6, 8, 3), value, dtype=np.uint8)
        for value in (6, 200, 7, 200, 10, 11, 12, 13, 14, 15)
    ]

    seam, cost = select_seam(
        np,
        current=current,
        current_start=0,
        following=following,
        following_start=6,
    )
    merged, record = merge_at_best_seam(
        np,
        current=current,
        current_start=0,
        following=following,
        following_start=6,
    )

    assert seam == 8
    assert cost == pytest.approx(0.0)
    assert record["seam_frame"] == 8
    assert len(merged) == 16
    assert int(merged[7][0, 0, 0]) == 7
    assert int(merged[8][0, 0, 0]) == 7


def test_quality_anchor_keeps_a_constrained_exact_core() -> None:
    np = pytest.importorskip("numpy")
    left = [np.full((4, 6, 3), value, dtype=np.uint8) for value in range(12)]
    anchor = [np.full((4, 6, 3), value, dtype=np.uint8) for value in range(6, 16)]
    right = [np.full((4, 6, 3), value, dtype=np.uint8) for value in range(10, 21)]

    merged, record = merge_quality_anchor(
        np,
        left=left,
        left_start=0,
        anchor=anchor,
        anchor_start=6,
        right=right,
        right_start=10,
        minimum_anchor_frames=3,
        blend_radius=1,
    )

    retained_start = record["retained_start_frame"]
    retained_end = record["retained_end_frame_exclusive"]
    assert record["retained_frames"] >= 3
    assert len(merged) == 21
    for index in range(retained_start, retained_end):
        assert np.array_equal(merged[index], anchor[index - 6])


def test_overlap_blend_preserves_length_and_softens_the_hard_step() -> None:
    np = pytest.importorskip("numpy")
    current = [np.full((4, 6, 3), 0, dtype=np.uint8) for _ in range(8)]
    following = [np.full((4, 6, 3), 100, dtype=np.uint8) for _ in range(8)]

    hard, _ = merge_at_best_seam(
        np,
        current=current,
        current_start=0,
        following=following,
        following_start=4,
    )
    blended, record = merge_at_best_seam(
        np,
        current=current,
        current_start=0,
        following=following,
        following_start=4,
        blend_radius=2,
    )

    hard_steps = [abs(int(hard[i][0, 0, 0]) - int(hard[i - 1][0, 0, 0])) for i in range(1, 12)]
    blended_steps = [
        abs(int(blended[i][0, 0, 0]) - int(blended[i - 1][0, 0, 0]))
        for i in range(1, 12)
    ]
    assert len(blended) == len(hard) == 12
    assert max(blended_steps) < max(hard_steps)
    assert record["blend_radius"] == 2


def test_seam_search_respects_stable_window_bounds() -> None:
    np = pytest.importorskip("numpy")
    current = [np.full((4, 6, 3), value, dtype=np.uint8) for value in range(20)]
    following = [
        np.full((4, 6, 3), value, dtype=np.uint8)
        for value in range(10, 30)
    ]

    seam, _ = select_seam(
        np,
        current=current,
        current_start=0,
        following=following,
        following_start=10,
        minimum_seam=14,
        maximum_seam=16,
    )

    assert 14 <= seam <= 16
    with pytest.raises(ValueError, match="constraints"):
        select_seam(
            np,
            current=current,
            current_start=0,
            following=following,
            following_start=10,
            minimum_seam=30,
        )
