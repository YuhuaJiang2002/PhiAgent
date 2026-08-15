from __future__ import annotations

import pytest

from phiagent.rendering.replacement_capacity import (
    JoyAIThroughputBenchmark,
    estimate_joyai_replacement_capacity,
)


def _estimate(**overrides: object) -> dict[str, object]:
    arguments = {
        "video_hours": 100.0,
        "fps": 24,
        "average_clip_frames": 660,
        "gpu_count": 8,
        "gpu_utilization": 0.85,
        "postprocess_workers": 1,
        "postprocess_utilization": 0.85,
    }
    arguments.update(overrides)
    return estimate_joyai_replacement_capacity(**arguments)


def test_100h_estimate_uses_measured_padding_and_throughput() -> None:
    result = _estimate()

    assert result["workload"] == {
        "source_frames": 8_640_000,
        "clip_count": 13_091,
        "full_clips": 13_090,
        "remainder_frames": 600,
        "generated_frames_with_padding": 8_705_451,
        "tail_padding_frames": 65_451,
        "tail_padding_percent": pytest.approx(0.7575347222),
        "legacy_protocol_small_files": 8_705_451,
    }
    assert result["compute"]["generation_gpu_hours"] == pytest.approx(384.3343889)
    assert result["compute"]["postprocess_worker_hours"] == pytest.approx(46.4036364)
    assert result["calendar"]["generation_hours"] == pytest.approx(56.5197631)
    assert result["calendar"]["postprocess_hours"] == pytest.approx(54.5925134)
    assert result["calendar"]["pipelined_hours"] == pytest.approx(56.5197631)
    assert result["recommendation"]["balanced_postprocess_workers"] == 1


def test_postprocessing_scales_to_balance_larger_gpu_pool() -> None:
    result = _estimate(gpu_count=32, postprocess_workers=4)

    assert result["recommendation"]["balanced_postprocess_workers"] == 4
    assert result["calendar"]["generation_hours"] == pytest.approx(14.1299408)
    assert result["calendar"]["postprocess_hours"] == pytest.approx(13.6481283)


def test_session_overhead_is_charged_per_clip() -> None:
    baseline = _estimate()
    with_overhead = _estimate(session_overhead_seconds=5.0)

    expected_extra_gpu_hours = 13_091 * 5 / 3600
    assert (
        with_overhead["compute"]["generation_gpu_hours"]
        - baseline["compute"]["generation_gpu_hours"]
    ) == pytest.approx(expected_extra_gpu_hours)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("video_hours", 0),
        ("gpu_count", 0),
        ("gpu_utilization", 1.1),
        ("postprocess_utilization", 0),
        ("session_overhead_seconds", -1),
    ],
)
def test_invalid_capacity_assumptions_fail(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _estimate(**{field: value})


def test_benchmark_rates_match_recorded_a800_run() -> None:
    benchmark = JoyAIThroughputBenchmark()

    assert benchmark.generation_fps == pytest.approx(6.2918669341)
    assert benchmark.postprocess_fps == pytest.approx(51.7200846329)
