from __future__ import annotations

import json

import pytest

from phiagent.evaluation.video_proxy import (
    DecodedFrames,
    evaluate_decoded_core_proxy,
    evaluate_decoded_proxy,
    resolve_ffmpeg,
    write_evaluation_evidence,
)
from phiagent.evaluation.object_instance import ObjectInstanceMetrics


def _sequence(frames: tuple[bytes, ...]) -> DecodedFrames:
    return DecodedFrames(frames=frames, width=8, height=8)


def _moving_frames() -> tuple[bytes, ...]:
    return tuple(
        bytes((x * 20 + y * 10 + time * (x + 1)) % 256 for y in range(8) for x in range(8))
        for time in range(5)
    )


def _object_metrics(score: float = 1.0) -> ObjectInstanceMetrics:
    return ObjectInstanceMetrics(
        contour_similarity=score,
        color_similarity=score,
        temporal_deformation=score,
        tracking_coverage=score,
        trajectory_similarity=score,
        lift_recall=score,
        object_consistency=score,
    )


def test_identical_motion_and_reference_receive_perfect_proxy_scores() -> None:
    frames = _moving_frames()
    decoded = _sequence(frames)

    metrics = evaluate_decoded_proxy(
        decoded,
        decoded,
        _sequence((frames[0],)),
        decoded,
        _object_metrics(),
    )

    assert metrics.motion_preservation == pytest.approx(1.0)
    assert metrics.target_identity == pytest.approx(1.0)
    assert metrics.object_consistency == pytest.approx(1.0)
    assert metrics.temporal_consistency == pytest.approx(1.0)


def test_static_candidate_fails_motion_preservation() -> None:
    frames = _moving_frames()
    static = _sequence((frames[0],) * len(frames))

    metrics = evaluate_decoded_proxy(
        _sequence(frames),
        _sequence(frames),
        _sequence((frames[0],)),
        static,
        _object_metrics(),
    )

    assert metrics.motion_preservation == 0.0


def test_instance_failure_reduces_object_consistency() -> None:
    frames = _moving_frames()
    uniform = tuple(bytes([frame_index * 5] * 64) for frame_index in range(5))

    metrics = evaluate_decoded_proxy(
        _sequence(frames),
        _sequence(frames),
        _sequence((uniform[0],)),
        _sequence(uniform),
        _object_metrics(0.2),
    )

    assert metrics.object_consistency < 0.8


def test_flicker_reduces_temporal_consistency() -> None:
    frames = _moving_frames()
    flicker = tuple(bytes([255 if index % 2 else 0] * 64) for index in range(5))

    metrics = evaluate_decoded_proxy(
        _sequence(frames),
        _sequence(frames),
        _sequence((flicker[0],)),
        _sequence(flicker),
        _object_metrics(),
    )

    assert metrics.temporal_consistency < 0.1
    assert metrics.candidate_temporal_jerk > metrics.reference_temporal_jerk


def test_late_regional_flicker_is_not_hidden_by_full_frame_average() -> None:
    frames = _moving_frames() + _moving_frames()
    candidate = list(frames)
    for frame_index in range(7, 10):
        frame = bytearray(candidate[frame_index])
        value = 255 if frame_index % 2 else 0
        for y in range(4):
            for x in range(4):
                frame[y * 8 + x] = value
        candidate[frame_index] = bytes(frame)

    metrics = evaluate_decoded_proxy(
        _sequence(frames),
        _sequence(frames),
        _sequence((frames[0],)),
        _sequence(tuple(candidate)),
        _object_metrics(),
    )

    assert metrics.late_temporal_consistency < metrics.global_temporal_consistency
    assert metrics.regional_temporal_consistency < 0.5
    assert metrics.temporal_consistency < 0.5


def test_core_proxy_keeps_object_gate_out_of_metric_result() -> None:
    frames = _moving_frames()
    decoded = _sequence(frames)

    result = evaluate_decoded_core_proxy(
        decoded,
        decoded,
        _sequence((frames[0],)),
        decoded,
    )

    assert result["motion_preservation"] == pytest.approx(1.0)
    assert result["temporal_consistency"] == pytest.approx(1.0)
    assert "object_consistency" not in result


def test_evidence_records_hashes_metrics_and_limitations(tmp_path) -> None:
    paths = {}
    for name in ("source", "reference", "target", "candidate", "metadata"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        paths[name] = path
    frames = _moving_frames()
    metrics = evaluate_decoded_proxy(
        _sequence(frames),
        _sequence(frames),
        _sequence((frames[0],)),
        _sequence(frames),
        _object_metrics(),
    )
    evidence = tmp_path / "evidence.json"

    write_evaluation_evidence(
        evidence,
        source=paths["source"],
        reference=paths["reference"],
        target_image=paths["target"],
        candidate=paths["candidate"],
        backend_metadata=paths["metadata"],
        ffmpeg=tmp_path / "ffmpeg",
        metrics=metrics,
        width=8,
        height=8,
        sample_fps=8.0,
        maximum_seconds=4.0,
    )

    payload = json.loads(evidence.read_text())
    assert payload["evaluator"] == "phiagent-local-video-evaluator-v4-object-trajectory"
    assert payload["metrics"]["motion_preservation"] == pytest.approx(1.0)
    assert "tracked-instance contour" in payload["limitations"][1]
    assert "sensitivity 32.0" in payload["limitations"][3]
    assert "global, late-window, and regional" in payload["limitations"][4]
    with pytest.raises(FileExistsError, match="already exists"):
        write_evaluation_evidence(
            evidence,
            source=paths["source"],
            reference=paths["reference"],
            target_image=paths["target"],
            candidate=paths["candidate"],
            backend_metadata=paths["metadata"],
            ffmpeg=tmp_path / "ffmpeg",
            metrics=metrics,
            width=8,
            height=8,
            sample_fps=8.0,
            maximum_seconds=4.0,
        )


def test_ffmpeg_preflight_has_actionable_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        resolve_ffmpeg(tmp_path / "missing-ffmpeg")
