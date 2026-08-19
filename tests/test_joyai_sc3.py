from __future__ import annotations

import json
from pathlib import Path

import pytest

from phiagent.acwm.schema import ACWMActionCondition, ActionRepresentation
from phiagent.rendering.joyai_video_edit import (
    JOYAI_MODEL_REVISION,
    JOYAI_REPOSITORY_REVISION,
    JOYAI_TEXT_ENCODER_REVISION,
)
from phiagent.world_model.joyai_sc3 import (
    CandidateScore,
    CarrierContract,
    ConsistencyThresholds,
    FitBlurPadTransform,
    JoyAISC3Config,
    JoyAISC3PreflightError,
    JoyAISC3Runner,
    VideoStream,
    compile_action_preserving_prompt,
    resampled_frame_count,
    select_consistent_candidate,
    validate_server_manifest,
    with_runtime_overrides,
)


def _action(*, frame: str = "camera:test_640x480") -> ACWMActionCondition:
    return ACWMActionCondition(
        label="carry-right",
        instruction="Carry the yellow bowl to the right and hold it.",
        timeline="approach, grasp, carry right, hold",
        representation=ActionRepresentation.CAMERA_RELATIVE_EEF_DELTA,
        coordinate_frame=frame,
        timestamps_s=(0.0, 1.0 / 15.0),
        channels=("delta_x_px", "delta_y_px"),
        values=((0.0, 0.0), (24.0, -2.0)),
    )


def _score(
    seed: int,
    *,
    action: float,
    embodiment: float = 0.9,
    interaction: float = 0.9,
    temporal: float = 0.9,
    background: float = 0.9,
    hard_gates: bool = True,
    human_review: bool | None = None,
) -> CandidateScore:
    return CandidateScore(
        seed=seed,
        action_adherence=action,
        embodiment_consistency=embodiment,
        object_interaction=interaction,
        temporal_consistency=temporal,
        background_consistency=background,
        hard_gates_passed=hard_gates,
        human_review_passed=human_review,
        evaluator="test-inverse-evaluator",
    )


def test_fit_blur_pad_transform_round_trips_camera_pixels() -> None:
    transform = FitBlurPadTransform.create(
        source_frame="camera:oscar_640x480_pixels",
        target_frame="camera:joyai_center_crop_1248x720",
        source_width=640,
        source_height=480,
        target_width=1248,
        target_height=720,
    )

    assert transform.scale == 1.5
    assert (transform.resized_width, transform.resized_height) == (960, 720)
    assert (transform.offset_x, transform.offset_y) == (144, 0)
    projected = transform.forward_xy(320.0, 240.0)
    assert projected == pytest.approx((624.0, 360.0))
    assert transform.inverse_xy(*projected) == pytest.approx((320.0, 240.0))
    prepare_filter = transform.prepare_filter(
        tail_padding_frames=0,
        output_fps=24.0,
        resample_fps=24.0,
    )
    assert "gblur=sigma=30" in prepare_filter
    assert "[bg][fg]overlay=144:0" in prepare_filter
    assert "tpad=stop_mode=clone:stop=1,fps=fps=24:round=near" in prepare_filter
    assert "mirror" not in prepare_filter
    assert transform.restore_filter().startswith("crop=960:720:144:0")


def test_model_cadence_resampling_is_exact_and_causally_padded() -> None:
    assert (
        resampled_frame_count(
            81,
            source_fps_numerator=15,
            source_fps_denominator=1,
            target_fps=24,
        )
        == 130
    )


def test_action_prompt_makes_carrier_motion_authoritative() -> None:
    prompt = compile_action_preserving_prompt(_action())

    assert "immutable action carrier" in prompt
    assert "Do not reinterpret, reverse, amplify, smooth, or replace" in prompt
    assert "Carry the yellow bowl to the right" in prompt
    assert "camera:test_640x480" in prompt


def test_candidate_selection_applies_hard_gates_before_inverse_error() -> None:
    thresholds = ConsistencyThresholds()
    scores = (
        _score(1, action=0.99, temporal=0.60),
        _score(2, action=0.88),
        _score(3, action=0.80),
    )

    assert select_consistent_candidate(scores, thresholds) == 1
    assert scores[1].inverse_action_error == pytest.approx(0.12)


def test_candidate_selection_keeps_best_failed_diagnostic() -> None:
    thresholds = ConsistencyThresholds()
    scores = (
        _score(1, action=0.70, temporal=0.70),
        _score(2, action=0.74, temporal=0.74),
    )

    assert select_consistent_candidate(scores, thresholds) == 1


def test_human_veto_cannot_win_candidate_selection() -> None:
    thresholds = ConsistencyThresholds()
    scores = (
        _score(1, action=0.99, human_review=False),
        _score(2, action=0.80, human_review=None),
    )

    assert select_consistent_candidate(scores, thresholds) == 1


def test_video_stream_parses_counted_frames_and_fractional_rate() -> None:
    stream = VideoStream.from_ffprobe(
        {
            "streams": [
                {
                    "width": 640,
                    "height": 480,
                    "avg_frame_rate": "30000/1001",
                    "nb_read_frames": "81",
                    "duration": "2.7027",
                }
            ]
        }
    )

    assert stream.frame_count == 81
    assert stream.fps == pytest.approx(29.97002997)


def test_video_stream_derives_duration_when_container_omits_it() -> None:
    stream = VideoStream.from_ffprobe(
        {
            "streams": [
                {
                    "width": 1248,
                    "height": 720,
                    "avg_frame_rate": "15/1",
                    "nb_read_frames": "81",
                }
            ]
        }
    )

    assert stream.duration_seconds == pytest.approx(5.4)


def test_server_manifest_requires_pinned_ready_two_gpu_service(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    payload = {
        "status": "WORKING",
        "stage": "joyai_server_ready",
        "source": {"revision": JOYAI_REPOSITORY_REVISION},
        "checkpoints": {
            "model_revision": JOYAI_MODEL_REVISION,
            "text_encoder_revision": JOYAI_TEXT_ENCODER_REVISION,
        },
        "gpu": {
            "selected": [
                {"logical_index": 0, "physical_index": 4},
                {"logical_index": 1, "physical_index": 7},
            ],
            "cuda_visible_devices": "4,7",
        },
        "runtime": {"cuda_available": True, "cuda_device_count": 2},
        "health_url": "http://127.0.0.1:18080/health",
    }
    manifest.write_text(json.dumps(payload))

    record = validate_server_manifest(manifest)
    assert record["physical_gpus"] == [4, 7]

    payload["stage"] = "joyai_server_stopped"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(JoyAISC3PreflightError, match="joyai_server_ready"):
        validate_server_manifest(manifest)


def test_runtime_seed_subset_cannot_escape_frozen_plan(tmp_path: Path) -> None:
    config = JoyAISC3Config(
        experiment_root=tmp_path / "runs",
        action_condition=tmp_path / "action.json",
        first_frame=tmp_path / "first.png",
        source_video=tmp_path / "source.mp4",
        carrier=CarrierContract(
            video=tmp_path / "carrier.mp4",
            coordinate_frame="camera:test",
            generator="test",
            generator_revision="revision",
            motion_authority="camera-image action only",
        ),
        client_script=tmp_path / "client.py",
        client_python=tmp_path / "python",
        evaluator_python=tmp_path / "python",
        evaluator_command=(
            "{python}",
            "evaluate.py",
            "{candidate}",
            "{condition}",
            "{first_frame}",
            "{source}",
            "{metadata}",
        ),
        candidate_seeds=(42, 101),
    )

    selected = with_runtime_overrides(config, candidate_seeds=(42,))
    assert selected.candidate_seeds == (42,)
    with pytest.raises(ValueError, match="not in the frozen"):
        with_runtime_overrides(config, candidate_seeds=(999,))


def test_prepare_only_writes_real_contract_without_model_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    action_path = tmp_path / "action.json"
    _action().to_json(action_path)
    carrier = tmp_path / "carrier.mp4"
    first_frame = tmp_path / "first.png"
    source = tmp_path / "source.mp4"
    client = tmp_path / "client.py"
    python = tmp_path / "python"
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    for path in (carrier, first_frame, source, client, python, ffmpeg, ffprobe):
        path.write_bytes(b"test")
    config = JoyAISC3Config(
        experiment_root=tmp_path / "runs",
        action_condition=action_path,
        first_frame=first_frame,
        source_video=source,
        carrier=CarrierContract(
            video=carrier,
            coordinate_frame="camera:test_640x480",
            generator="test action model",
            generator_revision="revision-1",
            motion_authority="image-space motion and timing only",
        ),
        client_script=client,
        client_python=python,
        evaluator_python=python,
        evaluator_command=(
            "{python}",
            "evaluate.py",
            "--candidate",
            "{candidate}",
            "--condition",
            "{condition}",
            "--first-frame",
            "{first_frame}",
            "--source",
            "{source}",
            "--metadata",
            "{metadata}",
        ),
        candidate_seeds=(42, 101),
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )
    runner = JoyAISC3Runner(config, project_root=Path(__file__).resolve().parents[1])
    source_stream = VideoStream(640, 480, 15, 1, 2, 2 / 15)
    prepared_stream = VideoStream(1248, 720, 24, 1, 9, 9 / 24)

    def fake_probe(path: Path) -> VideoStream:
        return prepared_stream if "joyai-1248x720" in path.name else source_stream

    def fake_command(command: list[str], log_path: Path, *, timeout: float) -> str:
        del log_path, timeout
        output = Path(command[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"prepared")
        return ""

    monkeypatch.setattr(runner, "_probe_video", fake_probe)
    monkeypatch.setattr(runner, "_probe_dimensions", lambda path: (640, 480))
    monkeypatch.setattr(runner, "_run_command", fake_command)
    monkeypatch.setattr(runner, "_git_state", lambda: {"head": "test"})
    monkeypatch.setattr(
        runner,
        "_package_state",
        lambda run_dir: {"path": str(run_dir / "packages.txt"), "sha256": "test"},
    )

    result = runner.run(prepare_only=True)

    assert result["status"] == "PARTIAL"
    assert result["stage"] == "joyai_sc3_prepared_not_run"
    assert result["model_inference"] == "NOT STARTED"
    assert result["acceptance"]["harness_preparation"] == "WORKING"
    assert result["preflight"]["causal_padding"] == {
        "source_action_frames": 2,
        "source_fps": 15.0,
        "model_deliverable_frames": 3,
        "model_fps": 24,
        "padded_frames": 9,
        "tail_clones": 6,
        "source_end_support_frames_before_resampling": 1,
        "temporal_resampling": "nearest_timestamp_no_interpolation",
        "trim_after_generation": True,
        "restore_source_fps_after_trim": True,
    }
