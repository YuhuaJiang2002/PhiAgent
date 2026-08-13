from __future__ import annotations

import json
import threading
import urllib.request
from argparse import Namespace
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from phiagent.acwm.adapters import (
    ACWMRenderRequest,
    ACWMRenderResult,
    BackendSupport,
)
from phiagent.acwm.numeric import (
    BWM_ACTION_FPS,
    BWM_ACTION_FRAMES,
    NumericActionKeyframe,
    NumericActionStatistics,
    compile_bwm_eef_action,
    compile_bwm_eef_payload,
)
from phiagent.acwm.robotwin import BWM_EEF_CHANNELS
from phiagent.acwm.schema import ACWMCase
from phiagent.acwm.worldarena import WORLD_ARENA_EEF_QUATERNION_CHANNELS
from phiagent.agent.numeric_action import (
    NumericActionJobManager,
    NumericActionScene,
    NumericActionVideoAgent,
)
from phiagent.learning.experience import load_experiences
from scripts.serve_numeric_acwm_demo import _handler_factory, parse_byte_range
from scripts.run_acwm_backend import _bwm_inference_command


def _row(*, offset: float = 0.0) -> tuple[float, ...]:
    values = [offset] * 14
    values[6] = 0.25
    values[13] = 0.75
    return tuple(values)


def _payload() -> dict[str, object]:
    start = list(_row())
    end = list(_row())
    end[0] = 0.14
    end[9] = -0.07
    end[13] = 0.2
    return {
        "instruction": "Move the dual-arm end effectors along this exact numeric action.",
        "keyframes": [
            {"frame": 0, "values": start},
            {"frame": BWM_ACTION_FRAMES - 1, "values": end},
        ],
    }


def _job_payload() -> dict[str, object]:
    return {**_payload(), "coordinate_frame": "robot_base:test"}


def test_numeric_keyframes_compile_to_exact_bwm_contract() -> None:
    start = _row()
    end = list(_row())
    end[0] = 0.56
    end[7] = -0.28

    compiled = compile_bwm_eef_action(
        label="numeric-test",
        instruction="Move both end effectors to the exact requested terminal state.",
        prompt="Render the robot following the supplied exact action sequence.",
        coordinate_frame="robot_base:test",
        keyframes=(
            NumericActionKeyframe(0, start),
            NumericActionKeyframe(BWM_ACTION_FRAMES - 1, tuple(end)),
        ),
    )

    condition = compiled.condition
    assert condition.channels == BWM_EEF_CHANNELS
    assert len(condition.values) == BWM_ACTION_FRAMES
    assert condition.fps == pytest.approx(BWM_ACTION_FPS)
    assert condition.values[0] == start
    assert condition.values[28][0] == pytest.approx(0.28)
    assert condition.values[28][7] == pytest.approx(-0.14)
    assert condition.values[-1] == tuple(end)
    assert compiled.summary["source_mode"] == "piecewise_linear_keyframes"
    assert compiled.summary["frames"] == BWM_ACTION_FRAMES


def test_exact_numeric_samples_are_not_interpolated_or_relabelled() -> None:
    samples = []
    for frame in range(BWM_ACTION_FRAMES):
        row = list(_row())
        row[0] = frame / 1000
        row[7] = -(frame * frame) / 100_000
        samples.append(row)

    compiled = compile_bwm_eef_payload(
        {
            "instruction": "Use every supplied EEF sample exactly as provided.",
            "prompt": "Generate the video from the exact per-frame robot action.",
            "samples": samples,
        },
        label="sampled-action",
        coordinate_frame="robot_base:measured",
    )

    assert compiled.source_mode == "exact_samples"
    assert compiled.condition.values == tuple(tuple(row) for row in samples)
    assert compiled.condition.coordinate_frame == "robot_base:measured"


def _quaternion_row(*, right_x: float = 0.0, right_w: float = 1.0) -> tuple[float, ...]:
    return (
        0.1,
        0.2,
        0.3,
        0.0,
        0.0,
        0.0,
        1.0,
        0.4,
        0.5,
        0.6,
        right_x,
        0.0,
        0.0,
        right_w,
    )


def test_quaternion_keyframes_use_unit_slerp() -> None:
    compiled = compile_bwm_eef_action(
        label="quaternion-action",
        instruction="Rotate the right EEF with an exact quaternion action.",
        prompt="Render the supplied quaternion-conditioned robot motion.",
        coordinate_frame="robot_base:quaternion-test",
        channels=WORLD_ARENA_EEF_QUATERNION_CHANNELS,
        keyframes=(
            NumericActionKeyframe(0, _quaternion_row()),
            NumericActionKeyframe(
                BWM_ACTION_FRAMES - 1,
                _quaternion_row(right_x=1.0, right_w=0.0),
            ),
        ),
        fps=30.0,
    )

    middle = compiled.condition.values[BWM_ACTION_FRAMES // 2]
    assert compiled.condition.fps == pytest.approx(30.0)
    assert compiled.source_mode == "piecewise_linear_position_slerp_quaternion_keyframes"
    assert middle[10] == pytest.approx(2**-0.5)
    assert middle[13] == pytest.approx(2**-0.5)
    assert sum(value * value for value in middle[10:14]) == pytest.approx(1.0)


def test_quaternion_samples_reject_non_unit_orientation() -> None:
    invalid = list(_quaternion_row())
    invalid[13] = 0.5
    with pytest.raises(ValueError, match="quaternion.*norm"):
        compile_bwm_eef_action(
            label="invalid-quaternion",
            instruction="Reject this invalid quaternion-conditioned action.",
            prompt="This invalid action must not reach the video model.",
            coordinate_frame="robot_base:quaternion-test",
            channels=WORLD_ARENA_EEF_QUATERNION_CHANNELS,
            samples=(tuple(invalid),) * BWM_ACTION_FRAMES,
            fps=30.0,
        )


def test_matching_action_statistics_validate_frame_channels_and_range(
    tmp_path: Path,
) -> None:
    stats_path = tmp_path / "action-stat.json"
    stats_path.write_text(
        json.dumps(
            {
                "state_pose": {
                    "coordinate_frame": "robot_base:quaternion-test",
                    "channels": list(WORLD_ARENA_EEF_QUATERNION_CHANNELS),
                    "min": [-1.0] * 14,
                    "max": [1.0] * 14,
                    "p01": [-0.9] * 14,
                    "p99": [0.9] * 14,
                }
            }
        )
    )
    stats = NumericActionStatistics.from_json(stats_path)
    condition = compile_bwm_eef_action(
        label="stats-action",
        instruction="Validate this action against its matching statistics.",
        prompt="Render this in-distribution quaternion action.",
        coordinate_frame="robot_base:quaternion-test",
        channels=WORLD_ARENA_EEF_QUATERNION_CHANNELS,
        samples=(_quaternion_row(),) * BWM_ACTION_FRAMES,
        fps=30.0,
    ).condition

    summary = stats.validate(condition)

    assert summary["outside_minmax_count"] == 0
    assert summary["outside_p01_p99_count"] == BWM_ACTION_FRAMES * 2


def test_bwm_command_receives_exact_frame_and_sampling_counts(tmp_path: Path) -> None:
    command = _bwm_inference_command(
        Namespace(
            repository=tmp_path / "bwm",
            config=tmp_path / "infer.yaml",
            base_model=tmp_path / "base",
            checkpoint=tmp_path / "checkpoint.safetensors",
            action_stats=tmp_path / "stats.json",
        ),
        data=tmp_path / "data",
        metadata_path=tmp_path / "episodes.jsonl",
        raw_outputs=tmp_path / "raw",
        action_type="eef_abs",
        seed=31,
        num_frames=BWM_ACTION_FRAMES,
        num_inference_steps=20,
        guidance_scale=3.5,
        fps=24,
        max_samples=1,
    )

    assert command[command.index("--action_type") + 1] == "eef_abs"
    assert command[command.index("--num_frames") + 1] == str(BWM_ACTION_FRAMES)
    assert command[command.index("--num_inference_steps") + 1] == "20"
    assert command[command.index("--cfg_scale") + 1] == "3.5"
    assert command[command.index("--fps") + 1] == "24"
    assert command[command.index("--seed") + 1] == "31"


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (
            {
                "instruction": "Use this malformed action sequence.",
                "keyframes": [
                    {"frame": 1, "values": _row()},
                    {"frame": 56, "values": _row()},
                ],
            },
            "frames 0 and 56",
        ),
        (
            {
                "instruction": "Use this malformed action sequence.",
                "keyframes": [
                    {"frame": 0, "values": (*_row()[:13], 1.2)},
                    {"frame": 56, "values": _row()},
                ],
            },
            "right_gripper_open",
        ),
        (
            {
                "instruction": "Use this malformed action sequence.",
                "samples": [_row()] * 2,
            },
            "57 rows",
        ),
    ),
)
def test_numeric_payload_rejects_ambiguous_or_invalid_actions(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compile_bwm_eef_payload(
            payload,
            label="invalid-action",
            coordinate_frame="robot_base:test",
        )


def test_numeric_payload_rejects_camera_frame() -> None:
    with pytest.raises(ValueError, match="robot_base"):
        compile_bwm_eef_payload(
            _payload(),
            label="invalid-frame",
            coordinate_frame="camera:rgb_pixels",
        )


def test_numeric_scene_enforces_configured_first_frame_state(tmp_path: Path) -> None:
    first_frame = tmp_path / "first.png"
    source_video = tmp_path / "source.mp4"
    first_frame.write_bytes(b"png")
    source_video.write_bytes(b"mp4")
    default = compile_bwm_eef_payload(
        _payload(),
        label="default-action",
        coordinate_frame="robot_base:test",
    ).condition
    scene = NumericActionScene(
        first_frame=first_frame,
        source_video=source_video,
        coordinate_frame="robot_base:test",
        default_condition=default,
    )
    changed_start = list(_row())
    changed_start[0] = 0.01
    changed_payload = {
        "instruction": "Move from a mismatched initial EEF state.",
        "keyframes": [
            {"frame": 0, "values": changed_start},
            {"frame": BWM_ACTION_FRAMES - 1, "values": _row()},
        ],
    }
    changed = compile_bwm_eef_payload(
        changed_payload,
        label="changed-action",
        coordinate_frame="robot_base:test",
    ).condition

    with pytest.raises(ValueError, match="does not match.*left_eef_pos_x_m"):
        scene.validate_initial_state(changed)


class _FakeBWMRenderer:
    name = "bwm"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[ACWMRenderRequest] = []

    def supports(self, case: ACWMCase) -> BackendSupport:
        supported = tuple(case.action.channels) == BWM_EEF_CHANNELS
        return BackendSupport(self.name, supported, () if supported else ("wrong channels",))

    def render_batch(
        self, requests: tuple[ACWMRenderRequest, ...]
    ) -> tuple[ACWMRenderResult, ...]:
        self.requests.extend(requests)
        if self.fail:
            raise RuntimeError("synthetic backend failure")
        results = []
        for request in requests:
            request.output.parent.mkdir(parents=True, exist_ok=True)
            request.output.write_bytes(b"fake-mp4")
            metadata = request.output.with_suffix(".metadata.json")
            metadata.write_text('{"backend": "fake-bwm"}\n')
            results.append(
                ACWMRenderResult(
                    backend=self.name,
                    case_id=request.case.case_id,
                    output=request.output,
                    metadata=metadata,
                    experiment_dir=request.experiment_dir,
                )
            )
        return tuple(results)


def _manager(tmp_path: Path, renderer: _FakeBWMRenderer) -> NumericActionJobManager:
    first_frame = tmp_path / "scene" / "first.png"
    source_video = tmp_path / "scene" / "source.mp4"
    first_frame.parent.mkdir(parents=True)
    first_frame.write_bytes(b"png")
    source_video.write_bytes(b"mp4")
    return NumericActionJobManager(
        NumericActionVideoAgent(
            renderer,
            project_root=tmp_path,
            provenance_provider=lambda _: {"test": True},
        ),
        scene=NumericActionScene(
            first_frame=first_frame,
            source_video=source_video,
            coordinate_frame="robot_base:test",
        ),
        jobs_root=tmp_path / "jobs",
        experiment_root=tmp_path / "experiments",
        ledger_path=tmp_path / "ledger.jsonl",
    )


def test_numeric_job_requires_explicit_matching_coordinate_frame(tmp_path: Path) -> None:
    manager = _manager(tmp_path, _FakeBWMRenderer())
    try:
        with pytest.raises(ValueError, match="requires coordinate_frame"):
            manager.submit(_payload())
        with pytest.raises(ValueError, match="scene requires"):
            manager.submit({**_payload(), "coordinate_frame": "robot_base:other"})
    finally:
        manager.close()


def test_numeric_job_generates_persists_and_publishes_video(tmp_path: Path) -> None:
    renderer = _FakeBWMRenderer()
    manager = _manager(tmp_path, renderer)
    try:
        submitted = manager.submit(_job_payload())
        completed = manager.wait(str(submitted["job_id"]))

        assert completed["status"] == "generated_pending_review"
        assert completed["video_url"] == (
            f"/api/numeric-jobs/{submitted['job_id']}/video"
        )
        assert manager.video_path(str(submitted["job_id"])).read_bytes() == b"fake-mp4"
        action = manager.action_payload(str(submitted["job_id"]))
        assert action is not None
        assert action["representation"] == "eef_absolute"
        assert action["coordinate_frame"] == "robot_base:test"
        assert len(action["values"]) == BWM_ACTION_FRAMES
        assert renderer.requests[0].case.action.values[-1][0] == pytest.approx(0.14)
        records = load_experiences(tmp_path / "ledger.jsonl")
        assert len(records) == 1
        assert records[0].status == "PARTIAL"
    finally:
        manager.close()


def test_numeric_job_records_backend_failure_without_preset_fallback(tmp_path: Path) -> None:
    manager = _manager(tmp_path, _FakeBWMRenderer(fail=True))
    try:
        submitted = manager.submit(_job_payload())
        with pytest.raises(RuntimeError, match="synthetic backend failure"):
            manager.wait(str(submitted["job_id"]))

        failed = manager.get(str(submitted["job_id"]))
        assert failed is not None
        assert failed["status"] == "failed"
        assert failed["video_url"] is None
        assert manager.video_path(str(submitted["job_id"])) is None
        records = load_experiences(tmp_path / "ledger.jsonl")
        assert len(records) == 1
        assert records[0].status == "BLOCKED"
    finally:
        manager.close()


def test_numeric_http_api_serves_contract_action_and_video_range(tmp_path: Path) -> None:
    manager = _manager(tmp_path, _FakeBWMRenderer())
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "index.html").write_text("<!doctype html><title>numeric test</title>")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_factory(manager, demo))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(f"{root}/api/numeric-capabilities") as response:
            capabilities = json.load(response)
        assert capabilities["action_contract"]["channels"][0]["unit"] == "m"

        request = urllib.request.Request(
            f"{root}/api/numeric-jobs",
            data=json.dumps(_job_payload()).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            submitted = json.load(response)
        manager.wait(submitted["job_id"])

        with urllib.request.urlopen(
            f"{root}/api/numeric-jobs/{submitted['job_id']}/action"
        ) as response:
            action = json.load(response)
        assert action["channels"] == list(BWM_EEF_CHANNELS)

        video_request = urllib.request.Request(
            f"{root}/api/numeric-jobs/{submitted['job_id']}/video",
            headers={"Range": "bytes=0-3"},
        )
        with urllib.request.urlopen(video_request) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 0-3/8"
            assert response.read() == b"fake"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        manager.close()


@pytest.mark.parametrize(
    ("header", "size", "expected"),
    (
        (None, 100, None),
        ("bytes=0-9", 100, (0, 9)),
        ("bytes=90-", 100, (90, 99)),
        ("bytes=-10", 100, (90, 99)),
        ("bytes=0-999", 100, (0, 99)),
    ),
)
def test_video_byte_ranges(
    header: str | None, size: int, expected: tuple[int, int] | None
) -> None:
    assert parse_byte_range(header, size) == expected


def test_video_byte_range_rejects_multiple_ranges() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        parse_byte_range("bytes=0-2,5-7", 100)
