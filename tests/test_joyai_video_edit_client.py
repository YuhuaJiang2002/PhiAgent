from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.run_joyai_video_edit_client import (
    OutputFrameSink,
    cleanup_protocol_frames,
    mux_outputs,
    resolve_throughput_options,
)


def test_mjpeg_sink_keeps_one_contiguous_bounded_spool(tmp_path: Path) -> None:
    sink = OutputFrameSink(tmp_path, "mjpeg")
    first = sink.write(0, b"\xff\xd8first\xff\xd9")
    second = sink.write(1, b"\xff\xd8second\xff\xd9")
    manifest = sink.to_manifest()
    sink.close()

    assert first["spool_offset"] == 0
    assert second["spool_offset"] == len(b"\xff\xd8first\xff\xd9")
    assert (tmp_path / "frames.mjpeg").read_bytes() == (
        b"\xff\xd8first\xff\xd9\xff\xd8second\xff\xd9"
    )
    assert manifest["frames"] == 2
    assert manifest["bytes"] == len((tmp_path / "frames.mjpeg").read_bytes())


def test_frame_sink_rejects_noncontiguous_output(tmp_path: Path) -> None:
    sink = OutputFrameSink(tmp_path, "mjpeg")
    try:
        with pytest.raises(ValueError, match="not contiguous"):
            sink.write(1, b"frame")
    finally:
        sink.close()


def test_cleanup_removes_only_known_mjpeg_spool(tmp_path: Path) -> None:
    spool = tmp_path / "frames.mjpeg"
    unrelated = tmp_path / "keep.txt"
    spool.write_bytes(b"frames")
    unrelated.write_text("keep")
    args = argparse.Namespace(keep_protocol_frames=False, frame_storage="mjpeg")

    cleanup_protocol_frames(args, tmp_path)

    assert not spool.exists()
    assert unrelated.read_text() == "keep"


def test_throughput_mode_resolves_bounded_dataset_defaults() -> None:
    args = argparse.Namespace(
        throughput_mode=True,
        input_loading=None,
        ack_window=None,
        frame_log_every=None,
        frame_storage=None,
        keep_protocol_frames=None,
        output_artifacts=None,
    )

    resolved = resolve_throughput_options(args)

    assert resolved.input_loading == "stream"
    assert resolved.ack_window == 8
    assert resolved.frame_log_every == 240
    assert resolved.frame_storage == "mjpeg"
    assert resolved.keep_protocol_frames is False
    assert resolved.output_artifacts == "review"


def test_throughput_mode_preserves_explicit_values_for_validation() -> None:
    args = argparse.Namespace(
        throughput_mode=True,
        input_loading="memory",
        ack_window=0,
        frame_log_every=0,
        frame_storage="files",
        keep_protocol_frames=True,
        output_artifacts="lossless",
    )

    resolved = resolve_throughput_options(args)

    assert resolved.ack_window == 0
    assert resolved.frame_log_every == 0
    assert resolved.keep_protocol_frames is True


def test_review_only_mjpeg_mux_uses_one_image_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_text("")
    (tmp_path / "frames.mjpeg").write_bytes(b"frames")
    args = argparse.Namespace(
        ffmpeg=ffmpeg,
        frame_storage="mjpeg",
        fps=24,
        output_artifacts="review",
    )
    observed: list[list[str]] = []

    def fake_run(
        command: list[str], *, capture_output: bool, text: bool, check: bool
    ) -> argparse.Namespace:
        observed.append(command)
        Path(command[-1]).write_bytes(b"review")
        return argparse.Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.run_joyai_video_edit_client.subprocess.run", fake_run)
    outputs, commands = mux_outputs(args, tmp_path)

    assert commands == observed
    assert len(commands) == 1
    assert commands[0][commands[0].index("-f") + 1] == "image2pipe"
    assert commands[0][-1].endswith("joyai-proposal-review.mp4")
    assert set(outputs) == {"review"}
