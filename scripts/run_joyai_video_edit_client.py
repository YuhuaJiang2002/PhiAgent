#!/usr/bin/env python3
"""Deterministic file-to-file client for the official JoyAI WebSocket server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.joyai_video_edit import (  # noqa: E402
    DEFAULT_SCISSORS_CONTRACT,
    JOYAI_MODEL_ID,
    JOYAI_MODEL_REVISION,
    JOYAI_REPOSITORY_REVISION,
    flower_full_stream_prompt,
    flower_repair_prompt,
    sha256_file,
    write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="ws://127.0.0.1:18080/ws")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-video", type=Path)
    inputs.add_argument(
        "--input-frame-dir",
        type=Path,
        help="pre-encoded protocol JPEGs named in lexical frame order",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path)
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument("--prompt")
    prompt_source.add_argument(
        "--prompt-file",
        type=Path,
        help="Read the exact editing instruction from UTF-8 text without shell interpolation.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("repair_window", "full_flower_stream"),
        default="repair_window",
        help="Use a checked-in prompt contract when --prompt is omitted.",
    )
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--expected-frames", type=int, default=33)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--output-quality", type=int, default=95)
    parser.add_argument(
        "--throughput-mode",
        action="store_true",
        help=(
            "Use bounded-memory dataset defaults: streamed input, one causal chunk "
            "per ACK, sampled frame logs, one MJPEG spool, review-only output, and "
            "spool cleanup after successful muxing."
        ),
    )
    parser.add_argument(
        "--input-loading",
        choices=("memory", "stream"),
        help="Input decode strategy (default: stream in throughput mode, otherwise memory).",
    )
    parser.add_argument(
        "--ack-window",
        type=int,
        help="Maximum submitted frames before waiting for server ACK (default: 8 in throughput mode, otherwise 1).",
    )
    parser.add_argument(
        "--frame-log-every",
        type=int,
        help="Log every Nth frame event; 0 keeps only boundary frames (default: 240 in throughput mode, otherwise 1).",
    )
    parser.add_argument(
        "--frame-storage",
        choices=("files", "mjpeg"),
        help="Protocol-frame staging format (default: mjpeg in throughput mode, otherwise files).",
    )
    parser.add_argument(
        "--keep-protocol-frames",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retain JPEG protocol frames after successful muxing.",
    )
    parser.add_argument(
        "--output-artifacts",
        choices=("both", "lossless", "review"),
        help="Final artifacts to encode (default: review in throughput mode, otherwise both).",
    )
    parser.add_argument(
        "--profile-timings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Synchronize CUDA stage timers (disabled by default for production throughput).",
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    return parser


def iter_input_frames(video: Path, *, width: int, height: int) -> Iterator[bytes]:
    """Decode exact pixels and JPEG-encode frames lazily for the official uplink."""

    try:
        import av
    except ImportError:  # pragma: no cover - local preparation fallback
        av = None
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional runtime only
        raise RuntimeError("JoyAI client requires Pillow") from exc
    if av is not None:
        with av.open(str(video)) as container:
            streams = container.streams.video
            if len(streams) != 1:
                raise ValueError("JoyAI input must contain exactly one video stream")
            stream = streams[0]
            if (stream.codec_context.width, stream.codec_context.height) != (width, height):
                raise ValueError(
                    f"JoyAI input is {stream.codec_context.width}x{stream.codec_context.height}; "
                    f"expected {width}x{height}"
                )
            arrays = (frame.to_ndarray(format="rgb24") for frame in container.decode(stream))
            for array in arrays:
                encoded = io.BytesIO()
                Image.fromarray(array, mode="RGB").save(
                    encoded, format="JPEG", quality=95, subsampling=0, optimize=False
                )
                yield encoded.getvalue()
        return

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional runtime has PyAV
        raise RuntimeError("JoyAI input decoding requires PyAV or OpenCV") from exc
    capture = cv2.VideoCapture(str(video))
    try:
        observed = (
            round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if observed != (width, height):
            raise ValueError(
                f"JoyAI input is {observed[0]}x{observed[1]}; expected {width}x{height}"
            )
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            encoded = io.BytesIO()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb, mode="RGB").save(
                encoded, format="JPEG", quality=95, subsampling=0, optimize=False
            )
            yield encoded.getvalue()
    finally:
        capture.release()


def decode_input_frames(video: Path, *, width: int, height: int) -> list[bytes]:
    """Decode all uplink frames for the compatibility, short-clip path."""

    return list(iter_input_frames(video, width=width, height=height))


def load_input_frame_dir(
    frame_dir: Path, *, width: int, height: int, expected_frames: int
) -> tuple[list[bytes], dict[str, Any]]:
    """Load exact JPEG uplink packets without a second lossy transcode."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional runtime only
        raise RuntimeError("JoyAI client requires Pillow") from exc
    paths = sorted(frame_dir.glob("*.jpg"))
    if len(paths) != expected_frames:
        raise ValueError(
            f"input frame directory has {len(paths)} JPEGs; expected {expected_frames}"
        )
    frames: list[bytes] = []
    members = []
    for index, path in enumerate(paths):
        payload = path.read_bytes()
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "JPEG":
                raise ValueError(f"input frame {path} is not JPEG")
            if image.size != (width, height):
                raise ValueError(
                    f"input frame {path} is {image.width}x{image.height}; "
                    f"expected {width}x{height}"
                )
        frames.append(payload)
        members.append(
            {
                "index": index,
                "name": path.name,
                "bytes": len(payload),
                "sha256": sha256_file(path),
            }
        )
    return frames, {
        "kind": "preencoded_protocol_jpeg_directory",
        "path": str(frame_dir),
        "frames": len(frames),
        "total_bytes": sum(len(frame) for frame in frames),
        "members": members,
    }


def _reference_data_url(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    suffix = resolved.suffix.lower()
    media = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{media};base64,{base64.b64encode(resolved.read_bytes()).decode('ascii')}"


async def _connect(url: str):
    try:
        from websockets.asyncio.client import connect
    except ImportError:  # pragma: no cover - compatibility with websockets < 14
        try:
            from websockets import connect
        except ImportError as exc:
            raise RuntimeError("JoyAI client requires the optional 'websockets' package") from exc
    return await connect(url, max_size=64 * 1024 * 1024, ping_interval=5, ping_timeout=60)


class OutputFrameSink:
    """Write protocol JPEGs either as files or one concatenated MJPEG stream."""

    def __init__(self, output: Path, storage: str) -> None:
        if storage not in {"files", "mjpeg"}:
            raise ValueError(f"unsupported frame storage: {storage}")
        self.storage = storage
        self.frame_count = 0
        self.total_bytes = 0
        self._stream: BinaryIO | None = None
        if storage == "files":
            self.path = output / "frames-jpeg"
            self.path.mkdir()
        else:
            self.path = output / "frames.mjpeg"
            self._stream = self.path.open("xb")

    def write(self, index: int, payload: bytes) -> dict[str, Any]:
        if index != self.frame_count:
            raise ValueError(
                f"output frame index {index} is not contiguous after {self.frame_count} frames"
            )
        if self.storage == "files":
            frame_path = self.path / f"{index:06d}.jpg"
            frame_path.write_bytes(payload)
            location = {"saved_path": str(frame_path)}
        else:
            if self._stream is None:
                raise RuntimeError("MJPEG frame sink is closed")
            offset = self._stream.tell()
            self._stream.write(payload)
            location = {"spool_path": str(self.path), "spool_offset": offset}
        self.frame_count += 1
        self.total_bytes += len(payload)
        return location

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": "jpeg_file_sequence" if self.storage == "files" else "concatenated_mjpeg",
            "path": str(self.path),
            "frames": self.frame_count,
            "bytes": self.total_bytes,
        }

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


class ProtocolRun:
    def __init__(
        self,
        frame_sink: OutputFrameSink,
        events_path: Path,
        expected_frames: int,
        frame_log_every: int,
    ) -> None:
        self.frame_sink = frame_sink
        self.events_path = events_path
        self.expected_frames = expected_frames
        self.frame_log_every = frame_log_every
        self.session_granted = asyncio.Event()
        self.started = asyncio.Event()
        self.outputs_done = asyncio.Event()
        self.finalized = asyncio.Event()
        self.ack_condition = asyncio.Condition()
        self.acked_frames = 0
        self.output_count = 0
        self.pending_output: dict[str, Any] | None = None
        self.error: str | None = None
        self.chunk_profiles: dict[str, dict[str, Any]] = {}
        self._events_stream = self.events_path.open("a", encoding="utf-8", buffering=1)

    def log(self, row: dict[str, Any]) -> None:
        self._events_stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def should_log_frame(self, one_based_index: int) -> bool:
        if one_based_index in {1, self.expected_frames}:
            return True
        return self.frame_log_every > 0 and one_based_index % self.frame_log_every == 0

    def close(self) -> None:
        self.frame_sink.close()
        self._events_stream.close()

    async def receiver(self, websocket: Any) -> None:
        try:
            async for message in websocket:
                now = time.time()
                if isinstance(message, str):
                    payload = json.loads(message)
                    kind = str(payload.get("type", "unknown"))
                    frame_event_index: int | None = None
                    if kind in {"frame_ack", "accepted", "chunk_done"}:
                        frame_event_index = int(payload.get("frames_in", 0))
                    elif kind == "output_frame":
                        frame_event_index = self.output_count + 1
                    if frame_event_index is None or self.should_log_frame(frame_event_index):
                        self.log(
                            {
                                "direction": "server_to_client",
                                "received_at": now,
                                "message": payload,
                            }
                        )
                    if kind == "session_granted":
                        self.session_granted.set()
                    elif kind == "started":
                        self.started.set()
                    elif kind in {"frame_ack", "accepted", "chunk_done"}:
                        async with self.ack_condition:
                            self.acked_frames = max(
                                self.acked_frames, int(payload.get("frames_in", 0))
                            )
                            self.ack_condition.notify_all()
                    elif kind == "output_frame":
                        if self.pending_output is not None:
                            self.error = (
                                "received output_frame metadata before prior binary payload"
                            )
                            self.outputs_done.set()
                            return
                        self.pending_output = payload
                        profile = payload.get("profile")
                        if isinstance(profile, dict):
                            key = str(profile.get("chunk_idx", len(self.chunk_profiles)))
                            self.chunk_profiles[key] = profile
                    elif kind == "recording_finalized":
                        if payload.get("ok") is not True:
                            self.error = f"server failed to finalize recording: {payload}"
                        self.finalized.set()
                    elif kind in {"error", "session_timeout"}:
                        self.error = str(payload.get("message") or payload)
                        self.outputs_done.set()
                        self.finalized.set()
                else:
                    if self.pending_output is None:
                        self.error = "received binary payload without output_frame metadata"
                        self.outputs_done.set()
                        return
                    index = self.output_count
                    location = self.frame_sink.write(index, bytes(message))
                    if self.should_log_frame(index + 1):
                        self.log(
                            {
                                "direction": "server_to_client",
                                "received_at": now,
                                "binary_bytes": len(message),
                                **location,
                                "output_meta": self.pending_output,
                            }
                        )
                    self.pending_output = None
                    self.output_count += 1
                    if self.output_count >= self.expected_frames:
                        self.outputs_done.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f"JoyAI receiver failed: {type(exc).__name__}: {exc}"
            self.log(
                {
                    "direction": "client_internal",
                    "received_at": time.time(),
                    "error": self.error,
                }
            )
            self.outputs_done.set()
            self.finalized.set()
            async with self.ack_condition:
                self.ack_condition.notify_all()

    async def wait_ack(self, minimum: int, timeout: float) -> None:
        async with self.ack_condition:
            await asyncio.wait_for(
                self.ack_condition.wait_for(lambda: self.acked_frames >= minimum or self.error is not None),
                timeout=timeout,
            )
        if self.error:
            raise RuntimeError(self.error)


def build_start_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build the exact session contract sent to the pinned JoyAI server."""

    return {
        "type": "start",
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "num_inference_steps": args.num_inference_steps,
        "use_pe": False,
        "gate_enabled": False,
        "seed": args.seed,
        "kv_reset_frames": 0,
        "output_quality": args.output_quality,
        "no_person_blank": False,
        "require_face": False,
        "person_count_reedit": False,
        "freeze_kv_on_static": False,
        "profile_timings": args.profile_timings,
        "uplink_codec": "jpeg",
        "downlink_codec": "jpeg",
    }


async def run_protocol(
    args: argparse.Namespace, frames: Iterable[bytes], output: Path
) -> dict[str, Any]:
    events = output / "protocol-events.jsonl"
    events.write_text("", encoding="utf-8")
    frame_sink = OutputFrameSink(output, args.frame_storage)
    state = ProtocolRun(
        frame_sink,
        events,
        args.expected_frames,
        args.frame_log_every,
    )
    websocket = None
    receiver = None
    started_wall = time.perf_counter()
    first_send_wall: float | None = None
    input_frames_sent = 0
    input_bytes_sent = 0
    frame_iterator = iter(frames)
    try:
        websocket = await _connect(args.server_url)
        receiver = asyncio.create_task(state.receiver(websocket))
        await asyncio.wait_for(state.session_granted.wait(), timeout=60)
        payload = build_start_payload(args)
        reference = _reference_data_url(args.reference_image)
        if reference:
            payload["ref_image"] = reference
        state.log({"direction": "client_to_server", "sent_at": time.time(), "message": {**payload, "ref_image": "<base64>"} if reference else payload})
        await websocket.send(json.dumps(payload))
        await asyncio.wait_for(state.started.wait(), timeout=600)

        first_send_wall = time.perf_counter()
        for index in range(1, args.expected_frames + 1):
            try:
                encoded = next(frame_iterator)
            except StopIteration as exc:
                raise ValueError(
                    f"decoded {index - 1} input frames; expected {args.expected_frames}"
                ) from exc
            meta = {
                "type": "frame_meta",
                "seq": index,
                "t_capture_ms": (index - 1) * 1000.0 / args.fps,
            }
            if state.should_log_frame(index):
                state.log(
                    {
                        "direction": "client_to_server",
                        "sent_at": time.time(),
                        "message": meta,
                    }
                )
            await websocket.send(json.dumps(meta))
            await websocket.send(encoded)
            input_frames_sent += 1
            input_bytes_sent += len(encoded)
            if index == args.expected_frames or index % args.ack_window == 0:
                await state.wait_ack(index, timeout=600)
        try:
            next(frame_iterator)
        except StopIteration:
            pass
        else:
            raise ValueError(
                f"input contains more than the expected {args.expected_frames} frames"
            )
        await asyncio.wait_for(state.outputs_done.wait(), timeout=args.timeout_seconds)
        if state.error:
            raise RuntimeError(state.error)
        if state.output_count != args.expected_frames:
            raise RuntimeError(
                f"JoyAI returned {state.output_count} frames; expected {args.expected_frames}"
            )
        inference_done_wall = time.perf_counter()
        finalize = {"type": "finalize_recording"}
        state.log({"direction": "client_to_server", "sent_at": time.time(), "message": finalize})
        await websocket.send(json.dumps(finalize))
        await asyncio.wait_for(state.finalized.wait(), timeout=300)
        if state.error:
            raise RuntimeError(state.error)
        return {
            "session_wall_seconds": time.perf_counter() - started_wall,
            "edit_wall_seconds": inference_done_wall - (first_send_wall or started_wall),
            "output_frames": state.output_count,
            "effective_output_fps": state.output_count
            / max(inference_done_wall - (first_send_wall or started_wall), 1e-9),
            "input_frames_sent": input_frames_sent,
            "input_bytes_sent": input_bytes_sent,
            "ack_window": args.ack_window,
            "frame_log_every": args.frame_log_every,
            "frame_storage": state.frame_sink.to_manifest(),
            "server_chunk_profiles": state.chunk_profiles,
            "protocol_events": str(events),
        }
    finally:
        close_iterator = getattr(frame_iterator, "close", None)
        if close_iterator is not None:
            close_iterator()
        if websocket is not None:
            await websocket.close()
        if receiver is not None:
            if not receiver.done():
                receiver.cancel()
            try:
                await receiver
            except asyncio.CancelledError:
                pass
        state.close()


def mux_outputs(args: argparse.Namespace, output: Path) -> tuple[dict[str, Any], list[list[str]]]:
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)
    if args.frame_storage == "files":
        input_args = ["-framerate", str(args.fps), "-i", str(output / "frames-jpeg/%06d.jpg")]
    else:
        input_args = [
            "-f",
            "image2pipe",
            "-framerate",
            str(args.fps),
            "-vcodec",
            "mjpeg",
            "-i",
            str(output / "frames.mjpeg"),
        ]
    lossless = output / "joyai-proposal-lossless.mkv"
    review = output / "joyai-proposal-review.mp4"
    commands_by_artifact = {
        "lossless": [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            *input_args,
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "bgr0",
            str(lossless),
        ],
        "review": [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            *input_args,
            "-c:v",
            "libx264",
            "-crf",
            "8",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(review),
        ],
    }
    selected = (
        ("lossless", "review")
        if args.output_artifacts == "both"
        else (args.output_artifacts,)
    )
    commands = [commands_by_artifact[name] for name in selected]
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    for name, command in zip(selected, commands):
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        (logs / f"mux-{name}.log").write_text(
            "$ " + shlex.join(command) + "\n" + completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode:
            raise RuntimeError(f"mux command failed; inspect {logs / f'mux-{name}.log'}")
    paths = {"lossless": lossless, "review": review}
    return {
        name: {"path": str(paths[name]), "sha256": sha256_file(paths[name])}
        for name in selected
    }, commands


def cleanup_protocol_frames(args: argparse.Namespace, output: Path) -> None:
    """Remove only the known successful staging artifacts."""

    if args.keep_protocol_frames:
        return
    if args.frame_storage == "mjpeg":
        (output / "frames.mjpeg").unlink()
        return
    frame_dir = output / "frames-jpeg"
    for index in range(args.expected_frames):
        (frame_dir / f"{index:06d}.jpg").unlink()
    frame_dir.rmdir()


def resolve_throughput_options(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve mode defaults while preserving explicit zero values for validation."""

    if args.input_loading is None:
        args.input_loading = "stream" if args.throughput_mode else "memory"
    if args.ack_window is None:
        args.ack_window = 8 if args.throughput_mode else 1
    if args.frame_log_every is None:
        args.frame_log_every = 240 if args.throughput_mode else 1
    if args.frame_storage is None:
        args.frame_storage = "mjpeg" if args.throughput_mode else "files"
    if args.keep_protocol_frames is None:
        args.keep_protocol_frames = not args.throughput_mode
    if args.output_artifacts is None:
        args.output_artifacts = "review" if args.throughput_mode else "both"
    return args


def _git_state() -> dict[str, Any]:
    state = {}
    for label, command in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
    }.items():
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        state[label] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    return state


def _package_state(output: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    path = output / "packages.txt"
    path.write_text(completed.stdout, encoding="utf-8")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "pip_freeze_path": str(path),
        "pip_freeze_sha256": sha256_file(path),
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    args = resolve_throughput_options(_parser().parse_args())
    prompt_file = None
    if args.prompt_file is not None:
        prompt_file = args.prompt_file.expanduser().resolve()
        if not prompt_file.is_file():
            raise FileNotFoundError(prompt_file)
        args.prompt = prompt_file.read_text(encoding="utf-8").strip()
        if not args.prompt:
            raise ValueError("prompt file must contain a non-empty editing instruction")
    if args.prompt is None:
        args.prompt = (
            flower_full_stream_prompt()
            if args.prompt_mode == "full_flower_stream"
            else flower_repair_prompt()
        )
    held_tool_contract = (
        DEFAULT_SCISSORS_CONTRACT.to_manifest()
        if args.prompt_mode == "full_flower_stream"
        else None
    )
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"JoyAI client experiment already exists: {output}")
    if args.expected_frames < 1 or (args.expected_frames - 1) % 8:
        raise ValueError("expected frames must satisfy frame_count = 1 + 8n")
    if args.fps <= 0 or args.timeout_seconds <= 0 or args.ack_window <= 0:
        raise ValueError("fps, timeout, and ack window must be positive")
    if args.frame_log_every < 0:
        raise ValueError("frame log interval must be non-negative")
    if args.input_video is not None:
        input_video = args.input_video.expanduser().resolve()
        if not input_video.is_file():
            raise FileNotFoundError(input_video)
        frames: Iterable[bytes]
        if args.input_loading == "stream":
            frames = iter_input_frames(input_video, width=args.width, height=args.height)
            input_kind = "video_streamed_then_jpeg95_444"
            decoded_frames = args.expected_frames
            uplink_bytes = None
        else:
            decoded = decode_input_frames(input_video, width=args.width, height=args.height)
            frames = decoded
            input_kind = "video_decoded_then_jpeg95_444"
            decoded_frames = len(decoded)
            uplink_bytes = sum(len(frame) for frame in decoded)
        input_record = {
            "kind": input_kind,
            "path": str(input_video),
            "sha256": sha256_file(input_video),
            "frames": decoded_frames,
            "uplink_bytes": uplink_bytes,
        }
    else:
        frame_dir = args.input_frame_dir.expanduser().resolve()
        if not frame_dir.is_dir():
            raise FileNotFoundError(frame_dir)
        frames, input_record = load_input_frame_dir(
            frame_dir,
            width=args.width,
            height=args.height,
            expected_frames=args.expected_frames,
        )
    if isinstance(frames, list) and len(frames) != args.expected_frames:
        raise ValueError(f"decoded {len(frames)} input frames; expected {args.expected_frames}")
    output.mkdir(parents=True)

    status = "PARTIAL"
    error: str | None = None
    protocol: dict[str, Any] = {}
    muxed: dict[str, Any] = {}
    commands: list[list[str]] = []
    try:
        protocol = asyncio.run(run_protocol(args, frames, output))
        input_record["frames"] = protocol["input_frames_sent"]
        input_record["uplink_bytes"] = protocol["input_bytes_sent"]
        muxed, commands = mux_outputs(args, output)
        cleanup_protocol_frames(args, output)
        protocol["frame_storage"]["retained"] = args.keep_protocol_frames
        status = "WORKING"
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        manifest = {
            "schema_version": "1.0.0",
            "status": status,
            "stage": "joyai_proposal_generation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "command": [sys.executable, *sys.argv],
            "command_shell": shlex.join([sys.executable, *sys.argv]),
            "git": _git_state(),
            "packages": _package_state(output),
            "input": input_record,
            "server_url": args.server_url,
            "model": {
                "id": JOYAI_MODEL_ID,
                "weights_revision": JOYAI_MODEL_REVISION,
                "repository_revision": JOYAI_REPOSITORY_REVISION,
            },
            "config": {
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "expected_frames": args.expected_frames,
                "seed": args.seed,
                "num_inference_steps": args.num_inference_steps,
                "output_quality": args.output_quality,
                "profile_timings": args.profile_timings,
                "throughput_mode": args.throughput_mode,
                "input_loading": args.input_loading,
                "ack_window": args.ack_window,
                "frame_log_every": args.frame_log_every,
                "frame_storage": args.frame_storage,
                "keep_protocol_frames": args.keep_protocol_frames,
                "output_artifacts": args.output_artifacts,
                "prompt": args.prompt,
                "prompt_file": str(prompt_file) if prompt_file is not None else None,
                "prompt_file_sha256": sha256_file(prompt_file) if prompt_file is not None else None,
                "reference_image": str(args.reference_image.expanduser().resolve()) if args.reference_image else None,
                "held_tool_contract": held_tool_contract,
            },
            "protocol": protocol,
            "outputs": muxed,
            "mux_commands": commands,
            "error": error,
            "model_authority": "proposal_only",
            "physical_evidence": False,
            "promotion_status": "NOT_EVALUATED",
        }
        write_json(output / "manifest.json", manifest)
    print(json.dumps({"experiment": str(output), "status": status, "metrics": protocol}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
