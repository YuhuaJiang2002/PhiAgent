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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    parser.add_argument("--prompt")
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
        "--profile-timings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Synchronize CUDA stage timers (disabled by default for production throughput).",
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    return parser


def decode_input_frames(video: Path, *, width: int, height: int) -> list[bytes]:
    """Decode exact pixels and JPEG-encode frames for the official uplink."""

    try:
        import av
    except ImportError:  # pragma: no cover - local preparation fallback
        av = None
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional runtime only
        raise RuntimeError("JoyAI client requires Pillow") from exc
    frames: list[bytes] = []
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
                frames.append(encoded.getvalue())
        return frames

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
            frames.append(encoded.getvalue())
    finally:
        capture.release()
    return frames


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


class ProtocolRun:
    def __init__(self, output_frames: Path, events_path: Path, expected_frames: int) -> None:
        self.output_frames = output_frames
        self.events_path = events_path
        self.expected_frames = expected_frames
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

    def log(self, row: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    async def receiver(self, websocket: Any) -> None:
        try:
            async for message in websocket:
                now = time.time()
                if isinstance(message, str):
                    payload = json.loads(message)
                    kind = str(payload.get("type", "unknown"))
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
                    elif kind == "frame_ack":
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
                    frame_path = self.output_frames / f"{index:06d}.jpg"
                    frame_path.write_bytes(bytes(message))
                    self.log(
                        {
                            "direction": "server_to_client",
                            "received_at": now,
                            "binary_bytes": len(message),
                            "saved_path": str(frame_path),
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


async def run_protocol(args: argparse.Namespace, frames: list[bytes], output: Path) -> dict[str, Any]:
    frame_dir = output / "frames-jpeg"
    frame_dir.mkdir()
    events = output / "protocol-events.jsonl"
    events.write_text("", encoding="utf-8")
    state = ProtocolRun(frame_dir, events, args.expected_frames)
    websocket = await _connect(args.server_url)
    receiver = asyncio.create_task(state.receiver(websocket))
    started_wall = time.perf_counter()
    first_send_wall: float | None = None
    try:
        await asyncio.wait_for(state.session_granted.wait(), timeout=60)
        payload: dict[str, Any] = {
            "type": "start",
            "prompt": args.prompt,
            "width": args.width,
            "height": args.height,
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
        reference = _reference_data_url(args.reference_image)
        if reference:
            payload["ref_image"] = reference
        state.log({"direction": "client_to_server", "sent_at": time.time(), "message": {**payload, "ref_image": "<base64>"} if reference else payload})
        await websocket.send(json.dumps(payload))
        await asyncio.wait_for(state.started.wait(), timeout=600)

        first_send_wall = time.perf_counter()
        for index, encoded in enumerate(frames, start=1):
            meta = {
                "type": "frame_meta",
                "seq": index,
                "t_capture_ms": (index - 1) * 1000.0 / args.fps,
            }
            state.log({"direction": "client_to_server", "sent_at": time.time(), "message": meta})
            await websocket.send(json.dumps(meta))
            await websocket.send(encoded)
            await state.wait_ack(index, timeout=600)
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
            "server_chunk_profiles": state.chunk_profiles,
            "protocol_events": str(events),
        }
    finally:
        await websocket.close()
        if not receiver.done():
            receiver.cancel()
        try:
            await receiver
        except asyncio.CancelledError:
            pass


def mux_outputs(args: argparse.Namespace, output: Path) -> tuple[dict[str, Any], list[list[str]]]:
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)
    frame_pattern = output / "frames-jpeg/%06d.jpg"
    lossless = output / "joyai-proposal-lossless.mkv"
    review = output / "joyai-proposal-review.mp4"
    commands = [
        [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-framerate", str(args.fps),
            "-i", str(frame_pattern), "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", str(lossless),
        ],
        [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-framerate", str(args.fps),
            "-i", str(frame_pattern), "-c:v", "libx264", "-crf", "8", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(review),
        ],
    ]
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    for index, command in enumerate(commands):
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        (logs / f"mux-{index:02d}.log").write_text(
            "$ " + shlex.join(command) + "\n" + completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode:
            raise RuntimeError(f"mux command failed; inspect {logs / f'mux-{index:02d}.log'}")
    return {
        "lossless": {"path": str(lossless), "sha256": sha256_file(lossless)},
        "review": {"path": str(review), "sha256": sha256_file(review)},
    }, commands


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
    args = _parser().parse_args()
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
    if args.fps <= 0 or args.timeout_seconds <= 0:
        raise ValueError("fps and timeout must be positive")
    if args.input_video is not None:
        input_video = args.input_video.expanduser().resolve()
        if not input_video.is_file():
            raise FileNotFoundError(input_video)
        frames = decode_input_frames(input_video, width=args.width, height=args.height)
        input_record = {
            "kind": "video_decoded_then_jpeg95_444",
            "path": str(input_video),
            "sha256": sha256_file(input_video),
            "frames": len(frames),
            "uplink_bytes": sum(len(frame) for frame in frames),
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
    if len(frames) != args.expected_frames:
        raise ValueError(f"decoded {len(frames)} input frames; expected {args.expected_frames}")
    output.mkdir(parents=True)

    status = "PARTIAL"
    error: str | None = None
    protocol: dict[str, Any] = {}
    muxed: dict[str, Any] = {}
    commands: list[list[str]] = []
    try:
        protocol = asyncio.run(run_protocol(args, frames, output))
        muxed, commands = mux_outputs(args, output)
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
                "prompt": args.prompt,
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
