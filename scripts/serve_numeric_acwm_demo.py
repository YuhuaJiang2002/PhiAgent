#!/usr/bin/env python3
"""Serve the demo and execute exact 14-D BWM action jobs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.adapters import BWMConfig, BWMRenderer  # noqa: E402
from phiagent.acwm.numeric import NumericActionStatistics  # noqa: E402
from phiagent.acwm.robotwin import BWM_EEF_CHANNELS  # noqa: E402
from phiagent.acwm.schema import ACWMActionCondition  # noqa: E402
from phiagent.acwm.worldarena import (  # noqa: E402
    WORLD_ARENA_EEF_QUATERNION_CHANNELS,
)
from phiagent.agent.numeric_action import (  # noqa: E402
    NumericActionJobManager,
    NumericActionScene,
    NumericActionVideoAgent,
)

_JOB_ROUTE = re.compile(r"/api/numeric-jobs/([0-9a-f]{32})")


def parse_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Parse one HTTP byte range, inclusive at both ends."""

    if header is None:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if match is None or (not match.group(1) and not match.group(2)):
        raise ValueError("unsupported byte range")
    start_text, end_text = match.groups()
    if not start_text:
        suffix = int(end_text)
        if suffix <= 0:
            raise ValueError("byte-range suffix must be positive")
        start = max(0, size - suffix)
        end = size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("byte range lies outside the video")
    return start, min(end, size - 1)


def _handler_factory(manager: NumericActionJobManager, demo_dir: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(demo_dir), **kwargs)

        def _json(self, status: HTTPStatus, payload: object, *, send_body: bool = True) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if send_body:
                self.wfile.write(encoded)

        def _video(self, job_id: str, *, send_body: bool) -> None:
            job = manager.get(job_id)
            if job is None:
                self._json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "unknown numeric action job"},
                    send_body=send_body,
                )
                return
            path = manager.video_path(job_id)
            if path is None:
                self._json(
                    HTTPStatus.CONFLICT,
                    {"error": f"video is unavailable while job status is {job['status']}"},
                    send_body=send_body,
                )
                return
            size = path.stat().st_size
            try:
                selected = parse_byte_range(self.headers.get("Range"), size)
            except ValueError as exc:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                if send_body:
                    self.wfile.write(json.dumps({"error": str(exc)}).encode())
                return
            start, end = selected or (0, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT if selected else HTTPStatus.OK
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            if selected:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not send_body:
                return
            remaining = end - start + 1
            with path.open("rb") as handle:
                handle.seek(start)
                while remaining:
                    block = handle.read(min(1024 * 1024, remaining))
                    if not block:
                        raise RuntimeError(f"video ended before byte {end}: {path}")
                    self.wfile.write(block)
                    remaining -= len(block)

        def do_POST(self) -> None:  # noqa: N802
            route = urlsplit(self.path).path
            if route != "/api/numeric-jobs":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                content_type = self.headers.get("Content-Type", "")
                if not content_type.lower().startswith("application/json"):
                    raise ValueError("Content-Type must be application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 256_000:
                    raise ValueError("numeric action request must contain 1-256000 bytes")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("numeric action request must be a JSON object")
                result = manager.submit(payload)
                self._json(HTTPStatus.ACCEPTED, result)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except RuntimeError as exc:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})

        def do_GET(self) -> None:  # noqa: N802
            route = urlsplit(self.path).path
            if route == "/api/numeric-capabilities":
                self._json(HTTPStatus.OK, manager.capabilities())
                return
            match = _JOB_ROUTE.fullmatch(route)
            if match:
                job = manager.get(match.group(1))
                self._json(
                    HTTPStatus.OK if job else HTTPStatus.NOT_FOUND,
                    job or {"error": "unknown numeric action job"},
                )
                return
            match = re.fullmatch(r"/api/numeric-jobs/([0-9a-f]{32})/action", route)
            if match:
                payload = manager.action_payload(match.group(1))
                self._json(
                    HTTPStatus.OK if payload else HTTPStatus.NOT_FOUND,
                    payload or {"error": "unknown numeric action job"},
                )
                return
            match = re.fullmatch(r"/api/numeric-jobs/([0-9a-f]{32})/video", route)
            if match:
                self._video(match.group(1), send_body=True)
                return
            super().do_GET()

        def do_HEAD(self) -> None:  # noqa: N802
            route = urlsplit(self.path).path
            match = re.fullmatch(r"/api/numeric-jobs/([0-9a-f]{32})/video", route)
            if match:
                self._video(match.group(1), send_body=False)
                return
            super().do_HEAD()

    return Handler


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--demo-dir", type=Path, default=root / "demo")
    parser.add_argument(
        "--jobs-root", type=Path, default=root / "outputs/numeric-action-demo-jobs"
    )
    parser.add_argument(
        "--experiment-root", type=Path, default=root / "outputs/numeric-action-video"
    )
    parser.add_argument("--ledger", type=Path, default=root / "experiences/ledger.jsonl")
    parser.add_argument("--first-frame", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--coordinate-frame", required=True)
    parser.add_argument("--default-condition", type=Path)
    parser.add_argument("--initial-state-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--action-profile",
        choices=("euler-gripper", "quaternion"),
        default="euler-gripper",
    )
    parser.add_argument("--action-sample-hz", type=float, default=24.0)
    parser.add_argument("--output-fps", type=int, default=24)
    parser.add_argument(
        "--bwm-repo", type=Path, default=root / "external/boundless-world-model"
    )
    parser.add_argument(
        "--bwm-base-model", type=Path, default=root / "checkpoints/Wan2.2-TI2V-5B"
    )
    parser.add_argument(
        "--bwm-checkpoint",
        type=Path,
        default=root / "checkpoints/BWM/step-12000.safetensors",
    )
    parser.add_argument("--bwm-action-stats", type=Path, required=True)
    parser.add_argument("--bwm-config", type=Path)
    parser.add_argument("--bwm-python", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=32 * 1024)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--maximum-queued-jobs", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    renderer = BWMRenderer(
        BWMConfig(
            repository=args.bwm_repo,
            base_model_dir=args.bwm_base_model,
            checkpoint_path=args.bwm_checkpoint,
            action_stats=args.bwm_action_stats,
            python_executable=args.bwm_python,
            config_path=args.bwm_config,
            gpu_index=args.gpu,
            minimum_free_gpu_mib=args.minimum_free_gpu_mib,
            output_fps=args.output_fps,
        ),
        project_root=project_root,
    )
    if args.preflight_only:
        print(json.dumps(renderer.preflight(), indent=2, sort_keys=True))
        return 0

    default_condition = (
        ACWMActionCondition.from_json(args.default_condition)
        if args.default_condition is not None
        else None
    )
    action_channels = (
        default_condition.channels
        if default_condition is not None
        else (
            WORLD_ARENA_EEF_QUATERNION_CHANNELS
            if args.action_profile == "quaternion"
            else BWM_EEF_CHANNELS
        )
    )
    action_sample_hz = (
        default_condition.fps
        if default_condition is not None
        else args.action_sample_hz
    )
    manager = NumericActionJobManager(
        NumericActionVideoAgent(renderer, project_root=project_root),
        scene=NumericActionScene(
            first_frame=args.first_frame,
            source_video=args.source_video,
            coordinate_frame=args.coordinate_frame,
            default_condition=default_condition,
            action_channels=action_channels,
            action_sample_hz=action_sample_hz,
            action_stats=NumericActionStatistics.from_json(args.bwm_action_stats),
            initial_state_tolerance=args.initial_state_tolerance,
        ),
        jobs_root=args.jobs_root,
        experiment_root=args.experiment_root,
        ledger_path=args.ledger,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        maximum_queued_jobs=args.maximum_queued_jobs,
    )
    demo_dir = args.demo_dir.expanduser().resolve()
    if not demo_dir.is_dir():
        raise ValueError(f"demo directory does not exist: {demo_dir}")
    server = ThreadingHTTPServer(
        (args.host, args.port),
        _handler_factory(manager, demo_dir),
    )
    print(
        f"Numeric AC-WM demo listening on http://{args.host}:{args.port}/ "
        f"for {args.coordinate_frame}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.close(wait=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
