#!/usr/bin/env python3
"""Serve the reviewed MiniMax-H3 action-control demo.

The HTTP service binds to loopback by default.  Static preset results remain
usable without a GPU job. Unmatched actions are rejected until an explicit
Hand2Dex-2 hand/bowl control video has been compiled and verified.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.learning.experience import ExperienceRecord, append_experience  # noqa: E402


class ActionDemo:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(__file__).resolve().parents[1]
        self.demo = args.demo_dir.expanduser().resolve()
        self.jobs_root = args.jobs_root.expanduser().resolve()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-action")

    def submit(self, instruction: str, timeline: str) -> dict[str, str]:
        instruction = " ".join(instruction.split())
        timeline = " ".join(timeline.split())
        if not 8 <= len(instruction) <= 700:
            raise ValueError("instruction must contain 8-700 characters")
        if len(timeline) > 1600:
            raise ValueError("timeline must contain at most 1600 characters")
        raise ValueError(
            "unmatched custom actions are not queued: compile and verify an explicit "
            "Hand2Dex-2 hand/bowl control video before H3 inference"
        )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is not None:
                return {key: value for key, value in job.items() if key not in {"instruction", "timeline", "traceback"}}
        path = self.jobs_root / job_id / "job.json"
        if not path.is_file():
            return None
        raw = json.loads(path.read_text())
        return {key: value for key, value in raw.items() if key not in {"instruction", "timeline", "traceback"}}

    def _update(self, job_id: str, **values: Any) -> None:
        with self.lock:
            self.jobs[job_id].update(values)
            snapshot = dict(self.jobs[job_id])
        self._persist(snapshot)

    def _persist(self, job: dict[str, Any]) -> None:
        path = self.jobs_root / job["job_id"] / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    def _run(
        self,
        command: list[str],
        *,
        allowed: tuple[int, ...] = (0,),
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            command, cwd=cwd, check=False, capture_output=True, text=True
        )
        if completed.returncode not in allowed:
            output = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"command failed with exit {completed.returncode}: {output[-1200:]}")
        return completed

    def _run_job(self, job_id: str) -> None:
        job = self.jobs[job_id]
        local_request_dir = self.jobs_root / job_id
        local_manifest = local_request_dir / "action-manifest.json"
        local_experiment = self.args.local_experiment_root.expanduser().resolve() / f"live-{job_id}"
        remote_experiment = f"{self.args.remote_root}/outputs/acwm-bowl-h3/live-{job_id}"
        remote_manifest = f"{self.args.remote_root}/demo/jobs/{job_id}.json"
        manifest = {
            "schema_version": "1.0.0",
            "actions": [
                {
                    "label": "live-action",
                    "instruction": job["instruction"],
                    "timeline": job["timeline"],
                }
            ],
        }
        local_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        try:
            self._update(job_id, status="running", detail="正在同步动作条件并执行 MiniMax-H3 真实推理。")
            self._run(
                [
                    "rsync",
                    "-av",
                    "--relative",
                    "phiagent/rendering/minimax_h3.py",
                    "scripts/run_minimax_h3_action_variants.py",
                    f"{self.args.ssh_host}:{self.args.remote_root}/",
                ],
                cwd=self.root,
            )
            self._run(
                ["ssh", self.args.ssh_host, "mkdir", "-p", f"{self.args.remote_root}/demo/jobs"]
            )
            self._run(["rsync", "-av", str(local_manifest), f"{self.args.ssh_host}:{remote_manifest}"])
            remote_command = [
                f"{self.args.remote_root}/.venv-h3/bin/python",
                f"{self.args.remote_root}/scripts/run_minimax_h3_action_variants.py",
                "--source-video",
                self.args.remote_source,
                "--robot-reference",
                self.args.remote_robot_reference,
                "--action-manifest",
                remote_manifest,
                "--diffsynth-repo",
                f"{self.args.remote_root}/external/DiffSynth-Studio",
                "--model-base-path",
                f"{self.args.remote_root}/checkpoints/h3-models",
                "--experiment-dir",
                remote_experiment,
                "--gpu",
                str(self.args.gpu),
                "--scene-reference-mode",
                "anchor_image",
                "--scene-domain",
                "tabletop_bowl",
            ]
            self._run(
                [
                    "ssh",
                    self.args.ssh_host,
                    f"cd {shlex.quote(self.args.remote_root)} && {shlex.join(remote_command)}",
                ]
            )
            self._update(job_id, detail="H3 推理完成；正在同步并检查视频可解码性。")
            local_experiment.mkdir(parents=True, exist_ok=False)
            self._run(
                [
                    "rsync",
                    "-av",
                    f"{self.args.ssh_host}:{remote_experiment}/",
                    f"{local_experiment}/",
                ]
            )
            final = local_experiment / "variants/live-action/raw-h3-nf4.mp4"
            self._run(
                [
                    str(self.args.ffprobe.expanduser().resolve()),
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate,nb_frames",
                    "-of",
                    "json",
                    str(final),
                ]
            )
            public_dir = self.demo / "showcase/live"
            public_dir.mkdir(parents=True, exist_ok=True)
            public_video = public_dir / f"{job_id}.mp4"
            shutil.copy2(final, public_video)
            self._update(
                job_id,
                status="completed",
                detail="真实场景 H3 text-only 实验已完成；未编译显式轨迹，动作语义仍需人工复核。",
                completed_at=datetime.now(timezone.utc).isoformat(),
                video_url=f"showcase/live/{job_id}.mp4",
                honest_status="PARTIAL",
                run_dir=str(local_experiment),
            )
            append_experience(
                self.root / "experiences/ledger.jsonl",
                ExperienceRecord(
                    record_id=f"2026-08-10.live-h3-action-{job_id.lower()}",
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    status="PARTIAL",
                    scope="live language-conditioned MiniMax-H3 real-scene video job",
                    summary="A user-entered text-only action instruction produced a decodable MiniMax-H3 video in the real Hand2Dex-2 tabletop scene.",
                    evidence=(
                        f"H3 experiment metadata: {local_experiment / 'metadata.json'}",
                        f"Public result: {public_video}",
                    ),
                    lessons=("Custom action jobs must expose semantic human review separately from pixel-lock metrics.",),
                    limitations=("The unmatched instruction used language conditioning without an explicit action-control trajectory and has not been semantically reviewed.",),
                    next_actions=("Compile a new explicit bowl/hand trajectory and run the matched terminal-state evaluator before treating it as action-adherent.",),
                    run_dir=str(local_experiment),
                    tags=("minimax-h3", "live-demo", "action-condition", "real-scene"),
                ),
            )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                detail=f"H3 作业未完成：{type(exc).__name__}: {str(exc)[:280]}",
                completed_at=datetime.now(timezone.utc).isoformat(),
                traceback=traceback.format_exc(),
            )
            try:
                append_experience(
                    self.root / "experiences/ledger.jsonl",
                    ExperienceRecord(
                        record_id=f"2026-08-10.live-h3-action-{job_id.lower()}-failed",
                        recorded_at=datetime.now(timezone.utc).isoformat(),
                        status="BLOCKED",
                        scope="live language-conditioned MiniMax-H3 real-scene video job",
                        summary="A user-entered H3 action job did not complete and no result was published.",
                        evidence=(f"Job record: {local_request_dir / 'job.json'}",),
                        lessons=("GPU and transport failures must remain explicit rather than falling back to a simulated video.",),
                        limitations=(str(exc)[:500],),
                        next_actions=("Inspect the persisted job record and retry in a fresh experiment directory after the blocking condition is resolved.",),
                        run_dir=str(local_experiment) if local_experiment.exists() else None,
                        tags=("minimax-h3", "live-demo", "failed"),
                    ),
                )
            except Exception:
                pass


def _handler_factory(app: ActionDemo) -> type[SimpleHTTPRequestHandler]:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, directory=str(app.demo), **kwargs)

        def _json(self, status: HTTPStatus, payload: object) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/jobs":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 20_000:
                    raise ValueError("invalid request size")
                raw = json.loads(self.rfile.read(length))
                if not isinstance(raw, dict):
                    raise ValueError("JSON body must be an object")
                result = app.submit(str(raw.get("instruction", "")), str(raw.get("timeline", "")))
                self._json(HTTPStatus.ACCEPTED, result)
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def do_GET(self) -> None:  # noqa: N802
            match = re.fullmatch(r"/api/jobs/([a-zA-Z0-9_-]+)", self.path)
            if match:
                job = app.get(match.group(1))
                self._json(HTTPStatus.OK if job else HTTPStatus.NOT_FOUND, job or {"error": "unknown job"})
                return
            super().do_GET()

    return Handler


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    bowl_control = root / "outputs/acwm-bowl-action-controls/20260810T121000Z-hand2dex2-v1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--demo-dir", type=Path, default=root / "demo")
    parser.add_argument("--jobs-root", type=Path, default=root / "outputs/minimax-h3-action-demo-jobs")
    parser.add_argument("--local-experiment-root", type=Path, default=root / "outputs/acwm-bowl-h3")
    parser.add_argument("--ssh-host", default="a800-1")
    parser.add_argument("--remote-root", default="/data0/jiangyuhua/PhiAgent-0")
    parser.add_argument("--remote-source", default="/data0/jiangyuhua/PhiAgent-0/outputs/acwm-bowl-action-controls/20260810T121000Z-hand2dex2-v1/input/real-scene-source-124f.mp4")
    parser.add_argument(
        "--remote-robot-reference",
        default=(
            "/data0/jiangyuhua/PhiAgent-0/outputs/acwm-bowl-action-controls/"
            "20260810T121000Z-hand2dex2-v1/input/robot-reference.png"
        ),
    )
    parser.add_argument("--anchor-mask", type=Path, default=bowl_control / "input/bowl-mask.png")
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/opt/homebrew/bin/ffprobe"))
    parser.add_argument("--gpu", type=int, default=2)
    return parser


def main() -> int:
    args = _parser().parse_args()
    app = ActionDemo(args)
    server = ThreadingHTTPServer((args.host, args.port), _handler_factory(app))
    print(f"MiniMax-H3 action demo listening on http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.executor.shutdown(wait=False, cancel_futures=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
