"""Command-line entry point for inspecting and running PhiAgent workflows."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .checkpoint import JsonFileCheckpointer
from .core import Command
from .registry import create_workflow, workflow_names


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phiagent-workflow",
        description="Inspect and run persistent PhiAgent research workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list built-in workflow recipes")

    describe = subparsers.add_parser("describe", help="print a graph contract")
    describe.add_argument("workflow", choices=workflow_names())
    describe.add_argument("--format", choices=("json", "mermaid"), default="json")

    run = subparsers.add_parser("run", help="start a new persistent workflow thread")
    run.add_argument("workflow", choices=workflow_names())
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--thread-id", required=True)

    resume = subparsers.add_parser("resume", help="resume an interrupted workflow thread")
    resume.add_argument("--output-dir", type=Path, required=True)
    value = resume.add_mutually_exclusive_group(required=True)
    value.add_argument("--resume-json", help="JSON value returned by the interrupt")
    value.add_argument("--resume-file", type=Path, help="JSON file returned by the interrupt")

    retry = subparsers.add_parser("retry", help="retry the node recorded by an ERROR checkpoint")
    retry.add_argument("--output-dir", type=Path, required=True)

    status = subparsers.add_parser("status", help="show the latest checkpoint")
    status.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "list":
        print(json.dumps({"workflows": list(workflow_names())}, indent=2))
        return 0
    if args.command == "describe":
        workflow = create_workflow(args.workflow)
        if args.format == "mermaid":
            print(workflow.mermaid(), end="")
        else:
            print(json.dumps(workflow.describe(), indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        return _run_new(args.workflow, args.config, args.output_dir, args.thread_id)
    if args.command == "resume":
        value = (
            json.loads(args.resume_json)
            if args.resume_json is not None
            else _read_json_value(args.resume_file)
        )
        return _continue(args.output_dir, Command(resume=value))
    if args.command == "retry":
        return _continue(args.output_dir, Command(retry=True))
    if args.command == "status":
        metadata = _load_run_metadata(args.output_dir)
        checkpointer = JsonFileCheckpointer(args.output_dir / "checkpoints")
        latest = checkpointer.load_latest(str(metadata["thread_id"]))
        print(json.dumps(latest, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    raise AssertionError(args.command)


def _run_new(workflow_name: str, config_path: Path, output_dir: Path, thread_id: str) -> int:
    config_path = config_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"experiment output directory already exists: {output_dir}")
    raw = _read_json_object(config_path)
    workspace_raw = Path(str(raw.get("workspace_root", Path.cwd())))
    workspace_root = (
        workspace_raw.resolve()
        if workspace_raw.is_absolute()
        else (Path.cwd() / workspace_raw).resolve()
    )
    state = {**raw, "workspace_root": str(workspace_root), "workflow": workflow_name}
    output_dir.mkdir(parents=True)
    (output_dir / "logs").mkdir()
    (output_dir / "artifacts").mkdir()
    _write_json(output_dir / "config.json", state)
    metadata = {
        "schema_version": "1.0.0",
        "workflow": workflow_name,
        "thread_id": thread_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config_path),
        "output_dir": str(output_dir),
        "seed": state.get("seed"),
        "command": list(sys.argv),
    }
    _write_json(output_dir / "run.json", metadata)
    _capture_provenance(output_dir, workspace_root)
    checkpointer = JsonFileCheckpointer(output_dir / "checkpoints")
    workflow = create_workflow(workflow_name, checkpointer=checkpointer)
    return _execute(workflow, state, output_dir, thread_id)


def _continue(output_dir: Path, command: Command) -> int:
    output_dir = output_dir.expanduser().resolve()
    metadata = _load_run_metadata(output_dir)
    workflow_name = str(metadata["workflow"])
    thread_id = str(metadata["thread_id"])
    workflow = create_workflow(
        workflow_name,
        checkpointer=JsonFileCheckpointer(output_dir / "checkpoints"),
    )
    return _execute(workflow, command, output_dir, thread_id)


def _execute(workflow: Any, value: Any, output_dir: Path, thread_id: str) -> int:
    events_path = output_dir / "logs" / "events.jsonl"
    terminal = None
    with events_path.open("a", encoding="utf-8") as event_log:
        for event in workflow.stream(
            value,
            config={
                "thread_id": thread_id,
                "artifact_root": str(output_dir / "artifacts"),
                "checkpoint_metadata": {
                    "workflow": workflow.name,
                    "version": workflow.version,
                },
            },
        ):
            event_log.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            if event["type"] in {"graph_completed", "graph_interrupted"}:
                terminal = event
    if terminal is None:
        raise RuntimeError("workflow did not emit a terminal event")
    _write_json(output_dir / "result.json", terminal)
    print(json.dumps(_compact_result(terminal), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _capture_provenance(output_dir: Path, workspace_root: Path) -> None:
    git = {}
    for name, argv in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
    }.items():
        completed = subprocess.run(
            argv,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        git[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.rstrip(),
            "stderr": completed.stderr.rstrip(),
        }
    _write_json(output_dir / "git.json", git)
    _write_json(
        output_dir / "host.json",
        {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
        },
    )
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    _write_json(output_dir / "packages.json", dict(sorted(packages.items())))


def _compact_result(event: Mapping[str, Any]) -> dict[str, Any]:
    state = event.get("state", {})
    if not isinstance(state, Mapping):
        state = {}
    return {
        "thread_id": event.get("thread_id"),
        "status": event.get("status"),
        "step": event.get("step"),
        "next_node": event.get("next_node"),
        "interrupts": event.get("interrupts", []),
        "workflow_outcome": state.get("workflow_outcome"),
        "result_path": "result.json",
    }


def _load_run_metadata(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"workflow output directory is missing: {output_dir}")
    return _read_json_object(output_dir / "run.json")


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _read_json_value(path: Path | None) -> Any:
    if path is None:
        raise ValueError("resume file is required")
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
