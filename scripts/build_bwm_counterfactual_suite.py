#!/usr/bin/env python3
"""Create an immutable BWM factual/action-swap diagnostic suite."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.counterfactual import (  # noqa: E402
    build_action_swap_suite,
    rebase_absolute_eef_future,
    validate_counterfactual_sources,
)
from phiagent.acwm.worldarena import attach_worldarena_lineage  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_state(commit: str | None, branch: str | None) -> dict[str, object]:
    if (commit is None) != (branch is None):
        raise ValueError("--git-commit and --git-branch must be supplied together")
    if commit is not None:
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ValueError("--git-commit must be a lowercase 40-character SHA-1")
        return {
            "commit": commit,
            "branch": branch,
            "resolution": "explicit immutable source snapshot",
            "dirty": True,
        }
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    current_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "diff", "--quiet"],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode != 0
    return {
        "commit": completed.stdout.strip(),
        "branch": current_branch,
        "resolution": "local Git checkout",
        "dirty": dirty,
    }


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("numpy", "opencv-python", "pyarrow", "torch"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-index-start", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--swap-mode",
        choices=("history_preserving_rebased_future", "whole_sequence"),
        default="history_preserving_rebased_future",
    )
    parser.add_argument("--git-commit")
    parser.add_argument("--git-branch")
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = args.source_metadata.expanduser().resolve()
    dataset_manifest_path = args.dataset_manifest.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"source metadata is missing or empty: {source}")
    if not dataset_manifest_path.is_file() or dataset_manifest_path.stat().st_size == 0:
        raise ValueError(f"dataset manifest is missing or empty: {dataset_manifest_path}")
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is missing: {dataset_root}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite counterfactual suite: {output}")
    source_rows = [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    source_rows = list(attach_worldarena_lineage(source_rows, dataset_manifest))
    output.mkdir(parents=True)
    effective_dataset_root = dataset_root
    derived_actions: dict[str, str] | None = None
    if args.swap_mode == "history_preserving_rebased_future":
        import pyarrow as pa
        import pyarrow.parquet as pq

        normalized = validate_counterfactual_sources(source_rows)
        bundle = output / "bundle"
        bundle.mkdir()
        (bundle / "source").symlink_to(dataset_root, target_is_directory=True)
        derived_root = bundle / "derived-actions"
        derived_root.mkdir()
        prefixed_rows = copy.deepcopy(list(normalized))
        for row in prefixed_rows:
            for field in ("action", "video"):
                row[field]["data"] = f"source/{row[field]['data']}"
        derived_actions = {}
        for index, source_row in enumerate(normalized):
            donor_row = normalized[(index + 1) % len(normalized)]
            source_action = dataset_root / str(source_row["action"]["data"])
            donor_action = dataset_root / str(donor_row["action"]["data"])
            source_table = pq.read_table(source_action)
            donor_table = pq.read_table(donor_action)
            source_start = int(source_row["action"]["start_frame"])
            source_end = int(source_row["action"]["end_frame"]) + 1
            donor_start = int(donor_row["action"]["start_frame"])
            donor_end = int(donor_row["action"]["end_frame"]) + 1
            source_state = source_table.column("observation.state").to_pylist()[
                source_start:source_end
            ]
            donor_state = donor_table.column("observation.state").to_pylist()[
                donor_start:donor_end
            ]
            rebased = rebase_absolute_eef_future(
                source_state,
                donor_state,
                history_frames=int(source_row["history_frames"]),
                rotation_representation="quaternion",
            )
            derived = derived_root / f"source-{index:04d}-rebased-future.parquet"
            state_type = source_table.schema.field("observation.state").type
            action_type = source_table.schema.field("action").type
            derived_table = pa.Table.from_arrays(
                [
                    pa.array(rebased, type=state_type),
                    pa.array(
                        source_table.column("action").to_pylist()[
                            source_start:source_end
                        ],
                        type=action_type,
                    ),
                ],
                names=("observation.state", "action"),
            )
            pq.write_table(derived_table, derived)
            derived_actions[str(source_row["source_episode"])] = (
                f"derived-actions/{derived.name}"
            )
        source_rows = prefixed_rows
        effective_dataset_root = bundle
    rows, pairs = build_action_swap_suite(
        source_rows,
        episode_index_start=args.episode_index_start,
        swapped_action_by_source_episode=derived_actions,
    )
    for row in rows:
        for field in ("action", "video"):
            path = effective_dataset_root / str(row[field]["data"])
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"{field} artifact is missing or empty: {path}")
    command = [sys.executable, *sys.argv]
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    _write_json(output / "git-state.json", _git_state(args.git_commit, args.git_branch))
    suite = output / "counterfactual.jsonl"
    suite.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    task_files = (
        PROJECT_ROOT / "phiagent" / "acwm" / "counterfactual.py",
        PROJECT_ROOT / "scripts" / "build_bwm_counterfactual_suite.py",
    )
    manifest = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "seed": args.seed,
        "method": "bwm_factual_action_swap_diagnostic_v1",
        "source_metadata": {
            "path": str(source),
            "sha256": _sha256(source),
            "rows": len(source_rows),
        },
        "dataset_manifest": {
            "path": str(dataset_manifest_path),
            "sha256": _sha256(dataset_manifest_path),
        },
        "dataset_root": str(effective_dataset_root),
        "source_dataset_root": str(dataset_root),
        "swap_mode": args.swap_mode,
        "conditioning_semantics": {
            "action_type": "eef_abs",
            "conditioning_column": "observation.state",
            "history": "source state preserved exactly",
            "future": (
                "donor XYZ displacement and quaternion-relative rotation rebased on "
                "the source history endpoint using translation plus SO(3) composition"
            ),
            "raw_action_column": "source sequence preserved; unused by eef_abs loader",
        },
        "suite": {
            "path": str(suite),
            "sha256": _sha256(suite),
            "rows": len(rows),
            "independent_source_episodes": len(
                {str(pair["independent_group_id"]) for pair in pairs}
            ),
        },
        "episode_index_start": args.episode_index_start,
        "pairs": pairs,
        "packages": _package_versions(),
        "task_file_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in task_files
        },
        "claim_boundary": (
            "The audit conditions BWM on observation.state, which is a realized absolute "
            "EEF trajectory rather than a low-level command. Action swapping tests whether "
            "generation changes under a history-preserving, frame-aligned robot-base EEF "
            "future. Because no physical counterfactual reference exists, it does not "
            "establish action correctness or SOTA."
        ),
    }
    _write_json(output / "manifest.json", manifest)
    (output / "build.log").write_text(
        f"compiled {len(rows)} rows from {len(pairs)} independent source episodes\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
