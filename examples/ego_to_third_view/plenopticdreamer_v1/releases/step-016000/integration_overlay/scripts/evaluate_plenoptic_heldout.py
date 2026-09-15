#!/usr/bin/env python3
"""Evaluate a completed target-free Plenoptic run against held-out target RGB."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.plenoptic import sha256_file, write_json, resolve_runtime_path  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--runtime-root", type=Path, required=True)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--source-video", type=Path, required=True)
    result.add_argument("--target-video", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--source-git-state", type=Path)
    return result


def contained(root: Path, value: Path, label: str) -> Path:
    root = root.resolve()
    path = resolve_runtime_path(root, value)
    allowed = (root, (root/'../DATASETS').resolve(), (root/'../OUTPUTS').resolve())
    if not any(base in path.parents for base in allowed):
        raise ValueError(f"{label} must be inside the runtime's code, DATASETS or OUTPUTS directories: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def main() -> None:
    args = parser().parse_args()
    runtime, run = args.runtime_root.resolve(), args.run_dir.resolve()
    run.relative_to((runtime / '../OUTPUTS').resolve())
    result_path, suite_path = run / "result.json", run / "suite.json"
    result = json.loads(contained(runtime, result_path, "result record").read_text())
    suite_bytes = contained(runtime, suite_path, "generation suite").read_bytes()
    generated = contained(runtime, Path(result["dit"]), "generated video")
    inference_path = contained(runtime, Path(result["generation"]), "inference record")
    source = contained(runtime, args.source_video, "source video")
    target = args.target_video.resolve()
    if not target.is_file() or target.stat().st_size == 0:
        raise FileNotFoundError(f"held-out target video is missing: {target}")
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    if (inference.get("status") != "generated"
            or inference.get("target_reference_used_during_generation") is not False
            or inference.get("has_target_reference") is not False):
        raise ValueError("inference record does not prove target-free generation")
    target_hash = sha256_file(target)
    generation_inputs = set(inference.get("input_sha256", {}).values())
    if target_hash in generation_inputs or str(target).encode() in suite_bytes:
        raise ValueError("held-out target appears in the generation inputs or suite")
    output = args.output.resolve()
    output.relative_to(run)
    if output.exists():
        raise FileExistsError(f"held-out evaluation output must be fresh: {output}")
    output.mkdir(parents=True)

    tools = runtime / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from plenoptic_data import decode_video
    from validation_image_metrics import measure_video_errors

    indices, hw = list(range(81)), (432, 768)
    def frames(path: Path):
        return decode_video(str(path), indices, hw, calibrated_crop=True).permute(1, 2, 3, 0).numpy()

    metrics = measure_video_errors(frames(generated), frames(target), frames(source))
    comparison = output / "source-generated-heldout.mp4"
    temporary = output / "source-generated-heldout.tmp.mp4"
    command = [
        "ffmpeg", "-v", "error", "-y", "-i", str(source), "-i", str(generated),
        "-i", str(target), "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[outv]",
        "-map", "[outv]", "-an", "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", "-threads", "4", str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(comparison)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate", "-of", "json",
        str(comparison),
    ], text=True))["streams"][0]
    if (int(probe["nb_read_frames"]), int(probe["width"]), int(probe["height"]),
            probe["r_frame_rate"]) != (81, 2304, 432, "15/1"):
        raise ValueError(f"unexpected comparison encoding: {probe}")
    git = (json.loads(args.source_git_state.read_text(encoding="utf-8"))
           if args.source_git_state else {"available": False})
    report = {
        "schema": 1,
        "status": "measured",
        "scope": "Held-out target pixels introduced only after target-free generation completed.",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "invocation": shlex.join([sys.executable, *sys.argv]),
        "source_git": git,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "generation_record": str(inference_path),
        "generation_created_at": inference.get("created_at"),
        "target_reference_used_during_generation": False,
        "has_target_reference_during_generation": False,
        "suite_sha256": sha256_file(suite_path),
        "source_sha256": sha256_file(source),
        "generated_sha256": sha256_file(generated),
        "heldout_target_sha256": target_hash,
        "heldout_target_absent_from_generation_inputs": True,
        "metrics": metrics,
        "comparison": str(comparison),
        "comparison_sha256": sha256_file(comparison),
        "encoded": probe,
    }
    write_json(output / "evaluation.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
