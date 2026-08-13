#!/usr/bin/env python3
"""Run a reproducible Qwen3-VL sparse long-video physical-observation probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROMPT = """
You are auditing a generated long robot video. The images are ordered samples from
one video and each image is followed by its exact frame index and timestamp.
Report only directly visible evidence. Never infer metric depth, camera scale,
robot joint angles/velocities, contact force, or force closure from appearance.
Return JSON only, with exactly this schema:
{
  "schema_version": "1.0.0",
  "observations": [{
    "frame_index": 0,
    "timestamp_s": 0.0,
    "left_hand_visibility": "clear|partial|occluded|absent",
    "right_hand_visibility": "clear|partial|occluded|absent",
    "finger_integrity": "normal|ambiguous|deformed|motion_blur|not_visible",
    "left_contact": "none|near|touching|grasping|unknown",
    "right_contact": "none|near|touching|grasping|unknown",
    "flower_motion": "moving|static|ambiguous",
    "camera_motion": "static|moving|ambiguous",
    "evidence_note": "one short visible fact"
  }],
  "failure_intervals": [{
    "start_s": 0.0,
    "end_s": 0.0,
    "failure": "short description",
    "confidence": 0.0
  }],
  "non_observable": {
    "metric_depth": true,
    "absolute_camera_scale": true,
    "full_q_qdot": true,
    "contact_force": true,
    "force_closure": true
  }
}
Emit one observation for every supplied image, using the supplied indices and
timestamps unchanged. "static" means no visible flower displacement compared
with adjacent samples; if evidence is insufficient use "ambiguous" or "unknown".
""".strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--samples", type=int, default=14)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=24 * 1024)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.splitlines():
        parts = [value.strip() for value in line.split(",")]
        if len(parts) != 7:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "physical_index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mib": int(parts[3]),
                "memory_free_mib": int(parts[4]),
                "memory_used_mib": int(parts[5]),
                "utilization_percent": int(parts[6]),
            }
        )
    return rows


def _video_info(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    stream = payload["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", maxsplit=1)
    fps = int(numerator) / int(denominator)
    frames = int(stream.get("nb_frames") or round(float(payload["format"]["duration"]) * fps))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frames": frames,
        "duration_s": float(payload["format"]["duration"]),
    }


def _sample_indices(total: int, count: int) -> list[int]:
    if count < 2 or total < count:
        raise ValueError("sample count must be at least two and no larger than frame count")
    return sorted({round(index * (total - 1) / (count - 1)) for index in range(count)})


def _extract_frames(video: Path, frame_dir: Path, indices: list[int]) -> list[dict[str, Any]]:
    frame_dir.mkdir(parents=True)
    rows = []
    for ordinal, frame_index in enumerate(indices):
        output = frame_dir / f"sample-{ordinal:02d}-f{frame_index:04d}.jpg"
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output),
        ]
        subprocess.run(command, check=True)
        if not output.is_file():
            raise RuntimeError(f"ffmpeg did not extract frame {frame_index}")
        rows.append({"frame_index": frame_index, "path": str(output), "sha256": _sha256(output)})
    return rows


def _parse_json_response(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response must contain one JSON object")
    return value


def _git_state(root: Path) -> dict[str, Any]:
    result = {}
    for name, command in (
        ("head", ["git", "rev-parse", "HEAD"]),
        ("status", ["git", "status", "--short"]),
    ):
        completed = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
        result[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return result


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    source_video = args.source_video.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {output_dir}")
    if not source_video.is_file() or not model_path.is_dir():
        raise FileNotFoundError("source video and model directory must exist")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    inventory = _gpu_inventory()
    matches = [row for row in inventory if row["physical_index"] == args.gpu]
    if len(matches) != 1:
        raise ValueError(f"physical GPU {args.gpu} is not present")
    selected = matches[0]
    if selected["memory_free_mib"] < args.minimum_free_gpu_mib:
        raise RuntimeError(
            f"GPU {args.gpu} has {selected['memory_free_mib']} MiB free; "
            f"need {args.minimum_free_gpu_mib} MiB"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    video = _video_info(source_video)
    indices = _sample_indices(video["frames"], args.samples)
    frame_rows = _extract_frames(source_video, output_dir / "frames", indices)
    for row in frame_rows:
        row["timestamp_s"] = row["frame_index"] / video["fps"]
    prompt_path = output_dir / "prompt.txt"
    prompt_path.write_text(PROMPT + "\n")
    metadata: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "seed": args.seed,
        "git": _git_state(project_root),
        "gpu_inventory": inventory,
        "selected_gpu": selected,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "source": {"path": str(source_video), "sha256": _sha256(source_video), **video},
        "model": {"name": args.model_name, "path": str(model_path)},
        "sampling": {"count": len(frame_rows), "frames": frame_rows},
        "prompt": {"path": str(prompt_path), "sha256": _sha256(prompt_path)},
        "limitations": [
            "Sparse VLM labels are model-derived hypotheses, not independent measurements.",
            "No metric depth, q/qdot, contact force, or force closure is observable here.",
        ],
    }
    _write_json(output_dir / "metadata.json", metadata)
    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("selected physical GPU must map to exactly one CUDA device")
        torch.manual_seed(args.seed)
        load_started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="sdpa",
            )
        except TypeError:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="sdpa",
            )
        model.eval()
        load_seconds = time.perf_counter() - load_started
        content: list[dict[str, Any]] = [{"type": "text", "text": PROMPT}]
        for row in frame_rows:
            content.extend(
                [
                    {
                        "type": "image",
                        "image": row["path"],
                        "max_pixels": args.max_pixels,
                    },
                    {
                        "type": "text",
                        "text": (
                            f"frame_index={row['frame_index']}; "
                            f"timestamp_s={row['timestamp_s']:.6f}"
                        ),
                    },
                ]
            )
        messages = [{"role": "user", "content": content}]
        try:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        image_patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", 14)
        image_inputs, video_inputs = process_vision_info(
            messages, image_patch_size=image_patch_size
        )
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(next(model.parameters()).device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        inference_seconds = time.perf_counter() - inference_started
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        completion = generated[:, prompt_tokens:]
        raw = processor.batch_decode(
            completion, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        raw_path = output_dir / "raw-response.txt"
        raw_path.write_text(raw + "\n")
        parsed = _parse_json_response(raw)
        parsed["source_video_sha256"] = metadata["source"]["sha256"]
        parsed["model"] = {
            "name": args.model_name,
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        }
        parsed["evidence_class"] = "foundation_model_estimate"
        parsed["independent_physical_groups"] = 0
        parsed["physical_gate_eligible"] = False
        report_path = output_dir / "observation-report.json"
        _write_json(report_path, parsed)
        metadata.update(
            {
                "status": "completed",
                "packages": {
                    "torch": torch.__version__,
                    "transformers": importlib.metadata.version("transformers"),
                    "qwen-vl-utils": importlib.metadata.version("qwen-vl-utils"),
                },
                "performance": {
                    "model_load_seconds": load_seconds,
                    "inference_seconds": inference_seconds,
                    "sampled_frames_per_inference_second": len(frame_rows) / inference_seconds,
                    "source_video_seconds_per_inference_second": video["duration_s"] / inference_seconds,
                    "wall_seconds": time.perf_counter() - started,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": int(completion.shape[-1]),
                    "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
                },
                "outputs": {
                    "raw_response": {"path": str(raw_path), "sha256": _sha256(raw_path)},
                    "observation_report": {
                        "path": str(report_path),
                        "sha256": _sha256(report_path),
                    },
                },
            }
        )
        _write_json(output_dir / "metadata.json", metadata)
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            }
        )
        _write_json(output_dir / "metadata.json", metadata)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
