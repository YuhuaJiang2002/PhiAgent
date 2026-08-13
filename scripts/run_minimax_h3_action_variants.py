#!/usr/bin/env python3
"""Generate matched language-conditioned MiniMax-H3 real-scene videos.

The heavyweight H3 pipeline is loaded once, then every action variant uses the
same source video, robot reference, seed and inference settings.  Only the
language action condition changes.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.minimax_h3 import (  # noqa: E402
    DIFFSYNTH_H3_COMMIT,
    MINIMAX_H3_MODELSCOPE_ID,
    MINIMAX_H3_NF4_MODEL_ID,
    H3ActionVariant,
    MiniMaxH3ValidationConfig,
    build_action_conditioned_flower_ref2va_prompt,
    build_action_conditioned_ego_bottle_ref2va_prompt,
    build_action_conditioned_tabletop_ref2va_prompt,
    file_sha256,
    verify_diffsynth_h3_source,
)
from phiagent.rendering.wan_animate import query_gpus  # noqa: E402


class _Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _compose_action_references(
    references: list[dict[str, Any]],
    *,
    continuation: Any | None,
    control_frames: Any | None,
) -> list[dict[str, Any]]:
    """Keep both recursive state and motion control in the H3 reference list."""

    result = list(references)
    if continuation is not None:
        result.append({"type": "image", "image": continuation})
    if control_frames is not None:
        result.append({"type": "video", "video": control_frames})
    return result


def _raise_on_termination(signum: int, _frame: object) -> None:
    raise RuntimeError(
        f"received {signal.Signals(signum).name}; action-variant experiment did not complete"
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _packages() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in (
        "torch",
        "transformers",
        "modelscope",
        "bitsandbytes",
        "safetensors",
        "accelerate",
        "huggingface-hub",
        "av",
        "diffsynth",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _git_state(root: Path) -> dict[str, object]:
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "available": status.returncode == 0,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
        "error": status.stderr.strip() if status.returncode != 0 else None,
    }


def _load_actions(path: Path) -> tuple[H3ActionVariant, ...]:
    raw = json.loads(path.read_text())
    entries = raw.get("actions") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or len(entries) < 1:
        raise ValueError("action manifest must contain a non-empty 'actions' list")
    actions = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("every action must be a JSON object")
        action = H3ActionVariant(
            label=str(entry.get("label", "")),
            instruction=str(entry.get("instruction", "")),
            timeline=str(entry.get("timeline", "")),
        )
        action.validate()
        actions.append(action)
    labels = [action.label for action in actions]
    if len(labels) != len(set(labels)):
        raise ValueError("action labels must be unique")
    return tuple(actions)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--robot-reference", type=Path, required=True)
    parser.add_argument("--action-manifest", type=Path, required=True)
    parser.add_argument("--diffsynth-repo", type=Path, required=True)
    parser.add_argument("--model-base-path", type=Path, required=True)
    parser.add_argument(
        "--experiment-root", type=Path, default=Path("outputs/minimax-h3-action-control")
    )
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=54 * 1024)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--reference-short-edge", type=int, default=768)
    parser.add_argument("--reference-video-short-edge", type=int, default=480)
    parser.add_argument(
        "--scene-reference-mode",
        choices=("video", "anchor_image", "control_video"),
        default="video",
        help="Use the full source motion or one real-video anchor frame as H3 scene reference.",
    )
    parser.add_argument("--scene-anchor-frame", type=int, default=60)
    parser.add_argument(
        "--scene-domain",
        choices=("flower", "tabletop_bowl", "ego_bottle"),
        default="flower",
        help="Select the domain-specific object and scene constraints used in H3 prompts.",
    )
    parser.add_argument(
        "--action-control-root",
        type=Path,
        help="Experiment root containing variants/<label>/action-control.mp4.",
    )
    parser.add_argument(
        "--continuation-reference-root",
        type=Path,
        help=(
            "Optional root containing variants/<label>/continuation.png from the "
            "preceding matched action window."
        ),
    )
    parser.add_argument("--vram-reserve-gib", type=float, default=8.0)
    parser.add_argument(
        "--lora-checkpoint",
        type=Path,
        help="Optional native H3 Ref2VA LoRA checkpoint to evaluate.",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if args.experiment_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment = args.experiment_root.expanduser().resolve() / f"{stamp}-{uuid4().hex[:8]}"
    else:
        experiment = args.experiment_dir.expanduser().resolve()
    metadata_path = experiment / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"experiment metadata already exists: {metadata_path}")

    source_video = args.source_video.expanduser().resolve()
    robot_reference = args.robot_reference.expanduser().resolve()
    action_manifest = args.action_manifest.expanduser().resolve()
    actions = _load_actions(action_manifest)
    lora_checkpoint = (
        args.lora_checkpoint.expanduser().resolve()
        if args.lora_checkpoint is not None
        else None
    )
    for source in (source_video, robot_reference, action_manifest, lora_checkpoint):
        if source is None:
            continue
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"experiment input does not exist or is empty: {source}")
    if not 0.0 < args.lora_scale <= 2.0:
        raise ValueError("--lora-scale must lie in (0, 2]")
    source_controls: dict[str, Path] = {}
    if args.scene_reference_mode == "control_video":
        if args.action_control_root is None:
            raise ValueError("--action-control-root is required for control_video mode")
        control_root = args.action_control_root.expanduser().resolve()
        for action in actions:
            source_control = control_root / "variants" / action.label / "action-control.mp4"
            if not source_control.is_file() or source_control.stat().st_size == 0:
                raise ValueError(f"action control video is missing: {source_control}")
            source_controls[action.label] = source_control
    source_continuations: dict[str, Path] = {}
    if args.continuation_reference_root is not None:
        continuation_root = args.continuation_reference_root.expanduser().resolve()
        for action in actions:
            source_continuation = (
                continuation_root / "variants" / action.label / "continuation.png"
            )
            if not source_continuation.is_file() or source_continuation.stat().st_size == 0:
                raise ValueError(
                    f"continuation reference is missing: {source_continuation}"
                )
            source_continuations[action.label] = source_continuation

    experiment.mkdir(parents=True, exist_ok=True)
    duration_seconds = args.num_frames / args.fps
    input_dir = experiment / "input"
    input_dir.mkdir(parents=True)
    copied_source = input_dir / f"source{source_video.suffix.lower()}"
    copied_robot = input_dir / f"robot-reference{robot_reference.suffix.lower()}"
    copied_manifest = input_dir / "action-variants.json"
    for source, destination in (
        (source_video, copied_source),
        (robot_reference, copied_robot),
        (action_manifest, copied_manifest),
    ):
        shutil.copy2(source, destination)
    copied_lora = None
    if lora_checkpoint is not None:
        copied_lora = input_dir / f"identity-topology-lora{lora_checkpoint.suffix.lower()}"
        shutil.copy2(lora_checkpoint, copied_lora)

    control_videos: dict[str, Path] = {}
    if args.scene_reference_mode == "control_video":
        for action in actions:
            copied_control = input_dir / "action-controls" / f"{action.label}.mp4"
            copied_control.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_controls[action.label], copied_control)
            control_videos[action.label] = copied_control
    continuation_images: dict[str, Path] = {}
    for action in actions:
        if action.label not in source_continuations:
            continue
        source_continuation = source_continuations[action.label]
        copied_continuation = (
            input_dir / "continuation-references" / f"{action.label}.png"
        )
        copied_continuation.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_continuation, copied_continuation)
        continuation_images[action.label] = copied_continuation

    prompts: dict[str, Path] = {}
    for action in actions:
        prompt_path = input_dir / "prompts" / f"{action.label}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        if args.scene_domain == "tabletop_bowl":
            if args.scene_reference_mode != "control_video":
                raise ValueError("tabletop_bowl currently requires control_video mode")
            prompt = build_action_conditioned_tabletop_ref2va_prompt(
                duration_seconds, action
            )
        elif args.scene_domain == "ego_bottle":
            if args.scene_reference_mode != "control_video":
                raise ValueError("ego_bottle currently requires control_video mode")
            prompt = build_action_conditioned_ego_bottle_ref2va_prompt(
                duration_seconds, action
            )
        else:
            prompt = build_action_conditioned_flower_ref2va_prompt(
                duration_seconds,
                action,
                scene_reference=args.scene_reference_mode,
            )
        if action.label in continuation_images:
            if args.scene_domain == "ego_bottle":
                continuation_state = (
                    "first-person camera viewpoint, two robot-hand poses, bottle holder, "
                    "grasp state, bottle placement and depth ordering"
                )
                forbidden_reset = "Do not reveal a robot head or torso"
            elif args.scene_domain == "tabletop_bowl":
                continuation_state = (
                    "robot-hand identity, hand pose, bowl holder, contact state, bowl "
                    "placement and depth ordering"
                )
                forbidden_reset = "Do not reset the bowl or hand"
            else:
                continuation_state = (
                    "robot identity, body pose, flower holder, grasp state, object "
                    "placement and depth ordering"
                )
                forbidden_reset = "Do not reset to the initial pose"
            prompt += (
                "\ncontinuation_reference:\n"
                "<Picture 3> is the same action's generated state at this window's "
                f"absolute start frame. Preserve its {continuation_state} while following "
                f"the remainder of <Video 1>. {forbidden_reset}, and do not copy state "
                "from another action variant.\n"
            )
        prompt_path.write_text(prompt)
        prompts[action.label] = prompt_path

    # The shared config performs geometry/GPU-contract validation.  Its prompt
    # field points at the first generated prompt; every prompt is validated and
    # hashed separately below.
    config = MiniMaxH3ValidationConfig(
        source_video=copied_source,
        robot_reference=copied_robot,
        prompt_file=prompts[actions[0].label],
        diffsynth_repo=args.diffsynth_repo.expanduser().resolve(),
        model_base_path=args.model_base_path.expanduser().resolve(),
        width=args.width,
        height=args.height,
        fps=args.fps,
        num_frames=args.num_frames,
        steps=args.steps,
        seed=args.seed,
        minimum_free_gpu_mib=args.minimum_free_gpu_mib,
        requested_gpu=args.gpu,
    )
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "method": (
            "minimax_h3_nf4_matched_language_action_condition_comparison"
            if len(actions) > 1
            else "minimax_h3_nf4_language_action_condition_generation"
        ),
        "status": "preflight_started",
        "honest_status": "NOT STARTED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git": _git_state(project_root),
        "config": {
            "source_video": str(config.source_video),
            "robot_reference": str(config.robot_reference),
            "action_manifest": str(copied_manifest),
            "diffsynth_repo": str(config.diffsynth_repo),
            "model_base_path": str(config.model_base_path),
            "width": config.width,
            "height": config.height,
            "fps": config.fps,
            "num_frames": config.num_frames,
            "steps": config.steps,
            "seed": config.seed,
            "minimum_free_gpu_mib": config.minimum_free_gpu_mib,
            "requested_gpu": config.requested_gpu,
            "reference_short_edge": args.reference_short_edge,
            "reference_video_short_edge": args.reference_video_short_edge,
            "scene_reference_mode": args.scene_reference_mode,
            "scene_domain": args.scene_domain,
            "scene_anchor_frame": args.scene_anchor_frame,
            "action_control_root": (
                str(args.action_control_root.expanduser().resolve())
                if args.action_control_root is not None
                else None
            ),
            "continuation_reference_root": (
                str(args.continuation_reference_root.expanduser().resolve())
                if args.continuation_reference_root is not None
                else None
            ),
            "vram_reserve_gib": args.vram_reserve_gib,
            "lora_checkpoint": str(copied_lora) if copied_lora is not None else None,
            "lora_checkpoint_sha256": (
                file_sha256(copied_lora) if copied_lora is not None else None
            ),
            "lora_scale": args.lora_scale if copied_lora is not None else None,
        },
        "matched_controls": [
            "same real-scene source video",
            "same robot identity reference",
            "same seed",
            "same MiniMax-H3 checkpoint and inference settings",
            "same continuation-reference policy for every matched action",
            (
                "only the action condition differs across variants: instruction, timeline, "
                "and its deterministically compiled control video"
                if args.scene_reference_mode == "control_video"
                else "only action instruction and timeline differ across variants"
            ),
        ],
        "model": {
            "base": "MiniMax-H3",
            "weights": MINIMAX_H3_NF4_MODEL_ID,
            "processor": MINIMAX_H3_MODELSCOPE_ID,
            "quantization": "third-party prequantized bitsandbytes NF4",
            "adaptation": "native Ref2VA LoRA" if copied_lora is not None else None,
        },
        "source_revision": DIFFSYNTH_H3_COMMIT,
        "actions": [
            {
                **asdict(action),
                "prompt": str(prompts[action.label]),
                "prompt_sha256": file_sha256(prompts[action.label]),
                "action_control": (
                    str(control_videos[action.label])
                    if action.label in control_videos
                    else None
                ),
                "action_control_sha256": (
                    file_sha256(control_videos[action.label])
                    if action.label in control_videos
                    else None
                ),
                "continuation_reference": (
                    str(continuation_images[action.label])
                    if action.label in continuation_images
                    else None
                ),
                "continuation_reference_sha256": (
                    file_sha256(continuation_images[action.label])
                    if action.label in continuation_images
                    else None
                ),
                "status": "pending",
            }
            for action in actions
        ],
        "limitations": [
            "This uses the third-party NF4 quantization, not the official BF16 checkpoint.",
            "This is real-scene video generation, not real-robot execution.",
            "Prompt compliance requires independent visual and action-adherence review.",
            "The source clip is silent; H3 audio is not an acceptance target.",
            (
                "H3 receives one real-video anchor frame instead of source temporal motion; "
                "the agent evaluator restores the aligned real source outside the subject support."
                if args.scene_reference_mode == "anchor_image"
                else (
                    "H3 receives an explicit per-action control video compiled from camera-pixel "
                    "arm trajectories; those controls are intermediate evidence, not final outputs."
                    if args.scene_reference_mode == "control_video"
                    else "The full source video can dominate prompt motion despite explicit exclusions."
                )
            ),
            (
                "Each second-window action receives only its own preceding generated frame as "
                "a continuation reference; it does not share another action's state."
                if continuation_images
                else "No generated continuation image is supplied for the first action window."
            ),
        ],
    }
    _write_json(metadata_path, record)
    for signal_name in ("SIGHUP", "SIGTERM"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), _raise_on_termination)

    log_path = experiment / "inference.log"
    try:
        config.validate()
        for path in prompts.values():
            if not path.is_file() or not path.read_text().strip():
                raise ValueError(f"generated prompt is empty: {path}")
        source_commit = verify_diffsynth_h3_source(config.diffsynth_repo)
        gpus, inventory_raw, processes_raw = query_gpus()
        selected = config.select_gpu(gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
        os.environ["PYTHONHASHSEED"] = str(config.seed)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(config.diffsynth_repo), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(config.model_base_path)
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
        sys.path.insert(0, str(config.diffsynth_repo))
        config.model_base_path.mkdir(parents=True, exist_ok=True)
        record.update(
            {
                "status": "preflight_passed" if args.preflight_only else "running",
                "honest_status": "PARTIAL" if not args.preflight_only else "NOT STARTED",
                "source_revision": source_commit,
                "selected_gpu": asdict(selected),
                "gpu_inventory": [asdict(gpu) for gpu in gpus],
                "gpu_inventory_raw": inventory_raw,
                "gpu_processes_raw": processes_raw,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "packages": _packages(),
                "inputs": {
                    "source_video": {"path": str(copied_source), "sha256": file_sha256(copied_source)},
                    "robot_reference": {"path": str(copied_robot), "sha256": file_sha256(copied_robot)},
                    "action_manifest": {"path": str(copied_manifest), "sha256": file_sha256(copied_manifest)},
                },
            }
        )
        _write_json(metadata_path, record)
        if args.preflight_only:
            print(json.dumps({"experiment": str(experiment), "status": record["status"]}))
            return 0

        with log_path.open("w", encoding="utf-8") as log:
            tee_out = _Tee(sys.stdout, log)
            tee_err = _Tee(sys.stderr, log)
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                import torch
                from PIL import Image
                from diffsynth.pipelines.minimax_h3_audio_video import (
                    MiniMaxH3Pipeline,
                    ModelConfig,
                )
                from diffsynth.utils.data.audio_video import read_video_audio, write_video_audio

                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise RuntimeError("selected physical GPU did not map to exactly one CUDA device")
                free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
                free_gib = free_bytes / 1024**3
                vram_limit_gib = free_gib - args.vram_reserve_gib
                if vram_limit_gib <= 8:
                    raise RuntimeError(
                        f"only {free_gib:.2f} GiB is free; the reserve leaves too little VRAM"
                    )
                record["runtime"] = {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "logical_devices": torch.cuda.device_count(),
                    "logical_gpu_name": torch.cuda.get_device_name(0),
                    "free_gib_at_load": free_gib,
                    "total_gib": total_bytes / 1024**3,
                    "vram_limit_gib": vram_limit_gib,
                }
                _write_json(metadata_path, record)
                vram_config = {
                    "offload_dtype": "disk",
                    "offload_device": "disk",
                    "onload_dtype": torch.bfloat16,
                    "onload_device": "cpu",
                    "preparing_dtype": torch.bfloat16,
                    "preparing_device": "cuda",
                    "computation_dtype": torch.bfloat16,
                    "computation_device": "cuda",
                }
                pipe = MiniMaxH3Pipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device="cuda",
                    model_configs=[
                        ModelConfig(
                            model_id=MINIMAX_H3_NF4_MODEL_ID,
                            origin_file_pattern="minimax-h3-ref2va-nf4.safetensors",
                            **vram_config,
                        ),
                        ModelConfig(
                            model_id=MINIMAX_H3_NF4_MODEL_ID,
                            origin_file_pattern="minimax-h3-text-encoder-nf4.safetensors",
                            **vram_config,
                        ),
                        ModelConfig(
                            model_id=MINIMAX_H3_NF4_MODEL_ID,
                            origin_file_pattern="video_vae_nf4.safetensors",
                            **vram_config,
                        ),
                        ModelConfig(
                            model_id=MINIMAX_H3_NF4_MODEL_ID,
                            origin_file_pattern="audio_vae_nf4.safetensors",
                            **vram_config,
                        ),
                    ],
                    processor_config=ModelConfig(
                        model_id=MINIMAX_H3_MODELSCOPE_ID,
                        origin_file_pattern="Ref2VA/processor/",
                    ),
                    vram_limit=vram_limit_gib,
                )
                if copied_lora is not None:
                    pipe.load_lora(
                        pipe.dit,
                        str(copied_lora),
                        alpha=args.lora_scale,
                    )
                frames, _, _ = read_video_audio(
                    str(config.source_video),
                    height=config.height,
                    width=config.width,
                    num_frames=config.num_frames,
                    fps=config.fps,
                    audio_sample_rate=pipe.audio_vae.sample_rate,
                )
                if len(frames) != config.num_frames:
                    raise RuntimeError(
                        f"reference decoder returned {len(frames)} frames, expected {config.num_frames}"
                    )
                robot = Image.open(config.robot_reference).convert("RGB")
                if not 0 <= args.scene_anchor_frame < len(frames):
                    raise ValueError("scene-anchor-frame is outside the decoded source clip")
                if args.scene_reference_mode in {"anchor_image", "control_video"}:
                    scene_anchor = frames[args.scene_anchor_frame]
                    if not isinstance(scene_anchor, Image.Image):
                        scene_anchor = Image.fromarray(scene_anchor)
                    scene_anchor = scene_anchor.convert("RGB")
                    scene_anchor_path = input_dir / f"scene-anchor-{args.scene_anchor_frame:03d}.png"
                    scene_anchor.save(scene_anchor_path)
                    record["inputs"]["scene_anchor"] = {
                        "path": str(scene_anchor_path),
                        "source_frame": args.scene_anchor_frame,
                        "frame": "camera:source_clip_pixels",
                        "sha256": file_sha256(scene_anchor_path),
                    }
                    _write_json(metadata_path, record)
                    references = [
                        {"type": "image", "image": robot},
                        {"type": "image", "image": scene_anchor},
                    ]
                else:
                    references = [
                        {"type": "image", "image": robot},
                        {"type": "video", "video": frames},
                    ]
                for index, action in enumerate(actions):
                    record["actions"][index]["status"] = "running"
                    record["actions"][index]["started_at"] = datetime.now(timezone.utc).isoformat()
                    _write_json(metadata_path, record)
                    print(f"ACTION_VARIANT_START {index + 1}/{len(actions)} {action.label}")
                    continuation = None
                    if action.label in continuation_images:
                        continuation = Image.open(
                            continuation_images[action.label]
                        ).convert("RGB")
                    control_frames = None
                    if args.scene_reference_mode == "control_video":
                        control_frames, _, _ = read_video_audio(
                            str(control_videos[action.label]),
                            height=config.height,
                            width=config.width,
                            num_frames=config.num_frames,
                            fps=config.fps,
                            audio_sample_rate=pipe.audio_vae.sample_rate,
                        )
                        if len(control_frames) != config.num_frames:
                            raise RuntimeError(
                                f"control decoder returned {len(control_frames)} frames for "
                                f"{action.label}, expected {config.num_frames}"
                            )
                    action_references = _compose_action_references(
                        references,
                        continuation=continuation,
                        control_frames=control_frames,
                    )
                    video, audio = pipe(
                        prompt=prompts[action.label].read_text().strip(),
                        height=config.height,
                        width=config.width,
                        num_frames=config.num_frames,
                        num_inference_steps=config.steps,
                        seed=config.seed,
                        references=action_references,
                        ref_image_short_edge=args.reference_short_edge,
                        ref_video_short_edge=args.reference_video_short_edge,
                        ref_video_max_pixels=config.height * config.width,
                    )
                    result = experiment / "variants" / action.label / "raw-h3-nf4.mp4"
                    result.parent.mkdir(parents=True, exist_ok=True)
                    write_video_audio(
                        video=video,
                        audio=audio,
                        output_path=str(result),
                        fps=config.fps,
                        audio_sample_rate=32000,
                    )
                    del video, audio
                    if continuation is not None:
                        continuation.close()
                    gc.collect()
                    torch.cuda.empty_cache()
                    subprocess.run(
                        ["ffmpeg", "-v", "error", "-i", str(result), "-f", "null", "-"],
                        check=True,
                    )
                    record["actions"][index].update(
                        {
                            "status": "completed",
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                            "result": str(result),
                            "result_sha256": file_sha256(result),
                            "bytes": result.stat().st_size,
                            "output_decoded": True,
                        }
                    )
                    _write_json(metadata_path, record)
                    print(f"ACTION_VARIANT_COMPLETE {index + 1}/{len(actions)} {action.label}")

        checkpoint_root = config.model_base_path / MINIMAX_H3_NF4_MODEL_ID
        checkpoint_files = sorted(path for path in checkpoint_root.rglob("*") if path.is_file())
        record.update(
            {
                "status": "completed",
                "honest_status": "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "checkpoint_files": [
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                    for path in checkpoint_files
                ],
                "acceptance": {
                    "all_inferences_completed": True,
                    "all_outputs_decoded": True,
                    "real_scene_input_used": True,
                    "matched_action_comparison": True,
                    "visual_quality_evaluated": False,
                    "action_adherence_evaluated": False,
                },
            }
        )
        _write_json(metadata_path, record)
        print(json.dumps({"experiment": str(experiment), "status": record["status"]}))
        return 0
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "honest_status": "BLOCKED",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(metadata_path, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
