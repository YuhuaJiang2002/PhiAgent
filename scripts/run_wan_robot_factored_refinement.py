#!/usr/bin/env python3
"""Run a sharp robot-factored Wan refinement over an action-specific driver.

The driver supplies motion/geometry.  The real Ego clip supplies the static scene.
Their masks are unioned before generation so source-human pixels cannot be copied
back into the generated character region.  No post-generation alpha repair or
temporal blur is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.wan_animate import (  # noqa: E402
    WAN22_COMMIT,
    WanAnimateConfig,
    WanAnimateRenderer,
    acquire_gpu_lease,
    query_gpus,
    select_gpu,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"command failed with exit code {code}: {' '.join(command)}")


def _video_info(path: Path) -> dict[str, int | float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = (int(value) for value in stream["r_frame_rate"].split("/"))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": numerator / denominator,
        "frames": int(stream["nb_read_frames"]),
    }


def _merge_factored_controls(
    real_source: Path,
    source_preprocess: Path | None,
    driver_preprocess: Path,
    output: Path,
    *,
    dilation_pixels: int,
    human_guard_y: float,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    source_mode = "wan_union" if source_preprocess is not None else "factored_guard"
    source_bg_path = source_preprocess / "src_bg.mp4" if source_preprocess else real_source
    source_bg = cv2.VideoCapture(str(source_bg_path))
    source_mask = (
        cv2.VideoCapture(str(source_preprocess / "src_mask.mp4"))
        if source_preprocess
        else None
    )
    driver_mask = cv2.VideoCapture(str(driver_preprocess / "src_mask.mp4"))
    controls = (
        (source_bg, driver_mask)
        if source_mask is None
        else (source_bg, source_mask, driver_mask)
    )
    if not all(item.isOpened() for item in controls):
        raise RuntimeError("could not decode factored Wan preprocessing controls")
    width = int(driver_mask.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(driver_mask.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(driver_mask.get(cv2.CAP_PROP_FPS))
    output.mkdir(parents=True, exist_ok=False)
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    bg_writer = cv2.VideoWriter(str(output / "src_bg.mkv"), fourcc, fps, (width, height))
    mask_writer = cv2.VideoWriter(str(output / "src_mask.mkv"), fourcc, fps, (width, height))
    if not bg_writer.isOpened() or not mask_writer.isOpened():
        raise RuntimeError("could not create lossless hybrid control videos")
    kernel_size = max(1, dilation_pixels * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    frames = 0
    union_fractions = []
    source_fractions = []
    driver_fractions = []
    try:
        while True:
            ok_driver, driver = driver_mask.read()
            if not ok_driver:
                break
            ok_bg, bg = source_bg.read()
            if not ok_bg:
                raise RuntimeError("real-source background ended before the action controls")
            ok_source, source = source_mask.read() if source_mask is not None else (True, None)
            if not ok_source:
                raise RuntimeError("source mask ended before the action controls")
            if bg.shape[:2] != (height, width):
                bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_LANCZOS4)
            source_binary = (
                cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) >= 127
                if source is not None
                else np.zeros((height, width), dtype=bool)
            )
            if source_preprocess is None:
                # Fail closed: the action mask removes the controlled robot, while
                # the lower guard prevents any uncertain human hand/sleeve pixel in
                # the real Ego source from being copied into the generated video.
                guard_start = int(round(height * human_guard_y))
                source_binary[guard_start:, :] = True
            driver_binary = cv2.cvtColor(driver, cv2.COLOR_BGR2GRAY) >= 127
            union = np.logical_or(source_binary, driver_binary).astype(np.uint8) * 255
            if dilation_pixels:
                union = cv2.dilate(union, kernel)
            factored_bg = bg.copy()
            factored_bg[union > 0] = 0
            union_bgr = cv2.cvtColor(union, cv2.COLOR_GRAY2BGR)
            bg_writer.write(factored_bg)
            mask_writer.write(union_bgr)
            source_fractions.append(float(np.mean(source_binary)))
            driver_fractions.append(float(np.mean(driver_binary)))
            union_fractions.append(float(np.mean(union > 0)))
            frames += 1
    finally:
        source_bg.release()
        if source_mask is not None:
            source_mask.release()
        driver_mask.release()
        bg_writer.release()
        mask_writer.release()
    if frames < 1:
        raise RuntimeError("hybrid control contains no frames")

    shutil.copy2(driver_preprocess / "src_pose.mp4", output / "src_pose.mp4")
    shutil.copy2(driver_preprocess / "src_ref.png", output / "src_ref.png")
    face_info = _video_info(driver_preprocess / "src_face.mp4")
    _run(
        [
            str(Path(shutil.which("ffmpeg") or "ffmpeg").resolve()),
            "-y",
            "-v",
            "error",
            "-i",
            str(driver_preprocess / "src_face.mp4"),
            "-vf",
            "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(output / "src_face.mp4"),
        ],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        log_path=output / "face-suppression.log",
    )
    for stem in ("src_bg", "src_mask"):
        _run(
            [
                str(Path(shutil.which("ffmpeg") or "ffmpeg").resolve()),
                "-y",
                "-v",
                "error",
                "-i",
                str(output / f"{stem}.mkv"),
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "0",
                "-pix_fmt",
                "yuv420p",
                str(output / f"{stem}.mp4"),
            ],
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            log_path=output / f"{stem}-encode.log",
        )
    return {
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "face_frames": face_info["frames"],
        "source_mask_mean_fraction": sum(source_fractions) / len(source_fractions),
        "driver_mask_mean_fraction": sum(driver_fractions) / len(driver_fractions),
        "union_mask_mean_fraction": sum(union_fractions) / len(union_fractions),
        "dilation_pixels": dilation_pixels,
        "source_mask_mode": source_mode,
        "human_guard_y": human_guard_y if source_preprocess is None else None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-source", type=Path, required=True)
    parser.add_argument("--action-driver", type=Path, required=True)
    parser.add_argument("--robot-reference", type=Path, required=True)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--sam2-repo", type=Path)
    parser.add_argument(
        "--driver-preprocess-override",
        type=Path,
        help="Validated task-pose/mask bundle to copy instead of rerunning fragile preprocessing.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=60 * 1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--clip-len", type=int, default=89)
    parser.add_argument("--reference-frames", type=int, choices=(1, 5), default=5)
    parser.add_argument("--mask-dilation", type=int, default=5)
    parser.add_argument(
        "--source-mask-mode",
        choices=("factored_guard", "wan_union"),
        default="factored_guard",
        help="factored_guard skips fragile human SAM2 preprocessing and masks the lower Ego region",
    )
    parser.add_argument(
        "--human-guard-y",
        type=float,
        default=0.52,
        help="top of the fail-closed lower-frame human-risk guard, as a frame fraction",
    )
    parser.add_argument("--prompt", default="Two coherent white and brushed-silver robot arms perform the controlled object manipulation.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.clip_len < 1 or (args.clip_len - 1) % 4:
        raise ValueError("--clip-len must satisfy 4n+1")
    if args.mask_dilation < 0:
        raise ValueError("--mask-dilation must be non-negative")
    if not 0.0 <= args.human_guard_y <= 1.0:
        raise ValueError("--human-guard-y must be in [0, 1]")
    inputs = {
        "real_source": args.real_source.expanduser().resolve(),
        "action_driver": args.action_driver.expanduser().resolve(),
        "robot_reference": args.robot_reference.expanduser().resolve(),
    }
    for label, path in inputs.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")
    driver_override = (
        args.driver_preprocess_override.expanduser().resolve()
        if args.driver_preprocess_override is not None
        else None
    )
    if driver_override is not None:
        for name in ("src_pose.mp4", "src_face.mp4", "src_mask.mp4", "src_ref.png", "manifest.json"):
            path = driver_override / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"driver preprocess override is incomplete: {path}")
    source_info = _video_info(inputs["real_source"])
    driver_info = _video_info(inputs["action_driver"])
    for field in ("frames", "width", "height", "fps"):
        if source_info[field] != driver_info[field]:
            raise ValueError(f"source and driver {field} differ: {source_info[field]} != {driver_info[field]}")

    config = WanAnimateConfig(
        wan_repo=args.wan_repo.expanduser().resolve(),
        checkpoint_dir=args.checkpoint_dir.expanduser().resolve(),
        sam2_repo=args.sam2_repo.expanduser().resolve() if args.sam2_repo else None,
        # Preserve the venv launcher path. Path.resolve() follows its interpreter
        # symlink to /usr/bin/python and silently drops the venv site-packages.
        python_executable=Path(os.path.abspath(args.python.expanduser())),
        gpu_index=args.gpu,
        minimum_free_gpu_mib=args.minimum_free_gpu_mib,
        resolution_width=args.width,
        resolution_height=args.height,
        fps=args.fps,
        frame_num=args.clip_len,
        infer_frames=80,
        reference_frames=args.reference_frames,
        mode="replacement",
        retarget=False,
        use_flux=False,
        use_relighting_lora=True,
        suppress_source_face_control=True,
        object_roi=(0.15, 0.25, 0.75, 0.74),
    )
    renderer = WanAnimateRenderer(config)
    preflight = renderer.preflight(select_cuda_device=True)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    selected = preflight["selected_gpu"]
    environment = renderer._execution_environment(  # noqa: SLF001
        selected["physical_index"], args.seed
    )

    root = args.experiment_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = root / f"{stamp}-{uuid4().hex[:8]}"
    experiment.mkdir()
    metadata_path = experiment / "metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "running",
        "method": "robot_factored_wan22_joint_replacement_fail_closed_no_post_blur",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "seed": args.seed,
        "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
        "preflight": preflight,
        "wan_commit": WAN22_COMMIT,
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in inputs.items()
        },
        "source_info": source_info,
        "driver_info": driver_info,
        "training_method_proxies": {
            "robot_factored_visible_geometry": "action-specific camera-frame driver supplies visible robot geometry",
            "history_context_guidance": f"{args.reference_frames} generated frames condition each Wan segment",
            "global_control_normalization": "one full-length camera-frame driver is preprocessed before segmented inference",
            "human_residual_prevention": (
                "driver subject mask plus fail-closed lower-frame human guard; source face control suppressed"
                if args.source_mask_mode == "factored_guard"
                else "source and driver masks are unioned and source face control is suppressed"
            ),
            "degradation_aware_route": "raw Wan candidate is retained; alpha repair and temporal filtering are disabled",
        },
        "packages": {},
    }
    if driver_override is not None:
        override_manifest = driver_override / "manifest.json"
        metadata["inputs"]["driver_preprocess_override"] = {
            "path": str(driver_override),
            "manifest_sha256": _sha256(override_manifest),
        }
    for name in ("torch", "torchvision", "opencv-python", "sam-2"):
        try:
            metadata["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            metadata["packages"][name] = None
    _write_json(metadata_path, metadata)

    lease_path, lease = acquire_gpu_lease(selected["physical_index"])
    try:
        current, _, _ = query_gpus()
        select_gpu(current, selected["physical_index"], args.minimum_free_gpu_mib)
        metadata["gpu_lease"] = str(lease_path)
        _write_json(metadata_path, metadata)
        driver_preprocess = experiment / "driver-preprocess"
        source_preprocess = None
        source_command = None
        if args.source_mask_mode == "wan_union":
            source_preprocess = experiment / "source-preprocess"
            source_preprocess.mkdir()
            source_command = renderer.build_preprocess_command(
                inputs["real_source"], inputs["robot_reference"], source_preprocess
            )
        driver_command = None
        if driver_override is None:
            driver_preprocess.mkdir()
            driver_command = renderer.build_preprocess_command(
                inputs["action_driver"], inputs["robot_reference"], driver_preprocess
            )
        else:
            shutil.copytree(driver_override, driver_preprocess)
        metadata["commands"] = {"driver_preprocess": driver_command}
        if source_command is not None:
            metadata["commands"]["source_preprocess"] = source_command
        _write_json(metadata_path, metadata)
        if source_command is not None:
            _run(
                source_command,
                cwd=config.wan_repo,
                env=environment,
                log_path=experiment / "source-preprocess.log",
            )
        if driver_command is not None:
            _run(
                driver_command,
                cwd=config.wan_repo,
                env=environment,
                log_path=experiment / "driver-preprocess.log",
            )
        hybrid = experiment / "hybrid-preprocess"
        metadata["hybrid_control"] = _merge_factored_controls(
            inputs["real_source"],
            source_preprocess,
            driver_preprocess,
            hybrid,
            dilation_pixels=args.mask_dilation,
            human_guard_y=args.human_guard_y,
        )
        raw_output = experiment / "wan-raw.mp4"
        generate_command = renderer.build_generate_command(
            hybrid, raw_output, args.prompt, args.seed
        )
        metadata["commands"]["generate"] = generate_command
        _write_json(metadata_path, metadata)
        _run(generate_command, cwd=config.wan_repo, env=environment, log_path=experiment / "generate.log")
        raw_info = _video_info(raw_output)
        final_output = experiment / "robot-factored-refined.mp4"
        target_frames = int(source_info["frames"])
        if int(raw_info["frames"]) > target_frames:
            raise RuntimeError("Wan output is longer than the requested source")
        clone_frames = target_frames - int(raw_info["frames"])
        timing = f"setpts=N/({args.fps}*TB)"
        filter_value = (
            f"{timing},tpad=stop_mode=clone:stop_duration={clone_frames / args.fps:.9f}"
            if clone_frames
            else timing
        )
        _run(
            [
                str(Path(shutil.which("ffmpeg") or "ffmpeg").resolve()),
                "-y",
                "-v",
                "error",
                "-i",
                str(raw_output),
                "-vf",
                filter_value,
                "-frames:v",
                str(target_frames),
                "-r",
                str(args.fps),
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "12",
                "-preset",
                "slow",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(final_output),
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            log_path=experiment / "finalize.log",
        )
        final_info = _video_info(final_output)
        if int(final_info["frames"]) != target_frames or abs(float(final_info["fps"]) - args.fps) > 1e-6:
            raise RuntimeError(f"final video geometry is invalid: {final_info}")
        metadata.update(
            {
                "status": "succeeded",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "raw_output": str(raw_output),
                "raw_output_sha256": _sha256(raw_output),
                "raw_output_info": raw_info,
                "final_output": str(final_output),
                "final_output_sha256": _sha256(final_output),
                "final_output_info": final_info,
                "cloned_terminal_frames": clone_frames,
                "postprocessing": {
                    "alpha_repair": False,
                    "source_object_overwrite": False,
                    "temporal_filter": False,
                    "blur": False,
                },
            }
        )
        _write_json(metadata_path, metadata)
        print(f"EXPERIMENT={experiment}")
        print(f"VIDEO={final_output}")
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": repr(exc),
            }
        )
        _write_json(metadata_path, metadata)
        raise
    finally:
        import fcntl

        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        lease.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
