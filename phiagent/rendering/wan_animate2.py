"""Pinned Wan-Animate-2 source, checkpoint, and config validation."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Sequence

from phiagent.rendering.wan_animate import GPUInfo

WAN_ANIMATE2_COMMIT = "3ad2fef7d61d6200c9c653e0fe47be7616b323f3"
WAN_ANIMATE2_MODEL_ID = "Wan-AI/Wan2.2-Animate-2-14B"
WAN_ANIMATE2_MODEL_REVISION = "3c1a1ccd035b9997478d288040358891a06bf682"
WAN_ANIMATE2_MODELSCOPE_REVISION = "7053fd05166cdd99a49896364d01c06c281a9d69"

_COMMON_CONFIG_PATHS = {
    "../ckpts/videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth": (
        "videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth"
    ),
    "../ckpts/videomodel/Wan-AI/umt5-xxl": "videomodel/Wan-AI/umt5-xxl",
    "../ckpts/videomodel/Wan-AI/vae.pth": "videomodel/Wan-AI/vae.pth",
    "../ckpts/videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": (
        "videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    ),
    "../ckpts/videomodel/Wan-AI/xlm-roberta-large": (
        "videomodel/Wan-AI/xlm-roberta-large"
    ),
}
_BASE_MODEL = "wan_animate_2/wan_animate_2_bf16.safetensors"
_DISTILLED_MODEL = "wan_animate_2/wan_animate_2_bf16_distillation.safetensors"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_wan_animate2_gpus(
    gpus: Sequence[GPUInfo],
    requested_indices: Sequence[int],
    minimum_free_mib: int,
) -> tuple[GPUInfo, GPUInfo]:
    if minimum_free_mib <= 0:
        raise ValueError("minimum_free_mib must be positive")
    by_index = {gpu.physical_index: gpu for gpu in gpus}
    if requested_indices:
        if len(requested_indices) != 2 or len(set(requested_indices)) != 2:
            raise ValueError("Wan-Animate-2 requires exactly two distinct physical GPUs")
        missing = [index for index in requested_indices if index not in by_index]
        if missing:
            raise ValueError(f"requested physical GPUs were not reported: {missing}")
        selected = tuple(by_index[index] for index in requested_indices)
    else:
        eligible = sorted(
            (gpu for gpu in gpus if gpu.free_mib >= minimum_free_mib),
            key=lambda gpu: gpu.free_mib,
            reverse=True,
        )
        if len(eligible) < 2:
            raise ValueError(
                "Wan-Animate-2 requires two GPUs with at least "
                f"{minimum_free_mib} MiB free; found {len(eligible)}"
            )
        selected = tuple(eligible[:2])
    busy = [gpu for gpu in selected if gpu.free_mib < minimum_free_mib]
    if busy:
        summary = ", ".join(
            f"GPU {gpu.physical_index}: {gpu.free_mib} MiB free" for gpu in busy
        )
        raise ValueError(
            f"selected GPUs do not meet the {minimum_free_mib} MiB requirement ({summary})"
        )
    return selected[0], selected[1]


def verify_wan_animate2_source(repo: Path) -> str:
    resolved = repo.expanduser().resolve()
    required = (
        resolved / "LICENSE",
        resolved / "infer" / "wan_animate_2_demo.py",
        resolved / "infer" / "wan_animate_2.yaml",
        resolved / "pipelines" / "wan_animate_2_pipeline.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Wan-Animate-2 source is missing required files: {missing}")
    license_text = required[0].read_text(errors="replace")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("Wan-Animate-2 source does not contain the reviewed Apache license")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=resolved,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or commit != WAN_ANIMATE2_COMMIT:
        raise ValueError(
            f"Wan-Animate-2 source is {commit or 'unreadable'}, "
            f"expected {WAN_ANIMATE2_COMMIT}"
        )
    return commit


def verify_wan_animate2_checkpoint(
    checkpoint_dir: Path, *, distilled: bool = False
) -> dict[str, str]:
    resolved = checkpoint_dir.expanduser().resolve()
    marker = resolved / ".phiagent-model-revision"
    accepted_revisions = {
        WAN_ANIMATE2_MODEL_REVISION,
        f"modelscope:{WAN_ANIMATE2_MODELSCOPE_REVISION}",
    }
    actual_revision = marker.read_text().strip() if marker.is_file() else ""
    if actual_revision not in accepted_revisions:
        raise ValueError(
            "Wan-Animate-2 checkpoint revision marker is missing or incorrect; "
            f"expected one of {sorted(accepted_revisions)}"
        )
    hashes: dict[str, str] = {}
    required = (*_COMMON_CONFIG_PATHS.values(), _DISTILLED_MODEL if distilled else _BASE_MODEL)
    for relative in required:
        path = resolved / relative
        if not path.exists():
            raise ValueError(f"Wan-Animate-2 checkpoint asset is missing: {path}")
        if path.is_file():
            if path.stat().st_size == 0:
                raise ValueError(f"Wan-Animate-2 checkpoint asset is empty: {path}")
            hashes[relative] = file_sha256(path)
    return hashes


def write_runtime_config(
    repo: Path,
    checkpoint_dir: Path,
    output: Path,
    *,
    distilled: bool = False,
) -> None:
    template_name = (
        "wan_animate_2_distillation.yaml" if distilled else "wan_animate_2.yaml"
    )
    template = repo.expanduser().resolve() / "infer" / template_name
    text = template.read_text()
    checkpoint = checkpoint_dir.expanduser().resolve()
    config_paths = {
        **_COMMON_CONFIG_PATHS,
        (
            "../ckpts/wan_animate_2/wan_animate_2_bf16_distillation.safetensors"
            if distilled
            else "../ckpts/wan_animate_2/wan_animate_2_bf16.safetensors"
        ): _DISTILLED_MODEL if distilled else _BASE_MODEL,
    }
    for configured, relative in config_paths.items():
        if text.count(configured) != 1:
            raise ValueError(f"unexpected Wan-Animate-2 config occurrence for {configured}")
        text = text.replace(configured, str(checkpoint / relative))
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(text)
