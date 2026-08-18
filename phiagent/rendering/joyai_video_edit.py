"""Pinned, dependency-free contracts for bounded and full-stream JoyAI edits.

The released model is deliberately treated as a visual proposal generator.
Bounded repair mode restores flowers, background, and endpoints exactly.  Full
stream mode instead lets one uninterrupted causal session reproduce source
object motion; its output must pass an explicit non-freezing/contact audit and
never acquires physical-evidence authority from appearance alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from math import ceil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


JOYAI_REPOSITORY = "https://github.com/jd-opensource/JoyAI-Video-Edit"
JOYAI_REPOSITORY_REVISION = "3478e4b8c9a79fe935157d1d477cd3e57bb41f1f"
JOYAI_SOURCE_REVISION_MARKER = ".phiagent-source-revision"
JOYAI_MODEL_ID = "jdopensource/JoyAI-Video-Edit"
JOYAI_MODEL_REVISION = "e14d9ac50d4ad8e9f91b655bfab270c02a43923b"
JOYAI_MODELSCOPE_MODEL_ID = "JD-OpenSource/JoyAI-Video-Edit"
JOYAI_MODELSCOPE_MODEL_REVISION = "1566385ac62baea53acbc25239668188a1e85c73"
JOYAI_TEXT_ENCODER_ID = "XiaomiMiMo/MiMo-VL-7B-RL-2508"
JOYAI_TEXT_ENCODER_REVISION = "4bfb270765825d2fa059011deb4c96fdd579be6f"
JOYAI_TEXT_ENCODER_MODELSCOPE_REVISION = "233aba9632b84d30d10fa00f856a83fe11401ef3"

JOYAI_DIT_RELATIVE_PATH = Path("JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth")
JOYAI_VAE_CONFIG_RELATIVE_PATH = Path("JoyAI-Video-Edit/vae/config.json")
JOYAI_VAE_WEIGHTS_RELATIVE_PATH = Path(
    "JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors"
)
JOYAI_TEXT_ENCODER_RELATIVE_PATH = Path("MiMo-VL-7B-RL-2508")

# Sizes are part of the released Hugging Face snapshot contract.  Exact checks
# catch LFS pointer files and truncated downloads before a GPU process starts.
JOYAI_DIT_BYTES = 32_527_662_903
JOYAI_VAE_BYTES = 1_534_679_470
JOYAI_MIN_TEXT_ENCODER_BYTES = 16_000_000_000
JOYAI_LARGE_FILE_CONTRACT = {
    JOYAI_DIT_RELATIVE_PATH: (
        JOYAI_DIT_BYTES,
        "b3904b6fda53d13b230918bb616f322d12cfb2337b0e8d9dc203cdabc36605ba",
    ),
    JOYAI_VAE_WEIGHTS_RELATIVE_PATH: (
        JOYAI_VAE_BYTES,
        "150315748d7c3307cdae2819ee651b32d58385668ca0c4db3d3dcd6e63b77e86",
    ),
    Path("MiMo-VL-7B-RL-2508/model-00001-of-00004.safetensors"): (
        4_612_695_408,
        "f93e07524d843d63c080eef4fd43d7c5b98ac7a17e8c56e48edfe89297d6bff3",
    ),
    Path("MiMo-VL-7B-RL-2508/model-00002-of-00004.safetensors"): (
        4_937_303_136,
        "9715248b5ff4357d2deb23669c83131f35934c981e88f648a9769976e533412e",
    ),
    Path("MiMo-VL-7B-RL-2508/model-00003-of-00004.safetensors"): (
        4_982_109_888,
        "5b8db223f443f8ed88b65017c65aaa9e201e386b942676e75bf1651202e2181f",
    ),
    Path("MiMo-VL-7B-RL-2508/model-00004-of-00004.safetensors"): (
        2_080_418_376,
        "8bcbeaac2b402096c0629435395003fa66eb14dfba682d3af02b4398001f5b0f",
    ),
}

SOURCE_FRAME = "camera:source_native_1280x720"
JOYAI_FRAME = "camera:joyai_center_crop_1248x720"
TIMELINE_FRAME = "absolute_frame_index:full_source_660"


class JoyAIPreflightError(RuntimeError):
    """Raised before inference when the pinned deployment contract is broken."""


@dataclass(frozen=True)
class JoyAIWindow:
    """Inclusive full-video frame range aligned to JoyAI's causal chunks."""

    start_frame: int
    end_frame: int
    seam_frame: int

    def validate(self) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("JoyAI window must be a non-negative inclusive range")
        if not self.start_frame <= self.seam_frame <= self.end_frame:
            raise ValueError("seam_frame must lie inside the JoyAI window")
        if (self.frame_count - 1) % 8:
            raise ValueError("JoyAI window length must satisfy frame_count = 1 + 8n")

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


DEFAULT_FLOWER_WINDOWS = (
    JoyAIWindow(start_frame=463, end_frame=495, seam_frame=479),
    JoyAIWindow(start_frame=543, end_frame=575, seam_frame=559),
)


@dataclass(frozen=True)
class HeldToolContract:
    """Timeline and topology contract for a rigid tool carried through an edit.

    The contract deliberately records a human-review requirement.  A text
    prompt or a single reference image cannot prove ring occupancy, attachment,
    or temporal persistence in the generated frames.
    """

    name: str
    source_start_frame: int
    source_end_frame: int
    holder: str
    topology: tuple[str, ...]
    required_review_frames: tuple[int, ...]

    def validate(self, *, total_frames: int = 660) -> None:
        if not self.name.strip() or not self.holder.strip():
            raise ValueError("held tool name and holder must be non-empty")
        if not 0 <= self.source_start_frame <= self.source_end_frame < total_frames:
            raise ValueError("held tool interval must lie in the source timeline")
        if not self.topology:
            raise ValueError("held tool topology invariants must be non-empty")
        if not self.required_review_frames:
            raise ValueError("held tool requires native-resolution review frames")
        if tuple(sorted(set(self.required_review_frames))) != self.required_review_frames:
            raise ValueError("held tool review frames must be sorted and unique")
        if any(
            frame < self.source_start_frame or frame > self.source_end_frame
            for frame in self.required_review_frames
        ):
            raise ValueError("held tool review frame lies outside its source interval")

    def to_manifest(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "review_authority": "native_resolution_human_veto",
            "automatic_promotion": False,
            "physical_evidence": False,
        }


DEFAULT_SCISSORS_CONTRACT = HeldToolContract(
    name="black-handled stainless-steel florist scissors",
    source_start_frame=398,
    source_end_frame=447,
    holder="robot_right_hand",
    topology=(
        "exactly_one_rigid_scissors",
        "two_handle_rings",
        "two_blades_joined_at_one_pivot",
        "robot_fingers_remain_inside_handle_rings",
        "pivot_remains_attached_to_robot_hand",
        "no_floating_disappearance_or_finger_tool_merge",
    ),
    required_review_frames=(398, 402, 406, 410, 414, 417, 423, 431, 439, 447),
)


def flower_repair_prompt() -> str:
    """A source-fidelity prompt; deterministic locks remain the real guardrail."""

    return (
        "Preserve every flower, stem, vase, background pixel, camera motion, "
        "timing, and robot identity. Refine only the visible robot hand and arm "
        "inside the supplied repair support. Keep one articulated five-finger "
        "hand with stable topology and color; remove finger deformation, motion "
        "smear, double edges, and local structural artifacts. Preserve the exact "
        "reach, grasp, insertion, release, stem contact, and flower response "
        "motion. Do not add, remove, recolor, freeze, or move flowers or stems."
    )


def flower_full_stream_prompt() -> str:
    """Prompt a causal full-stream appearance edit without freezing flowers."""

    return (
        "Replace only the florist's human appearance with the exact silver humanoid "
        "robot shown in the reference image. Preserve the source video timeline, "
        "camera, background, table, vase, every flower and every stem. Each flower "
        "and stem must follow its original frame-by-frame source trajectory: while "
        "grasped it moves continuously with the contacting robot hand without lag, "
        "after release it remains supported by the vase or table, and it never "
        "freezes, floats, hangs unsupported, teleports, or moves before contact. "
        "Retarget the florist's complete torso, arm, wrist, hand, grasp, insertion, "
        "and release motion to one coherent robot with two articulated five-finger "
        "hands. During the source trimming action, preserve the one black-handled "
        "stainless-steel florist scissors as a distinct rigid tool held by the robot's "
        "right hand: the fingers remain closed through its handles, the pivot remains "
        "attached to the hand, and its two blades follow the exact source trajectory "
        "and opening angle. The held scissors must never disappear, float, become a "
        "background object, merge into a flower, or turn into fingers. Preserve flower "
        "identity, count, color, geometry, depth ordering, occlusion, motion blur, and "
        "physical response. Keep every robot part boundary, metallic material, texture, "
        "color, and highlight temporally stable after motion alignment. Adjacent frames "
        "must not show local shape popping, crawling texture, shimmering patches, "
        "duplicate contours, exposure pulsing, color pulsing, or a visual reset at a "
        "causal chunk boundary. Preserve natural source motion instead of freezing or "
        "smoothing away hand, tool, flower, or stem motion. Do not invent flower motion "
        "before a causal interaction, and do not add or remove flowers, limbs, people, "
        "tools, text, cuts, camera motion, or scene changes."
    )


def causal_padded_frame_count(frame_count: int, *, chunk_frames: int = 8) -> int:
    """Return the shortest JoyAI stream length ``1 + n * chunk_frames``."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be positive")
    return 1 + ceil(max(0, frame_count - 1) / chunk_frames) * chunk_frames


def causal_tail_padding_frames(frame_count: int, *, chunk_frames: int = 8) -> int:
    """Return how many cloned tail frames are required for a complete chunk."""

    return causal_padded_frame_count(frame_count, chunk_frames=chunk_frames) - frame_count


@dataclass(frozen=True)
class JoyAIFlowerEditContract:
    """Reproducible source-anchored challenger configuration."""

    windows: tuple[JoyAIWindow, ...] = DEFAULT_FLOWER_WINDOWS
    source_width: int = 1280
    source_height: int = 720
    model_width: int = 1248
    model_height: int = 720
    crop_left: int = 16
    crop_top: int = 0
    transform_kind: str = "integer_center_crop_no_rescale"
    resized_width: int | None = None
    fps: int = 24
    seed: int = 42
    num_inference_steps: int = 2
    output_quality: int = 95
    prompt: str = flower_repair_prompt()

    def validate(self) -> None:
        if not self.windows:
            raise ValueError("at least one JoyAI repair window is required")
        for window in self.windows:
            window.validate()
        ordered = sorted(self.windows, key=lambda row: row.start_frame)
        if list(self.windows) != ordered:
            raise ValueError("JoyAI windows must be ordered by start frame")
        for previous, following in zip(ordered, ordered[1:]):
            if previous.end_frame >= following.start_frame:
                raise ValueError("JoyAI windows must not overlap")
        if min(self.source_width, self.source_height, self.model_width, self.model_height) <= 0:
            raise ValueError("video dimensions must be positive")
        if self.crop_left < 0 or self.crop_top < 0:
            raise ValueError("crop offsets must be non-negative")
        if self.transform_kind == "integer_center_crop_no_rescale":
            if self.resized_width is not None:
                raise ValueError("integer crop must not declare a resized width")
            if self.crop_left + self.model_width > self.source_width:
                raise ValueError("horizontal crop leaves the source frame")
            if self.crop_top + self.model_height > self.source_height:
                raise ValueError("vertical crop leaves the source frame")
        elif self.transform_kind == "isotropic_fit_height_then_center_crop":
            if self.resized_width is None or self.resized_width < self.model_width:
                raise ValueError("fit-height transform requires resized_width >= model_width")
            if self.crop_left + self.model_width > self.resized_width or self.crop_top != 0:
                raise ValueError("model crop leaves the explicitly resized frame")
        else:
            raise ValueError(f"unsupported source-to-JoyAI transform: {self.transform_kind}")
        if self.fps <= 0 or self.num_inference_steps <= 0:
            raise ValueError("fps and inference steps must be positive")
        if not 1 <= self.output_quality <= 100:
            raise ValueError("output quality must be in [1, 100]")
        if not self.prompt.strip():
            raise ValueError("prompt must be non-empty")

    def to_manifest(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": "1.0.0",
            "model_authority": "proposal_only",
            "promotion_authority": (
                "deterministic object/background/endpoint validators plus "
                "native-resolution human veto"
            ),
            "model": {
                "repository": JOYAI_REPOSITORY,
                "repository_revision": JOYAI_REPOSITORY_REVISION,
                "weights": JOYAI_MODEL_ID,
                "weights_revision": JOYAI_MODEL_REVISION,
                "weights_modelscope_mirror": JOYAI_MODELSCOPE_MODEL_ID,
                "weights_modelscope_revision": JOYAI_MODELSCOPE_MODEL_REVISION,
                "text_encoder": JOYAI_TEXT_ENCODER_ID,
                "text_encoder_revision": JOYAI_TEXT_ENCODER_REVISION,
                "text_encoder_modelscope_revision": (
                    JOYAI_TEXT_ENCODER_MODELSCOPE_REVISION
                ),
            },
            "coordinate_frames": {
                "source": f"camera:source_native_{self.source_width}x{self.source_height}",
                "joyai_input": JOYAI_FRAME,
                "timeline": TIMELINE_FRAME,
                "source_to_joyai": {
                    "kind": self.transform_kind,
                    "x_joyai": (
                        f"x_source * ({self.resized_width}/{self.source_width}) - {self.crop_left}"
                        if self.resized_width is not None
                        else f"x_source - {self.crop_left}"
                    ),
                    "y_joyai": (
                        f"y_source * ({self.model_height}/{self.source_height})"
                        if self.resized_width is not None
                        else f"y_source - {self.crop_top}"
                    ),
                    "crop_left_px": self.crop_left,
                    "crop_top_px": self.crop_top,
                    "resized_width_px": self.resized_width,
                    "resized_height_px": self.model_height if self.resized_width is not None else None,
                },
            },
            "config": {
                **asdict(self),
                "windows": [asdict(window) for window in self.windows],
            },
            "immutable_state": [
                "flowers",
                "per_stem_pixels_and_motion",
                "vase",
                "background",
                "camera_motion",
                "window_endpoints",
                "all_frames_outside_edit_support",
            ],
            "claim_scope": "perceptually plausible synthetic video data",
            "physical_evidence": False,
        }


def validate_upstream_checkout(repository: Path) -> dict[str, Any]:
    """Require the audited JoyAI source revision and Apache-2.0 license."""

    root = repository.expanduser().resolve()
    if not root.is_dir():
        raise JoyAIPreflightError(f"JoyAI repository is missing: {root}")
    source_kind = "git"
    if (root / ".git").exists():
        try:
            revision = subprocess.run(
                ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise JoyAIPreflightError(f"could not inspect JoyAI checkout: {exc}") from exc
    else:
        marker = root / JOYAI_SOURCE_REVISION_MARKER
        if not marker.is_file():
            raise JoyAIPreflightError(
                "JoyAI archive checkout is missing its pinned source revision marker: "
                f"{marker}"
            )
        revision = marker.read_text(encoding="utf-8").strip()
        source_kind = "revision-marked-archive"
    if revision != JOYAI_REPOSITORY_REVISION:
        raise JoyAIPreflightError(
            f"JoyAI revision mismatch: observed {revision}, expected {JOYAI_REPOSITORY_REVISION}"
        )
    license_path = root / "LICENSE"
    if not license_path.is_file() or "Apache License" not in license_path.read_text(
        encoding="utf-8", errors="replace"
    ):
        raise JoyAIPreflightError("JoyAI Apache license file is missing or unexpected")
    server = root / "deploy/xvideo/serving/serve_joyomni_streaming.py"
    if not server.is_file():
        raise JoyAIPreflightError(f"JoyAI server entrypoint is missing: {server}")
    return {
        "repository": str(root),
        "revision": revision,
        "source_kind": source_kind,
        "license": "Apache-2.0",
        "server_entrypoint": str(server),
    }


def _tree_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def validate_checkpoint_layout(
    checkpoint_root: Path, *, verify_large_hashes: bool = False
) -> dict[str, Any]:
    """Fail before loading CUDA when released weights are absent/truncated."""

    root = checkpoint_root.expanduser().resolve()
    dit = root / JOYAI_DIT_RELATIVE_PATH
    vae_config = root / JOYAI_VAE_CONFIG_RELATIVE_PATH
    vae = root / JOYAI_VAE_WEIGHTS_RELATIVE_PATH
    text_encoder = root / JOYAI_TEXT_ENCODER_RELATIVE_PATH
    model_marker = dit.parents[1] / ".phiagent-model-revision"
    text_encoder_marker = text_encoder / ".phiagent-model-revision"
    text_encoder_index = text_encoder / "model.safetensors.index.json"
    required = (
        vae_config,
        text_encoder,
        model_marker,
        text_encoder_marker,
        text_encoder_index,
        *(root / relative for relative in JOYAI_LARGE_FILE_CONTRACT),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise JoyAIPreflightError(f"JoyAI checkpoint layout is incomplete: {missing}")
    observed_revisions = {
        "model": model_marker.read_text(encoding="utf-8").strip(),
        "text_encoder": text_encoder_marker.read_text(encoding="utf-8").strip(),
    }
    accepted_revisions = {
        "model": {
            JOYAI_MODEL_REVISION,
            f"modelscope:{JOYAI_MODELSCOPE_MODEL_REVISION}",
        },
        "text_encoder": {
            JOYAI_TEXT_ENCODER_REVISION,
            f"modelscope:{JOYAI_TEXT_ENCODER_MODELSCOPE_REVISION}",
        },
    }
    rejected_revisions = {
        label: revision
        for label, revision in observed_revisions.items()
        if revision not in accepted_revisions[label]
    }
    if rejected_revisions:
        raise JoyAIPreflightError(
            "JoyAI checkpoint revision markers do not match the pinned release: "
            f"{rejected_revisions}; accepted markers are {accepted_revisions}"
        )
    observed_sizes = {
        str(relative): (root / relative).stat().st_size
        for relative in JOYAI_LARGE_FILE_CONTRACT
    }
    expected_sizes = {
        str(relative): expected_size
        for relative, (expected_size, _) in JOYAI_LARGE_FILE_CONTRACT.items()
    }
    if observed_sizes != expected_sizes:
        raise JoyAIPreflightError(
            "JoyAI released weight size mismatch (LFS pointer/truncation likely): "
            f"{observed_sizes} != {expected_sizes}"
        )
    text_encoder_bytes = _tree_size(text_encoder)
    if text_encoder_bytes < JOYAI_MIN_TEXT_ENCODER_BYTES:
        raise JoyAIPreflightError(
            "MiMo text encoder is incomplete: "
            f"{text_encoder_bytes} bytes < {JOYAI_MIN_TEXT_ENCODER_BYTES}"
        )
    hashes: dict[str, str] = {}
    if verify_large_hashes:
        hashes = {
            str(relative): sha256_file(root / relative)
            for relative in JOYAI_LARGE_FILE_CONTRACT
        }
        expected_hashes = {
            str(relative): expected_hash
            for relative, (_, expected_hash) in JOYAI_LARGE_FILE_CONTRACT.items()
        }
        if hashes != expected_hashes:
            raise JoyAIPreflightError(
                f"JoyAI released weight hash mismatch: {hashes} != {expected_hashes}"
            )
    return {
        "checkpoint_root": str(root),
        "dit": {"path": str(dit), "bytes": dit.stat().st_size},
        "vae": {"path": str(vae), "bytes": vae.stat().st_size},
        "vae_config": {"path": str(vae_config), "sha256": sha256_file(vae_config)},
        "text_encoder": {"path": str(text_encoder), "tree_bytes": text_encoder_bytes},
        "large_file_hashes": hashes,
        "model_revision": JOYAI_MODEL_REVISION,
        "text_encoder_revision": JOYAI_TEXT_ENCODER_REVISION,
        "observed_revision_markers": observed_revisions,
    }


def build_server_argv(
    *,
    python_executable: Path,
    repository: Path,
    checkpoint_root: Path,
    record_dir: Path,
    host: str,
    port: int,
) -> tuple[str, ...]:
    """Build the official server command with A800-safe BF16 placement."""

    root = repository.expanduser().resolve()
    checkpoints = checkpoint_root.expanduser().resolve()
    return (
        os.path.abspath(python_executable.expanduser()),
        str(root / "deploy/xvideo/serving/serve_joyomni_streaming.py"),
        "--dit-ckpt",
        str(checkpoints / JOYAI_DIT_RELATIVE_PATH),
        "--vae-ckpt",
        str((checkpoints / JOYAI_VAE_CONFIG_RELATIVE_PATH).parent),
        "--text-encoder-ckpt",
        str(checkpoints / JOYAI_TEXT_ENCODER_RELATIVE_PATH),
        "--record-dir",
        str(record_dir.expanduser().resolve()),
        "--device",
        "cuda:0",
        "--vae-encode-device",
        "cuda:1",
        "--vae-decode-device",
        "cuda:1",
        "--vae-pseudo-device",
        "cuda:1",
        "--postprocess-device",
        "cuda:1",
        "--host",
        host,
        "--port",
        str(port),
        "--height",
        "720",
        "--width",
        "1248",
        "--num-inference-steps",
        "2",
        "--seed",
        "42",
        "--fps",
        "24",
        "--no-use-pe",
        "--no-online-gate",
        "--uplink-codec",
        "mjpeg",
        "--downlink-codec",
        "mjpeg",
        "--output-quality",
        "95",
        "--profile-timings",
        "--kv-reset-frames",
        "0",
        "--max-inflight-chunks",
        "0",
        "--push-frame-timeout-s",
        "600",
        "--inference-lock-timeout-s",
        "600",
        "--session-close-timeout-s",
        "60",
        "--record-bitrate",
        "20000000",
        "--record-segment-seconds",
        "300",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def windows_from_ranges(ranges: Sequence[Sequence[int]]) -> tuple[JoyAIWindow, ...]:
    """Parse ``start,end,seam`` triples for CLIs without importing heavy deps."""

    windows = tuple(JoyAIWindow(*(int(value) for value in row)) for row in ranges)
    for window in windows:
        window.validate()
    return windows
