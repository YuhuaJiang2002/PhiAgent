"""SC3-inspired action-intent harness for the frozen JoyAI video editor.

JoyAI-Video-Edit is an instruction-guided video editor, not an action-conditioned
world model.  This harness therefore requires an external RGB action carrier whose
motion and timing are authoritative.  JoyAI may refine appearance, lighting, and
local realism, while a separate inverse visual evaluator checks whether the edited
video still expresses the requested action.

The module has no CUDA, torch, checkpoint, or simulator dependency at import time.
GPU selection and model loading remain owned by the pinned JoyAI server launcher.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shlex
import shutil
import socket
import string
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from phiagent.acwm.schema import ACWMActionCondition
from phiagent.rendering.joyai_video_edit import (
    JOYAI_FRAME,
    JOYAI_MODEL_REVISION,
    JOYAI_REPOSITORY_REVISION,
    JOYAI_TEXT_ENCODER_REVISION,
    causal_padded_frame_count,
    sha256_file,
    write_json,
)


class JoyAISC3PreflightError(RuntimeError):
    """Raised before inference when the action-rendering contract is unsafe."""


_SCORE_FIELDS = (
    "action_adherence",
    "embodiment_consistency",
    "object_interaction",
    "temporal_consistency",
    "background_consistency",
)
_EVALUATOR_PLACEHOLDERS = {
    "candidate",
    "condition",
    "evidence_root",
    "first_frame",
    "metadata",
    "project_root",
    "python",
    "seed",
    "source",
}


def _absolute_without_resolving(path: Path) -> Path:
    """Keep virtualenv interpreter symlinks instead of resolving their targets."""

    return Path(os.path.abspath(path.expanduser()))


def _resolve_path(value: str, *, base_dir: Path, executable: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return _absolute_without_resolving(path) if executable else path.resolve()


def _optional_path(value: object, *, base_dir: Path, executable: bool = False) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("optional path values must be non-empty strings or null")
    return _resolve_path(value, base_dir=base_dir, executable=executable)


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise JoyAISC3PreflightError(f"{label} is missing or empty: {path}")
    return path


def _parse_fraction(value: str) -> Fraction:
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise JoyAISC3PreflightError(f"invalid video frame rate: {value!r}") from exc
    if result <= 0:
        raise JoyAISC3PreflightError(f"video frame rate must be positive: {value!r}")
    return result


def resampled_frame_count(
    frame_count: int,
    *,
    source_fps_numerator: int,
    source_fps_denominator: int,
    target_fps: int,
) -> int:
    """Round a positive frame-duration ratio to the nearest target-frame count."""

    if min(
        frame_count,
        source_fps_numerator,
        source_fps_denominator,
        target_fps,
    ) <= 0:
        raise ValueError("frame count and source/target FPS values must be positive")
    exact = Fraction(
        frame_count * target_fps * source_fps_denominator,
        source_fps_numerator,
    )
    return int(exact + Fraction(1, 2))


@dataclass(frozen=True)
class VideoStream:
    """Exact metadata needed to bind a carrier video to an action timeline."""

    width: int
    height: int
    fps_numerator: int
    fps_denominator: int
    frame_count: int
    duration_seconds: float

    @property
    def fps(self) -> float:
        return self.fps_numerator / self.fps_denominator

    @classmethod
    def from_ffprobe(cls, payload: Mapping[str, Any]) -> "VideoStream":
        streams = payload.get("streams")
        if not isinstance(streams, list) or len(streams) != 1:
            raise JoyAISC3PreflightError("action carrier must contain exactly one video stream")
        row = streams[0]
        if not isinstance(row, Mapping):
            raise JoyAISC3PreflightError("ffprobe video stream payload is malformed")
        rate = _parse_fraction(str(row.get("avg_frame_rate", "")))
        frame_value = row.get("nb_read_frames") or row.get("nb_frames")
        try:
            width = int(row["width"])
            height = int(row["height"])
            frame_count = int(frame_value)
        except (KeyError, TypeError, ValueError) as exc:
            raise JoyAISC3PreflightError(
                "ffprobe did not report width, height, and frame count"
            ) from exc
        format_row = payload.get("format")
        duration_value = row.get("duration")
        if duration_value is None and isinstance(format_row, Mapping):
            duration_value = format_row.get("duration")
        try:
            duration = (
                float(duration_value) if duration_value is not None else frame_count / float(rate)
            )
        except (TypeError, ValueError) as exc:
            raise JoyAISC3PreflightError("ffprobe video duration is malformed") from exc
        if min(width, height, frame_count) <= 0 or not math.isfinite(duration) or duration <= 0:
            raise JoyAISC3PreflightError("carrier video metadata must be finite and positive")
        return cls(
            width=width,
            height=height,
            fps_numerator=rate.numerator,
            fps_denominator=rate.denominator,
            frame_count=frame_count,
            duration_seconds=duration,
        )


@dataclass(frozen=True)
class FitBlurPadTransform:
    """An explicit isotropic transform over a non-authoritative blurred background."""

    source_frame: str
    target_frame: str
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    scale_numerator: int
    scale_denominator: int
    resized_width: int
    resized_height: int
    offset_x: int
    offset_y: int
    background_blur_sigma: float

    @classmethod
    def create(
        cls,
        *,
        source_frame: str,
        target_frame: str,
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
        background_blur_sigma: float = 30.0,
    ) -> "FitBlurPadTransform":
        if not source_frame.startswith("camera:") or not target_frame.startswith("camera:"):
            raise ValueError("JoyAI image transforms require named camera frames")
        if min(source_width, source_height, target_width, target_height) <= 0:
            raise ValueError("image dimensions must be positive")
        if not math.isfinite(background_blur_sigma) or background_blur_sigma <= 0:
            raise ValueError("background_blur_sigma must be finite and positive")
        scale = min(Fraction(target_width, source_width), Fraction(target_height, source_height))
        width_fraction = source_width * scale
        height_fraction = source_height * scale
        if width_fraction.denominator != 1 or height_fraction.denominator != 1:
            raise ValueError(
                "fit transform must produce integer resized dimensions; "
                "choose a target with an exact rational fit"
            )
        resized_width = int(width_fraction)
        resized_height = int(height_fraction)
        horizontal = target_width - resized_width
        vertical = target_height - resized_height
        if horizontal < 0 or vertical < 0 or horizontal % 2 or vertical % 2:
            raise ValueError("fit transform requires non-negative even center padding")
        return cls(
            source_frame=source_frame,
            target_frame=target_frame,
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
            scale_numerator=scale.numerator,
            scale_denominator=scale.denominator,
            resized_width=resized_width,
            resized_height=resized_height,
            offset_x=horizontal // 2,
            offset_y=vertical // 2,
            background_blur_sigma=background_blur_sigma,
        )

    @property
    def scale(self) -> float:
        return self.scale_numerator / self.scale_denominator

    def forward_xy(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def inverse_xy(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale

    def prepare_filter(
        self,
        *,
        tail_padding_frames: int,
        output_fps: float,
        resample_fps: float | None = None,
    ) -> str:
        if tail_padding_frames < 0 or output_fps <= 0:
            raise ValueError("tail padding must be non-negative and fps must be positive")
        if resample_fps is not None and resample_fps <= 0:
            raise ValueError("resample_fps must be positive when provided")
        has_padding = bool(self.offset_x or self.offset_y)
        if has_padding:
            filter_graph = (
                "split=2[bg][fg];"
                f"[bg]scale={self.target_width}:{self.target_height}:"
                "force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={self.target_width}:{self.target_height},"
                f"gblur=sigma={self.background_blur_sigma:g}[bg];"
                f"[fg]scale={self.resized_width}:{self.resized_height}:"
                "flags=lanczos[fg];"
                f"[bg][fg]overlay={self.offset_x}:{self.offset_y}"
            )
        else:
            filter_graph = (
                f"scale={self.resized_width}:{self.resized_height}:flags=lanczos"
            )
        if resample_fps is not None:
            # Some FFmpeg versions drop the final frame duration after overlay.
            # One cloned source-rate frame preserves the closed source interval;
            # the explicit output frame limit removes any excess after resampling.
            filter_graph += ",tpad=stop_mode=clone:stop=1"
            filter_graph += f",fps=fps={resample_fps:g}:round=near"
        if tail_padding_frames:
            filter_graph += (
                f",tpad=stop_mode=clone:stop={tail_padding_frames}"
            )
        return filter_graph

    def restore_filter(self) -> str:
        return (
            f"crop={self.resized_width}:{self.resized_height}:"
            f"{self.offset_x}:{self.offset_y},"
            f"scale={self.source_width}:{self.source_height}:flags=lanczos"
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "kind": "isotropic_fit_over_blurred_cover_background",
            "scale": self.scale,
            "padding_background": {
                "coordinate_authority": False,
                "kind": "aspect_preserving_cover_crop_gaussian_blur",
                "gaussian_sigma": self.background_blur_sigma,
            },
            "forward_xy": (
                f"{self.target_frame} = {self.source_frame} * "
                f"({self.scale_numerator}/{self.scale_denominator}) + "
                f"({self.offset_x}, {self.offset_y})"
            ),
            "inverse_xy": (
                f"{self.source_frame} = ({self.target_frame} - "
                f"({self.offset_x}, {self.offset_y})) * "
                f"({self.scale_denominator}/{self.scale_numerator})"
            ),
        }


@dataclass(frozen=True)
class CarrierContract:
    """Provenance and authority boundary for an action-conditioned RGB carrier."""

    video: Path
    coordinate_frame: str
    generator: str
    generator_revision: str
    motion_authority: str
    physical_evidence: bool = False

    def validate(self, *, require_files: bool = True) -> None:
        if not self.coordinate_frame.startswith("camera:"):
            raise ValueError("carrier coordinate frame must be explicitly camera-relative")
        if not self.generator.strip() or not self.generator_revision.strip():
            raise ValueError("carrier generator and revision must be non-empty")
        if not self.motion_authority.strip():
            raise ValueError("carrier motion authority must be non-empty")
        if self.physical_evidence:
            raise ValueError("a generated RGB carrier cannot be declared physical evidence")
        if require_files:
            _require_file(self.video, "action carrier video")


@dataclass(frozen=True)
class ConsistencyThresholds:
    """Automatic gates used after inverse visual action recovery."""

    action_adherence: float = 0.75
    embodiment_consistency: float = 0.75
    object_interaction: float = 0.75
    temporal_consistency: float = 0.75
    background_consistency: float = 0.75

    def __post_init__(self) -> None:
        for field in _SCORE_FIELDS:
            value = getattr(self, field)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field} threshold must be finite and in [0, 1]")


@dataclass(frozen=True)
class CandidateScore:
    """Validated inverse-consistency result for one whole-stream JoyAI seed."""

    seed: int
    action_adherence: float
    embodiment_consistency: float
    object_interaction: float
    temporal_consistency: float
    background_consistency: float
    hard_gates_passed: bool
    human_review_passed: bool | None
    evaluator: str
    evidence: str | None = None

    def __post_init__(self) -> None:
        if self.seed < 0 or not self.evaluator.strip():
            raise ValueError("candidate seed must be non-negative and evaluator must be named")
        for field in _SCORE_FIELDS:
            value = getattr(self, field)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field} score must be finite and in [0, 1]")
        if self.human_review_passed not in {True, False, None}:
            raise ValueError("human_review_passed must be true, false, or null")

    @property
    def inverse_action_error(self) -> float:
        return 1.0 - self.action_adherence

    @property
    def mean_score(self) -> float:
        return sum(getattr(self, field) for field in _SCORE_FIELDS) / len(_SCORE_FIELDS)

    def constraint_margin(self, thresholds: ConsistencyThresholds) -> float:
        margin = min(getattr(self, field) - getattr(thresholds, field) for field in _SCORE_FIELDS)
        return margin if self.hard_gates_passed else min(margin, -1.0)

    def automatic_pass(self, thresholds: ConsistencyThresholds) -> bool:
        return self.hard_gates_passed and all(
            getattr(self, field) >= getattr(thresholds, field) for field in _SCORE_FIELDS
        )

    def visual_selection_pass(self, thresholds: ConsistencyThresholds) -> bool:
        return self.automatic_pass(thresholds) and self.human_review_passed is not False

    def promotion_pass(self, thresholds: ConsistencyThresholds) -> bool:
        return self.automatic_pass(thresholds) and self.human_review_passed is True

    @classmethod
    def from_payload(cls, seed: int, payload: Mapping[str, Any]) -> "CandidateScore":
        human = payload.get("human_review_passed")
        if human not in {True, False, None}:
            raise ValueError("evaluator returned an invalid human_review_passed value")
        hard_gates = payload.get("hard_gates_passed")
        if not isinstance(hard_gates, bool):
            raise ValueError("evaluator must return a boolean hard_gates_passed value")
        missing = set(_SCORE_FIELDS) - payload.keys()
        if missing:
            raise ValueError(f"evaluator score is missing fields: {sorted(missing)}")
        return cls(
            seed=seed,
            **{field: float(payload[field]) for field in _SCORE_FIELDS},
            hard_gates_passed=hard_gates,
            human_review_passed=human,
            evaluator=str(payload.get("evaluator", "")),
            evidence=str(payload["evidence"]) if payload.get("evidence") else None,
        )


def select_consistent_candidate(
    scores: Sequence[CandidateScore], thresholds: ConsistencyThresholds
) -> int:
    """Select by inverse action error after enforcing all automatic hard gates."""

    if not scores:
        raise ValueError("at least one candidate score is required")
    eligible = [
        (index, score)
        for index, score in enumerate(scores)
        if score.visual_selection_pass(thresholds)
    ]
    if eligible:
        return min(
            eligible,
            key=lambda item: (
                item[1].inverse_action_error,
                -item[1].constraint_margin(thresholds),
                -item[1].mean_score,
                item[1].seed,
            ),
        )[0]
    return max(
        enumerate(scores),
        key=lambda item: (
            item[1].constraint_margin(thresholds),
            item[1].action_adherence,
            item[1].mean_score,
            -item[1].seed,
        ),
    )[0]


def compile_action_preserving_prompt(action: ACWMActionCondition) -> str:
    """Compile text that makes carrier motion authoritative instead of generative."""

    return (
        "Photorealistically refine the supplied real-camera robot video. Treat every "
        "input frame as an immutable action carrier: preserve the exact robot pose, "
        "object pose, object trajectory, contact timing, release timing, camera, "
        "framing, and frame count. Do not reinterpret, reverse, amplify, smooth, or "
        "replace the carrier motion. Improve only appearance realism, local boundaries, "
        "lighting integration, material detail, shadows, and temporally coherent "
        "occlusion. Keep one robot and one manipulated object with stable identities; "
        "do not add limbs, duplicate the object, move the background, or create motion "
        "before contact. "
        f"Requested action intent: {action.instruction.strip()} "
        f"Authoritative timeline: {action.timeline.strip()} "
        f"The action values are defined in {action.coordinate_frame}; text is not "
        "permission to change that coordinate-frame trajectory."
    )


@dataclass(frozen=True)
class JoyAISC3Config:
    """Configuration for carrier -> JoyAI -> inverse-consistency best-of-N."""

    experiment_root: Path
    action_condition: Path
    first_frame: Path
    source_video: Path
    carrier: CarrierContract
    client_script: Path
    client_python: Path
    evaluator_python: Path
    evaluator_command: tuple[str, ...]
    candidate_seeds: tuple[int, ...]
    server_url: str = "ws://127.0.0.1:18080/ws"
    server_manifest: Path | None = None
    source_git_state: Path | None = None
    ffmpeg: Path = Path("ffmpeg")
    ffprobe: Path = Path("ffprobe")
    target_width: int = 1248
    target_height: int = 720
    target_coordinate_frame: str = JOYAI_FRAME
    num_inference_steps: int = 2
    output_quality: int = 95
    timeout_seconds: float = 3600.0
    padding_blur_sigma: float = 30.0
    model_fps: int = 24
    thresholds: ConsistencyThresholds = ConsistencyThresholds()

    def validate(self, *, require_files: bool = True, require_server: bool = False) -> None:
        self.carrier.validate(require_files=require_files)
        if not self.target_coordinate_frame.startswith("camera:"):
            raise ValueError("JoyAI target coordinate frame must be explicitly camera-relative")
        if (self.target_width, self.target_height) != (1248, 720):
            raise ValueError("the pinned JoyAI server contract requires exactly 1248x720")
        if not self.candidate_seeds or len(set(self.candidate_seeds)) != len(self.candidate_seeds):
            raise ValueError("candidate seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.candidate_seeds):
            raise ValueError("candidate seeds must be non-negative")
        if self.num_inference_steps <= 0 or not 1 <= self.output_quality <= 100:
            raise ValueError("inference steps must be positive and quality must be in [1, 100]")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if not math.isfinite(self.padding_blur_sigma) or self.padding_blur_sigma <= 0:
            raise ValueError("padding_blur_sigma must be finite and positive")
        if self.model_fps != 24:
            raise ValueError("the pinned JoyAI deployment contract requires model_fps=24")
        parsed = urlparse(self.server_url)
        if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
            raise ValueError("server_url must be an absolute ws:// or wss:// URL")
        if not self.evaluator_command:
            raise ValueError("an inverse visual evaluator command is required")
        placeholders = {
            field_name
            for argument in self.evaluator_command
            for _, field_name, _, _ in string.Formatter().parse(argument)
            if field_name is not None
        }
        unknown = placeholders - _EVALUATOR_PLACEHOLDERS
        if unknown:
            raise ValueError(f"unknown evaluator command placeholders: {sorted(unknown)}")
        required_placeholders = {
            "candidate",
            "condition",
            "first_frame",
            "metadata",
            "source",
        }
        missing_placeholders = required_placeholders - placeholders
        if missing_placeholders:
            raise ValueError(
                "evaluator command is missing required placeholders: "
                f"{sorted(missing_placeholders)}"
            )
        if require_server and self.server_manifest is None:
            raise JoyAISC3PreflightError(
                "real inference requires --server-manifest from the pinned GPU launcher"
            )
        if require_files:
            for path, label in (
                (self.action_condition, "action condition"),
                (self.first_frame, "real first frame"),
                (self.source_video, "real source video"),
                (self.client_script, "JoyAI client script"),
                (self.client_python, "JoyAI client Python"),
                (self.evaluator_python, "evaluator Python"),
                (self.ffmpeg, "ffmpeg"),
                (self.ffprobe, "ffprobe"),
            ):
                _require_file(path, label)
            if self.server_manifest is not None:
                _require_file(self.server_manifest, "JoyAI server manifest")
            if self.source_git_state is not None:
                _require_file(self.source_git_state, "source Git state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "experiment_root": str(self.experiment_root),
            "action_condition": str(self.action_condition),
            "first_frame": str(self.first_frame),
            "source_video": str(self.source_video),
            "carrier": {
                **asdict(self.carrier),
                "video": str(self.carrier.video),
            },
            "client_script": str(self.client_script),
            "client_python": str(self.client_python),
            "evaluator_python": str(self.evaluator_python),
            "evaluator_command": list(self.evaluator_command),
            "candidate_seeds": list(self.candidate_seeds),
            "server_url": self.server_url,
            "server_manifest": (
                str(self.server_manifest) if self.server_manifest is not None else None
            ),
            "source_git_state": (
                str(self.source_git_state) if self.source_git_state is not None else None
            ),
            "ffmpeg": str(self.ffmpeg),
            "ffprobe": str(self.ffprobe),
            "target_width": self.target_width,
            "target_height": self.target_height,
            "target_coordinate_frame": self.target_coordinate_frame,
            "num_inference_steps": self.num_inference_steps,
            "output_quality": self.output_quality,
            "timeout_seconds": self.timeout_seconds,
            "padding_blur_sigma": self.padding_blur_sigma,
            "model_fps": self.model_fps,
            "thresholds": asdict(self.thresholds),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, config_dir: Path) -> "JoyAISC3Config":
        carrier_raw = raw.get("carrier")
        if not isinstance(carrier_raw, Mapping):
            raise ValueError("JoyAI SC3 config requires a carrier object")
        thresholds_raw = raw.get("thresholds", {})
        if not isinstance(thresholds_raw, Mapping):
            raise ValueError("thresholds must be a JSON object")
        ffmpeg_value = raw.get("ffmpeg") or shutil.which("ffmpeg")
        ffprobe_value = raw.get("ffprobe") or shutil.which("ffprobe")
        if not isinstance(ffmpeg_value, str) or not isinstance(ffprobe_value, str):
            raise ValueError("ffmpeg and ffprobe must be configured or available on PATH")
        client_python_value = raw.get("client_python") or sys.executable
        evaluator_python_value = raw.get("evaluator_python") or sys.executable
        evaluator_command = raw.get("evaluator_command")
        if not isinstance(evaluator_command, list) or not all(
            isinstance(value, str) and value for value in evaluator_command
        ):
            raise ValueError("evaluator_command must be a non-empty JSON string array")
        seeds = raw.get("candidate_seeds")
        if not isinstance(seeds, list):
            raise ValueError("candidate_seeds must be a JSON array")
        physical_evidence = carrier_raw.get("physical_evidence", False)
        if type(physical_evidence) is not bool:
            raise ValueError("carrier physical_evidence must be a JSON boolean")
        return cls(
            experiment_root=_resolve_path(str(raw["experiment_root"]), base_dir=config_dir),
            action_condition=_resolve_path(str(raw["action_condition"]), base_dir=config_dir),
            first_frame=_resolve_path(str(raw["first_frame"]), base_dir=config_dir),
            source_video=_resolve_path(str(raw["source_video"]), base_dir=config_dir),
            carrier=CarrierContract(
                video=_resolve_path(str(carrier_raw["video"]), base_dir=config_dir),
                coordinate_frame=str(carrier_raw["coordinate_frame"]),
                generator=str(carrier_raw["generator"]),
                generator_revision=str(carrier_raw["generator_revision"]),
                motion_authority=str(carrier_raw["motion_authority"]),
                physical_evidence=physical_evidence,
            ),
            client_script=_resolve_path(str(raw["client_script"]), base_dir=config_dir),
            client_python=_resolve_path(
                str(client_python_value), base_dir=config_dir, executable=True
            ),
            evaluator_python=_resolve_path(
                str(evaluator_python_value), base_dir=config_dir, executable=True
            ),
            evaluator_command=tuple(evaluator_command),
            candidate_seeds=tuple(int(seed) for seed in seeds),
            server_url=str(raw.get("server_url", "ws://127.0.0.1:18080/ws")),
            server_manifest=_optional_path(            raw.get("server_manifest"), base_dir=config_dir
            ),
            source_git_state=_optional_path(
            raw.get("source_git_state"), base_dir=config_dir),
            ffmpeg=_resolve_path(str(ffmpeg_value), base_dir=config_dir),
            ffprobe=_resolve_path(str(ffprobe_value), base_dir=config_dir),
            target_width=int(raw.get("target_width", 1248)),
            target_height=int(raw.get("target_height", 720)),
            target_coordinate_frame=str(raw.get("target_coordinate_frame", JOYAI_FRAME)),
            num_inference_steps=int(raw.get("num_inference_steps", 2)),
            output_quality=int(raw.get("output_quality", 95)),
            timeout_seconds=float(raw.get("timeout_seconds", 3600.0)),
            padding_blur_sigma=float(raw.get("padding_blur_sigma", 30.0)),
            model_fps=int(raw.get("model_fps", 24)),
            thresholds=ConsistencyThresholds(
                **{field: float(thresholds_raw.get(field, 0.75)) for field in _SCORE_FIELDS}
            ),
        )


def validate_server_manifest(path: Path) -> dict[str, Any]:
    """Require evidence that the pinned two-GPU JoyAI service is currently ready."""

    resolved = _require_file(path.expanduser().resolve(), "JoyAI server manifest")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise JoyAISC3PreflightError("JoyAI server manifest must contain one JSON object")
    if payload.get("status") != "WORKING" or payload.get("stage") != "joyai_server_ready":
        raise JoyAISC3PreflightError(
            "JoyAI server manifest must be WORKING at stage joyai_server_ready"
        )
    source = payload.get("source")
    checkpoints = payload.get("checkpoints")
    gpu = payload.get("gpu")
    runtime = payload.get("runtime")
    if not all(isinstance(value, Mapping) for value in (source, checkpoints, gpu, runtime)):
        raise JoyAISC3PreflightError(
            "JoyAI server manifest lacks source, checkpoint, GPU, or runtime evidence"
        )
    assert isinstance(source, Mapping)
    assert isinstance(checkpoints, Mapping)
    assert isinstance(gpu, Mapping)
    assert isinstance(runtime, Mapping)
    if source.get("revision") != JOYAI_REPOSITORY_REVISION:
        raise JoyAISC3PreflightError("JoyAI server source revision is not pinned")
    if checkpoints.get("model_revision") != JOYAI_MODEL_REVISION:
        raise JoyAISC3PreflightError("JoyAI server model revision is not pinned")
    if checkpoints.get("text_encoder_revision") != JOYAI_TEXT_ENCODER_REVISION:
        raise JoyAISC3PreflightError("JoyAI server text encoder revision is not pinned")
    selected = gpu.get("selected")
    visible = gpu.get("cuda_visible_devices")
    if not isinstance(selected, list) or len(selected) != 2 or not isinstance(visible, str):
        raise JoyAISC3PreflightError("JoyAI server manifest must record two selected physical GPUs")
    try:
        ordered = sorted(selected, key=lambda row: int(row["logical_index"]))
        physical = [int(row["physical_index"]) for row in ordered]
    except (KeyError, TypeError, ValueError) as exc:
        raise JoyAISC3PreflightError("JoyAI selected GPU records are malformed") from exc
    try:
        visible_indices = [int(value) for value in visible.split(",")]
    except ValueError as exc:
        raise JoyAISC3PreflightError("CUDA_VISIBLE_DEVICES is malformed") from exc
    if len(set(physical)) != 2 or physical != visible_indices:
        raise JoyAISC3PreflightError(
            "selected physical GPUs do not match recorded CUDA_VISIBLE_DEVICES"
        )
    if runtime.get("cuda_available") is not True or runtime.get("cuda_device_count") != 2:
        raise JoyAISC3PreflightError("JoyAI runtime did not validate exactly two CUDA devices")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "status": payload["status"],
        "stage": payload["stage"],
        "physical_gpus": physical,
        "cuda_visible_devices": visible,
        "source_revision": source["revision"],
        "model_revision": checkpoints["model_revision"],
        "text_encoder_revision": checkpoints["text_encoder_revision"],
        "health_url": payload.get("health_url"),
    }


class JoyAISC3Runner:
    """Prepare, generate, inverse-check, and select whole-stream JoyAI candidates."""

    def __init__(
        self,
        config: JoyAISC3Config,
        *,
        config_path: Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.config = config
        self.config_path = config_path.expanduser().resolve() if config_path else None
        self.project_root = (
            (project_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
        )

    def _new_run_dir(self) -> Path:
        root = self.config.experiment_root
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = root / f"{stamp}-{uuid.uuid4().hex[:8]}-joyai-sc3"
        run_dir.mkdir()
        return run_dir

    def _probe_video(self, path: Path) -> VideoStream:
        command = [
            str(self.config.ffprobe),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames,nb_frames,duration:format=duration",
            "-of",
            "json",
            str(path),
        ]
        completed = subprocess.run(
            command,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if completed.returncode:
            raise JoyAISC3PreflightError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
        return VideoStream.from_ffprobe(json.loads(completed.stdout))

    def _probe_dimensions(self, path: Path) -> tuple[int, int]:
        command = [
            str(self.config.ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ]
        completed = subprocess.run(
            command,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if completed.returncode:
            raise JoyAISC3PreflightError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
        payload = json.loads(completed.stdout)
        streams = payload.get("streams")
        if not isinstance(streams, list) or len(streams) != 1:
            raise JoyAISC3PreflightError(f"{path} must contain exactly one image stream")
        try:
            return int(streams[0]["width"]), int(streams[0]["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise JoyAISC3PreflightError(f"ffprobe dimensions are malformed for {path}") from exc

    def _run_command(self, command: Sequence[str], log_path: Path, *, timeout: float) -> str:
        completed = subprocess.run(
            list(command),
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "$ "
            + shlex.join(command)
            + "\n\nSTDOUT\n"
            + completed.stdout
            + "\nSTDERR\n"
            + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode:
            raise RuntimeError(
                f"command failed with exit code {completed.returncode}; inspect {log_path}"
            )
        return completed.stdout

    def _git_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for name, arguments in {
            "head": ("rev-parse", "HEAD"),
            "branch": ("branch", "--show-current"),
            "status": ("status", "--short"),
        }.items():
            completed = subprocess.run(
                ["git", *arguments],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            state[name] = {
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        return state

    def _package_state(self, run_dir: Path) -> dict[str, Any]:
        completed = subprocess.run(
            [str(self.config.evaluator_python), "-m", "pip", "freeze"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        path = run_dir / "packages.txt"
        path.write_text(completed.stdout, encoding="utf-8")
        return {
            "python": str(self.config.evaluator_python),
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
            "path": str(path),
            "sha256": sha256_file(path),
        }

    def _preflight(
        self, *, require_server: bool
    ) -> tuple[
        ACWMActionCondition,
        VideoStream,
        FitBlurPadTransform,
        dict[str, Any] | None,
    ]:
        self.config.validate(require_files=True, require_server=require_server)
        action = ACWMActionCondition.from_json(self.config.action_condition)
        if action.coordinate_frame != self.config.carrier.coordinate_frame:
            raise JoyAISC3PreflightError(
                "action and carrier coordinate frames differ: "
                f"{action.coordinate_frame!r} != {self.config.carrier.coordinate_frame!r}"
            )
        carrier_stream = self._probe_video(self.config.carrier.video)
        if carrier_stream.frame_count != len(action.timestamps_s):
            raise JoyAISC3PreflightError(
                f"carrier has {carrier_stream.frame_count} frames but action has "
                f"{len(action.timestamps_s)} timestamps"
            )
        if not math.isclose(carrier_stream.fps, action.fps, rel_tol=0, abs_tol=1e-4):
            raise JoyAISC3PreflightError(
                f"carrier runs at {carrier_stream.fps} FPS but action runs at {action.fps} FPS"
            )
        first_size = self._probe_dimensions(self.config.first_frame)
        if first_size != (carrier_stream.width, carrier_stream.height):
            raise JoyAISC3PreflightError(
                f"first frame is {first_size}, carrier is "
                f"{(carrier_stream.width, carrier_stream.height)}"
            )
        transform = FitBlurPadTransform.create(
            source_frame=action.coordinate_frame,
            target_frame=self.config.target_coordinate_frame,
            source_width=carrier_stream.width,
            source_height=carrier_stream.height,
            target_width=self.config.target_width,
            target_height=self.config.target_height,
            background_blur_sigma=self.config.padding_blur_sigma,
        )
        server = (
            validate_server_manifest(self.config.server_manifest)
            if require_server and self.config.server_manifest is not None
            else None
        )
        return action, carrier_stream, transform, server

    def _prepare_commands(
        self,
        *,
        run_dir: Path,
        carrier_stream: VideoStream,
        transform: FitBlurPadTransform,
    ) -> tuple[list[str], list[str], Path, Path, int, int]:
        prepared = run_dir / "input" / "action-carrier-joyai-1248x720.mkv"
        reference = run_dir / "input" / "first-frame-joyai-1248x720.png"
        prepared.parent.mkdir(parents=True)
        model_frames = resampled_frame_count(
            carrier_stream.frame_count,
            source_fps_numerator=carrier_stream.fps_numerator,
            source_fps_denominator=carrier_stream.fps_denominator,
            target_fps=self.config.model_fps,
        )
        padded_frames = causal_padded_frame_count(model_frames)
        tail_padding = padded_frames - model_frames
        carrier_command = [
            str(self.config.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self.config.carrier.video),
            "-vf",
            transform.prepare_filter(            tail_padding_frames=tail_padding,
            output_fps=self.config.model_fps,
            resample_fps=self.config.model_fps,),
            "-an",
            "-frames:v",
            str(padded_frames),
            "-r",
            str(self.config.model_fps),
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "bgr0",
            str(prepared),
        ]
        reference_command = [
            str(self.config.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self.config.first_frame),
            "-vf",
            transform.prepare_filter(
                tail_padding_frames=0,
                output_fps=self.config.model_fps,
            ),
            "-frames:v",
            "1",
            str(reference),
        ]
        return (
            carrier_command,
            reference_command,
            prepared,
            reference,
            model_frames,
            padded_frames,
        )

    def _client_command(
        self,
        *,
        seed: int,
        output_dir: Path,
        prepared: Path,
        reference: Path,
        prompt_path: Path,
        carrier_stream: VideoStream,
        padded_frames: int,
    ) -> list[str]:
        return [
            str(self.config.client_python),
            str(self.config.client_script),
            "--server-url",
            self.config.server_url,
            "--input-video",
            str(prepared),
            "--output-dir",
            str(output_dir),
            "--reference-image",
            str(reference),
            "--prompt-file",
            str(prompt_path),
            "--width",
            str(self.config.target_width),
            "--height",
            str(self.config.target_height),
            "--fps",
            str(self.config.model_fps),
            "--expected-frames",
            str(padded_frames),
            "--seed",
            str(seed),
            "--num-inference-steps",
            str(self.config.num_inference_steps),
            "--output-quality",
            str(self.config.output_quality),
            "--throughput-mode",
            "--output-artifacts",
            "both",
            "--frame-log-every",
            str(padded_frames),
            "--timeout-seconds",
            f"{self.config.timeout_seconds:.12g}",
        ]

    def _restore_commands(
        self,
        *,
        raw_dir: Path,
        candidate_dir: Path,
        carrier_stream: VideoStream,
        transform: FitBlurPadTransform,
        model_frames: int,
    ) -> tuple[list[str], list[str], Path, Path]:
        lossless = candidate_dir / "candidate-restored-lossless.mkv"
        review = candidate_dir / "candidate-restored-review.mp4"
        restore = [
            str(self.config.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_dir / "joyai-proposal-lossless.mkv"),
            "-vf",
            (
                f"{transform.restore_filter()},trim=end_frame={model_frames},"
                "setpts=PTS-STARTPTS,"
                f"fps=fps={carrier_stream.fps:.12g}:round=near"
            ),
            "-an",
            "-frames:v",
            str(carrier_stream.frame_count),
            "-r",
            f"{carrier_stream.fps_numerator}/{carrier_stream.fps_denominator}",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "bgr0",
            str(lossless),
        ]
        encode = [
            str(self.config.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(lossless),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "8",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(review),
        ]
        return restore, encode, lossless, review

    def _evaluator_command(
        self,
        *,
        seed: int,
        candidate: Path,
        metadata: Path,
        evidence_root: Path,
    ) -> list[str]:
        values = {
            "candidate": str(candidate),
            "condition": str(self.config.action_condition),
            "evidence_root": str(evidence_root),
            "first_frame": str(self.config.first_frame),
            "metadata": str(metadata),
            "project_root": str(self.project_root),
            "python": str(self.config.evaluator_python),
            "seed": str(seed),
            "source": str(self.config.source_video),
        }
        return [argument.format_map(values) for argument in self.config.evaluator_command]

    def _initial_manifest(self, *, run_dir: Path, prepare_only: bool) -> dict[str, Any]:
        source_files = (
            self.project_root / "phiagent" / "world_model" / "joyai_sc3.py",
            self.project_root / "scripts" / "run_joyai_sc3_harness.py",
            self.config.client_script,
        )
        source_git_state = None
        if self.config.source_git_state is not None:
            payload = json.loads(self.config.source_git_state.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("source Git state must contain one JSON object")
            source_git_state = {
                "path": str(self.config.source_git_state),
                "sha256": sha256_file(self.config.source_git_state),
                "state": dict(payload),
            }
        return {
            "schema_version": "1.0.0",
            "status": "PARTIAL",
            "stage": "joyai_sc3_initializing",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "command": [sys.executable, *sys.argv],
            "command_shell": shlex.join([sys.executable, *sys.argv]),
            "prepare_only": prepare_only,
            "method": (
                "authoritative_action_carrier_then_joyai_residual_rendering_"
                "then_inverse_visual_consistency"
            ),
            "sc3_equivalence": False,
            "physical_evidence": False,
            "config": self.config.to_dict(),
            "config_source": (
                {
                    "path": str(self.config_path),
                    "sha256": sha256_file(self.config_path),
                }
                if self.config_path is not None
                else None
            ),
            "git": self._git_state(),
            "source_git_state": source_git_state,
            "packages": self._package_state(run_dir),
            "source_files": {
                str(path): sha256_file(path) for path in source_files if path.is_file()
            },
            "error": None,
        }

    def run(self, *, prepare_only: bool = False) -> dict[str, Any]:
        run_dir = self._new_run_dir()
        manifest_path = run_dir / "manifest.json"
        manifest = self._initial_manifest(run_dir=run_dir, prepare_only=prepare_only)
        write_json(manifest_path, manifest)
        try:
            action, carrier_stream, transform, server = self._preflight(
                require_server=not prepare_only
            )
            prompt = compile_action_preserving_prompt(action)
            contracts = run_dir / "contracts"
            contracts.mkdir()
            prompt_path = contracts / "action-preserving-prompt.txt"
            prompt_path.write_text(prompt + "\n", encoding="utf-8")
            action_snapshot = contracts / "action-condition.json"
            write_json(action_snapshot, action.to_dict())
            (
                carrier_command,
                reference_command,
                prepared,
                reference,
                model_frames,
                padded_frames,
            ) = self._prepare_commands(
                run_dir=run_dir,
                carrier_stream=carrier_stream,
                transform=transform,
            )
            self._run_command(
                carrier_command,
                run_dir / "logs" / "prepare-carrier.log",
                timeout=600,
            )
            self._run_command(
                reference_command,
                run_dir / "logs" / "prepare-reference.log",
                timeout=120,
            )
            prepared_stream = self._probe_video(prepared)
            if (
                prepared_stream.width,
                prepared_stream.height,
                prepared_stream.frame_count,
            ) != (self.config.target_width, self.config.target_height, padded_frames):
                raise RuntimeError(
                    "prepared JoyAI carrier does not match the target dimension/frame contract"
                )
            if not math.isclose(
                prepared_stream.fps,
                self.config.model_fps,
                rel_tol=0,
                abs_tol=1e-6,
            ):
                raise RuntimeError("prepared JoyAI carrier does not run at model_fps")
            client_plans = [
                {
                    "seed": seed,
                    "command": self._client_command(
                        seed=seed,
                        output_dir=run_dir / "candidates" / f"seed-{seed}" / "joyai-raw",
                        prepared=prepared,
                        reference=reference,
                        prompt_path=prompt_path,
                        carrier_stream=carrier_stream,
                        padded_frames=padded_frames,
                    ),
                }
                for seed in self.config.candidate_seeds
            ]
            manifest.update(
                {
                    "stage": "joyai_sc3_prepared",
                    "preflight": {
                        "server": server,
                        "carrier": {
                            **asdict(carrier_stream),
                            "fps": carrier_stream.fps,
                            "path": str(self.config.carrier.video),
                            "sha256": sha256_file(self.config.carrier.video),
                            "contract": {
                                **asdict(self.config.carrier),
                                "video": str(self.config.carrier.video),
                            },
                        },
                        "action": {
                            "path": str(self.config.action_condition),
                            "sha256": sha256_file(self.config.action_condition),
                            "coordinate_frame": action.coordinate_frame,
                            "representation": action.representation.value,
                            "timestamps": len(action.timestamps_s),
                            "fps": action.fps,
                        },
                        "first_frame": {
                            "path": str(self.config.first_frame),
                            "sha256": sha256_file(self.config.first_frame),
                        },
                        "source_video": {
                            "path": str(self.config.source_video),
                            "sha256": sha256_file(self.config.source_video),
                        },
                        "transform": transform.to_manifest(),
                        "causal_padding": {
                            "source_action_frames": carrier_stream.frame_count,
                            "source_fps": carrier_stream.fps,
                            "model_deliverable_frames": model_frames,
                            "model_fps": self.config.model_fps,
                            "padded_frames": padded_frames,
                            "tail_clones": padded_frames - model_frames,
                            "source_end_support_frames_before_resampling": 1,
                            "temporal_resampling": "nearest_timestamp_no_interpolation",
                            "trim_after_generation": True,
                            "restore_source_fps_after_trim": True,
                        },
                    },
                    "prepared_inputs": {
                        "carrier": {
                            "path": str(prepared),
                            "sha256": sha256_file(prepared),
                            "stream": asdict(prepared_stream),
                        },
                        "reference": {
                            "path": str(reference),
                            "sha256": sha256_file(reference),
                        },
                        "prompt": {
                            "path": str(prompt_path),
                            "sha256": sha256_file(prompt_path),
                        },
                        "action_snapshot": {
                            "path": str(action_snapshot),
                            "sha256": sha256_file(action_snapshot),
                        },
                    },
                    "commands": {
                        "prepare_carrier": carrier_command,
                        "prepare_reference": reference_command,
                        "candidates": client_plans,
                    },
                    "model_inference": "NOT STARTED" if prepare_only else "RUNNING",
                }
            )
            write_json(manifest_path, manifest)
            if prepare_only:
                manifest.update(
                    {
                        "status": "PARTIAL",
                        "stage": "joyai_sc3_prepared_not_run",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "acceptance": {
                            "harness_preparation": "WORKING",
                            "joyai_inference": "NOT STARTED",
                            "inverse_consistency": "NOT STARTED",
                            "human_review": "NOT STARTED",
                            "demo": "PARTIAL",
                        },
                    }
                )
                write_json(manifest_path, manifest)
                return manifest

            candidate_records = []
            scores = []
            for plan in client_plans:
                seed = int(plan["seed"])
                candidate_dir = run_dir / "candidates" / f"seed-{seed}"
                candidate_dir.mkdir(parents=True)
                raw_dir = candidate_dir / "joyai-raw"
                client_command = list(plan["command"])
                self._run_command(
                    client_command,
                    run_dir / "logs" / f"seed-{seed}-client.log",
                    timeout=self.config.timeout_seconds + 900,
                )
                restore, encode, lossless, review = self._restore_commands(
                    raw_dir=raw_dir,
                    candidate_dir=candidate_dir,
                    carrier_stream=carrier_stream,
                    transform=transform,
                    model_frames=model_frames,
                )
                self._run_command(
                    restore,
                    run_dir / "logs" / f"seed-{seed}-restore.log",
                    timeout=900,
                )
                self._run_command(
                    encode,
                    run_dir / "logs" / f"seed-{seed}-encode.log",
                    timeout=900,
                )
                restored_stream = self._probe_video(review)
                if (
                    restored_stream.width,
                    restored_stream.height,
                    restored_stream.frame_count,
                ) != (
                    carrier_stream.width,
                    carrier_stream.height,
                    carrier_stream.frame_count,
                ):
                    raise RuntimeError(
                        f"restored candidate seed {seed} violates the carrier stream contract"
                    )
                metadata_path = candidate_dir / "candidate-metadata.json"
                write_json(
                    metadata_path,
                    {
                        "schema_version": "1.0.0",
                        "seed": seed,
                        "condition": str(self.config.action_condition),
                        "carrier": str(self.config.carrier.video),
                        "prompt": str(prompt_path),
                        "joyai_manifest": str(raw_dir / "manifest.json"),
                        "transform": transform.to_manifest(),
                        "lossless": {
                            "path": str(lossless),
                            "sha256": sha256_file(lossless),
                        },
                        "review": {
                            "path": str(review),
                            "sha256": sha256_file(review),
                        },
                        "physical_evidence": False,
                    },
                )
                evidence_root = candidate_dir / "evaluation"
                evaluator_command = self._evaluator_command(
                    seed=seed,
                    candidate=review,
                    metadata=metadata_path,
                    evidence_root=evidence_root,
                )
                stdout = self._run_command(
                    evaluator_command,
                    run_dir / "logs" / f"seed-{seed}-evaluator.log",
                    timeout=900,
                )
                score_payload = json.loads(stdout)
                if not isinstance(score_payload, Mapping):
                    raise ValueError("inverse evaluator must emit one JSON object")
                score = CandidateScore.from_payload(seed, score_payload)
                scores.append(score)
                candidate_records.append(
                    {
                        "seed": seed,
                        "joyai_manifest": str(raw_dir / "manifest.json"),
                        "lossless": str(lossless),
                        "review": str(review),
                        "metadata": str(metadata_path),
                        "evaluator_command": evaluator_command,
                        "score": {
                            **asdict(score),
                            "inverse_action_error": score.inverse_action_error,
                            "mean_score": score.mean_score,
                            "constraint_margin": score.constraint_margin(self.config.thresholds),
                            "automatic_pass": score.automatic_pass(self.config.thresholds),
                            "visual_selection_pass": score.visual_selection_pass(
                                self.config.thresholds
                            ),
                            "promotion_pass": score.promotion_pass(self.config.thresholds),
                        },
                        "commands": {
                            "client": client_command,
                            "restore": restore,
                            "encode": encode,
                        },
                    }
                )
                manifest["candidates"] = candidate_records
                write_json(manifest_path, manifest)

            selected_index = select_consistent_candidate(scores, self.config.thresholds)
            selected = scores[selected_index]
            selected_record = candidate_records[selected_index]
            any_automatic = selected.automatic_pass(self.config.thresholds)
            promoted = selected.promotion_pass(self.config.thresholds)
            manifest.update(
                {
                    "status": "WORKING" if promoted else "PARTIAL",
                    "stage": (
                        "joyai_sc3_accepted"
                        if promoted
                        else (
                            "joyai_sc3_pending_human_review"
                            if any_automatic and selected.human_review_passed is None
                            else "joyai_sc3_rejected_by_consistency"
                        )
                    ),
                    "model_inference": "COMPLETED",
                    "candidates": candidate_records,
                    "selection": {
                        "policy": (
                            "automatic-gates-then-minimum-inverse-action-error-"
                            "then-margin-then-mean-then-seed"
                        ),
                        "selected_index": selected_index,
                        "selected_seed": selected.seed,
                        "selected_review": selected_record["review"],
                        "automatic_pass": any_automatic,
                        "human_review_passed": selected.human_review_passed,
                        "promotion_pass": promoted,
                        "thresholds": asdict(self.config.thresholds),
                    },
                    "acceptance": {
                        "harness_preparation": "WORKING",
                        "joyai_inference": "WORKING",
                        "inverse_consistency": ("WORKING" if any_automatic else "PARTIAL"),
                        "human_review": (
                            "WORKING"
                            if selected.human_review_passed is True
                            else (
                                "PARTIAL"
                                if selected.human_review_passed is False
                                else "NOT STARTED"
                            )
                        ),
                        "demo": "WORKING" if promoted else "PARTIAL",
                    },
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            write_json(manifest_path, manifest)
            return manifest
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.SubprocessError,
            json.JSONDecodeError,
        ) as exc:
            manifest.update(
                {
                    "status": "PARTIAL",
                    "stage": "joyai_sc3_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            write_json(manifest_path, manifest)
            raise


def load_config(path: Path) -> JoyAISC3Config:
    resolved = _require_file(path.expanduser().resolve(), "JoyAI SC3 config")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("JoyAI SC3 config must contain one JSON object")
    return JoyAISC3Config.from_dict(payload, config_dir=resolved.parent)


def with_runtime_overrides(
    config: JoyAISC3Config,
    *,
    client_python: Path | None = None,
    evaluator_python: Path | None = None,
    server_manifest: Path | None = None,
    server_url: str | None = None,
    source_git_state: Path | None = None,
    experiment_root: Path | None = None,
    candidate_seeds: Sequence[int] | None = None,
) -> JoyAISC3Config:
    """Apply CLI runtime locations without weakening the checked-in contract."""

    selected_seeds = (
        tuple(int(seed) for seed in candidate_seeds)
        if candidate_seeds is not None
        else config.candidate_seeds
    )
    unknown_seeds = set(selected_seeds) - set(config.candidate_seeds)
    if unknown_seeds:
        raise ValueError(
            f"runtime seeds are not in the frozen candidate set: {sorted(unknown_seeds)}"
        )
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("runtime candidate seeds must be non-empty and unique")
    return replace(
        config,
        experiment_root=(
            experiment_root.expanduser().resolve()
            if experiment_root is not None
            else config.experiment_root
        ),
        candidate_seeds=selected_seeds,
        client_python=(
            _absolute_without_resolving(client_python)
            if client_python is not None
            else config.client_python
        ),
        evaluator_python=(
            _absolute_without_resolving(evaluator_python)
            if evaluator_python is not None
            else config.evaluator_python
        ),
        server_manifest=(
            server_manifest.expanduser().resolve()
            if server_manifest is not None
            else config.server_manifest
        ),
        server_url=server_url or config.server_url,
        source_git_state=(
            source_git_state.expanduser().resolve()
            if source_git_state is not None
            else config.source_git_state
        ),
    )
