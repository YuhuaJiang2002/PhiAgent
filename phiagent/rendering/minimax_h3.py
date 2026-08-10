"""Lightweight configuration helpers for optional MiniMax-H3 validation.

The actual H3, Torch, and DiffSynth imports deliberately live in the GPU entry
point.  Importing :mod:`phiagent` therefore continues to require neither CUDA
nor the 50+ GiB quantized checkpoint used by the validation experiment.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from phiagent.rendering.wan_animate import GPUInfo, PreflightError, select_gpu


DIFFSYNTH_H3_COMMIT = "b1c02ce76aabc989f6bf534756b2da84532249e5"
MINIMAX_H3_MODEL_ID = "MiniMaxAI/MiniMax-H3"
MINIMAX_H3_MODELSCOPE_ID = "MiniMax/MiniMax-H3"
MINIMAX_H3_NF4_MODEL_ID = "DiffSynth-Studio/MiniMax-H3-NF4"


@dataclass(frozen=True)
class H3ActionVariant:
    """One language-conditioned action for a matched H3 comparison.

    ``label`` is deliberately filesystem-safe because experiment runners use it
    as an output-directory name.  The instruction and timeline stay separate:
    the former is the user-facing command, while the latter gives H3 explicit
    temporal constraints without silently rewriting the command.
    """

    label: str
    instruction: str
    timeline: str

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", self.label):
            raise ValueError(
                "action label must contain 1-64 lowercase letters, digits, '-' or '_'"
            )
        if not self.instruction.strip():
            raise ValueError("action instruction must not be empty")
        if not self.timeline.strip():
            raise ValueError("action timeline must not be empty")


@dataclass(frozen=True)
class H3LongWindow:
    """One H3 window in the absolute source camera timeline."""

    index: int
    start_frame: int
    frame_count: int
    source_frames: int
    padded_frames: int

    @property
    def end_frame_exclusive(self) -> int:
        return self.start_frame + self.frame_count


def plan_h3_long_windows(
    full_frame_count: int,
    *,
    window_frames: int = 124,
    overlap_frames: int = 28,
) -> tuple[H3LongWindow, ...]:
    """Cover a timeline with overlapping H3-compatible windows.

    The final start is pinned to the last real frame.  This can make the final
    overlap larger than requested, which is preferable to padding a long tail
    or dropping source frames.
    """

    if full_frame_count < window_frames:
        raise ValueError("full_frame_count must be at least window_frames")
    if window_frames < 5 or (window_frames - 5) % 17:
        raise ValueError("window_frames must satisfy window_frames = 17n + 5")
    if not 1 <= overlap_frames < window_frames:
        raise ValueError("overlap_frames must be in [1, window_frames - 1]")
    stride = window_frames - overlap_frames
    final_start = full_frame_count - window_frames
    starts = [0]
    while starts[-1] < final_start:
        candidate = min(starts[-1] + stride, final_start)
        if candidate == starts[-1]:
            break
        starts.append(candidate)
    return tuple(
        H3LongWindow(
            index=index,
            start_frame=start,
            frame_count=window_frames,
            source_frames=min(window_frames, full_frame_count - start),
            padded_frames=max(0, window_frames - (full_frame_count - start)),
        )
        for index, start in enumerate(starts)
    )


def flower_epl_phase(frame: int, full_frame_count: int = 660) -> str:
    """Return the reproducible coarse EPL phase for an absolute source frame."""

    if not 0 <= frame < full_frame_count:
        raise ValueError("frame must be inside the source timeline")
    progress = frame / max(1, full_frame_count - 1)
    if progress < 0.12:
        return "approach"
    if progress < 0.25:
        return "pregrasp"
    if progress < 0.38:
        return "grasp"
    if progress < 0.72:
        return "manipulate"
    if progress < 0.86:
        return "release"
    return "retract"


def build_flower_window_epl_constraint(
    start_frame: int,
    frame_count: int,
    *,
    full_frame_count: int = 660,
    contact_start_frame: int = 236,
    contact_end_frame_exclusive: int = 316,
) -> str:
    """Describe absolute EPL and hard-contact constraints for one H3 window."""

    if frame_count <= 0 or start_frame < 0:
        raise ValueError("window start and length must be positive")
    end = start_frame + frame_count
    if end > full_frame_count:
        raise ValueError("window must be inside the full source timeline")
    phases: list[str] = []
    for frame in range(start_frame, end):
        phase = flower_epl_phase(frame, full_frame_count)
        if phase not in phases:
            phases.append(phase)
    contact_overlap_start = max(start_frame, contact_start_frame)
    contact_overlap_end = min(end, contact_end_frame_exclusive)
    contact_text = (
        f"This window overlaps the hard flower-contact interval at absolute frames "
        f"[{contact_overlap_start}, {contact_overlap_end}). Preserve the same flower "
        "stem identity, robot-finger/stem attachment, vase-depth ordering, continuous "
        "contact, and source timing frame by frame."
        if contact_overlap_start < contact_overlap_end
        else "This window does not overlap the hard flower-contact interval."
    )
    return (
        "\nepl_window_constraint:\n"
        f"This is absolute source frames [{start_frame}, {end}) of {full_frame_count}; "
        f"the ordered EPL phases present are {', '.join(phases)}. Do not restart, "
        "retime, or summarize the action at the window boundary. The first and last "
        "frames are ordinary continuation frames of one 27.5-second shot. "
        + contact_text
    )


def file_sha256(path: Path) -> str:
    """Hash an experiment input or output without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_h3_frame_count(frame_count: int) -> int:
    """Round up to the frame-count contract used by the DiffSynth H3 VAE."""

    current = max(int(frame_count), 5)
    while (current - 5) % 17:
        current += 1
    return current


def build_flower_ref2va_prompt(duration_seconds: float) -> str:
    """Build the six-section prompt required by H3 full-reference mode."""

    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    return f"""subject_definitions:
<Subject 1> is the complete silver-and-graphite humanoid service robot in <Picture 1>, with a compact smooth head, a dark face panel, articulated shoulders, elbows and wrists, and two five-finger dexterous hands.
<Subject 2> is the flower-arranging workspace in <Video 1>, including every flower, petal, stem, leaf, vase, glass, table, ribbon, shelf, paper roll, wall decoration, window, shadow and reflection.
<Subject 3> is the florist's complete pose and motion trajectory in <Video 1>, including torso orientation, gaze direction, both arm trajectories, both hand trajectories, timing and flower-stem interactions; her human appearance is not retained.
<Video 1> is the source video for the target video edit and supplies the exact camera, framing, temporal structure and manipulation motion.

summary:
[video editing + reference generation] The target video is an edited version of <Video 1>. Replace only the human florist with <Subject 1>, transfer <Subject 3> to the robot, and preserve <Subject 2> and the source camera exactly.

retention_analysis:
<Subject 1> (appears throughout [Shot 1]): fully_preserved - retain the robot's material, proportions, head, dark face panel, articulated arms and two five-finger hands consistently in every frame.
<Subject 2> (appears throughout [Shot 1]): fully_preserved - retain the workspace geometry, colors, lighting, depth of field, flowers, stems, tools and all foreground occlusions without additions, removals or motion changes.
<Subject 3> (appears throughout [Shot 1]): attribute_transfer - transfer the florist's complete body, arm, wrist, hand and finger motion timing to <Subject 1> while removing all human skin, face, hair and clothing.
<Video 1> (complete visual and temporal structure): fully_preserved - preserve the original single shot, camera position, framing, pacing and every non-human scene element.

detailed_description:
The target video is a photorealistic, single-shot source-video edit with the same natural daylight, exposure, white balance, reflections and depth of field as <Video 1>.
[Shot 1] For the complete {duration_seconds:.3f}-second shot, the camera and every element of <Subject 2> remain fixed to their exact source positions and source motion. The human florist is absent from the first frame onward. In precisely her place is <Subject 1>, at the same scale and depth, following <Subject 3> frame by frame. The robot matches every torso lean, head direction, shoulder rotation, elbow bend, wrist path, five-finger hand pose and manipulation timing from <Video 1>. Flowers, petals, leaves and stems that pass in front of the florist in the source remain in front of the robot with unchanged silhouettes and trajectories. Objects formerly behind the florist remain behind the robot. No source flower, stem, vase, glass, table item, ribbon, shelf item, paper roll, wall feature, window detail, shadow or reflection is regenerated or displaced. The robot remains a single coherent identity with two arms, two hands and exactly five fingers on each hand; it has no human skin, face, hair, shirt, apron or other clothing. Do not introduce extra limbs, tools, objects, text, logos, glow, sparks, camera movement, reframing or scene cuts.

overall_soundscape:
No dialogue or newly invented sound is required; visual preservation and synchronized manipulation motion are the validation targets.

non_diegetic_music:
None.
"""


def build_action_conditioned_flower_ref2va_prompt(
    duration_seconds: float,
    action: H3ActionVariant,
    *,
    scene_reference: str = "video",
) -> str:
    """Build a Ref2VA prompt where language, not source-human motion, is the action.

    The real video remains a full-frame visual reference for the camera, scene,
    lighting, object identities and temporal duration.  Its human motion is
    explicitly excluded from the target trajectory so matched prompts can test
    whether MiniMax-H3 responds to different action conditions.
    """

    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    if scene_reference not in {"video", "anchor_image", "control_video"}:
        raise ValueError(
            "scene_reference must be 'video', 'anchor_image', or 'control_video'"
        )
    action.validate()
    instruction = " ".join(action.instruction.split())
    timeline = " ".join(action.timeline.split())
    if scene_reference == "video":
        scene_token = "<Video 1>"
        scene_definition = (
            "<Video 1> is the real-scene visual reference. It supplies the exact camera, "
            "framing, duration, natural lighting, workspace geometry, background and object "
            "identities. The source florist is removed. Her body trajectory, arm path, hand "
            "path and manipulation motion are explicitly not the target action and must not be copied."
        )
        source_exclusion = (
            "preserve scene appearance, object identity, camera, framing and duration, but do not "
            "transfer the source person's pose, limb trajectory, hand motion or action timing."
        )
        real_scene_phrase = "the same real room as <Video 1>"
    elif scene_reference == "anchor_image":
        scene_token = "<Picture 2>"
        scene_definition = (
            "<Picture 2> is a real-scene anchor frame extracted from the existing source video. "
            "It supplies the exact real camera viewpoint, natural lighting, workspace geometry, "
            "background and object identities. Any source person visible in the anchor is removed "
            "and supplies no target pose, limb path, hand motion or action timing."
        )
        source_exclusion = (
            "preserve the real scene appearance, object identity, camera and framing, while deriving "
            "all temporal motion exclusively from <Subject 3>."
        )
        real_scene_phrase = "the real room and camera shown in <Picture 2>"
        control_definition = ""
        control_retention = ""
        control_execution = ""
    else:
        scene_token = "<Picture 2>"
        scene_definition = (
            "<Picture 2> is a real-scene anchor frame extracted from the existing source video. "
            "It supplies the exact real camera viewpoint, natural lighting, workspace geometry, "
            "background and object identities. Any source person visible in the anchor is removed "
            "and supplies no target pose, limb path, hand motion or action timing."
        )
        source_exclusion = (
            "preserve the real scene appearance, object identity, camera and framing, while deriving "
            "all temporal robot motion from <Subject 3> and <Video 1>."
        )
        real_scene_phrase = "the real room and camera shown in <Picture 2>"
        control_definition = (
            "\n<Video 1> is an intermediate action-control video compiled from the language "
            "instruction. It supplies the exact robot shoulder, elbow, wrist and hand trajectories, "
            "two-hand timing and final pose. Transfer only its motion; improve its robot rendering, "
            "ignore its CONTROL ONLY caption, and do not copy compression or rigid-part artifacts."
        )
        control_retention = (
            "\n<Video 1> (motion and timing reference only): attribute_transfer - transfer its "
            "visibly different arm trajectories and hand timing exactly to <Subject 1>, while "
            "taking identity from <Picture 1> and the real scene from <Picture 2>."
        )
        control_execution = (
            " Match the action-control pose in <Video 1> frame by frame; the language command "
            "defines its semantics and <Video 1> defines the explicit kinematics."
        )
    if scene_reference == "video":
        control_definition = ""
        control_retention = ""
        control_execution = ""
    return f"""subject_definitions:
<Subject 1> is the complete silver-and-graphite humanoid service robot in <Picture 1>, with a compact smooth head, a dark face panel, articulated shoulders, elbows and wrists, and two five-finger dexterous hands.
<Subject 2> is the real flower-arranging workspace in {scene_token}, including every flower, petal, stem, leaf, vase, glass, table, ribbon, shelf, paper roll, wall decoration, window, shadow and reflection.
<Subject 3> is the commanded target action: \"{instruction}\"
{scene_definition}
{control_definition}

summary:
[language action-conditioned real-scene video generation] Generate a photorealistic video in {scene_token}. Replace the source florist with <Subject 1>. Keep <Subject 2> and the camera fixed, while making the robot execute <Subject 3> instead of the source person's motion.

retention_analysis:
<Subject 1> (appears throughout [Shot 1]): fully_preserved - retain the robot's silver-and-graphite material, proportions, head, dark face panel, two articulated arms and two five-finger hands consistently in every frame.
<Subject 2> (appears throughout [Shot 1]): fully_preserved - retain the real workspace geometry, lighting, depth of field, flowers, vase, glass, table, tools and all background details. Existing objects may move only when physically contacted by the commanded robot action.
<Subject 3> (appears throughout [Shot 1]): fully_preserved - follow the commanded robot action and its explicit timeline; this language-conditioned action overrides any human motion visible in {scene_token}.
{scene_token} (scene and camera reference only): attribute_transfer - {source_exclusion}
{control_retention}

detailed_description:
The target is one photorealistic {duration_seconds:.3f}-second shot in {real_scene_phrase}, with the same static camera, framing, natural daylight, exposure, white balance, reflections and depth of field. The human florist is absent from the first frame onward. <Subject 1> occupies the same plausible work area and executes only this command: \"{instruction}\"{control_execution}
Action timeline: {timeline}
The commanded motion must be plainly visible and mechanically feasible: continuous shoulder-elbow-wrist kinematics, stable torso, coherent grasp contact, exactly two arms, exactly two hands and exactly five fingers on each hand. Preserve causal contact: a flower or tool moves only while held or pushed, never before contact and never after release. Preserve correct depth ordering when the robot passes behind or in front of flowers. Keep the robot identity and scale stable. Do not imitate the source florist's original motion, and do not average the command with that motion. Do not add human skin, face, hair, clothing, extra limbs, extra objects, text, logos, glow, sparks, camera motion, reframing, cuts or scene changes.

overall_soundscape:
No dialogue or newly invented sound is required; the validation targets are scene preservation, robot identity and visible compliance with the language-conditioned action.

non_diegetic_music:
None.
"""


def build_action_conditioned_tabletop_ref2va_prompt(
    duration_seconds: float,
    action: H3ActionVariant,
) -> str:
    """Build a three-reference H3 prompt for a real tabletop AC-WM test."""

    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    action.validate()
    instruction = " ".join(action.instruction.split())
    timeline = " ".join(action.timeline.split())
    return f"""subject_definitions:
<Subject 1> is the photorealistic silver dexterous robot hand and articulated forearm in <Picture 1>. Preserve its exact metallic panels, joints, five fingers, proportions and scale.
<Subject 2> is the real fixed-camera laboratory tabletop in <Picture 2>, including the white cloth, dark machine frame, cables, fixtures, shadows, reflections and one yellow handled bowl.
<Subject 3> is the commanded counterfactual action: "{instruction}"
<Video 1> is an intermediate action-control video compiled from <Subject 3>. It specifies the exact robot-hand trajectory, bowl trajectory, contact timing and terminal bowl state. Transfer its motion and object state only; ignore its CONTROL ONLY caption and rendering artifacts.

summary:
[action-conditioned world-model generation] Generate one photorealistic video in the real scene from <Picture 2>. Replace the source human arm with <Subject 1>. Execute <Subject 3> and make the yellow bowl reach the commanded terminal state shown by <Video 1>.

retention_analysis:
<Subject 1> (throughout [Shot 1]): fully_preserved - one coherent silver robot forearm, one five-finger hand, stable materials and geometry.
<Subject 2> (throughout [Shot 1]): fully_preserved - fixed real camera and unchanged laboratory background; only the robot and causally contacted bowl may move.
<Subject 3> (throughout [Shot 1]): fully_preserved - action semantics and terminal state override the motion in the source scene.
<Video 1> (motion/state reference only): attribute_transfer - follow its hand path, bowl path, contact interval and final position frame by frame while improving visual quality from <Picture 1> and <Picture 2>.

detailed_description:
The target is one photorealistic {duration_seconds:.3f}-second fixed-camera shot. Begin from the same bowl position and robot approach pose in all matched variants. The human arm is absent from frame one. Execute only: "{instruction}"
Action timeline: {timeline}
The hand must contact the yellow bowl before it moves and maintain causal contact while pushing, pulling or lifting it. The bowl remains one rigid yellow handled bowl with constant identity, plausible scale and correct occlusion. Its final image-plane location must visibly match <Video 1> and remain held for the final second. Keep the table cloth and upper machinery static. Do not average this command with another action. Do not add a second bowl, second arm, human skin, extra fingers, text, logos, camera motion, reframing, cuts, glow or particles.

overall_soundscape:
No dialogue or invented sound is required; the acceptance targets are action-conditioned bowl state, robot consistency and real-scene preservation.

non_diegetic_music:
None.
"""


@dataclass(frozen=True)
class MiniMaxH3ValidationConfig:
    """Validated inputs for one Ref2VA NF4 experiment."""

    source_video: Path
    robot_reference: Path
    prompt_file: Path
    diffsynth_repo: Path
    model_base_path: Path
    width: int = 832
    height: int = 480
    fps: int = 24
    num_frames: int = 124
    steps: int = 20
    seed: int = 20260810
    minimum_free_gpu_mib: int = 54 * 1024
    requested_gpu: int | None = None

    def validate(self) -> None:
        for label, path in (
            ("source video", self.source_video),
            ("robot reference", self.robot_reference),
            ("prompt file", self.prompt_file),
        ):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"{label} does not exist or is empty: {path}")
        if self.width <= 0 or self.height <= 0 or self.width % 32 or self.height % 32:
            raise ValueError("H3 width and height must be positive multiples of 32")
        if self.fps != 24:
            raise ValueError("the released H3 pipeline requires 24 FPS")
        if self.num_frames < 5 or (self.num_frames - 5) % 17:
            raise ValueError("H3 num_frames must satisfy num_frames = 17n + 5")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.minimum_free_gpu_mib <= 0:
            raise ValueError("minimum_free_gpu_mib must be positive")

    def select_gpu(self, inventory: list[GPUInfo]) -> GPUInfo:
        return select_gpu(inventory, self.requested_gpu, self.minimum_free_gpu_mib)


def verify_diffsynth_h3_source(repo: Path) -> str:
    """Require the reviewed H3-capable DiffSynth revision and source files."""

    required = (
        repo / "LICENSE",
        repo / "diffsynth" / "pipelines" / "minimax_h3_audio_video.py",
        repo / "diffsynth" / "models" / "minimax_h3_dit.py",
        repo / "diffsynth" / "models" / "minimax_h3_text_encoder.py",
        repo / "diffsynth" / "models" / "minimax_h3_video_vae.py",
        repo / "diffsynth" / "models" / "minimax_h3_audio_vae.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise PreflightError("DiffSynth H3 source files are missing: " + ", ".join(missing))
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip() if completed.returncode == 0 else ""
    if actual != DIFFSYNTH_H3_COMMIT:
        raise PreflightError(
            f"DiffSynth H3 source is {actual or 'unreadable'}, expected {DIFFSYNTH_H3_COMMIT}"
        )
    return actual
