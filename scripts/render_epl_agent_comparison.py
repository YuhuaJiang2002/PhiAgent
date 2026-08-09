#!/usr/bin/env python3
"""Render an EPL-conditioned versus EPL-masked repair-policy comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.training.epl_agent import (  # noqa: E402
    RepairAction,
    encode_example,
    feature_names,
    generate_policy_examples,
)

ACTION_COLORS = {
    RepairAction.CLAMP_LIMITS: (80, 155, 255),
    RepairAction.SMOOTH_TRAJECTORY: (85, 205, 145),
    RepairAction.RETIME_TRAJECTORY: (255, 180, 80),
    RepairAction.SHIFT_ALIGNMENT: (190, 125, 255),
    RepairAction.CONTACT_SAFE_REPLAN: (255, 95, 105),
    RepairAction.ACCEPT: (120, 220, 220),
}
DIAGNOSTIC_NAMES = (
    "joint limit",
    "position noise",
    "temporal scale",
    "timing shift",
    "contact loss",
    "collision",
    "reachability",
    "slip",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_file(root: Path, name: str) -> Path:
    matches = tuple(root.glob(f"*/{name}"))
    if len(matches) != 1:
        raise ValueError(f"expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def _load_model(checkpoint_path: Path):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = torch.nn.Sequential(
        torch.nn.Linear(checkpoint["input_dim"], checkpoint["hidden_dim"]),
        torch.nn.ReLU(),
        torch.nn.Linear(checkpoint["hidden_dim"], checkpoint["hidden_dim"]),
        torch.nn.ReLU(),
        torch.nn.Linear(checkpoint["hidden_dim"], len(checkpoint["actions"])),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _predict(model, examples, include_epl: bool) -> list[int]:
    import torch

    features = torch.tensor(
        [encode_example(example, include_epl=include_epl) for example in examples],
        dtype=torch.float32,
    )
    with torch.no_grad():
        return model(features).argmax(dim=1).tolist()


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def _center_text(draw, box, text: str, font, fill) -> None:
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    draw.text(
        ((left + right - width) / 2, (top + bottom - height) / 2),
        text,
        font=font,
        fill=fill,
    )


def _wrapped_action(action: RepairAction) -> str:
    words = action.name.split("_")
    if len(words) <= 2:
        return " ".join(words)
    midpoint = math.ceil(len(words) / 2)
    return " ".join(words[:midpoint]) + "\n" + " ".join(words[midpoint:])


def _center_multiline(draw, box, text: str, font, fill) -> None:
    left, top, right, bottom = box
    text_box = draw.multiline_textbbox((0, 0), text, font=font, spacing=8, align="center")
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    draw.multiline_text(
        ((left + right - width) / 2, (top + bottom - height) / 2),
        text,
        font=font,
        fill=fill,
        spacing=8,
        align="center",
    )


def _render_intro(path: Path, epl_accuracy: float, masked_accuracy: float) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1280, 720), (15, 20, 31))
    draw = ImageDraw.Draw(image)
    _center_text(
        draw,
        (80, 90, 1200, 210),
        "Does EPL help an agent choose repairs?",
        _font(48, bold=True),
        (238, 243, 255),
    )
    _center_text(
        draw,
        (80, 215, 1200, 275),
        "Matched held-out synthetic repair-action classification",
        _font(25),
        (155, 170, 195),
    )
    draw.rounded_rectangle((95, 340, 615, 555), radius=24, fill=(45, 50, 65))
    draw.rounded_rectangle((665, 340, 1185, 555), radius=24, fill=(35, 70, 58))
    _center_text(draw, (95, 355, 615, 425), "WITHOUT EPL", _font(28, True), (225, 230, 240))
    _center_text(draw, (665, 355, 1185, 425), "WITH EPL", _font(28, True), (225, 245, 235))
    _center_text(
        draw,
        (95, 430, 615, 535),
        f"{masked_accuracy:.1%}",
        _font(64, True),
        (255, 185, 95),
    )
    _center_text(
        draw,
        (665, 430, 1185, 535),
        f"{epl_accuracy:.1%}",
        _font(64, True),
        (95, 235, 165),
    )
    _center_text(
        draw,
        (80, 610, 1200, 675),
        "Same model, seed, split and diagnostics; only EPL phase/contact features differ",
        _font(22),
        (185, 195, 215),
    )
    image.save(path)


def _draw_prediction_panel(
    draw,
    box,
    title: str,
    prediction: RepairAction,
    truth: RepairAction,
    show_epl: bool,
) -> None:
    left, top, right, bottom = box
    correct = prediction is truth
    background = (31, 84, 62) if correct else (94, 38, 45)
    draw.rounded_rectangle(box, radius=22, fill=background)
    _center_text(draw, (left, top + 12, right, top + 65), title, _font(25, True), (245, 248, 255))
    _center_multiline(
        draw,
        (left + 25, top + 90, right - 25, top + 190),
        _wrapped_action(prediction),
        _font(27, True),
        ACTION_COLORS[prediction],
    )
    _center_text(
        draw,
        (left, top + 205, right, top + 250),
        "CORRECT" if correct else "WRONG",
        _font(24, True),
        (120, 245, 170) if correct else (255, 145, 145),
    )
    _center_text(
        draw,
        (left + 15, top + 270, right - 15, bottom - 15),
        "phase + contact visible" if show_epl else "phase + contact masked",
        _font(19),
        (205, 220, 225),
    )


def _render_case(path: Path, case_index: int, case) -> None:
    from PIL import Image, ImageDraw

    example, masked_prediction, epl_prediction = case
    truth = example.action
    image = Image.new("RGB", (1280, 720), (15, 20, 31))
    draw = ImageDraw.Draw(image)
    draw.text((55, 35), f"Held-out masked failure #{case_index + 1}", font=_font(31, True), fill=(238, 243, 255))
    draw.text(
        (55, 87),
        f"EPL phase: {example.phase.value.upper()}     contact: {example.contact_state.value.upper()}",
        font=_font(25, True),
        fill=(115, 205, 255),
    )
    draw.text(
        (55, 130),
        f"Ground truth repair: {truth.name.replace('_', ' ')}",
        font=_font(25, True),
        fill=ACTION_COLORS[truth],
    )
    draw.text((55, 190), "Agent diagnostics", font=_font(22, True), fill=(210, 220, 238))
    for index, (name, value) in enumerate(zip(DIAGNOSTIC_NAMES, example.diagnostics)):
        y = 235 + index * 48
        draw.text((60, y), name, font=_font(18), fill=(175, 188, 210))
        draw.rounded_rectangle((205, y + 2, 490, y + 24), radius=8, fill=(42, 50, 66))
        draw.rounded_rectangle(
            (205, y + 2, 205 + min(value, 1.0) * 285, y + 24),
            radius=8,
            fill=(95, 175, 245),
        )
        draw.text((500, y), f"{value:.2f}", font=_font(17), fill=(185, 195, 215))
    _draw_prediction_panel(
        draw,
        (600, 190, 910, 620),
        "WITHOUT EPL",
        RepairAction(masked_prediction),
        truth,
        False,
    )
    _draw_prediction_panel(
        draw,
        (935, 190, 1245, 620),
        "WITH EPL",
        RepairAction(epl_prediction),
        truth,
        True,
    )
    draw.text(
        (55, 675),
        "Synthetic diagnostic example - not a robot rollout",
        font=_font(17),
        fill=(130, 140, 160),
    )
    image.save(path)


def _render_summary(path: Path, epl_accuracy: float, masked_accuracy: float, count: int) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1280, 720), (15, 20, 31))
    draw = ImageDraw.Draw(image)
    _center_text(draw, (80, 75, 1200, 165), "Matched test-set result", _font(45, True), (238, 243, 255))
    draw.text((130, 250), "WITHOUT EPL", font=_font(29, True), fill=(220, 225, 235))
    draw.text((130, 330), f"{masked_accuracy:.1%}", font=_font(74, True), fill=(255, 180, 90))
    draw.text((720, 250), "WITH EPL", font=_font(29, True), fill=(220, 240, 230))
    draw.text((720, 330), f"{epl_accuracy:.1%}", font=_font(74, True), fill=(95, 235, 165))
    gain = epl_accuracy - masked_accuracy
    _center_text(
        draw,
        (80, 455, 1200, 545),
        f"EPL gain: +{gain * 100:.1f} percentage points",
        _font(37, True),
        (115, 205, 255),
    )
    _center_text(
        draw,
        (80, 570, 1200, 650),
        f"{count} held-out examples; displayed cases are the first 8 masked failures corrected by EPL",
        _font(20),
        (165, 178, 200),
    )
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epl-arm", type=Path, required=True)
    parser.add_argument("--masked-arm", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/epl-agent-video"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--case-seconds", type=float, default=1.2)
    parser.add_argument("--maximum-cases", type=int, default=8)
    args = parser.parse_args()
    if args.fps <= 0 or args.case_seconds <= 0 or args.maximum_cases < 1:
        raise ValueError("FPS, case duration, and maximum cases must be positive")

    epl_metadata_path = _single_file(args.epl_arm, "metadata.json")
    masked_metadata_path = _single_file(args.masked_arm, "metadata.json")
    epl_checkpoint = _single_file(args.epl_arm, "policy.pt")
    masked_checkpoint = _single_file(args.masked_arm, "policy.pt")
    epl_metadata = json.loads(epl_metadata_path.read_text())
    masked_metadata = json.loads(masked_metadata_path.read_text())
    seed = int(epl_metadata["config"]["seed"])
    if seed != int(masked_metadata["config"]["seed"]):
        raise ValueError("comparison arms must use the same seed")
    for key in ("examples", "epochs", "batch_size", "hidden_dim", "learning_rate"):
        if epl_metadata["config"][key] != masked_metadata["config"][key]:
            raise ValueError(f"comparison arms differ in {key}")

    examples = generate_policy_examples(int(epl_metadata["config"]["examples"]), seed)
    indices = list(range(len(examples)))
    random.Random(seed + 1).shuffle(indices)
    test_indices = indices[round(len(indices) * 0.85) :]
    test_examples = [examples[index] for index in test_indices]
    epl_model = _load_model(epl_checkpoint)
    masked_model = _load_model(masked_checkpoint)
    epl_predictions = _predict(epl_model, test_examples, include_epl=True)
    masked_predictions = _predict(masked_model, test_examples, include_epl=False)
    truths = [int(example.action) for example in test_examples]
    epl_accuracy = sum(p == t for p, t in zip(epl_predictions, truths)) / len(truths)
    masked_accuracy = sum(p == t for p, t in zip(masked_predictions, truths)) / len(truths)
    selected = [
        (example, masked_prediction, epl_prediction)
        for example, masked_prediction, epl_prediction, truth in zip(
            test_examples, masked_predictions, epl_predictions, truths
        )
        if masked_prediction != truth and epl_prediction == truth
    ][: args.maximum_cases]
    if not selected:
        raise ValueError("no masked failures corrected by EPL were found")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = args.experiment_root.expanduser().resolve() / f"{stamp}-{uuid4().hex[:8]}"
    frames_dir = experiment / "frames"
    frames_dir.mkdir(parents=True)
    intro = frames_dir / "intro.png"
    summary_frame = frames_dir / "summary.png"
    _render_intro(intro, epl_accuracy, masked_accuracy)
    case_frames = []
    for index, case in enumerate(selected):
        frame = frames_dir / f"case-{index:02d}.png"
        _render_case(frame, index, case)
        case_frames.append(frame)
    _render_summary(summary_frame, epl_accuracy, masked_accuracy, len(test_examples))

    concat_path = experiment / "frames.txt"
    lines = [f"file '{intro}'", "duration 2.0"]
    for frame in case_frames:
        lines.extend((f"file '{frame}'", f"duration {args.case_seconds}"))
    lines.extend((f"file '{summary_frame}'", "duration 2.5", f"file '{summary_frame}'"))
    concat_path.write_text("\n".join(lines) + "\n")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the comparison video")
    output = experiment / "epl-comparison.mp4"
    command = [
        ffmpeg,
        "-v",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-vf",
        f"fps={args.fps},format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        str(output),
    ]
    subprocess.run(command, check=True)
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "synthetic_epl_agent_policy_comparison",
        "seed": seed,
        "selection_rule": "first masked failures on held-out split corrected by EPL",
        "selected_cases": len(selected),
        "held_out_examples": len(test_examples),
        "epl_accuracy": epl_accuracy,
        "masked_accuracy": masked_accuracy,
        "gain": epl_accuracy - masked_accuracy,
        "feature_names": feature_names(),
        "epl_metadata": str(epl_metadata_path.resolve()),
        "epl_metadata_sha256": _sha256(epl_metadata_path),
        "masked_metadata": str(masked_metadata_path.resolve()),
        "masked_metadata_sha256": _sha256(masked_metadata_path),
        "epl_checkpoint_sha256": _sha256(epl_checkpoint),
        "masked_checkpoint_sha256": _sha256(masked_checkpoint),
        "ffmpeg_command": command,
        "output": str(output),
        "output_sha256": _sha256(output),
        "limitations": [
            "This video visualizes synthetic repair-action classification.",
            "It is not generated robot motion, simulation, or real-robot evidence.",
        ],
    }
    manifest_path = experiment / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"experiment": str(experiment), **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
