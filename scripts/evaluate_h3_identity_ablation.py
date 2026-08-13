#!/usr/bin/env python3
"""Frozen DINOv2 plus pixel-contract evaluation for an H3 identity ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402
from phiagent.training.h3_identity_rsi import (  # noqa: E402
    IdentityMetrics,
    IdentityPromotionContract,
    TopologyReviewEvidence,
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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


def _decode(path: Path) -> list[object]:
    import av

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_image().convert("RGB"))
    if not frames:
        raise ValueError(f"video contains no decodable frames: {path}")
    return frames


def _masked_crop(image: object, mask: object) -> object:
    from PIL import Image

    resized_mask = mask.resize(image.size, Image.Resampling.NEAREST)
    bbox = resized_mask.getbbox()
    if bbox is None:
        raise ValueError("identity mask is empty")
    black = Image.new("RGB", image.size)
    selected = Image.composite(image, black, resized_mask)
    return selected.crop(bbox)


def _features(model: object, processor: object, crops: list[object], device: str) -> object:
    import torch
    import torch.nn.functional as functional

    batches = []
    for start in range(0, len(crops), 8):
        inputs = processor(images=crops[start : start + 8], return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            embedding = outputs.pooler_output
            batches.append(functional.normalize(embedding.float(), dim=-1).cpu())
    return torch.cat(batches)


def _unit_cosines(reference: object, features: object) -> list[float]:
    import torch

    values = torch.matmul(features, reference).tolist()
    return [max(0.0, min(1.0, (float(value) + 1.0) / 2.0)) for value in values]


def _adjacent_unit_cosines(features: object) -> list[float]:
    import torch

    if len(features) < 2:
        return [1.0]
    values = torch.sum(features[:-1] * features[1:], dim=-1).tolist()
    return [max(0.0, min(1.0, (float(value) + 1.0) / 2.0)) for value in values]


def _pixel_arrays(frames: list[object], width: int = 224, height: int = 128) -> object:
    import numpy as np
    from PIL import Image

    return np.stack(
        [
            np.asarray(
                frame.resize((width, height), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            / 255.0
            for frame in frames
        ]
    )


def _scene_score(frames: list[object], scene: object, mask: object) -> float:
    import numpy as np
    from PIL import Image

    width, height = 224, 128
    scene_array = (
        np.asarray(scene.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
        / 255.0
    )
    outside = (
        np.asarray(mask.resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8) < 128
    )
    if not outside.any():
        raise ValueError("identity mask leaves no scene pixels")
    values = []
    for frame in frames:
        array = (
            np.asarray(frame.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
            / 255.0
        )
        values.append(float(np.abs(array - scene_array)[outside].mean()))
    return max(0.0, min(1.0, 1.0 - sum(values) / len(values)))


def _temporal_score(arrays: object) -> float:
    import numpy as np

    if len(arrays) < 3:
        return 1.0
    deltas = np.mean(np.abs(arrays[1:] - arrays[:-1]), axis=(1, 2, 3))
    jerk = float(np.mean(np.abs(deltas[1:] - deltas[:-1])))
    return max(0.0, min(1.0, 1.0 - 4.0 * jerk))


def _motion_profile_score(baseline_arrays: object, candidate_arrays: object) -> float:
    import numpy as np

    baseline = np.mean(np.abs(baseline_arrays[1:] - baseline_arrays[:-1]), axis=(1, 2, 3))
    candidate = np.mean(np.abs(candidate_arrays[1:] - candidate_arrays[:-1]), axis=(1, 2, 3))
    scale = max(float(np.mean(np.abs(baseline))), 1e-3)
    error = float(np.mean(np.abs(candidate - baseline))) / scale
    return max(0.0, min(1.0, math.exp(-error)))


def _finite_action_value(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"action {name} must be finite and in [0, 1], got {value!r}")
    return number


ACTION_METRIC_NAMES = ("motion_preservation", "epl_minimum", "object_lock")
ACTION_CONTEXT_INPUTS = ("source", "motion_reference", "robot_reference", "anchor_mask")


def _action_input_digest(payload: dict[str, object], name: str, path: Path) -> str:
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get(name), dict):
        raise ValueError(f"action evaluation has no {name!r} input evidence: {path}")
    digest = inputs[name].get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"action evaluation has invalid {name!r} SHA-256 evidence: {path}")
    return digest


def _action_score(
    path: Path, video_sha256: str
) -> tuple[dict[str, float], dict[str, object]]:
    """Load the scorecard bound to ``video_sha256``, failing closed on mismatch."""

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"action evaluation must be a JSON object: {path}")
    inputs = payload.get("inputs")
    outputs = payload.get("outputs")
    rounds = payload.get("rounds")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict) or not isinstance(rounds, list):
        raise ValueError(f"action evaluation lacks digest-bound inputs/outputs/rounds: {path}")

    scorecard: object = None
    binding: dict[str, object] | None = None
    raw_h3 = inputs.get("raw_h3")
    if isinstance(raw_h3, dict) and raw_h3.get("sha256") == video_sha256:
        raw_rounds = [
            entry
            for entry in rounds
            if isinstance(entry, dict)
            and isinstance(entry.get("repair"), dict)
            and entry["repair"].get("name") == "raw-h3"
        ]
        if len(raw_rounds) != 1:
            raise ValueError(f"action evaluation must contain exactly one raw-h3 round: {path}")
        scorecard = raw_rounds[0].get("scorecard")
        binding = {
            "scope": "inputs.raw_h3",
            "video_sha256": video_sha256,
            "round": raw_rounds[0].get("round"),
            "scored_output_sha256": raw_rounds[0].get("output_sha256"),
        }
    elif outputs.get("final_sha256") == video_sha256:
        scorecard = payload.get("best_scorecard")
        binding = {"scope": "outputs.final", "video_sha256": video_sha256}
    else:
        matching_rounds = [
            entry
            for entry in rounds
            if isinstance(entry, dict) and entry.get("output_sha256") == video_sha256
        ]
        if len(matching_rounds) == 1:
            scorecard = matching_rounds[0].get("scorecard")
            binding = {
                "scope": "round.output",
                "video_sha256": video_sha256,
                "round": matching_rounds[0].get("round"),
            }
    if binding is None:
        raise ValueError(
            f"action evaluation is not bound to candidate video SHA-256 {video_sha256}: {path}"
        )
    if not isinstance(scorecard, dict):
        raise ValueError(f"action evaluation binding has no scorecard: {path}")
    values = {
        name: _finite_action_value(scorecard.get(name), name) for name in ACTION_METRIC_NAMES
    }
    return values, binding


def _action_adherence(
    baseline: dict[str, float], candidate: dict[str, float]
) -> tuple[float, dict[str, float]]:
    """Require every action component to avoid relative regression."""

    ratios = {
        name: (
            1.0
            if baseline[name] == 0.0 and candidate[name] >= baseline[name]
            else min(1.0, candidate[name] / max(baseline[name], 1e-9))
        )
        for name in ACTION_METRIC_NAMES
    }
    return min(ratios.values()), ratios


def _validate_action_context(
    baseline_path: Path, candidate_path: Path
) -> dict[str, str]:
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise ValueError("action evaluations must be JSON objects")
    matched: dict[str, str] = {}
    for name in ACTION_CONTEXT_INPUTS:
        baseline_digest = _action_input_digest(baseline, name, baseline_path)
        candidate_digest = _action_input_digest(candidate, name, candidate_path)
        if baseline_digest != candidate_digest:
            raise ValueError(
                f"action evaluation context mismatch for {name!r}: "
                f"{baseline_digest} != {candidate_digest}"
            )
        matched[name] = baseline_digest
    return matched


def _build_metrics(
    *,
    features: object,
    reference: object,
    frames: list[object],
    scene: object,
    mask: object,
    arrays: object,
    motion_adherence: float,
    action_adherence: float,
    topology_integrity: float,
) -> tuple[IdentityMetrics, dict[str, object]]:
    import torch

    similarities = _unit_cosines(reference, features)
    temporal_cosines = _adjacent_unit_cosines(features)
    appearance_proxy = min(min(similarities), min(temporal_cosines))
    metrics = IdentityMetrics(
        reference_identity_mean=sum(similarities) / len(similarities),
        reference_identity_worst=min(similarities),
        cross_frame_identity=sum(temporal_cosines) / len(temporal_cosines),
        topology_integrity=topology_integrity,
        motion_adherence=motion_adherence,
        action_adherence=action_adherence,
        scene_preservation=_scene_score(frames, scene, mask),
        temporal_consistency=_temporal_score(arrays),
    )
    diagnostics = {
        "reference_similarities": similarities,
        "adjacent_frame_similarities": temporal_cosines,
        "reference_similarity_std": float(torch.tensor(similarities).std(unbiased=False)),
        "appearance_proxy_not_used_as_topology": appearance_proxy,
        "topology_metric": "full-frame SHA-bound semantic evidence passing fraction",
    }
    return metrics, diagnostics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--scene-image", type=Path, required=True)
    parser.add_argument("--identity-mask", type=Path, required=True)
    parser.add_argument("--baseline-topology-evidence", type=Path, required=True)
    parser.add_argument("--candidate-topology-evidence", type=Path, required=True)
    parser.add_argument("--baseline-action-evaluation", type=Path, required=True)
    parser.add_argument("--candidate-action-evaluation", type=Path, required=True)
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=6 * 1024)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = {
        "baseline": args.baseline.expanduser().resolve(),
        "candidate": args.candidate.expanduser().resolve(),
        "reference_image": args.reference_image.expanduser().resolve(),
        "scene_image": args.scene_image.expanduser().resolve(),
        "identity_mask": args.identity_mask.expanduser().resolve(),
        "baseline_topology_evidence": args.baseline_topology_evidence.expanduser().resolve(),
        "candidate_topology_evidence": args.candidate_topology_evidence.expanduser().resolve(),
        "baseline_action_evaluation": args.baseline_action_evaluation.expanduser().resolve(),
        "candidate_action_evaluation": args.candidate_action_evaluation.expanduser().resolve(),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{name} is missing or empty: {path}")
    gpus, inventory_raw, processes_raw = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    os.environ["HF_HOME"] = str(args.model_cache.expanduser().resolve())

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda"
    model_source = Path(args.model).expanduser()
    load_options: dict[str, object] = {"cache_dir": os.environ["HF_HOME"]}
    if model_source.exists():
        model_id = str(model_source.resolve())
        load_options["local_files_only"] = True
    else:
        model_id = args.model
        load_options["revision"] = args.model_revision
    processor = AutoImageProcessor.from_pretrained(model_id, **load_options)
    model = AutoModel.from_pretrained(model_id, **load_options).eval().to(device)
    baseline_frames = _decode(paths["baseline"])
    candidate_frames = _decode(paths["candidate"])
    if len(baseline_frames) != len(candidate_frames):
        raise ValueError("baseline and candidate frame counts differ")
    baseline_topology = TopologyReviewEvidence.load(paths["baseline_topology_evidence"])
    candidate_topology = TopologyReviewEvidence.load(paths["candidate_topology_evidence"])
    if baseline_topology.video_sha256 != _sha256(paths["baseline"]):
        raise ValueError("baseline topology evidence digest does not match the video")
    if candidate_topology.video_sha256 != _sha256(paths["candidate"]):
        raise ValueError("candidate topology evidence digest does not match the video")
    if baseline_topology.total_frames != len(baseline_frames):
        raise ValueError("baseline topology evidence frame count does not match the video")
    if candidate_topology.total_frames != len(candidate_frames):
        raise ValueError("candidate topology evidence frame count does not match the video")
    contract = IdentityPromotionContract()
    reference_image = Image.open(paths["reference_image"]).convert("RGB")
    scene_image = Image.open(paths["scene_image"]).convert("RGB")
    mask = Image.open(paths["identity_mask"]).convert("L")
    reference_crop = _masked_crop(reference_image, mask)
    baseline_crops = [_masked_crop(frame, mask) for frame in baseline_frames]
    candidate_crops = [_masked_crop(frame, mask) for frame in candidate_frames]
    reference_feature = _features(model, processor, [reference_crop], device)[0]
    baseline_features = _features(model, processor, baseline_crops, device)
    candidate_features = _features(model, processor, candidate_crops, device)
    baseline_arrays = _pixel_arrays(baseline_frames)
    candidate_arrays = _pixel_arrays(candidate_frames)
    baseline_video_sha256 = _sha256(paths["baseline"])
    candidate_video_sha256 = _sha256(paths["candidate"])
    action_context = _validate_action_context(
        paths["baseline_action_evaluation"], paths["candidate_action_evaluation"]
    )
    baseline_action_score, baseline_action_binding = _action_score(
        paths["baseline_action_evaluation"], baseline_video_sha256
    )
    candidate_action_score, candidate_action_binding = _action_score(
        paths["candidate_action_evaluation"], candidate_video_sha256
    )
    action_adherence, action_component_ratios = _action_adherence(
        baseline_action_score, candidate_action_score
    )
    baseline_metrics, baseline_diagnostics = _build_metrics(
        features=baseline_features,
        reference=reference_feature,
        frames=baseline_frames,
        scene=scene_image,
        mask=mask,
        arrays=baseline_arrays,
        motion_adherence=1.0,
        action_adherence=1.0,
        topology_integrity=baseline_topology.passing_fraction(
            contract.minimum_topology_review_confidence
        ),
    )
    candidate_metrics, candidate_diagnostics = _build_metrics(
        features=candidate_features,
        reference=reference_feature,
        frames=candidate_frames,
        scene=scene_image,
        mask=mask,
        arrays=candidate_arrays,
        motion_adherence=_motion_profile_score(baseline_arrays, candidate_arrays),
        action_adherence=action_adherence,
        topology_integrity=candidate_topology.passing_fraction(
            contract.minimum_topology_review_confidence
        ),
    )
    assessment = contract.assess(baseline_metrics, candidate_metrics, candidate_topology)
    model_inventory_root = Path(model_id) if model_source.exists() else Path(os.environ["HF_HOME"])
    model_files = sorted(
        path
        for path in model_inventory_root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".safetensors"}
    )
    payload = {
        "schema_version": "1.0.0",
        "method": "frozen_dinov2_identity_plus_pixel_non_regression_contract",
        "status": "WORKING" if assessment.passed else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "git": _git_state(Path(__file__).resolve().parents[1]),
        "seed": 0,
        "deterministic_feature_extraction": True,
        "selected_gpu": asdict(selected),
        "gpu_inventory_raw": inventory_raw,
        "gpu_processes_raw": processes_raw,
        "torch": torch.__version__,
        "model": args.model,
        "resolved_model": model_id,
        "model_revision": args.model_revision,
        "model_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in model_files
        ],
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
        },
        "frames": len(baseline_frames),
        "baseline": {"metrics": asdict(baseline_metrics), "diagnostics": baseline_diagnostics},
        "candidate": {
            "metrics": asdict(candidate_metrics),
            "diagnostics": candidate_diagnostics,
        },
        "action_evidence": {
            "matched_context_sha256": action_context,
            "baseline": {
                "metrics": baseline_action_score,
                "binding": baseline_action_binding,
            },
            "candidate": {
                "metrics": candidate_action_score,
                "binding": candidate_action_binding,
            },
            "component_ratios": action_component_ratios,
            "aggregation": "minimum normalized candidate/baseline ratio over every action component",
        },
        "contract": asdict(contract),
        "assessment": assessment.to_dict(),
        "limitations": [
            "DINOv2 masked similarity is used only for appearance identity, never as topology evidence.",
            "Motion non-regression measures similarity to the matched baseline motion profile, not robot-base control accuracy.",
            "Action evidence must be SHA-bound to each assessed video and share source, motion-control, robot-reference, and mask context.",
            "Action non-regression is the minimum normalized candidate/baseline ratio over task-motion, EPL, and object-lock; one regressing component cannot be hidden by another.",
            "A publication candidate still requires part-level topology review and at least three held-out identities/scenes.",
        ],
    }
    _write_json(output, payload)
    print(json.dumps(payload["assessment"], indent=2, sort_keys=True))
    return 0 if assessment.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
