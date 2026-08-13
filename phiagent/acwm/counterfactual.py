"""Leakage-safe metadata construction for BWM action counterfactual audits."""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from phiagent.acwm.promotion import paired_bootstrap_lower_bound


def _require_mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"metadata row requires an object-valued {key!r}")
    return value


def _require_path(payload: Mapping[str, Any], label: str) -> str:
    value = payload.get("data")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} requires a non-empty data path")
    return value


def validate_counterfactual_sources(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate frozen test rows before actions are exchanged between episodes."""

    if len(rows) < 2:
        raise ValueError("counterfactual construction requires at least two source rows")
    normalized: list[dict[str, Any]] = []
    source_episodes: set[str] = set()
    episode_indices: set[int] = set()
    contracts: set[tuple[str, int, int]] = set()
    independent_groups: set[str] = set()
    for position, source in enumerate(rows):
        row = copy.deepcopy(dict(source))
        if row.get("split") != "test":
            raise ValueError(f"source row {position} is not from the frozen test split")
        source_episode = row.get("source_episode")
        if not isinstance(source_episode, str) or not source_episode.strip():
            raise ValueError(f"source row {position} requires source_episode")
        if source_episode in source_episodes:
            raise ValueError(f"duplicate source_episode {source_episode!r}")
        source_episodes.add(source_episode)
        independent_group = row.get("independent_group_id")
        if not isinstance(independent_group, str) or not independent_group.strip():
            raise ValueError(
                f"source row {position} requires a lineage-derived independent_group_id"
            )
        independent_groups.add(independent_group)
        try:
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            history_frames = int(row["history_frames"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"source row {position} has invalid integer fields") from exc
        if episode_index < 0 or episode_index in episode_indices:
            raise ValueError(f"invalid or duplicate episode_index {episode_index}")
        episode_indices.add(episode_index)
        if length < 2 or not 0 < history_frames < length:
            raise ValueError("history_frames must be positive and shorter than length")
        coordinate_frame = row.get("coordinate_frame")
        if (
            not isinstance(coordinate_frame, str)
            or not coordinate_frame.startswith("robot_base:")
        ):
            raise ValueError("BWM counterfactual actions require an explicit robot_base frame")
        action = _require_mapping(row, "action")
        video = _require_mapping(row, "video")
        _require_path(action, "action")
        _require_path(video, "video")
        for label, payload in (("action", action), ("video", video)):
            start = int(payload.get("start_frame", -1))
            end = int(payload.get("end_frame", -1))
            if start < 0 or end - start + 1 != length:
                raise ValueError(f"{label} interval does not match row length")
        contracts.add((coordinate_frame, length, history_frames))
        normalized.append(row)
    if len(independent_groups) < 2:
        raise ValueError(
            "counterfactual construction requires at least two independent source groups"
        )
    if len(contracts) != 1:
        raise ValueError(
            "counterfactual sources must share coordinate frame, length, and history"
        )
    return tuple(sorted(normalized, key=lambda row: str(row["source_episode"])))


def build_action_swap_suite(
    rows: Sequence[Mapping[str, Any]],
    *,
    episode_index_start: int = 100_000,
    swapped_action_by_source_episode: Mapping[str, str] | None = None,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Build factual/action-swapped pairs with a deterministic derangement.

    The source video and prompt remain unchanged. Only the complete, frame-aligned
    action sequence is exchanged with the next sorted test episode.
    """

    if episode_index_start < 0:
        raise ValueError("episode_index_start must be non-negative")
    sources = validate_counterfactual_sources(rows)
    output_rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        donor = next(
            (
                sources[(source_index + offset) % len(sources)]
                for offset in range(1, len(sources))
                if sources[(source_index + offset) % len(sources)][
                    "independent_group_id"
                ]
                != source["independent_group_id"]
            ),
            None,
        )
        if donor is None:
            raise ValueError("no cross-group action donor is available")
        if source["source_episode"] == donor["source_episode"]:
            raise ValueError("action donor must differ from the source episode")
        factual_index = episode_index_start + source_index * 2
        swapped_index = factual_index + 1
        trial_id = f"{source['source_episode']}::action-swap"
        common = {
            "schema_version": "1.0.0",
            "trial_id": trial_id,
            "source_episode": source["source_episode"],
            "source_episode_index": int(source["episode_index"]),
            "independent_group_id": source["independent_group_id"],
            "action_coordinate_frame": source["coordinate_frame"],
            "claim_scope": "diagnostic_action_sensitivity_without_counterfactual_ground_truth",
        }
        factual = copy.deepcopy(source)
        factual["episode_index"] = factual_index
        factual["group_id"] = f"{source['group_id']}::factual"
        factual["counterfactual"] = {
            **common,
            "variant": "factual",
            "action_source_episode": source["source_episode"],
            "action_source_episode_index": int(source["episode_index"]),
            "paired_episode_index": swapped_index,
        }
        swapped = copy.deepcopy(source)
        swapped["episode_index"] = swapped_index
        swapped["group_id"] = f"{source['group_id']}::swapped"
        if swapped_action_by_source_episode is None:
            swapped["action"] = copy.deepcopy(donor["action"])
        else:
            derived_path = swapped_action_by_source_episode.get(
                str(source["source_episode"])
            )
            if not isinstance(derived_path, str) or not derived_path.strip():
                raise ValueError(
                    f"missing derived action for {source['source_episode']!r}"
                )
            swapped["action"] = copy.deepcopy(source["action"])
            swapped["action"]["data"] = derived_path
            swapped["action"]["start_frame"] = 0
            swapped["action"]["end_frame"] = int(source["length"]) - 1
        swapped["counterfactual"] = {
            **common,
            "variant": "swapped",
            "action_source_episode": donor["source_episode"],
            "action_source_episode_index": int(donor["episode_index"]),
            "paired_episode_index": factual_index,
        }
        output_rows.extend((factual, swapped))
        pairs.append(
            {
                "trial_id": trial_id,
                "source_episode": source["source_episode"],
                "source_episode_index": int(source["episode_index"]),
                "independent_group_id": source["independent_group_id"],
                "factual_episode_index": factual_index,
                "swapped_episode_index": swapped_index,
                "factual_action": _require_path(source["action"], "action"),
                "swapped_action": _require_path(swapped["action"], "action"),
                "donor_action": _require_path(donor["action"], "action"),
                "swapped_action_source_episode": donor["source_episode"],
                "swapped_action_independent_group_id": donor["independent_group_id"],
            }
        )
    return tuple(output_rows), tuple(pairs)


def _wrap_radians(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _quaternion_normalize(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError("quaternion must contain x, y, z, w")
    quaternion = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-8:
        raise ValueError("quaternion norm must be positive")
    normalized = tuple(value / norm for value in quaternion)
    return normalized[0], normalized[1], normalized[2], normalized[3]


def _quaternion_multiply(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = _quaternion_normalize(left)
    rx, ry, rz, rw = _quaternion_normalize(right)
    return _quaternion_normalize(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def _quaternion_inverse(values: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, z, w = _quaternion_normalize(values)
    return -x, -y, -z, w


def rebase_absolute_eef_future(
    source: Sequence[Sequence[float]],
    donor: Sequence[Sequence[float]],
    *,
    history_frames: int,
    rotation_representation: str = "euler_gripper",
) -> tuple[tuple[float, ...], ...]:
    """Preserve source history and apply the donor's future EEF displacement.

    Future donor poses are expressed relative to the donor's last history state
    and rebased onto the source's corresponding state. ``euler_gripper`` uses
    two XYZ + Euler XYZ + gripper blocks. ``quaternion`` uses two XYZ +
    quaternion XYZW blocks and composes relative rotations on SO(3).
    """

    source_rows = tuple(tuple(float(value) for value in row) for row in source)
    donor_rows = tuple(tuple(float(value) for value in row) for row in donor)
    if len(source_rows) != len(donor_rows) or len(source_rows) < 2:
        raise ValueError("source and donor actions must have equal non-trivial length")
    if not 0 < history_frames < len(source_rows):
        raise ValueError("history_frames must be positive and shorter than the sequence")
    if any(len(row) != 14 for row in (*source_rows, *donor_rows)):
        raise ValueError("absolute EEF rows must contain exactly 14 channels")
    if any(not math.isfinite(value) for row in (*source_rows, *donor_rows) for value in row):
        raise ValueError("absolute EEF rows must contain only finite values")
    if rotation_representation not in {"euler_gripper", "quaternion"}:
        raise ValueError("rotation_representation must be euler_gripper or quaternion")
    anchor_index = history_frames - 1
    source_anchor = source_rows[anchor_index]
    donor_anchor = donor_rows[anchor_index]
    result = list(source_rows[:history_frames])
    position_channels = (0, 1, 2, 7, 8, 9)
    for donor_row in donor_rows[history_frames:]:
        row = list(source_anchor)
        for channel in position_channels:
            row[channel] = source_anchor[channel] + (
                donor_row[channel] - donor_anchor[channel]
            )
        if rotation_representation == "euler_gripper":
            for channel in (3, 4, 5, 10, 11, 12):
                delta = _wrap_radians(donor_row[channel] - donor_anchor[channel])
                row[channel] = _wrap_radians(source_anchor[channel] + delta)
            for channel in (6, 13):
                row[channel] = donor_row[channel]
        else:
            for start in (3, 10):
                donor_delta = _quaternion_multiply(
                    _quaternion_inverse(donor_anchor[start : start + 4]),
                    donor_row[start : start + 4],
                )
                row[start : start + 4] = _quaternion_multiply(
                    source_anchor[start : start + 4],
                    donor_delta,
                )
        result.append(tuple(row))
    return tuple(result)


def aggregate_counterfactual_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Average inference seeds within each independent source episode."""

    if not records:
        raise ValueError("counterfactual aggregation requires records")
    grouped: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    independent_unit_by_record: dict[tuple[str, str], str] = {}
    for record in records:
        model_id = str(record.get("model_id", "")).strip()
        trial_id = str(record.get("trial_id", "")).strip()
        if not model_id or not trial_id:
            raise ValueError("every counterfactual record requires model_id and trial_id")
        try:
            seed = int(record["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("every counterfactual record requires an integer seed") from exc
        raw_metrics = record.get("metrics")
        if not isinstance(raw_metrics, Mapping) or not raw_metrics:
            raise ValueError("every counterfactual record requires metrics")
        metrics = {str(key): float(value) for key, value in raw_metrics.items()}
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError("counterfactual metrics must be finite")
        trial = grouped.setdefault(model_id, {}).setdefault(trial_id, {})
        if seed in trial:
            raise ValueError(f"duplicate record for {model_id}/{trial_id}/seed-{seed}")
        trial[seed] = metrics
        independent_unit = str(record.get("independent_unit_id", trial_id)).strip()
        if not independent_unit:
            raise ValueError("independent_unit_id cannot be empty")
        key = model_id, trial_id
        previous_unit = independent_unit_by_record.setdefault(key, independent_unit)
        if previous_unit != independent_unit:
            raise ValueError("a trial cannot change independent_unit_id across seeds")
    models: dict[str, dict[str, Any]] = {}
    expected_trials: set[str] | None = None
    expected_seeds: set[int] | None = None
    expected_metrics: set[str] | None = None
    for model_id, trials in sorted(grouped.items()):
        if expected_trials is None:
            expected_trials = set(trials)
        elif set(trials) != expected_trials:
            raise ValueError("models do not contain identical counterfactual trials")
        per_trial: dict[str, dict[str, float]] = {}
        for trial_id, seeds in sorted(trials.items()):
            if expected_seeds is None:
                expected_seeds = set(seeds)
            elif set(seeds) != expected_seeds:
                raise ValueError("counterfactual trials do not contain identical seeds")
            metric_names = {name for values in seeds.values() for name in values}
            if any(set(values) != metric_names for values in seeds.values()):
                raise ValueError("seed records do not contain identical metrics")
            if expected_metrics is None:
                expected_metrics = metric_names
            elif metric_names != expected_metrics:
                raise ValueError("counterfactual trials do not contain identical metrics")
            per_trial[trial_id] = {
                metric: statistics.fmean(values[metric] for values in seeds.values())
                for metric in sorted(metric_names)
            }
        per_unit_lists: dict[str, list[dict[str, float]]] = {}
        for trial_id, values in per_trial.items():
            unit = independent_unit_by_record[(model_id, trial_id)]
            per_unit_lists.setdefault(unit, []).append(values)
        per_unit = {
            unit: {
                metric: statistics.fmean(row[metric] for row in unit_rows)
                for metric in sorted(expected_metrics or ())
            }
            for unit, unit_rows in sorted(per_unit_lists.items())
        }
        models[model_id] = {
            "seeds": sorted(expected_seeds or ()),
            "raw_trials": len(per_trial),
            "independent_trials": len(per_unit),
            "per_trial": per_unit,
            "mean": {
                metric: statistics.fmean(
                    values[metric] for values in per_unit.values()
                )
                for metric in sorted(expected_metrics or ())
            },
        }
    return models


def compare_counterfactual_models(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_model: str,
    baseline_model: str,
    primary_metrics: Mapping[str, str],
    minimum_independent_trials: int = 20,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260812,
) -> dict[str, Any]:
    """Compare diagnostic metrics after averaging inference seeds per episode."""

    if minimum_independent_trials < 2:
        raise ValueError("minimum_independent_trials must be at least two")
    models = aggregate_counterfactual_records(records)
    if candidate_model not in models or baseline_model not in models:
        raise ValueError("candidate and baseline models must both be present")
    comparisons = []
    all_positive = True
    for metric, direction in primary_metrics.items():
        if direction not in {"higher", "lower"}:
            raise ValueError(f"invalid direction for {metric!r}")
        candidate = {
            trial_id: float(values[metric])
            for trial_id, values in models[candidate_model]["per_trial"].items()
        }
        baseline = {
            trial_id: float(values[metric])
            for trial_id, values in models[baseline_model]["per_trial"].items()
        }
        evidence = paired_bootstrap_lower_bound(
            candidate,
            baseline,
            higher_is_better=direction == "higher",
            seed=seed + len(comparisons),
            iterations=bootstrap_iterations,
            confidence=confidence,
        )
        positive = float(evidence["lower_confidence_bound"]) > 0.0
        all_positive = all_positive and positive
        comparisons.append(
            {
                "metric": metric,
                "direction": direction,
                "positive_lower_bound": positive,
                **evidence,
            }
        )
    trial_count = int(models[candidate_model]["independent_trials"])
    decision_eligible = trial_count >= minimum_independent_trials
    return {
        "schema_version": "1.0.0",
        "candidate_model": candidate_model,
        "baseline_model": baseline_model,
        "models": models,
        "minimum_independent_trials": minimum_independent_trials,
        "decision_eligible": decision_eligible,
        "all_primary_metric_lower_bounds_positive": all_positive,
        "audit_passed": decision_eligible and all_positive,
        "comparisons": comparisons,
        "claim_boundary": (
            "This diagnostic has no physical counterfactual target. Passing can establish "
            "greater action sensitivity with factual non-regression on the named suite, "
            "not causal action correctness, task success, or SOTA."
        ),
    }
