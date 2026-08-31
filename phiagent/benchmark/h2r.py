"""Clean-room reproduction of the public H2R-Bench scoring equations.

This module follows arXiv:2608.13049v1 Appendix B.6. It is not the unpublished
official evaluator. Judge providers are deliberately external: this module only
validates structured outputs and performs deterministic aggregation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from phiagent.benchmark.schema import VisualEvidence


CONTACT_DIMENSIONS = (
    "contact_region_transfer",
    "contact_establishment",
    "manipulation_mode_transfer",
    "temporally_supported_object_response",
    "embodiment_compatible_contact_strategy",
)
EMBODIMENT_DIMENSIONS = (
    "robot_actor_presence",
    "human_absence",
    "embodiment_category_match",
    "end_effector_correctness",
    "structural_consistency",
)
EMBODIMENT_WEIGHTS = {
    "robot_actor_presence": 0.20,
    "human_absence": 0.20,
    "embodiment_category_match": 0.20,
    "end_effector_correctness": 0.25,
    "structural_consistency": 0.15,
}


def _score(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer rubric score")
    score = int(value)
    if score != float(value) or score not in range(5):
        raise ValueError(f"{label} must be an integer in [0, 4]")
    return score


def _unit(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return number


@dataclass(frozen=True)
class WeightedCriterion:
    criterion_id: str
    description: str
    weight: float

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WeightedCriterion":
        weight = float(payload["weight"])
        criterion_id = str(payload["id"]).strip()
        description = str(payload["description"]).strip()
        if not criterion_id or not description or not math.isfinite(weight) or weight <= 0:
            raise ValueError("H2R criteria require an ID, description, and positive weight")
        return cls(criterion_id, description, weight)


@dataclass(frozen=True)
class H2RAnnotation:
    goal_predicates: tuple[WeightedCriterion, ...]
    action_events: tuple[WeightedCriterion, ...]
    applicable_contact_dimensions: tuple[str, ...]
    contact_specification: dict[str, Any]
    target_embodiment: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "H2RAnnotation":
        goals = tuple(WeightedCriterion.from_dict(item) for item in payload["goal_predicates"])
        actions = tuple(WeightedCriterion.from_dict(item) for item in payload["action_events"])
        applicable = tuple(str(value) for value in payload["applicable_contact_dimensions"])
        if not goals or not actions:
            raise ValueError("H2R annotations require goal predicates and action events")
        for label, criteria in (("goal", goals), ("action", actions)):
            identifiers = [criterion.criterion_id for criterion in criteria]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"duplicate {label} criterion IDs")
        if not applicable or len(set(applicable)) != len(applicable):
            raise ValueError("applicable contact dimensions must be unique and non-empty")
        if any(value not in CONTACT_DIMENSIONS for value in applicable):
            raise ValueError("unknown H2R contact dimension")
        contact = payload.get("contact_specification", {})
        target = payload.get("target_embodiment", {})
        if not isinstance(contact, Mapping) or not isinstance(target, Mapping):
            raise ValueError("contact specification and target embodiment must be objects")
        return cls(goals, actions, applicable, dict(contact), dict(target))


@dataclass(frozen=True)
class H2RJudgeOutput:
    judge_id: str
    goal_scores: dict[str, int]
    action_scores: dict[str, int]
    source_grounded: bool
    contact_scores: dict[str, int]
    embodiment_hard_failure: bool
    embodiment_scores: dict[str, int]
    evidence_frame_indices: dict[str, tuple[int, ...]]
    rationale: dict[str, str]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "H2RJudgeOutput":
        def scores(name: str) -> dict[str, int]:
            raw = payload.get(name)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{name} must be an object")
            return {str(key): _score(value, f"{name}.{key}") for key, value in raw.items()}

        raw_frames = payload.get("evidence_frame_indices", {})
        raw_rationale = payload.get("rationale", {})
        if not isinstance(raw_frames, Mapping) or not isinstance(raw_rationale, Mapping):
            raise ValueError("judge evidence indices and rationales must be objects")
        frame_indices = {
            str(key): tuple(int(index) for index in value)
            for key, value in raw_frames.items()
        }
        if any(index < 0 or index >= 25 for values in frame_indices.values() for index in values):
            raise ValueError("H2R evidence frame indices must refer to the 25-frame budget")
        source_grounded = payload.get("source_grounded")
        hard_failure = payload.get("embodiment_hard_failure")
        if not isinstance(source_grounded, bool) or not isinstance(hard_failure, bool):
            raise ValueError("H2R judge hard decisions must be boolean")
        return cls(
            judge_id=str(payload["judge_id"]).strip(),
            goal_scores=scores("goal_scores"),
            action_scores=scores("action_scores"),
            source_grounded=source_grounded,
            contact_scores=scores("contact_scores"),
            embodiment_hard_failure=hard_failure,
            embodiment_scores=scores("embodiment_scores"),
            evidence_frame_indices=frame_indices,
            rationale={str(key): str(value) for key, value in raw_rationale.items()},
        )


def _weighted_score(
    criteria: tuple[WeightedCriterion, ...], scores: Mapping[str, int], label: str
) -> float:
    expected = {criterion.criterion_id for criterion in criteria}
    if set(scores) != expected:
        raise ValueError(f"{label} scores do not match annotation IDs")
    denominator = sum(criterion.weight for criterion in criteria)
    return sum(criterion.weight * scores[criterion.criterion_id] / 4.0 for criterion in criteria) / denominator


def _weighted_coverage(
    criteria: tuple[WeightedCriterion, ...], scores: Mapping[str, int], threshold: int
) -> float:
    denominator = sum(criterion.weight for criterion in criteria)
    return sum(
        criterion.weight
        for criterion in criteria
        if scores[criterion.criterion_id] >= threshold
    ) / denominator


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric")
    return sum(values) / len(values)


def aggregate_h2r_judges(
    annotation: H2RAnnotation,
    judges: tuple[H2RJudgeOutput, ...],
    *,
    video_quality_components: Mapping[str, float],
    strict_three_judges: bool = True,
) -> VisualEvidence:
    """Apply H2R equations S1--S12 to structured judge outputs."""

    if strict_three_judges and len(judges) != 3:
        raise ValueError("the H2R reproduction requires exactly three independent judges")
    if not judges or len({judge.judge_id for judge in judges}) != len(judges):
        raise ValueError("H2R judges must be non-empty and uniquely identified")
    quality_names = {"imaging_quality", "aesthetic_quality", "temporal_stability", "motion_smoothness"}
    if set(video_quality_components) != quality_names:
        raise ValueError("M5 requires exactly the four public H2R quality components")
    quality = _mean([_unit(video_quality_components[name], name) for name in sorted(quality_names)])

    goals: list[float] = []
    actions: list[float] = []
    contacts: list[float] = []
    embodiments: list[float] = []
    goal_coverage_3: list[float] = []
    goal_coverage_4: list[float] = []
    action_coverage_3: list[float] = []
    action_coverage_4: list[float] = []
    source_grounding_failures = 0
    embodiment_hard_failures = 0

    goal_ids = {item.criterion_id for item in annotation.goal_predicates}
    action_ids = {item.criterion_id for item in annotation.action_events}
    applicable_contacts = set(annotation.applicable_contact_dimensions)
    for judge in judges:
        if set(judge.goal_scores) != goal_ids or set(judge.action_scores) != action_ids:
            raise ValueError(f"judge {judge.judge_id} criterion keys do not match the case")
        if set(judge.contact_scores) != applicable_contacts:
            raise ValueError(f"judge {judge.judge_id} contact keys do not match applicability")
        if set(judge.embodiment_scores) != set(EMBODIMENT_DIMENSIONS):
            raise ValueError(f"judge {judge.judge_id} embodiment keys are incomplete")
        goals.append(_weighted_score(annotation.goal_predicates, judge.goal_scores, "goal"))
        actions.append(_weighted_score(annotation.action_events, judge.action_scores, "action"))
        goal_coverage_3.append(_weighted_coverage(annotation.goal_predicates, judge.goal_scores, 3))
        goal_coverage_4.append(_weighted_coverage(annotation.goal_predicates, judge.goal_scores, 4))
        action_coverage_3.append(_weighted_coverage(annotation.action_events, judge.action_scores, 3))
        action_coverage_4.append(_weighted_coverage(annotation.action_events, judge.action_scores, 4))
        if judge.source_grounded:
            contacts.append(_mean([judge.contact_scores[name] / 4.0 for name in annotation.applicable_contact_dimensions]))
        else:
            source_grounding_failures += 1
            contacts.append(0.0)
        if judge.embodiment_hard_failure:
            embodiment_hard_failures += 1
            embodiments.append(0.0)
        else:
            embodiments.append(
                sum(EMBODIMENT_WEIGHTS[name] * judge.embodiment_scores[name] / 4.0 for name in EMBODIMENT_DIMENSIONS)
            )

    return VisualEvidence(
        goal_completion=_mean(goals),
        action_completion=_mean(actions),
        contact_transfer=_mean(contacts),
        embodiment_correctness=_mean(embodiments),
        video_quality=quality,
        judge_count=len(judges),
        evidence_frames=25,
        protocol="phiagent_h2r_reproduction_arxiv_2608.13049v1",
        diagnostics={
            "goal_coverage_ge3": _mean(goal_coverage_3),
            "goal_coverage_eq4": _mean(goal_coverage_4),
            "action_coverage_ge3": _mean(action_coverage_3),
            "action_coverage_eq4": _mean(action_coverage_4),
            "source_grounding_failure_rate": source_grounding_failures / len(judges),
            "embodiment_hard_failure_rate": embodiment_hard_failures / len(judges),
            "quality_components": dict(video_quality_components),
            "judge_ids": [judge.judge_id for judge in judges],
        },
    )
