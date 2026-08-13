"""Reference graph for 20-second-plus flower-arranging video replacement."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from phiagent.agent.perceptual_video_harness import (
    PERCEPTUAL_DEMO_GATES,
    PerceptualCandidate,
    foundation_model_roles,
    select_display_candidate,
)

from .checkpoint import CheckpointStore
from .core import END, NodeContext, StateGraph


FLOWER_WORKFLOW_NAME = "flower-long-video"
FLOWER_WORKFLOW_VERSION = "1.0.0"
FLOWER_VISUAL_GATES = (*PERCEPTUAL_DEMO_GATES, "late_projected_contact_visible")

_POLICY_FLOORS = {
    "min_duration_seconds": 20.0,
    "background_exact_fraction": 0.99,
    "flower_exact_fraction": 1.0,
    "flower_dynamic_frame_fraction": 0.95,
    "late_projected_contact_recall": 0.95,
}


def build_flower_long_video_workflow(
    *, checkpointer: CheckpointStore | None = None,
):
    """Build the first migrated domain workflow.

    The graph audits existing outputs or pauses for high-resolution review.  It
    never upgrades image-space depth/contact proposals to physical evidence.
    Generation scripts can be attached through ``SubprocessNode`` without
    importing their heavyweight runtimes into this module.
    """

    graph = StateGraph(reducers={"quality_gates": _merge_mapping})
    graph.add_node("validate_request", _validate_request)
    graph.add_node("lock_evidence_lineage", _lock_evidence_lineage)
    graph.add_node("audit_long_horizon", _audit_long_horizon)
    graph.add_node("audit_adversarial_critic", _audit_adversarial_critic)
    graph.add_node("high_resolution_review", _high_resolution_review)
    graph.add_node("decide_promotion", _decide_promotion)
    graph.add_node("finalize_display", _finalize_display)
    graph.add_node("plan_architecture_repair", _plan_architecture_repair)
    graph.set_entry_point("validate_request")
    graph.add_edge("validate_request", "lock_evidence_lineage")
    graph.add_edge("lock_evidence_lineage", "audit_long_horizon")
    graph.add_edge("audit_long_horizon", "audit_adversarial_critic")
    graph.add_edge("audit_adversarial_critic", "high_resolution_review")
    graph.add_edge("high_resolution_review", "decide_promotion")
    graph.add_conditional_edges(
        "decide_promotion",
        _promotion_route,
        {"accepted": "finalize_display", "rejected": "plan_architecture_repair"},
    )
    graph.add_edge("finalize_display", END)
    graph.add_edge("plan_architecture_repair", END)
    return graph.compile(
        name=FLOWER_WORKFLOW_NAME,
        version=FLOWER_WORKFLOW_VERSION,
        checkpointer=checkpointer,
        max_steps=32,
    )


def _validate_request(state: Mapping[str, Any]) -> dict[str, Any]:
    if state.get("workflow") not in {None, FLOWER_WORKFLOW_NAME}:
        raise ValueError(f"workflow must be {FLOWER_WORKFLOW_NAME!r}")
    candidate_id = str(state.get("candidate_id", "")).strip()
    if not candidate_id:
        raise ValueError("candidate_id is required")
    claim_scope = str(state.get("claim_scope", ""))
    if claim_scope not in {
        "perceptually plausible synthetic display data",
        "perceptually plausible synthetic video data",
    }:
        raise ValueError("this workflow accepts only the perceptual synthetic-display scope")
    workspace_root = Path(str(state.get("workspace_root", "."))).expanduser().resolve()
    if not workspace_root.is_dir():
        raise ValueError(f"workspace_root does not exist: {workspace_root}")
    evidence = state.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence must be an object")
    for name in ("manifest", "adversarial_audit", "promotion"):
        if not str(evidence.get(name, "")).strip():
            raise ValueError(f"evidence.{name} is required")
    policy = _validate_policy(state.get("policy", {}))
    generation_recipe = {
        "representation": "persistent_full_timeline_plus_object_factored_local_repairs",
        "action_phases": ["reach", "grasp", "transport", "insert", "release"],
        "immutable_layers": ["source_flowers", "source_background", "flower_response_motion"],
        "editable_layer": "tracked_robot_or_person_support_only",
        "proposal_models": list(foundation_model_roles()),
        "promotion_authority": [
            "post_decode_object_lock",
            "all_frame_temporal_audit",
            "adversarial_critic",
            "native_resolution_human_veto",
        ],
    }
    return {
        "candidate_id": candidate_id,
        "workspace_root": str(workspace_root),
        "policy": policy,
        "policy_sha256": _json_sha256(policy),
        "generation_recipe": generation_recipe,
        "quality_gates": {},
    }


def _lock_evidence_lineage(state: Mapping[str, Any]) -> dict[str, Any]:
    workspace_root = Path(str(state["workspace_root"]))
    raw_paths = state["evidence"]
    if not isinstance(raw_paths, Mapping):
        raise ValueError("evidence must remain an object")
    resolved = {
        name: _resolve_path(workspace_root, str(raw_path))
        for name, raw_path in raw_paths.items()
        if raw_path is not None and str(raw_path).strip()
    }
    for name in ("manifest", "adversarial_audit", "promotion"):
        if name not in resolved:
            raise ValueError(f"missing required evidence path: {name}")
    documents = {name: _read_json(path) for name, path in resolved.items()}
    manifest = documents["manifest"]
    audit = documents["adversarial_audit"]
    prior_promotion = documents["promotion"]
    human_review = documents.get("human_review")

    if manifest.get("physical_evidence") is not False:
        raise ValueError("flower perceptual manifest must explicitly set physical_evidence=false")
    coordinate_frames = manifest.get("coordinate_frames")
    if not isinstance(coordinate_frames, Mapping):
        raise ValueError("manifest must name its coordinate frames")
    if not str(coordinate_frames.get("source", "")).startswith("camera:"):
        raise ValueError("manifest source measurements require a named camera frame")

    file_hashes = {name: _sha256(path) for name, path in resolved.items()}
    promotion_inputs = prior_promotion.get("inputs", [])
    if not isinstance(promotion_inputs, list):
        raise ValueError("promotion.inputs must be an array")
    promotion_bindings = {}
    for item in promotion_inputs:
        if not isinstance(item, Mapping):
            continue
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        expected = str(item.get("sha256", ""))
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"promotion lineage mismatch: {path}")
        promotion_bindings[str(path)] = expected
    for name in ("manifest", "adversarial_audit"):
        if str(resolved[name]) not in promotion_bindings:
            raise ValueError(f"promotion does not bind current {name} evidence")
    if human_review is not None and str(resolved["human_review"]) not in promotion_bindings:
        raise ValueError("promotion does not bind the current human review")

    output_name = str(state.get("video_output", "review_video"))
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get(output_name), Mapping):
        raise ValueError(f"manifest does not declare outputs.{output_name}")
    video_record = outputs[output_name]
    video_path = Path(str(video_record.get("path", ""))).expanduser().resolve()
    if not video_path.is_file():
        raise ValueError(f"candidate video is missing: {video_path}")
    video_sha256 = _sha256(video_path)
    if video_sha256 != video_record.get("sha256"):
        raise ValueError("candidate video hash does not match its manifest")

    metrics = manifest.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("manifest.metrics must be an object")
    candidates = audit.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("adversarial audit must contain exactly one scoped candidate")
    audit_candidate = candidates[0]
    if not isinstance(audit_candidate, Mapping):
        raise ValueError("adversarial candidate must be an object")

    evidence_summary = {
        "manifest": {
            "path": str(resolved["manifest"]),
            "sha256": file_hashes["manifest"],
            "claim_scope": manifest.get("claim_scope"),
            "physical_evidence": manifest.get("physical_evidence"),
            "coordinate_frames": dict(coordinate_frames),
            "metrics": dict(metrics),
            "limitations": list(manifest.get("limitations", [])),
        },
        "adversarial_audit": {
            "path": str(resolved["adversarial_audit"]),
            "sha256": file_hashes["adversarial_audit"],
            "adversarial_audit_pass": audit.get("adversarial_audit_pass"),
            "candidate": {
                "adversarial": audit_candidate.get("adversarial"),
                "summary": audit_candidate.get("summary"),
                "wall_seconds": audit_candidate.get("wall_seconds"),
            },
        },
        "human_review": human_review,
        "prior_promotion": prior_promotion,
    }
    return {
        "resolved_evidence": {name: str(path) for name, path in resolved.items()},
        "evidence_sha256": file_hashes,
        "candidate_video": {"path": str(video_path), "sha256": video_sha256},
        "evidence_summary": evidence_summary,
        "lineage_locked": True,
    }


def _audit_long_horizon(state: Mapping[str, Any]) -> dict[str, Any]:
    summary = state["evidence_summary"]
    metrics = summary["manifest"]["metrics"]
    policy = state["policy"]
    frames = int(metrics.get("frames", 0))
    seconds = float(metrics.get("video_seconds", 0.0))
    postdecode = metrics.get("postencode_lossless_lock_audit", {})
    if not isinstance(postdecode, Mapping):
        postdecode = {}
    gates = {
        "duration_at_least_20_seconds": seconds >= policy["min_duration_seconds"],
        "full_video_decodes": int(postdecode.get("decoded_frames", -1)) == frames and frames > 0,
        "native_background_locked": float(
            postdecode.get("native_background_exact_fraction", -1.0)
        )
        >= policy["background_exact_fraction"],
        "flower_pixels_locked": float(postdecode.get("flower_exact_fraction", -1.0))
        >= policy["flower_exact_fraction"],
        "flower_response_not_frozen": float(
            metrics.get("source_flower_dynamic_frame_fraction", -1.0)
        )
        >= policy["flower_dynamic_frame_fraction"],
    }

    transition_p95 = float(metrics.get("frame_transition_delta_p95", 0.0))
    risk_threshold = transition_p95 * policy["route_boundary_outlier_multiplier"]
    raw_boundaries = metrics.get("route_boundary_transition_deltas", {})
    risk_windows = []
    if isinstance(raw_boundaries, Mapping):
        for raw_frame, raw_delta in raw_boundaries.items():
            frame = int(raw_frame)
            delta = float(raw_delta)
            if delta > risk_threshold:
                radius = int(policy["repair_context_frames"])
                risk_windows.append(
                    {
                        "kind": "route_transition_outlier",
                        "frame": frame,
                        "range_inclusive": [max(0, frame - radius), min(frames - 1, frame + radius)],
                        "transition_delta": delta,
                        "threshold": risk_threshold,
                        "repair_contract": (
                            "JoyAI source-anchored 1+8n causal bridge; deterministic "
                            "flower/background/endpoint projection"
                        ),
                    }
                )

    audit_summary = summary["adversarial_audit"]["candidate"].get("summary", {})
    late_proxy = _nested_get(
        audit_summary,
        "sections",
        "at_or_after_20_seconds",
        "metrics",
        "hand_replacement_coverage",
        "violation_fraction",
    )
    diagnostic_debt = []
    if late_proxy is not None and float(late_proxy) > 0:
        diagnostic_debt.append(
            {
                "name": "legacy_late_hand_replacement_proxy",
                "violation_fraction": float(late_proxy),
                "promotion_authority": False,
                "reason": "anchor-derived RGB-alpha proxy is diagnostic, not a morphology oracle",
            }
        )
    return {
        "quality_gates": gates,
        "risk_windows": risk_windows,
        "diagnostic_debt": diagnostic_debt,
        "long_horizon_metrics": {
            "frames": frames,
            "video_seconds": seconds,
            "effective_fps": frames / seconds if seconds > 0 else None,
            "flower_dynamic_frame_fraction": metrics.get(
                "source_flower_dynamic_frame_fraction"
            ),
            "route_transition_outliers": len(risk_windows),
        },
    }


def _audit_adversarial_critic(state: Mapping[str, Any]) -> dict[str, Any]:
    audit = state["evidence_summary"]["adversarial_audit"]
    candidate = audit["candidate"]
    adversarial = candidate.get("adversarial", {})
    if not isinstance(adversarial, Mapping):
        adversarial = {}
    attack_gates = adversarial.get("gates", {})
    if not isinstance(attack_gates, Mapping):
        attack_gates = {}
    detected = (
        audit.get("adversarial_audit_pass") is True
        and adversarial.get("all_attacks_detected") is True
        and len(attack_gates) >= 4
        and all(value is True for value in attack_gates.values())
    )
    late_contact = _nested_get(
        candidate.get("summary", {}),
        "sections",
        "at_or_after_20_seconds",
        "projected_contact_recall",
    )
    contact_diagnostic_pass = (
        late_contact is not None
        and float(late_contact) >= state["policy"]["late_projected_contact_recall"]
    )
    return {
        "quality_gates": {
            "adversarial_attacks_detected": detected,
            "late_projected_contact_visible": contact_diagnostic_pass,
        },
        "adversarial_result": {
            "all_attacks_detected": detected,
            "attack_gates": dict(attack_gates),
            "sampled_frames": adversarial.get("sampled_frames"),
            "late_projected_contact_recall": late_contact,
            "projected_contact_diagnostic_pass": contact_diagnostic_pass,
            "physical_contact_evidence": False,
        },
        "physical_promotion": {
            "eligible": False,
            "independent_physical_groups": 1,
            "required_independent_groups": 2,
            "gates": {
                "metric_camera": False,
                "exact_robot_q_qdot": False,
                "persistent_per_stem_3d": False,
                "sensor_or_solver_force": False,
            },
            "reason": "RGB projections and model critics cannot establish metric force closure",
        },
    }


def _high_resolution_review(
    state: Mapping[str, Any], context: NodeContext
) -> dict[str, Any]:
    review = state["evidence_summary"].get("human_review")
    if review is None:
        review = context.interrupt(
            {
                "kind": "native_resolution_video_review",
                "candidate_video": state["candidate_video"],
                "risk_windows": state.get("risk_windows", []),
                "required_boolean_gates": [
                    "human_residue_absent",
                    "canonical_hand_topology_locked",
                    "intermittent_hand_smear_absent",
                    "long_term_robot_identity_stable",
                    "high_resolution_review_pass",
                ],
                "scope": "perceptually plausible synthetic display data only",
            },
            key="native-resolution-veto",
        )
    if not isinstance(review, Mapping):
        raise ValueError("human review must be an object")
    raw_gates = review.get("gates", {})
    if not isinstance(raw_gates, Mapping):
        raise ValueError("human review gates must be an object")
    gates = {
        name: raw_gates.get(name) is True
        for name in (
            "human_residue_absent",
            "canonical_hand_topology_locked",
            "intermittent_hand_smear_absent",
            "long_term_robot_identity_stable",
        )
    }
    gates["high_resolution_review_pass"] = review.get("high_resolution_review_pass") is True
    return {
        "quality_gates": gates,
        "human_review_result": {
            "decision": review.get("decision"),
            "reviewer": review.get("reviewer"),
            "gates": gates,
            "notes": list(review.get("notes", [])),
            "utility": float(review.get("utility", 0.0)),
        },
    }


def _decide_promotion(state: Mapping[str, Any]) -> dict[str, Any]:
    gates = state.get("quality_gates", {})
    if set(gates) != set(FLOWER_VISUAL_GATES):
        missing = sorted(set(FLOWER_VISUAL_GATES) - set(gates))
        extra = sorted(set(gates) - set(FLOWER_VISUAL_GATES))
        raise ValueError(f"quality gate mismatch: missing={missing}, extra={extra}")
    metrics = state["evidence_summary"]["manifest"]["metrics"]
    decision = select_display_candidate(
        [
            PerceptualCandidate(
                candidate_id=str(state["candidate_id"]),
                gates=tuple((name, bool(gates[name])) for name in PERCEPTUAL_DEMO_GATES),
                utility=float(state["human_review_result"]["utility"]),
                wall_seconds=float(metrics.get("compositor_wall_seconds", 0.0)),
                evidence_path=str(state["resolved_evidence"]["manifest"]),
            )
        ]
    )
    failed_flower_gates = [name for name in FLOWER_VISUAL_GATES if gates[name] is not True]
    if failed_flower_gates:
        decision = {
            **decision,
            "status": "PARTIAL",
            "selected_candidate": None,
            "failed_flower_gates": failed_flower_gates,
        }
    else:
        decision = {**decision, "failed_flower_gates": []}
    prior = state["evidence_summary"]["prior_promotion"]
    prior_consistent = (
        prior.get("status") == decision["status"]
        and prior.get("selected_candidate") == decision["selected_candidate"]
        and prior.get("physical_evidence") is False
    )
    return {
        "promotion_decision": decision,
        "prior_promotion_consistent": prior_consistent,
    }


def _promotion_route(state: Mapping[str, Any]) -> str:
    decision = state.get("promotion_decision", {})
    if isinstance(decision, Mapping) and decision.get("status") == "DISPLAY_READY":
        return "accepted"
    return "rejected"


def _finalize_display(state: Mapping[str, Any]) -> dict[str, Any]:
    risk_windows = list(state.get("risk_windows", []))
    next_iteration = _architecture_actions((), risk_windows)
    return {
        "workflow_outcome": {
            "status": "DISPLAY_READY",
            "claim_scope": "perceptually plausible synthetic video data",
            "candidate_video": state["candidate_video"],
            "all_hard_visual_gates_pass": True,
            "physical_evidence": False,
            "physical_promotion": False,
            "quality_debt_count": len(risk_windows)
            + len(state.get("diagnostic_debt", [])),
        },
        "next_quality_iteration": next_iteration,
    }


def _plan_architecture_repair(state: Mapping[str, Any]) -> dict[str, Any]:
    gates = state.get("quality_gates", {})
    failed = tuple(name for name in FLOWER_VISUAL_GATES if gates.get(name) is not True)
    return {
        "workflow_outcome": {
            "status": "PARTIAL",
            "claim_scope": "perceptually plausible synthetic video data",
            "candidate_video": state["candidate_video"],
            "failed_hard_visual_gates": list(failed),
            "physical_evidence": False,
            "physical_promotion": False,
        },
        "next_quality_iteration": _architecture_actions(failed, state.get("risk_windows", [])),
    }


def _architecture_actions(
    failed: tuple[str, ...], risk_windows: Any
) -> dict[str, Any]:
    actions = []
    if any(
        name in failed
        for name in (
            "native_background_locked",
            "flower_pixels_locked",
            "flower_response_not_frozen",
        )
    ):
        actions.append(
            {
                "architecture": "immutable_source_state_projection",
                "change": "factor flowers/background/response motion out of the generative state",
                "acceptance": "post-decode exact locks plus non-frozen all-frame motion",
            }
        )
    if any(
        name in failed
        for name in (
            "canonical_hand_topology_locked",
            "intermittent_hand_smear_absent",
            "long_term_robot_identity_stable",
        )
    ):
        actions.append(
            {
                "architecture": "persistent_embodiment_state_with_canonical_topology",
                "change": "condition all action phases on one identity state and articulated hand prior",
                "acceptance": "native-resolution topology/identity veto over every risk window",
            }
        )
    if risk_windows:
        actions.append(
            {
                "architecture": "joyai_source_anchored_causal_bridge",
                "change": (
                    "regenerate only measured 1+8n seam neighborhoods with JoyAI's "
                    "first-chunk sink and recent causal KV, then project immutable source state"
                ),
                "windows": list(risk_windows),
                "foundation_model": "jdopensource/JoyAI-Video-Edit@7c36b253",
                "acceptance": (
                    "route transition no longer an all-frame outlier; flower/background/outside-window/"
                    "endpoint locks exact; native-resolution topology veto passes"
                ),
            }
        )
    if "adversarial_attacks_detected" in failed:
        actions.append(
            {
                "architecture": "critic_counterexample_training",
                "change": "add missed attack families to a sealed adversarial split",
                "acceptance": "all attacks detected without weakening any visual gate",
            }
        )
    if "late_projected_contact_visible" in failed:
        actions.append(
            {
                "architecture": "contact_phase_object_motion_coupling",
                "change": "condition grasp/insert/release on persistent flower-response state",
                "acceptance": "late projected contact plus non-frozen flower response pass together",
            }
        )
    if not actions:
        actions.append(
            {
                "architecture": "bounded_phase_local_challenger",
                "change": "challenge only measured reach/grasp/insert/release windows",
                "acceptance": "strict Pareto improvement with immutable object locks",
            }
        )
    return {
        "failed_gates": list(failed),
        "threshold_changes_allowed": False,
        "actions": actions,
        "model_authority": "proposal_only",
        "promotion_authority": "deterministic validators plus native-resolution human veto",
    }


def _validate_policy(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("policy must be an object")
    policy = {
        **_POLICY_FLOORS,
        "route_boundary_outlier_multiplier": 3.0,
        "repair_context_frames": 16,
        **dict(raw),
    }
    for name, floor in _POLICY_FLOORS.items():
        value = float(policy[name])
        if not math.isfinite(value) or value < floor:
            raise ValueError(f"policy.{name} cannot weaken the frozen floor {floor}")
        policy[name] = value
    multiplier = float(policy["route_boundary_outlier_multiplier"])
    if not math.isfinite(multiplier) or multiplier <= 0 or multiplier > 3.0:
        raise ValueError("route_boundary_outlier_multiplier must be in (0, 3.0]")
    policy["route_boundary_outlier_multiplier"] = multiplier
    context = int(policy["repair_context_frames"])
    if context < 8:
        raise ValueError("repair_context_frames must be at least 8")
    policy["repair_context_frames"] = context
    return policy


def _merge_mapping(left: Any, right: Any) -> dict[str, Any]:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise TypeError("mapping reducer received a non-mapping value")
    return {**left, **right}


def _resolve_path(workspace_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    resolved = path.resolve() if path.is_absolute() else (workspace_root / path).resolve()
    if not resolved.is_file():
        raise ValueError(f"evidence file does not exist: {resolved}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must contain an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nested_get(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current
