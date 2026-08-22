from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from phiagent.harness.task_reasoning import TSHIRT_FOLD_TASK, TaskEntity, TaskReasoningRequest
from phiagent.harness.tshirt_fold_strategy import (
    LEFT_THEN_RIGHT,
    RIGHT_THEN_LEFT,
    SIMULTANEOUS,
    VIEWER_LEFT,
    TshirtFoldStrategy,
    TshirtFoldStrategyReasoningPlugin,
)
from phiagent.harness.tshirt_positive_reference import (
    CAMERA_PIXEL_REFERENCE,
    COMPACT_IN_PLACE,
    VISUAL_QUALITY_REVIEW,
    PositiveFoldReference,
    PositiveReferenceBank,
    compile_reference_conditioning,
    compile_reference_conditioning_batch,
    load_positive_reference_bank,
)


def _request() -> TaskReasoningRequest:
    return TaskReasoningRequest(
        task_id="multi-strategy-tshirt-fold",
        task_type=TSHIRT_FOLD_TASK,
        instruction="Generate several distinct two-arm folding strategies.",
        coordinate_frame="camera:tshirt_fold_1024x768_pixels",
        duration_seconds=8.0,
        entities=(
            TaskEntity("lower_left_robot", "manipulator", "lower-left robot"),
            TaskEntity("upper_right_robot", "manipulator", "upper-right robot"),
            TaskEntity("viewer_left_sleeve", "cloth_part", "viewer-left sleeve"),
            TaskEntity("viewer_right_sleeve", "cloth_part", "viewer-right sleeve"),
            TaskEntity("shirt_body", "cloth_body", "gray shirt body"),
        ),
        available_evidence=("three user-reviewed RGB videos",),
        unavailable_evidence=("metric cloth state", "force", "joint trajectory"),
        user_constraints=("preserve distinct strategies", "do not relax hard gates"),
    )


def _reference(
    reference_id: str,
    sleeve_order: str,
    payload: bytes,
    video_path: str,
) -> PositiveFoldReference:
    return PositiveFoldReference(
        reference_id=reference_id,
        video_path=video_path,
        video_sha256=hashlib.sha256(payload).hexdigest(),
        coordinate_frame="camera:tshirt_fold_1024x768_pixels",
        width=1024,
        height=768,
        fps=24.0,
        frame_count=192,
        duration_seconds=8.0,
        sleeve_order=sleeve_order,
        terminal_behavior=COMPACT_IN_PLACE,
        review_scope=VISUAL_QUALITY_REVIEW,
        review_state="passed",
        observed_strengths=("stable camera", "persistent two-arm appearance"),
        unavailable_evidence=("metric geometry", "force", "robot execution"),
    )


def _bank() -> PositiveReferenceBank:
    return PositiveReferenceBank(
        "user-accepted-three-strategy-v1",
        (
            _reference("alternating", LEFT_THEN_RIGHT, b"a", "media/a.mp4"),
            _reference("staged", RIGHT_THEN_LEFT, b"b", "media/b.mp4"),
            _reference("synchronized", SIMULTANEOUS, b"c", "media/c.mp4"),
        ),
    )


def test_reference_conditioning_preserves_every_hard_gate_and_terminal_plan() -> None:
    plan = TshirtFoldStrategyReasoningPlugin().analyze_strategy(
        _request(), TshirtFoldStrategy(LEFT_THEN_RIGHT, VIEWER_LEFT)
    )

    conditioning = compile_reference_conditioning(plan, _bank())

    assert conditioning.reference_id == "alternating"
    assert conditioning.task_plan_sha256 == plan.plan_sha256
    assert conditioning.non_overrideable_hard_gate_ids == tuple(
        gate.gate_id for gate in plan.verification_gates
    )
    assert "visual generation quality only" in conditioning.prompt_addendum
    assert "left-then-right-place-left" in conditioning.terminal_strategy_id
    assert "real-robot task success" in conditioning.claim_boundary


def test_reference_conditioning_rejects_cross_strategy_reference() -> None:
    plan = TshirtFoldStrategyReasoningPlugin().analyze_strategy(
        _request(), TshirtFoldStrategy(LEFT_THEN_RIGHT, VIEWER_LEFT)
    )

    with pytest.raises(ValueError, match="does not match"):
        compile_reference_conditioning(plan, _bank(), reference_id="staged")


def test_reference_batch_uses_one_reviewed_prior_for_each_sleeve_order() -> None:
    plugin = TshirtFoldStrategyReasoningPlugin()
    plans = tuple(
        plugin.analyze_strategy(_request(), TshirtFoldStrategy(order, VIEWER_LEFT))
        for order in (LEFT_THEN_RIGHT, RIGHT_THEN_LEFT, SIMULTANEOUS)
    )

    compiled = compile_reference_conditioning_batch(plans, _bank())

    assert tuple(item.reference_id for item in compiled) == (
        "alternating",
        "staged",
        "synchronized",
    )
    assert len({item.conditioning_sha256 for item in compiled}) == 3


def test_bank_hash_and_media_hash_fail_closed(tmp_path: Path) -> None:
    bank = _bank()
    for reference, payload in zip(bank.references, (b"a", b"b", b"c")):
        path = tmp_path / reference.video_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"positive_reference_bank": bank.to_dict()}), encoding="utf-8"
    )

    loaded = load_positive_reference_bank(manifest, repo_root=tmp_path)

    assert loaded.bank_sha256 == bank.bank_sha256
    assert loaded.to_dict()["conditioning_scope"] == CAMERA_PIXEL_REFERENCE
    (tmp_path / bank.references[0].video_path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="video hash mismatch"):
        load_positive_reference_bank(manifest, repo_root=tmp_path)


def test_bank_declared_hash_and_reference_review_scope_are_non_relaxable() -> None:
    payload = _bank().to_dict()
    payload["bank_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bank hash mismatch"):
        PositiveReferenceBank.from_dict(payload)

    with pytest.raises(ValueError, match="visual-quality review"):
        replace(_bank().references[0], review_scope="physical_execution")
