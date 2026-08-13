from __future__ import annotations

import json

import pytest

from phiagent.training.h3_identity_rsi import (
    DomainCurriculumContract,
    H3IdentityRound,
    IdentityClipSpec,
    IdentityDatasetPlan,
    IdentityMetrics,
    IdentityPromotionContract,
    TopologyFrameReview,
    TopologyReviewEvidence,
    build_diffsynth_metadata,
    choose_next_round,
)
from scripts.evaluate_h3_identity_ablation import (
    _action_adherence,
    _action_score,
    _validate_action_context,
)


def _topology_evidence(*, failed_frame: int | None = None) -> TopologyReviewEvidence:
    frames = []
    for index in range(3):
        passed = index != failed_frame
        frames.append(
            TopologyFrameReview(
                frame_index=index,
                single_robot_subject=True,
                single_head_torso_chain=True,
                exactly_two_arms=True,
                left_shoulder_attachment=passed,
                right_shoulder_attachment=True,
                continuous_arm_segments=True,
                stable_robot_proportions=True,
                no_extra_or_missing_limbs=passed,
                no_human_residual=True,
                confidence=0.99,
                note="same-shoulder duplicate" if not passed else "",
                decoded_frame_sha256=f"{index + 1:064x}",
                unique_left_shoulder_origin=passed,
                unique_right_shoulder_origin=True,
                arm_roots_clear_of_head_and_neck=passed,
            )
        )
    return TopologyReviewEvidence(
        video_sha256="a" * 64,
        total_frames=3,
        reviewer="test-reviewer",
        review_method="unit-test",
        frames=tuple(frames),
    )


def _metrics(**overrides: float) -> IdentityMetrics:
    values = {
        "reference_identity_mean": 0.70,
        "reference_identity_worst": 0.63,
        "cross_frame_identity": 0.68,
        "topology_integrity": 0.85,
        "motion_adherence": 0.60,
        "action_adherence": 0.60,
        "scene_preservation": 0.95,
        "temporal_consistency": 0.90,
    }
    values.update(overrides)
    return IdentityMetrics(**values)


def test_promotion_requires_identity_gain_and_capability_non_regression() -> None:
    baseline = _metrics()
    improved = _metrics(
        reference_identity_worst=0.67,
        cross_frame_identity=0.72,
        topology_integrity=1.0,
        motion_adherence=0.595,
        scene_preservation=0.947,
    )
    assessment = IdentityPromotionContract().assess(baseline, improved, _topology_evidence())
    assert assessment.passed
    assert assessment.identity_gain == pytest.approx(0.04)


def test_promotion_rejects_identity_gain_that_hides_motion_collapse() -> None:
    baseline = _metrics()
    candidate = _metrics(
        reference_identity_worst=0.69,
        cross_frame_identity=0.73,
        topology_integrity=1.0,
        motion_adherence=0.40,
    )
    assessment = IdentityPromotionContract().assess(baseline, candidate, _topology_evidence())
    assert not assessment.passed
    assert "motion_non_regression" in assessment.failed_gates()
    next_round = choose_next_round(((H3IdentityRound("r0-smoke-r8", 8, 5e-5, 1, 1), assessment),))
    assert next_round is not None
    assert next_round.name == "r2-conservative-r16"


def test_promotion_rejects_action_regression_and_routes_conservatively() -> None:
    baseline = _metrics()
    candidate = _metrics(
        reference_identity_worst=0.69,
        cross_frame_identity=0.73,
        topology_integrity=1.0,
        action_adherence=0.40,
    )
    assessment = IdentityPromotionContract().assess(baseline, candidate, _topology_evidence())
    assert not assessment.passed
    assert "action_non_regression" in assessment.failed_gates()
    next_round = choose_next_round(((H3IdentityRound("r0-smoke-r8", 8, 5e-5, 1, 1), assessment),))
    assert next_round is not None
    assert next_round.name == "r2-conservative-r16"


def test_out_of_table_failed_round_requires_new_structural_route() -> None:
    candidate = _metrics(
        reference_identity_worst=0.69,
        cross_frame_identity=0.73,
        topology_integrity=1.0,
        action_adherence=0.40,
    )
    assessment = IdentityPromotionContract().assess(
        _metrics(), candidate, _topology_evidence()
    )
    extended = H3IdentityRound("r4b-domain-r16", 16, 1e-5, 3, 1)
    assert choose_next_round(((extended, assessment),)) is None


def test_legacy_metrics_default_action_adherence_to_one() -> None:
    payload = {
        name: value for name, value in _metrics().__dict__.items() if name != "action_adherence"
    }
    assert IdentityMetrics.from_dict(payload).action_adherence == 1.0


def test_action_score_cannot_hide_object_lock_collapse(tmp_path) -> None:
    evaluation = tmp_path / "action-evaluation.json"
    video_sha256 = "a" * 64
    evaluation.write_text(
        json.dumps(
            {
                "inputs": {
                    "raw_h3": {"sha256": video_sha256},
                },
                "outputs": {"final_sha256": "b" * 64},
                "best_scorecard": {
                    "motion_preservation": 0.91,
                    "epl_minimum": 0.88,
                    "object_lock": 0.02,
                },
                "rounds": [
                    {
                        "round": 0,
                        "repair": {"name": "raw-h3"},
                        "output_sha256": "c" * 64,
                        "scorecard": {
                            "motion_preservation": 0.91,
                            "epl_minimum": 0.88,
                            "object_lock": 0.02,
                        },
                    }
                ],
            }
        )
    )
    scores, binding = _action_score(evaluation, video_sha256)
    adherence, ratios = _action_adherence(
        {"motion_preservation": 0.92, "epl_minimum": 0.90, "object_lock": 1.0},
        scores,
    )
    assert binding["scope"] == "inputs.raw_h3"
    assert ratios["object_lock"] == pytest.approx(0.02)
    assert adherence == pytest.approx(0.02)


def test_action_score_rejects_evidence_for_another_video(tmp_path) -> None:
    evaluation = tmp_path / "action-evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "inputs": {"raw_h3": {"sha256": "a" * 64}},
                "outputs": {"final_sha256": "b" * 64},
                "rounds": [],
            }
        )
    )
    with pytest.raises(ValueError, match="not bound to candidate video"):
        _action_score(evaluation, "c" * 64)


def test_action_adherence_cannot_hide_motion_regression_behind_object_floor() -> None:
    baseline = {
        "motion_preservation": 0.60,
        "epl_minimum": 0.50,
        "object_lock": 0.0005,
    }
    candidate = {
        "motion_preservation": 0.54,
        "epl_minimum": 0.51,
        "object_lock": 0.0006,
    }
    adherence, ratios = _action_adherence(baseline, candidate)
    assert ratios["object_lock"] == 1.0
    assert adherence == pytest.approx(0.90)


def test_action_context_must_match(tmp_path) -> None:
    common_inputs = {
        name: {"sha256": str(index) * 64}
        for index, name in enumerate(
            ("source", "motion_reference", "robot_reference", "anchor_mask"), start=1
        )
    }
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps({"inputs": common_inputs}))
    mismatched = json.loads(json.dumps(common_inputs))
    mismatched["motion_reference"]["sha256"] = "f" * 64
    candidate.write_text(json.dumps({"inputs": mismatched}))
    with pytest.raises(ValueError, match="context mismatch"):
        _validate_action_context(baseline, candidate)


def test_promotion_rejects_one_bad_topology_frame_even_with_high_identity() -> None:
    baseline = _metrics(topology_integrity=0.60)
    candidate = _metrics(
        reference_identity_mean=0.99,
        reference_identity_worst=0.99,
        cross_frame_identity=0.99,
        topology_integrity=2 / 3,
    )
    assessment = IdentityPromotionContract().assess(
        baseline, candidate, _topology_evidence(failed_frame=1)
    )
    assert not assessment.passed
    assert "topology" in assessment.failed_gates()
    assert assessment.topology_failed_frames == (1,)
    assert dict(assessment.topology_failure_histogram) == {
        "left_shoulder_attachment": 1,
        "no_extra_or_missing_limbs": 1,
        "unique_left_shoulder_origin": 1,
        "arm_roots_clear_of_head_and_neck": 1,
    }


def test_promotion_rejects_unreviewed_or_partial_topology_evidence() -> None:
    baseline = _metrics(topology_integrity=0.60)
    candidate = _metrics(topology_integrity=1.0)
    missing = IdentityPromotionContract().assess(baseline, candidate)
    assert not missing.passed
    assert "topology_evidence" in missing.failed_gates()

    full = _topology_evidence()
    partial = TopologyReviewEvidence(
        video_sha256=full.video_sha256,
        total_frames=full.total_frames,
        reviewer=full.reviewer,
        review_method=full.review_method,
        frames=full.frames[:-1],
    )
    incomplete = IdentityPromotionContract().assess(baseline, candidate, partial)
    assert not incomplete.passed
    assert "topology_full_frame_coverage" in incomplete.failed_gates()


def test_promotion_rejects_topology_labels_without_per_frame_digests() -> None:
    evidence = _topology_evidence()
    unbound = TopologyReviewEvidence(
        video_sha256=evidence.video_sha256,
        total_frames=evidence.total_frames,
        reviewer=evidence.reviewer,
        review_method=evidence.review_method,
        frames=tuple(
            TopologyFrameReview(
                **{
                    **frame.__dict__,
                    "decoded_frame_sha256": None,
                }
            )
            for frame in evidence.frames
        ),
    )
    assessment = IdentityPromotionContract().assess(
        _metrics(), _metrics(topology_integrity=1.0), unbound
    )
    assert not assessment.passed
    assert "topology_decoded_frame_digests" in assessment.failed_gates()


def test_promotion_rejects_legacy_topology_without_kinematic_detail() -> None:
    evidence = _topology_evidence()
    legacy = TopologyReviewEvidence(
        video_sha256=evidence.video_sha256,
        total_frames=evidence.total_frames,
        reviewer=evidence.reviewer,
        review_method=evidence.review_method,
        frames=tuple(
            TopologyFrameReview(
                **{
                    **frame.__dict__,
                    "unique_left_shoulder_origin": None,
                    "unique_right_shoulder_origin": None,
                    "arm_roots_clear_of_head_and_neck": None,
                }
            )
            for frame in evidence.frames
        ),
    )
    assessment = IdentityPromotionContract().assess(
        _metrics(), _metrics(topology_integrity=1.0), legacy
    )
    assert not assessment.passed
    assert "topology_kinematic_detail" in assessment.failed_gates()


def test_topology_evidence_rejects_string_boolean(tmp_path) -> None:
    payload = {
        "schema_version": "1.0.0",
        "video_sha256": "b" * 64,
        "total_frames": 1,
        "reviewer": "reviewer",
        "review_method": "manual",
        "frames": [
            {
                "frame_index": 0,
                "single_robot_subject": "false",
                "single_head_torso_chain": True,
                "exactly_two_arms": True,
                "left_shoulder_attachment": True,
                "right_shoulder_attachment": True,
                "continuous_arm_segments": True,
                "stable_robot_proportions": True,
                "no_extra_or_missing_limbs": True,
                "no_human_residual": True,
                "confidence": 0.99,
            }
        ],
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="JSON booleans"):
        TopologyReviewEvidence.load(path)


def test_plan_rejects_test_subject_leakage(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    common = {
        "prompt": "Keep the referenced robot identity consistent.",
        "source_start_seconds": 0,
        "reference_frame": 0,
        "license_id": "Apache-2.0",
        "source_uri": "https://example.invalid/source",
        "review_status": "accepted",
    }
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "name": "leaky-plan",
                "fps": 24,
                "width": 448,
                "height": 256,
                "num_frames": 39,
                "clips": [
                    {
                        **common,
                        "clip_id": "train-a",
                        "subject_id": "same-subject",
                        "scene_id": "scene-a",
                        "split": "train",
                        "source_video": "outputs/a.mp4",
                    },
                    {
                        **common,
                        "clip_id": "train-b",
                        "subject_id": "second-subject",
                        "scene_id": "scene-b",
                        "split": "train",
                        "source_video": "outputs/b.mp4",
                    },
                    {
                        **common,
                        "clip_id": "test-a",
                        "subject_id": "same-subject",
                        "scene_id": "scene-c",
                        "split": "test",
                        "source_video": "outputs/c.mp4",
                    },
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="test subjects leak"):
        IdentityDatasetPlan.load(plan_path)


def test_diffsynth_metadata_keeps_reference_explicit(tmp_path) -> None:
    payload = {
        "schema_version": "1.0.0",
        "name": "valid-plan",
        "fps": 24,
        "width": 448,
        "height": 256,
        "num_frames": 39,
        "clips": [
            {
                "clip_id": "train-a",
                "subject_id": "subject-a",
                "scene_id": "scene-a",
                "split": "train",
                "source_video": "outputs/a.mp4",
                "prompt": "  keep   identity  ",
                "source_start_seconds": 0,
                "reference_frame": 0,
                "license_id": "Apache-2.0",
                "source_uri": "https://example.invalid/a",
                "review_status": "accepted",
                "source_crop": [96, 0, 448, 256],
            },
            {
                "clip_id": "train-b",
                "subject_id": "subject-b",
                "scene_id": "scene-b",
                "split": "train",
                "source_video": "outputs/b.mp4",
                "prompt": "keep identity",
                "source_start_seconds": 0,
                "reference_frame": 0,
                "license_id": "Apache-2.0",
                "source_uri": "https://example.invalid/b",
                "review_status": "accepted",
            },
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload))
    plan = IdentityDatasetPlan.load(path)
    assert plan.clips[0].source_crop == (96, 0, 448, 256)
    metadata = build_diffsynth_metadata(((plan.clips[0], "clips/a.mp4", "references/a.png"),))
    assert metadata[0]["prompt"] == "keep identity"
    assert metadata[0]["input_audio"] == "clips/a.mp4"
    assert metadata[0]["references"] == [{"type": "image", "image": "references/a.png"}]


def test_plan_rejects_non_integer_source_crop(tmp_path) -> None:
    payload = {
        "schema_version": "1.0.0",
        "name": "invalid-crop-plan",
        "fps": 24,
        "width": 448,
        "height": 256,
        "num_frames": 39,
        "clips": [
            {
                "clip_id": f"train-{index}",
                "subject_id": f"subject-{index}",
                "scene_id": f"scene-{index}",
                "split": "train",
                "source_video": f"outputs/{index}.mp4",
                "prompt": "keep identity",
                "source_start_seconds": 0,
                "reference_frame": 0,
                "license_id": "Apache-2.0",
                "source_uri": "https://example.invalid/source",
                "review_status": "accepted",
                "source_crop": [96, 0, 448, "256"],
            }
            for index in range(2)
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="only JSON integers"):
        IdentityDatasetPlan.load(path)


def test_domain_curriculum_requires_diverse_disjoint_splits() -> None:
    required = DomainCurriculumContract().required_training_tags
    clips = []
    for index in range(3):
        clips.append(
            IdentityClipSpec(
                clip_id=f"train-{index}",
                subject_id=f"train-robot-{index}",
                scene_id=f"train-scene-{index}",
                split="train",
                source_video=f"outputs/train-{index}.mp4",
                prompt="preserve both shoulder roots",
                source_start_seconds=0,
                reference_frame=0,
                license_id="Pexels License plus project render",
                source_uri=f"https://example.invalid/train-{index}",
                review_status="accepted",
                curriculum_tags=required if index == 0 else ("real-background",),
            )
        )
    for split, offset in (("validation", 0), ("test", 1), ("test", 2)):
        clips.append(
            IdentityClipSpec(
                clip_id=f"{split}-{offset}",
                subject_id=f"heldout-robot-{offset}",
                scene_id=f"heldout-scene-{offset}",
                split=split,
                source_video=f"outputs/{split}-{offset}.mp4",
                prompt="held-out shoulder-root evaluation",
                source_start_seconds=0,
                reference_frame=0,
                license_id="Pexels License plus project render",
                source_uri=f"https://example.invalid/{split}-{offset}",
                review_status="accepted",
                curriculum_tags=("real-background",),
            )
        )
    plan = IdentityDatasetPlan(
        name="domain-diverse-r4",
        fps=24,
        width=448,
        height=256,
        num_frames=39,
        clips=tuple(clips),
    )
    plan.validate()
    assessment = DomainCurriculumContract().assess(plan)
    assert assessment.passed
    assert not assessment.failed_gates()


def test_domain_curriculum_rejects_missing_real_domain_coverage() -> None:
    clips = tuple(
        IdentityClipSpec(
            clip_id=f"train-{index}",
            subject_id=f"robot-{index}",
            scene_id="one-render-domain",
            split="train",
            source_video=f"outputs/train-{index}.mp4",
            prompt="generic topology",
            source_start_seconds=0,
            reference_frame=0,
            license_id="project render",
            source_uri=f"https://example.invalid/{index}",
            review_status="partial",
        )
        for index in range(3)
    )
    plan = IdentityDatasetPlan(
        name="undercovered-r4",
        fps=24,
        width=448,
        height=256,
        num_frames=39,
        clips=clips,
    )
    plan.validate()
    assessment = DomainCurriculumContract().assess(plan)
    assert not assessment.passed
    assert "training_scene_diversity" in assessment.failed_gates()
    assert "required_training_tags" in assessment.failed_gates()
    assert "heldout_scene_diversity" in assessment.failed_gates()
