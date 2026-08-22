"""JoyAI repair planning for persistent, cooperative dual-gripper T-shirt folding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .task_reasoning import (
    SCHEMA_VERSION,
    TSHIRT_FOLD_TASK,
    ReasoningPluginDescriptor,
    TaskReasoningPlan,
    TaskReasoningRequest,
    VerificationGate,
    _canonical_sha256,
)
from .tshirt_fold_strategy import (
    LEFT_THEN_RIGHT,
    VIEWER_RIGHT,
    TshirtFoldStrategy,
    TshirtFoldStrategyReasoningPlugin,
)


@dataclass(frozen=True)
class GripperClothContactContract:
    """One phase-local visual contract for a named gripper and cloth patch."""

    phase_id: str
    active_gripper: str
    active_patch: str
    anchor_gripper: str
    anchor_patch: str
    supported_motion: str


CONTACT_DETAIL_CONTRACTS = {
    contract.phase_id: contract
    for contract in (
        GripperClothContactContract(
            "fold_viewer_left_sleeve",
            "lower-left gripper",
            "same viewer-left cuff patch",
            "upper-right gripper",
            "same viewer-right shoulder patch",
            "one continuous cuff-to-shoulder inward fold",
        ),
        GripperClothContactContract(
            "fold_viewer_right_sleeve",
            "upper-right gripper",
            "same viewer-right cuff patch",
            "lower-left gripper",
            "same settled left-fold and adjacent torso patch",
            "one continuous cuff-to-shoulder inward fold",
        ),
        GripperClothContactContract(
            "fold_body_bottom_to_top",
            "both grippers",
            "two opposite lower-hem corner patches",
            "both grippers",
            "same two lower-hem patches",
            "one level bilateral lower-hem lift and body fold",
        ),
        GripperClothContactContract(
            "move_folded_bundle_viewer_right",
            "both grippers",
            "two opposite compact-bundle side-edge patches",
            "both grippers",
            "same two compact-bundle side-edge patches",
            "one level bilateral bundle transport",
        ),
    )
}


class TshirtCooperativeJoyAIRepairPlugin:
    """Plan a continuous two-arm edit with local handoffs and no home resets."""

    descriptor = ReasoningPluginDescriptor(
        name="tshirt-cooperative-joyai-repair-planner",
        version="2.1.0",
        stage="reasoning",
        description=(
            "Replans a generated T-shirt video as persistent two-gripper manipulation: "
            "one gripper anchors while the other folds, roles swap locally, and both "
            "grippers fold and transport the body without returning home."
        ),
        capabilities=(
            "language_analysis",
            "task_expansion",
            "joyai_video_repair",
            "persistent_workspace_contact",
            "dual_gripper_cooperation",
            "local_role_handoff",
            "cloth_material_conservation",
            "persistent_two_jaw_topology",
            "named_cloth_patch_binding",
            "contact_deformation_causality",
            "supported_release_microsequence",
            TSHIRT_FOLD_TASK,
        ),
        deterministic=True,
        heavyweight=False,
    )

    def analyze(self, request: TaskReasoningRequest) -> TaskReasoningPlan:
        if request.task_type != TSHIRT_FOLD_TASK:
            raise ValueError(f"unsupported task type: {request.task_type}")
        base = TshirtFoldStrategyReasoningPlugin().analyze_strategy(
            request,
            TshirtFoldStrategy(LEFT_THEN_RIGHT, VIEWER_RIGHT),
        )
        payload = base.to_dict()
        payload["schema_version"] = SCHEMA_VERSION
        payload["plugin"] = asdict(self.descriptor)
        payload["task_id"] = f"{base.task_id}--joyai-cooperative"
        payload["language_analysis"]["normalized_instruction"] = (
            "Preserve the exact real scene, one intact T-shirt, two original white arms, "
            "and two original black parallel two-jaw grippers. Each gripper always has one "
            "rigid palm, one connected wrist collar, and exactly two opposing jaw plates; "
            "never replace a jaw with fingers or merge cloth into the gripper. Before every "
            "cloth motion, open jaws bracket one named cloth patch, both inner jaw faces "
            "close onto that same patch, and a short preload hold shows local compression "
            "without sliding. During motion, the same texture patch stays pinned between the "
            "same jaw faces while the partner gripper anchors a second named patch. Fold the "
            "left cuff, exchange roles locally, fold the right cuff, grasp opposite lower-hem "
            "corners, fold the body, then grasp opposite compact-bundle side edges and move "
            "viewer-right. Lower and settle before opening; after release the cloth stays "
            "still and both grippers remain nearby instead of returning home."
        )
        payload["language_analysis"]["temporal_modifiers"] = [
            "both contacts establish before the first cloth motion",
            "left active fold with right anchor",
            "local role handoff without home-pose retreat",
            "right active fold with left anchor",
            "bilateral hem grasp before body fold",
            "bilateral support through viewer-right transport",
            "release only after table support and terminal settling",
            "open jaws visibly bracket the named cloth patch before closure",
            "closed jaw faces retain the same cloth texture patch without slip",
            "cloth settles for multiple frames before jaws open",
        ]
        payload["language_analysis"]["ambiguity_resolutions"] = [
            *payload["language_analysis"]["ambiguity_resolutions"],
                (
                    "Remain in the cloth workspace means neither gripper returns to its "
                    "initial retracted pose between manipulation phases."
                ),
                (
                    "Cooperation means one active cuff trajectory plus one visible "
                    "shoulder/torso anchor during sleeve folds, followed by bilateral "
                    "hem and bundle support; it does not imply force measurements."
                ),
                (
                    "Visible contact means cloth lies between two identifiable inner jaw "
                    "faces with local compression and occlusion; mere silhouette overlap, "
                    "surface hovering, or cloth passing through a jaw is a hard failure."
                ),
                (
                    "Gripper consistency means one connected wrist, one rigid black palm, "
                    "and exactly two persistent opposing jaw plates for each original arm; "
                    "human-like fingers, extra tips, fused jaws, or detached wrists fail."
                ),
        ]

        cooperative_gates = (
            VerificationGate(
                "both_grippers_remain_in_cloth_workspace",
                (
                    "After first contact, both grippers remain near the shirt and never "
                    "return to their initial retracted poses before terminal settling."
                ),
                "native_resolution_human_review",
            ),
            VerificationGate(
                "cooperative_anchor_active_role_visible",
                (
                    "During each sleeve fold, one gripper visibly guides the cuff while "
                    "the other visibly anchors the corresponding shoulder or torso."
                ),
                "native_resolution_human_review",
            ),
            VerificationGate(
                "local_handoff_without_home_reset",
                (
                    "Between sleeve folds, both grippers reposition along the cloth by "
                    "short continuous paths without withdrawing to their starting poses."
                ),
                "native_resolution_human_review",
            ),
            VerificationGate(
                "bilateral_hem_fold_and_bundle_transport",
                (
                    "Both grippers grasp opposite lower-hem corners before the body fold "
                    "and support opposite sides of the completed bundle during transport."
                ),
                "native_resolution_human_review",
            ),
            VerificationGate(
                "release_only_after_supported_settle",
                (
                    "A gripper releases cloth only after that cloth is visibly supported "
                    "by the tabletop or the other maintained gripper and has settled."
                ),
                "native_resolution_human_review",
            ),
            VerificationGate(
                "persistent_two_jaw_gripper_topology",
                (
                    "Each original gripper preserves one wrist-connected rigid palm and "
                    "exactly two opposing jaw plates in every frame; no fingers, extra tips, "
                    "fusion, wrist detachment, or identity swap appears."
                ),
                "native_resolution_human_review",
            ),
            VerificationGate(
                "dual_gripper_temporal_stability",
                (
                    "Each tracked gripper follows a smooth pose path: centroid, principal "
                    "axis, and mask area have bounded three-frame second differences with "
                    "no alternating one-frame twitch, scale pulse, or pose reversal."
                ),
                "automatic_proxy",
            ),
            VerificationGate(
                "dual_gripper_sharpness_persistent",
                (
                    "Mask-interior Laplacian detail for both grippers stays above the "
                    "absolute and canonical-reference sharpness floors in every frame; "
                    "motion blur and soft donor patches are hard failures."
                ),
                "automatic_proxy",
            ),
            VerificationGate(
                "redqueen_evaluator_attack_replay_passed",
                (
                    "The frozen evaluator epoch rejects every bounded jitter, blur, "
                    "identity, topology, and contact attack in its hash-bound Red Queen "
                    "pool without regressing any known-positive anchor."
                ),
                "automatic_proxy",
            ),
            VerificationGate(
                "jaw_closure_precedes_named_patch_motion",
                (
                    "Open jaws first bracket the named cloth patch, then both inner faces "
                    "close and hold briefly; the patch remains still until closure completes."
                ),
                "native_resolution_human_review",
            ),
            VerificationGate(
                "same_cloth_patch_pinned_without_slip",
                (
                    "The same visible texture patch remains between the same two inner jaw "
                    "faces through lift, fold, or transport without sliding, reassignment, "
                    "penetration, disappearance, or texture replacement."
                ),
                "native_resolution_human_review",
            ),
            VerificationGate(
                "local_contact_deformation_and_occlusion",
                (
                    "Jaw closure produces bounded local cloth compression or wrinkling and "
                    "a physically ordered occlusion boundary while the rest of the garment "
                    "does not move prematurely."
                ),
                "native_resolution_human_review",
            ),
            VerificationGate(
                "supported_release_microsequence",
                (
                    "The carried patch is lowered onto visible support, remains settled for "
                    "multiple frames, then the jaws open while cloth stays in place and the "
                    "gripper makes only a short local withdrawal."
                ),
                "native_resolution_human_review",
            ),
            VerificationGate(
                "precontact_cloth_state_static",
                (
                    "Before both required jaw closures, the complete shirt preserves "
                    "its position, contour, opacity, weave, seams, sleeves, hem, and "
                    "table support without any fold, lift, drag, fade, or replacement."
                ),
                "automatic_proxy",
            ),
            VerificationGate(
                "no_unexplained_cloth_state_change",
                (
                    "Every cloth-state change is a continuous deformation connected to "
                    "closed-jaw motion on named patches; no disappearance, transparency, "
                    "crossfade, sudden shape replacement, self-motion, or unsupported jump."
                ),
                "automatic_proxy",
            ),
            VerificationGate(
                "manipulation_stage_requires_native_contact_start",
                (
                    "Any stage that folds, lifts, or transports cloth starts only from a "
                    "hash-bound keyframe whose bilateral named-patch clamp passed native "
                    "contact review."
                ),
                "native_resolution_human_review",
            ),
            VerificationGate(
                "manipulation_stage_requires_native_gripper_identity",
                (
                    "Any cloth-manipulation stage starts only after both original "
                    "wrist-connected grippers match a hash-bound canonical identity "
                    "reference and pass native two-jaw topology review."
                ),
                "native_resolution_human_review",
            ),
        )
        payload["verification_gates"] = [
            *payload["verification_gates"],
            *(asdict(gate) for gate in cooperative_gates),
        ]
        phases = {phase["phase_id"]: phase for phase in payload["phases"]}
        self._rewrite_phase(
            phases,
            "establish_viewer_left_two_point_contact",
            directive=(
                "Both wrist-connected parallel grippers approach once with exactly two open "
                "jaw plates. The lower-left jaws bracket one viewer-left cuff patch and the "
                "upper-right jaws bracket one viewer-right shoulder patch. Cloth stays still "
                "while both jaw gaps close; each patch becomes visibly sandwiched between two "
                "inner faces with a small local crease before either arm loads it."
            ),
            invariant=(
                "Both grippers stay in the cloth workspace; no home-pose return or arm reset."
            ),
            gates=(
                "both_grippers_remain_in_cloth_workspace",
                "cooperative_anchor_active_role_visible",
                "persistent_two_jaw_gripper_topology",
                "dual_gripper_temporal_stability",
                "dual_gripper_sharpness_persistent",
                "redqueen_evaluator_attack_replay_passed",
                "jaw_closure_precedes_named_patch_motion",
                "local_contact_deformation_and_occlusion",
                "precontact_cloth_state_static",
            ),
        )
        self._rewrite_phase(
            phases,
            "fold_viewer_left_sleeve",
            directive=(
                "Keep the same viewer-left cuff texture patch pinned between the same two "
                "lower-left inner jaw faces with no sliding. The upper-right jaws retain the "
                "same shoulder patch. Guide the full cuff-to-shoulder sleeve inward in one "
                "continuous fold; show bending at the shoulder seam and bounded wrinkles at "
                "both contacts, never cloth-jaw penetration or material reassignment."
            ),
            invariant=(
                "Active and anchor contacts remain visible; neither gripper retreats or releases early."
            ),
            gates=(
                "cooperative_anchor_active_role_visible",
                "both_grippers_remain_in_cloth_workspace",
                "release_only_after_supported_settle",
                "persistent_two_jaw_gripper_topology",
                "same_cloth_patch_pinned_without_slip",
                "local_contact_deformation_and_occlusion",
                "precontact_cloth_state_static",
                "no_unexplained_cloth_state_change",
            ),
        )
        self._rewrite_phase(
            phases,
            "settle_viewer_left_sleeve",
            directive=(
                "Settle the folded left sleeve. Keep the lower-left gripper close as a "
                "local torso anchor while the upper-right gripper slides continuously "
                "along the cloth workspace toward the viewer-right cuff; neither returns home."
            ),
            invariant="The first fold stays supported throughout the local role handoff.",
            gates=(
                "local_handoff_without_home_reset",
                "release_only_after_supported_settle",
            ),
        )
        self._rewrite_phase(
            phases,
            "establish_viewer_right_two_point_contact",
            directive=(
                "Without any arm reset, the upper-right two open jaw plates bracket one named "
                "viewer-right cuff texture patch. The lower-left two-jaw gripper brackets the "
                "settled left-fold/torso anchor patch. Both gaps close before the right cuff "
                "moves, producing local compression without swallowing or replacing cloth."
            ),
            invariant="Both grippers remain close to cloth through the role exchange.",
            gates=(
                "local_handoff_without_home_reset",
                "cooperative_anchor_active_role_visible",
                "both_grippers_remain_in_cloth_workspace",
                "persistent_two_jaw_gripper_topology",
                "jaw_closure_precedes_named_patch_motion",
                "local_contact_deformation_and_occlusion",
                "precontact_cloth_state_static",
            ),
        )
        self._rewrite_phase(
            phases,
            "fold_viewer_right_sleeve",
            directive=(
                "Keep the same viewer-right cuff texture patch pinned between the same two "
                "upper-right inner jaw faces, and keep the same settled left-fold/torso patch "
                "between the lower-left jaw faces. Fold the full right cuff inward through "
                "one shoulder-seam bend with persistent two-jaw silhouettes, bounded contact "
                "wrinkles, no slip, no cloth-jaw fusion, and unchanged sleeve material."
            ),
            invariant="Both roles remain visible and the anchor prevents torso sliding.",
            gates=(
                "cooperative_anchor_active_role_visible",
                "both_grippers_remain_in_cloth_workspace",
                "release_only_after_supported_settle",
                "persistent_two_jaw_gripper_topology",
                "same_cloth_patch_pinned_without_slip",
                "local_contact_deformation_and_occlusion",
                "precontact_cloth_state_static",
                "no_unexplained_cloth_state_change",
            ),
        )
        self._rewrite_phase(
            phases,
            "fold_body_bottom_to_top",
            directive=(
                "Both grippers move locally, not home, to opposite lower-hem corners. Both "
                "wrist-connected two-jaw tools target one named corner patch. Each pair of "
                "open jaws brackets its corner, closes with the "
                "same patch visible between both inner faces, and holds briefly before lift. "
                "Keep both patches pinned while lifting at matched height and speed and hinge "
                "the complete lower body upward as one cloth layer."
            ),
            invariant="Both hem contacts persist and neither side leads by a large jump.",
            gates=(
                "local_handoff_without_home_reset",
                "bilateral_hem_fold_and_bundle_transport",
                "release_only_after_supported_settle",
                "persistent_two_jaw_gripper_topology",
                "jaw_closure_precedes_named_patch_motion",
                "same_cloth_patch_pinned_without_slip",
                "local_contact_deformation_and_occlusion",
                "precontact_cloth_state_static",
                "no_unexplained_cloth_state_change",
                "manipulation_stage_requires_native_contact_start",
                "manipulation_stage_requires_native_gripper_identity",
            ),
        )
        self._rewrite_phase(
            phases,
            "compress_bundle_without_stretch",
            directive=(
                "Both grippers stay beside the bundle, square opposite edges together, "
                "and keep every cloth layer supported without stretching or disappearing."
            ),
            invariant="No gripper returns to its initial pose between body fold and transport.",
            gates=(
                "both_grippers_remain_in_cloth_workspace",
                "bilateral_hem_fold_and_bundle_transport",
            ),
        )
        self._rewrite_phase(
            phases,
            "move_folded_bundle_viewer_right",
            directive=(
                "Both grippers support opposite sides of the compact bundle. Each two-jaw "
                "tool closes on one side-edge patch, not on the top surface and not on empty "
                "air. The same two patches remain "
                "pinned while both arms translate viewer-right together. Lower the full base "
                "onto the table, hold it motionless for multiple frames, open both jaw gaps, "
                "verify the cloth does not follow either jaw, then withdraw only locally."
            ),
            invariant="Bilateral support persists throughout transport; no single-arm dragging.",
            gates=(
                "bilateral_hem_fold_and_bundle_transport",
                "release_only_after_supported_settle",
                "persistent_two_jaw_gripper_topology",
                "same_cloth_patch_pinned_without_slip",
                "supported_release_microsequence",
                "no_unexplained_cloth_state_change",
                "manipulation_stage_requires_native_contact_start",
                "manipulation_stage_requires_native_gripper_identity",
            ),
        )
        self._rewrite_phase(
            phases,
            "terminal_bundle_hold",
            directive=(
                "Hold one intact compact shirt viewer-right. Both grippers remain nearby "
                "after release; do not snap back to their initial poses in the final frames."
            ),
            invariant="No terminal arm reset, bundle drift, unfolding, or scene change.",
            gates=("both_grippers_remain_in_cloth_workspace",),
        )
        payload["global_constraints"] = [
            *payload["global_constraints"],
                (
                    "After first contact, both grippers stay in the cloth workspace and "
                    "reposition locally; returning to either initial pose is a hard failure."
                ),
                (
                    "Every sleeve fold visibly contains one active cuff guide and one "
                    "shoulder/torso anchor using the two original grippers."
                ),
                (
                    "The body fold and final transport use bilateral support; one gripper "
                    "may not drag, throw, or teleport the shirt alone."
                ),
                "No gripper releases moving or unsupported cloth.",
                (
                    "At every contact, the active cloth patch must be visibly between two "
                    "inner jaw faces; silhouette overlap or a gripper resting on top is not contact."
                ),
                (
                    "The black grippers remain rigid mechanical two-jaw tools, never soft "
                    "hands, fingers, claws, merged black cloth, or changing jaw counts."
                ),
                (
                    "Contact detail is local: preserve garment texture, seams, sleeve length, "
                    "layer count, and all pixels outside the declared gripper/contact support."
                ),
                (
                    "Before required jaws close, the shirt is visually static. Afterwards "
                    "every cloth change must remain continuously attached to closed-jaw motion; "
                    "a disappearing, fading, self-moving, or suddenly replaced garment fails."
                ),
                (
                    "A fold, lift, or transport generation stage may start only from a "
                    "hash-bound contact keyframe with native contact approval."
                ),
                (
                    "The same stage also requires native approval that both original "
                    "grippers match a hash-bound canonical two-jaw identity reference."
                ),
        ]
        payload["claim_boundary"] = (
            "This JoyAI repair plan constrains camera-frame visual evidence for persistent "
            "two-gripper cooperation, local handoffs, cloth continuity, and task order. "
            "Its automatic cloth tracks and native-resolution gripper reviews do not "
            "establish force, friction, calibrated 3-D cloth geometry, robot joint "
            "feasibility, collision safety, or physical execution."
        )
        payload.pop("plan_sha256", None)
        return TaskReasoningPlan.from_dict(
            {**payload, "plan_sha256": _canonical_sha256(payload)}
        )

    @staticmethod
    def _rewrite_phase(
        phases: dict[str, dict[str, Any]],
        phase_id: str,
        *,
        directive: str,
        invariant: str,
        gates: tuple[str, ...],
    ) -> None:
        phase = phases[phase_id]
        phase["language_directive"] = directive
        phase["invariants"] = list(dict.fromkeys((*phase["invariants"], invariant)))
        phase["gate_ids"] = list(dict.fromkeys((*phase["gate_ids"], *gates)))

    @staticmethod
    def contact_detail_prompt(
        *,
        phase_id: str,
        frame_start: int,
        frame_end: int,
        plan_sha256: str,
    ) -> str:
        """Compile a phase-only JoyAI prompt focused on jaw/cloth consistency."""

        if phase_id not in CONTACT_DETAIL_CONTRACTS:
            raise ValueError(f"phase has no contact-detail contract: {phase_id}")
        if frame_start < 0 or frame_end < frame_start:
            raise ValueError("invalid contact-detail frame window")
        contract = CONTACT_DETAIL_CONTRACTS[phase_id]
        bilateral = contract.active_gripper == "both grippers"
        active_verb = "use" if bilateral else "uses"
        anchor_verb = "retain" if bilateral else "retains"
        patch_pronoun = "those exact texture patches" if bilateral else "that exact texture patch"
        jaw_owner = (
            "their corresponding two inner jaw faces"
            if bilateral
            else "its two inner jaw faces"
        )
        return (
            f"Correct only frames {frame_start}-{frame_end} for phase "
            f"{contract.phase_id}; preserve timing, camera, arm paths, garment shape, "
            "and every pixel outside the two gripper/contact neighborhoods. "
            "Identity lock: preserve the two original white arms. Each black gripper "
            "must remain one rigid palm connected to one wrist collar with exactly two "
            "opposing parallel jaw plates; no fingers, extra tips, fused jaws, detached "
            "wrist, identity swap, or black cloth merged into a tool. "
            f"Active contact: {contract.active_gripper} {active_verb} open jaws to bracket the "
            f"{contract.active_patch}; both inner faces close before cloth motion, hold "
            f"briefly, and keep {patch_pronoun} pinned with no slip. "
            f"Anchor contact: {contract.anchor_gripper} {anchor_verb} the "
            f"{contract.anchor_patch} between {jaw_owner}. "
            f"Motion: {contract.supported_motion}. At each closure show a small local "
            "compression crease and correct foreground/background occlusion, not mere "
            "silhouette overlap, hovering, or cloth passing through a jaw. Preserve cloth "
            "opacity, weave, seams, sleeve length, width, layer count, and continuous "
            "attachment to the shirt. Lower onto visible support and hold settled before "
            "opening; cloth remains still after opening and each arm withdraws only locally. "
            "Render no text, guide marks, borders, UI, or watermark. "
            f"Hash-bound cooperative plan: {plan_sha256}."
        )
