---
name: physical-video-planner
description: Expand a manipulation-video request into a hash-bound language and physical phase plan, freeze material/contact/task hard gates, run failure-aware test-time scaling, and audit candidates without allowing aggregate scores to override failed gates. Use for foundation-model robot videos that must preserve object or cloth identity, geometry, contact causality, action order, and continuous motion.
---

# Physical Video Planner

Treat MiniMax-H3 and other foundation models as proposal generators. Physical,
material, temporal, task-order, and human-review gates are the acceptance
authority.

## Locate the implementation

Run `git rev-parse --show-toplevel` from the active PhiAgent workspace. Require
these files before using the workflow:

- `scripts/compile_physical_video_plan.py`
- `scripts/build_tshirt_length_preserving_carrier.py` for carrier-guided T-shirt retakes
- `scripts/build_tshirt_articulated_contact_carrier.py` when the cloth carrier
  must be coupled to connected dual-arm motion
- `scripts/run_agentic_acwm.py`
- `scripts/evaluate_tshirt_fold_candidate.py` for T-shirt tasks
- `phiagent/harness/task_reasoning.py`
- `phiagent/harness/test_time_scaling.py`

Do not hardcode a user or host path. If the active repository is not PhiAgent,
stop and explain that the runtime implementation is unavailable.

## Immutable decision rules

1. Keep camera, world, and robot-base frames distinct. Viewer-left and
   viewer-right are camera-frame relations only.
2. A proposal fails when any required hard gate is false or unavailable.
   Never use a mean score, a preferred-looking seed, or more inference steps to
   override a failed gate.
3. Test-time scaling may add seeds, inference steps, or gate-specific repair
   prompts. It may not relax thresholds, delete gates, reuse an output
   directory, or silently change the tracking contract.
4. Tracker loss is a failure, not an occlusion exemption. Preserve material
   endpoint identity and reacquire it before making a claim.
5. Image-space material tracks are visual production evidence, not calibrated
   3-D cloth geometry. Pixels or overlap do not establish contact force,
   collision safety, joint feasibility, or real-robot success.
6. Native-resolution human review remains a veto after automatic gates pass.

## T-shirt folding contract

For a left-sleeve, right-sleeve, body-fold, move-aside task, require independent
hard gates for:

- the exact first frame;
- exactly one persistent shirt and two original sleeves;
- viewer-left cuff-to-shoulder material length and segment conservation;
- viewer-right cuff-to-shoulder material length and segment conservation;
- cuff and shoulder-seam identity through overlap;
- stabilizing and cuff-side contact before sleeve motion;
- the viewer-left fold settling before viewer-right motion;
- no cut, dissolve, crossfade, teleport, or single-frame material jump;
- body folding only after both sleeves settle;
- bundle transport only after body folding;
- a fixed camera and background;
- a compact, stable terminal bundle.

## Workflow

1. Inspect the real first frame and define one named camera frame. Write a typed
   request JSON containing the instruction, entities, available evidence,
   unavailable evidence, and user constraints.
2. Run `scripts/compile_physical_video_plan.py` in a new output directory. Read
   the complete plan and verify its SHA-256 before generating.
3. Freeze the candidate-independent tracking contract: initial material
   landmarks, both gripper point sets, phase frame windows, background patches,
   contact distance, and all other thresholds. Do not tune it after viewing
   candidate results. Fail closed if gripper tracks are absent.
4. Run `scripts/run_agentic_acwm.py --plan-only` with
   `--task-reasoning-plan`, `--test-time-scaling-config`, and the relevant
   tracking contract. Confirm prompt expansion, coordinate-frame equality,
   candidate allocation, and hashes.
5. Run GPU preflight. The harness must inventory physical GPUs, bind a UUID in
   `CUDA_VISIBLE_DEVICES`, validate pinned model/source hashes, and save the
   selection.
6. Generate the first scaling round. Evaluate every candidate independently.
   If an evaluator crashes, preserve the candidate, mark it fail-closed, and do
   not spend another scaling round until the evaluator is repaired.
7. For ordinary hard-gate failures, use their exact IDs to build the next repair
   prompts and increase compute according to the frozen scaling policy. Do not
   change thresholds or the initial material tracks.
8. If scaled static-reference proposals repeatedly fail the same cloth-motion
   gates, treat that as an architecture failure rather than adding more seeds.
   Predeclare a new experiment and build a continuous carrier with rigid sleeve
   rotations using `scripts/build_tshirt_length_preserving_carrier.py`. Verify
   analytically that every frozen sleeve segment length is unchanged before
   using the carrier as H3 Video 1. Reject a cloth-only carrier that makes the
   garment move while both manipulators remain stationary. The carrier remains
   a control proposal, not acceptance evidence, and all original output gates
   remain frozen.
9. Before sending a cloth carrier back to H3, reject it if the garment moves
   while the manipulators remain stationary or if raster arm motion creates
   detached links. When no exact robot URDF/MJCF and calibrated camera are
   available, use `scripts/build_tshirt_articulated_contact_carrier.py` to
   compile explicitly synthetic fixed-base planar camera rigs. Require a
   complete per-frame node chain, `q`, and `qdot`; conserved link lengths; a
   bounded joint step and tip step; and named gripper contact before the
   corresponding sleeve motion. Record that this is a proposal-control rig,
   not the unidentified real robot, metric kinematics, force, collision safety,
   or executable commands. Do not replace visual output failures with analytic
   carrier claims, and do not let a protected compositing pass override a
   failed first-frame, background, material-tracking, or human-review gate.
10. Rank only candidates that passed every automatic hard gate. Then require a
   candidate-SHA-bound native-resolution human review before acceptance.
11. Append every success, rejection, failure, or blocker to
   `experiences/ledger.jsonl` through `scripts/experience_ledger.py`.

## Reporting

Report automatic gate results separately from diagnostic scores. Use
`PARTIAL` when a visual candidate exists but calibrated physical evidence or
explicit user review is missing. Never describe camera-pixel sleeve tracking as
metric inextensibility or real-world execution.
