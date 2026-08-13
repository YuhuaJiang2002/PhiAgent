---
name: acwm-long-action-control
description: Build, train, evaluate, and package sharp, human-free, stateful 8-12 second AC-WM action-control comparisons from licensed public egocentric datasets. Use when asked to extend a short AC-WM manipulation clip, make or improve a 10-second Ego embodied-task demo, remove hand ghosts or blur, preserve grasp or object-holder state across model windows, optimize a domain-matched action or repair module, or compare multiple controlled robot-arm actions. Default new demos to public Ego data such as EPIC-KITCHENS; never route them to the legacy flower scene.
---

# AC-WM Long Action Control

Operate from the PhiAgent repository root. Preserve existing dirty-worktree changes and create a new immutable experiment directory for every attempt.

## Workflow

1. Read `AGENTS.md`, `docs/ACWM_WORKFLOW.md`, `docs/STATUS.md`, and the relevant existing experiment manifests. Default this repository's current public-Ego route to the annotated EPIC-KITCHENS `P03_28` bottle interval; require a new licensed and annotated preparation manifest before choosing another scene. Confirm that every action uses the same real source interval, robot reference, seed, frame count, model revision, and inference settings. Retain the dataset license, official annotations, download verification, source hash, and exact interval.

2. Represent long actions in a task-specific JSON file under `configs/`. Keep phases contiguous, declare `object_name` and `coordinate_frame`, and name the object holder at every transition. Never relabel a `camera:*` 2D trajectory as `robot_base:*` EEF, joints, or pointmap input.

3. Compile a 240-frame control for each action with the matching domain adapter. Use `scripts/build_epic_ego_bottle_action_controls.py` for the default EPIC-KITCHENS bottle case. Treat these videos as intermediate controls, not generated results; do not use the legacy flower adapter for a new Ego demo.

4. Prepare two legal H3 windows with `scripts/prepare_minimax_h3_long_action_windows.py`. Use 24 FPS, 240 total frames, 124 frames per window, and 8 overlap frames unless a new tested contract justifies different geometry. Require the generated manifest to prove exact 10-second coverage and explicit cross-window state.

5. Train or select a repair-routing module only against metrics and labels valid for the current object/task domain. Use held-action folds and hard capability non-regression. Never claim that a flower-trained policy is bottle/Ego-trained; if Ego supervision is insufficient, retain deterministic bounded repair and report the routing module as `NOT STARTED` or `PARTIAL` rather than transferring an invalid checkpoint.

6. Run the first matched action window with `scripts/run_minimax_h3_action_variants.py` in `control_video` mode. Let the runner inspect physical GPUs, select or validate one, set `CUDA_VISIBLE_DEVICES`, and record the selection. Do not bypass model, checkpoint, revision, or memory preflight. Treat H3 output as an action/geometry driver until it passes the final visual gates.

7. Extract each action's frame 116 with `scripts/prepare_h3_action_continuations.py`. Run the second matched window with `--continuation-reference-root`. Never pass one action's continuation frame to another action. When `control_video` is also enabled, verify that the actual model reference sequence contains robot image, scene image, continuation image, and control video; the control must append to, never replace, the recursive continuation.

8. For visual recovery, build the raw 240-frame drivers with `scripts/build_h3_raw_action_drivers.py`, then refine each action jointly with `scripts/run_wan_robot_factored_refinement.py`. Use the action driver for robot pose and subject support and the real Ego clip only for clear background pixels. Default to `--source-mask-mode factored_guard`: it skips the fragile human-mask preprocessing path and combines the driver mask with a fail-closed lower-frame human-risk guard. Before spending full generation compute, render a ten-frame contact sheet of `hybrid-preprocess/src_bg.mp4`; reject fragmented, high-frequency, or isolated black support shapes. Empty-mask fallbacks must use smooth pose capsules plus task-object support, not raw scene-difference texture. Suppress source face control. Preserve the raw Wan candidate; do not alpha-composite the source person or run temporal smoothing afterward.

9. Evaluate dense frames from the raw driver, refined candidate, real source, and accepted cabbage reference. Never use the flower validation script to score an Ego bottle result. Reject on any visible human skin, hand, sleeve, translucent source-person residual, or conditioning-shaped black artifact; reject if foreground detail is visibly blurred or background sharpness materially regresses. Also score action motion, object lock, robot identity, temporal consistency, background, action distinctness, and holder-transition causality. Verify the declared terminal holder directly in the last phase; visual cleanliness cannot promote a clip whose object remains grasped when the instruction requires support and release. A decodable clip or a favorable average metric cannot override either human-removal, sharpness, or task-terminal rejection.

10. Package the three individual 10-second actions, a labeled source/old/new/action comparison, poster, manifest, exact geometry, and hashes. Use a hard seam chosen inside declared overlap for H3-only fallbacks; never cross-dissolve robot identities. Prefer one stateful Wan replacement pass over repairing two separately composited final clips.

11. Append every concluded attempt to `experiences/ledger.jsonl` through `scripts/experience_ledger.py`. Update `docs/STATUS.md` with only `WORKING`, `PARTIAL`, `NOT STARTED`, or `BLOCKED` claims supported by the actual acceptance run.

## Acceptance boundary

Report `WORKING` only when all window gates pass, all outputs decode to exactly 240 frames at 24 FPS, action variants are measurably distinct, and dense human review finds zero source-person residual and no material blur regression against the accepted cabbage route. Any hand ghost, sleeve ghost, translucent human remnant, or destructive smoothing is a hard rejection. Otherwise deliver useful artifacts as `PARTIAL` and name failed gates.

Run the domain adapter's focused unit tests, the repository's interpreter-explicit test suite, skill-creator `quick_validate.py`, and SkillHone `static_check.py`. Fix and rerun any failing validator. Stop only after the video manifests prove exact geometry and the honest status matches the failed or passed gates.

Never describe the result as real-robot execution, calibrated contact physics, a native single-pass 10-second H3 trajectory, official BF16 H3, or PhiZero. The released path uses third-party NF4 H3, 2D camera controls, RGB continuation, and recorded real-world observations.

## Gotchas

- H3 frame counts must satisfy `num_frames = 17n + 5`; 240 frames must therefore be split into legal windows.
- The second window must repeat the absolute action state in its prompt and receive its own prior continuation image.
- A declared continuation path is not proof that H3 received it. Keep the reference-composition regression test and stop at 0/20 if a control-video branch overwrites the continuation image.
- Preserve source objects after robot compositing, but reject repairs that improve object lock by materially regressing action motion or identity.
- Never restore the source background inside an uncertain human/robot support region. Fail closed by expanding generation support; a larger generated region is safer than reintroducing a source hand.
- If official replacement preprocessing returns an empty human mask on any frame, do not patch the pinned dependency in place. Record the failure and switch to the driver-mask plus conservative human-guard route in a new experiment.
- Do not feed line-art controls directly to Wan when SAM2 can lose the subject. First generate a realistic H3 robot driver, then use that driver for Wan subject segmentation and motion support.
- A nonempty fallback mask can still be invalid. Reject appearance-difference masks with scene-texture holes or many small components; their topology can leak into Wan as black blocks even when mean coverage looks reasonable.
- Use recent methods as design constraints, not retrospective training claims: robot-factored rendering for explicit visible geometry, LongVie-style global control normalization and history context, and degradation-aware raw-candidate routing. State clearly when a base model was used without reproducing a paper's training.
- Do not use stock web footage as if it were an Ego dataset. Bind every public-dataset demo to its official license and annotation evidence.
- Do not reuse an experiment directory after any preflight, training, inference, evaluation, or packaging attempt.
