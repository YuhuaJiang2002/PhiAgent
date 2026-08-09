# Experiment discipline and measured runs

Every production run uses a unique directory and refuses to overwrite it.
Cosmos runs save the exact robot/object trajectories, camera, verification
record, scene-asset hashes, control video and hash, config, command,
source/model revisions, Git state, hostname, packages, GPU inventory/processes,
selected physical GPU, seed, logs, failures, and output. Wan baseline runs
preserve equivalent runtime provenance. `human_to_sim.py` saves EPL,
trajectory, retarget diagnostics, simulation result, optional videos, and a
manifest.

Measured on `a800-1` on 2026-08-08:

| Run | Evidence | Result |
| --- | --- | --- |
| `simulation/push-measure` | `tabletop_push.xml` + fixed trajectory | 1000 steps, 133 contacts, 0 forbidden collisions, 0.254 m object displacement |
| `e2e/human-to-sim-20260808-01` | explicitly synthetic 60-frame teacher fixture | EPL phases `approach,approach,approach,manipulate`; task success; 3.7 mm object-goal error |
| `repair/demo` | deliberately over-limit push trajectory | 10 tool actions; repaired and accepted; 2 JSONL negative examples |
| `sharpa-wave-sim-20260808-02` | explicitly synthetic 21-point right-hand fixture; official Sharpa Wave MJCF at `6eea427e` | 22-DOF bounded trajectory; accepted MuJoCo 3.3.7 rendered rollout on physical A800 GPU 4 |
| `e2e/multi-embodiment-20260808-01` | one EPL, two linear embodiment configs | masks `[1,0]` and `[1,1]`; both physically replayed |
| `phizero-agentic-proxy/20260808T092301Z-d0d14730` | official hand2dex case 1 source/reference + Wan proxy, seed 42 | 89-frame baseline visibly flickered; evaluator v2 measured temporal 0.694. Temporal-only denoising raised it to 0.753 with identity 0.942 and object proxy 0.847; still rejected because motion 0.625 < 0.75 |
| `phizero-agentic-corrections/case1-sam2-inpaint-trajectory` | official hand2dex case 1; rejected replacement candidate; SAM2 source/candidate object masks | old tool removed, one source-trajectory tool restored and visibly lifted; keyframes pass manual object-count/action review, but overall candidate remains rejected for temporal inconsistency |
| `phizero-agentic-corrections/case1-ghost-temporal2-k9-r5` | case-1 SAM2 repair; swept candidate-object mask over +/-2 frames; 9x9 dilation; Telea radius 5 | selected among four bounded ghost-removal settings; temporal proxy 0.371 vs 0.313 baseline, motion 0.839, identity 0.972; still below final temporal gate |
| `phizero-agentic-corrections/case1-edge-hard-pngmask-yuv444-crf12` | selected swept-mask repair rebuilt from lossless SAM2 PNG masks | binary source alpha has zero support outside the object; exterior-ring leakage 5.84 to 0 before encode; High 4:4:4 CRF 12 avoids 4:2:0 chroma bleed |
| `phizero-agentic-corrections/case1-confidence-routed-raw` | case-1 raw replacement candidate with SAM2 source/candidate tracks | candidate mask present throughout, area ratio 1.53, lift recall 1.0; destructive inpainting disabled to remove apex torso scars; trajectory similarity 0.492, so still not accepted |
| `robot-model-contrast/sharpa-vs-allegro-20260809` | matched case-1 Wan settings; Sharpa versus pinned Wonik Allegro first-frame model | Allegro motion 0.812 and identity 0.973, but object/temporal gates failed; apex has duplicate hands and lost contact; retained as explicitly labelled failure contrast |
| `robot-model-contrast/sharpa-vs-unitree-full-20260809` | matched case-1 Wan settings; full-body pinned Unitree G1 reference, with and without source tool | pure G1 motion 0.138; G1+tool 0.180; detached limbs and multi-object trails; proves full identity replacement alone does not solve initial-pose mismatch |
| `robot-model-contrast/sharpa-vs-sunday-memo-20260809` | exact Sharpa replacement-mode settings; only target reference changed to official Sunday Memo hero frame | Memo identity and lift-return arm motion are coherent; turquoise tool becomes a white box, so object gate fails; internal comparison only |
| `robot-model-contrast/sunday-memo-source-scene-v2` | identity-stable Memo raw output; SAM2 source-human, Memo, and tool masks | replacement limited to 27.99% mean frame area; source table/background retained and tool restored; Memo hands remain wrong, so result is partial and not accepted |
| `robot-model-contrast/sharpa-vs-sunday-wan2-deghosted-20260809` | Wan-Animate-2 distilled, two A800 GPUs, isolated Memo reference; source-table blend | hand trails removed and grippers crisp; source table below row 375 has pre-encode MAE 0; Memo identity drifts and tool contact is misaligned, so PARTIAL |
| `phizero-singleclip97/20260808T120641Z-30f2a572` | case 1 Wan Animate, one 97-frame diffusion clip | motion 0.829 and identity 0.965; rejected on object color 0.719 and regional temporal consistency 0.279 |
| `epl-agent-campaign-v3` + `distributed-epl-campaign-v4` | matched EPL-conditioned vs EPL-masked repair-action MLP, seeds 42-50 across `a800-2`, `a800-3`, `a800-4`, and `zhaoli` | EPL 1.000 mean test accuracy vs masked 0.8906; matched gain 0.1094 +/- 0.0039 population SD on deterministic synthetic examples |
| `epl-agent-video/20260808T162447Z-285b7f5d` | seed-42 matched policy checkpoints, same 3,000 held-out examples | 14.17-second side-by-side classification video; 89.3% without EPL vs 100.0% with EPL; eight masked failures corrected by EPL shown |
| `long-human-retarget/20260809T0410Z-allegro-20p7s` | pinned dex-retargeting video and source revision; MediaPipe + Dexpilot Allegro retargeting | one uncut 20.70-second human input; 621/621 frames detected and rendered; synchronized comparison manually reviewed at six times; gesture conversion only, without object manipulation |
| `long-human-composite/20260809T1055Z-allegro-background-locked-v2` | same 621-frame input and Allegro trajectory; lossless source-frame hand-mask replacement | 20.70-second synchronized comparison; post-decode audit found zero RGB channel differences outside the hand replacement mask on every frame; six-time manual review passed |
| `long-human-shadow-arm/20260809T1910Z-shadow-smoothed-stable-scale-v2` | same uncut input; pinned MediaPipe + Dexpilot 24-DOF Shadow Hand; sigma-4 zero-phase joint smoothing; fixed hand scale | 621/621 frames, 20.70 seconds; 8-9 second maximum frame delta reduced 0.4634 to 0.2607; zero post-decode RGB differences outside the replacement mask; 7.5-9.5 second frame review passed |
| `full-robot-flower-demo/20260809T074000Z-pexels5893642/render-v6` | temporal-union full-person screen-space replacement | REJECTED after full-video review: clean-plate smearing, flower-mask remnants, and distorted procedural arm links remained despite sparse keyframe checks |
| `articulated-flower-demo/20260809T120000Z-full660-v3` | same continuous Pexels 5893642 source; MuJoCo G1 arm IK; wrist-attached articulated Sharpa or Allegro models; clean 3D target scene | REJECTED for README visual use despite passing geometry gates: it removes ghosting and mesh distortion but differs too much from the real-scene generated demos |
| `epl-apple-comparison/20260809T0052-case3` | pinned official hand2dex case 3 source/reference + coarse EPL phase annotation | 3-second side-by-side human-hand to dexterous-hand apple-like grasp/lift; official reference, not a new generated result |
| `phiagent-apple-comparison/20260809T0104-case3` | official case-3 human source + PhiAgent Wan replacement output | visually reproduces robot-hand apple grasp; motion 0.854, identity 0.945; rejected on object 0.004 and regional temporal 0.444 |
| `three-hand-apple-comparison/20260809T0315` | human source + PhiAgent Sharpa + PhiAgent Linker L20 | three-column generated comparison; Linker motion 0.668, identity 0.908, object 0.000, temporal 0.002; failed strict gates |
| `three-hand-apple-comparison/20260809T1100-confidence-routed` | human source + PhiAgent Sharpa + confidence-routed Linker replacement raw | learned replacement/SAM2/relighting route; preserves stable raw candidate and avoids destructive repair; visually improved but still partial |
| `vendor-hand-apple/20260809T0355Z` | matched case-3 Wan proxies for Wonik Allegro and Shadow Robot Hand plus pinned Sharpa reference | 89-frame 2x2 comparison; Allegro 0.330/0.653/0.004/0.001 and Shadow 0.401/0.781/0.004/0.001 for motion/identity/object/temporal; both rejected |
| `vendor-hand-overlay/20260809T1120Z` | source, Sharpa reference, and explicitly labelled full-arm Allegro/Shadow screen-space overlays | removes the visible human arm, preserves the original background and tracked apple; Allegro uses a procedural forearm because the pinned model is hand-only; visualization, not inference or physics-valid retargeting |
| `vendor-confidence-routed/20260809T1205Z` | Allegro/Shadow rerun with the Linker confidence-routed recipe: same-scene full arm, replacement, SAM2, relighting LoRA, frame 89, seed 42 | both routes preserve raw candidates; background/arm improve, but vendor hands become Sharpa-like and strict object/temporal gates fail |
| `four-embodiment-apple-comparison/20260809T1635` | preserved human/Sharpa/Linker outputs plus Robotiq 2F-85-conditioned replacement | gripper attempt motion 0.848, identity 0.919, object 0.004, temporal 0.350; failed to preserve two-finger morphology |
| `control/20260808T091723Z-b82a5ee9` | resampled trajectory + MuJoCo control render | 61 frames at 640x480/30 FPS; task success; 131 contacts; 0 forbidden collisions; 4.38 mm goal error; all hashes verified |
| `real-world-demo/20260808T092850Z-3f146ae9` | official real-camera `hand2dex_1` source + same-scene robot target; Wan proxy, seed 42 | 89 frames at 896x512/30 FPS; proxy accepted; mean 0.8994; motion 0.7786; identity 0.9493; object 0.8698; temporal 1.0000 |
| `hand2dex-2-midframe-norelight-proxy/20260808T141126Z-4ac3ef3d` | official real-camera `hand2dex_2`; 1.2-second robot-hand anchor; corrected object ROI; T5 CPU; no relighting; seed 42 | 77 frames at 896x512/30 FPS; rejected; mean 0.7328; motion 0.6971; identity 0.9239; object 0.8574; temporal 0.4527 |
| `hand2dex-2-deghost-final` | best `hand2dex_2` candidate + recorded character/object masks; masked `hqdn3d=7` | motion 0.8745; identity 0.9342; object 0.7698; temporal 0.6295; improved but still rejected on temporal gate |
| `hand2dex-2-deghost-v4` | diagnosed duplicate-object frames; tracked source-object layer reconstruction; robot restoration; three-frame cosine transitions | duplicate container removed at 1.7 s; object 0.8494; identity 0.9227; motion 0.6950; temporal 0.5316; visual-priority PARTIAL |
| `alt-hand-confidence-routed-final` | clean-apex raw candidate + SAM2 hand-instance mask; graphite material-only edit; no object repair/filtering | object present all frames; area ratio 1.613; lift recall 1.0; trajectory similarity 0.9839; no start/middle/end composite ghosting; appearance change only |
| `sharpa-vace-authorized-ablation` | Apache-2.0 Sharpa-only procedural data; 12 train/4 held-out clips; rank-4 VACE-1.3B LoRA; matched 5 epochs x 60 steps, seed 42 | held-out clip 012: geometry control SSIM/edge/PSNR 0.6920/0.8582/29.56 vs neutral 0.6826/0.8399/28.54 and zero-shot 0.4647/0.5102/12.73; synthetic-domain evidence only |

The synthetic fixture verifies software wiring only. It is watermarked “NOT A
MODEL RESULT” and is not evidence for real-video perception quality.

For the PhiZero Figure 8(b) target, one experiment record must bind the human-hand
source clip, edited Sharpa first frame, encoded discrete physical-language
sequence, source-domain adaptation revision, tokenizer/decoder revisions, seed,
logs, and transferred video. The token sequence must be reused unchanged during
transfer. EPL trajectories, simulation rollouts, Cosmos output, and
Wan2.2-Animate output may be reported as auxiliary evidence but cannot satisfy
the PhiZero reproduction claim.

The Cosmos adapter initializes `alignment_report.json` with
`status=not_evaluated` and `accepted=false`, then replaces it with measured
per-frame edge SSIM after successful generation. The report remains
`accepted=false` because structural edge agreement is not pose-level
robot/object alignment. Successful generation is therefore not a successful
visual-binding experiment.
