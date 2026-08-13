# Latest open models for long-horizon flower replacement

Evidence cutoff: **2026-08-13**. Overall status: **PARTIAL** for the real
Pexels clip.

## One-sentence thesis

Existing human-to-robot video generators fail on flower arranging because
monocular pixels do not contain metric scale, hidden material identity, exact
robot topology, or measured contact force; PhiAgent should instead compile
public geometry models into one persistent multi-stem/robot/contact state and
allow a video model to change only a masked appearance residual.

## First-principles information boundary

The 27.5-second RGB clip directly observes color, time, visible boundaries, and
2-D motion. It can support proposals for relative depth, camera motion, human
pose, hand shape, point tracks, and contact phase. It cannot independently
recover:

- an absolute metre without a registered measurement or calibrated baseline;
- geometry and background hidden in every frame;
- unique stem identity through a fully ambiguous occlusion;
- the 73 unobserved G1 plus bilateral-Sharpa coordinates;
- measured force, friction, stiffness, or support reaction;
- exact robot topology from diffusion pixels.

Therefore learned metric depth, point maps, and inverse force remain proposals
or model-conditioned explanations until independent evidence binds them.

## Verified public methods

`NOW` means public code and required weights can process custom inputs.
`GATED` means the method is relevant but lacks required sensing, training, or
released artifacts. `UNUSABLE` means no executable public method exists for this
task.

| Layer | Method | Status | Public artifact and exact boundary |
| --- | --- | --- | --- |
| Camera | [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4) | NOW/proposal | Custom video depth/camera; Apache code, larger weights CC-BY-NC; no independent scale |
| Camera | [MoGe-2](https://github.com/microsoft/MoGe/tree/925b8ed835a7a9cdb7578ba15c658a0afc969030) | NOW/proposal | MIT metric-depth prior; learned scale is not calibration |
| Camera | [MegaSaM](https://github.com/mega-sam/mega-sam/tree/a27b4e633c5cc0828a62ed943ef9f6505705fd3f) | NOW/proposal | Apache dynamic monocular reconstruction; no measured metre |
| 4-D | [CARI4D](https://github.com/NVlabs/CARI4D/tree/f359eebbd3d96c68fc7130b08744b46f9e8ebf68) | NOW/limited | Released custom-video path and checkpoint; rigid object pose and binary hand contact; explicitly weak for partial bodies/long occlusion; NVIDIA research license |
| 4-D | [V-DPM](https://github.com/eldar/vdpm/tree/5e2a57cf6007dfb0511a8b396a0805089b9edcc4) | NOW/proposal | MIT code, CC-BY-NC/VGGT weights; dynamic point maps and camera; relative scale, no stem semantics |
| 4-D | [SpatialTrackerV2](https://github.com/henry123-boy/SpaTrackerV2/tree/7e12274c52077860cebfe007a6290777db43b63c) | NOW/proposal | RGB or calibrated RGB-D 3-D tracks; public weights; CC-BY-NC code; finite windows and no stem IDs |
| 4-D | [LongDPM](https://arxiv.org/abs/2605.17303) | GATED | Correct long-window association idea; no code or weights found |
| Hand/object | [VideoManip](https://github.com/hychen-naza/VideoManip/tree/9d0f286af35d73f4252d115d968e0a9c06542b9c) | NOW/partial | Runnable custom monocular hand plus rigid-object reconstruction; withheld policy stages, no Sharpa, no root license |
| Robot | [GMR](https://github.com/YanjieZe/GMR/tree/bb1bbe40774794fceb2a7c579a3464a28e68c844) | NOW/scaffold | MIT GVHMR-to-G1 retargeting; no object/contact model |
| Robot hand | [dex-retargeting](https://github.com/dexsuite/dex-retargeting/tree/3f56141bc8bd2760d5e452e382937269554ebb21) | NOW/scaffold | MIT exact custom-hand URDF optimization; contact must be added |
| DLO | [TrackDLO](https://github.com/RMDLO/trackdlo/tree/c05921e0c3b83a52fe97a6e35f1dd032648df31a) / [MultiDLO](https://github.com/RMDLO/multidlo/tree/e0b7fa35739731a96ac7569952c00414ca2ad968) | NOW with RGB-D | MIT topology-preserving single/multiple DLO tracking; cannot metrically consume the existing RGB-only clip |
| DLO physics | [DEFT](https://github.com/roahmlab/DEFT/tree/5781c70c7737fb84b8bd43261e3ed00ef2fd0fbc) | GATED/legal | Released differentiable branched rods, data and checkpoints; trained from motion capture; no repository license found |
| Deformable twin | [PhysTwin](https://github.com/Jianghanxiao/PhysTwin/tree/54106c6357e369955bc21ea77f012fbd5867165c) | GATED/sensing | MIT code/data; custom inputs require RGB, depth, calibration and metadata |
| Deformable twin | [DeformMaster](https://github.com/CAN-Lee/DeformMaster/tree/c7b3510a38b3fccbfe12cc6557aaf58d9ea823dc) | GATED/training | Structured MPM, residual, compliant hand and Gaussian appearance; custom preprocessing exists, full training absent; monocular scale remains learned |
| Contact | [C2Dex](https://github.com/K-Jie/C2Dex_code/tree/eae9248accaedd7a61dccad562f8048bb9e6c36f) | UNUSABLE | Closest canonical-contact idea; README/figures only, no code/weights/license; rigid-object assumption |
| Contact | [ContactFlow](https://github.com/rpl-bonn/contactflow) | GATED | Strong embodiment-independent 3-D contact trajectory representation; implementation/weights forthcoming |
| Visual | [Wan2.2-Animate](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc) | NOW/residual only | Apache code/weights; short single-person windows, proportion and background failures; cannot own topology/contact |
| Visual | [VACE](https://github.com/ali-vilab/VACE) | NOW/rejected | Public models; current target-independent PhiAgent adapter fails semantic human review |
| Visual | [X-Humanoid](https://arxiv.org/abs/2512.04537) | UNUSABLE | Closest paired full-body replacement precedent; no public code, weights, or paired data found |
| Visual | [H2R-Grounder](https://github.com/showlab/H2R-Grounder) | UNUSABLE | Code/models forthcoming; project reports thin-object grasping as a bad case |
| Visual | [HandEdit](https://arxiv.org/abs/2608.12122) | Evaluator only | URDF-conditioned image benchmark/LoRAs; framewise, egocentric, non-Sharpa, restrictive dataset terms |

Paper-only or unavailable methods are not dependencies: CHOIR, ObjRetarget,
HOWTransfer, Human2Humanoid (2026), PhysHanDI, PGRD, BendTwin, LongDPM, and
ContactFlow remain architectural references.

## Closest-prior novelty matrix

| Axis | C2Dex | VideoManip | PhysHanDI | DeformMaster | X-Humanoid | PhiAgent target |
| --- | --- | --- | --- | --- | --- | --- |
| Optimization object | Rigid canonical contact | Hand + rigid mesh/pose | Hand + one deformable object | MPM/Gaussian twin | Generated human replacement | One 660-frame typed robot/multi-stem/contact state |
| Supervision | Demonstration + RL | Monocular RGB + APIs | Calibrated sparse RGB-D | Monocular pseudo-depth or captured data | 17+ h paired synthetic video | Independent calibration when metric; otherwise explicit relative ambiguity |
| Exact G1 + Sharpa | No | No | No | No | Fixed Optimus | Hash-bound 29+22+22 assets |
| Persistent material IDs | No | Rigid object | One object | Physics particles | No | One ID and arc coordinate per stem/cut child |
| Force provenance | Contact policy | Contact proposal | Model-conditioned | Simulator | None | Sensor or coupled rod/pad residual with covariance |
| Long source lock | No | No | No | Object renderer | Generative | Source pixels immutable outside edit support |
| Public end-to-end | No | Partial | No | Partial | No | Partial until real metric observations exist |

The defensible novelty is not a new depth model or generator. It is the
fail-closed state object joining exact target topology, persistent multi-rod
material coordinates, explicit contact/force provenance, source-locked
compositing, and immutable 660-frame lineage.

## Selected architecture

1. **Camera proposals:** run DA3 and MoGe-2; use MegaSaM as a dynamic-camera
   critic. Fit scale/shift only to registered RGB-D, fiducials, calibrated
   stereo, or another independent known-length observation.
2. **Track proposals:** run V-DPM and SpatialTrackerV2 independently. Prompt
   immutable stem/head instances externally; neither model owns semantics.
3. **Persistent multi-rod state:** jointly optimize every frame and stem with
   fixed material IDs, arc length, root/free modes, topology-event lineage,
   temporal dynamics, ID-swap audit, and covariance over complete occlusion.
4. **Human/contact initialization:** use VideoManip/CARI4D only for visible hand
   and phase initialization. Express contact in stem material arc coordinates,
   not a rigid canonical object frame.
5. **Exact retarget:** initialize body/arms with GMR and optimize the complete
   G1 plus bilateral-Sharpa state with exact assets, limits, balance,
   self-collision and contact. Diffusion never owns geometry.
6. **Force:** prefer tactile or wrist force/torque. Otherwise solve one coupled
   pad-friction/rod inverse-dynamics system and propagate covariance.
7. **Rendering:** render exact geometry/depth/normals/flow/link and stem IDs;
   restore source-evidenced flowers/background; use Wan only for a bounded
   masked relighting residual.
8. **Long horizon:** one absolute state over frames 0--659. Overlapping neural
   windows may change RGB residuals but cannot become physical memory or change
   IDs.

## New executable state layer

`phiagent/perception/flower_track_routing.py` pins the verified public tracker
revisions and prevents monocular proposals from opening a metric gate.

`phiagent/perception/multistem_rod_optimizer.py` jointly optimizes persistent
multi-stem centerlines, fixed/free roots, arc lengths, temporal state,
occlusion covariance, and exact small-set identity assignment audits.

`scripts/optimize_multistem_flower_state.py` validates proposal/source/frame
lineage and requires a matching calibration hash before emitting metric state.

The real 17-frame `active-pink-stem-01` compatibility run at
`outputs/foundation-contact/20260813T030500Z-multistem-active-pink-v3`
demonstrates why the old observation must be replaced:

- exact material projection lowers temporal segment CV from 1.857 to
  `1.26e-14`;
- the required observation displacement has p95 `1.5965` times the full stem
  length, far above the frozen 0.10 bound;
- identity continuity passes for the one available stem;
- no independent scale is present;
- the correct result remains `PARTIAL`.

This is decision-relevant negative evidence: harder optimization cannot repair a
mask skeleton that crosses flower-head/background depth discontinuities.
V-DPM/SpatialTracker point identities are the next required input.

## Decisive experiment stack

1. V-DPM versus SpatialTrackerV2 versus both, with the same frames, masks,
   calibration proposal, optimizer and compute budget.
2. Per-frame skeleton lifting versus persistent point identities plus hard rod
   constraints.
3. Single best proposal versus retained occlusion hypotheses, with an
   identity-swap attack.
4. Relative monocular result versus the same tracks aligned to independent
   RGB-D/stereo anchors.
5. GMR position-only retarget versus full G1/Sharpa contact optimization.
6. Uncoupled visual contact versus coupled pad/rod wrench residual.
7. Exact renderer only versus bounded Wan relighting residual; geometry and
   source pixels frozen.
8. Multiple scenes, bouquets, cameras and material parameters; scene is the
   independent unit.

## Immutable acceptance gates

- source/model/license hashes and 660-frame lineage all bind;
- metric camera: at least 20 registered measurements from at least two
  independent groups, held-group p95 error at most 6%, scale SD at most 2%;
- robot: all 73 coordinates, limits/velocity, reprojection at most 8 px and
  wrong-asset rejection;
- stems: zero ID swaps, every manipulated stem, segment CV at most 0.12,
  normalized observation residual p95 at most 0.10, explicit cut lineage;
- contact: named modes, gap at most 3 mm, bounded penetration, no post-release
  force;
- force: rank six where required, coupled residual p95 at most 0.08 N, finite
  covariance, response on every driven frame;
- unchanged source pixels bit-identical, flower IoU at least 0.95 and accepted
  depth ordering;
- exactly 660 decoded frames at 24 FPS, no identity/color drift;
- any independent human semantic veto rejects.

## Honest completion boundary

A stronger Pexels-only **visual** result is feasible now, but it must remain
non-calibrated, force-unverified and ambiguity-bearing. A real physical
`WORKING` claim still requires a new calibrated observation of the original
scene. No paper or open model found at this cutoff can create that missing
information from Pexels RGB alone.
