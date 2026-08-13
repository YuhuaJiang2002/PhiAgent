# H3 robot identity and topology RSI

Evidence date: 2026-08-11. Overall status: **PARTIAL**.

## Failure being corrected

The former H3 review represented robot identity with an anchor-frame masked
grayscale similarity. That proxy can remain high when an arm originates behind
the head, a shoulder-to-hand chain becomes ambiguous, or the head and torso
change proportions. A separate `topology_integrity` field existed in the native
LoRA promotion contract, but no full-frame evidence was required to produce
that value. It was therefore a placeholder rather than an effective gate.

The held-out 124-frame `inspect-flower` baseline demonstrates the gap. Its
identity proxy is approximately 0.71, yet semantic review finds a drifting
raised-arm shoulder origin from frame 24 onward and a broken head-to-torso chain
from frame 42 onward.

## Fail-closed review contract

`compile_robot_topology_review.py` now binds review evidence to the SHA-256 of
one decoded candidate, computes a SHA-256 over the decoded pixels of every
individual frame, and expands non-overlapping reviewed ranges into one record
for every video frame. Promotion fails if even one per-frame digest is absent;
this prevents a label sheet for another decode or an unbound range summary from
being treated as full-frame evidence. Every frame must pass all of these gates
with at least 0.95 confidence:

1. one robot subject;
2. one connected head-to-torso chain;
3. exactly two visible arms;
4. a unique anatomical left-shoulder attachment;
5. a unique anatomical right-shoulder attachment;
6. continuous upper-arm, elbow, forearm, wrist, and hand segments;
7. stable robot proportions;
8. no extra or missing limbs; and
9. no residual human anatomy;
10. exactly one left-shoulder origin;
11. exactly one right-shoulder origin; and
12. both arm roots remain visibly outside the head/neck region.

Coverage and aggregation are strict: every decoded frame must be reviewed, and
one failed gate in one frame rejects the candidate. String values such as
`"false"` are rejected rather than coerced to true. `IdentityPromotionContract`
also checks that the metric presented to RSI exactly matches the score derived
from the bound evidence. Legacy nine-gate evidence remains readable as history
but cannot promote a new candidate without all three shoulder-root details.

The baseline review is under
`outputs/h3-identity-topology-rsi/20260810T211009Z-baseline-review-v1`.
It covers all 124 frames, passes 24, rejects 100, reports 100 failed shoulder
attachments and stable-proportion frames, and reports 82 failed head-to-torso
frames. The annotated video SHA-256 is
`f23f3bc50bd1b3980c9ac42b0cc5aaf2c04ffa0434b6723ffd86ad19376e2ce8`.

## Targeted supervision

`configs/h3_identity_topology_rsi_v2.json` defines six 39-frame, 448x256,
24-FPS positive training clips from two separately rendered articulated robot
subjects. The prompts explicitly cover normal approach/release motion and the
difficult crossed-arm/self-occlusion case. They require unique shoulder origins,
continuous arm chains, stable proportions, and no duplicate limbs. A separate
v14 real-scene rigid clip is validation-only and a separate v19 clip is a
subject-disjoint topology-only test. Those clips remain `PARTIAL` overall and
are not treated as physical flower-manipulation successes.

The compiled remote training dataset is
`/data0/jiangyuhua/PhiAgent-0/outputs/h3-identity-datasets/20260811T052000Z-topology-v2-train`.
Its manifest SHA-256 is
`1118ee94b53e3dac68bf662e5ccf6a4615d1bc905b576f2dac77bcfe351159c6`.

Epoch-0 review exposed a supervision-scale problem: the two synthetic robots
are topologically clean, but their shoulder joints occupy too few pixels in
the full-body training frames. The leakage-safe v3 fallback curriculum under
`configs/h3_identity_topology_closeup_rsi_v3.json` therefore mixes two
full-body clips with eight declared upper-body crops from the same two
synthetic subjects. The crops cover reach, cross-body occlusion, head
clearance, and release, while v14/v19 same-real-scene artifacts remain
validation/test only. The compiled ten-clip train directory is
`/data0/jiangyuhua/PhiAgent-0/outputs/h3-identity-datasets/20260811T074000Z-topology-closeup-v3-train`;
its manifest SHA-256 is
`14f4a55133e41dd7d79d90792a4c23545bf61ea2b998a57e40987843d5f340f0`.
It is staged, not yet evidence of model improvement.

## Bounded RSI results

Round r0 used a rank-8 native Ref2VA LoRA, learning rate `5e-5`, one dataset
repeat, one epoch, and seed 20260811. The dedicated H3 environment completed
6/6 steps and produced a 32,825,120-byte checkpoint with SHA-256
`2a8ee75e76cda3ef610ffcf71eac7c11d2adab621d9d4f38a7920482dd66cce2`.

The frozen held-out inference used the same real scene, robot reference, action
control, seed, dimensions, frame count, and 20 inference steps as the baseline.
The candidate differs from baseline (SSIM 0.984521 and PSNR 40.609 dB) but has
the same semantic topology result: 24/124 passing frames. Motion rises from
0.565746 to 0.567128 and EPL from 0.511592 to 0.513314; temporal consistency
changes from 0.953630 to 0.953424. These remain within the capability
non-regression contract, but topology and identity-gain gates fail. The result
is therefore rejected, not a demo-quality improvement.

A second frozen DINOv2 cross-check is stricter than the original flower-motion
proxy. Mean reference identity changes from 0.907582 to 0.910069, but worst-frame
identity falls from 0.883382 to 0.880120 and matched motion adherence is only
0.938565, below the 0.99 non-regression floor. Scene preservation changes from
0.948382 to 0.948156 and temporal consistency from 0.997785 to 0.997761, both
inside tolerance. DINOv2 is used only for appearance identity; the 0.193548
topology score still comes exclusively from full-frame semantic evidence.

The r0 assessment is under
`outputs/h3-identity-topology-rsi/20260810T212900Z-r0-assessment-v1`. The bounded
search policy therefore selects the lower-learning-rate
`r2-conservative-r16` route. A higher-learning-rate rank-16 route was stopped
after seven steps when the stronger motion result arrived; it emitted no
checkpoint and remains recorded as a controlled failed attempt. No round will
be promoted unless topology is 1.0 over every frame and motion-profile,
task-action/EPL, scene, and temporal metrics remain within their frozen
tolerances.

## Conservative r2 checkpoint result

The conservative rank-16 round uses learning rate `2e-5`, twelve dataset
repeats, two epochs, and seed 20260811. Its first epoch completed 72/72 steps
and produced a 65,626,168-byte checkpoint with SHA-256
`f022c783de9b4d76c9ecfa8d3384cba89a32e67c49d44d5b73a9e0bf89b8bac0`.
The matched epoch-0 candidate contains 124/124 frames and has SHA-256
`3e6b14f29211418214782fd77682f6250448777ed409a38360a7364f6dc3feb7`.

Epoch 0 is rejected. Dense robot crops and boundary review again locate the
first shoulder/proportion failure at frame 24 and the head-to-torso collapse
at frame 42. The digest-bound evidence therefore remains 24/124 passing
frames, with 100 left-shoulder/proportion failures and 82 head-to-torso
failures. Its annotated hard-reject video is
`outputs/h3-identity-topology-rsi/20260811T072500Z-r2-epoch0-review-v1/topology-review.mp4`
with SHA-256
`7d3bce7bdbdc8a5fa6dd2a76b7fc417c16ae5ad6d13dffef86c52c98fc9722da`.

Frozen DINOv2 mean appearance changes from 0.907582 to 0.907675 and the
worst-frame value changes from 0.883382 to 0.886156, but matched motion-profile
adherence is only 0.941179, below the 0.99 non-regression floor. The separate
flower-task proxy changes motion from 0.565746 to 0.564189 and EPL minimum from
0.511592 to 0.509141. Their conservative normalized action adherence is
0.995209, inside its 0.99 floor. Both gates are retained: the task proxy
prevents gross action loss while the stricter matched profile catches temporal
motion changes that its scalar phase score can hide.
The final epoch remains a separate candidate and must pass the same full
contract; epoch-0 evidence cannot be averaged with it.

## Final r2 replication and action-evidence correction

A separate epoch-1 replication with checkpoint SHA-256
`d58651beb2d3c9e42ac0363c3b0d1002e3d4ab312faa10eca512637ec17d72ce`
produced candidate SHA-256
`959ee744a1484334c2dd4f8d19814177bd41f8511c611ad3fc2d9343f1f13823`.
It preserves two visible arms but reproduces the baseline failure pattern:
24/124 frames pass, left-shoulder attachment and proportions fail on 100 frames,
and the head-to-torso chain fails on 82 frames. Worst-frame DINOv2 appearance
falls to 0.800488 and matched pixel-motion adherence is 0.934159.

This run exposed an action-review bug. The previous evaluator accepted an
action JSON without proving that its scorecard described the assessed video,
and reduced motion, EPL, and object lock to one minimum raw value. A shared
near-zero object score could therefore hide motion/EPL loss. The corrected
review now:

1. binds baseline and candidate scorecards to the assessed video SHA-256;
2. requires identical source, motion-control, robot-reference, and mask hashes;
3. compares motion, EPL minimum, and object lock component by component; and
4. uses the worst normalized component ratio as action adherence.

For the second epoch-1 replication, the component ratios are 0.967217 motion,
0.960113 EPL, and 1.0 object lock. Action adherence is therefore 0.960113 and
fails the 0.99 gate. The authoritative frame/action-bound assessment is
`outputs/h3-identity-topology-rsi/20260811T083700Z-r2-epoch1-assessment-v4-full-contract`.
It requires all 124 decoded-frame digests and all twelve semantic gates. The
detailed review counts 100 non-unique left-shoulder origins and 100 arm roots
that do not remain clear of the head/neck region, then rejects the candidate on
identity gain, identity floor, topology, matched motion, and action.

The next bounded candidate was the rank-32 r3 round trained on the ten-clip
close-up curriculum. Epoch 0 completed 120 steps and produced checkpoint
SHA-256
`ce324f0950530e3e007e8a918c2bce15bb1d5484759deb136be936801bef7831`.
Matched inference at LoRA scales 1.0, 0.5, and 0.25 was reviewed independently.
All three candidates reproduce the same 24/124 passing topology result: 100
frames fail left-shoulder attachment, unique left-shoulder origin, head/neck
clearance, and stable proportions; 82 of those also fail the head-to-torso
chain. Their matched motion-profile ratios are 0.912370, 0.908665, and
0.968811, all below the 0.99 floor. The component-wise action ratios are
0.998123, 1.0, and 0.986822 respectively, so the 0.25 candidate additionally
fails the action gate. Mean and worst DINOv2 identity also remain below the
baseline for every scale. Epoch 1 was therefore interrupted after 35/120 steps
under the early-stop rule; no later checkpoint was emitted or claimed.

The nominally identical r2 replications also produced different checkpoint
hashes and failure modes despite sharing a seed. The former launcher seeded
Python, NumPy, Torch, and CUDA but did not require deterministic CUDA kernels.
Future rounds now set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, disable cuDNN
benchmarking, enable deterministic cuDNN and strict Torch deterministic
algorithms, and persist the wrapper hash and determinism settings. The running
r3 process predates that change, so every one of its checkpoints remains an
independent candidate; determinism is not claimed retroactively.

## Real-background r4b curriculum and bounded result

The r3 rejection changed the next round's data contract instead of increasing
LoRA capacity again. Six Pexels-licensed real-scene sources were assigned to
three train scenes, one validation scene, and two test scenes. The sources and
license metadata are frozen under
`outputs/h3-identity-domain-sources/20260811T031200Z-pexels-domain-v4`.
Original people and hands are not topology-positive supervision: each scene
uses a visually reviewed, person-free ROI as a softened background plate, then
adds a deterministic full-body articulated robot with per-frame joint and
topology truth.

The first generated curriculum at
`outputs/h3-identity-domain-curricula/20260811T033000Z-r4-six-domain-topology`
was rejected after its overview exposed a residual human forearm in one train
background. The corrected r4b curriculum narrows that ROI and lives under
`outputs/h3-identity-domain-curricula/20260811T034500Z-r4b-six-domain-topology`.
Its fail-closed domain contract passes: three train identities/scenes and three
held-out identities/scenes are mutually disjoint by subject, scene, and source;
all four high-risk action tags and the full-body, unique-shoulder, head-clearance,
and real-background tags are present. The compiled 12-clip training manifest is
`outputs/h3-identity-datasets/20260811T035000Z-r4b-domain-train/manifest.json`
with SHA-256
`31dd657193736fd78cf86af97de5d359f9d3110c329fe419f1d63c3af6e5d14b`.

The strict deterministic r4b native Ref2VA run used rank 16, learning rate
`1e-5`, three dataset repeats, one epoch, seed 20260811, and physical A800 GPU
0. It completed 36 steps and retained step-12, step-24, and step-36 checkpoints
with SHA-256 values
`8e59e413f5b2d42c2bdac3e036f34774b7b6ff1dced879d3ed06a4114263e91f`,
`fd5bd2cfeade526e0f65fd9edc4fed3e09c98b9efd48ae01d7d63d4b9dc7b884`,
and `464c66259f86a8260dcbe44d05b658aee8d1ee2d82ea43ac9d38758ea4503ac8`.

The step-12 matched held-out result has a small appearance-only signal: DINOv2
mean reference identity changes from 0.907582 to 0.909375 and worst-frame
identity from 0.883382 to 0.889964. This is not an accepted improvement.
Topology remains exactly 24/124 passing frames; 100 frames fail left-shoulder
attachment, unique left root, head/neck clearance, and proportions, and 82 fail
the head-to-torso chain. Matched motion falls from 1.0 to 0.978634 and action
adherence to 0.824577. The separate flower-task scorecard also rejects the
candidate with object lock 0.000419. The corrected full assessment is
`outputs/h3-identity-topology-rsi/20260811T051500Z-r4b-step12-assessment-v2-structural-stop`;
it returns `REQUIRE_STRUCTURAL_BACKBONE` rather than cycling to an exhausted
r0-r3 LoRA setting.

One final bounded check used the step-36 checkpoint at LoRA scale 0.5 because
the step-12 DINOv2 values showed a small appearance signal. Its dense full-video
review reproduces the same 24/124 topology result and failure histogram. The
candidate is about 40.52 dB PSNR from the frozen baseline, indicating a minor
appearance perturbation rather than a structural repair. No r4b checkpoint is
promoted or releaseable, and this qkv/out-projection LoRA route is stopped until
the optimization objective or backbone exposes explicit structural control.

## Fail-closed task-bound delivery

`route_h3_identity_delivery.py` now prevents both a rejected learned candidate
and an unrelated fallback from being surfaced as the workflow result. A
fallback must pass the same complete 12-gate promotion assessment as the
candidate. Both metric bundles must bind the same current source, action-control
video, robot reference, anchor mask, frozen baseline, baseline action
evaluation, identity mask, reference image, and scene image by SHA-256. The
assessments must also bind the exact encoded 124-frame videos and metric files.
Historical booleans or a partial gate list are invalid inputs. If neither branch
passes, the router writes a blocked manifest and no video.

The earlier delivery under
`outputs/h3-identity-rsi/20260811T085300Z-r3-reject-structure-fallback-v2` is
user-rejected and superseded. Its old v19 action booleans described a different
660-frame trajectory and did not establish correspondence to the current
`inspect-flower` control. A post-hoc task-bound audit of the exact delivered
video finds 124/124 visible topology frames but only 0.445897 motion adherence
and 0.278505 action adherence relative to the current baseline. Raw
current-control scores fall from 0.570023 to 0.165170 for motion preservation
and from 0.516750 to 0.143918 for EPL minimum. It therefore fails both
`motion_non_regression` and `action_non_regression`.

The corrected routing record is
`outputs/h3-identity-rsi/20260811T090310Z-task-bound-delivery-block-v2/manifest.json`.
It reports `blocked`, has `output: null`, and contains no `delivered.mp4`. This
is the intended safe outcome until a candidate passes structure, identity,
motion, action, scene, and temporal gates together.

## Meaning of “perfect” in this experiment

“Perfect” is limited to the declared observable acceptance contract: 124/124
frames pass every visible 2-D topology gate, the identity floor improves by at
least 0.02, and motion-profile, task-action/EPL, scene, and temporal
capabilities do not regress beyond their declared tolerances. It does not mean
universal model correctness, hidden
3-D joint validity, force-closure, or real-robot execution. A candidate that
fails the full-video contract is retained as evidence and triggers another
reviewed round or `REGENERATE_WORLD_MODEL_CANDIDATE`; it is never presented as a
successful visual repair.
