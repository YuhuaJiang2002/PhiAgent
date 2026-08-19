# JoyAI SC3-inspired action-carrier demo

Evidence date: 2026-08-19. Status: `PARTIAL`.

## One-sentence thesis

Because the released JoyAI editor has no numerical action input, the demo makes a
frame-explicit action-conditioned RGB carrier authoritative for motion, uses JoyAI
only as a photorealistic residual renderer, and selects whole-stream seeds by an
independent inverse visual action check.

## Why this is the main path

SC3-Eval jointly trains forward dynamics, inverse dynamics, and cross-view
inpainting around numerical robot actions. Its training data and checkpoint are not
public. JoyAI-Video-Edit instead accepts a causal RGB stream, text instruction, and
optional reference image. Passing only a language command to JoyAI would therefore
be prompt-conditioned editing, not action-conditioned world modeling.

The implemented harness uses the strongest honest composition available from the
released interfaces:

```text
real first frame + explicit camera-frame action
  -> frozen action model produces an authoritative RGB motion carrier
  -> exact reversible 640x480 -> 1248x720 camera-image transform
  -> JoyAI whole-stream residual rendering, four frozen seeds
  -> inverse visual evaluator recovers the requested object motion
  -> hard gates, human veto, then minimum inverse-action error
```

`phiagent/world_model/joyai_sc3.py` owns this counterfactual action-carrier path.
`phiagent/world_model/joyai_action_intent.py` is retained as an information-rich
source-demonstration control: it compiles typed semantic phases and independent
audit templates for the existing 27.5-second flower video, but does not turn text
into numerical action conditioning. Its current hand-authored phase schedule
failed boundary-frame semantic review and is retained as a negative structural
control, not an accepted action label. The two paths are complementary rather
than interchangeable.

## Frozen first case

The first counterfactual case is the checked-in OSCAR carry-right result:

- action: `demo/showcase/oscar-acwm-carry-right-action.json`;
- carrier: `demo/showcase/oscar-acwm-carry-right.mp4`;
- real first frame:
  `outputs/acwm-oscar-conditions/20260810T131648Z-hand2dex2-v2/input/first-frame.png`;
- config: `configs/joyai/sc3_oscar_carry_right_best_of_4_v1.json`;
- action frame: `camera:oscar_640x480_pixels`;
- source action timeline: 81 frames at 15 FPS;
- model timeline: nearest-timestamp resampling to 130 frames at JoyAI's native
  24 FPS, followed by seven cloned tail frames to satisfy
  137=`1 + 8 * 17`;
- JoyAI seeds: 42, 101, 131, and 181.

The image transform is explicit and reversible:

```text
x_joyai = 1.5 * x_oscar + 144
y_joyai = 1.5 * y_oscar
```

The 640x480 carrier is isotropically resized to 960x720 and centered over a
non-authoritative blurred cover background. This avoids mirror padding that would
duplicate the robot or bowl near an image boundary. Generated frames are
center-cropped and resized back before inverse evaluation. Camera, world, and
robot-base coordinates are never relabelled.

## CPU preparation

The non-CUDA preparation command validates the action/carrier frame and FPS
contracts, creates a lossless native JoyAI input and reference image, freezes the
prompt, and writes all four candidate commands:

```bash
.venv-wuji/bin/python scripts/run_joyai_sc3_harness.py \
  --config configs/joyai/sc3_oscar_carry_right_best_of_4_v1.json \
  --prepare-only
```

Accepted preparation evidence:

- run:
  `outputs/joyai-sc3-action-demo/20260819T050615Z-3d5d8ec5-joyai-sc3`;
- status: `PARTIAL`, stage `joyai_sc3_prepared_not_run`;
- prepared carrier: FFV1, 1248x720, 24 FPS, 137 frames;
- prepared carrier SHA-256:
  `279ef1980418fd07ce598ab8c2886506949785d5a44dfdd9357079b661be827f`;
- aligned reference SHA-256:
  `6118c1a0a898e729fde400d7842ee113b3147debee8e85b9a760102bec98917c`;
- action JSON SHA-256:
  `0bb87748cc2d1e59d8c778c094a3ddaad1f1d1541c441f2fa3adf0860eb16358`.

The first preparation attempt exposed that FFV1 Matroska may omit stream-level
duration. The probe now reads container duration and otherwise derives duration
from counted frames and the exact rational FPS. The failed run remains preserved at
`outputs/joyai-sc3-action-demo/20260819T041203Z-7f635853-joyai-sc3`.
The next numerically valid run used mirrored side padding; full-frame visual
inspection found that it duplicated the terminal robot and bowl outside the
authoritative center support. It remains preserved at
`outputs/joyai-sc3-action-demo/20260819T041344Z-f4b7650e-joyai-sc3` and was
superseded by the blurred-cover policy above. The final config freezes Gaussian
sigma 30, native model FPS 24, and its SHA-256 is
`6165a58dc2b71a420dd417942605e487f5e67a471a53430f48b6ca18a2c9d231`.
An identity-render round trip first trims the seven protocol frames and then
returns to 81 frames at 15 FPS; it measured all-channel SSIM 0.986845 against the
original carrier.

Remote FFmpeg 4.4 initially produced 135 rather than 137 frames because its
overlay filter removed the terminal source-frame duration. That failed attempt is
preserved at
`a800-1:/data0/jiangyuhua/PhiAgent-0/outputs/joyai-sc3-gpu/20260819T045500Z-source-bound-v1/experiments/smoke-seed42/20260819T050258Z-0b2761d3-joyai-sc3`.
The corrected filter adds one source-rate terminal support clone before
nearest-timestamp resampling, adds exactly seven clones afterward, and caps the
output at 137 frames. A fresh remote preflight then passed at
`a800-1:/data0/jiangyuhua/PhiAgent-0/outputs/joyai-sc3-gpu/20260819T050700Z-source-bound-v2/experiments/preflight-exact-frames/20260819T050746Z-3afd6222-joyai-sc3`.

## Run on the audited GPU service

First launch the pinned JoyAI service. The server launcher inventories physical
GPUs, validates exactly two devices and free memory, sets
`CUDA_VISIBLE_DEVICES`, verifies source/checkpoint hashes, and persists the
selection:

```bash
python scripts/launch_joyai_video_edit_server.py \
  --repository <pinned-JoyAI-checkout> \
  --checkpoint-root <pinned-checkpoint-root> \
  --output-dir outputs/joyai-server/<new-run> \
  --python <joyai-venv>/bin/python \
  --physical-gpu 4 \
  --physical-gpu 5
```

While that manifest is `WORKING` at `joyai_server_ready`, run:

```bash
python scripts/run_joyai_sc3_harness.py \
  --config configs/joyai/sc3_oscar_carry_right_best_of_4_v1.json \
  --client-python <joyai-venv>/bin/python \
  --evaluator-python <phiagent-venv>/bin/python \
  --server-manifest outputs/joyai-server/<ready-run>/manifest.json \
  --server-url ws://127.0.0.1:18080/ws
```

The harness refuses inference without a ready server manifest containing the exact
JoyAI source, model, text-encoder, physical-GPU, logical-placement, and
`CUDA_VISIBLE_DEVICES` evidence.

## Measured A800 result

The source-bound run used the pinned JoyAI 0811 service on physical A800 GPUs 4
and 5. The server manifest was `WORKING` at `joyai_server_ready`, with source
revision `3478e4b8c9a79fe935157d1d477cd3e57bb41f1f`, model revision
`e14d9ac50d4ad8e9f91b655bfab270c02a43923b`, and text-encoder revision
`4bfb270765825d2fa059011deb4c96fdd579be6f`.

A one-seed smoke completed 137 generated frames, restored exactly 81 frames at
15 FPS, and passed all automatic gates. The frozen four-seed experiment then
completed without a model or protocol failure:

| Seed | Action | Object | Embodiment | Temporal | Background | Automatic |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0.9438 | 0.6420 | 0.9133 | 1.0000 | 0.8491 | fail |
| 101 | 0.9460 | 0.6543 | 0.9179 | 1.0000 | 0.8485 | fail |
| 131 | 0.9452 | 0.5679 | 0.9233 | 1.0000 | 0.8479 | fail |
| **181** | **0.9327** | **0.8395** | **0.9161** | **1.0000** | **0.8452** | **pass** |

The immutable batch run is
`a800-1:/data0/jiangyuhua/PhiAgent-0/outputs/joyai-sc3-gpu/20260819T050700Z-source-bound-v2/experiments/best-of-four/20260819T051052Z-1f71bd8b-joyai-sc3`.
Its locally mirrored manifest SHA-256 is
`9531a67d5729485b5cfa4f0b3742fd3ff15d28c62e51ca75c4a03342dfa454bc`.
Seed 181 is the only eligible batch realization. Its restored review SHA-256 is
`5d36a9816076fe25d97f578d552d838583f6b5443eaf63139eef226f566dc450`,
and its all-channel SSIM against the authoritative carrier is 0.969928. The four
sessions generated at 7.05--7.20 output FPS.

An independent seed-181 repeat also passed every automatic gate: action 0.9227,
object 0.8272, embodiment 0.9216, temporal 0.9292, and background 0.8420.
However, the two seed-181 outputs have all-channel SSIM 0.941755 despite identical
decoded input, prompt, reference, settings, and seed. Seed 42 also changed its
object score from 0.7654 in the smoke to 0.6420 in the batch. The measured
CUDA-graph deployment is therefore gate-stable for seed 181 in two attempts but
not pixel-deterministic. Every realized candidate must be identified by run path
and SHA-256, not seed alone.

The packaged diagnostic is:

- packaging run:
  `outputs/joyai-sc3-showcase/20260819T063000Z-seed181-confirmed-partial-v3`;
- provenance-complete manifest SHA-256:
  `0b3a2fdcd8db59492a378339273210f8d004b2a2fbfe415f4674b33da58a7189`;
- `demo/showcase/joyai-sc3-action-carrier-partial.mp4`, SHA-256
  `919dc58991218f098608732fbaceb100d24f77de6317ff3758c2a91639d2091a`;
- `demo/showcase/joyai-sc3-action-carrier-partial-poster.jpg`, SHA-256
  `eebffd132d4f5143c99aa030172c0512e84ce1d27e4ede5d75aebe4029f83261`.

Every frame labels the left panel as motion authority and the right panel as the
JoyAI residual render, and states `PARTIAL`, automatic gates 2/2, human review
pending, and not physical execution or contact evidence.

## Same-first-frame user-intent contrast

The carrier-versus-residual view above is useful for auditing what JoyAI changed,
but it is not the primary capability example because the two panels are expected
to be similar. The primary user-facing demo now holds the real first frame fixed
and changes the requested action:

| Intent | Selected seed | Action | Object | Requested terminal |
| --- | ---: | ---: | ---: | --- |
| Lift up and hold | 101 | 0.9934 | 0.9259 | (260.7, 126.6) |
| Carry right and hold | 181 | 0.9327 | 0.8395 | (435.7, 126.6) |

Both action contracts start the yellow bowl at `(262.9, 290.8)` in
`camera:oscar_640x480_pixels`, and both generated runs bind the same real
first-frame SHA-256
`7ea743ac225e7cd3890cc8553d10c203bb88988a213f19309cdbd5e0790b652f`.
The requested terminal targets are exactly 175 pixels apart. The evaluator
observed lift-up at `(256.6, 92.6)` and carry-right at `(449.2, 76.2)`, a
generated endpoint separation of 193.288 pixels.

All four lift-up seeds passed the automatic gates, and seed 101 minimized inverse
action error. The exact lift review SHA-256 is
`d9e251bfad9bc18ad37fbd17d68171e7f2343254f76c0f10ed0aeea06458af66`.
It is paired with the exact carry-right seed-181 review already reported above.

The primary differentiated-intent artifact is:

- `demo/showcase/joyai-sc3-two-intents-partial.mp4`, SHA-256
  `39210fbe0cde11ec1461beb3e90f3a3c4fc3b2daf0cbaf60f19850c4f2f86bc2`;
- `demo/showcase/joyai-sc3-two-intents-partial-poster.jpg`, SHA-256
  `3908f10ceb74af99bdfa4a987fe165aeaf5c89d0efec34950e7f4a002e7c5bbf`;
- provenance-complete manifest SHA-256
  `cfd5e7afd6a9d763766b608cf4d8065f097bb7c506f6bed3f8d2decc2643696c`.

The two panels show generated JoyAI rollouts rather than carrier-versus-output.
Blue and orange overlays expose the requested vertical and rightward object
trajectories and target points. The pixel separation is evidence of image-space
intent differentiation only, not metric robot control.

## Inverse consistency and selection

Each seed is one uninterrupted causal stream. After removing protocol tail clones,
the harness returns from 24 to 15 FPS without motion interpolation, reverses the
camera-image transform, and runs the existing bowl evaluator against the original
action condition. Five automatic gates are required:

1. requested yellow-bowl action adherence;
2. robot embodiment consistency;
3. object presence and non-duplication;
4. temporal consistency;
5. fixed real-background consistency.

Candidates failing any gate are ineligible. A native-resolution human rejection is
also a veto. Among eligible candidates, the winner minimizes
`1 - action_adherence`, then maximizes the minimum gate margin and mean score, then
uses the lower seed as a deterministic tie break. Human review must be explicitly
true before the demo can become `WORKING`.

This mirrors SC3's test-time consistency principle, but its image-space inverse
error is not calibrated to SC3's normalized 7-D action threshold.

## Acceptance boundary

The harness, real-input preparation, JoyAI inference, exact stream restoration,
inverse scoring, best-of-four selection, independent automatic confirmation, and
diagnostic packaging have run. Native-resolution human review is still
`NOT STARTED`, so the end-to-end demo remains `PARTIAL`.

Even a visually accepted candidate establishes only a perceptually plausible
counterfactual in a captured-camera scene. It does not establish:

- equivalence to SC3-Eval or its reported policy-ranking correlation;
- cross-view consistency;
- metric camera or robot-base calibration;
- exact joint or end-effector execution;
- 3-D contact, force closure, collision safety, or real-robot success.

The next acceptance test is independent native-resolution human review of the
packaged MP4. A stronger scientific claim additionally needs held-out action
types, a compute-matched carrier-only control, scene-level uncertainty, and an
independent numerical action observer. Failed seeds and failed runs remain in the
experiment history.

## Public references

- SC3-Eval: <https://arxiv.org/abs/2606.18610>
- JoyAI-Video-Edit:
  <https://github.com/jd-opensource/JoyAI-Video-Edit>
