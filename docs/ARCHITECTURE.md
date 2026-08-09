# Architecture

## PhiZero reproduction mainline

The target is Figure 8(b), Human Hand to Sharpa Dexterous Hand Transfer. It uses
the paper's learned video-tokenization route, not explicit hand tracking,
kinematic retargeting, or a two-arm object handover:

    HumanHandVideo
      -> adapted PhiZero spatiotemporal encoder
      -> transition-level Q-Former, 32 queries per adjacent latent pair
      -> FSQ(8, 5, 5, 5, 5, 5)
      -> unchanged 25K-vocabulary PhysicalLanguageTokenSequence

    first(HumanHandVideo)
      -> first-frame appearance edit: human hand to Sharpa dexterous hand
      -> TargetFirstFrame

    PhysicalLanguageTokenSequence + TargetFirstFrame
      -> PhiZero physical-language-conditioned Wan2.2-5B diffusion decoder
      -> TransferredSharpaVideo

For the paper's standard 33-frame representation, nine latent temporal states
produce `(9 - 1) * 32 = 256` discrete symbols. The Physical Language Reasoner is
not required for this transfer demo because the tokens are encoded from the
source video rather than predicted from text.

`PhysicalLanguageTokenSequence` and PhiAgent's existing EPL are different.
PhiZero tokens are learned FSQ indices and are not directly grounded in
interpretable physical variables. EPL remains an auxiliary robotics
representation and must not be reported as a reproduction of the tokenizer.

## Auxiliary PhiAgent pipelines

The explicit robotics route remains useful for downstream analysis:

    HumanVideo -> perception -> EPL -> retargeting -> simulation/repair
      -> trajectory-conditioned Cosmos3-Nano rendering

The trajectory-rendering contract binds accepted robot/object trajectories,
camera calibration, scene assets, control video, prompt, and verification
evidence. This is a separate method and cannot satisfy Figure 8(b).
The MuJoCo control producer resamples the verified joint path at video FPS,
reruns verification, exports measured object poses in a named robot-base frame,
and hashes the complete bundle. Cosmos output is compared frame-by-frame with
the control using edge SSIM; this is a diagnostic and never substitutes for the
pending pose-level alignment gate.

Native Wan2.2-Animate is also an independent motion-transfer baseline. It does
not implement PhiZero's transition Q-Former, FSQ bottleneck, source-domain
adaptation, or physical-language-conditioned Wan2.2-5B decoder.

The agentic proxy wraps released visual-transfer backends in a bounded loop:

    source + Sharpa first-frame candidates
      -> multi-backend/seed candidate generation
      -> local evaluator: motion, identity, object, temporal scores
      -> accept all thresholds or feed diagnoses to the repair agent
      -> revised prompt/seed and another bounded round

All traces use the method label `agentic_proxy_not_official_phizero`. This path
targets a similar visual result without claiming the unreleased representation.

Wan-Animate-2 is an additional direct-driving proxy:

    human driving clip + pose-matched clean Sharpa reference
      -> two-GPU Wan-Animate-2 distilled inference
      -> local evaluation and bounded temporal repair

It removes the original Animate pose/face-control preprocessing dependency. A
pose-matched case-1 run visibly replaces the human hand with one Sharpa hand, but
object and temporal gates remain failed. This is not learned physical language or
unchanged-token decoding.

For long-duration and low-latency output, a deterministic hybrid baseline avoids
full-frame generation:

    source video + separately rendered robot layer + source object track
      -> object-relative screen-space robot placement
      -> robot-only alpha composition over unchanged source pixels
      -> exact source-object pixel restoration
      -> optional downstream robot-region enhancement

The current baseline names both source and robot image-pixel frames and does not
infer a camera/world transform. It is intentionally PARTIAL: object-relative
placement is not a calibrated 3D wrist pose, exposed human-hand pixels are not
yet removed, and no generative region enhancement has run. This path targets
stable long-video engineering rather than exact PhiZero reproduction.

ArtiCraft remains an independent asset-generation research path:

    text description / reference image
      -> pinned mini-ArtiCraft subprocess
      -> isolated USDZ candidate + upstream run evidence
      -> explicit target-simulator conversion and physical validation
      -> calibrated simulation asset

Generated assets do not bypass simulation gates, but they are not inputs
required by the paper's Sharpa video-transfer protocol.

## PhiZero acceptance gates

1. Pin the paper, project-page revision, official code/model revisions, and all
   three public `hand2dex` source/transferred reference pairs.
2. Run the released tokenizer and decoder rather than approximating their outputs
   with EPL, Wan2.2-Animate, Cosmos, or kinematic retargeting.
3. Confirm source-domain adaptation uses HRDexDB human videos without paired
   robot data or cross-embodiment correspondence.
4. Reuse the source physical-language sequence unchanged and change only the
   first-frame embodiment condition.
5. Persist source video, edited first frame, tokens, configs, model/source
   revisions, seed, logs, and transferred video.
6. Evaluate motion preservation, Sharpa identity/geometry, object interaction,
   and temporal consistency against the three pinned 3-second, 896x512 pairs.

## Coordinate and unit contract

Every 3D point carries a `FrameRef`; every pose is named `target_T_source`.
Transforms reject mismatched composition, and projection only accepts camera
frames. EPL v0.1 uses metres, seconds, and XYZW unit quaternions. Robot joint
ordering is name-based and checked against declared limits. The current robot
trajectory schema supports revolute joints in radians; arbitrary prismatic URDF
support is not claimed.

HaMeR outputs are converted from root-relative MANO/OpenPose-ordered joints plus
the official full-frame camera translation into metric camera-frame keypoints.
FoundationPose text matrices are interpreted exactly as `camera_T_object`.
Absolute dex-retargeting POSITION mode is rejected until a named
camera-to-robot-base calibration transform is supplied; vector modes are
translation invariant.

## Simulation acceptance

`SimulationRequest` distinguishes required contact pairs from forbidden contact
pairs and can specify terminal object-position goals. MuJoCo returns all contacts
but counts a collision only when its geom pair is explicitly forbidden. Physical
validity, task success, collision, contact, reachability, joint limits, object
poses, and rollout paths remain separate fields.

The agent loop invokes explicit simulate, collision, contact, and reachability
checks. Its current safe repair is declared joint-limit clamping; unsupported
collision/contact repairs fail rather than fabricate success. These checks apply
to the auxiliary robotics pipeline and are not claimed as the PhiZero method.
