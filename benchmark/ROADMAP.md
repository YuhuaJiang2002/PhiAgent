# PhiAgent-Bench implementation and release path

Evidence date: 2026-09-04. Status terms refer to accepted evidence, not merely
the presence of code.

## What the v0.2 alpha can do now

- Parse versioned suites/submissions and enforce exact case coverage and named
  coordinate frames.
- Score H2R visual transfer, metric geometry, synchronized action, repeated
  simulation, repeated blind real trials, efficiency, and matched-budget policy
  utility without allowing one level to compensate for a failed lower gate.
- Expand a method manifest into deterministic, dependency-ordered jobs; resume
  successful work; hash outputs and logs; detect tampering; and compile selected
  candidates into a submission.
- Turn a complete per-step simulator trace into L4 evidence by deriving IK,
  joint-limit, velocity, forbidden-collision, and singularity rates.
- Produce a randomized ten-trial/three-session real schedule only for L4-passing
  cases, while keeping hardware control disabled in the repository adapter.
- Invoke pinned external RoboWM-Bench and HarnessEval-W checkouts without
  copying their code or silently extending their evidence authority.

These paths are covered by CPU fixtures. They establish the benchmark harness,
not a finished public benchmark result.

## Required before batch scoring is public

1. Freeze a core test set with task-family balance, source licenses, hashes,
   hidden annotations, initial states, calibration, and target embodiments. The
   current public clips are only an L1 pilot because they lack synchronized
   metric/action truth.
2. Implement model-specific workers for PhiAgent, H3, Wan, JoyAI, and LingBot
   against the generic v0.2 job contract. Each worker must record model/checkpoint
   hashes, exact generation parameters, physical GPUs, latency, VRAM, and every
   rejected candidate.
3. Calibrate three independent H2R judges and the robot-specific HarnessEval-W
   skill set on a held-out human-rated set. Report both panels separately until
   inter-rater reliability and score mapping are frozen.
4. Add calibrated L2/L3 cases with camera poses, depth/mesh or landmarks,
   synchronized q/EEF/gripper/contact telemetry, and per-side event timing.
5. Run at least five frozen initial-state simulation episodes per case in the
   same backend revision. Export full traces, contact/task predicates, renders,
   and logs; do not infer L4 from a rendered video.
6. After safety review, connect a site-owned executor to the evidence contract.
   Run at least ten blind trials over three sessions for each eligible
   case/embodiment and retain failures, interventions, timeouts, and safety logs.
7. Freeze leaderboard policies and publish separate visual, action, simulation,
   real, efficiency, and policy-utility tables. A single composite rank may be
   added only after the component tables and hard gates remain visible.

## Practical first release

Use two or three reference embodiments rather than claiming every commercial
robot immediately: dual RM65-B + AG2F90-D as the company anchor, one common 6-DoF
arm, and one 7-DoF arm. Broader URDF coverage belongs in an adapter-qualification
track. An asset advances from source-pinned to kinematic-, simulation-, and
hardware-validated only after the corresponding tests; a downloadable URDF is
not proof that dynamics, collision meshes, controller semantics, or safety
limits are correct.

For the first external release, the defensible stopping point is a frozen L1--L4
leaderboard plus a clearly labelled RM65 real pilot. A multi-vendor L5 ecosystem
requires partner sites, standardized reset fixtures, calibration tooling, and
independent reviewers; it cannot be truthfully simulated by this repository.
