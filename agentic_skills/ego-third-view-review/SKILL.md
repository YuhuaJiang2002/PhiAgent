---
name: ego-third-view-review
description: Review and repair ego-to-third-person reconstruction, human-arm simulation controls, and DiT videos using source-world geometry and event timing. Use for this pipeline's visual QA or batch repair, not generic video editing or robot-execution certification.
---

# Ego → third-view visual review and bounded repair

Read `examples/ego_to_third_view/AUTOMATION.md` for the active bundle/review schema and
`examples/ego_to_third_view/evidence/v35-v8-lessons.md` when changing geometry, H3
runtime, or timing. Paths are relative to the repository root, not this skill folder.

## First establish what is authoritative

- The original ego observation defines operator location, action sequence and time.
  The third-person camera observes that action; it is **not** the actor's origin.
  Infer the actor's side from source-world evidence, not left/right screen position.
  A long table edge is one example, never a universal default.
- Preserve the accepted EGO/SIM when asked only to repair DiT timing or appearance.
  Do not rerun upstream models merely to export another comparison.
- Separate observed, estimated and imposed geometry. No metric-scale evidence or
  ambiguous actor origin means `needs_input`, not a guessed confident acceptance.
- Each scene supplies its own identities/materials/background/visible object faces.
  Do not inject this example's whiteboards, clock, cylinders, green mat or timestamps.

## Review actual pixels before approving

Read the exact `review_request.json`. Check its bound source/scene/rig/events/video
hashes. Inspect the full chronological contact sheets and full-resolution frames
around every reach, grasp, carry, stop, release and suspected discontinuity.
Do not infer visual success from a report, prompt, first/last frames, or other
clip's acceptance. Use the environment's available image/video viewer. If viewing
is unavailable, leave the review unknown/needs_input.

Check all nine schema fields, with these specific rejection criteria:

- `actor_origin`: arms emerge from the original ego operator's physical location,
  not the observer camera or a newly invented person across the table.
- `limb_lengths`: constant plausible upper arm/forearm lengths, connected shoulder,
  elbow and wrist. An unreachable hand target requires scale/origin/track repair;
  never lengthen limbs, stretch mesh, or reframe to conceal it.
- `torso`: one shared stable torso, smooth action-driven yaw, continuous elbows;
  no full-body bobbing/translation driven by each hand. Do not replace this with
  a completely frozen facing direction. Genuine locomotion needs a separate adapter.
- `contacts`: no air gap, obvious penetration, slipping, or object-independent
  hand motion through a held interval. Silhouette overlap is not 3-D contact proof.
- `identity`: object count, rigid shape and visible side remain consistent; no
  duplicate hands, vanishing objects, changing clock back into a dial, etc.
- `occlusion`: wrists connect naturally and hands/objects/body/table occlude in
  sensible depth order. Do not paint hands on top to bypass bad geometry.
- `camera`: fixed framing and scene continuity; no cuts at arbitrary 4/8-second
  boundaries, zoom/restart, or unstable partition/background seams.
- `timing`: compare SIM and DiT at matching times, including held pauses and the
  final hold/release state. Match event IDs; do not fabricate missing actions.
- `background`: preserve this scene's accepted constraints; do not add furniture,
  equipment or appearance changes that were not requested.

For accept, report the **actually viewed** full frame coverage using
`inspected_frames: {ranges: [[first,last], ...]}` and all nine checks as pass.
For partial coverage or uncertainty, do not invent viewed frames or a pass.

## Route corrections, preserve the evidence

Use the issue codes in `automation/contracts.py`. Geometry/identity/origin errors
go upstream; contact jitter/elbow/torso errors to stabilization; camera/occlusion
to rendering; DiT anatomy/cuts to generation. A repair request must include visual
evidence and the affected interval. Keep old candidates and failed attempts.

Only when the picture/action sequence is acceptable and timing is the defect,
provide reviewed `dit_events` with the same IDs as SIM and request `dit_timing`.
Use the pipeline's monotone PCHIP path; preserve EGO/SIM and duration. Never call
exact interpolation through chosen anchors independent validation. Check held-out
motion progress too, and disclose repeated-frame cadence. A time map cannot fix
wrong poses, missing actions, or reversed event order.

Run `batch.py --execute --resume` only within the user's authorized batch and
repair budget. Stop at `REPAIR_BUDGET_EXHAUSTED`, `NEEDS_INPUT`, or missing resource
authority; report the specific blocker instead of retrying indefinitely.

## Expensive inference and claims

Before H3, require accepted source/SIM timing and geometry, plus the padded
reference-frame check. Soft Ref2VA conditioning can still drift; inspect afterward.
Use the existing same-NUMA GPU selector and resident worker when authorized.
Never waive arm constraints, consumed-output numerical gates, or NUMA isolation
to obtain a faster/cleaner-looking result. Do not stop unrelated GPU work.
Sparse progress logging alone is not evidence of a hung job.

Report measured load, request and denoising times separately. A flag being enabled,
a fake-backend lifecycle test, or a published benchmark on another workload is
not an observed speedup here. Do not claim arbitrary-video generalization from
the whiteboard example or coordinate-invariance unit tests.
