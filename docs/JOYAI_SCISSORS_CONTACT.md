# JoyAI scissors and late-contact pipeline

Status: **PARTIAL** (2026-08-13).  The final label stays PARTIAL because the
image-space contact test is not 3-D contact or force-closure evidence, and the
small scissors handle rings are not independently observable in every native
frame.

## What changed

The full-stream prompt treats the florist scissors as a distinct rigid object,
not a texture on the hand.  `HeldToolContract` pins its source interval
(frames 398–447), holder (`robot_right_hand`), topology, and ten native-resolution
review frames.  Tool review is a human veto and never an automatic promotion.

The JoyAI proposal is one uninterrupted 665-frame causal session.  Five cloned
tail frames complete the model's `1 + 8n` chunk contract and are removed without
interpolation to recover the exact 660-frame, 24 fps, 27.5 s timeline.  A
tool-bearing robot reference is a semantic conditioning anchor only; it is not
accepted as evidence about generated frames.

Late hand–flower repair is audit-driven.  It selects only source-required
contact frames that the candidate misses, expands them into small temporal
intervals, and copies donor pixels only where they add a robot replacement
inside tracked hand support.  Flower pixels are immutable.  If a residual
one-pixel image-space gap remains, `project_missing_contact` grows existing
generated-hand pixels through tracked hand support and stops immediately when
the fixed projected-contact invariant passes.  It cannot cross the protected
flower mask and it carries `physical_evidence = false` semantics.

## Reproducible measurements

- Official JoyAI full-stream proposal: 665 generated frames in 105.692 s,
  **6.292 generated fps** (4.37× real time for a 24 fps video).
- The scissors-conditioned raw candidate reaches 6/11 late projected contacts,
  up from the reported 5/11 baseline but still below the fixed 95% gate.
- Final deterministic union: 660 frames in 12.761 s, **51.720 repair fps**.  It
  changes 25,548 donor pixels across 14 audit-selected frames, never writes a
  protected flower pixel, and needs no synthetic contact-pixel growth.
- The exact final audit reaches **11/11 = 100%** projected-contact recall at or
  after 20 s.  Ten of eleven image gates pass; late hand edge energy remains
  below its anchor-fitted lower bound in 22/180 late frames (12.22%, over the
  allowed 10%).  Color, contact, and topology attacks are detected, while the
  structure-ghost attack is not, so adversarial promotion remains rejected.
- Native scissors review finds a hand-bound black/silver tool trajectory across
  the required interval, but the two handle rings and finger-through-ring state
  are not separable in every sampled frame; this gate remains PARTIAL.

Primary local evidence:

- `outputs/joyai-flower-edit/20260813T105500Z-full665-scissors-contact-seed42-v2`
- `outputs/joyai-flower-edit/20260813T110000Z-full660-scissors-contact-final-v1`
- `outputs/joyai-flower-edit/20260813T105500Z-contact-projected-union-v2`
- `outputs/joyai-flower-edit/20260813T104500Z-contact-projected-audit-v1`
- `outputs/joyai-flower-edit/20260813T113000Z-scissors-contact-projected-union-v2`
- `outputs/joyai-flower-edit/20260813T113500Z-scissors-contact-projected-audit-v2`

## Acceptance boundary

The current 2-D audit establishes only projected adjacency, replacement
coverage, topology proxies, palette/skin leakage limits, and adversarial
sensitivity.  It does not establish metric depth, contact force, collision-free
kinematics, or force closure.  Those require an independent depth/telemetry or
verified physics source and remain outside this visual-data milestone.
