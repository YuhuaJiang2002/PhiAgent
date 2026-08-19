# JoyAI action-intent real-scene demo harness

Evidence date: 2026-08-19. Status: `PARTIAL`.

## Outcome

The source-demonstration harness is implemented and compiled against a real
27.5-second human flower-arrangement input. It converts a proposed typed action
intent into one immutable full-stream JoyAI RV2V request, plans four independent
seeds, and emits hash-bound inverse-action audit templates. A candidate cannot be
selected when any action, source-motion, object-identity, embodiment-identity,
temporal, or native-resolution human gate fails.

The code path is structurally valid, but its hand-authored nine-phase timeline is
not accepted action annotation. A deterministic boundary-frame review found that
frame 0 already shows the florist holding the bouquet, so the frozen
`observe -> approach -> grasp` opening labels are not aligned to the source. The
claimed scissors interval is also not established by the sampled boundary frame.
The v5 compile is retained as a provenance-preserving negative result and must not
be sent to GPU inference as an action-conditioned demo without relabelling.

This remains a useful source-motion control, but it is not the main
counterfactual demo. The main path is the frame-aligned numerical action-carrier
composition in `docs/JOYAI_SC3_ACTION_CARRIER.md`.

```text
real demonstration (motion authority)
  + typed action/object/phase intent
  + target robot reference
  -> one causal JoyAI RV2V proposal per seed
  -> independent video-to-action phase recovery
  -> hard-gated best-of-N visual selection
```

JoyAI is an instruction-guided causal video editor, not a released numerical
action-conditioned forward/inverse dynamics model. The v1 harness therefore
rejects `prompt_only` motion authority. The source demonstration supplies the
actual wrist, object, tool, contact-side occlusion, camera, and timing signal;
text names and constrains those events. This avoids presenting a prompt as if it
were a 7-D/10-D robot action.

## Implemented contracts

`phiagent/world_model/joyai_action_intent.py` provides a standard-library-only
ABI for:

- frame-explicit proposed audit windows with a named camera frame;
- persistent object identities and invariants;
- JoyAI's exact `1 + 8n` causal frame requirement and deterministic tail trim;
- frozen source/reference SHA-256 bindings;
- deterministic prompt and best-of-N command compilation;
- independent, hash-bound inverse-action observations;
- fail-closed candidate selection where a mean score cannot override one failed
  visual gate;
- an unconditional `physical_promotable = false` boundary for this v1 visual
  path.

The frozen diagnostic config is
`configs/joyai/action_intent_flower_sc3_demo_v1.json`. Its 660 deliverable
frames are partitioned into nine contiguous proposed windows. These windows
exercise the ABI but are not source-ground-truth phases. Five cloned tail frames
complete the 665-frame JoyAI protocol input and are excluded from the proposed
timeline.

## Reproduce the local compile

Compilation is CPU-only and refuses to overwrite an experiment directory:

```bash
.venv-wuji/bin/python scripts/run_joyai_action_intent_harness.py compile \
  --config configs/joyai/action_intent_flower_sc3_demo_v1.json \
  --output-dir outputs/joyai-action-intent/<new-run> \
  --server-url ws://127.0.0.1:18080/ws
```

The original human source is 1280x720, 24 FPS, and 660 frames with SHA-256
`f12ef50ca6050df10990cc759dd74ffc772eca8d927c9c4d1845e9fa1aafe065`.
The accepted CPU preparation center-crops it without rescaling and clones five
tail frames into a 1248x720 FFV1 input with SHA-256
`bcea083a6fa4781ca16e0f41a7cd88abc92fcd761c1dce81b74860e5a4eff3d1`.
The retained immutable negative compile is
`outputs/joyai-action-intent/20260819T131000Z-flower-sc3-demo-compile-v5`.
It additionally binds every harness implementation source by SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `26689fa115f7896f23ea79f8429535ee0b0a8266e2b39371e83057d3eec33bbf` |
| `compiled-action-prompt.txt` | `4497a8674f0e2d9db7f63b688f502d8437d0ef9945418ae04396fbf1ac08a6ba` |
| `candidate-plan.json` | `8eb22b9c3c0b8379e39a343927c7b0f956c8f8ede37fa0a7eab8bce12949fbbf` |

The run directory also contains one incomplete audit template for each seed.
Incomplete templates fail closed. Selection re-hashes the absolute candidate
video path and rejects reports whose declared SHA-256 does not match the file.

## Run on the audited GPU host

First start the pinned 0811 JoyAI service with
`scripts/launch_joyai_video_edit_server.py`. That launcher inventories physical
GPUs, validates exactly two requested devices, sets `CUDA_VISIBLE_DEVICES`,
pins the source/checkpoints/runtime, and saves the server manifest. Then create
a fresh inference directory:

```bash
python scripts/run_joyai_action_intent_harness.py run \
  --config configs/joyai/action_intent_flower_sc3_demo_v1.json \
  --output-dir outputs/joyai-action-intent/<new-gpu-run> \
  --server-manifest outputs/joyai-server/<ready-run>/manifest.json \
  --server-url ws://127.0.0.1:18080/ws
```

Do not execute the frozen v5 phase schedule as an accepted action demo. After an
independently relabelled config exists, a one-seed smoke can add
`--candidate-limit 1`. The planned full run uses frozen seeds
17, 42, 73, and 101. Each seed is one uninterrupted 665-frame causal session;
the harness does not restart at phase boundaries or hide seams with
interpolation.

After an independent observer fills the audit reports, select with:

```bash
python scripts/run_joyai_action_intent_harness.py select \
  --config configs/joyai/action_intent_flower_sc3_demo_v1.json \
  --audit-report <seed-17-audit.json> \
  --audit-report <seed-42-audit.json> \
  --output <new-selection-report.json>
```

## Acceptance boundary and next experiment

The CPU contract and real-input compilation are working, but the semantic
timeline failed source-video spot review before GPU execution. No candidate was
generated, and the path remains `PARTIAL`. Relabelling would require a frozen
independent annotation pass before any 0811 candidate is generated. For the
current demo, use the main 81-frame action-carrier path instead.

Even a visually accepted result remains a perceptual real-scene rendering. It
does not establish SC3-Eval equivalence, metric camera calibration, exact robot
joint trajectories, persistent metric stem geometry, physical contact, force,
safe control, or real-robot task success. A later SC3-like physical path must
add a true action-conditioned dynamics adapter and independent scale/contact
evidence rather than weakening these boundaries.
