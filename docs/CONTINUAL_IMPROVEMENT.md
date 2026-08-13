# Evidence-backed continual improvement

PhiAgent uses recursive self-improvement only in the engineering sense: observe
measured outcomes, preserve the evidence, propose a bounded change, run held-out
regressions, and keep the change only when the relevant acceptance contract
passes. It does not treat model self-critique, visual appeal, or a rising proxy
score as proof of improvement.

## Current evidence snapshot

The canonical detailed sources remain [`STATUS.md`](STATUS.md),
[`EXPERIMENTS.md`](EXPERIMENTS.md), and [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).
As of 2026-08-10, the complete top-level status inventory is:

| Status | Entries | Meaning |
| --- | ---: | --- |
| WORKING | 30 | The stated, narrowly scoped acceptance evidence exists. |
| PARTIAL | 62 | Some evidence exists, but at least one required gate or scope is missing. |
| BLOCKED | 5 | A named external prerequisite prevents the acceptance test. |
| NOT STARTED | 6 | No claim-eligible acceptance run exists. |

`EXPERIMENTS.md` contains 38 measured-run rows, and `STATUS.md` contains 103
top-level entries in total. The counts are mechanically
checked by `scripts/experience_ledger.py summary`; this document summarizes the
cross-cutting lessons instead of copying or replacing the detailed evidence.

## What has succeeded

- Explicit contracts work: named coordinate frames, validated transforms,
  physical-GPU selection, pinned revisions, immutable run directories, hashes,
  seeds, manifests, and deterministic acceptance rules have produced auditable
  component results.
- Deterministic simulation and bounded repair work on their declared fixtures.
  The tabletop push, joint-limit repair, multi-embodiment smoke path, and control
  rendering have measured physical or structural evidence.
- Matched ablation is valuable. Nine EPL-conditioned versus EPL-masked seeds
  show a repeatable synthetic classification gain; the ledger preserves the
  limitation that this is not simulator or real-robot evidence.
- Masked compositing can preserve unaffected scene pixels exactly. Full-video,
  post-decode audits are stronger evidence than attractive keyframes.
- Preflight checks prevent wasted GPU runs when checkpoints, CUDA extensions,
  memory, or physical GPU selection are wrong.

## What has failed or remained partial

- Visual-transfer candidates often score well on motion or target identity while
  failing object retention, regional temporal stability, contact, or robot
  morphology. No aggregate score may hide a failed required gate.
- Sparse keyframe inspection missed long-video smearing, remnants, deformations,
  and localized flicker. Whole-video and high-jerk-region evaluation is required.
- Source-object restoration can also restore adjacent human pixels; stronger
  temporal smoothing can remove motion; and inpainting can introduce scars.
  Repairs therefore need matched regression evaluation, not visual intuition.
- A target image with the wrong scene, pose, camera, wrist entry, or embodiment
  can cause duplication or identity collapse. Replacing only the reference image
  does not solve alignment.
- Historical proxy metrics have produced false confidence. The legacy object
  score accepted a visibly dropped object, which motivated stricter instance,
  deformation, coverage, trajectory, and lift checks.
- Exact PhiZero reproduction is still blocked by unreleased implementation,
  tokenizer/decoder weights, adaptation artifacts, and licensed inputs. EPL,
  Wan, and Cosmos are not silently relabelled as the missing PhiZero method.
- No learned component or real robot has yet satisfied the primary project goal.

## Persistent decision history

`experiences/ledger.jsonl` is append-only. Each line records:

- one of `WORKING`, `PARTIAL`, `BLOCKED`, or `NOT STARTED`;
- a scope-limited statement rather than a broad milestone claim;
- evidence references, lessons, limitations, and next actions;
- an optional immutable experiment directory;
- `supersedes` links when new evidence corrects an earlier conclusion.

Do not edit an old conclusion to make history look cleaner. Append a correcting
record and supersede the old ID. `WORKING` and `PARTIAL` require evidence;
`PARTIAL` and `BLOCKED` require explicit limitations; every incomplete state
requires a next action.

Validate and summarize the full history with:

```bash
python3 scripts/experience_ledger.py validate
python3 scripts/experience_ledger.py summary
```

To add a future result, first create a single JSON object that follows the
schema exercised in `tests/test_experience_learning.py`, then run:

```bash
python3 scripts/experience_ledger.py add --record /path/to/new-record.json
```

## Optimization gate

Every improvement cycle follows the same order:

1. Select one failed or partial acceptance criterion and its immutable evidence.
2. Form one falsifiable hypothesis and name the files or parameters allowed to
   change.
3. Preserve a baseline and keep evaluation inputs separate from optimization
   instructions.
4. Create a new experiment directory and save configuration, command, Git state,
   host, packages, seed, GPU inventory and selection, logs, outputs, and failures.
5. Run the targeted acceptance test plus held-out regressions. GPU/model tests
   must fail with a useful prerequisite diagnosis when unavailable.
6. Append the outcome even when it fails. Keep a change only when every required
   gate passes; otherwise revert the candidate or retain it as explicitly
   labelled negative evidence.
7. Update `STATUS.md` only to the evidence-supported scope. A SkillHone
   optimization run is `PARTIAL` until it has run on real PhiAgent inputs and its
   held-out regression passes.

SkillHone is installed outside the repository as an optional orchestration
adapter. It must remain isolated from `phiagent` imports, keep its evaluation
split private from the optimizer, and never merge a proposed skill change merely
because its training/probe score rose.
