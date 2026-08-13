---
name: evolve-demo-video-factory
description: Build, train, validate, and operate PhiAgent's guarded batch demo-video data flywheel. Use when asked to solidify successful agentic video-generation behavior in the environment, collect complete recipe tournaments, train a lightweight recipe router, export planner-distillation preferences, reduce GPU candidate cost, run repeated production batches, or evolve the video factory without weakening action, identity, object, temporal, background, or human-review gates.
---

# Evolve Demo Video Factory

Operate from the PhiAgent repository root. Read `AGENTS.md`,
`docs/DEMO_VIDEO_DATA_FLYWHEEL.md`, `docs/STATUS.md`, and the domain's existing
experiment manifests before changing or running a campaign.

## Workflow

1. Choose one narrow video domain and one immutable acceptance contract. Start
   from `configs/demo_factory/agentic_video_demo_contract_v1.json`, but create a
   separate domain/checkpoint when metrics or repairs differ. Never reuse a
   flower router for Ego video, or conflate `camera:*` controls with
   `robot_base:*` actions.

2. For native AC-WM, create case manifests using the required schema in
   `docs/DEMO_VIDEO_DATA_FLYWHEEL.md`, then use
   `scripts/run_acwm_demo_factory_worker.py`. For another generator/evaluator,
   wrap it as an equivalent command adapter. Keep every candidate inside
   `{attempt_dir}` and preserve source license/hash, explicit action coordinate
   frame, model/evaluator revisions, seed, prompts, controls, and evaluator
   evidence.

3. Bootstrap on at least two independent scene, object, subject, or embodiment
   groups. For AC-WM, use `scripts/build_acwm_demo_factory_campaign.py`; it must
   reject ambiguous licenses, hashes, revisions, frames, human-review paths, or
   same-source groups. Set `collect_all_recipes=true` and make maximum attempts
   equal the recipe count. Run `scripts/run_demo_factory_batch.py` without a
   policy. Do not use early-stop production traces as if they measured every
   repair.

4. Train with `scripts/train_demo_factory_router.py`. Keep groups intact and
   require every recipe in every group. Reject scene leakage, missing baselines,
   baseline-unbound contexts, human-review gaps, or incomplete tournaments.

5. Inspect `held-group-evaluation.json`. Promote only when every emitted gate
   passes. Aggregate utility never overrides action, embodiment, object,
   temporal, background, non-regression, or human-review failure.

6. Run production with the promoted `policy.json` and
   `collect_all_recipes=false`. The runner must inventory/select a physical GPU,
   set `CUDA_VISIBLE_DEVICES`, take a lease, and save selection before any GPU
   worker launches. Never bypass preflight to increase throughput.

7. Preserve a fixed exploration share of genuinely new groups with complete
   tournaments. Append their immutable `episodes.jsonl` files to the next
   training run. Keep rejected checkpoints and failed workers as evidence.

8. Use `distillation-preferences.jsonl` only as candidate planner supervision.
   Train a planner LoRA/SFT or preference adapter on training groups, then make
   it pass the identical held-group and cost gates before replacing the ridge
   router. Treat generator LoRA training as a separate, licensed-data campaign
   with frozen real-input acceptance groups.

9. Append every concluded bootstrap, training, or production attempt to
   `experiences/ledger.jsonl` via `scripts/experience_ledger.py`. Update
   `docs/STATUS.md` only to WORKING, PARTIAL, BLOCKED, or NOT STARTED evidence
   actually established by the run.

Use `scripts/import_ego_repair_history_to_demo_factory.py` only to migrate the
existing measured Ego repair tournament. Preserve null/failed human decisions.
Treat a non-promoted replay as negative evidence; never lower the contract to
turn historical failures into accepted examples.

## Stop conditions

- Stop before batch execution if the case license, group split, worker command,
  model/checkpoint revision, input hash, evaluator, GPU state, or coordinate
  frame is ambiguous.
- Stop promotion on any held-group capability regression even if acceptance,
  aesthetics, or average utility rises.
- Stop production when an explicit human veto, source-person residual, object
  loss, action failure, identity/topology failure, or invalid terminal state is
  present.
- Never call accepted generated video real-robot execution or physical-contact
  evidence.

Run the focused CPU tests, targeted Ruff checks, and skill-creator
`quick_validate.py` after any harness or skill change. A software test is not a
real video acceptance run; report that boundary explicitly.
