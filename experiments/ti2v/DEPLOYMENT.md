# Runtime and reproduction interfaces

## Source and dependencies

Pure logic modules do not require CUDA, PyTorch, or model weights. `TI2VBackend` calls remote service adapters. SkillAdam is imported only when registering or running the optimizer; install its pinned author checkout separately under the experiment's `upstream/` path. See [model-pins.json](configs/model-pins.json) and [SKILLADAM_EXPLORATION.md](SKILLADAM_EXPLORATION.md).

The remote controller needs FFmpeg/FFprobe, `jsonschema`, and a separate Python environment capable of running XGrammar. `protocol.json` points to that environment through `grammar_python`. Generation, Qwen, and official scoring services use their separately validated environments; controller dependencies are not a complete service installation recipe.

The generation service requires complete weight attestation for the selected MiniMax-H3 partition, pinned runtime source, and validated GPUs. Save physical device selection, CUDA_VISIBLE_DEVICES, UUIDs, process ownership, commands, package versions, and native-video receipts. Third-party model repositories are not vendored here.

## Files for paired generation

| File or directory | Contract |
|---|---|
| `source/` | Frozen copy of this package, used as the import root |
| `protocol.json` | `expected_hostname`, deadline, native/auxiliary budgets, `generator`, `auxiliary`, `grammar_python`, and `native_pools` |
| `experiment.json` | `reserved_at`, `max_seconds`, `generation_workers`, `owned_native_pools`, and `stage_budgets.generate` |
| `inputs.json` | `records` with `case_id`, `seed`, literal `instruction`, `initial`, `initial_sha256`, `base`, and `base_sha256` |
| `parent-selections.json` | `base_audit` for the full manifest; historical FAIL/UNKNOWN statuses are preserved |
| `planning/plans-frozen.json` | parent/combined_extended/factored_extended arms, complete `prompts` per record, and `readiness=READY_FOR_GENERATION` |
| `planning/state.json` | `plans_sha256` binding the plan file |
| `source-manifest.json` | Relative paths to SHA-256 for frozen inputs, configuration, and source; excludes changing outputs |
| `qwen.lock` | One shared lock target across tasks using the same Qwen service |
| `score-queue/` | Official scoring requests and results handled by a separate service |

`scripts.ti2v_frozen_manifest.freeze_manifest(run_root)` creates the paired-generation manifest. It includes required inputs, configuration, and source, excluding changing logs and state. Missing files and source symlinks escaping the run directory cause failure. The optimizer campaign has its own frozen input/configuration manifest, including case splits and `initial-skill.md`; do not substitute the paired-generation template for it.

The `auxiliary` configuration must provide the actual endpoint and served model name, with exact revision, service process, and GPU binding recorded. The CPU controller requires `CUDA_VISIBLE_DEVICES=''`; GPU services use validated physical devices. Resolve and freeze placeholders in [paired-generation.example.json](configs/paired-generation.example.json) on the server.

The current manifest checks expect 20 cases and seeds 20260910, 20260911, and 20260912. A different dataset requires a new protocol and corresponding manifest validation; development scores cannot be relabeled as complete test-set results.

## Service contracts

Native generation pools accept `.job.json` under `queue/` and return `.job.result.json` with paths, hashes, and generation receipts. Their `execution.json` must report readiness. Startup warmup counts toward actual cost; pool limits must cover it and every distinct prompt.

Scoring requests contain a `selected` list of case, seed, video path, and hash, and are submitted only after selections are frozen. Results must be `SCORED`, with five metrics per case and official evidence hashes. Pin the EWMBench wrapper, per-video caption seed, and video processing protocol. Reference futures stay at the scoring service; scores must not leak into generation or selection. The SkillAdam optimizer receives its declared training/validation score feedback after those rollouts have been selected and scored.

The project environment supplies queue deployment and cross-server transfer. This release contains method, controller, and verification code rather than an automatic model installer. Missing services leave a run waiting; do not start model inference or benchmark experiments on the workstation.

## Interpretation

Model-based checks, schema validity, and template admission do not establish human ground truth or video quality. Report raw candidates and selected outputs, grouping the three seeds by case. Sparse image checks do not establish 3-D geometry, contact force, joint feasibility, or real-robot success. Detailed environment and per-call evidence remain in the experiment archive; the public export retains relevant metrics and binding hashes.
