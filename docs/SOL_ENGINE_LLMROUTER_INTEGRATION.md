# Sol-Engine and LLMRouter integration contract

Status: **PARTIAL**.  PhiAgent now has dependency-free planning and acceptance
contracts plus a measured four-A800 reproduction of the released H3 full-opt
profile; broader task and hardware coverage remains follow-up work.

Sol-Attn is integrated as a pinned external runtime, never as a replacement for
PhiAgent's geometry or physical validators.  The current target is the official
Sol-Engine MiniMax-H3 4x A100 profile, which maps to four A800-SXM4 GPUs
(SM80).  The control and optimized runs must keep the checkpoint revision,
prompt, seed, steps, resolution, duration, GPU set, and SGLang topology fixed.
Only `H3_SOL_PROFILE` differs: `dense` versus `fullopt_exact`.

`scripts/plan_sol_engine_h3_ab.py` checks the physical GPU inventory, writes the
selection and two Docker launchers into a new experiment directory, and requires
the Sol-Engine source marker `6fb7eb11c3435555ec6d6adf0d5572d339d2c6eb`.  It
does not run a fallback path.  The optimized benchmark must report sparse calls
on all four measured ranks.  Acceptance requires at least 1.15x measured
generation speedup plus matched-input automated quality, temporal, action,
physical, and human-review evidence.  The gates say that the synthetic video
passed a specified evaluation; they do not make it real robot evidence.
The planner writes a false-by-default `quality_evidence.template.json` and
`scripts/assess_sol_engine_h3_ab.py` produces a fail-closed acceptance decision
from the two benchmarks and that evidence.
`scripts/evaluate_sol_engine_h3_video_pair.py` supplies deterministic matched-
input, frame-quality, and temporal evidence, but intentionally leaves action,
physical, and human-review gates false for a generic video pair.
`scripts/evaluate_and_assess_sol_engine_h3_pair.py` chains those two steps once
both `out.mp4` artifacts exist.  The assessor independently rejects any drift
in model partition, prompt hash, seed, generation parameters, or distributed
topology, even if a caller incorrectly marks the quality evidence as matched.

The model-selection part follows LLMRouter's endpoint-routing idea, but the
unit of routing is a complete PhiAgent backend profile, not a token-level fusion
of MiniMax, Wan, LingBot, and JoyAI.  Heterogeneous DiT models cannot share
weights, latents, or KV caches.  `phiagent.routing.route_request` first applies
capability, quality-tier, latency-budget, and physical-gate constraints; only
then does it select the lowest measured-median-latency profile.  This is the
baseline that a learned LLMRouter policy must beat offline using PhiAgent's own
request/profile outcome records.

`build_llmrouter_training_rows` exports those records with an oracle label only
when a candidate has passed generated-video, automated-quality, action,
physical, and review gates.  Failed or unevaluated outputs remain as unlabelled
examples; they are not silently converted into cheap-model preference data.

The external sources are pinned but intentionally not vendored:

- Sol-Engine: `NVlabs/Sana`, revision `6fb7eb11c3435555ec6d6adf0d5572d339d2c6eb`.
- LLMRouter: `ulab-uiuc/LLMRouter`, revision `b5a54da822fea6134ca7af55700685fc8431575f`.

## A800 reproduction versus the released A100 result

The matched four-A800 run at
`/mnt/goosefs/guoyijun/experiments/h3-sol-engine-ab-a800-20260816-009`
used the released 1344x768, 124-frame, 50-step, seed-0 workload.  The following
comparison uses Sol-Engine's published four-A100 numbers as the reference:

| Metric | Sol-Engine A100 release | PhiAgent A800 reproduction | Difference |
|---|---:|---:|---:|
| Dense inference | 217.32 s | 216.600 s | -0.33% |
| Full-opt inference | 61.28 s | 61.684 s | +0.66% |
| End-to-end speedup | 3.55x | 3.511x | -1.09% |

The A800 benchmark records the portable Triton Sol-Attn backend on all four
ranks, 18.0% effective route density on the sampled sparse call, and
FirstBlockCache reuse on 33 of 49 calls.  Peak reserved memory was 47,724 MiB
per GPU for dense and 48,098 MiB for full-opt.  This reproduces the composed
**Sol-Attn + FirstBlockCache** release profile; it is not a Sol-Attn-only
ablation.

Our additional 64-frame comparison recorded PSNR 16.740, SSIM 0.677, and
temporal relative error 0.127.  Sol-Engine's release methodology ranks lossy
generative profiles with aligned pairwise Gemini artifact severity and aligned
LPIPS, without an absolute PSNR/SSIM delivery threshold.  Consequently the
local PSNR/SSIM telemetry is reported for reproducibility but is not directly
comparable to a published A100 PSNR/SSIM value, because the release does not
publish one for this H3 profile.  The structured record is
`experiences/h3-sol-engine-a800-fullopt-v1-partial-record.json`.

## Verification commands

The integration stays importable without Torch, CUDA, or model checkpoints:

```bash
python -m pytest tests/test_sol_engine_router.py -q
ruff check phiagent/acceleration phiagent/routing \
  scripts/plan_sol_engine_h3_ab.py scripts/assess_sol_engine_h3_ab.py \
  scripts/evaluate_sol_engine_h3_video_pair.py \
  scripts/evaluate_and_assess_sol_engine_h3_pair.py tests/test_sol_engine_router.py
```

The GPU planner additionally requires the pinned external source, the H3 FL2VA
checkpoint, Docker, and four idle physical GPUs.  It creates a fresh experiment
directory with exact dense and full-opt launchers and records the selected GPU
inventory rather than silently choosing a fallback.
