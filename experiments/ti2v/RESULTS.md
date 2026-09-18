# TI2V development results

Evidence snapshot: September 18, 2026. All completed results below use the previously opened 20-case development set with three seeds per case. They do not establish a full public-benchmark SOTA result.

## Complete comparison under one scoring protocol

| Method | BLEU ↑ | CLIP ↑ | hsd ↑ | dyn ↑ | ndtw ↑ |
|---|---:|---:|---:|---:|---:|
| MiniMax | 0.201205 | 88.915442 | 0.298517 | 0.204933 | 0.293650 |
| Historical Ours-v2 | 0.215548 | 89.223632 | 0.337533 | 0.250167 | 0.333533 |
| VideoWeaver adaptation | 0.206252 | 89.472307 | 0.332883 | 0.291683 | 0.357150 |
| SkillAdam full-loop adaptation | 0.199855 | 89.133862 | 0.335967 | 0.259583 | 0.329717 |
| PhiAgent-RSI, prespecified primary | 0.217046 | 89.317654 | 0.320183 | 0.236467 | 0.327150 |

Run `20260916T032105Z` scored all 60 outputs per row with the official metric code and a common per-video caption-seed protocol. VideoWeaver and SkillAdam are TI2V adaptations; historical optimization costs differ. Ours-v2 has higher BLEU and hsd means than VideoWeaver and lower CLIP, dyn, and ndtw. The paired intervals in [METHOD.md](METHOD.md) all cross zero.

## SkillAdam integration and crossed ablation

SkillAdam completed two rounds and accepted both updates. The separate bounded PhiAgent optimizer rejected both and retained its initial skill. Small validation-batch gains did not establish complete-set superiority. The integration and frozen learned skill are documented in [SKILLADAM_EXPLORATION.md](SKILLADAM_EXPLORATION.md).

| Optimizer and frame sampling | BLEU ↑ | CLIP ↑ | hsd ↑ | dyn ↑ | ndtw ↑ |
|---|---:|---:|---:|---:|---:|
| PhiAgent bounded updates + uniform | 0.211015 | 89.192610 | 0.325833 | 0.245550 | 0.331750 |
| PhiAgent bounded updates + event (prespecified primary) | 0.217046 | 89.317654 | 0.320183 | 0.236467 | 0.327150 |
| SkillAdam + uniform | 0.199855 | 89.133862 | 0.335967 | 0.259583 | 0.329717 |
| SkillAdam + PhiAgent event sampling | 0.205104 | 88.906126 | 0.319867 | 0.249867 | 0.312417 |

Each row contains 60/60 results. Within each optimizer, both sampling variants reuse the same generated candidates. The SkillAdam-plus-event combination increased BLEU but reduced the other four metrics relative to SkillAdam plus uniform sampling. Under both optimizers, event sampling reduced hsd, dyn, and ndtw. These negative results are retained; evaluator development is outside the current work.

The [evidence export](evidence/skilladam-exploration.json) copies existing server-produced means, decisions, and hashes without recomputing scores. Its optimization checkpoint was collected before final evaluation; the separate `final_results` section comes from the completed 60-output scoring archive.

## Relation-proposal diagnostics

The first two bounded relation-repair studies produced 0/60 admissible structured edits and stopped under their declared conditions without generating new videos. The first four-arm study also produced no admissible edits because the actor fields were inconsistent. After correcting that redundant encoding, the study obtained:

| Arm | Admissible edits | Cases covered | Calls | Tokens |
|---|---:|---:|---:|---:|
| Joint / original vocabulary | 15/60 | 6/20 | 78 | 228,254 |
| Factored / original vocabulary | 11/60 | 8/20 | 73 | 199,818 |
| Joint / extended vocabulary | 17/60 | 8/20 | 79 | 239,560 |
| Factored / extended vocabulary | 13/60 | 8/20 | 75 | 209,000 |

Verification covered 305 raw responses, 5,185 image payloads, and all 240 arm decisions. The two four-arm studies together used 553 calls and 1,594,622 tokens. In the original-vocabulary arms, nine and seven edits respectively referred to withdrawal that the task did not request. Those records are retained but excluded from the new generation plan.

These counts measure proposal coverage, not video-quality improvement. The summary and plans have SHA-256 bindings:

- Summary: `ac1c424d455c744a631ea4b6b63625ab38b1f0ffbae04f79abce743452444186`
- Plans: `57c92a11a836dc356ad5cde27429046888fe526f0e276d27f5d5fadca71afefc`

## Paired generation

The three arms are parent, joint extended repair, and factored extended repair. Every arm retains all 60 records. The 17 and 13 edits pass the literal action precondition; sharing identical complete prompts and inputs requires 77 distinct native videos.

Recovery run `20260918T095225Z` was frozen and submitted remotely after rechecking MiniMax weights, using physical GPUs 0 and 1 on H200-2. The scoring service retains the original wrapper. The collected snapshot includes the first completed native video, eight seconds at 1024×768, with verified SHA-256 `c0d2bdd26b12bc3dfa5d5a138218159252310e3d2e2b430894062995aa7bbefd`. Complete quality scores are pending.

Preparation `20260918T094146Z` stopped after the full template-condition audit, before any task video. Startup warmup costs remain recorded separately. The first three-arm preparation, `20260918T094601Z`, incorrectly included its changing preparation log in the frozen manifest and was blocked by startup verification, also before any task video. Recovery corrected only the manifest scope: model, prompts, cases, selector, and original deadline were preserved. Warmup and failure receipts are retained for each attempt.

This paired run has its own frozen parent and is not a completed SkillAdam-plus-repair experiment. The proposed composition and the comparison needed to isolate its contribution are described in [SKILLADAM_EXPLORATION.md](SKILLADAM_EXPLORATION.md).
