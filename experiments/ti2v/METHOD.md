# Task-grounded relational repair for TI2V

Method ID: `task_grounded_relational_repair_v1`. The primary arm is `combined_extended`; its factored-decision ablation is `factored_extended`. The proposal entry is [`method.propose`](method.py), and paired generation is handled by [run_ti2v_extended_generation.py](scripts/run_ti2v_extended_generation.py).

This candidate complements the reusable-skill optimization explored with SkillAdam. SkillAdam learns shared instructions from training rollouts; this compiler limits a task-specific edit to one supported action relation. The completed SkillAdam integration, learned endpoint constraints, and the negative SkillAdam-plus-event-sampling result are documented in [SKILLADAM_EXPLORATION.md](SKILLADAM_EXPLORATION.md). A frozen SkillAdam skill followed by this compiler is a prospective composition, not a completed result.

## Algorithm

Robot-video failures can involve releasing an object before support, transferring it before the receiver grasps it, or losing contact while pushing. The method binds the original failure record, literal instruction, visible entities, and evidence frames to a finite relation vocabulary, then compiles one skill edit for the next generation.

1. Read the parent's original five-gate audit. Keep the parent skill when no explicit FAIL exists, and preserve UNKNOWN statuses. Bind the instruction, initial image, and parent video by SHA-256.
2. Make two fixed proposal calls over the initial image and sampled frames. The joint arm records entities, failures, and a relation, then refines the relation once. The factored arm records observations before choosing a relation. The second stage cannot change entity bindings or expand the inherited failure-frame evidence.
3. Apply the existing grounding check and a literal task precondition. Recognized task families are pickup, placement, handover, and pushing; an unknown family causes abstention. For example, placing an object permits support-before-release but does not imply an additional withdrawal action.
4. Compile one declared skill-slot replacement, on one line with at most 45 English words. Preserve task, input, entity, and evidence bindings. Save rejected proposals as well as accepted ones.
5. Generate with fixed model, camera constraints, seed, and sampling settings. Use the unchanged selector: a candidate must pass all five gates and repair an explicit parent failure. Otherwise retain the parent; fallback does not certify its success.

The six relations are contact-before-transport, maintain-grasp, release-before-withdrawal, support-before-release, receiver-grasp-before-giver-release, and maintain-contact-during-push. Templates are in [repair_factorial.py](integrations/skilladam_ti2v/repair_factorial.py); compilation constraints are in [relational_repair.py](integrations/skilladam_ti2v/relational_repair.py). The literal precondition uses a limited English rule set.

## Comparison with VideoWeaver

[VideoWeaver: Evaluating and Evolving Skills for Agentic Long Video Generation](https://arxiv.org/abs/2606.08091) already uses execution traces, final videos, and intermediate evidence to evaluate and evolve skills. Our hypothesis concerns the granularity of robot TI2V edits: each change must express one task-required relation supported by named entities and inherited failure frames.

This restriction could reduce additional actions and prompt drift while preserving appearance and camera behavior. SkillAdam provides a complementary way to learn reusable constraints across cases. Whether the combined approach improves the currently weaker dyn and ndtw metrics requires a direct experiment; proposal acceptance alone does not establish video quality.

| Component | Observed behavior | Quality hypothesis to test |
|---|---|---|
| Extended vocabulary | Joint arm produced 17/60 admissible edits across eight cases | Supported relations improve generated action order |
| Literal task precondition | Rejects unrequested withdrawal; unknown tasks retain the parent | Fewer extra actions improve task fidelity |
| One-slot edit | Compiler enforces location and length | Motion improves without disrupting scene or camera |
| Joint/factored decision | Extended arms produced 17/60 and 13/60 edits | Joint decisions yield greater video-quality gains |
| Frozen SkillAdam skill + repair | Compatible caller-supplied-skill interface; composition not yet run | Reusable constraints and case-specific repair complement one another |

## Completed historical comparison

These numbers describe historical Ours-v2, not the new repair candidate. The development comparison covers 20 cases with three seeds each. VideoWeaver uses a Qwen + MiniMax TI2V adaptation; historical optimization costs were not matched.

| Official metric ↑ | Ours-v2 | VideoWeaver adaptation | Paired difference | 99% interval for difference |
|---|---:|---:|---:|---:|
| BLEUScore | 0.215548 | 0.206252 | +0.009296 | [−0.049842, 0.067407] |
| CLIPScore | 89.223632 | 89.472307 | −0.248675 | [−1.902750, 1.424408] |
| hsd | 0.337533 | 0.332883 | +0.004650 | [−0.049184, 0.061953] |
| dyn | 0.250167 | 0.291683 | −0.041517 | [−0.116985, 0.029667] |
| ndtw | 0.333533 | 0.357150 | −0.023617 | [−0.102184, 0.046734] |

Differences and intervals come from the completed server-side [paired analysis](evidence/historical-paired-analysis.json): paired case-cluster bootstrap, 10,000 replicates, seed 20260918. Every interval crosses zero. The BLEU and hsd mean advantages therefore do not establish overall significant superiority. [RESULTS.md](RESULTS.md) includes the full comparison set and the SkillAdam crossed ablation.

## Current validation

Run `20260918T095225Z` compares the unchanged parent, joint extended repair, and factored extended repair. All arms retain 60 records. Identical full prompts and settings share generated bytes, requiring 77 distinct videos. The collected snapshot contains a verified first native-video receipt; complete quality scores are pending. The run uses its frozen implementation and parent, which this publication entry does not replace.

The candidate has no established overall advantage over VideoWeaver. These development cases have been used repeatedly; a formal claim requires complete public-benchmark comparisons, component ablations, and independent confirmation. Human review is excluded, and evaluator improvements belong to collaborators.

## Interface

After preparing the backend, parent audit, skill, and sampled frames on an authorized server:

```python
from pathlib import Path
from method import propose

plan = propose(
    backend, row, base_audit, parent_skill,
    Path(new_run) / "proposal", images, positions,
    arm="combined_extended",
)
# plan["skill"] is used for the next generation; abstention retains parent_skill.
# plan.json retains the raw plan; method-decision.json records the final decision.
```

`row` must contain instruction, initial, initial_sha256, base_sha256, and seed. Only these fields are forwarded; official scores, reference futures, and competitor outputs are excluded from the proposal input. Deployment and service contracts are in [DEPLOYMENT.md](DEPLOYMENT.md).
