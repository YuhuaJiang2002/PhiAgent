"""Connect the bounded meta-RSI controller to the existing remote TI2V backend.

Call only inside a newly reserved and source-frozen remote campaign. Construction
of TI2VBackend checks the authorized host and disabled controller GPU visibility;
its services retain GPU selection and generation/scoring provenance. This adapter
never changes the old runner or any existing run directory.
"""
from phiagent.harness.video_meta_rsi import MetaRSI, Partition, ScoreCard, METRICS, GATES, text_hash
from scripts.run_ti2v_skill_optimization import bounded_edit
from .backend import obj, TEXT, save, sha

FIELDS = {'hypothesis': TEXT, 'expected_effect': TEXT, 'risk': TEXT}
POLICY_SCHEMA = obj({**FIELDS, 'policy': TEXT})
SKILL_SCHEMA = obj({**FIELDS, 'edits': {'type': 'array', 'maxItems': 2,
                                   'items': obj({'old': TEXT, 'new': TEXT})}})


def run_meta_rsi(backend, rows, initial_skill, initial_policy, protocol_sha256, output_dir, *, terms=2):
    """Return a frozen release; final scoring must be scheduled separately.

    Existing historically opened cases must not be renamed as an unseen test.
    The caller supplies a new frozen manifest with four case-disjoint partitions.
    """
    from pathlib import Path
    import json
    from concurrent.futures import ThreadPoolExecutor
    if protocol_sha256 != sha(backend.root / "protocol.json"):
        raise ValueError("Controller contract must bind the actual frozen remote protocol")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=False)
    by_split = {name: [r for r in rows if r['split'] == name]
                for name in ('train', 'validation', 'meta_validation', 'test')}
    if sum(len(v) for v in by_split.values()) != len(rows):
        raise ValueError('Unknown row split')
    partitions = [Partition(name, tuple((r['case_id'], r['seed']) for r in group))
                  for name, group in by_split.items()]

    def record(stage, data):
        path = out/(stage + '.json')
        if path.exists():
            raise FileExistsError('Do not overwrite a prior RSI receipt')
        save(path, data)

    def rollouts(skill, part):
        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(lambda row: backend.rollout(row, skill), by_split[part.name]))

    def observe(skill, part, stage):
        records = rollouts(skill, part)
        return [{'case_id': r['case_id'], 'seed': r['seed'], 'skill_sha256': text_hash(skill),
                 'video_sha256': r['candidate_sha256'], 'trace': json.dumps(r['trajectory']),
                 'gates': {g: r['candidate_audit']['gates'][g]['status'] for g in GATES}}
                for r in records]

    def propose_policy(policy, signal, stage):
        prompt = ('Propose a task-general policy for improving reusable robotic video skills. '
                  'Use only these training observations. State a hypothesis, expected effect, and risk. '
                  'The policy is at most 220 words and guides diagnosis and two-line skill edits. '
                  'You cannot edit the evaluator, physical checks, split membership, budgets, or promotion rule. '
                  'Do not embed case identifiers, reference futures, metric targets or memorized scene answers.\n'
                  'CURRENT POLICY:\n' + policy + '\nTRAINING SIGNAL:\n' + json.dumps(signal))
        proposal, usage = backend.query([{'role': 'user', 'content': prompt}], POLICY_SCHEMA, stage)
        record(stage + '-usage', usage)
        return proposal

    def propose_skill(policy, skill, signal, stage):
        prompt = ('Only propose at most two exact existing non-heading line substitutions or deletions. '
                  'Preserve literal task, entity, endpoint, and visible contact order. '
                  'No evaluator identities, case identifiers, scene-specific answers, future references or extra actions. '
                  'The policy below is proposal guidance; it cannot change these rules. '
                  'Each replacement is at most 55 words, whole skill at most 300 words.\n'
                  'POLICY:\n' + policy + '\nSKILL:\n' + skill + '\nTRAINING ONLY:\n' + json.dumps(signal))
        proposal, usage = backend.query([{'role': 'user', 'content': prompt}], SKILL_SCHEMA, stage)
        record(stage + '-usage', usage)
        return proposal

    def evaluate(skill, part, stage):
        records = rollouts(skill, part)
        result = backend.score(records, stage)
        expected = {r['case_id'] + ':' + str(r['seed']) for r in records}
        if set(result['metrics']) != expected:
            raise ValueError('Incomplete or extra official result rows')
        values = []
        for r in records:
            scores = result['metrics'][r['case_id'] + ':' + str(r['seed'])]
            if set(scores) != set(METRICS):
                raise ValueError('Require exactly the frozen five official metrics')
            values.append((r['case_id'], r['seed'], tuple(scores[m] for m in METRICS)))
        return ScoreCard(text_hash(skill), part.sha256, protocol_sha256,
                         result['official_evidence_sha256'], tuple(values))

    controller = MetaRSI(partitions, protocol_sha256, observe=observe, propose_policy=propose_policy,
                         propose_skill=propose_skill, apply_patch=bounded_edit, evaluate=evaluate, record=record)
    try:
        return controller.run(initial_skill, initial_policy, terms=terms)
    except Exception as exc:
        record("failure", {"type": type(exc).__name__, "message": str(exc)})
        raise
