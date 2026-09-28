"""Remote adapter for the prospectively frozen three-arm RSI comparison."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json

from .backend import TI2VBackend, save, sha, digest, GATES
from .meta_rsi import POLICY_SCHEMA, SKILL_SCHEMA
from phiagent.harness.video_meta_rsi import Partition, ScoreCard, METRICS, text_hash
from phiagent.harness.video_pareto_rsi import ParetoRSI, Assessment
from scripts.run_ti2v_skill_optimization import bounded_edit


class ComparisonBackend(TI2VBackend):
    def __init__(self, root, method):
        super().__init__(root, method)
        self.phase = 'optimization'
        self.state.update(optimization_native_calls=0, final_native_calls=0,
                          base_audit_cache_hits=0, official_score_cache_hits=0)
        manifest = json.loads((self.root/'base-audits.json').read_text())
        if manifest['auxiliary'] != self.cfg['auxiliary']:
            raise ValueError('Base audits require the same pinned auxiliary identity')
        self.base_audits = {(r['case_id'], r['seed']): r for r in manifest['records']}
        self.update()

    def audit(self, row, video, folder, mode='uniform'):
        if str(video) != str(row['base']):
            return super().audit(row, video, folder, mode)
        item = self.base_audits[(row['case_id'], row['seed'])]
        for name in ('instruction', 'initial_sha256', 'base_sha256'):
            if item[name] != row[name]:
                raise ValueError('Stale common base audit binding: ' + name)
        if mode != 'uniform' or sha(video) != item['base_sha256']:
            raise ValueError('Base audit sampling/video mismatch')
        folder.mkdir(parents=True, exist_ok=False)
        save(folder/'audit.json', item['audit'])
        save(folder/'reuse.json', {'source': item['source'], 'source_sha256': item['source_sha256'],
             'base_manifest_sha256': sha(self.root/'base-audits.json'),
             'common_to_all_arms': True, 'new_auxiliary_calls': 0})
        with self.lock:
            self.state['base_audit_cache_hits'] += 1; self.update()
        return item['audit'], []

    def native(self, row, prompt, folder):
        with self.lock:
            key = self.phase + '_native_calls'
            limit = 68 if self.phase == 'optimization' else 60
            if self.state[key] >= limit:
                raise RuntimeError('Native phase budget exhausted; final reserve protected')
            self.state[key] += 1; self.update()
        return super().native(row, prompt, folder)

    def score(self, records, stage):
        key = digest({'outputs': sorted((r['case_id'], r['seed'], r['sha256']) for r in records),
                      'protocol_sha256': sha(self.root/'protocol.json')})
        cache = self.out/'score-cache'/(key+'.json')
        if cache.exists():
            with self.lock:
                self.state['official_score_cache_hits'] += 1; self.update()
            return json.loads(cache.read_text())
        result = super().score(records, stage); save(cache, result)
        return result


def run_comparison_arm(backend, rows, skill, policy, protocol, output_dir, arm):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=False)
    groups = {n: [r for r in rows if r['split'] == n]
              for n in ('train', 'validation', 'meta_validation', 'test')}
    parts = [Partition(n, tuple((r['case_id'], r['seed']) for r in group))
             for n, group in groups.items()]
    if protocol != sha(backend.root/'protocol.json'):
        raise ValueError('Protocol binding mismatch')

    def record(stage, value):
        path = out/(stage+'.json')
        if path.exists():
            raise FileExistsError('Immutable RSI receipt already exists')
        save(path, value)

    def rollouts(value, part):
        with ThreadPoolExecutor(max_workers=2) as pool:
            return list(pool.map(lambda r: backend.rollout(r, value), groups[part.name]))

    def observe(value, part, stage):
        rows = rollouts(value, part)
        return [{'case_id': r['case_id'], 'seed': r['seed'], 'skill_sha256': text_hash(value),
                 'video_sha256': r['candidate_sha256'], 'trace': json.dumps(r['trajectory']),
                 'gates': {g: r['candidate_audit']['gates'][g]['status'] for g in GATES}} for r in rows]

    def propose_policy(value, signal, stage):
        prompt = ('Improve a task-general policy for making two-line edits to a robotic video skill. '
                  'Use only the supplied training traces and proposal history. Diagnose failed or repeated '
                  'repairs, preserve successful behavior, and change how useful edits are proposed. '
                  'Return hypothesis, expected_effect, risk, and policy (at most 220 words). '
                  'You cannot change checks, evaluators, partitions, model identity, budgets or acceptance rules. '
                  'No case identifiers, metric targets, reference futures or memorized scene answers.\n'
                  'CURRENT POLICY:\n'+value+'\nTRAINING ONLY:\n'+json.dumps(signal))
        result, usage = backend.query([{'role': 'user', 'content': prompt}], POLICY_SCHEMA, stage)
        record(stage+'-usage', usage)
        return result

    def propose_skill(value, current, signal, stage):
        prompt = ('Propose at most two exact existing non-heading whole-line edits or deletions. '
                  'Each replacement at most 55 words; entire skill at most 300 words. '
                  'Preserve literal task, named arm/object/endpoint and visible action order. '
                  'No extra actions, hidden states, future references, case IDs or evaluator identities. '
                  'Avoid repeating prior proposals; use a different supported repair if available, '
                  'otherwise abstain with an empty edits array. The policy is guidance only.\n'
                  'POLICY:\n'+value+'\nSKILL:\n'+current+'\nTRAINING ONLY:\n'+json.dumps(signal))
        result, usage = backend.query([{'role': 'user', 'content': prompt}], SKILL_SCHEMA, stage,
                                       seed=20260917 + signal['proposal_slot'])
        record(stage+'-usage', usage)
        return result

    def evaluate(value, part, stage):
        rows = rollouts(value, part); result = backend.score(rows, stage)
        expected = {r['case_id']+':'+str(r['seed']) for r in rows}
        if set(result['metrics']) != expected:
            raise ValueError('Incomplete official scores')
        values = []
        for r in rows:
            v = result['metrics'][r['case_id']+':'+str(r['seed'])]
            if set(v) != set(METRICS):
                raise ValueError('Metric identity mismatch')
            values.append((r['case_id'], r['seed'], tuple(v[m] for m in METRICS)))
        card = ScoreCard(text_hash(value), part.sha256, protocol,
                         result['official_evidence_sha256'], tuple(values))
        return Assessment(card, tuple((r['case_id'], r['seed'], r['sha256']) for r in rows))

    controller = ParetoRSI(parts, protocol, observe=observe, propose_policy=propose_policy,
        propose_skill=propose_skill, apply_patch=bounded_edit, evaluate=evaluate, record=record)
    try:
        return controller.run(skill, policy, arm=arm)
    except Exception as exc:
        record('failure', {'type': type(exc).__name__, 'message': str(exc)})
        raise
