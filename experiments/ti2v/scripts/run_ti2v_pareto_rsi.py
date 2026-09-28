"""Authorized remote three-arm comparison, final scoring and paired reporting."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback


def run_arm(root, method, cfg, rows):
    from integrations.skilladam_ti2v.backend import save, sha, GATES
    from integrations.skilladam_ti2v.pareto_rsi import ComparisonBackend, run_comparison_arm
    from phiagent.harness.video_meta_rsi import METRICS
    out = root/method; out.mkdir(exist_ok=False)
    state = {'status': 'OPTIMIZING', 'started_at': time.time(), 'hostname': socket.gethostname(),
             'scope': cfg['scope'], 'protocol_sha256': sha(root/'protocol.json')}
    save(out/'state.json', state)
    try:
        backend = ComparisonBackend(root, method)
        optrows = [r for r in rows if r['seed'] == cfg['optimizer_seed']]
        release = run_comparison_arm(backend, optrows, (root/'initial-skill.md').read_text(),
                 (root/'initial-policy.md').read_text(), sha(root/'protocol.json'), out/'optimizer', method)
        (out/'final-skill.md').write_text(release['skill']); (out/'final-policy.md').write_text(release['policy'])
        backend.phase = 'final'
        state.update(status='FINAL_GENERATION', release=release, optimizer_finished_at=time.time())
        save(out/'state.json', state)
        with ThreadPoolExecutor(max_workers=2) as pool:
            selected = list(pool.map(lambda r: backend.rollout(r, release['skill']), rows))
        save(out/'final-selection.json', {'records': selected, 'release': release,
                                         'all_selections_committed': True})
        state.update(status='FINAL_SCORING'); save(out/'state.json', state)
        result = backend.score(selected, 'final60'); save(out/'final-scores.json', result)
        expected = {r['case_id']+':'+str(r['seed']) for r in rows}
        if set(result['metrics']) != expected:
            raise ValueError('Final score coverage mismatch')
        # Diagnostic raw candidate bank includes failed/unknown candidates unchanged.
        raw = [{**r, 'video': r['candidate'], 'sha256': r['candidate_sha256']} for r in selected]
        raw_result = backend.score(raw, 'diagnostic-raw-final60')
        save(out/'raw-final-scores.json', raw_result)
        means = lambda keys, values: {m: sum(values[k][m] for k in keys)/len(keys) for m in METRICS}
        isolated = {r['case_id']+':'+str(r['seed']) for r in rows if r['split'] == 'test'}
        usage = [json.loads(p.read_text()) for p in (backend.out/'calls').glob('*/usage.json')]
        tokens = {k: sum(u.get(k, 0) for u in usage) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        summary = {'all60_development_means': means(expected, result['metrics']),
            'optimizer_isolated24_means': means(isolated, result['metrics']),
            'diagnostic_raw60_means': means(expected, raw_result['metrics']),
            'candidate_adoptions': sum(r['decision']['selected'] == 'candidate' for r in selected),
            'candidate_all_pass': sum(all(r['candidate_audit']['gates'][g]['status'] == 'PASS' for g in GATES) for r in selected),
            'physical_success_established': False, 'optimizer_isolated_cases_historically_opened': True,
            'cost': {k: v for k, v in backend.state.items() if k.endswith('_calls') or k.endswith('_hits')},
            'tokens': tokens, 'proposal_slots': release['proposal_slots_consumed'],
            'full_public_benchmark': False, 'SOTA_established': False,
            'official_evidence_sha256': result['official_evidence_sha256']}
        save(out/'summary.json', summary)
        state.update(status='DEVELOPMENT_COMPLETE', finished_at=time.time()); save(out/'state.json', state)
        return {'status': 'DEVELOPMENT_COMPLETE', 'summary': summary}
    except BaseException as exc:
        state.update(status='BLOCKED', error=f'{type(exc).__name__}: {exc}', finished_at=time.time())
        save(out/'state.json', state); (out/'failure.txt').write_text(traceback.format_exc())
        return state


def compare(root, methods, rows, completed):
    """Remote-only case-cluster bootstrap; fixed family of 15 paired tests."""
    import random
    from itertools import combinations
    from integrations.skilladam_ti2v.backend import save
    from phiagent.harness.video_meta_rsi import METRICS
    result = {'arms': completed, 'all_complete': all(v['status'] == 'DEVELOPMENT_COMPLETE' for v in completed.values()),
              'scope': 'historically opened development only', 'SOTA_established': False}
    if result['all_complete']:
        scores = {m: json.loads((root/m/'final-scores.json').read_text())['metrics'] for m in methods}
        result['paired_case_cluster_intervals'] = {}
        for split in ('all60', 'optimizer_isolated24'):
            cases = sorted({r['case_id'] for r in rows if split == 'all60' or r['split'] == 'test'})
            rng = random.Random(20260917)
            samples = [[rng.randrange(len(cases)) for _ in cases] for _ in range(20000)]
            comparisons = {}
            for a, b in combinations(methods, 2):
                comparison = {}
                for metric in METRICS:
                    deltas = [sum(scores[b][c+':'+str(s)][metric]-scores[a][c+':'+str(s)][metric]
                                  for s in (20260910, 20260911, 20260912))/3 for c in cases]
                    bootstrap = sorted(sum(deltas[i] for i in indices)/len(cases) for indices in samples)
                    alpha = .05/15
                    comparison[metric] = {'mean_delta_b_minus_a': sum(deltas)/len(deltas),
                        'familywise95_interval': [bootstrap[int(len(bootstrap)*alpha/2)],
                                                  bootstrap[min(len(bootstrap)-1, int(len(bootstrap)*(1-alpha/2)))]],
                        'case_clusters': len(cases)}
                comparisons[b+' minus '+a] = comparison
            result['paired_case_cluster_intervals'][split] = comparisons
    save(root/'comparison.json', result)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    root = ap.parse_args().root
    if socket.gethostname() != 'yxys-node-214-41-3-2' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Authorized remote CPU controller only')
    sys.path.insert(0, str(root/'source'))
    from phiagent.experiments import audit_experiment_timestamps
    from integrations.skilladam_ti2v.backend import save, sha
    from phiagent.harness.video_pareto_rsi import ARMS
    audit = audit_experiment_timestamps(root, require_reservation=True)
    save(root/'timestamp-audit.json', audit.to_dict())
    if not audit.valid:
        raise RuntimeError(audit.issues)
    for name, expected in json.loads((root/'source-manifest.json').read_text()).items():
        if sha(root/name) != expected:
            raise ValueError('Frozen source mismatch: '+name)
    cfg = json.loads((root/'config.json').read_text()); protocol = json.loads((root/'protocol.json').read_text())
    rows = json.loads((root/'inputs.json').read_text())['records']
    if len(rows) != 60 or len({r['case_id'] for r in rows}) != 20:
        raise ValueError('Full development set required')
    try:
        deadline = time.monotonic()+2400
        while time.monotonic() < deadline:
            states = [json.loads((Path(p)/'execution.json').read_text()).get('status')
                      if (Path(p)/'execution.json').exists() else 'STARTING' for p in protocol['native_pools']]
            if all(s == 'FRAMEWORK_POOL_READY' for s in states):
                break
            if any(s in ('BLOCKED', 'FRAMEWORK_POOL_STOPPED') for s in states):
                raise RuntimeError('Native preparation failed: '+repr(states))
            time.sleep(10)
        else:
            raise TimeoutError('Native startup expired')
        completed = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(run_arm, root, m, cfg, rows): m for m in ARMS}
            for future in as_completed(futures):
                completed[futures[future]] = future.result()
        compare(root, ARMS, rows, completed)
    finally:
        for p in protocol['native_pools']:
            if Path(p).parent == root:
                (Path(p)/'STOP').touch()


if __name__ == '__main__':
    main()
