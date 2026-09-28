"""Analyze completed relational-repair evidence on the authorized server only."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time


METRICS = ('BLEUScore', 'CLIPScore', 'hsd', 'dyn', 'ndtw')


def sha(target):
    return hashlib.sha256(Path(target).read_bytes()).hexdigest()


def analyze(root):
    if socket.gethostname() != 'yxys-node-214-41-3-2' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Analysis is remote-only, with controller GPU visibility disabled')
    output = root / 'analysis'
    output.mkdir(exist_ok=False)
    experiment = json.loads((root / 'experiment.json').read_text())
    state = json.loads((root / 'planning/state.json').read_text())
    if state['status'] not in ('READY_FOR_GENERATION', 'STOP_NO_STRUCTURED_INTERVENTION'):
        raise ValueError('Planning is not complete')
    plans_file = root / 'planning/plans-frozen.json'
    if sha(plans_file) != state['plans_sha256']:
        raise ValueError('Frozen plans no longer match their receipt')
    plans = json.loads(plans_file.read_text())
    records = plans['records']
    if len(records) != 60 or len({(record['case_id'], record['seed']) for record in records}) != 60:
        raise ValueError('Incomplete planning population')
    summary = {
        'at': time.time(), 'hostname': socket.gethostname(),
        'run_id': experiment['run_id'], 'cases': 20, 'seeds_per_case': 3,
        'scope': 'Repeatedly opened development; no unseen confirmation or admitted innovation',
        'source_sha256': sha(__file__), 'protocol_sha256': sha(root / 'protocol.json'),
        'experiment_sha256': sha(root / 'experiment.json'), 'plans_sha256': sha(plans_file),
        'coverage': plans['coverage'], 'arms': {}, 'official_metrics': None,
        'counterfactual_quality_not_measured_by_coverage': True,
    }
    parent_records = json.loads((root / 'parent-selections.json').read_text())['records']
    source_gates = {}
    for record in parent_records:
        for gate, verdict in record['base_audit']['gates'].items():
            source_gates.setdefault(gate, Counter())[verdict['status']] += 1
    summary['uncalibrated_source_gate_counts'] = {gate: dict(counts) for gate, counts in source_gates.items()}
    for arm in ('free_form', 'structured', 'instruction_only'):
        summary['arms'][arm] = {
            'status_counts': dict(Counter(record[arm + '_status'] for record in records)),
            'applied_cases': sorted({record['case_id'] for record in records
                                    if record[arm + '_status'] == 'APPLIED'}),
        }
        if arm != 'free_form':
            summary['arms'][arm]['reason_counts'] = dict(Counter(record[arm + '_reason'] for record in records))
    guard_statuses = {}
    evidence_counts = Counter()
    for index, record in enumerate(records):
        folder = root / 'planning/cases' / f'{index:03d}'
        for name, expected in record['plan_hashes'].items():
            if sha(folder / name) != expected:
                raise ValueError('Individual plan receipt changed')
        relation = json.loads((folder / 'relation-plan.json').read_text())
        for check, verdict in relation.get('guard', {}).items():
            guard_statuses.setdefault(check, Counter())[verdict['status']] += 1
        for assessment in relation.get('assessments', {}).get('assessments', []):
            evidence_counts[assessment['decision']] += 1
    summary['structured_guard_counts'] = {name: dict(counts) for name, counts in guard_statuses.items()}
    summary['structured_gate_assessment_counts'] = dict(evidence_counts)
    planning_usage = json.loads((root / 'relational_plan/execution/state.json').read_text())
    summary['cost'] = {'native_calls': planning_usage['native_calls'],
                       'auxiliary_calls': planning_usage['auxiliary_calls'],
                       'planning_seconds': state['finished_at'] - state['started_at'],
                       'inherited_base_generation_and_audits_excluded': True}
    token_totals = Counter()
    usage_files = list((root / 'relational_plan/execution/calls').glob('*/usage.json'))
    for usage_file in usage_files:
        usage = json.loads(usage_file.read_text())
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            if isinstance(usage.get(key), int):
                token_totals[key] += usage[key]
    summary['cost']['planning_usage_receipts'] = len(usage_files)
    summary['cost']['planning_tokens'] = dict(token_totals)
    if not plans['coverage']['structured']:
        if (root / 'generation').exists() or planning_usage['native_calls'] != 0:
            raise ValueError('Generation occurred despite the predeclared zero-coverage stop')
        summary.update(status='STOPPED_AT_PREREGISTERED_ZERO_COVERAGE_GATE',
                       conclusion='No valid structured intervention; quality comparison and skill learning not run')
    elif not (root / 'generation/scores.json').exists():
        summary.update(status='PLANNING_COMPLETE_AWAITING_GENERATION',
                       conclusion='Valid interventions exist; no quality result yet')
    else:
        import numpy as np
        generation = json.loads((root / 'generation/state.json').read_text())
        if generation['status'] != 'DEVELOPMENT_COMPLETE':
            raise ValueError('Generation and official scoring are incomplete')
        scores = json.loads((root / 'generation/scores.json').read_text())
        identities = [record['case_id'] + ':' + str(record['seed']) for record in records]
        cases = sorted({record['case_id'] for record in records})
        score_values = {}
        means = {}
        for name, receipt in scores.items():
            if receipt['status'] != 'SCORED' or receipt['videos'] != 60 or set(receipt['metrics']) != set(identities):
                raise ValueError('Incomplete official scoring receipt: ' + name)
            values = np.array([[receipt['metrics'][identity][metric] for metric in METRICS] for identity in identities])
            if not np.isfinite(values).all():
                raise ValueError('Non-finite official score')
            score_values[name] = values
            means[name] = dict(zip(METRICS, values.mean(axis=0).tolist()))
        differences = score_values['structured_selected'] - score_values['free_form_selected']
        case_means = np.array([differences[[record['case_id'] == case_id for record in records]].mean(axis=0)
                               for case_id in cases])
        generator = np.random.default_rng(experiment['statistics']['seed'])
        indices = generator.integers(0, len(cases), size=(experiment['statistics']['replicates'], len(cases)))
        bootstraps = case_means[indices].mean(axis=1)
        bounds = np.quantile(bootstraps, [0.005, 0.995], axis=0)
        summary['official_metrics'] = {'means': means, 'primary_selected_structured_minus_free_form': {
            metric: {'difference': float(differences[:, index].mean()),
                     'case_cluster_99_percent_interval': bounds[:, index].tolist()}
            for index, metric in enumerate(METRICS)}, 'numpy_version': np.__version__,
            'raw_candidates_and_other_controls': 'exploratory mechanism comparisons',
            'scope': 'Five-metric primary family; repeated development remains exploratory'}
        generation_usage = json.loads((root / 'relational_generate/execution/state.json').read_text())
        for key in ('native_calls', 'auxiliary_calls'):
            summary['cost'][key] += generation_usage[key]
        summary.update(status='DEVELOPMENT_COMPLETE', conclusion='Inspect all contrasts; no generalization claim')
    prior_cost = experiment.get('prior_campaign_cost', {'native_calls': 0, 'auxiliary_calls': 0})
    summary['cost']['campaign_native_calls'] = summary['cost']['native_calls'] + prior_cost['native_calls']
    summary['cost']['campaign_auxiliary_calls'] = summary['cost']['auxiliary_calls'] + prior_cost['auxiliary_calls']
    if experiment.get('parent_run'):
        parent_file = root.parent / experiment['parent_run'] / 'analysis/summary.json'
        if sha(parent_file) != experiment['parent_summary_sha256']:
            raise ValueError('Parent experiment summary binding changed')
        parent_summary = json.loads(parent_file.read_text())
        summary['parent_run'] = experiment['parent_run']
        summary['parent_summary_sha256'] = sha(parent_file)
        summary['cost']['campaign_planning_tokens'] = dict(
            Counter(summary['cost']['planning_tokens']) + Counter(parent_summary['cost']['planning_tokens']))
        summary['cost']['sum_of_planning_stage_seconds'] = (
            summary['cost']['planning_seconds'] + parent_summary['cost']['planning_seconds'])
    if summary['cost']['campaign_native_calls'] > 300 or summary['cost']['campaign_auxiliary_calls'] > 720:
        raise ValueError('Recorded usage exceeds the frozen global ceiling')
    for value in summary['cost']['planning_tokens'].values():
        if not math.isfinite(value):
            raise ValueError('Non-finite usage receipt')
    target = output / 'summary.json'
    target.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root)


if __name__ == '__main__':
    main()