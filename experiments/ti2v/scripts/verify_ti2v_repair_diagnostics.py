"""Remote exact-payload audit of method diagnostics; no inference or metric calls."""
import argparse
import base64
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


def sha(target):
    return hashlib.sha256(Path(target).read_bytes()).hexdigest()


def image_hashes(content):
    hashes = []
    for item in content:
        if item['type'] != 'image_url':
            continue
        url = item['image_url']['url']
        if not url.startswith(('data:image/png;base64,', 'data:image/jpeg;base64,')):
            raise ValueError('Unexpected image transport')
        hashes.append(hashlib.sha256(base64.b64decode(url.split(',', 1)[1], validate=True)).hexdigest())
    return hashes


def initial_binding_barriers(task):
    barriers = []
    if not task['validated']['task_binding_valid']:
        barriers.append('INVALID_LITERAL_BINDING')
    if task['value']['object_identifiable'] != 'YES':
        barriers.append('TARGET_NOT_IDENTIFIED')
    if task['value']['initial_object_state'] in ('NOT_IDENTIFIABLE', 'UNKNOWN'):
        barriers.append('INITIAL_OBJECT_STATE_UNRESOLVED')
    if task['validated']['contradictions']:
        barriers.append('CONTRADICTORY_APPLICABILITY_FIELDS')
    return barriers


def record_decision(root):
    if socket.gethostname() != 'yxys-node-214-41-3-2' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Decision analysis is remote CPU-only')
    summary_file = root / 'diagnosis/summary.json'
    verification_file = root / 'verification.json'
    summary = json.loads(summary_file.read_text())
    verification = json.loads(verification_file.read_text())
    if verification['status'] != 'PASS' or verification['summary_sha256'] != sha(summary_file):
        raise ValueError('A verified complete diagnostic is required')
    tasks_file = root / 'diagnosis/task-applicability-frozen.json'
    records_file = root / 'diagnosis/records.json'
    if sha(tasks_file) != summary['tasks_sha256'] or sha(records_file) != summary['records_sha256']:
        raise ValueError('Frozen task or output records changed')
    tasks = json.loads(tasks_file.read_text())
    records = json.loads(records_file.read_text())
    barriers = {case_id: initial_binding_barriers(task) for case_id, task in tasks.items()}
    clean_templates = Counter()
    for case_id, task in tasks.items():
        if not barriers[case_id]:
            clean_templates.update(name for name, verdict in task['validated']['template_applicability'].items()
                                   if verdict == 'APPLICABLE')
    failure_groups = Counter()
    for record in records:
        for comparison in record['diagnosis']['gates'].values():
            if comparison['original_status'] != 'FAIL':
                continue
            if barriers[record['case_id']]:
                group = 'INITIAL_ENTITY_OR_BINDING_UNRESOLVED'
            elif comparison['diagnosis'] == 'REPEATED_FAILURE_WITH_TEMPLATE_TOPIC':
                group = 'TEMPLATE_TOPIC_ONLY_EXACT_REPAIRABILITY_UNTESTED'
            else:
                group = comparison['diagnosis']
            failure_groups[group] += 1
    decision = {
        'status': 'DIAGNOSTIC_COMPLETE_NEXT_EXPERIMENT_SPECIFIED_NOT_LAUNCHED',
        'analysis_type': 'Post-hoc method-side decision analysis; original summaries and verdicts unchanged',
        'source_sha256': sha(__file__), 'run_id': summary['run_id'],
        'summary_sha256': sha(summary_file), 'verification_sha256': sha(verification_file),
        'binding_barriers_by_case': {case_id: reasons for case_id, reasons in barriers.items() if reasons},
        'cases_without_reported_initial_binding_barriers': sum(not reasons for reasons in barriers.values()),
        'applicable_template_cases_without_reported_initial_binding_barriers': {
            name: clean_templates[name] for name in summary['template_case_counts']},
        'failure_observations_by_binding_and_scope': dict(failure_groups),
        'scope_counts_are_not_repair_success_or_human_ground_truth': True,
        'decision': 'NO_NEW_GENERATION_OR_RSI; specify a separate proposal-stage factorized binding and action-coverage study',
        'next_study': {
            'status': 'NOT_STARTED',
            'factor_A': ['joint_failure_and_template_abstention', 'separate_entity_binding_failure_relation_and_template_match'],
            'factor_B': ['original_three_templates', 'original_templates_plus_support_release_handover_and_push_contact'],
            'expanded_relations': ['support_before_release', 'receiver_grasp_before_giver_release', 'maintain_contact_during_push'],
            'unchanged': ['original five gates and selector', 'official scorer', 'all twenty cases and three seeds',
                          'model revision, input bytes, seeds and frames', 'no official scores or reference futures in the proposer'],
            'binding_rules': ['unnamed actor remains unspecified, never invent left or right',
                              'unidentified target or unresolved initial state cannot justify an executable repair',
                              'template applicability is distinct from a visible violation of that template',
                              'placement must not acquire an unrequested withdrawal'],
            'isolation': 'Equal proposal and token ceilings; freeze all four arms before calls; report all failures and abstentions',
            'progression': 'Proceed to separately frozen generation only if an evidence-bound edit changes an actual prompt; coverage alone is not effectiveness',
            'stop': 'No valid structured intervention stops before generation; do not increase RSI depth or relax physical checks',
        },
        'human_review_requested': False, 'evaluator_development_started': False,
        'population_accuracy_established': False, 'new_model_calls_for_this_analysis': 0,
    }
    target = root / 'followup-decision.json'
    if target.exists():
        raise FileExistsError('Do not overwrite a recorded decision')
    target.write_text(json.dumps(decision, indent=2, allow_nan=False))
    print(json.dumps(decision, indent=2, allow_nan=False))


def verify(root):
    if socket.gethostname() != 'yxys-node-214-41-3-2' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Remote CPU-only artifact verification required')
    sys.path.insert(0, str(root / 'source'))
    from integrations.skilladam_ti2v.backend import GATE_SCHEMA, JUDGE
    from integrations.skilladam_ti2v.relational_repair import TEMPLATES
    from integrations.skilladam_ti2v.repair_diagnostics import TASK_SCHEMA, TASK_PROMPT, diagnose_input, validate_task_diagnosis
    protocol = json.loads((root / 'protocol.json').read_text())
    experiment = json.loads((root / 'experiment.json').read_text())
    state = json.loads((root / 'diagnosis/state.json').read_text())
    if state['status'] != 'DIAGNOSTIC_COMPLETE':
        raise ValueError('Require a complete diagnostic')
    if sha(root / 'diagnosis/summary.json') != state['summary_sha256']:
        raise ValueError('Diagnostic summary binding changed')
    summary = json.loads((root / 'diagnosis/summary.json').read_text())
    for relative, expected in json.loads((root / 'source-manifest.json').read_text()).items():
        if sha(root / relative) != expected:
            raise ValueError('Frozen artifact changed: ' + relative)
    for name, key in (('records.json', 'records_sha256'), ('task-applicability-frozen.json', 'tasks_sha256')):
        if sha(root / 'diagnosis' / name) != summary[key]:
            raise ValueError('Diagnostic records changed')
    rows = json.loads((root / 'inputs.json').read_text())['records']
    records = json.loads((root / 'diagnosis/records.json').read_text())
    tasks = json.loads((root / 'diagnosis/task-applicability-frozen.json').read_text())
    inherited = {(row['case_id'], row['seed']): row for row in json.loads((root / 'parent-selections.json').read_text())['records']}
    if len(rows) != len(records) or len(rows) != 60 or len(tasks) != 20:
        raise ValueError('Incomplete population')
    calls = sorted((root / 'diagnostic_model/execution/calls').iterdir())
    if experiment.get('recovery_parent'):
        calls.insert(0, root / 'retained-first-call')
    if len(calls) != 80:
        raise ValueError('Incorrect call count')
    checked_images = 0
    old_rollouts = Path('/opt/phiagent/runs/ti2v-skill-optimization/20260916T032105Z/recovery-2/phiagent/execution/rollouts')
    old_index = {}
    for target in old_rollouts.glob('*/selection.json'):
        item = json.loads(target.read_text())
        key = (item['case_id'], item['seed'], item['candidate_sha256'], item.get('audit_mode'))
        old_index.setdefault(key, []).append((target.parent, item))
    historical_frame_matches = 0

    def check_call(folder, expected_prompt, expected_hashes, schema, seed):
        request = json.loads((folder / 'request.json').read_text())
        response = json.loads((folder / 'response.json').read_text())
        parsed = json.loads((folder / 'value.json').read_text())
        if response['model'] != protocol['auxiliary']['served_model'] or request['model'] != response['model']:
            raise ValueError('Model identity mismatch')
        if request['temperature'] != 0 or request['seed'] != seed or request['max_tokens'] != 2048:
            raise ValueError('Inference settings changed')
        if request['chat_template_kwargs'] != {'enable_thinking': False}:
            raise ValueError('Inference policy changed')
        if response['choices'][0]['finish_reason'] != 'stop' or json.loads(response['choices'][0]['message']['content']) != parsed:
            raise ValueError('Parsed response is not bound to the complete native reply')
        if json.loads((folder / 'schema.json').read_text()) != schema:
            raise ValueError('Output schema changed')
        messages = request['messages']
        if len(messages) != 1 or messages[0]['role'] != 'user':
            raise ValueError('Unexpected context supplied to the observer')
        content = messages[0]['content']
        if content[0]['text'] != expected_prompt or image_hashes(content) != expected_hashes:
            raise ValueError('Task or image payload changed')
        return parsed

    for index, (case_id, task) in enumerate(tasks.items()):
        row = next(row for row in rows if row['case_id'] == case_id)
        prompt = TASK_PROMPT + '\nFROZEN TEMPLATES:\n' + json.dumps(TEMPLATES) + '\nLITERAL TASK:\n' + row['instruction']
        value = check_call(calls[index], prompt, [row['initial_sha256']], TASK_SCHEMA, experiment['seed'])
        if value != task['value'] or validate_task_diagnosis(value, row['instruction']) != task['validated']:
            raise ValueError('Task validation record drifted')
        checked_images += 1
    for index, (row, record) in enumerate(zip(rows, records)):
        if (row['case_id'], row['seed']) != (record['case_id'], record['seed']):
            raise ValueError('Output identity mismatch')
        folder = root / 'diagnosis/cases' / f'{index:03d}'
        sampling_file = folder / 'replay/uniform/sampling.json'
        sampling = json.loads(sampling_file.read_text())
        if sha(sampling_file) != record['sampling_sha256'] or sampling['video_sha256'] != row['base_sha256']:
            raise ValueError('Sampling is not bound to the original video')
        frame_paths = sorted((folder / 'replay/uniform').glob('*.jpg'))
        frame_hashes = [sha(target) for target in frame_paths]
        if len(frame_paths) != 16 or frame_hashes != sampling['frame_hashes']:
            raise ValueError('Sampled frame bytes changed')
        observed = check_call(calls[20 + index], JUDGE + row['instruction'],
                              [row['initial_sha256']] + frame_hashes, GATE_SCHEMA, row['seed'])
        if observed != record['replay_audit']:
            raise ValueError('Observed gates changed during analysis')
        old = inherited[(row['case_id'], row['seed'])]['base_audit']
        old_record = inherited[(row['case_id'], row['seed'])]
        old_key = (row['case_id'], row['seed'], old_record['candidate_sha256'], 'uniform')
        old_matches = [(directory, item) for directory, item in old_index.get(old_key, [])
                       if item['base_audit'] == old]
        if len(old_matches) != 1:
            raise ValueError('Historical audit lacks a unique rollout binding')
        old_frames = old_matches[0][0] / 'base-audit/uniform'
        old_sampling = json.loads((old_frames / 'sampling.json').read_text())
        if old_sampling['video_sha256'] != row['base_sha256'] or old_sampling['indices'] != sampling['indices']:
            raise ValueError('Historical observer used a different video or temporal sample')
        if old_sampling['frame_hashes'] != frame_hashes or [sha(target) for target in sorted(old_frames.glob('*.jpg'))] != frame_hashes:
            raise ValueError('Historical and replay frame bytes differ')
        historical_frame_matches += 1
        expected = diagnose_input(old, observed, tasks[row['case_id']]['validated']['template_applicability'])
        if expected != record['diagnosis']:
            raise ValueError('Failure comparison changed')
        checked_images += 17
    if protocol['native_pools'] or list((root / 'score-queue').glob('*.request.json')):
        raise ValueError('Diagnostic must not start generation or official scoring')
    output = root / 'verification.json'
    if output.exists():
        raise FileExistsError('Do not overwrite a verification receipt')
    result = {'status': 'PASS', 'source_sha256': sha(__file__), 'run_id': experiment['run_id'],
              'summary_sha256': sha(root / 'diagnosis/summary.json'), 'verified_model_calls': 80,
              'verified_image_payloads': checked_images, 'verified_task_records': 20,
              'verified_gate_records': 300, 'all_original_verdicts_preserved': True,
              'historical_rollout_and_frame_bindings_verified': historical_frame_matches,
              'historical_and_replay_frame_bytes_identical': historical_frame_matches == 60,
              'observer_prompt_and_schema_unchanged': True, 'new_generation_calls': 0,
              'official_metric_requests': 0, 'population_accuracy_established': False,
              'note': 'Artifact verification, not independent human or semantic ground truth'}
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--record-decision', action='store_true')
    args = parser.parse_args()
    if args.record_decision:
        record_decision(args.root)
    else:
        verify(args.root)