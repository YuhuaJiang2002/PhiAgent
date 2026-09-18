"""Verify and analyze a completed remote proposal factorial without inference."""
import argparse
import base64
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time


def sha(target):
    return hashlib.sha256(Path(target).read_bytes()).hexdigest()


def factorial_contrasts(values):
    required = {'combined_original', 'factored_original', 'combined_extended', 'factored_extended'}
    if set(values) != required:
        raise ValueError('Require all four paired arm values')
    combined_original = values['combined_original']
    factored_original = values['factored_original']
    combined_extended = values['combined_extended']
    factored_extended = values['factored_extended']
    return {
        'separation_with_original': factored_original - combined_original,
        'separation_with_extended': factored_extended - combined_extended,
        'extension_with_combined': combined_extended - combined_original,
        'extension_with_factored': factored_extended - factored_original,
        'average_separation': ((factored_original - combined_original)
                               + (factored_extended - combined_extended)) / 2,
        'average_extension': ((combined_extended - combined_original)
                              + (factored_extended - factored_original)) / 2,
        'interaction': (factored_extended - combined_extended) - (factored_original - combined_original),
    }


def record_interpretation(root):
    if socket.gethostname() != 'yxys-node-214-41-3-2' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Interpretation is remote CPU-only')
    verification_path = root / 'verification/result.json'
    verification = json.loads(verification_path.read_text())
    summary_path = root / 'proposal-study/summary.json'
    plans_path = root / 'proposal-study/plans-frozen.json'
    if verification['status'] != 'PASS' or verification['summary_sha256'] != sha(summary_path):
        raise ValueError('Require a verified complete factorial')
    if verification['plans_sha256'] != sha(plans_path):
        raise ValueError('Frozen plans changed after verification')
    summary = json.loads(summary_path.read_text())
    records = json.loads(plans_path.read_text())
    rows = json.loads((root / 'inputs.json').read_text())['records']
    diagnostics_root = Path('/opt/phiagent/runs/ti2v-repair-diagnostics/20260918T064838Z')
    task_file = diagnostics_root / 'diagnosis/task-applicability-frozen.json'
    diagnostic_summary = json.loads((diagnostics_root / 'diagnosis/summary.json').read_text())
    if sha(task_file) != diagnostic_summary['tasks_sha256']:
        raise ValueError('Prior task-applicability record changed')
    tasks = json.loads(task_file.read_text())
    arms = tuple(summary['arms'])
    admitted = {arm: {(record['case_id'], record['seed']) for record in records
                      if record['arms'][arm]['status'] == 'APPLIED'} for arm in arms}
    contrasts = {}
    for before, after in (('combined_original', 'factored_original'),
                          ('combined_extended', 'factored_extended'),
                          ('combined_original', 'combined_extended'),
                          ('factored_original', 'factored_extended')):
        gained = sorted(admitted[after] - admitted[before])
        lost = sorted(admitted[before] - admitted[after])
        contrasts[after + '_minus_' + before] = {
            'gained_outputs': len(gained), 'lost_outputs': len(lost),
            'net_admitted_outputs': len(gained) - len(lost),
            'gained_case_seed_identities': gained, 'lost_case_seed_identities': lost,
        }
    conditional = []
    for row, record in zip(rows, records):
        if (row['case_id'], row['seed']) != (record['case_id'], record['seed']):
            raise ValueError('Input and proposal identities differ')
        for arm in arms:
            plan = record['arms'][arm]
            withdrawal = tasks[row['case_id']]['value']['withdrawal_explicitly_requested']
            if (plan['status'] == 'APPLIED' and plan['relation_id'] == 'release_before_withdrawal'
                    and withdrawal != 'YES'):
                conditional.append({'case_id': row['case_id'], 'seed': row['seed'], 'arm': arm,
                    'instruction': row['instruction'], 'prior_withdrawal_required': withdrawal,
                    'plan_sha256': plan['plan_sha256'],
                    'issue': 'The guard admitted a template whose withdrawal condition is not established by the prior task diagnosis'})
    campaign_cost = Counter()
    for cost in verification['cumulative_arm_cost'].values():
        campaign_cost.update(cost)
    result = {
        'run_id': summary['run_id'], 'analysis_type': 'Post-hoc paired-set and conditional-template audit; native admissions unchanged',
        'source_sha256': sha(__file__), 'summary_sha256': sha(summary_path),
        'verification_sha256': sha(verification_path), 'prior_task_diagnosis_sha256': sha(task_file),
        'paired_admission_transitions': contrasts,
        'conditional_template_flags': conditional,
        'conditional_flags_per_arm': {arm: sum(flag['arm'] == arm for flag in conditional) for arm in arms},
        'campaign_cost': dict(campaign_cost),
        'inference': {
            'template_extension': 'Two net additional admitted prompts under each evidence-processing path; gains and losses both occur',
            'factoring': 'Four fewer admitted prompts under either vocabulary; lower count alone does not establish lower precision or quality',
            'interaction': 'Zero on the observed admission-count scale; no equivalence or significance claim',
            'shared_actor_fix': 'First run remained invalid; the four-arm repeat, not between-run improvement, isolates the two intended factors',
        },
        'generation': {
            'status': 'NOT_STARTED', 'primary_candidate': 'combined_extended',
            'candidate_admissions': summary['arms']['combined_extended']['admitted'],
            'candidate_case_count': summary['arms']['combined_extended']['admitted_cases'],
            'suggested_controls': ['unchanged_parent', 'factored_extended'],
            'reason': 'The extended-vocabulary arms supply changed prompts without the observed withdrawal-template conflict; quality remains unmeasured',
            'boundary': 'Freeze a separate paired-generation protocol with all sixty inputs, unchanged gates/selector/scorer, full raw and selected results, and exact-prompt reuse before any generation',
            'human_or_evaluator_work_required_in_this_task': False,
        },
        'new_model_calls': 0, 'historical_statuses_changed': False,
        'quality_or_physical_validity_established': False,
    }
    target = root / 'verification/interpretation.json'
    if target.exists():
        raise FileExistsError('Do not overwrite a prior interpretation')
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    compact = {key: value for key, value in result.items() if key != 'conditional_template_flags'}
    print(json.dumps(compact, indent=2, allow_nan=False))


def verify(root):
    if socket.gethostname() != 'yxys-node-214-41-3-2' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('This artifact analysis must run on the authorized remote CPU')
    sys.path.insert(0, str(root / 'source'))
    from integrations.skilladam_ti2v.backend import GATES
    from integrations.skilladam_ti2v.relational_repair import EDIT_SLOT, TEMPLATES
    from integrations.skilladam_ti2v.residual_repair import CONTROL_SUFFIX
    from integrations.skilladam_ti2v.repair_factorial import ARMS, EXTENDED_TEMPLATES, prepare_factorial_repair
    from scripts.run_ti2v_relational_repair import prompt_from_skill, verify_coverage

    state = json.loads((root / 'proposal-study/state.json').read_text())
    summary_path = root / 'proposal-study/summary.json'
    if state['status'] != 'PROPOSAL_STUDY_COMPLETE' or sha(summary_path) != state['summary_sha256']:
        raise ValueError('A complete source-bound experiment is required')
    summary = json.loads(summary_path.read_text())
    experiment = json.loads((root / 'experiment.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    if sha(root / 'protocol.json') != summary['protocol_sha256'] or sha(root / 'experiment.json') != summary['experiment_sha256']:
        raise ValueError('Protocol or experiment changed')
    for relative, expected in json.loads((root / 'source-manifest.json').read_text()).items():
        if sha(root / relative) != expected:
            raise ValueError('Frozen source or input changed: ' + relative)
    frozen_backend = root / 'source/integrations/skilladam_ti2v/backend.py'
    parent = Path(experiment['parent_run'])
    if sha(frozen_backend) != sha(parent / 'source/integrations/skilladam_ti2v/backend.py'):
        raise ValueError('Original observer, selector or transport changed')
    if any(EXTENDED_TEMPLATES[name] != value for name, value in TEMPLATES.items()):
        raise ValueError('Extended vocabulary modified an original template')
    if set(EXTENDED_TEMPLATES) - set(TEMPLATES) != {
            'support_before_release', 'receiver_grasp_before_giver_release', 'maintain_contact_during_push'}:
        raise ValueError('Unexpected expanded vocabulary')
    rows = json.loads((root / 'inputs.json').read_text())['records']
    verify_coverage(rows)
    original = {(item['case_id'], item['seed']): item for item in
                json.loads((root / 'parent-selections.json').read_text())['records']}
    plans_path = root / 'proposal-study/plans-frozen.json'
    if sha(plans_path) != summary['plans_sha256']:
        raise ValueError('Frozen plan manifest changed')
    records = json.loads(plans_path.read_text())
    if len(records) != len(rows):
        raise ValueError('Plan population differs')
    output = root / 'verification'
    output.mkdir(exist_ok=False)
    calls = iter(sorted((root / 'factorial/execution/calls').iterdir()))
    call_bindings = []
    arm_cost = {arm: Counter() for arm in ARMS}
    stage_counts = {arm: {name: Counter() for name in (
        'first_binding', 'second_binding', 'first_failure_status', 'second_failure_status',
        'first_gate_decisions', 'second_gate_decisions', 'invalid_errors')} for arm in ARMS}
    apply_intent_inputs = {arm: 0 for arm in ARMS}
    invalid_or_unbound = Counter()
    started_at = time.time()

    class RecordedBackend:
        def __init__(self, arm):
            self.arm = arm

        def visual_query(self, prompt, images, schema, stage, seed, positions):
            folder = next(calls)
            if not folder.name.endswith('-' + stage):
                raise ValueError('Call order or arm identity mismatch')
            request = json.loads((folder / 'request.json').read_text())
            response = json.loads((folder / 'response.json').read_text())
            value = json.loads((folder / 'value.json').read_text())
            usage = json.loads((folder / 'usage.json').read_text())
            if json.loads((folder / 'schema.json').read_text()) != schema:
                raise ValueError('Schema or vocabulary differs from frozen arm')
            if request['model'] != protocol['auxiliary']['served_model'] or response['model'] != request['model']:
                raise ValueError('Model identity differs')
            if request['seed'] != seed or request['temperature'] != 0 or request['max_tokens'] != 2048:
                raise ValueError('Call budget or sampling settings changed')
            if request['chat_template_kwargs'] != {'enable_thinking': False}:
                raise ValueError('Thinking policy changed')
            if response['choices'][0]['finish_reason'] != 'stop':
                raise ValueError('Incomplete native response')
            if json.loads(response['choices'][0]['message']['content']) != value or response['usage'] != usage:
                raise ValueError('Parsed result or usage is not bound to native response')
            messages = request['messages']
            if len(messages) != 1 or messages[0]['role'] != 'user':
                raise ValueError('Unexpected hidden context')
            content = messages[0]['content']
            if content[0] != {'type': 'text', 'text': prompt} or len(content) != 1 + 2 * len(images):
                raise ValueError('Actual proposal context differs from the frozen prompt')
            for index, image in enumerate(images):
                label = 'Image0 REAL INITIAL' if index == 0 else f'Image{index} GENERATED normalized_time={positions[index-1]:.5f}'
                if content[1 + 2 * index] != {'type': 'text', 'text': label}:
                    raise ValueError('Image ordering or temporal label mismatch')
                url = content[2 + 2 * index]['image_url']['url']
                if not url.startswith(('data:image/png;base64,', 'data:image/jpeg;base64,')):
                    raise ValueError('External image reference is not allowed')
                actual = base64.b64decode(url.split(',', 1)[1], validate=True)
                if hashlib.sha256(actual).hexdigest() != sha(image):
                    raise ValueError('Image bytes differ from declared input')
            arm_cost[self.arm]['calls'] += 1
            arm_cost[self.arm].update({key: usage[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')})
            call_bindings.append({'call': folder.name, 'arm': self.arm,
                                  'request_sha256': sha(folder / 'request.json'),
                                  'response_sha256': sha(folder / 'response.json'),
                                  'image_payloads': len(images), 'usage': usage})
            return value, usage

    skill = CONTROL_SUFFIX.strip() + '\n' + EDIT_SLOT
    matched_historical_frames = 0
    old_rollouts = Path('/opt/phiagent/runs/ti2v-skill-optimization/20260916T032105Z/recovery-2/phiagent/execution/rollouts')
    old_index = {}
    for target in old_rollouts.glob('*/selection.json'):
        item = json.loads(target.read_text())
        key = (item['case_id'], item['seed'], item['candidate_sha256'], item.get('audit_mode'))
        old_index.setdefault(key, []).append((target.parent, item))
    no_source_failure = 0
    admitted_records = []
    with tempfile.TemporaryDirectory(dir=output) as replay_directory:
        for index, (row, record) in enumerate(zip(rows, records)):
            if (row['case_id'], row['seed']) != (record['case_id'], record['seed']) or set(record['arms']) != set(ARMS):
                raise ValueError('Case-seed pairing or arm coverage differs')
            folder = root / 'proposal-study/cases' / f'{index:03d}'
            if sha(folder / 'frames/sampling.json') != record['sampling_sha256']:
                raise ValueError('Sampling manifest changed')
            sample = json.loads((folder / 'frames/sampling.json').read_text())
            images = sorted((folder / 'frames').glob('*.jpg'))
            if len(images) != 16 or [sha(image) for image in images] != sample['frame_hashes']:
                raise ValueError('Uniform sampled frame bytes changed')
            if sha(row['base']) != row['base_sha256'] or sample['video_sha256'] != row['base_sha256'] or sha(row['initial']) != row['initial_sha256']:
                raise ValueError('Input media binding changed')
            frame_count = int(sample['probe']['streams'][0]['nb_read_frames'])
            expected_indices = [round(position * (frame_count - 1) / 15) for position in range(16)]
            if sample['indices'] != expected_indices:
                raise ValueError('Uniform observation schedule changed')
            positions = [position / (frame_count - 1) for position in sample['indices']]
            parent_record = original[(row['case_id'], row['seed'])]
            audit = parent_record['base_audit']
            before = deepcopy(audit)
            if not any(audit['gates'][gate]['status'] == 'FAIL' for gate in GATES):
                no_source_failure += 1
            old_key = (row['case_id'], row['seed'], parent_record['candidate_sha256'], 'uniform')
            matches = [directory for directory, item in old_index.get(old_key, []) if item['base_audit'] == audit]
            if len(matches) != 1:
                raise ValueError('Original audit is not uniquely bound to its rollout')
            historical = json.loads((matches[0] / 'base-audit/uniform/sampling.json').read_text())
            if any(sample[key] != historical[key] for key in ('video_sha256', 'indices', 'frame_hashes')):
                raise ValueError('Historical and factorial observer inputs differ')
            matched_historical_frames += 1
            order = ARMS[index % 4:] + ARMS[:index % 4]
            for arm in order:
                plan_path = folder / arm / 'plan.json'
                if sha(plan_path) != record['arms'][arm]['plan_sha256']:
                    raise ValueError('Individual proposal changed')
                saved = json.loads(plan_path.read_text())
                replay = prepare_factorial_repair(RecordedBackend(arm), row, audit, skill,
                    Path(replay_directory) / f'{index:03d}' / arm, arm, images, positions)
                if replay != saved:
                    raise ValueError('Saved proposal does not replay under its frozen controller')
                if saved.get('error'):
                    stage_counts[arm]['invalid_errors'][saved['error']] += 1
                for stage, prefix in (('first_stage', 'first'), ('second_stage', 'second')):
                    stage_value = saved.get(stage)
                    if stage_value:
                        stage_counts[arm][prefix + '_binding'][stage_value['binding_status']] += 1
                        for assessment in stage_value['assessments']:
                            stage_counts[arm][prefix + '_failure_status'][assessment['failure_visible']] += 1
                            if 'decision' in assessment:
                                stage_counts[arm][prefix + '_gate_decisions'][assessment['decision']] += 1
                apply_intent_inputs[arm] += any(entry['decision'] == 'APPLY'
                    for entry in saved.get('second_stage', {}).get('assessments', []))
                prompt = prompt_from_skill(row['instruction'], replay['skill'])
                if prompt != record['arms'][arm]['prompt'] or replay['status'] != record['arms'][arm]['status']:
                    raise ValueError('Final prompt or admission status changed')
                if replay['status'] == 'APPLIED':
                    if prompt == record['parent_prompt'] or any(check['status'] != 'PASS' for check in replay['guard'].values()):
                        raise ValueError('Admitted proposal lacks a changed prompt or passing guard')
                    admitted_records.append({'case_id': row['case_id'], 'seed': row['seed'], 'arm': arm,
                                             'relation_id': replay['relation_id'], 'plan_sha256': sha(plan_path),
                                             'evidence_frame_ids': replay['evidence_frame_ids']})
                else:
                    invalid_or_unbound[replay['reason']] += 1
            if audit != before:
                raise ValueError('Original five-gate audit was changed')
    if next(calls, None) is not None:
        raise ValueError('Extra model calls outside the paired protocol')
    for arm in ARMS:
        if dict(arm_cost[arm]) != summary['arms'][arm]['cost']:
            raise ValueError('Per-arm cost does not match native receipts')
        if arm_cost[arm]['calls'] > experiment['max_calls_per_arm'] or arm_cost[arm]['total_tokens'] > experiment['max_tokens_per_arm']:
            raise ValueError('Per-arm budget exceeded')
        admitted = [record for record in records if record['arms'][arm]['status'] == 'APPLIED']
        if len(admitted) != summary['arms'][arm]['admitted'] or len({record['case_id'] for record in admitted}) != summary['arms'][arm]['admitted_cases']:
            raise ValueError('Admission counts changed')
    cumulative_cost = {arm: dict(arm_cost[arm]) for arm in ARMS}
    if experiment.get('repeat_parent'):
        prior_root = Path(experiment['repeat_parent'])
        prior_file = prior_root / 'proposal-study/summary.json'
        if sha(prior_file) != experiment['repeat_parent_summary_sha256']:
            raise ValueError('Repeat parent summary binding changed')
        prior_summary = json.loads(prior_file.read_text())
        for arm in ARMS:
            prior_cost = prior_summary['arms'][arm]['cost']
            if prior_cost != experiment['prior_arm_cost'][arm]:
                raise ValueError('Prior arm cost omitted or altered')
            cumulative_cost[arm] = {key: arm_cost[arm][key] + prior_cost[key]
                                    for key in ('calls', 'prompt_tokens', 'completion_tokens', 'total_tokens')}
            if cumulative_cost[arm] != summary['campaign_arm_cost'][arm]:
                raise ValueError('Reported campaign cost differs from both receipts')
            if cumulative_cost[arm]['calls'] > experiment['max_calls_per_arm'] or cumulative_cost[arm]['total_tokens'] > experiment['max_tokens_per_arm']:
                raise ValueError('Cumulative paired-run budget exceeded')
        if state['finished_at'] > experiment['campaign_started_at'] + experiment['max_seconds']:
            raise ValueError('Original campaign deadline exceeded')
    if protocol['native_pools'] or list((root / 'score-queue').glob('*.request.json')):
        raise ValueError('Proposal study must not dispatch generation or scores')
    counts = {arm: summary['arms'][arm]['admitted'] for arm in ARMS}
    case_contrasts = {}
    for case_id in sorted({row['case_id'] for row in rows}):
        case_counts = {arm: sum(record['arms'][arm]['status'] == 'APPLIED' for record in records
                               if record['case_id'] == case_id) / 3 for arm in ARMS}
        case_contrasts[case_id] = {'admitted_fraction_by_arm': case_counts,
                                 'contrasts': factorial_contrasts(case_counts)}
    result = {
        'status': 'PASS', 'run_id': experiment['run_id'], 'source_sha256': sha(__file__),
        'summary_sha256': sha(summary_path), 'plans_sha256': sha(plans_path),
        'frozen_factorial_sha256': sha(root / 'source/integrations/skilladam_ti2v/repair_factorial.py'),
        'frozen_backend_sha256': sha(frozen_backend), 'model_calls_verified': len(call_bindings),
        'actual_image_payloads_verified': sum(record['image_payloads'] for record in call_bindings),
        'historical_frame_bindings_verified': matched_historical_frames,
        'paired_outputs': 60, 'cases': 20, 'no_reported_source_failure_outputs': no_source_failure,
        'admitted_counts': counts, 'admitted_records': admitted_records,
        'stage_counts': {arm: {name: dict(values) for name, values in stages.items()}
                 for arm, stages in stage_counts.items()},
        'inputs_with_second_stage_apply_intent': apply_intent_inputs,
        'cumulative_arm_cost': cumulative_cost,
        'prior_run': experiment.get('repeat_parent'),
        'descriptive_count_contrasts': factorial_contrasts(counts),
        'paired_case_contrasts': case_contrasts,
        'analysis_status': 'Descriptive paired proposal-admission analysis, not a video-quality result or statistical significance claim',
        'same_model_guard_is_independent_truth': False, 'original_gates_selector_transport_unchanged': True,
        'new_generation_or_official_score_calls': 0,
        'generation_decision': 'Review evidence-bound candidates and freeze a separate protocol; no automatic generation',
        'seconds': time.time() - started_at,
    }
    (output / 'calls.json').write_text(json.dumps(call_bindings, indent=2, allow_nan=False))
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key not in ('paired_case_contrasts', 'admitted_records')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--interpret', action='store_true')
    args = parser.parse_args()
    if args.interpret:
        record_interpretation(args.root)
    else:
        verify(args.root)