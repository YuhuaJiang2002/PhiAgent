"""Remote method-side diagnosis; no human review, evaluator change or generation."""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback


PARENT_RUN = '/opt/phiagent/runs/ti2v-relational-repair/20260918T060617Z'
HOSTNAME = 'yxys-node-214-41-3-2'
FILES = (
    'integrations/skilladam_ti2v/__init__.py',
    'integrations/skilladam_ti2v/adapter.py',
    'integrations/skilladam_ti2v/backend.py',
    'integrations/skilladam_ti2v/residual_repair.py',
    'integrations/skilladam_ti2v/relational_repair.py',
    'integrations/skilladam_ti2v/repair_diagnostics.py',
    'scripts/run_ti2v_repair_diagnostics.py',
    'tests/test_ti2v_relational_repair.py',
    'configs/ti2v_method_scope_v2.json',
)


def sha(target):
    return hashlib.sha256(Path(target).read_bytes()).hexdigest()


def save(target, value):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(target.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False))
    pending.replace(target)


def summarize(records, tasks):
    gates = {}
    for record in records:
        for gate, comparison in record['diagnosis']['gates'].items():
            entry = gates.setdefault(gate, {'transitions': Counter(), 'diagnoses': Counter(),
                                           'original_witnesses': Counter(), 'replay_witnesses': Counter()})
            entry['transitions'][comparison['original_status'] + '->' + comparison['replay_status']] += 1
            entry['diagnoses'][comparison['diagnosis']] += 1
            entry['original_witnesses'][comparison['original_witness']['status']] += 1
            entry['replay_witnesses'][comparison['replay_witness']['status']] += 1
    cases = {}
    for case_id, task in tasks.items():
        members = [record for record in records if record['case_id'] == case_id]
        cases[case_id] = {
            'action_kind': task['value']['action_kind'],
            'object_identifiable': task['value']['object_identifiable'],
            'template_applicability': task['validated']['template_applicability'],
            'missing_dedicated_templates': task['validated']['missing_dedicated_templates'],
            'task_contradictions': task['validated']['contradictions'],
            'task_binding_errors': task['validated']['binding_errors'],
            'original_failure_outputs': sum(any(row['original_status'] == 'FAIL'
                                              for row in member['diagnosis']['gates'].values()) for member in members),
            'replay_failure_outputs': sum(any(row['replay_status'] == 'FAIL'
                                            for row in member['diagnosis']['gates'].values()) for member in members),
        }
    return {
        'outputs': len(records), 'cases': len(tasks),
        'gate_counts': {gate: {name: dict(counts) for name, counts in entry.items()} for gate, entry in gates.items()},
        'action_case_counts': dict(Counter(task['value']['action_kind'] for task in tasks.values())),
        'template_case_counts': {name: dict(Counter(task['validated']['template_applicability'][name]
                                                  for task in tasks.values()))
                                 for name in next(iter(tasks.values()))['validated']['template_applicability']},
        'missing_template_case_counts': dict(Counter(name for task in tasks.values()
                                                     for name in task['validated']['missing_dedicated_templates'])),
        'invalid_task_binding_cases': [case_id for case_id, task in tasks.items()
                           if not task['validated']['task_binding_valid']],
        'case_diagnostics': cases,
        'diagnostic_decision': 'KEEP_GENERATION_AND_RSI_OFF; use evidence reproducibility and action coverage to define a separate next study',
        'original_verdicts_changed': False, 'official_metrics_computed': False,
        'human_review_performed': False, 'population_accuracy_established': False,
        'repeated_judge_is_independent_ground_truth': False,
    }


def run_remote(root):
    if socket.gethostname() != HOSTNAME or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Authorized remote CPU controller required')
    sys.path.insert(0, str(root / 'source'))
    from integrations.skilladam_ti2v.backend import TI2VBackend
    from integrations.skilladam_ti2v.relational_repair import TEMPLATES
    from integrations.skilladam_ti2v.repair_diagnostics import (
        TASK_PROMPT, TASK_SCHEMA, diagnose_input, validate_task_diagnosis,
    )
    for relative, expected in json.loads((root / 'source-manifest.json').read_text()).items():
        if sha(root / relative) != expected:
            raise ValueError('Frozen source changed: ' + relative)
    experiment = json.loads((root / 'experiment.json').read_text())
    rows = json.loads((root / 'inputs.json').read_text())['records']
    parents = json.loads((root / 'parent-selections.json').read_text())['records']
    original = {(row['case_id'], row['seed']): row for row in parents}
    if len(rows) != 60 or len({(row['case_id'], row['seed']) for row in rows}) != 60:
        raise ValueError('Require all sixty frozen inputs')
    grouped = {}
    for row in rows:
        grouped.setdefault(row['case_id'], []).append(row)
        if sha(row['initial']) != row['initial_sha256'] or sha(row['base']) != row['base_sha256']:
            raise ValueError('Input bytes changed')
    if len(grouped) != 20 or any(len(group) != 3 for group in grouped.values()):
        raise ValueError('Require twenty complete three-seed cases')
    if set(original) != {(row['case_id'], row['seed']) for row in rows}:
        raise ValueError('Original audit coverage changed')
    output = root / 'diagnosis'
    output.mkdir(exist_ok=False)
    backend = TI2VBackend(root, 'diagnostic_model')
    if experiment.get('recovery_parent_started_at'):
        backend.deadline = time.monotonic() + max(0, experiment['recovery_parent_started_at'] + 3600 - time.time())
    state = {'status': 'RUNNING', 'completed_tasks': 0, 'completed_outputs': 0,
             'started_at': time.time(), 'hostname': socket.gethostname(), 'argv': sys.argv,
             'source_manifest_sha256': sha(root / 'source-manifest.json')}

    def update():
        state['updated_at'] = time.time()
        save(output / 'state.json', state)
        print(json.dumps({name: state[name] for name in ('status', 'completed_tasks', 'completed_outputs')}), flush=True)

    try:
        update()
        tasks = {}
        for index, (case_id, group) in enumerate(grouped.items()):
            if time.monotonic() >= backend.deadline:
                raise TimeoutError('Diagnostic budget expired')
            row = group[0]
            if any(member['instruction'] != row['instruction'] or member['initial_sha256'] != row['initial_sha256'] for member in group):
                raise ValueError('Case-level task or initial image differs by seed')
            folder = output / 'tasks' / f'{index:03d}'
            folder.mkdir(parents=True, exist_ok=False)
            prompt = TASK_PROMPT + '\nFROZEN TEMPLATES:\n' + json.dumps(TEMPLATES) + '\nLITERAL TASK:\n' + row['instruction']
            if index == 0 and experiment.get('recovery_parent'):
                cached = root / 'retained-first-call'
                request = json.loads((cached / 'request.json').read_text())
                schema = json.loads((cached / 'schema.json').read_text())
                content = request['messages'][0]['content']
                if schema != TASK_SCHEMA or content[0]['text'] != prompt or request['seed'] != experiment['seed']:
                    raise ValueError('Recovery would change the cached model request')
                if request['model'] != backend.cfg['auxiliary']['served_model']:
                    raise ValueError('Recovery model identity differs')
                images = [item['image_url']['url'] for item in content if item['type'] == 'image_url']
                if len(images) != 1 or hashlib.sha256(base64.b64decode(images[0].split(',', 1)[1])).hexdigest() != row['initial_sha256']:
                    raise ValueError('Recovery initial-image payload differs')
                value = json.loads((cached / 'value.json').read_text())
                usage = json.loads((cached / 'usage.json').read_text())
                save(folder / 'request-reuse.json', {'parent': experiment['recovery_parent'],
                     'request_sha256': sha(cached / 'request.json'), 'value_sha256': sha(cached / 'value.json'),
                     'new_model_requests': 0, 'prior_cost_retained': True})
            else:
                value, usage = backend.visual_query(prompt, [row['initial']], TASK_SCHEMA,
                                                    'template-applicability', experiment['seed'])
            validated = validate_task_diagnosis(value, row['instruction'])
            tasks[case_id] = {'case_id': case_id, 'instruction': row['instruction'],
                              'initial_sha256': row['initial_sha256'], 'value': value,
                              'validated': validated, 'usage': usage}
            save(folder / 'task.json', tasks[case_id])
            state['completed_tasks'] = len(tasks)
            update()
        save(output / 'task-applicability-frozen.json', tasks)
        records = []
        for index, row in enumerate(rows):
            if time.monotonic() >= backend.deadline:
                raise TimeoutError('Diagnostic budget expired')
            folder = output / 'cases' / f'{index:03d}'
            folder.mkdir(parents=True, exist_ok=False)
            observed, usage = backend.audit(row, row['base'], folder / 'replay')
            sampling = json.loads((folder / 'replay/uniform/sampling.json').read_text())
            if sampling['video_sha256'] != row['base_sha256'] or len(sampling['frame_hashes']) != 16:
                raise ValueError('Video/frame binding mismatch')
            positions = sampling['indices']
            if positions[0] != 0 or any(left >= right for left, right in zip(positions, positions[1:])):
                raise ValueError('Frame ordering mismatch')
            task = tasks[row['case_id']]['validated']
            record = {'case_id': row['case_id'], 'seed': row['seed'],
                      'initial_sha256': row['initial_sha256'], 'video_sha256': row['base_sha256'],
                      'sampling_sha256': sha(folder / 'replay/uniform/sampling.json'),
                      'task_diagnosis_contradictions': task['contradictions'],
                      'diagnosis': diagnose_input(original[(row['case_id'], row['seed'])]['base_audit'],
                                                 observed, task['template_applicability']),
                      'replay_audit': observed, 'usage': usage}
            save(folder / 'diagnostic.json', record)
            records.append(record)
            state['completed_outputs'] = len(records)
            update()
        save(output / 'records.json', records)
        summary = summarize(records, tasks)
        calls = sorted((backend.out / 'calls').glob('*/usage.json'))
        inherited_calls = 1 if experiment.get('recovery_parent') else 0
        if inherited_calls:
            calls.append(root / 'retained-first-call/usage.json')
        tokens = Counter()
        for target in calls:
            usage = json.loads(target.read_text())
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                if isinstance(usage.get(key), int):
                    tokens[key] += usage[key]
        if backend.state['auxiliary_calls'] + inherited_calls != 80 or len(calls) != 80 or backend.state['native_calls'] != 0:
            raise ValueError('Diagnostic call ledger is incomplete or exceeds protocol')
        summary.update(run_id=experiment['run_id'],
                       protocol_sha256=sha(root / 'protocol.json'),
                       experiment_sha256=sha(root / 'experiment.json'),
                       records_sha256=sha(output / 'records.json'),
                       tasks_sha256=sha(output / 'task-applicability-frozen.json'),
                       input_sha256=sha(root / 'inputs.json'),
                       source_audit_sha256=sha(root / 'parent-selections.json'),
                         cost={'auxiliary_calls': 80, 'new_calls_in_recovery_run': backend.state['auxiliary_calls'],
                             'retained_parent_calls': inherited_calls, 'usage_receipts': len(calls), 'tokens': dict(tokens),
                             'native_calls': 0, 'official_score_calls': 0, 'seconds': time.time() - state['started_at']})
        save(output / 'summary.json', summary)
        state.update(status='DIAGNOSTIC_COMPLETE', summary_sha256=sha(output / 'summary.json'), finished_at=time.time())
    except BaseException as error:
        state.update(status='BLOCKED', error=f'{type(error).__name__}: {error}', finished_at=time.time())
        (output / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        update()


def prepare_remote(root):
    if socket.gethostname() != HOSTNAME or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Remote preparation requires authorized hostname and empty CUDA visibility')
    for relative, expected in json.loads((root / 'package-sha256.json').read_text()).items():
        if sha(root / relative) != expected:
            raise ValueError('Transferred package changed')
    scope = json.loads((root / 'source/configs/ti2v_method_scope_v2.json').read_text())
    if scope['human_review']['enabled'] or scope['evaluator_development']['active_in_this_task']:
        raise ValueError('Unexpected method/evaluator responsibility boundary')
    parent = Path(PARENT_RUN)
    protocol = json.loads((parent / 'protocol.json').read_text())
    experiment = json.loads((root / 'experiment.json').read_text())
    if sha(parent / 'source/integrations/skilladam_ti2v/backend.py') != sha(root / 'source/integrations/skilladam_ti2v/backend.py'):
        raise ValueError('Original observer, transport or selector changed')
    inherited_calls = 0
    if experiment.get('recovery_parent'):
        previous = Path(experiment['recovery_parent'])
        previous_state = json.loads((previous / 'diagnosis/state.json').read_text())
        if previous_state['status'] != 'BLOCKED' or previous_state['completed_tasks'] != 0:
            raise ValueError('Unexpected recovery parent state')
        call_folders = list((previous / 'diagnostic_model/execution/calls').glob('*'))
        if len(call_folders) != 1 or not (call_folders[0] / 'usage.json').is_file():
            raise ValueError('Recovery requires exactly one completed parent call')
        shutil.copytree(call_folders[0], root / 'retained-first-call')
        save(root / 'recovery-binding.json', {'parent_state': previous_state,
             'parent_state_sha256': sha(previous / 'diagnosis/state.json'),
             'retained_call_files': {target.name: sha(target) for target in call_folders[0].iterdir() if target.is_file()},
             'policy': 'Record invalid evidence without ending the population audit; no model request retry'})
        inherited_calls = 1
    protocol.update(max_seconds=3600, max_native_calls_per_method=0,
                    max_auxiliary_calls_per_method=80 - inherited_calls, native_pools=[])
    save(root / 'protocol.json', protocol)
    for name in ('inputs.json', 'parent-selections.json'):
        shutil.copy2(parent / name, root / name)
    (root / 'qwen.lock').symlink_to((parent / 'qwen.lock').resolve())
    (root / 'score-queue').mkdir()
    save(root / 'gpu-before.json', {'at': time.time(), 'controller_CUDA_VISIBLE_DEVICES': '',
         'inventory': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)})
    (root / 'packages.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    files = [target for target in root.rglob('*') if target.is_file() and target.name != 'qwen.lock']
    save(root / 'source-manifest.json', {str(target.relative_to(root)): sha(target) for target in files})
    command = [sys.executable, str(root / 'source/scripts/run_ti2v_repair_diagnostics.py'), '--run-remote', '--root', str(root)]
    with (root / 'controller.log').open('x') as log:
        process = subprocess.Popen(command, cwd=root, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch = {'pid': process.pid, 'command': command, 'hostname': socket.gethostname(), 'at': time.time(),
              'source_manifest_sha256': sha(root / 'source-manifest.json')}
    save(root / 'launch.json', launch)
    print(json.dumps(launch))


def submit(repo, control_path, recover_from=None):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    local = repo / 'outputs/ti2v-repair-diagnostics' / run_id
    remote = Path('/opt/phiagent/runs/ti2v-repair-diagnostics') / run_id
    local.mkdir(parents=True, exist_ok=False)
    for relative in FILES:
        target = local / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, target)
    experiment = {'run_id': run_id, 'parent_run': PARENT_RUN, 'seed': 20260918,
                  'max_seconds': 3600, 'max_auxiliary_calls': 80,
                  'unique_initial_task_queries': 20, 'unchanged_observer_replays': 60,
                  'max_native_calls': 0, 'max_official_score_calls': 0,
                  'purpose': 'Separate failure repeatability, witness usability and template applicability',
                  'no_retries': True, 'no_rsi': True, 'no_human_review': True,
                  'evaluator_development': False, 'no_historical_verdict_replacement': True,
                  'inputs': 'All same twenty opened cases and three seeds; not unseen confirmation',
                  'validity_boundary': 'Same-model repeatability is not independent calibration or population accuracy',
                  'action_catalog': 'Task-level method coverage only; not new metric, supervision or executable repair',
                  'decision_rule': 'No automatic repair or generation launch; freeze diagnostic and choose separate follow-up only after integrity verification'}
    if recover_from:
        previous_locations = json.loads((recover_from / 'locations.json').read_text())
        previous_state = json.loads((recover_from / 'collected/diagnosis/state.json').read_text())
        if previous_state['status'] != 'BLOCKED' or previous_state['completed_tasks'] != 0:
            raise ValueError('Recovery is limited to the first-call validation failure')
        experiment.update(recovery_parent=previous_locations['h2_root'],
                          recovery_parent_started_at=previous_state['started_at'],
                          recovery_reason='Invalid literal-task quote is retained as an invalid observation, not accepted evidence; continue without repeating the completed request')
    save(local / 'experiment.json', experiment)
    save(local / 'git-state.json', {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
         'status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True),
         'source_sha256': {relative: sha(repo / relative) for relative in FILES}})
    binding_code = '''import json,subprocess,time
from pathlib import Path
process=Path('/proc/1580945')
arguments=process.joinpath('cmdline').read_bytes().split(b'\\0')
environment=process.joinpath('environ').read_bytes().split(b'\\0')
cuda=next(item.split(b'=',1)[1].decode() for item in environment if item.startswith(b'CUDA_VISIBLE_DEVICES='))
assert cuda=='2' and b'qwen38-27b-robotics-framework-v1' in arguments
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
assert any(line.startswith('2, GPU-3105bf40-69e9-3577-57d6-dc389411707b,') for line in inventory.splitlines())
print(json.dumps({'at':time.time(),'service_pid':1580945,'CUDA_VISIBLE_DEVICES':cuda,'physical_uuid':'GPU-3105bf40-69e9-3577-57d6-dc389411707b','inventory':inventory,'served_model':'qwen38-27b-robotics-framework-v1'}))
'''
    binding = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'h200-1', 'python3 -'],
                             input=binding_code, capture_output=True, text=True, check=True, timeout=120)
    save(local / 'auxiliary-binding.json', json.loads(binding.stdout))
    save(local / 'package-sha256.json', {str(target.relative_to(local)): sha(target) for target in local.rglob('*') if target.is_file()})
    ssh_options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20']
    if control_path:
        ssh_options += ['-S', str(control_path)]
    command = 'mkdir -p ' + shlex.quote(str(remote.parent)) + ' && mkdir ' + shlex.quote(str(remote)) + ' && tar -xzf - -C ' + shlex.quote(str(remote))
    with subprocess.Popen(['tar', '--no-xattrs', '-czf', '-', '-C', str(local), '.'], stdout=subprocess.PIPE,
                          env={**os.environ, 'COPYFILE_DISABLE': '1'}) as archive:
        subprocess.run(['ssh', *ssh_options, 'h200-2', command], stdin=archive.stdout, check=True, timeout=180)
        archive.stdout.close()
        if archive.wait() != 0:
            raise RuntimeError('Frozen source transfer failed')
    launch_command = 'CUDA_VISIBLE_DEVICES= python3 ' + shlex.quote(str(remote / 'source/scripts/run_ti2v_repair_diagnostics.py')) + ' --prepare-remote --root ' + shlex.quote(str(remote))
    launch = subprocess.run(['ssh', *ssh_options, 'h200-2', launch_command], capture_output=True, text=True, check=True, timeout=180)
    save(local / 'launch.json', json.loads(launch.stdout))
    locations = {'run_id': run_id, 'local_root': str(local), 'h2_root': str(remote),
                 'ssh_control_paths': {'h200-2': str(control_path)} if control_path else {},
                 'status': 'REMOTE_DIAGNOSIS_LAUNCHED'}
    save(local / 'locations.json', locations)
    save(local.parent / 'CURRENT.json', locations)
    print(json.dumps(locations, indent=2))


def collect(locations_file):
    locations = json.loads(locations_file.read_text())
    local = Path(locations['local_root']) / 'collected'
    local.mkdir(exist_ok=True)
    options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20']
    control_path = locations.get('ssh_control_paths', {}).get('h200-2')
    if control_path:
        options += ['-S', control_path]
    code = '''import base64,hashlib,json
from pathlib import Path
root=Path(ROOT)
paths=[root/name for name in ['experiment.json','protocol.json','source-manifest.json','launch.json','auxiliary-binding.json','recovery-binding.json','verification.json','followup-decision.json','diagnosis/state.json','diagnosis/summary.json','diagnosis/records.json','diagnosis/task-applicability-frozen.json','diagnostic_model/execution/state.json']]
paths+=list((root/'diagnosis/tasks').glob('*/task.json'))
out={}
for target in paths:
 if target.is_file():
  content=target.read_bytes()
  out[str(target.relative_to(root))]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':base64.b64encode(content).decode()}
print(json.dumps(out))
'''.replace('ROOT', repr(locations['h2_root']))
    response = subprocess.run(['ssh', *options, 'h200-2', 'python3 -'], input=code, capture_output=True,
                              text=True, check=True, timeout=120)
    hashes = {}
    for relative, record in json.loads(response.stdout).items():
        target = local / relative
        if not target.resolve().is_relative_to(local.resolve()):
            raise ValueError('Collection escaped its run directory')
        content = base64.b64decode(record['bytes'], validate=True)
        if hashlib.sha256(content).hexdigest() != record['sha256']:
            raise ValueError('Collection byte mismatch')
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_suffix(target.suffix + '.pending')
        pending.write_bytes(content)
        pending.replace(target)
        hashes[relative] = record['sha256']
    save(local.parent / 'collection.json', {'at': time.time(), 'exact_remote_hashes': hashes})
    state = json.loads((local / 'diagnosis/state.json').read_text())
    print(json.dumps(state, indent=2))
    if state['status'] == 'BLOCKED':
        raise RuntimeError(state['error'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    parser.add_argument('--prepare-remote', action='store_true')
    parser.add_argument('--run-remote', action='store_true')
    parser.add_argument('--collect', type=Path)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--ssh-control-path', type=Path)
    parser.add_argument('--recover-from', type=Path)
    args = parser.parse_args()
    if args.prepare_remote:
        prepare_remote(args.root)
    elif args.run_remote:
        run_remote(args.root)
    elif args.collect:
        collect(args.collect)
    elif args.submit:
        submit(Path(__file__).resolve().parents[1], args.ssh_control_path, args.recover_from)
    else:
        parser.error('Choose one explicit execution mode')


if __name__ == '__main__':
    main()