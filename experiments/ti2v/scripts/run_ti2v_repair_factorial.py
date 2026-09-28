"""Remote-only frozen two-factor proposal study and exact-byte collection."""
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
import time
import traceback


HOSTNAME = 'yxys-node-214-41-3-2'
PARENT = '/opt/phiagent/runs/ti2v-relational-repair/20260918T060617Z'
FILES = (
    'integrations/skilladam_ti2v/__init__.py',
    'integrations/skilladam_ti2v/adapter.py',
    'integrations/skilladam_ti2v/backend.py',
    'integrations/skilladam_ti2v/residual_repair.py',
    'integrations/skilladam_ti2v/relational_repair.py',
    'integrations/skilladam_ti2v/repair_diagnostics.py',
    'integrations/skilladam_ti2v/repair_factorial.py',
    'scripts/run_ti2v_relational_repair.py',
    'scripts/run_ti2v_repair_factorial.py',
    'scripts/verify_ti2v_repair_diagnostics.py',
    'tests/test_ti2v_relational_repair.py',
    'configs/ti2v_method_scope_v2.json',
)


def sha(target):
    return hashlib.sha256(Path(target).read_bytes()).hexdigest()


def save(target, value):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(target)


def ensure_remote():
    if socket.gethostname() != HOSTNAME or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Authorized remote CPU controller with empty CUDA visibility required')


class MeteredBackend:
    def __init__(self, backend, experiment):
        self.backend = backend
        self.experiment = experiment
        self.arm = None
        self.cost = {}

    def visual_query(self, *arguments):
        entry = self.cost.setdefault(self.arm, {'calls': 0, 'prompt_tokens': 0,
                                                'completion_tokens': 0, 'total_tokens': 0})
        prior = self.experiment.get('prior_arm_cost', {}).get(self.arm, {})
        if entry['calls'] + prior.get('calls', 0) >= self.experiment['max_calls_per_arm']:
            raise RuntimeError('Frozen arm call limit reached')
        if entry['total_tokens'] + prior.get('total_tokens', 0) + 24000 > self.experiment['max_tokens_per_arm']:
            raise RuntimeError('Frozen arm token reserve exhausted')
        if time.monotonic() >= self.backend.deadline:
            raise TimeoutError('Frozen experiment deadline reached')
        entry['calls'] += 1
        value, usage = self.backend.visual_query(*arguments)
        if any(type(usage.get(key)) is not int for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
            raise ValueError('Missing authoritative token usage')
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            entry[key] += usage[key]
        if usage['total_tokens'] > 24000 or entry['total_tokens'] + prior.get('total_tokens', 0) > self.experiment['max_tokens_per_arm']:
            raise RuntimeError('Token reservation exceeded; preserve response and stop')
        return value, usage


def run(root):
    ensure_remote()
    sys.path.insert(0, str(root / 'source'))
    from integrations.skilladam_ti2v.backend import TI2VBackend
    from integrations.skilladam_ti2v.relational_repair import EDIT_SLOT
    from integrations.skilladam_ti2v.residual_repair import CONTROL_SUFFIX
    from integrations.skilladam_ti2v.repair_factorial import ARMS, prepare_factorial_repair
    from scripts.run_ti2v_relational_repair import verify_coverage, prompt_from_skill

    for relative, expected in json.loads((root / 'source-manifest.json').read_text()).items():
        if sha(root / relative) != expected:
            raise ValueError('Frozen input changed: ' + relative)
    experiment = json.loads((root / 'experiment.json').read_text())
    rows = json.loads((root / 'inputs.json').read_text())['records']
    verify_coverage(rows)
    parent = json.loads((root / 'parent-selections.json').read_text())['records']
    parents = {(record['case_id'], record['seed']): record for record in parent}
    if set(parents) != {(row['case_id'], row['seed']) for row in rows}:
        raise ValueError('Source audit coverage differs')
    for row in rows:
        if sha(row['initial']) != row['initial_sha256'] or sha(row['base']) != row['base_sha256']:
            raise ValueError('Source media hash differs')
    output = root / 'proposal-study'
    output.mkdir(exist_ok=False)
    backend = TI2VBackend(root, 'factorial')
    if experiment.get('campaign_started_at'):
        backend.deadline = time.monotonic() + max(0, experiment['campaign_started_at'] + experiment['max_seconds'] - time.time())
    meter = MeteredBackend(backend, experiment)
    state = {'status': 'RUNNING', 'completed_outputs': 0, 'started_at': time.time(),
             'hostname': socket.gethostname(), 'argv': sys.argv,
             'source_manifest_sha256': sha(root / 'source-manifest.json')}

    def update():
        state.update(updated_at=time.time(), cost=meter.cost)
        save(output / 'state.json', state)
        print(json.dumps({key: state[key] for key in ('status', 'completed_outputs')}), flush=True)

    records = []
    try:
        update()
        skill = CONTROL_SUFFIX.strip() + '\n' + EDIT_SLOT
        for index, row in enumerate(rows):
            folder = output / 'cases' / f'{index:03d}'
            folder.mkdir(parents=True, exist_ok=False)
            audit = parents[(row['case_id'], row['seed'])]['base_audit']
            before = json.dumps(audit, sort_keys=True)
            images, positions = backend.frames(row['base'], folder / 'frames')
            record = {'case_id': row['case_id'], 'seed': row['seed'], 'arms': {},
                      'parent_prompt': prompt_from_skill(row['instruction'], skill),
                      'sampling_sha256': sha(folder / 'frames/sampling.json')}
            arm_order = ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)]
            for arm in arm_order:
                meter.arm = arm
                result = prepare_factorial_repair(meter, row, audit, skill, folder / arm,
                                                  arm, images, positions)
                record['arms'][arm] = {
                    'status': result['status'], 'reason': result['reason'],
                    'stage_reached': result['stage_reached'],
                    'relation_id': result.get('relation_id'),
                    'prompt': prompt_from_skill(row['instruction'], result['skill']),
                    'plan_sha256': sha(folder / arm / 'plan.json'),
                    'guard_called': 'guard' in result,
                    'guard_rejections': [name for name, check in result.get('guard', {}).items()
                                         if check['status'] != 'PASS'],
                }
            if before != json.dumps(audit, sort_keys=True):
                raise ValueError('Inherited audit was mutated')
            save(folder / 'record.json', record)
            records.append(record)
            state['completed_outputs'] = len(records)
            update()
        save(output / 'plans-frozen.json', records)
        arm_summary = {}
        for arm in ARMS:
            applied = [record for record in records if record['arms'][arm]['status'] == 'APPLIED']
            arm_summary[arm] = {
                'inputs': len(records), 'admitted': len(applied),
                'admitted_cases': len({record['case_id'] for record in applied}),
                'reasons': dict(Counter(record['arms'][arm]['reason'] for record in records)),
                'stages': dict(Counter(record['arms'][arm]['stage_reached'] for record in records)),
                'guard_calls': sum(record['arms'][arm]['guard_called'] for record in records),
                'guard_rejections': dict(Counter(name for record in records
                                                for name in record['arms'][arm]['guard_rejections'])),
                'relations': dict(Counter(record['arms'][arm]['relation_id'] for record in applied)),
                'changed_prompts': sum(record['arms'][arm]['prompt'] != record['parent_prompt'] for record in records),
                'cost': meter.cost.get(arm, {}),
            }
        receipts = sorted((backend.out / 'calls').glob('*/usage.json'))
        total = Counter()
        for target in receipts:
            for key, value in json.loads(target.read_text()).items():
                if key in ('prompt_tokens', 'completion_tokens', 'total_tokens') and type(value) is int:
                    total[key] += value
        if len(receipts) != backend.state['auxiliary_calls'] or sum(item['calls'] for item in meter.cost.values()) != len(receipts):
            raise ValueError('Usage receipt mismatch')
        if backend.state['native_calls'] or backend.state['score_jobs']:
            raise ValueError('Proposal protocol cannot generate or score')
        summary = {
            'run_id': experiment['run_id'], 'outputs': 60, 'cases': 20, 'arms': arm_summary,
            'scope': 'Same opened development cases, not independent confirmation or quality scores',
            'cost': {'auxiliary_calls': len(receipts), 'usage_receipts': len(receipts), **dict(total),
                     'native_calls': 0, 'official_score_jobs': 0},
            'plans_sha256': sha(output / 'plans-frozen.json'),
            'protocol_sha256': sha(root / 'protocol.json'),
            'experiment_sha256': sha(root / 'experiment.json'),
            'original_audit_changed': False, 'evaluator_changed': False,
            'quality_improvement_established': False, 'human_calibration_performed': False,
            'next_stage': 'SEPARATELY_FREEZE_PAIRED_GENERATION' if any(item['admitted'] for item in arm_summary.values())
                          else 'STOP_NO_ADMISSIBLE_INTERVENTION',
        }
        if experiment.get('repeat_parent'):
            summary['repeat_parent'] = experiment['repeat_parent']
            summary['repeat_parent_summary_sha256'] = experiment['repeat_parent_summary_sha256']
            summary['campaign_arm_cost'] = {
                arm: {key: meter.cost.get(arm, {}).get(key, 0) + experiment['prior_arm_cost'][arm].get(key, 0)
                      for key in ('calls', 'prompt_tokens', 'completion_tokens', 'total_tokens')}
                for arm in ARMS}
        save(output / 'summary.json', summary)
        state.update(status='PROPOSAL_STUDY_COMPLETE', summary_sha256=sha(output / 'summary.json'))
    except BaseException as error:
        state.update(status='BLOCKED', error=f'{type(error).__name__}: {error}')
        (output / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        state['finished_at'] = time.time()
        update()


def prepare(root):
    ensure_remote()
    for relative, expected in json.loads((root / 'package-sha256.json').read_text()).items():
        if sha(root / relative) != expected:
            raise ValueError('Transferred package mismatch')
    parent = Path(PARENT)
    if sha(parent / 'source/integrations/skilladam_ti2v/backend.py') != sha(root / 'source/integrations/skilladam_ti2v/backend.py'):
        raise ValueError('Frozen observer, selector or transport changed')
    experiment = json.loads((root / 'experiment.json').read_text())
    if experiment.get('repeat_parent'):
        repeat_parent = Path(experiment['repeat_parent'])
        parent_summary = repeat_parent / 'proposal-study/summary.json'
        if sha(parent_summary) != experiment['repeat_parent_summary_sha256']:
            raise ValueError('Repeat parent summary changed')
        for name in ('inputs.json', 'parent-selections.json'):
            if sha(repeat_parent / name) != sha(parent / name):
                raise ValueError('Paired repeat inputs changed')
    scope = json.loads((root / 'source/configs/ti2v_method_scope_v2.json').read_text())
    if scope['human_review']['enabled'] or scope['evaluator_development']['active_in_this_task']:
        raise ValueError('Unexpected evaluator responsibility change')
    protocol = json.loads((parent / 'protocol.json').read_text())
    protocol.update(max_seconds=experiment['max_seconds'], max_native_calls_per_method=0,
                    max_auxiliary_calls_per_method=4 * experiment['max_calls_per_arm'], native_pools=[])
    save(root / 'protocol.json', protocol)
    for filename in ('inputs.json', 'parent-selections.json'):
        shutil.copy2(parent / filename, root / filename)
    (root / 'qwen.lock').symlink_to((parent / 'qwen.lock').resolve())
    save(root / 'gpu-before.json', {'at': time.time(), 'CUDA_VISIBLE_DEVICES': '',
         'inventory': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)})
    (root / 'packages.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    save(root / 'source-manifest.json', {str(target.relative_to(root)): sha(target)
         for target in root.rglob('*') if target.is_file() and not target.is_symlink()})
    command = [sys.executable, str(root / 'source/scripts/run_ti2v_repair_factorial.py'), '--run', '--root', str(root)]
    with (root / 'controller.log').open('x') as log:
        process = subprocess.Popen(command, cwd=root, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch = {'pid': process.pid, 'command': command, 'hostname': socket.gethostname(), 'at': time.time(),
              'source_manifest_sha256': sha(root / 'source-manifest.json')}
    save(root / 'launch.json', launch)
    print(json.dumps(launch))


def submit(repo, control_path, repeat_from=None):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    local = repo / 'outputs/ti2v-repair-factorial' / run_id
    remote = Path('/opt/phiagent/runs/ti2v-repair-factorial') / run_id
    local.mkdir(parents=True, exist_ok=False)
    for relative in FILES:
        destination = local / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, destination)
    experiment = {
        'run_id': run_id, 'parent_run': PARENT, 'seed': 20260918,
        'max_seconds': 5400, 'max_calls_per_arm': 180, 'max_tokens_per_arm': 500000,
        'max_native_calls': 0, 'max_official_score_jobs': 0,
        'arms': ['combined_original', 'factored_original', 'combined_extended', 'factored_extended'],
        'primary_outcome': 'Admitted proposal count on all sixty inputs, grouped by twenty cases',
        'factors': ['joint observation/proposal versus separated observation then proposal',
                    'original three relations versus six action-covering relations'],
        'matched_allocation': 'Two predeclared decision stages plus at most one identical grounding guard per eligible input',
        'shared_changes': 'An absent task actor remains explicitly unspecified in every arm; no invented arm enters an edit',
        'second_stage': 'Preserve first-stage binding and failure evidence; no UNKNOWN/FAIL override',
        'no_call_retries': True, 'no_historical_verdict_replacement': True,
        'arm_order': 'Rotate the four arms by frozen input index',
        'generation_rule': 'Never launch generation here; changed admissible prompts require a separately frozen paired study',
        'scope': 'Opened development only; no evaluator development, human review or new RSI level',
    }
    if repeat_from:
        prior_locations = json.loads((repeat_from / 'locations.json').read_text())
        prior_state = json.loads((repeat_from / 'collected/proposal-study/state.json').read_text())
        prior_summary_file = repeat_from / 'collected/proposal-study/summary.json'
        prior_summary = json.loads(prior_summary_file.read_text())
        prior_experiment = json.loads((repeat_from / 'experiment.json').read_text())
        if prior_state['status'] != 'PROPOSAL_STUDY_COMPLETE' or prior_summary['next_stage'] != 'STOP_NO_ADMISSIBLE_INTERVENTION':
            raise ValueError('This interface repeat requires a completed zero-admission parent')
        deadline = prior_state['started_at'] + prior_experiment['max_seconds']
        if time.time() >= deadline:
            raise TimeoutError('Original campaign deadline has passed')
        experiment.update(max_seconds=prior_experiment['max_seconds'],
                          max_calls_per_arm=prior_experiment['max_calls_per_arm'],
                          max_tokens_per_arm=prior_experiment['max_tokens_per_arm'],
                          campaign_started_at=prior_state['started_at'],
                          repeat_parent=prior_locations['h2_root'],
                          repeat_parent_summary_sha256=sha(prior_summary_file),
                          prior_arm_cost={arm: prior_summary['arms'][arm]['cost'] for arm in experiment['arms']},
                          common_interface_revision='Constrain actor quote and redundant binding enum to exact named left/right arm/gripper spans, or both unspecified; all four arms share the same rule',
                          original_frozen_priority_restored=True,
                          interpretation='New within-run factorial after an interface correction; between-run differences are post-hoc and not independent confirmation')
    save(local / 'experiment.json', experiment)
    save(local / 'git-state.json', {
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
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
    save(local / 'package-sha256.json', {str(target.relative_to(local)): sha(target)
                                       for target in local.rglob('*') if target.is_file()})
    options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20']
    if control_path:
        options += ['-S', str(control_path)]
    locations = {'run_id': run_id, 'local_root': str(local), 'h2_root': str(remote),
                 'ssh_control_path': str(control_path) if control_path else None}
    save(local / 'locations.json', locations)
    command = 'mkdir -p ' + shlex.quote(str(remote.parent)) + ' && mkdir ' + shlex.quote(str(remote)) + ' && tar -xzf - -C ' + shlex.quote(str(remote))
    with subprocess.Popen(['tar', '--no-xattrs', '-czf', '-', '-C', str(local), '.'], stdout=subprocess.PIPE,
                          env={**os.environ, 'COPYFILE_DISABLE': '1'}) as archive:
        subprocess.run(['ssh', *options, 'h200-2', command], stdin=archive.stdout, check=True, timeout=180)
        archive.stdout.close()
        if archive.wait() != 0:
            raise RuntimeError('Frozen package transfer failed')
    remote_command = 'CUDA_VISIBLE_DEVICES= python3 ' + shlex.quote(str(remote / 'source/scripts/run_ti2v_repair_factorial.py')) + ' --prepare --root ' + shlex.quote(str(remote))
    launched = subprocess.run(['ssh', *options, 'h200-2', remote_command], capture_output=True, text=True, check=True, timeout=180)
    save(local / 'launch.json', json.loads(launched.stdout))
    save(local.parent / 'CURRENT.json', locations)
    print(json.dumps(locations, indent=2))


def collect(locations_file, control_path=None):
    locations = json.loads(locations_file.read_text())
    local = Path(locations['local_root']) / 'collected'
    options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20']
    connection = str(control_path) if control_path else locations['ssh_control_path']
    if connection:
        options += ['-S', connection]
    remote_code = '''import base64,hashlib,json
from pathlib import Path
root=Path(ROOT)
paths=[root/name for name in ['experiment.json','protocol.json','source-manifest.json','launch.json','gpu-before.json','auxiliary-binding.json','proposal-study/state.json','proposal-study/summary.json','proposal-study/plans-frozen.json','factorial/execution/state.json','verification/result.json','verification/calls.json','verification/interpretation.json']]
paths+=list((root/'proposal-study/cases').glob('*/*/plan.json'))
result={}
for target in paths:
 if target.is_file():
  content=target.read_bytes()
  result[str(target.relative_to(root))]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':base64.b64encode(content).decode()}
print(json.dumps(result))
'''.replace('ROOT', repr(locations['h2_root']))
    response = subprocess.run(['ssh', *options, 'h200-2', 'python3 -'], input=remote_code,
                              capture_output=True, text=True, check=True, timeout=180)
    bindings = {}
    for relative, record in json.loads(response.stdout).items():
        destination = local / relative
        if not destination.resolve().is_relative_to(local.resolve()):
            raise ValueError('Collection path escaped the run')
        content = base64.b64decode(record['bytes'], validate=True)
        if hashlib.sha256(content).hexdigest() != record['sha256']:
            raise ValueError('Transferred artifact hash differs')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        bindings[relative] = record['sha256']
    save(local.parent / 'collection.json', {'at': time.time(), 'exact_remote_hashes': bindings})
    state = json.loads((local / 'proposal-study/state.json').read_text())
    print(json.dumps(state, indent=2))
    if state['status'] == 'BLOCKED':
        raise RuntimeError(state['error'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--submit', action='store_true')
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--collect', type=Path)
    parser.add_argument('--ssh-control-path', type=Path)
    parser.add_argument('--repeat-from', type=Path)
    arguments = parser.parse_args()
    if arguments.submit:
        submit(Path(__file__).resolve().parents[1], arguments.ssh_control_path, arguments.repeat_from)
    elif arguments.prepare:
        prepare(arguments.root)
    elif arguments.run:
        run(arguments.root)
    else:
        collect(arguments.collect, arguments.ssh_control_path)


if __name__ == '__main__':
    main()