"""Remote paired generation of two frozen extended-vocabulary proposal arms plus a matched parent."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback


ARMS = ('parent', 'combined_extended', 'factored_extended')


def prompt_from_skill(instruction, skill):
    if not instruction.strip() or not skill.strip():
        raise ValueError('Missing task or skill')
    return instruction + '\n' + skill


def distinct_prompt_groups(prompts):
    if set(prompts) != set(ARMS):
        raise ValueError('All predeclared arms are required')
    groups = {}
    for arm in ARMS:
        groups.setdefault(prompts[arm], []).append(arm)
    return groups


def verify_coverage(rows):
    identities = {(row['case_id'], row['seed']) for row in rows}
    cases = Counter(row['case_id'] for row in rows)
    if len(rows) != 60 or len(identities) != 60 or len(cases) != 20:
        raise ValueError('This development protocol requires all 20 cases and 60 unique outputs')
    if set(cases.values()) != {3} or any(
            {row['seed'] for row in rows if row['case_id'] == case_id}
            != {20260910, 20260911, 20260912} for case_id in cases):
        raise ValueError('Require the same three frozen seeds per case')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('generate',), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'source'))
    from integrations.skilladam_ti2v.backend import TI2VBackend, choose, save, sha
    from integrations.skilladam_ti2v.relational_repair import (
        EDIT_SLOT, NEUTRAL_TEMPLATES, prepare_instruction_repair,
        prepare_relational_repair, replace_slot,
    )
    from integrations.skilladam_ti2v.residual_repair import CONTROL_SUFFIX, prepare_repair

    protocol = json.loads((root / 'protocol.json').read_text())
    experiment = json.loads((root / 'experiment.json').read_text())
    if socket.gethostname() != protocol['expected_hostname'] or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Authorized remote CPU controller with GPU visibility disabled required')
    for relative, expected in json.loads((root / 'source-manifest.json').read_text()).items():
        if sha(root / relative) != expected:
            raise ValueError('Frozen artifact changed: ' + relative)
    rows = json.loads((root / 'inputs.json').read_text())['records']
    verify_coverage(rows)
    parent_records = json.loads((root / 'parent-selections.json').read_text())['records']
    parent = {(record['case_id'], record['seed']): record for record in parent_records}
    if set(parent) != {(row['case_id'], row['seed']) for row in rows}:
        raise ValueError('Parent selection coverage mismatch')
    for row in rows:
        for key in ('initial', 'base'):
            if sha(row[key]) != row[key + '_sha256']:
                raise ValueError('Source media changed: ' + row['case_id'])
    output = root / ('planning' if args.stage == 'plan' else 'generation')
    output.mkdir(exist_ok=False)
    backend = TI2VBackend(root, 'extended_' + args.stage)
    limits = experiment['stage_budgets'][args.stage]
    backend.cfg['max_native_calls_per_method'] = limits['native_calls']
    backend.cfg['max_auxiliary_calls_per_method'] = limits['auxiliary_calls']
    deadline = experiment['reserved_at'] + experiment['max_seconds']
    backend.deadline = time.monotonic() + max(0, deadline - time.time())
    state = {'status': 'RUNNING', 'stage': args.stage, 'completed': 0,
             'expected': len(rows), 'hostname': socket.gethostname(), 'argv': sys.argv,
             'started_at': time.time(), 'protocol_sha256': sha(root / 'protocol.json'),
             'source_manifest_sha256': sha(root / 'source-manifest.json'),
             'scope': 'Opened development cases; not unseen confirmation'}

    def update():
        state['updated_at'] = time.time()
        save(output / 'state.json', state)

    try:
        update()
        skill = CONTROL_SUFFIX.strip() + '\n' + EDIT_SLOT
        frozen_path = root / 'planning/plans-frozen.json'
        plan_state = json.loads((root / 'planning/state.json').read_text())
        if sha(frozen_path) != plan_state['plans_sha256']:
            raise ValueError('Plan bytes changed after freezing')
        frozen = json.loads(frozen_path.read_text())
        if frozen['readiness'] != 'READY_FOR_GENERATION' or tuple(frozen['arms']) != ARMS:
            raise ValueError('Frozen audited extended-vocabulary generation plan is required')
        plans = {(record['case_id'], record['seed']): record for record in frozen['records']}
        if set(plans) != set(parent):
            raise ValueError('Frozen plan coverage mismatch')
        state['plans_sha256'] = sha(frozen_path)
        update()
        abort = threading.Event()

        def execute(index, row):
            if abort.is_set() or time.time() >= deadline:
                raise TimeoutError('Frozen experiment deadline reached')
            folder = output / 'cases' / f'{index:03d}'
            folder.mkdir(parents=True, exist_ok=False)
            plan = plans[(row['case_id'], row['seed'])]
            result = {}
            for prompt, aliases in distinct_prompt_groups(plan['prompts']).items():
                if abort.is_set():
                    raise RuntimeError('Retain in-flight receipts; another case failed')
                target = folder / aliases[0]
                target.mkdir()
                video = backend.native(row, prompt, target)
                audit, usages = backend.audit(row, video, target / 'audit')
                base_audit = parent[(row['case_id'], row['seed'])]['base_audit']
                decision = choose(base_audit, audit)
                selected = video if decision['selected'] == 'candidate' else row['base']
                record = {'case_id': row['case_id'], 'seed': row['seed'],
                          'video': selected, 'sha256': sha(selected),
                          'candidate': video, 'candidate_sha256': sha(video),
                          'base_audit': base_audit, 'candidate_audit': audit,
                          'decision': decision, 'usage': usages,
                          'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                          'shared_arms': aliases, 'physical_success_established': False}
                for arm in aliases:
                    result[arm] = {**record, 'arm': arm}
            save(folder / 'selections.json', result)
            return index, result

        ordered = {}
        with ThreadPoolExecutor(max_workers=experiment['generation_workers']) as executor:
            futures = [executor.submit(execute, index, row) for index, row in enumerate(rows)]
            try:
                for future in as_completed(futures):
                    index, record = future.result()
                    ordered[index] = record
                    state['completed'] = len(ordered)
                    update()
            except BaseException:
                abort.set()
                for future in futures:
                    future.cancel()
                raise
        selections = {arm: [ordered[index][arm] for index in range(len(rows))] for arm in ARMS}
        save(output / 'selections-frozen.json', selections)
        state.update(status='SELECTIONS_FROZEN', selections_sha256=sha(output / 'selections-frozen.json'))
        update()
        for pool_name in experiment['owned_native_pools']:
            (Path(pool_name) / 'STOP').touch()
        scores = {}
        state['status'] = 'OFFICIAL_SCORING'
        update()
        for arm in ARMS:
            chosen = selections[arm]
            raw = [{**record, 'video': record['candidate'], 'sha256': record['candidate_sha256']}
                   for record in chosen]
            for kind, records in (('raw', raw), ('selected', chosen)):
                name = arm + '_' + kind
                scores[name] = backend.score(records, name)
                save(output / ('scores-' + name + '.json'), scores[name])
        save(output / 'scores.json', scores)
        state['status'] = 'DEVELOPMENT_COMPLETE'
        state['finished_at'] = time.time()
    except BaseException as error:
        state.update(status='BLOCKED', error=f'{type(error).__name__}: {error}', finished_at=time.time())
        (output / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        update()


if __name__ == '__main__':
    main()