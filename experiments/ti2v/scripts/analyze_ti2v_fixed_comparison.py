"""Remote-only post-hoc paired analysis of already frozen official CSVs."""
import argparse
import base64
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys


METRICS = ('BLEUScore', 'CLIPScore', 'hsd', 'dyn', 'ndtw')
METHODS = ('ours_v2', 'videoweaver', 'minimax')


def load_rows(stream):
    records = {}
    for row in csv.DictReader(stream):
        fields = ('task_id', 'episode_id', 'trial_id')
        if not all(str(row.get(field, '')).strip().isdigit() for field in fields):
            continue
        identity = tuple(row[field] for field in fields)
        if identity in records:
            raise ValueError('Duplicate case/trial identity')
        values = [float(row[metric]) for metric in METRICS]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Nonfinite official metric')
        records[identity] = values
    counts = Counter(identity[:2] for identity in records)
    if len(records) != 60 or len(counts) != 20 or set(counts.values()) != {3}:
        raise ValueError('Require twenty complete three-trial cases')
    return records


def sha(target):
    return hashlib.sha256(target.read_bytes()).hexdigest()


def analyze(root):
    if socket.gethostname() != 'yxys-node-214-41-3-2' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run benchmark statistics only on the authorized remote CPU controller')
    import numpy as np
    if np.__version__ != '2.3.5':
        raise RuntimeError('Require pinned NumPy 2.3.5')
    config = json.loads((root / 'config.json').read_text())
    for relative, digest in config['source_sha256'].items():
        if sha(root / relative) != digest:
            raise ValueError('Frozen input hash changed')
    records = {}
    for method in METHODS:
        with (root / f'{method}.csv').open(newline='') as stream:
            records[method] = load_rows(stream)
    identities = sorted(records['ours_v2'])
    if any(set(values) != set(identities) for values in records.values()):
        raise ValueError('Paired method identities differ')
    cases = sorted({identity[:2] for identity in identities})
    resamples = np.random.default_rng(config['seed']).integers(0, len(cases), size=(config['replicates'], len(cases)))
    target = np.asarray([records['ours_v2'][identity] for identity in identities])
    contrasts = {}
    for baseline in ('videoweaver', 'minimax'):
        other = np.asarray([records[baseline][identity] for identity in identities])
        delta = target - other
        by_case = np.stack([delta[[index for index, identity in enumerate(identities) if identity[:2] == case]].mean(0) for case in cases])
        lower, upper = np.quantile(by_case[resamples].mean(1), [0.005, 0.995], axis=0)
        contrasts['ours_v2_minus_' + baseline] = {
            metric: {'difference': float(delta[:, column].mean()),
                     'ci99': [float(lower[column]), float(upper[column])],
                     'target_mean': float(target[:, column].mean()),
                     'baseline_mean': float(other[:, column].mean())}
            for column, metric in enumerate(METRICS)}
    result = {
        'status': 'ANALYSIS_COMPLETE', 'run_id': config['run_id'], 'numpy': np.__version__,
        'scope': 'Post-hoc development analysis on previously opened cases; not unseen confirmation or budget-matched superiority',
        'case_count': 20, 'trials_per_case': 3, 'outputs_per_method': 60,
        'seed': config['seed'], 'replicates': config['replicates'],
        'ci_method': 'Paired case-cluster bootstrap; 99% percentile intervals; five-metric Bonferroni approximation within each contrast, exploratory across contrasts',
        'source_sha256': config['source_sha256'], 'contrasts': contrasts,
        'new_generation_calls': 0, 'new_auxiliary_calls': 0, 'official_metrics_recomputed': False,
    }
    (root / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (root / 'execution.json').write_text(json.dumps({
        'hostname': socket.gethostname(), 'argv': sys.argv, 'CUDA_VISIBLE_DEVICES': '',
        'gpu_inventory': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader'], text=True),
        'python': sys.version, 'config_sha256': sha(root / 'config.json'),
        'result_sha256': sha(root / 'result.json')}, indent=2))
    (root / 'packages.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    print(json.dumps({'sha256': sha(root / 'result.json'),
                      'bytes': base64.b64encode((root / 'result.json').read_bytes()).decode()}))


def submit(control_path):
    repo = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    local = repo / 'outputs/ti2v-fixed-comparison-analysis' / run_id
    remote = Path('/opt/phiagent/runs/ti2v-fixed-comparison-analysis') / run_id
    local.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, local / 'analyze.py')
    for method in METHODS:
        shutil.copy2(repo / f'paper/phiagent-technical-report-overleaf/evidence/results/optimization-{method}.csv', local / f'{method}.csv')
    config = {'run_id': run_id, 'seed': 20260918, 'replicates': 10000,
              'contrasts': ['ours_v2_minus_videoweaver', 'ours_v2_minus_minimax'],
              'source_sha256': {target.name: sha(target) for target in local.iterdir() if target.is_file()}}
    (local / 'config.json').write_text(json.dumps(config, indent=2))
    (local / 'git-state.json').write_text(json.dumps({
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True)}, indent=2))
    options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-S', str(control_path)]
    command = 'mkdir -p ' + shlex.quote(str(remote.parent)) + ' && mkdir ' + shlex.quote(str(remote)) + ' && tar -xzf - -C ' + shlex.quote(str(remote))
    with subprocess.Popen(['tar', '--no-xattrs', '-czf', '-', '-C', str(local), '.'], stdout=subprocess.PIPE,
                          env={**os.environ, 'COPYFILE_DISABLE': '1'}) as archive:
        subprocess.run(['ssh', *options, 'h200-2', command], stdin=archive.stdout, check=True, timeout=180)
        archive.stdout.close()
        if archive.wait() != 0:
            raise RuntimeError('Source transfer failed')
    command = 'CUDA_VISIBLE_DEVICES= /dev/shm/phiagent-h3-h200-cu128-v2/bin/python ' + shlex.quote(str(remote / 'analyze.py')) + ' --run --root ' + shlex.quote(str(remote))
    response = subprocess.run(['ssh', *options, 'h200-2', command], capture_output=True, text=True, check=True, timeout=180)
    receipt = json.loads(response.stdout)
    content = base64.b64decode(receipt['bytes'], validate=True)
    if hashlib.sha256(content).hexdigest() != receipt['sha256']:
        raise ValueError('Result transfer changed bytes')
    (local / 'result.json').write_bytes(content)
    destination = repo / 'paper/phiagent-technical-report-overleaf/evidence/fixed-comparison-uncertainty.json'
    destination.write_bytes(content)
    print(json.dumps({'run_id': run_id, 'remote_root': str(remote), 'result_sha256': receipt['sha256'],
                      'destination': str(destination), 'status': 'ANALYSIS_COMPLETE'}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--submit', action='store_true')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--ssh-control-path', type=Path)
    arguments = parser.parse_args()
    if arguments.run:
        analyze(arguments.root)
    else:
        if not arguments.ssh_control_path:
            parser.error('Submission requires the existing run-owned SSH control path')
        submit(arguments.ssh_control_path)


if __name__ == '__main__':
    main()