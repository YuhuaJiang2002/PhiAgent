#!/usr/bin/env python3
"""Run one reviewed ego-view reconstruction stage with provenance and GPU preflight."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
GPU_STAGES = {'run_vggt_omega_ego', 'export_hawor_mano_meshes',
              'run_sam3d_ego_metric', 'track_ego_foundationpose',
              'refine_sam3d_layout', 'refine_static_multiview',
              'fix_clock_upright', 'render_lab_demo', 'audit_fp_trajectories',
              'run_sam2_video_point_track'}


def capture(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed: {result.stderr.strip()}')
    return result.stdout.strip()


def select_gpu(value):
    snapshot = capture(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total',
                        '--format=csv,noheader,nounits'])
    rows = [tuple(part.strip() for part in line.split(',')) for line in snapshot.splitlines()]
    matches = [row for row in rows if value in row[:2]]
    if len(matches) != 1:
        raise ValueError(f'GPU {value!r} is not a unique visible physical GPU')
    # Use UUID, which remains unambiguous inside a restricted container.
    return matches[0][1], snapshot


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(x.stem for x in (HERE/'stages').glob('*.py') if x.stem != 'runtime'))
    p.add_argument('--root', type=Path, required=True, help='External dependency root (third_party/, envs/)')
    p.add_argument('--run', type=Path, required=True, help='Input/intermediate bundle; use a new bundle for new experiments')
    p.add_argument('--output', type=Path, required=True, help='New per-stage execution directory; must not exist')
    p.add_argument('--gpu', help='Explicit physical GPU index or UUID; required for GPU stages')
    p.add_argument('--seed', type=int, default=20260905)
    # Adapter flags follow a literal -- to avoid collisions with launcher flags.
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index('--') if '--' in argv else len(argv)
    a = p.parse_args(argv[:split]); extra = argv[split+1:]
    if a.stage in GPU_STAGES and a.gpu is None:
        p.error('this stage requires --gpu; inspect shared GPU usage before choosing')
    if not a.root.is_dir() or not a.run.is_dir():
        p.error('--root and --run must be existing directories')
    env = os.environ.copy()
    gpu, snapshot = select_gpu(a.gpu) if a.gpu is not None else ('', 'CPU stage')
    env.update(CUDA_VISIBLE_DEVICES=gpu, PHI_EGO_ROOT=str(a.root.resolve()),
               PHI_EGO_RUN=str(a.run.resolve()), PHI_EGO_OUTPUT=str(a.output.resolve()),
               PYTHONHASHSEED=str(a.seed))
    a.output.mkdir(parents=True, exist_ok=False)
    cmd = [sys.executable, str(HERE/'stages'/f'{a.stage}.py'), *extra]
    record = {'stage': a.stage, 'command': cmd, 'root': str(a.root.resolve()),
              'run': str(a.run.resolve()), 'output': str(a.output.resolve()),
              'hostname': socket.gethostname(), 'started_at': datetime.now(timezone.utc).isoformat(),
              'gpu_uuid': gpu, 'gpu_snapshot': snapshot, 'seed': a.seed,
              'seed_note': 'hash seed only; model seeds remain explicit adapter arguments',
              'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
    for field, command in [('git_commit', ['git', 'rev-parse', 'HEAD']),
                           ('git_status', ['git', 'status', '--porcelain'])]:
        try:
            record[field] = capture(command, HERE)
        except RuntimeError as exc:
            record[field] = str(exc)
    target = a.output/'execution.json'
    target.write_text(json.dumps(record, indent=2))
    with (a.output/'stage.log').open('w') as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode, finished_at=datetime.now(timezone.utc).isoformat())
    target.write_text(json.dumps(record, indent=2))
    print(json.dumps({'returncode': result.returncode, 'execution': str(target), 'log': str(a.output/'stage.log')}))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
