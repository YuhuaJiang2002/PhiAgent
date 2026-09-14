#!/usr/bin/env python3
"""Read source and one complete train scene per dataset; no model imports."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid


def command(args, cwd=None, timeout=60):
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {'returncode': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()[:2000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def source_report(repo):
    base = 'cosmos_transfer2/_src/predict2/camera/'
    paths = {
        'rays': base + 'utils.py',
        'network': base + 'networks/dit_multiview_camera_ar.py',
        'model': base + 'models/multiview_camera_ar_video2world_model.py',
        'conditioner': base + 'configs/multiview_camera/conditioner.py',
        'data_registration': base + 'configs/multiview_camera/data.py',
        'experiments': base + 'configs/multiview_camera/experiment/exp_2b.py',
        'inference': base + 'inference/multiview_camera_ar_video2world.py',
        'retrieval': base + 'datasets/camera_conditioned/dataset_utils.py',
        'train_entry': 'scripts/train.py',
    }
    report = {
        'commit': command(['git', 'rev-parse', 'HEAD'], repo),
        'tracked_changes': command(['git', 'status', '--porcelain', '--untracked-files=no'], repo),
        'files': {},
    }
    for name, relative in paths.items():
        path = repo / relative
        row = {'path': relative, 'exists': path.is_file()}
        if path.is_file():
            source = path.read_text()
            row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            try:
                tree = ast.parse(source)
                row['syntax_ok'] = True
                if name == 'data_registration':
                    row['functions'] = {
                        n.name: ast.get_source_segment(source, n)
                        for n in tree.body if isinstance(n, ast.FunctionDef)
                    }
                if name == 'model':
                    row['batch_keys'] = sorted({
                        n.slice.value for n in ast.walk(tree)
                        if isinstance(n, ast.Subscript)
                        and isinstance(n.value, ast.Name) and n.value.id == 'data_batch'
                        and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)
                    })
                    row['condition_chunk_indices'] = sorted({
                        n.slice.value for n in ast.walk(tree)
                        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                        and n.value.id in ('raw_state_cond_chunks', 'latent_state_cond_list')
                        and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, int)
                    })
            except SyntaxError as exc:
                row['syntax_error'] = str(exc)
        report['files'][name] = row
    return report


def pick_scene(roots):
    first_problem = None
    for root in roots:
        if not root.is_dir():
            continue
        for camera in root.rglob('camera_extrinsics.json'):
            scene = camera.parent.parent
            if 'train' not in scene.parts:
                continue
            videos = [scene / 'videos' / f'cam{i:02d}.mp4' for i in range(1, 11)]
            missing = [str(p) for p in videos if not p.is_file() or p.stat().st_size == 0]
            if not missing:
                return scene
            if first_problem is None:
                first_problem = missing[:2]
    raise RuntimeError(f'尚未找到含 10 路视频的 train 场景；搜索目录={roots}；缺失示例={first_problem}')


def data_report(roots, project_root):
    scene = pick_scene(roots)
    print('检查样本：', scene, flush=True)
    camera = scene / 'cameras/camera_extrinsics.json'
    raw_bytes = camera.read_bytes()
    data = json.loads(raw_bytes)
    report = {
        'scene': os.path.relpath(scene, project_root),
        'camera_json_sha256': hashlib.sha256(raw_bytes).hexdigest(),
        'camera_json_type': type(data).__name__,
        'videos_bytes': {f'cam{i:02d}': (scene / 'videos' / f'cam{i:02d}.mp4').stat().st_size for i in range(1, 11)},
    }
    if isinstance(data, dict):
        frames = sorted((int(m.group(1)), k) for k in data if (m := re.fullmatch(r'frame(\d+)', k)))
        report['top_keys'] = list(data)[:12]
        report['frame_count'] = len(frames)
        ids = [i for i, _ in frames]
        report['consecutive_from_zero'] = ids == list(range(len(ids))) and bool(ids)
        report['missing_cameras_by_frame'] = {
            k: [f'cam{i:02d}' for i in range(1, 11) if not isinstance(data[k], dict) or f'cam{i:02d}' not in data[k]]
            for _, k in frames
        }
        report['missing_cameras_by_frame'] = {k: v for k, v in report['missing_cameras_by_frame'].items() if v}
        report['first_last_examples'] = {
            k: {c: data[k].get(c) for c in ('cam01', 'cam02')}
            if isinstance(data[k], dict) else data[k]
            for _, k in frames[:1] + frames[-1:]
        }
        sample = data[frames[0][1]]['cam01'] if frames else None
        report['pose_value_type'] = type(sample).__name__
        if isinstance(sample, str):
            rows = re.findall(r'\[([^\[\]]+)\]', sample)
            matrix = [[float(x) for x in row.split()] for row in rows]
            assert len(matrix) == 4 and all(len(row) == 4 for row in matrix), 'Expected a 4x4 pose string'
            assert all(math.isfinite(x) for row in matrix for x in row), 'Non-finite pose'
            report['raw_pose_matrix'] = matrix
            report['raw_translation_last_row'] = matrix[3][:3]
            report['raw_affine_last_column_valid'] = all(abs(matrix[i][3]) < 1e-5 for i in range(3)) and abs(matrix[3][3] - 1) < 1e-5
    else:
        report['schema_preview'] = repr(data)[:1000]

    report['ffprobe'] = {}
    executable = shutil.which('ffprobe')
    for cam in ('cam01', 'cam02'):
        if executable is None:
            report['ffprobe'][cam] = {'status': 'pending: host ffprobe unavailable'}
            continue
        print('检查视频：', cam, flush=True)
        result = command([
            executable, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
            '-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration',
            '-of', 'json', str(scene / 'videos' / (cam + '.mp4')),
        ], timeout=180)
        if result.get('returncode') == 0:
            try:
                result['metadata'] = json.loads(result['stdout'])
                del result['stdout']
            except ValueError:
                result['error'] = 'ffprobe output is not JSON'
        report['ffprobe'][cam] = result
    report['sample_validation'] = 'passed' if (
        report.get('frame_count') == 81 and report.get('consecutive_from_zero')
        and not report.get('missing_cameras_by_frame')
        and report.get('raw_affine_last_column_valid')
        and all(v.get('returncode') == 0 and any(
            s.get('nb_read_frames') == '81' and s.get('width') == 1280 and s.get('height') == 1280
            for s in v.get('metadata', {}).get('streams', [])) for v in report['ffprobe'].values())
    ) else 'failed'
    report['validation_scope'] = 'one complete scene: 10 video files, 81 camera frames, full ffprobe decode of cam01 and cam02'
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--syncam', type=Path)
    parser.add_argument('--multicam', type=Path)
    parser.add_argument('--datasets', nargs='+', choices=['syncam', 'multicam'], default=['syncam', 'multicam'])
    args = parser.parse_args()
    root = args.root.resolve()
    report = {'repository': source_report(root / 'cosmos-transfer2.5'), 'datasets': {}}
    pin = root / 'download-state/source.json'
    if pin.is_file():
        report['repository']['source_pin'] = json.loads(pin.read_text())
    for name, label in [('syncam', 'SynCamVideo'), ('multicam', 'MultiCamVideo')]:
        if name not in args.datasets:
            continue
        custom = getattr(args, name)
        roots = [custom.resolve()] if custom else [
            root / 'datasets/extracted' / name,
            root / 'datasets/extracted' / label,
            root / 'datasets/raw' / label,
            root / 'datasets' / (label + '-Dataset'),
        ]
        try:
            report['datasets'][name] = data_report(roots, root)
        except Exception as exc:
            report['datasets'][name] = {'error': str(exc)}
    output = root / 'reports' / ('step01-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6])
    output.mkdir(parents=True)
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    (output / 'report.json').write_text(text)
    for name, value in report['datasets'].items():
        (root / 'reports' / f'{name}-sample-check.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(text)
    print('报告：', output / 'report.json')
    if any(v.get('sample_validation') != 'passed' for v in report['datasets'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
