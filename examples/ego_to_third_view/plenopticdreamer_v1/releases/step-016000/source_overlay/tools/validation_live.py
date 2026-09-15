"""Publish completed cases immediately, without publishing partial loss averages."""
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import plenoptic_paths as layout
from validation_artifacts import CASE_ID, RUN_ID, archive_case, read, retain_file, retain_json

ROOT = layout.OUTPUTS_ROOT / 'validation'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run_directory(run_id):
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError('Invalid validation run ID')
    return ROOT / 'runs' / run_id


@contextmanager
def publication_lock():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / '.report.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def write_index(record):
    """The manifest is the completion marker; video bytes are installed first."""
    record = dict(record, updated_at=datetime.now().astimezone().isoformat())
    folder = run_directory(record['run_id'])
    atomic_json(folder / 'manifest.json', record)
    atomic_json(ROOT / 'live/latest.json', dict(record, manifest=str(folder / 'manifest.json')))
    rows = [f"# Validation {record['run_id']}", '',
            f"Checkpoint: {record['checkpoint_step']}; status: {record['status']}", '',
            'Completed videos are retained immediately. Loss history is updated only after the full fixed suite completes.', '',
            '| Kind | Video | Generated | Comparison |', '|---|---|---|---|']
    for kind in ('clips', 'groups'):
        for item in record.get(kind, []):
            generated = os.path.relpath(ROOT / item['generated'], folder)
            compared = os.path.relpath(ROOT / item['comparison'], folder)
            rows.append(f"| {kind} | {item['case_id']} | [MP4]({generated}) | [MP4]({compared}) |")
    path = folder / 'README.md'
    temporary = path.with_name('.README.' + str(os.getpid()) + '.tmp')
    temporary.write_text('\n'.join(rows) + '\n')
    temporary.replace(path)
    return record


def initialize_run(run, checkpoint, metadata, job):
    """Retain the exact inference snapshot so stage two never loads a later step."""
    run = Path(run)
    run_directory(run.name)
    checkpoint = Path(checkpoint)
    actual = sha256(checkpoint)
    if metadata.get('snapshot_sha256') != actual:
        raise ValueError('Inference checkpoint differs from its packing receipt')
    pinned = ROOT / 'checkpoints' / (actual + '.pt')
    with publication_lock():
        retain_file(checkpoint, pinned)
        retain_json(pinned.with_suffix('.json'), {
            key: metadata[key] for key in ('snapshot_sha256', 'snapshot_bytes', 'checkpoint_step',
                                          'source_sha256', 'source_bytes', 'tensor_count') if key in metadata})
        record = dict(schema=1, run_id=run.name, status='preparing',
            checkpoint_step=metadata['checkpoint_step'], checkpoint=str(pinned),
            checkpoint_sha256=actual, checkpoint_source_sha256=metadata.get('source_sha256'),
            clips=[], groups=[], loss_history_complete=False,
            pipeline_config=job.get('pipeline_config'), video_config=job.get('video_config'),
            stage='stage1', created_at=datetime.now().astimezone().isoformat())
        retain_json(run_directory(run.name) / 'submission.json', record)
        return write_index(record)


def register_inputs(run):
    run = Path(run)
    with publication_lock():
        record = read(run_directory(run.name) / 'manifest.json', {})
        if not record:
            raise ValueError('Validation publication was not initialized')
        for name in ('suite.json', 'custom_suite.json', 'preflight.json', 'video_inputs.json',
                     'pipeline-preparation.json', 'pipeline-config.json'):
            path = run / name
            if path.is_file():
                retain_json(run_directory(run.name) / name, read(path))
        record['status'] = 'generating'
        return write_index(record)


def compare_case(directory):
    directory = Path(directory)
    info = read(directory / 'inference.json', {})
    if (directory / 'comparison.json').is_file():
        comparison = read(directory / 'comparison.json')
        if (comparison.get('status') != 'passed'
                or comparison.get('checkpoint_step') != info.get('checkpoint_step')
                or not (directory / 'comparison.mp4').is_file()):
            raise ValueError('An existing comparison is incomplete or belongs to another checkpoint')
        return comparison
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    subprocess.run([sys.executable, '-B', str(layout.ROOT / 'tools/compare_inference.py'), str(directory)],
                   cwd=layout.ROOT, env=environment, check=True, timeout=300,
                   stdout=subprocess.DEVNULL)
    return read(directory / 'comparison.json', {})


def publish_case(run, summary, *, full_video=False):
    run = Path(run)
    name = summary['case_id']
    directory = run / summary['video_directory']
    directory.resolve().relative_to(run.resolve())
    info = read(directory / 'inference.json', {})
    comparison = compare_case(directory)
    if not summary.get('qualitative_only'):
        if 'image_metrics' not in comparison:
            raise ValueError('A fixed validation case has no calibrated image metrics')
        summary['image_metrics'] = comparison['image_metrics']
    with publication_lock():
        path = run_directory(run.name) / 'manifest.json'
        record = read(path, {})
        if not record:
            raise ValueError('Validation publication was not initialized')
        entry, _, _ = archive_case(ROOT, record, name,
            directory / 'generated.mp4', directory / 'comparison.mp4', info, comparison)
        entry = dict(entry, frames=info['frames'],
            generated_sha256=sha256(directory / 'generated.mp4'),
            comparison_sha256=sha256(directory / 'comparison.mp4'))
        key = 'groups' if full_video else 'clips'
        previous = {row['case_id']: row for row in record[key]}
        if name in previous and previous[name] != entry:
            raise ValueError('A published case changed')
        previous[name] = entry
        record[key] = list(previous.values())
        record['status'] = 'generating'
        write_index(record)
    print(json.dumps(dict(event='video_published', run_id=run.name, case_id=name,
                          full_video=full_video, generated=str(ROOT / entry['generated']),
                          comparison=str(ROOT / entry['comparison']))), flush=True)
    return entry


def publish_ready_groups(run, records):
    """Publish a complete source video while other source videos are still running."""
    from custom_validation_comparison import assemble_group
    run = Path(run)
    suite = read(run / 'custom_suite.json', {})
    summaries = {row['case_id']: row for row in records}
    published = {item['case_id'] for item in read(run_directory(run.name) / 'manifest.json', {}).get('groups', [])}
    result = []
    for group in suite.get('video_groups', []):
        if group['id'] not in published and all(segment['case_id'] in summaries for segment in group['segments']):
            summary = assemble_group(run, group, summaries)
            publish_case(run, summary, full_video=True)
            result.append(summary)
    return result


def finish_run(run, status, error=None):
    if status not in ('complete', 'failed', 'cancelled'):
        raise ValueError('Unknown final publication status')
    with publication_lock():
        record = read(run_directory(Path(run).name) / 'manifest.json', {})
        if not record:
            return None
        record.update(status=status, error=error, loss_history_complete=status == 'complete')
        return write_index(record)


def refinement_directory(run_id):
    run_directory(run_id)  # Validate with the same run-ID rules.
    return ROOT / 'refinements' / run_id


def write_refinement_index(record):
    folder = refinement_directory(record['run_id'])
    record = dict(record, updated_at=datetime.now().astimezone().isoformat())
    atomic_json(folder / 'manifest.json', record)
    atomic_json(ROOT / 'refinements/latest.json', dict(record, manifest=str(folder/'manifest.json')))
    lines = [f"# Refinement {record['run_id']}", '',
        f"Source run: {record['source_run_id']}; checkpoint: {record['checkpoint_step']}; status: {record['status']}", '',
        'Comparisons show stage one on the left and refinement on the right. Quality requires visual review.', '',
        '| Kind | Video | Refined | Comparison |', '|---|---|---|---|']
    for kind in ('clips','groups'):
        for item in record[kind]:
            lines.append(f"| {kind} | {item['case_id']} | [MP4]({item['generated']}) | [MP4]({item['comparison']}) |")
    path = folder/'README.md'
    temporary = path.with_name('.README.'+str(os.getpid())+'.tmp')
    temporary.write_text('\n'.join(lines)+'\n')
    temporary.replace(path)
    return record


def initialize_refinement(run, request):
    folder = refinement_directory(Path(run).name)
    retain_json(folder/'request.json', request)
    record = dict(schema=1, stage='stage2', run_id=folder.name,
        source_run_id=request['source_run_id'], checkpoint_step=request['checkpoint_step'],
        checkpoint=request['checkpoint'], checkpoint_sha256=request['checkpoint_sha256'],
        parameters=request['parameters'], uses_actual_stage1_rgb=True, real_target_ground_truth_used=False,
        status='preparing', clips=[], groups=[], created_at=datetime.now().astimezone().isoformat())
    if (folder/'manifest.json').exists():
        raise FileExistsError('Refinement run already exists')
    return write_refinement_index(record)


def publish_refinement(run, case_id, files, info, *, full_video=False):
    if not CASE_ID.fullmatch(case_id):
        raise ValueError('Invalid refinement case ID')
    folder = refinement_directory(Path(run).name)
    kind = 'groups' if full_video else 'clips'
    target = folder/kind/case_id
    record = read(folder/'manifest.json')
    if (info.get('stage') != 'stage2' or info.get('uses_actual_stage1_rgb') is not True
            or info.get('checkpoint_sha256') != record['checkpoint_sha256']):
        raise ValueError('Refinement publication provenance mismatch')
    for name in ('generated.mp4','comparison.mp4'):
        retain_file(Path(files)/name, target/name)
    retain_json(target/'inference.json', info)
    entry = dict(case_id=case_id, frames=info['frames'],
                 generated=str((target/'generated.mp4').relative_to(folder)),
                 comparison=str((target/'comparison.mp4').relative_to(folder)),
                 inference=str((target/'inference.json').relative_to(folder)),
                 generated_sha256=sha256(target/'generated.mp4'),
                 comparison_sha256=sha256(target/'comparison.mp4'))
    previous = {item['case_id']:item for item in record[kind]}
    if case_id in previous and previous[case_id] != entry:
        raise ValueError('Published refinement changed')
    previous[case_id] = entry
    record[kind] = list(previous.values())
    record['status'] = 'refining'
    write_refinement_index(record)
    print(__import__('json').dumps(dict(event='refinement_published', full_video=full_video,
        case_id=case_id, generated=str(target/'generated.mp4'), comparison=str(target/'comparison.mp4'))), flush=True)
    return entry


def finish_refinement(run, status, error=None):
    if status not in ('complete','failed','cancelled'):
        raise ValueError('Invalid refinement completion status')
    path = refinement_directory(Path(run).name)/'manifest.json'
    record = read(path, {})
    if record:
        record.update(status=status, error=error, quality_status='pending_visual_review')
        return write_refinement_index(record)

