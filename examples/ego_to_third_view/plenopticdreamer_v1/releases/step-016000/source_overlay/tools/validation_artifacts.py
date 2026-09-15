"""Publish checkpoint-named videos while retaining previous videos and loss history."""
import csv
from datetime import datetime
import filecmp
import json
import math
import os
from pathlib import Path
import re
import shutil

LEGACY_RUN = re.compile(r'\d{8}-\d{6}-\d{6}(?:-h20-1)?')
CASE_ID = re.compile(r'[a-z0-9][a-z0-9_-]*')
RUN_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')
DIRECTORIES = ('videos', 'curves', 'metrics')


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def valid_record(row):
    if row.get('status') != 'evaluated':
        return False
    if not all(key in row for key in ('checkpoint_step', 'suite_sha256', 'k', 'cases', 'target_mse')):
        raise ValueError('Incomplete validation metrics')
    if not row['cases'] or not math.isfinite(float(row['target_mse'])):
        raise ValueError('Invalid validation mean')
    expected = sum(float(c['target_mse']) for c in row['cases'])/len(row['cases'])
    if not math.isclose(float(row['target_mse']), expected, rel_tol=1e-10):
        raise ValueError('Validation mean does not match its supervised cases')
    return True


def load_history(root, run=None):
    """Merge the compact history and legacy runs before removing any old output."""
    root = Path(root)
    records, sources = {}, {}
    for path in (root/'history.json', root/'metrics/history.json'):
        for row in read(path, []) or []:
            if not valid_record(row):
                continue
            key = row.get('run_id', row.get('directory'))
            if not key:
                raise ValueError('Validation history lacks a run ID')
            records[key] = dict(row, run_id=key)
    candidates = [p for p in root.iterdir()
                  if p.is_dir() and LEGACY_RUN.fullmatch(p.name)]
    if run is not None:
        run = Path(run).resolve()
        run.relative_to(root.resolve())
        if run not in [p.resolve() for p in candidates]:
            candidates.append(run)
    for path in candidates:
        row = read(path/'validation.json', {})
        if not valid_record(row):
            continue
        records[path.name] = dict(row, run_id=path.name)
        sources[path.name] = path
    result = []
    for row in records.values():
        result.append({k:v for k,v in row.items() if not k.startswith('_') and k != 'directory'})
    result.sort(key=lambda r:(r.get('created_at', r['run_id']), r['run_id']))
    if run is not None and run.name not in records:
        raise ValueError('Requested run has no valid completed metrics')
    return result, sources


def last_training_step(path):
    path = Path(path)
    if not path.is_file():
        return -1
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-128*1024))
        lines = stream.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
            if 'step' in row:
                return int(row['step'])
        except (ValueError, TypeError):
            continue
    return -1


def retain_file(source, target):
    """Keep immutable video bytes; an existing different file is never replaced."""
    source, target = Path(source), Path(target)
    if source.is_symlink() or not source.is_file() or target.is_symlink():
        raise ValueError(f'Invalid retained file: {source} -> {target}')
    if target.exists():
        if target.is_file() and (source.samefile(target) or filecmp.cmp(source, target, shallow=False)):
            return
        raise FileExistsError(f'Refusing to overwrite retained output: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        with source.open('rb') as incoming, target.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=8*1024*1024)
        shutil.copystat(source, target)


def retain_json(path, value):
    """Archive metadata once; report refreshes may only reuse identical metadata."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f'Invalid retained metadata: {path}')
    if path.exists():
        if path.is_file() and read(path) == value:
            return
        raise FileExistsError(f'Refusing to overwrite retained metadata: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def retain_tree(source, target, skip=()):
    """Merge old outputs into staging without changing either copy's contents."""
    source, target = Path(source), Path(target)
    if source.is_symlink():
        raise ValueError(f'Invalid validation video directory: {source}')
    if not source.exists():
        return
    if not source.is_dir():
        raise ValueError(f'Expected a validation video directory: {source}')
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source)
        if relative.as_posix() in skip:
            continue
        destination = target/relative
        if path.is_symlink():
            raise ValueError(f'Unexpected link in retained videos: {path}')
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            retain_file(path, destination)


def confined(root, value):
    root = Path(root).resolve()
    path = (root/value).resolve()
    path.relative_to(root)
    return path


def archive_case(stage, record, name, generated, compared, info, comparison):
    """Store one case under its run ID and actual checkpoint step."""
    run_id, step = record['run_id'], record['checkpoint_step']
    if (not isinstance(name, str) or not CASE_ID.fullmatch(name)
            or not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id)
            or type(step) is not int or step < 1):
        raise ValueError('Invalid case ID, run ID or checkpoint step')
    if (info.get('status') != 'generated'
            or info.get('checkpoint_step') != step
            or info.get('case_id', name) != name
            or info.get('target_reference_used_during_generation') is not False
            or comparison.get('status') != 'passed'
            or comparison.get('checkpoint_step') != step):
        raise ValueError(f'Incomplete or mismatched video: {name}')
    if record.get('checkpoint_sha256') and info.get('checkpoint_sha256') != record['checkpoint_sha256']:
        raise ValueError(f'Video checkpoint fingerprint differs: {name}')
    relative = Path('videos')/name/run_id
    generated_name = f'generated_checkpoint{step:06d}.mp4'
    comparison_name = f'comparison_checkpoint{step:06d}.mp4'
    for source, filename in ((generated, generated_name), (compared, comparison_name)):
        source = Path(source)
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f'Missing completed video: {source}')
        retain_file(source, Path(stage)/relative/filename)
    entry = dict(case_id=name,
        generated=(relative/generated_name).as_posix(),
        comparison=(relative/comparison_name).as_posix(),
        inference=(relative/'inference.json').as_posix(),
        comparison_metadata=(relative/'comparison.json').as_posix())
    info = dict(info, case_id=name, run_id=run_id,
                video='../OUTPUTS/validation/'+entry['generated'])
    comparison = dict(comparison, case_id=name, run_id=run_id,
        comparison='../OUTPUTS/validation/'+entry['comparison'],
        inference='../OUTPUTS/validation/'+entry['inference'])
    for key in ('preview_images', 'preview_frame_index'):
        comparison.pop(key, None)
    retain_json(Path(stage)/entry['inference'], info)
    retain_json(Path(stage)/entry['comparison_metadata'], comparison)
    return entry, info, comparison


def retain_published(root, stage):
    """Retain this host's history, including older flat filenames, before a swap."""
    root, stage = Path(root), Path(stage)
    previous = read(root/'metrics/latest.json', {})
    migrated = set()
    for case in previous.get('cases', []):
        name = case['case_id']
        if not isinstance(name, str) or not CASE_ID.fullmatch(name):
            raise ValueError('Unsafe previous validation case ID')
        generated, compared = [confined(root, case[key]) for key in ('generated', 'comparison')]
        for path in (generated, compared):
            path.relative_to((root/'videos').resolve())
        info_path = confined(root, case.get('inference', f'metrics/inference/{name}.json'))
        comparison_path = confined(root, case.get('comparison_metadata', f'metrics/comparisons/{name}.json'))
        archive_case(stage, previous, name, generated, compared,
                     read(info_path, {}), read(comparison_path, {}))
        # The bytes now live in the checkpoint-named archive. Do not copy old flat aliases.
        for path in (generated, compared, info_path, comparison_path):
            try:
                migrated.add(path.relative_to((root/'videos').resolve()).as_posix())
            except ValueError:
                pass
    retain_tree(root/'videos', stage/'videos', migrated)


def prepare_bundle(root, stage, history, sources):
    """Stage complete, checked latest videos and the history before publication."""
    root, stage = Path(root), Path(stage)
    for name in DIRECTORIES:
        (stage/name).mkdir(parents=True)
    if not history:
        raise ValueError('No completed validation is available')
    latest = history[-1]
    previous = read(root/'metrics/latest.json', {})
    source = sources.get(latest['run_id'])
    if source is None and previous.get('run_id') != latest['run_id']:
        raise ValueError('Latest validation has no available video source')
    save(stage/'metrics/history.json', history)
    save(stage/'metrics/latest_validation.json', latest)
    video_cases = []
    previous_cases = {case['case_id']: case for case in previous.get('cases', [])}
    # Archive any explicitly supplied private workspace even if its timestamp sorts earlier.
    pending = [(row, sources[row['run_id']], False) for row in history[:-1]
               if row['run_id'] in sources
               and sources[row['run_id']].parent.resolve() == (root/'.work').resolve()]
    pending.append((latest, source, True))
    for record, case_source, is_latest in pending:
        if record.get('loss_only'):
            continue
        for case in record['cases']+record.get('qualitative_cases', []):
            if case.get('generation_skipped'):
                if case.get('qualitative_only') or not record.get('skip_standard_videos'):
                    raise ValueError('Unexpected skipped video in completed validation')
                continue
            name = case['case_id']
            if not isinstance(name, str) or not CASE_ID.fullmatch(name):
                raise ValueError('Unsafe validation case ID')
            if case_source is not None:
                folder = confined(case_source, case['video_directory'])
                folder.relative_to(root.resolve())
                generated, compared = folder/'generated.mp4', folder/'comparison.mp4'
                info_path, comparison_path = folder/'inference.json', folder/'comparison.json'
            else:
                if name not in previous_cases:
                    raise ValueError(f'Latest published video is unavailable: {name}')
                entry = previous_cases[name]
                generated, compared = [confined(root, entry[key]) for key in ('generated', 'comparison')]
                info_path = confined(root, entry.get('inference', f'metrics/inference/{name}.json'))
                comparison_path = confined(root, entry.get('comparison_metadata', f'metrics/comparisons/{name}.json'))
            for video in (generated, compared):
                video.resolve().relative_to(root.resolve())
            entry, info, comparison = archive_case(stage, record, name, generated, compared,
                                                  read(info_path, {}), read(comparison_path, {}))
            if is_latest:
                save(stage/'metrics/inference'/f'{name}.json', info)
                save(stage/'metrics/comparisons'/f'{name}.json', comparison)
                video_cases.append(entry)
    candidates = [root/'metrics/training_context']
    candidates += [path/'training_context' for path in sources.values()]
    available = [p for p in candidates if (p/'metrics.jsonl').is_file()]
    if available:
        context = max(available, key=lambda p:(last_training_step(p/'metrics.jsonl'),
                      (p/'metrics.jsonl').stat().st_mtime_ns))
        destination = stage/'metrics/training_context'
        destination.mkdir()
        for name in ('metrics.jsonl', 'config.json', 'capture.json'):
            if (context/name).is_file():
                shutil.copy2(context/name, destination/name)
    inputs = stage/'metrics/inputs'
    if source is not None:
        for name in ('suite.json', 'custom_suite.json', 'preflight.json'):
            if (source/name).is_file():
                inputs.mkdir(exist_ok=True)
                shutil.copy2(source/name, inputs/name)
    elif (root/'metrics/inputs').is_dir():
        shutil.copytree(root/'metrics/inputs', inputs)
    workflow = root/'metrics/workflow.json'
    if source is None and workflow.is_file():
        if read(workflow, {}).get('run_id') != latest['run_id']:
            raise ValueError('Published workflow does not match the latest validation')
        shutil.copy2(workflow, stage/'metrics/workflow.json')
    with (stage/'metrics/history.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run_id', 'step', 'k', 'suite', 'case', 'noise_level',
                         'target_mse', 'training_normalized_mse'])
        for row in history:
            for case in row['cases']:
                for noise in case['losses']:
                    writer.writerow([row['run_id'], row['checkpoint_step'], row['k'],
                        row['suite_sha256'], case['case_id'], noise['noise_level'],
                        noise['target_mse'], noise['training_normalized_mse']])
    manifest = dict(schema=2, run_id=latest['run_id'],
        checkpoint_step=latest['checkpoint_step'], k=latest['k'],
        video_checkpoint_step=latest['checkpoint_step'] if video_cases else None,
        cases=video_cases, validation_runs=len(history), execution_host='h20-1',
        videos='videos', curves='curves', metrics='metrics',
        video_retention='keep_previous_runs',
        updated_at=datetime.now().astimezone().isoformat())
    save(stage/'metrics/latest.json', manifest)
    return manifest


def check_bundle(stage):
    stage = Path(stage)
    manifest = read(stage/'metrics/latest.json', {})
    history = read(stage/'metrics/history.json', [])
    if manifest.get('schema') != 2 or len(history) != manifest.get('validation_runs'):
        raise ValueError('Incomplete published validation history')
    if not history or history[-1]['run_id'] != manifest['run_id']:
        raise ValueError('Latest validation does not match retained history')
    for row in history:
        valid_record(row)
    for case in manifest['cases']:
        for key in ('generated', 'comparison'):
            path = stage/case[key]
            path.resolve().relative_to(stage.resolve())
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError('Published video is missing')
    if not (stage/'curves/health-dashboard.png').is_file():
        raise ValueError('Training/validation curves are missing')
    if any(list((stage/name).rglob('*.html')) for name in DIRECTORIES):
        raise ValueError('HTML is not part of the validation output')


def commit_bundle(stage, root):
    """Retain old videos on either host, then swap directories with rollback."""
    stage, root = Path(stage), Path(root)
    retain_published(root, stage)
    check_bundle(stage)
    changed = []
    try:
        for name in DIRECTORIES:
            target, old = root/name, stage/('previous-'+name)
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                raise ValueError('Validation output directory was replaced unexpectedly')
            had_old = target.exists()
            if had_old:
                target.rename(old)
            try:
                (stage/name).rename(target)
            except BaseException:
                if had_old:
                    old.rename(target)
                raise
            changed.append((target, old, had_old))
    except BaseException:
        for target, old, had_old in reversed(changed):
            target.rename(stage/target.name)
            if had_old:
                old.rename(target)
        raise


def cleanup_legacy(root):
    """Remove obsolete loose reports; retain legacy run folders and their videos."""
    root = Path(root)
    check_bundle(root)
    retained = {r['run_id'] for r in read(root/'metrics/history.json', [])}
    targets = []
    for path in root.iterdir():
        if path.is_symlink():
            continue
        if path.is_dir() and LEGACY_RUN.fullmatch(path.name):
            row = read(path/'validation.json', {})
            if valid_record(row) and path.name not in retained:
                raise ValueError('Refusing to discard unretained validation metrics')
            # Older folders can contain videos absent from the published manifest.
            continue
        elif path.is_file() and (path.suffix == '.html'
                or path.name in ('history.json', 'latest.json', 'diagnostics.json')
                or (path.suffix in ('.png', '.svg') and
                    path.stem.startswith(('health', 'comparison', 'fixed-frame', 'validation_loss')))):
            targets.append(path)
    deleted_files, deleted_bytes = 0, 0
    for path in targets:
        files = [p for p in path.rglob('*') if p.is_file()] if path.is_dir() else [path]
        deleted_files += len(files)
        deleted_bytes += sum(p.stat().st_size for p in files)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return dict(deleted_files=deleted_files, deleted_bytes=deleted_bytes)
