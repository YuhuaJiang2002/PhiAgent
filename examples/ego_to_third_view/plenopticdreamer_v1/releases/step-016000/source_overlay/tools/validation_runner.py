#!/usr/bin/env python3
"""Submit checkpoint validation to h20-1; never control the training job."""
import plenoptic_paths as layout
import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

import prepare_plenoptic as prepare
from validation_resources import availability, check_devices, parse_gpu_ids

TARGET_HOST = 'h20-1'
TARGET_ADDRESS = '192.168.26.61'
HEAD_HOST = 'h20-4'
HOST = socket.gethostname().split('.')[0]
STATE = layout.rooted('download-state/fixed-validation-h20-1.json')
SUBMISSION = layout.rooted('download-state/fixed-validation-submission.json')
DEFAULT_CHECKPOINT = layout.rooted('outputs/basic_stage1_24gpu/latest.pt')
VALIDATION_ROOT = layout.rooted('outputs/validation')
REPORT_ENTRY = VALIDATION_ROOT/'curves/health-dashboard.png'
SSH = ['ssh','-i',str(layout.SSH_KEY),
       '-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=10']


class ValidationCancelled(InterruptedError):
    """A termination request for this validation process."""


def interrupt_validation(signum, frame):
    raise ValidationCancelled(f'Validation cancelled by signal {signum}')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def identity_active(value):
    try:
        proc = Path('/proc')/str(value['pid'])
        return (value.get('host') == HOST and (proc/'stat').read_text().split()[21] == value['start_ticks']
                and 'validation_runner.py _' in (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except (KeyError,OSError):
        return False


def public_state(value):
    """Keep CLI status useful without printing all archived camera matrices."""
    result = dict(value)
    config = result.pop('pipeline_config', None)
    videos = result.pop('video_config', None)
    request = result.pop('refinement_config', None)
    if config:
        result['pipeline_profile'] = dict(source_file=config.get('source_file'),
            source_sha256=config.get('source_sha256'), name=config['profile'].get('name','custom'),
            generation=config['profile']['generation'],
            preview_segments=config['profile'].get('preview_segments'),
            parameter_variants=[item['id'] for item in config['profile'].get('tuning_variants',[])])
    if videos:
        result['source_videos'] = [entry['id'] for entry in videos.get('videos', [])]
    if request:
        result['refinement'] = dict(source_run_id=request.get('source_run_id'),
            videos=[entry['id'] for entry in request.get('videos', [])],
            parameters=request.get('parameters'), checkpoint_sha256=request['checkpoint_sha256'],
            uses_actual_stage1_rgb=request.get('uses_actual_stage1_rgb'))
    return result


def update(path, phase, **details):
    value = dict(read(path), **details, phase=phase, updated_at=datetime.now().astimezone().isoformat())
    prepare.save_json(path,value)
    print(json.dumps(public_state(value),ensure_ascii=False),flush=True)


def command(argv, timeout=3600, capture=False):
    process = subprocess.Popen(argv,cwd=prepare.ROOT,start_new_session=True,
        stdout=subprocess.PIPE if capture else None,text=True)
    try:
        output,_ = process.communicate(timeout=timeout)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode,argv,output)
        return output
    except BaseException:
        # Only the process group created for this validation command is eligible.
        try: os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError: pass
        try: process.wait(timeout=20)
        except subprocess.TimeoutExpired: pass
        try: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()
        raise


def remote(argv, **kwargs):
    return command(SSH+['root@'+TARGET_ADDRESS,
        'cd '+shlex.quote(str(prepare.ROOT))+' && '+shlex.join(argv)],**kwargs)


def wait_for_devices(path, requested_gpu_indices=None, on_target=False, video_batch_size=1):
    """Keep a four-GPU validation queued without controlling other jobs."""
    remote_args = ['python3', 'tools/validation_resources.py', '--json',
                   '--video-batch-size', str(video_batch_size)]
    if requested_gpu_indices is not None:
        remote_args += ['--gpus', ','.join(map(str, requested_gpu_indices))]

    while True:
        value = (
            json.loads(remote(remote_args, capture=True, timeout=30))
            if on_target else availability(requested_gpu_indices, video_batch_size)
        )
        if value['ready']:
            return value['devices']
        update(path, 'waiting_for_gpus', context_parallel_size=4, gpu_availability=value)
        time.sleep(15)


def snapshot(source, destination):
    """Pin one atomically published checkpoint; do not request a new save."""
    source = source.resolve()
    if source == destination.resolve():
        if not source.is_file(): raise FileNotFoundError(source)
        return
    if destination.exists(): raise FileExistsError(destination)
    # Shared checkpoint receipts include ctime. Adding/removing a hard link
    # changes that fingerprint even when the checkpoint bytes stay identical.
    if not layout.SHARED_FILESYSTEM:
        try:
            os.link(source,destination)
            return
        except OSError as exc:
            import errno
            if exc.errno != errno.EXDEV: raise
    # An open descriptor remains bound to the published inode if latest.pt changes.
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        shutil.copyfileobj(incoming,outgoing,length=8*1024*1024)


def inference_snapshot(source, destination):
    """Pin once, then remove training-only tensors from the transferred copy."""
    if Path(source).resolve() == destination.resolve():
        if not destination.is_file():
            raise FileNotFoundError(destination)
        return  # h20-1 received this exact immutable snapshot from the head.
    source = Path(source).resolve()
    metadata = read(source.with_suffix('.json'))
    if (source.parent == (VALIDATION_ROOT/'checkpoints').resolve()
            and source.name == str(metadata.get('snapshot_sha256'))+'.pt'):
        from validation_live import sha256
        if sha256(source) != metadata['snapshot_sha256']:
            raise ValueError('Archived inference checkpoint changed')
        snapshot(source,destination)
        if sha256(destination) != metadata['snapshot_sha256']:
            raise ValueError('Copied inference checkpoint differs from its archive')
        shutil.copyfile(source.with_suffix('.json'),destination.with_suffix('.json'))
        return  # Parameter comparisons retain the same bytes, including checkpoint provenance.
    training = destination.with_name('.training-checkpoint.pt')
    snapshot(Path(source), training)
    try:
        command(['bash', '-c', 'source tools/activate_h20.sh && exec env '
                 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 ionice -c 3 '
                 'python tools/prepare_inference_checkpoint.py "$@"', 'inference-snapshot',
                 '--source', str(training), '--output', str(destination)], timeout=600)
    finally:
        training.unlink(missing_ok=True)
        destination.with_suffix('.packing.tmp').unlink(missing_ok=True)


def capture_training_context(checkpoint, run, force=False):
    """Copy small training diagnostics without loading or requesting a checkpoint."""
    context = run/'training_context'
    if not force and (context/'capture.json').is_file():
        return  # A head-provided snapshot is authoritative on h20-1.
    source = Path(checkpoint).resolve().parent
    context.mkdir(parents=True,exist_ok=True)
    info = dict(source_directory=str(source), captured_on=HOST,
                captured_at=datetime.now().astimezone().isoformat(), status='unavailable')
    metrics = source/'metrics.jsonl'
    if metrics.is_file():
        with metrics.open('rb') as stream:
            raw = stream.read(metrics.stat().st_size)
        temporary = context/'metrics.jsonl.tmp'
        temporary.write_bytes(raw)
        temporary.replace(context/'metrics.jsonl')
        info.update(status='captured',metrics_bytes=len(raw))
        config = source/'resolved_config.json'
        if config.is_file():
            prepare.save_json(context/'config.json',read(config))
    prepare.save_json(context/'capture.json',info)


def render_report(run=None):
    if HOST != TARGET_HOST:
        raise RuntimeError('Report generation is permitted only on h20-1')
    args = [str(run)] if run is not None else []
    command(['bash','-c','source tools/activate_h20.sh && exec env '
             'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
             'nice -n 10 python tools/validation_report.py "$@"',
             'validation-report',*args],timeout=600)


def copy_reports_to_head():
    """Mirror only the published videos, curves and retained metrics."""
    if HOST != HEAD_HOST:
        raise RuntimeError('Only h20-4 may pull the h20-1 validation reports')
    if layout.SHARED_FILESYSTEM:
        return
    VALIDATION_ROOT.mkdir(parents=True,exist_ok=True)
    from validation_artifacts import DIRECTORIES, cleanup_legacy, commit_bundle
    with (VALIDATION_ROOT/'.report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(prefix='.mirror-',dir=VALIDATION_ROOT) as temporary:
            stage = Path(temporary)
            for name in DIRECTORIES:
                (stage/name).mkdir()
            # One shared lock keeps all published directories from the same bundle.
            # Reuse unchanged local files so a curve refresh need not resend videos.
            filters = ['--include=/'+name+'/***' for name in DIRECTORIES]
            command(['nice','-n','19','ionice','-c','3','rsync','-a','--bwlimit=10240',
                     '--link-dest='+str(VALIDATION_ROOT),
                     '--rsync-path='+shlex.join(['flock','-s',str(VALIDATION_ROOT/'.report.lock'),
                                                'nice','-n','19','ionice','-c','3','rsync']),
                     *filters,'--exclude=*','-e',shlex.join(SSH),
                     'root@'+TARGET_ADDRESS+':'+str(VALIDATION_ROOT)+'/',
                     str(stage)+'/'],timeout=900)
            commit_bundle(stage,VALIDATION_ROOT)
        cleanup_legacy(VALIDATION_ROOT)


def refresh_training_context():
    """Send a small, fresh log snapshot so report refreshes the current curves."""
    if HOST != HEAD_HOST:
        raise RuntimeError('Training diagnostics must be captured on h20-4')
    VALIDATION_ROOT.mkdir(parents=True,exist_ok=True)
    with (VALIDATION_ROOT/'.report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        capture_training_context(DEFAULT_CHECKPOINT,VALIDATION_ROOT/'metrics',force=True)
        if layout.SHARED_FILESYSTEM:
            return
        context = VALIDATION_ROOT/'metrics/training_context'
        remote(['flock','-x',str(VALIDATION_ROOT/'.report.lock'),
                'mkdir','-p',str(context)],timeout=30)
        command(['nice','-n','19','ionice','-c','3','rsync','-a','--bwlimit=10240',
                 '--rsync-path='+shlex.join(['flock','-x',str(VALIDATION_ROOT/'.report.lock'),
                                            'nice','-n','19','ionice','-c','3','rsync']),
                 '-e',shlex.join(SSH),str(context)+'/',
                 'root@'+TARGET_ADDRESS+':'+str(context)+'/'],timeout=120)


def checkpoint_fingerprint(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]


def latest_checkpoint_sync_status():
    """Prove that shared latest.pt matches the newest completed source save."""
    audit = layout.OUTPUTS_ROOT / 'migration/20260912-shared'
    source = layout.LEGACY_ROOT / 'outputs/basic_stage1_24gpu'
    mirror = read(audit / 'mirror-status.json')
    source_metadata = read(source / 'latest.json')
    shared_metadata = read(DEFAULT_CHECKPOINT.with_suffix('.json'))
    receipt = read(audit / 'training-checkpoints/latest.pt.json')
    try:
        source_fingerprint = checkpoint_fingerprint(source / 'latest.pt')
        shared_fingerprint = checkpoint_fingerprint(DEFAULT_CHECKPOINT)
    except OSError:
        source_fingerprint = shared_fingerprint = None
    source_step = source_metadata.get('step')
    shared_step = shared_metadata.get('step')
    receipt_step = receipt.get('payload', {}).get('step')
    ready = (
        mirror.get('status') in ('watching_source_training', 'complete')
        and type(source_step) is int
        and source_step == shared_step == mirror.get('step') == receipt_step
        and isinstance(receipt.get('sha256'), str) and len(receipt['sha256']) == 64
        and source_fingerprint is not None
        and receipt.get('source_fingerprint') == source_fingerprint
        and receipt.get('verified') == shared_fingerprint
    )
    return dict(
        ready=ready, source_step=source_step, shared_step=shared_step,
        mirror_step=mirror.get('step'), receipt_step=receipt_step,
        mirror_status=mirror.get('status'),
        receipt_sha256=receipt.get('sha256'),
        source_fingerprint_matches=receipt.get('source_fingerprint') == source_fingerprint,
        shared_fingerprint_matches=receipt.get('verified') == shared_fingerprint,
    )


def synchronize_latest_checkpoint(path, destination, timeout=900):
    """Wait for the source mirror and pin one snapshot while its handoff is locked."""
    if HOST != HEAD_HOST:
        raise RuntimeError('Latest checkpoint synchronization is only available on h20-4')
    audit = layout.OUTPUTS_ROOT / 'migration/20260912-shared'
    lock_path = audit / 'source-mirror-handoff.lock'
    destination = Path(destination)
    deadline = time.monotonic() + timeout
    last = None
    while True:
        result = None
        with lock_path.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                result = dict(ready=False, mirror_status='copying_checkpoint', mirror_lock='busy')
            else:
                result = latest_checkpoint_sync_status()
                if result.get('ready'):
                    update(path, 'preparing_snapshot', checkpoint=str(DEFAULT_CHECKPOINT),
                           checkpoint_step=result['shared_step'], latest_checkpoint_sync=result)
                    inference_snapshot(DEFAULT_CHECKPOINT, destination)
                    pinned = read(destination.with_suffix('.json'))
                    after = latest_checkpoint_sync_status()
                    stable = (
                        after.get('ready')
                        and after.get('shared_step') == result.get('shared_step')
                        and pinned.get('checkpoint_step') == result.get('shared_step')
                        and pinned.get('source_sha256') == result.get('receipt_sha256')
                    )
                    if stable:
                        return dict(after, checkpoint=str(DEFAULT_CHECKPOINT),
                                    checkpoint_step=after['shared_step'],
                                    pinned_checkpoint=str(destination),
                                    pinned_checkpoint_sha256=pinned.get('snapshot_sha256'))
                    # A source save can complete while the inference copy is being
                    # packed. Discard the stale candidate, let the mirror catch up,
                    # and pin again instead of silently running the older step.
                    destination.unlink(missing_ok=True)
                    destination.with_suffix('.json').unlink(missing_ok=True)
                    destination.with_suffix('.packing.tmp').unlink(missing_ok=True)
                    result = dict(after, ready=False,
                                  retry_reason='source_advanced_while_pinning')
        if result != last:
            update(path, 'synchronizing_latest_checkpoint', latest_checkpoint_sync=result)
            last = result
        if result.get('mirror_status') == 'failed':
            raise RuntimeError('Latest checkpoint mirror failed: ' + json.dumps(result, sort_keys=True))
        if time.monotonic() >= deadline:
            raise RuntimeError('Latest checkpoint did not synchronize before timeout: '
                               + json.dumps(result, sort_keys=True))
        time.sleep(2)


def remove_workspace(run):
    """Remove only a private validation workspace, never published/training data."""
    run = Path(run).resolve()
    work = (VALIDATION_ROOT/'.work').resolve()
    relative = run.relative_to(work)
    if not relative.parts:
        raise ValueError('Cannot remove the validation workspace root')
    if run.exists():
        shutil.rmtree(run)
    try:
        work.rmdir()
    except (FileNotFoundError,OSError):
        pass


def status():
    value = read(STATE)
    output = value.get('work_output',value.get('output'))
    run = layout.rooted(output) if output else None
    progress = read(run/'progress.json') if run else {}
    proof = read(run/'preflight.json') if run else {}
    cases = progress.get('cases',[])
    if not cases and value.get('phase') == 'complete':
        latest = read(VALIDATION_ROOT/'metrics/latest_validation.json')
        if latest.get('run_id') == value.get('run_id',run.name if run else None):
            cases = latest.get('cases',[])+latest.get('segment_qualitative_cases',latest.get('qualitative_cases',[]))
            progress = dict(cases=cases,total_cases=latest.get('generation_total_cases',latest.get('total_cases',len(cases))))
    total = progress.get('total_cases',proof.get('cases',value.get('total_cases')))
    if total is None:
        total = len(read(layout.rooted('configs/plenoptic/fixed_validation_v2.json')).get('cases',[]))
        total += len(read(layout.rooted('configs/plenoptic/custom_validation.json')).get('cases',[]))
    from validation_live import run_directory
    if run and value.get('stage') == 'stage2':
        from validation_live import refinement_directory
        manifest = refinement_directory(run.name)/'manifest.json'
    else:
        manifest = run_directory(run.name)/'manifest.json' if run else None
    published = read(manifest) if manifest else {}
    if value.get('stage') == 'stage2' and not cases:
        cases = published.get('clips', [])
        total = value.get('total_cases', len(cases))
    return dict(public_state(value),active=identity_active(value),completed_cases=len(cases),total_cases=total,
                live_manifest=str(manifest) if manifest else None,
                published_clips=len(published.get('clips', [])),
                published_full_videos=len(published.get('groups', [])),
                last_completed_case=cases[-1]['case_id'] if cases else None)


def worker():
    if HOST != TARGET_HOST:
        raise RuntimeError('GPU validation is permitted only on h20-1')
    with STATE.with_suffix('.start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
    with STATE.with_suffix('.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        job = read(STATE)
        requested_gpu_indices = job.get('requested_gpu_indices')
        memory_batch_size = 1 if job['loss_only'] else job.get('video_batch_size', 1)
        run = layout.rooted(job.get('work_output',job['output']))
        pinned = run/'checkpoint.pt'
        if job.get('refinement_config'):
            return refine_worker(sys.modules[__name__], job, run, pinned)
        error = None
        cancelled = None
        signal.signal(signal.SIGTERM,interrupt_validation)
        signal.signal(signal.SIGINT,interrupt_validation)
        try:
            devices = wait_for_devices(
                STATE, requested_gpu_indices=requested_gpu_indices, video_batch_size=memory_batch_size
            )
            execution = ['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,devices))]
            inference_snapshot(Path(job['checkpoint']),pinned)
            capture_training_context(Path(job['checkpoint']),run)
            from validation_live import initialize_run
            initialize_run(run, pinned, read(pinned.with_suffix('.json')), job)
            update(STATE,'preflight',gpu_indices=devices,context_parallel_size=len(devices),
                   gpu_policy='fixed_cp4_batch_aware_memory_allow_shared_gpus')
            args = ['--checkpoint',str(pinned),'--output',str(run),
                    '--video-batch-size',str(job.get('video_batch_size',1)), '--publish-progress']
            if job['loss_only']: args.append('--loss-only')
            if job.get('skip_standard_videos'): args.append('--skip-standard-videos')
            if job.get('standard_only'):
                empty = run/'empty-custom-suite.json'
                prepare.save_json(empty, dict(schema=1, name='fixed-only', cases=[]))
                args += ['--custom-suite', str(empty)]
            if job.get('video_config'):
                update(STATE, 'preparing_video_batch')
                config_path = run/'video_inputs.json'
                suite_path = run/'video_batch_suite.json'
                prepare.save_json(config_path, job['video_config'])
                if job.get('pipeline_config'):
                    profile_path = run/'pipeline-config.json'
                    prepare.save_json(profile_path, job['pipeline_config']['profile'])
                    command([
                        'bash', '-c',
                        'source tools/activate_h20.sh && exec python tools/prepare_video_pipeline.py "$@"',
                        'prepare-video-pipeline', '--config', str(profile_path),
                        '--suite-output', str(suite_path),
                        '--receipt-output', str(run/'pipeline-preparation.json'),
                    ], timeout=1800)
                else:
                    command([
                        'bash', '-c',
                        'source tools/activate_h20.sh && exec python tools/prepare_custom_validation.py "$@"',
                        'prepare-video-batch',
                        '--batch-config', str(config_path),
                        '--output-dir', str(layout.rooted('datasets/custom_validation/batches')/run.name),
                        '--suite-output', str(suite_path),
                    ], timeout=1800)
                args += ['--custom-suite', str(suite_path)]
            command(execution+['bash','tools/run_validation_gpu.sh','--check-only',*args],timeout=600)
            proof = read(run/'preflight.json')
            # CPU preparation can outlast GPU availability; select again before loading.
            devices = wait_for_devices(STATE, requested_gpu_indices=requested_gpu_indices,
                                       video_batch_size=memory_batch_size)
            check_devices(devices, video_batch_size=memory_batch_size)
            execution = ['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,devices))]
            update(STATE,'evaluating',checkpoint_step=proof['checkpoint_step'],
                   total_cases=proof['cases'],gpu_indices=devices)
            command(execution+['bash','tools/run_validation_gpu.sh',*args],
                    timeout=job.get('evaluation_timeout_seconds', 5400))
            update(STATE,'rendering_report')
            report = read(run/'validation.json')
            if not job['loss_only']:
                for case in report['cases']+report.get('qualitative_cases',[]):
                    if case.get('generation_skipped'):
                        continue
                    if not (run/case['video_directory']/'comparison.json').is_file():
                        command(['bash','compare_inference.sh',str(run/case['video_directory'])],timeout=300)
                    if not case.get('qualitative_only'):
                        comparison = read(run/case['video_directory']/'comparison.json')
                        if 'image_metrics' not in comparison:
                            raise RuntimeError('Calibrated validation case has no image-error measurements')
                        case['image_metrics'] = comparison['image_metrics']
                prepare.save_json(run/'validation.json',report)
            if job.get('video_config'):
                update(STATE, 'assembling_full_videos')
                command([
                    'bash', '-c',
                    'source tools/activate_h20.sh && exec python tools/custom_validation_comparison.py "$@"',
                    'assemble-video-batch', '--assemble-run', str(run),
                ], timeout=1800)
            render_report(run)
        except ValidationCancelled as exc:
            cancelled = str(exc)
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            pinned.unlink(missing_ok=True)
        from validation_live import finish_run
        finish_run(run, 'cancelled' if cancelled else 'failed' if error else 'complete', error or cancelled)
        if not error and not cancelled:
            remove_workspace(run)
        # A concurrent CPU redraw or head mirror must include the completion receipt.
        with (VALIDATION_ROOT/'.report.lock').open('a') as report_lock:
            fcntl.flock(report_lock, fcntl.LOCK_EX)
            update(STATE,'cancelled' if cancelled else 'failed' if error else 'complete',error=error,
                   output=job['output'] if error or cancelled else str(layout.relative(VALIDATION_ROOT)),
                   report=str(layout.relative(REPORT_ENTRY)) if REPORT_ENTRY.exists() else None,
                   videos=str(layout.relative(VALIDATION_ROOT/'videos')),
                   curves=str(layout.relative(VALIDATION_ROOT/'curves')),run_report=None,
                   **({'cancellation_reason':cancelled} if cancelled else {}))
            prepare.save_json((run if error or cancelled else VALIDATION_ROOT/'metrics')/'workflow.json',read(STATE))
        if error: raise RuntimeError(error)


def dispatch():
    if HOST != HEAD_HOST: raise RuntimeError('Only h20-4 may submit to h20-1')
    with SUBMISSION.with_suffix('.start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
    job = read(SUBMISSION)
    run = layout.rooted(job.get('work_output',job['output']))
    pinned = run/'checkpoint.pt'
    error = None
    cancelled = None
    signal.signal(signal.SIGTERM,interrupt_validation)
    signal.signal(signal.SIGINT,interrupt_validation)
    try:
        update(SUBMISSION,'checking_target')
        remote_state = json.loads(remote(['python3','tools/validation_runner.py','status'],capture=True,timeout=30))
        if remote_state.get('active'): raise RuntimeError('h20-1 already has a running validation')

        requested_gpu_indices = job.get('requested_gpu_indices')
        wait_for_devices(
            SUBMISSION,
            requested_gpu_indices=requested_gpu_indices,
            on_target=True,
            video_batch_size=1 if job['loss_only'] else job.get('video_batch_size',1),
        )
        if job.get('refinement_config'):
            pin_refinement_checkpoint(sys.modules[__name__], job['refinement_config'], pinned)
        else:
            if not job.get('synchronize_latest_checkpoint'):
                raise RuntimeError('Stage-one validation requires a synchronized latest checkpoint')
            synchronized = synchronize_latest_checkpoint(SUBMISSION, pinned)
            checkpoint = Path(synchronized['checkpoint'])
            update(SUBMISSION, 'snapshot_ready', checkpoint=str(checkpoint),
                   checkpoint_step=synchronized['checkpoint_step'],
                   latest_checkpoint_sync=synchronized)
            capture_training_context(checkpoint,run)
        # Test actual visibility of this unique file, not equality of path names.
        refinement_submission = run/'refinement-submission.json'
        if job.get('refinement_config'):
            prepare.save_json(refinement_submission, job['refinement_config'])
        probe = run/'submission.json'
        prepare.save_json(probe,dict(run=job['output'],host=HOST))
        probe_result = remote(['python3','-c',
            'from pathlib import Path; import hashlib; p=Path('+repr(str(probe))+'); '
            'print(hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "missing")'],capture=True,timeout=30).strip()
        import hashlib
        shared = probe_result == hashlib.sha256(probe.read_bytes()).hexdigest()
        update(SUBMISSION,'submitting_to_h20_1',shared_visibility=shared)
        if not shared and job.get('refinement_config'):
            raise RuntimeError('The two-stage pipeline requires the migrated shared workspace on both hosts')
        if not shared:
            remote(['mkdir','-p',str(run)],timeout=30)
            # A new destination has no delta history; send one coherent snapshot.
            command(['nice','-n','19','ionice','-c','3','rsync','-a','--whole-file',
                     '--bwlimit=65536',
                     '--rsync-path=nice -n 19 ionice -c 3 rsync',
                     '-e',shlex.join(SSH),str(pinned),str(pinned.with_suffix('.json')),
                     'root@'+TARGET_ADDRESS+':'+str(run)+'/'],timeout=1800)
            if not job.get('refinement_config'):
                command(['rsync','-a','-e',shlex.join(SSH),str(run/'training_context'),
                         'root@'+TARGET_ADDRESS+':'+str(run)+'/'],timeout=60)
        update(SUBMISSION,'validating_on_h20_1')
        args = ['python3','tools/validation_runner.py','refine' if job.get('refinement_config') else 'start',
                '--checkpoint',str(pinned),'--output',job['output'],
                '--video-batch-size',str(job.get('video_batch_size',1))]
        if not job.get('refinement_config'):
            args.append('--synchronized-latest')
        if job['loss_only']: args.append('--loss-only')
        if job.get('skip_standard_videos'): args.append('--skip-standard-videos')
        if job.get('standard_only'): args.append('--standard-only')
        if requested_gpu_indices is not None:
            args += ['--gpus', ','.join(map(str, requested_gpu_indices))]
        if job.get('refinement_config'):
            from validation_live import sha256
            args += ['--refinement-config-file', str(refinement_submission),
                     '--refinement-config-sha256', sha256(refinement_submission)]
        elif not job['loss_only'] and not job.get('standard_only'):
            if job.get('pipeline_config'):
                args += ['--pipeline-config-json', json.dumps(job['pipeline_config'], ensure_ascii=False)]
            else:
                args += ['--video-config-json',
                         json.dumps(job.get('video_config') or dict(videos=[]), ensure_ascii=False)]
        args += ['--evaluation-timeout-seconds',
                 str(job.get('evaluation_timeout_seconds', 5400))]
        remote(args,timeout=30)
        while True:
            target = json.loads(remote(['python3','tools/validation_runner.py','status'],
                                       capture=True,timeout=30))
            if target.get('run_id') != run.name:
                raise RuntimeError('h20-1 validation run identity changed')
            if target.get('phase') == 'complete':
                break
            if target.get('phase') == 'cancelled':
                raise ValidationCancelled('Validation cancelled on h20-1')
            if target.get('phase') == 'failed' or not target.get('active'):
                raise RuntimeError('h20-1 validation exited: '+str(target.get('error')))
            update(SUBMISSION,'waiting_for_gpus' if target.get('phase') == 'waiting_for_gpus'
                   else 'validating_on_h20_1',target_phase=target.get('phase'),
                   completed_cases=target.get('completed_cases'),total_cases=target.get('total_cases'),
                   published_clips=target.get('published_clips'),
                   published_full_videos=target.get('published_full_videos'),
                   live_manifest=target.get('live_manifest'),
                   gpu_availability=target.get('gpu_availability'),context_parallel_size=4)
            time.sleep(15)
        # h20-1 computes and renders; h20-4 receives a copy for local viewing.
        completed = json.loads(remote(['python3','tools/validation_runner.py','status'],
                                      capture=True,timeout=30))
        if completed.get('phase') != 'complete':
            raise RuntimeError('h20-1 did not complete validation: '+str(completed.get('error')))
        if job.get('refinement_config'):
            update(SUBMISSION,'complete',checkpoint_step=completed['checkpoint_step'],
                report_host=TARGET_HOST,report_hosts=[HEAD_HOST,TARGET_HOST],
                output=completed['output'],report=completed.get('report'),videos=completed.get('videos'),
                live_manifest=completed.get('live_manifest'),published_clips=completed.get('published_clips'),
                published_full_videos=completed.get('published_full_videos'),error=None)
            return
        update(SUBMISSION,'copying_results_to_h20_4',report_host=TARGET_HOST)
        copy_reports_to_head()
        update(SUBMISSION,'complete',checkpoint_step=completed['checkpoint_step'],
               report_host=HEAD_HOST,report_hosts=[HEAD_HOST,TARGET_HOST],
               output=str(layout.relative(VALIDATION_ROOT)),
               report=str(layout.relative(REPORT_ENTRY)),
               videos=str(layout.relative(VALIDATION_ROOT/'videos')),
               curves=str(layout.relative(VALIDATION_ROOT/'curves')),
               run_report=None,error=None)
    except ValidationCancelled as exc:
        cancelled = str(exc)
        update(SUBMISSION,'cancelled',error=None,cancellation_reason=cancelled)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        update(SUBMISSION,'failed',error=error)
    finally:
        pinned.unlink(missing_ok=True)
        if not error and not cancelled:
            remove_workspace(run)
    if error: raise RuntimeError(error)


def external_video_geometry(entry):
    """Validate camera assumptions before submitting or waiting for GPUs."""
    if not isinstance(entry, dict):
        raise ValueError('Each video configuration must be an object')
    if 'side_degrees' in entry:
        raise ValueError('Batch videos use an external camera; replace side_degrees with azimuth')
    geometry = {key: entry.get(key, default) for key, default in (
        ('azimuth', 60.), ('distance_scale', 2.5), ('source_pitch', 55.), ('target_pitch', 25.))}
    values = [*geometry.values(), entry.get('hfov_degrees', 70.), entry.get('nominal_depth', 1.)]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError('Video camera geometry must use finite numbers')
    if not (0 < abs(geometry['azimuth']) < 120 and 1 < geometry['distance_scale'] <= 3
            and 0 < geometry['source_pitch'] < 85 and 0 < geometry['target_pitch'] < 85
            and 30 <= entry.get('hfov_degrees', 70.) <= 120
            and 0 < entry.get('nominal_depth', 1.) <= 10):
        raise ValueError('Video camera requires 0 < |azimuth| < 120, 1 < distance-scale <= 3, '
                         '0 < source/target-pitch < 85, 30 <= hfov <= 120, and 0 < nominal-depth <= 10')
    return geometry


def start(args, submission=False):

    requested_gpu_indices = (
        parse_gpu_ids(args.gpus) if args.gpus is not None else None
    )
    refinement_config = None
    video_config = pipeline_config = None
    if args.action == 'refine':
        if (args.loss_only or args.standard_only or args.video or args.video_prompt or args.video_config_json
                or args.pipeline_config or args.pipeline_config_json or args.preview_segments):
            raise ValueError('Refinement accepts a completed stage-one archive and refinement parameters')
        from refinement_inputs import make_request, verify_request
        if args.refinement_config_file:
            from refinement_inputs import checked_file
            request_path = checked_file(args.refinement_config_file, args.refinement_config_sha256)
            refinement_config = verify_request(read(request_path))
        elif args.refinement_config_json:
            refinement_config = verify_request(json.loads(args.refinement_config_json))
        else:
            refinement_config = make_request(args.source_run,args.source_video,strength=args.refinement_strength,
                steps=args.refinement_steps,guidance=args.refinement_guidance,seed=args.refinement_seed,
                contexts=args.refinement_contexts)
        args.checkpoint = refinement_config['checkpoint']
        args.video_batch_size = 1
    else:
        if (args.source_run or args.source_video or args.refinement_config_json or args.refinement_config_file
                or args.refinement_config_sha256
                or any(value is not None for value in (args.refinement_strength,args.refinement_steps,
                                                       args.refinement_guidance,args.refinement_seed,args.refinement_contexts))):
            raise ValueError('Use ./validate.sh refine for stage-two inputs')
        video_config = None
        pipeline_config = None
        from video_pipeline import DEFAULT_PROFILE, load_profile
        if (args.pipeline_config or args.pipeline_config_json) and (args.video or args.video_prompt or args.video_config_json or args.loss_only or args.standard_only):
            raise ValueError('A pipeline profile cannot be combined with --video, --loss-only or --standard-only')
        if args.standard_only and (args.video or args.video_prompt or args.video_config_json):
            raise ValueError('--standard-only cannot include custom videos')
        if args.pipeline_config_json:
            pipeline_config = json.loads(args.pipeline_config_json)
            pipeline_config['profile'] = load_profile(pipeline_config['profile'])
        elif args.pipeline_config:
            profile_path = layout.rooted(args.pipeline_config)
            import hashlib
            pipeline_config = dict(profile=load_profile(profile_path), source_file=str(profile_path),
                                   source_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest())
        elif not (args.standard_only or args.loss_only or args.video or args.video_prompt or args.video_config_json):
            profile_path = layout.rooted(DEFAULT_PROFILE)
            if profile_path.is_file() and read(profile_path).get('enabled'):
                import hashlib
                pipeline_config = dict(profile=load_profile(profile_path), source_file=str(profile_path),
                                       source_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest())
        if args.preview_segments:
            if not pipeline_config:
                raise ValueError('--preview-segments requires the default video profile or --pipeline-config')
            selected = [int(value) for value in args.preview_segments.split(',')]
            if not selected or any(value < 1 for value in selected) or len(selected) != len(set(selected)):
                raise ValueError('--preview-segments requires distinct positive segment numbers')
            pipeline_config['profile']['preview_segments'] = selected
        if pipeline_config:
            video_config = dict(videos=pipeline_config['profile']['videos'])
            args.skip_standard_videos = args.skip_standard_videos or pipeline_config['profile'].get('skip_standard_videos', False)
        if args.loss_only and (args.video or args.video_prompt or args.video_config_json):
            raise ValueError('--video cannot be combined with --loss-only')
        if args.video_config_json:
            if args.video_prompt:
                raise ValueError('Internal video configuration cannot be combined with --video-prompt')
            selected = json.loads(args.video_config_json)
            if not isinstance(selected, dict) or not isinstance(selected.get('videos'), list):
                raise ValueError('Internal video configuration requires a videos list')
            if selected['videos']:
                selected['videos'] = [dict(item, **external_video_geometry(item))
                                      for item in selected['videos']]
                video_config = selected
        elif args.video or args.video_prompt:
            if len(args.video) != len(args.video_prompt):
                raise ValueError('Provide exactly one --video-prompt for each --video')
            import re
            videos, names = [], set()
            geometry = external_video_geometry(dict(
                azimuth=args.video_azimuth, distance_scale=args.video_distance_scale,
                source_pitch=args.video_source_pitch, target_pitch=args.video_target_pitch))
            for index, (source, prompt) in enumerate(zip(args.video, args.video_prompt), 1):
                if not source.strip() or not prompt.strip():
                    raise ValueError('Video paths and prompts must be nonempty')
                name = re.sub(r'[^a-z0-9-]+', '-', Path(source).stem.lower()).strip('-')
                name = name or f'video-{index:02d}'
                if name in names:
                    raise ValueError(f'Duplicate output name: {name}; use distinct video filenames')
                names.add(name)
                videos.append(dict(id=name, source=source, prompt=prompt.strip(),
                                   source_camera=args.video_source_camera, **geometry))
            video_config = dict(videos=videos)
    if not refinement_config:
        if submission:
            if args.skip_standard_videos:
                raise ValueError('./validate.sh start always generates the standard validation videos; '
                                 'use loss when no videos should be generated')
            if args.synchronized_latest:
                raise ValueError('--synchronized-latest is reserved for the h20-4 dispatcher')
            if args.checkpoint is not None:
                raise ValueError('Stage-one validation always uses the synchronized latest checkpoint; '
                                 'do not pass --checkpoint')
            args.checkpoint = str(DEFAULT_CHECKPOINT)
        elif not args.synchronized_latest or args.checkpoint is None:
            raise RuntimeError('Run stage-one validation on h20-4 so the latest checkpoint can be synchronized')
    evaluation_timeout_seconds = args.evaluation_timeout_seconds
    if evaluation_timeout_seconds is None:
        evaluation_timeout_seconds = 21600 if video_config or refinement_config else 5400
    if evaluation_timeout_seconds <= 0:
        raise ValueError('--evaluation-timeout-seconds must be positive')
    path = SUBMISSION if submission else STATE

    with path.with_suffix('.start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if identity_active(read(path)):
            if args.wait or refinement_config: raise RuntimeError('A validation/refinement job is already running; finish it before submitting stage two')
            print('Validation is already running. Use ./validate.sh status.');return
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute(): checkpoint = layout.rooted(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError('Checkpoint is not visible on this host. Run ./validate.sh on h20-4 '
                                    'to synchronize and submit its latest saved snapshot to h20-1.')
        label = datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'-h20-1'
        output = args.output or 'outputs/validation/.work/'+label
        run = (layout.rooted(output)).resolve()
        relative = run.relative_to((VALIDATION_ROOT/'.work').resolve())
        if not relative.parts:
            raise ValueError('Use a unique run directory inside outputs/validation/.work')
        if (run/'validation.json').exists() or (run/'workflow.json').exists(): raise FileExistsError(run)
        run.mkdir(parents=True,exist_ok=True)
        output = str(layout.relative(run))
        log = layout.rooted(f'download-state/fixed-validation-{label}.log')
        action = '_dispatch' if submission else '_run'
        with log.open('a') as stream:
            process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),action],cwd=prepare.ROOT,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)

        prepare.save_json(path, dict(
            pid=process.pid,
            host=HOST,
            execution_host=TARGET_HOST,
            start_ticks=(Path('/proc')/str(process.pid)/'stat').read_text().split()[21],
            phase='starting',
            output=output,
            work_output=output,
            run_id=run.name,
            checkpoint=str(checkpoint),
            log=str(layout.relative(log)),
            loss_only=args.loss_only,
            standard_only=args.standard_only,
            skip_standard_videos=getattr(args, 'skip_standard_videos', False),
            requested_gpu_indices=requested_gpu_indices,
            video_config=video_config,
            pipeline_config=pipeline_config,
            refinement_config=refinement_config,
            synchronize_latest_checkpoint=bool(submission and not refinement_config),
            latest_checkpoint_synchronized=bool(args.synchronized_latest),
            stage='stage2' if refinement_config else 'stage1',
            video_batch_size=args.video_batch_size,
            evaluation_timeout_seconds=evaluation_timeout_seconds,
            training_control='none',
            updated_at=datetime.now().astimezone().isoformat(),
        ))

        print('Validation submitted to h20-1.' if submission else 'Validation started on h20-1.',flush=True)
        print('Videos: '+str(VALIDATION_ROOT/'refinements'/run.name if refinement_config else VALIDATION_ROOT/'videos')+'\nCurves: '+str(VALIDATION_ROOT/'curves')
              +'\nLive archive: '+str(VALIDATION_ROOT/('refinements' if refinement_config else 'runs')/run.name/'README.md')
              +'\nStatus: ./validate.sh status\nLog: '+str(layout.relative(log)),flush=True)
    if args.wait:
        while identity_active(read(path)): time.sleep(5)
        result = read(path)
        if result.get('phase') != 'complete': raise RuntimeError(str(result.get('error','Validation worker exited')))


def pin_refinement_checkpoint(api, request, destination):
    from validation_live import sha256
    source = Path(request['checkpoint'])
    if sha256(source) != request['checkpoint_sha256']:
        raise ValueError('Stage-one checkpoint changed before refinement')
    if not destination.exists():
        api.snapshot(source, destination)
    if sha256(destination) != request['checkpoint_sha256']:
        raise ValueError('Pinned refinement checkpoint differs from stage one')
    if source.with_suffix('.json').is_file() and source.resolve() != destination.resolve():
        metadata = destination.with_suffix('.json')
        if metadata.exists():
            if metadata.read_bytes() != source.with_suffix('.json').read_bytes():
                raise ValueError('Pinned refinement checkpoint metadata changed')
        else:
            shutil.copyfile(source.with_suffix('.json'), metadata)


def refine_worker(api, job, run, pinned):
    import validation_live as live
    from validation_live import atomic_json
    from validation_artifacts import retain_json
    request = job['refinement_config']
    error = cancelled = None
    signal.signal(signal.SIGTERM, api.interrupt_validation)
    signal.signal(signal.SIGINT, api.interrupt_validation)
    try:
        pin_refinement_checkpoint(api, request, pinned)
        live.initialize_refinement(run, request)
        archive = live.refinement_directory(run.name)
        request_path = archive/'request.json'
        api.update(api.STATE, 'preparing_refinement_inputs', checkpoint_step=request['checkpoint_step'],
                   stage='stage2', live_manifest=str(archive/'manifest.json'))
        api.command(['bash','-c',
            'source tools/activate_h20.sh && exec env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '
            'python tools/refinement_inputs.py "$@"', 'prepare-refinement',
            '--request',str(request_path),'--destination',str(archive/'inputs')],timeout=1800)
        args = ['--request',str(request_path),'--jobs',str(archive/'jobs.json'),
                '--checkpoint',str(pinned),'--output',str(run)]
        api.update(api.STATE, 'preflight')
        api.command(['bash','tools/run_validation_gpu.sh','--refine','--check-only',*args],timeout=1800)
        proof = api.read(run/'preflight.json')
        retain_json(archive/'preflight.json', proof)
        devices = api.wait_for_devices(api.STATE, job.get('requested_gpu_indices'), video_batch_size=1)
        api.check_devices(devices, video_batch_size=1)
        api.update(api.STATE, 'refining', gpu_indices=devices, context_parallel_size=4,
                   checkpoint_step=request['checkpoint_step'], total_cases=proof['cases'])
        api.command(['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,devices)),
                     'bash','tools/run_validation_gpu.sh','--refine',*args],timeout=job['evaluation_timeout_seconds'])
        retain_json(archive/'refinement.json', api.read(run/'refinement.json'))
        receipt = api.read(archive/'manifest.json')
        if len(receipt['groups']) != len(request['videos']) or len(receipt['clips']) != proof['cases']:
            raise ValueError('Refinement exited without publishing all requested videos')
    except api.ValidationCancelled as exc:
        cancelled = str(exc)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        pinned.unlink(missing_ok=True)
    phase = 'cancelled' if cancelled else 'failed' if error else 'complete'
    live.finish_refinement(run, phase, error or cancelled)
    api.update(api.STATE, phase, error=error, cancellation_reason=cancelled,
               output=str(layout.relative(live.refinement_directory(run.name))),
               videos=str(live.refinement_directory(run.name)), curves=None, report=str(live.refinement_directory(run.name)/'README.md'))
    atomic_json(live.refinement_directory(run.name)/'workflow.json',api.read(api.STATE))
    if not error and not cancelled:
        api.remove_workspace(run)
    if error:
        raise RuntimeError(error)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',choices=['start','loss','refine','status','report','_run','_dispatch'],default='start')
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('--source-run', help='Stage-one run ID or manifest path; default is its latest published run')
    source_group.add_argument('--source-video', help='Published full stage-one MP4 with adjacent inference.json')
    parser.add_argument('--refinement-config-json', help=argparse.SUPPRESS)
    parser.add_argument('--refinement-config-file', help=argparse.SUPPRESS)
    parser.add_argument('--refinement-config-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--refinement-strength', type=float, help='Actual initial noise sigma; default comes from the selected stage-one profile')
    parser.add_argument('--refinement-steps', type=int, help='Partial-noise denoising steps, default 20')
    parser.add_argument('--refinement-guidance', type=float, help='Refinement CFG, default 1.5')
    parser.add_argument('--refinement-seed', type=int)
    parser.add_argument('--refinement-contexts', type=int, choices=(1,2,3,4),
                        help='Ego conditioning copies during refinement; default follows the source run/profile')
    parser.add_argument('--loss-only',action='store_true')
    parser.add_argument('--preview-segments', help='Generate selected prepared clips, e.g. 3,8; fixed losses still run')
    parser.add_argument('--standard-only',action='store_true',
                        help='Run the fixed suite without the default ego-video profile')
    pipeline_group = parser.add_mutually_exclusive_group()
    pipeline_group.add_argument('--pipeline-config', help='Custom two-stage video profile; default is configs/plenoptic/video_pipeline.json')
    pipeline_group.add_argument('--pipeline-config-json', help=argparse.SUPPRESS)
    parser.add_argument('--skip-standard-videos',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--checkpoint', help=argparse.SUPPRESS)
    parser.add_argument('--synchronized-latest', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output')
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--gpus', metavar='GPU_IDS',
                    help='Exactly four comma-separated physical GPU indices on h20-1')
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument('--video', action='append', default=[], metavar='MP4',
                             help='Repeat for each full video; paired with --video-prompt in order')
    video_group.add_argument('--video-config-json', help=argparse.SUPPRESS)
    parser.add_argument('--video-prompt', action='append', default=[], metavar='TEXT')
    parser.add_argument('--video-batch-size', type=int, choices=(1, 2), default=1,
                        help='Custom clips generated together (default: 1 reduces GPU waiting)')
    parser.add_argument('--video-source-camera', choices=['rotation-proxy', 'fixed'],
                        default='rotation-proxy')
    parser.add_argument('--video-azimuth', type=float, default=60.,
                        help='Shared external-camera azimuth in degrees around the assumed activity center (default: 60)')
    parser.add_argument('--video-distance-scale', type=float, default=2.5,
                        help='Shared target/source distance ratio to the assumed activity center (default: 2.5; not metres)')
    parser.add_argument('--video-source-pitch', type=float, default=55.,
                        help='Shared assumed first source camera downward pitch in degrees (default: 55; not measured)')
    parser.add_argument('--video-target-pitch', type=float, default=25.,
                        help='Shared external camera downward pitch in degrees (default: 25)')
    parser.add_argument('--evaluation-timeout-seconds', type=int,
                        help='Default: 21600 for full-video batches, otherwise 5400')
    args = parser.parse_args()
    if args.action == 'loss':
        args.action = 'start'
        args.loss_only = True
    if args.refinement_config_json and args.refinement_config_file:
        parser.error('Use one refinement configuration transport')
    if HOST not in (HEAD_HOST,TARGET_HOST):
        raise RuntimeError('Use h20-4 to submit or h20-1 to evaluate; other training nodes are not permitted')
    STATE.parent.mkdir(parents=True,exist_ok=True)
    if args.action == 'status':
        if HOST == TARGET_HOST: print(json.dumps(dict(status(),validation_entry=str(VALIDATION_ROOT)),ensure_ascii=False,indent=2))
        else:
            value = read(SUBMISSION)
            target = json.loads(remote(['python3','tools/validation_runner.py','status'],capture=True,timeout=30))
            print(json.dumps(dict(submission=public_state(value),active=identity_active(value),h20_1=target,
                                 validation_entry=str(VALIDATION_ROOT)),ensure_ascii=False,indent=2))
    elif args.action == 'report':
        # Completed metrics are protected by the report lock; a GPU job may keep
        # waiting or evaluating in its private workspace while they are redrawn.
        if HOST == TARGET_HOST:
            render_report()
        else:
            refresh_training_context()
            remote(['python3','tools/validation_runner.py','report'],timeout=600)
            copy_reports_to_head()
        print('Videos on '+HOST+': '+str(VALIDATION_ROOT/'videos')
              +'\nCurves on '+HOST+': '+str(VALIDATION_ROOT/'curves'),flush=True)
    elif args.action == '_run': worker()
    elif args.action == '_dispatch': dispatch()
    else: start(args,submission=HOST==HEAD_HOST)


if __name__ == '__main__':
    main()
