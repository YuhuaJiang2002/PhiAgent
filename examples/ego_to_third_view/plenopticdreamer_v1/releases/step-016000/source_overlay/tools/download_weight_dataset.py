#!/usr/bin/env python3
import plenoptic_paths as layout
import ast
from collections import deque
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
import uuid
HERE = Path(__file__).resolve().parent
ROOT = layout.ROOT
PREPARE = HERE / 'prepare_plenoptic.py'
# Inherit the interpreter that successfully started this process; do not resolve
# a venv interpreter symlink to the underlying system Python.
PYTHON = Path(sys.executable)
QUEUE = layout.SHARED_CACHE_ROOT / os.environ.get('PLENOPTIC_QUEUE_NAME', 'assets-queue')
STEPS = [['weights'], ['syncam'], ['multicam']]
ATTEMPTS = int(os.getenv('PLENOPTIC_QUEUE_ATTEMPTS', '3'))
RETRY_SECONDS = int(os.getenv('PLENOPTIC_QUEUE_RETRY_SECONDS', '60'))

def save(path, data):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)

def environment():
    env = os.environ.copy()
    env.pop('HF_TOKEN', None)
    env.pop('HUGGING_FACE_HUB_TOKEN', None)
    env['PYTHONUNBUFFERED'] = '1'
    env['PLENOPTIC_ROOT'] = str(ROOT)
    return env

def load_prepare():
    if not PREPARE.is_file():
        raise RuntimeError(f'缺少同目录文件：{PREPARE.name}')
    ast.parse(PREPARE.read_text())
    spec = importlib.util.spec_from_file_location('prepare', PREPARE)
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    if prepare.ROOT != ROOT:
        raise RuntimeError('两个脚本的项目根目录不一致；请一并更新 prepare_plenoptic.py 和 download_dataset.py')
    return prepare

def checked_step(arguments):
    prepare = load_prepare()
    from huggingface_hub import get_token, get_hf_file_metadata, hf_hub_url, hf_hub_download
    checks = layout.rooted('download-state/verified')
    checks.mkdir(parents=True, exist_ok=True)
    manifest_path = layout.rooted('download-state/asset-manifest.json')
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    names = ['base', 'vae', 'reason'] if arguments == ['weights'] else arguments
    for name in names:
        repo, kind, revision, directory, filenames = prepare.JOBS[name]
        destination = prepare.download_directory(directory)
        public_asset = kind == 'dataset' or name == 'reason'
        token = False if public_asset else (get_token() or False)
        for filename in filenames:
            path = layout.rooted(directory) / filename
            key = hashlib.sha256(f'{repo}/{revision}/{filename}'.encode()).hexdigest()
            receipt = checks / (key + '.json')
            record = json.loads(receipt.read_text()) if receipt.exists() else {}
            pinned = manifest.get(name, {}).get(filename, {})
            pinned_hash = pinned.get('etag')
            if isinstance(pinned_hash, str) and re.fullmatch(r'(?:[a-f0-9]{40}|[a-f0-9]{64})', pinned_hash):
                if record and (record.get('etag') != pinned_hash or record.get('size') != pinned['size']):
                    raise RuntimeError(f'固定版本清单与缓存元数据不符：{name}/{filename}')
                if not record:
                    record = {'size': pinned['size'], 'etag': pinned_hash}
                    save(receipt, record)

            def fingerprint():
                stat = path.stat()
                return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]
            if path.is_file() and record.get('verified') == fingerprint():
                print(f'跳过已校验文件：{name}/{filename}', flush=True)
                continue
            if 'etag' not in record:
                print(f'读取固定版本元数据：{name}/{filename}', flush=True)
                url = hf_hub_url(repo, filename, repo_type=kind, revision=revision)
                metadata = prepare.retry(lambda: get_hf_file_metadata(url, token=token, timeout=60))
                if metadata.size is None or metadata.etag is None or metadata.commit_hash != revision:
                    raise RuntimeError(f'无法验证固定版本：{filename}')
                record = {'size': metadata.size, 'etag': metadata.etag.strip('"')}
                if pinned.get('size') is not None and record['size'] != pinned['size']:
                    raise RuntimeError(f'固定版本清单与远端大小不符：{name}/{filename}')
                save(receipt, record)
            if not path.is_file() or path.stat().st_size != record['size']:
                print(f'下载/续传：{name}/{filename}', flush=True)
                if public_asset and record['size'] > 1024**3 and shutil.which('aria2c'):
                    # Public assets only: no HF credentials are passed to aria2 or redirects.
                    staging = prepare.CACHE_ROOT / 'aria2' / name
                    staging.mkdir(parents=True, exist_ok=True)
                    partial = staging / (filename + '.partial')
                    candidates = list((destination / '.cache/huggingface/download').glob('*.' + record['etag'] + '.incomplete'))
                    if not partial.exists() and len(candidates) == 1:
                        candidates[0].rename(partial)
                    url = hf_hub_url(repo, filename, repo_type=kind, revision=revision)
                    subprocess.run([
                        'aria2c', '--continue=true', '--auto-file-renaming=false',
                        '--allow-overwrite=false', '--file-allocation=none',
                        '--max-connection-per-server=16', '--split=16', '--min-split-size=4M',
                        '--lowest-speed-limit=128K',
                        '--max-tries=20', '--retry-wait=10', '--connect-timeout=20', '--timeout=60',
                        '--summary-interval=60', '--console-log-level=warn', '--download-result=hide',
                        '--dir=' + str(staging), '--out=' + partial.name,
                        url,
                    ], check=True)
                    if partial.stat().st_size != record['size']:
                        raise RuntimeError(f'完整性检查失败（大小）：{filename}')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partial.replace(path)
                else:
                    prepare.retry(lambda: hf_hub_download(repo_id=repo, repo_type=kind, revision=revision, filename=filename, local_dir=str(destination), token=token, force_download=False))
            if not path.is_file() or path.stat().st_size != record['size']:
                raise RuntimeError(f'完整性检查失败（大小）：{filename}')
            print(f'校验完整文件：{name}/{filename}', flush=True)
            before = fingerprint()
            etag = record['etag']
            digest = hashlib.sha256() if len(etag) == 64 else hashlib.sha1()
            if len(etag) == 40:
                digest.update(f"blob {record['size']}\x00".encode())
            elif len(etag) != 64:
                raise RuntimeError(f'不支持的校验格式：{filename}')
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != etag or before != fingerprint():
                raise RuntimeError(f'完整性检查失败（哈希或文件变化）：{filename}')
            record['verified'] = before
            save(receipt, record)
            print(f'校验完成：{name}/{filename}', flush=True)
        prepare.save_json(layout.rooted('download-state') / f'{name}.json', {
            'repo': repo, 'revision': revision, 'directory': directory,
            'files': filenames, 'status': 'verified',
        })
        if name in ('syncam', 'multicam'):
            subprocess.run([str(PYTHON), str(HERE / 'prepare_datasets.py'), name], check=True)
            subprocess.run([str(PYTHON), str(HERE / '01_inspect_plenoptic.py'), '--datasets', name], check=True)

def latest():
    path = QUEUE / 'latest.json'
    if not path.exists():
        raise RuntimeError('还没有启动记录，请先执行 start')
    return json.loads(path.read_text())

def latest_run_dir(info):
    path = Path(info['run_dir'])
    if not path.is_absolute():
        return layout.rooted(path)
    # Old releases recorded absolute paths. Prefer the migrated local run.
    migrated = QUEUE / path.name
    return migrated if migrated.is_dir() else path

def status():
    info = latest()
    run_dir = latest_run_dir(info)
    state = json.loads((run_dir / 'status.json').read_text())
    with (QUEUE / 'run.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            active = False
        except BlockingIOError:
            active = True
    state['运行锁仍被持有'] = active
    if not active and state['status'] in ('starting', 'running', 'retrying'):
        state['提示'] = '任务已退出；可用 start 重启并复用断点'
    print(json.dumps(state, ensure_ascii=False, indent=2))

def selected_steps(arguments):
    groups = {'all': ['weights', 'syncam', 'multicam'],
              'datasets': ['syncam', 'multicam'],
              'weights': ['weights'], 'syncam': ['syncam'], 'multicam': ['multicam'],
              'base': ['base'], 'vae': ['vae'], 'reason': ['reason']}
    names = []
    for argument in arguments or ['all']:
        if argument not in groups:
            raise RuntimeError('下载任务请选择 all、datasets、weights、base、vae、reason、syncam 或 multicam')
        names.extend(groups[argument])
    return [[name] for name in dict.fromkeys(names)]

def start(arguments=None):
    steps = selected_steps(arguments)
    legacy_lock = layout.rooted('download-queue/run.lock')
    if legacy_lock.exists():
        with legacy_lock.open('a') as legacy:
            try:
                fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print('旧的 background_download.py 队列仍在运行；请先等它结束。')
                return
    if ATTEMPTS < 1 or RETRY_SECONDS < 0:
        raise RuntimeError('重试次数必须 >= 1，重试间隔必须 >= 0')
    prepare = load_prepare()
    # Re-enter the locally recreated environment, even when invoked from conda
    # base. Worker and step processes then inherit this same interpreter.
    prepare.use_download_environment()
    try:
        from huggingface_hub import get_token, get_hf_file_metadata, hf_hub_url, hf_hub_download
    except ImportError as exc:
        raise RuntimeError('当前下载环境依赖不完整；请执行：python3 prepare_plenoptic.py setup') from exc
    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('已有队列持有运行锁，本次没有重复启动。')
            return
        name = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
        run_dir = QUEUE / name
        run_dir.mkdir()
        log_path = run_dir / 'queue.log'
        log_relative = str(layout.relative(log_path))
        save(run_dir / 'status.json', {'status': 'starting', 'log': log_relative, 'steps': steps})
        try:
            with log_path.open('ab', buffering=0) as log:
                process = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '_worker', str(run_dir), str(lock.fileno())], cwd=HERE, env=environment(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        except OSError as exc:
            save(run_dir / 'status.json', {'status': 'failed', 'log': log_relative, 'error': f'{type(exc).__name__}: {exc}'})
            raise
        save(QUEUE / 'latest.json', {'pid': process.pid, 'run_dir': str(layout.relative(run_dir))})
        print(f'后台已启动，PID={process.pid}\n日志：{log_path}\n用 status 或 log 查看进度。')

def fatal_failure(log_path, offset):
    with log_path.open('rb') as handle:
        handle.seek(max(offset, log_path.stat().st_size - 65536))
        text = handle.read().decode(errors='replace').lower()
    return any((word in text for word in ('401 client error', '403 client error', '404 client error', 'gatedrepoerror', 'no space left on device', 'permission denied', 'permissionerror', 'syntaxerror', 'modulenotfounderror', '完整性检查失败', '无法验证固定版本', '不支持的校验格式', '低于脚本预留值', 'unauthorized', 'authentication required')))

def worker(run_dir, lock_fd):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    state_path = run_dir / 'status.json'
    log_path = run_dir / 'queue.log'
    steps = json.loads(state_path.read_text()).get('steps', STEPS)
    state = {'pid': os.getpid(), 'status': 'running', 'completed': [], 'steps': steps, 'log': str(layout.relative(log_path))}
    child = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)

    def update(**changes):
        state.update(changes)
        state['updated_at'] = datetime.now().astimezone().isoformat()
        save(state_path, state)
    try:
        for index, step in enumerate(steps, 1):
            label = ' '.join(step)
            for attempt in range(1, ATTEMPTS + 1):
                update(status='running', step=f'{index}/{len(steps)} {label}', attempt=attempt)
                print(f"\n[{state['updated_at']}] 开始 {label}，第 {attempt}/{ATTEMPTS} 轮", flush=True)
                offset = log_path.stat().st_size
                child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '_step', *step], cwd=HERE, env=environment(), stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock_fd,))
                update(child_pid=child.pid)
                code = child.wait()
                child = None
                update(child_pid=None, last_exit_code=code)
                if code == 0:
                    state['completed'].append(label)
                    update()
                    print(f'完成：{label}', flush=True)
                    break
                if code < 0 or attempt == ATTEMPTS or fatal_failure(log_path, offset):
                    update(status='failed')
                    print(f'队列停止在 {label}，exit={code}；未执行后续步骤。', flush=True)
                    return
                update(status='retrying')
                print(f'{RETRY_SECONDS} 秒后重跑当前步骤，保留原缓存。', flush=True)
                time.sleep(RETRY_SECONDS)
        update(status='completed', step='全部完成')
        print('\n所选下载任务及校验全部完成。', flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error_type=type(exc).__name__, child_pid=None)
        raise
    finally:
        os.close(lock_fd)

def log():
    path = latest_run_dir(latest()) / 'queue.log'
    with path.open(errors='replace') as handle:
        for line in deque(handle, maxlen=30):
            print(line, end='', flush=True)
        while True:
            line = handle.readline()
            if line:
                print(line, end='', flush=True)
            else:
                time.sleep(1)
if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'status'
    try:
        if action == '_worker':
            worker(Path(sys.argv[2]), int(sys.argv[3]))
        elif action == '_step':
            checked_step(sys.argv[2:])
        elif action == 'start':
            start(sys.argv[2:])
        elif action == 'status':
            status()
        elif action == 'log':
            log()
        else:
            raise RuntimeError('用法：python3 tools/download_weight_dataset.py start [all|datasets|weights|base|vae|reason|syncam|multicam] 或 status|log')
    except KeyboardInterrupt:
        print('\n已退出当前进程。退出 log 查看不会停止后台队列。')
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
