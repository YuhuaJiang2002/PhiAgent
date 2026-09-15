"""Portable code/data layout, including stable identities for old checkpoints."""
import copy
import json
import os
from pathlib import Path
import socket


ROOT = Path(__file__).resolve().parents[1]
SHARED_FILESYSTEM = True
DATASETS_ROOT = (ROOT / '../DATASETS').resolve()
OUTPUTS_ROOT = (ROOT / '../OUTPUTS').resolve()
CHECKPOINTS_ROOT = (ROOT / '../checkpoints').resolve()
ENVIRONMENTS_ROOT = (ROOT / '../ENVIRONMENTS').resolve()
TRAIN_ENV = ENVIRONMENTS_ROOT / 'plenoptic-h20'
DOWNLOAD_ENV = ENVIRONMENTS_ROOT / 'plenoptic-download'
SHARED_CACHE_ROOT = (ROOT / '../CACHE/plenoptic-repro/shared').resolve()
CACHE_ROOT = SHARED_CACHE_ROOT.parent / 'nodes' / socket.gethostname()
STATE_ROOT = OUTPUTS_ROOT / '.runtime' / socket.gethostname()
SHARED_STATE_ROOT = OUTPUTS_ROOT / '.runtime/shared'
INPUTS_ROOT = DATASETS_ROOT / 'inputs'
SSH_KEY = SHARED_CACHE_ROOT / 'cluster-ssh/id_ed25519'
LEGACY_ROOT = Path('/data3/llq/plenoptic-repro')


def dataset_root(name, kind='extracted'):
    names = {'syncam': 'symcam', 'symcam': 'symcam', 'multicam': 'multicam'}
    if name not in names or kind not in ('raw', 'extracted'):
        raise ValueError('Unknown first-stage dataset or storage kind')
    return DATASETS_ROOT / names[name] / kind


def rooted(value):
    """Resolve filesystem paths while leaving manifest/caption identities intact.

    Released scene manifests and caption ledgers retain their original logical
    strings. Resolving those strings here avoids changing checkpoint dataset
    hashes merely because the physical storage directory moved.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        for previous in (LEGACY_ROOT, ROOT):
            try:
                path = path.relative_to(previous)
                break
            except ValueError:
                pass
        else:
            return Path(os.path.normpath(path))
    mappings = [
        ('datasets/extracted/syncam', dataset_root('syncam')),
        ('datasets/extracted/multicam', dataset_root('multicam')),
        ('datasets/raw/SynCamVideo', dataset_root('syncam', 'raw')),
        ('datasets/raw/MultiCamVideo', dataset_root('multicam', 'raw')),
        ('datasets', DATASETS_ROOT), ('outputs', OUTPUTS_ROOT),
        ('checkpoints', CHECKPOINTS_ROOT), ('download-state', STATE_ROOT),
        ('inputs', INPUTS_ROOT), ('tools/download-env', DOWNLOAD_ENV),
        ('cosmos-transfer2.5/.venv', TRAIN_ENV),
    ]
    for prefix, destination in mappings:
        try:
            return Path(os.path.normpath(destination / path.relative_to(prefix)))
        except ValueError:
            pass
    return Path(os.path.normpath(ROOT / path))


def relative(value):
    """Return a command/report path relative to the code directory."""
    return Path(os.path.relpath(rooted(value), ROOT))


def require_workspace_path(value):
    path = rooted(value).resolve()
    for base in (ROOT, DATASETS_ROOT, OUTPUTS_ROOT, CHECKPOINTS_ROOT):
        if path == base or base in path.parents:
            return path
    raise ValueError('Path is outside the configured project/data/output/checkpoint directories: ' + str(path))


def output_path(value):
    path = rooted(value).resolve()
    if path == OUTPUTS_ROOT or OUTPUTS_ROOT not in path.parents:
        raise ValueError('Training output must be a run directory under ../OUTPUTS')
    return path


def logical(value):
    """Canonical storage identity used only for exact-resume path comparisons."""
    path = rooted(value).resolve()
    mappings = [
        (dataset_root('syncam'), 'datasets/extracted/syncam'),
        (dataset_root('multicam'), 'datasets/extracted/multicam'),
        (dataset_root('syncam', 'raw'), 'datasets/raw/SynCamVideo'),
        (dataset_root('multicam', 'raw'), 'datasets/raw/MultiCamVideo'),
        (DATASETS_ROOT, 'datasets'), (OUTPUTS_ROOT, 'outputs'),
        (CHECKPOINTS_ROOT, 'checkpoints'), (STATE_ROOT, 'download-state'),
        (ROOT, ''),
    ]
    for base, prefix in mappings:
        try:
            return str(Path(prefix) / path.relative_to(base))
        except ValueError:
            pass
    return str(path)


def canonical_data_config(data):
    result = copy.deepcopy(data)
    for key in ('captions', 'generated_manifest'):
        if result.get(key):
            result[key] = logical(result[key])
    if 'manifests' in result:
        result['manifests'] = [logical(value) for value in result['manifests']]
    return result


def resume_config_equal(saved, current, keys):
    for key in keys:
        a, b = saved.get(key), current.get(key)
        if key == 'data':
            a, b = canonical_data_config(a), canonical_data_config(b)
        if a != b:
            return False
    return True


def conflicting_training():
    """Read-only check: protect existing jobs even when their state is local."""
    conflicts = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            tokens = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if any(Path(token).name in ('train_plenoptic.py', 'train_tabletop.py', 'launch_plenoptic.sh')
                   for token in tokens):
                conflicts.append({'pid': int(proc.name), 'cwd': os.readlink(proc / 'cwd')})
        except OSError:
            continue
    return conflicts


def ensure_migration_complete():
    state = OUTPUTS_ROOT / 'migration/20260912-shared/migration-state.json'
    if state.is_file():
        record = json.loads(state.read_text())
        if record.get('status') != 'complete':
            raise RuntimeError('Shared migration has not passed all data, checkpoint and node checks yet')


def handoff_source_mirror():
    """Prevent a background source copy from overwriting a newly resumed job."""
    if socket.gethostname() != 'h20-4':
        return
    import fcntl
    from datetime import datetime, timezone
    audit = OUTPUTS_ROOT / 'migration/20260912-shared'
    mirror = audit / 'mirror-status.json'
    handoff = audit / 'mirror-handoff.json'
    if not mirror.is_file():
        return
    with (audit / 'source-mirror-handoff.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A completed source checkpoint is still being mirrored; check tools/migration_status.py')
        if handoff.is_file() and json.loads(handoff.read_text()).get('status') == 'shared_training':
            return
        if any(Path(item['cwd']) == LEGACY_ROOT for item in conflicting_training()):
            raise RuntimeError('The original local training is still active')
        state = json.loads(mirror.read_text())
        if state.get('status') not in ('watching_source_training', 'complete'):
            raise RuntimeError('The source checkpoint mirror has not finished its current snapshot')
        original = LEGACY_ROOT / 'outputs/basic_stage1_24gpu/latest.json'
        copied = OUTPUTS_ROOT / 'basic_stage1_24gpu/latest.json'
        copied_step = json.loads(copied.read_text()).get('step') if copied.is_file() else None
        original_step = json.loads(original.read_text()).get('step') if original.is_file() else copied_step
        if copied_step is None or copied_step != original_step or copied_step != state.get('step'):
            raise RuntimeError('The shared checkpoint is behind the completed source checkpoint; wait for the mirror')
        checkpoint = OUTPUTS_ROOT / 'basic_stage1_24gpu/latest.pt'
        receipt = json.loads((audit / 'training-checkpoints/latest.pt.json').read_text())
        stamp = checkpoint.stat()
        if receipt.get('verified') != [stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns, stamp.st_ino]:
            raise RuntimeError('The shared checkpoint differs from its completed-copy receipt')
        record = dict(status='shared_training', checkpoint_step=copied_step,
                      project=str(ROOT), handed_off_at=datetime.now(timezone.utc).isoformat())
        temporary = handoff.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(handoff)


def description():
    return {'root': str(ROOT), 'datasets': str(DATASETS_ROOT), 'outputs': str(OUTPUTS_ROOT),
            'checkpoints': str(CHECKPOINTS_ROOT), 'environment': str(TRAIN_ENV),
            'cache': str(CACHE_ROOT), 'node_state': str(STATE_ROOT),
            'shared_filesystem': SHARED_FILESYSTEM}
