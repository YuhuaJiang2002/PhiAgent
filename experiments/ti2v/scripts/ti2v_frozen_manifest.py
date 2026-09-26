"""Hash only immutable inputs/configuration/source, never live run outputs."""
import hashlib
from pathlib import Path

REQUIRED = (
    'protocol.json', 'experiment.json', 'inputs.json', 'parent-selections.json',
    'planning/plans-frozen.json', 'planning/state.json', 'git-state.json',
)

def freeze_manifest(root):
    root = Path(root).resolve()
    paths = [root / name for name in REQUIRED]
    paths += [p for p in (root / 'source').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    if not (root / 'source').is_dir() or not paths[len(REQUIRED):]:
        raise ValueError('A nonempty frozen source tree is required')
    result = {}
    for path in sorted(paths):
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('Missing or escaped immutable artifact: ' + str(path))
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        result[str(path.relative_to(root))] = digest.hexdigest()
    return result
