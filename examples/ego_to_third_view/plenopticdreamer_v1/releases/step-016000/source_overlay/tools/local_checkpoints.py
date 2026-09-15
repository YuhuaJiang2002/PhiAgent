#!/usr/bin/env python3
"""Bind fixed public checkpoint registry entries to verified local assets."""
import plenoptic_paths as layout
import hashlib
import json

import prepare_plenoptic as prepare

# Registry UUIDs from the pinned Cosmos source; only these public assets are bound.
REGISTRY = {
    'base': ['81edfebe-bd6a-4039-8c1d-737df1a790bf'],
    'vae': ['685afcaa-4de2-42fe-b7b9-69f7a2dee4d8'],
    'reason': ['7219c6c7-f878-4137-bbdb-76842ea85e70', 'cb3e3ffa-7b08-4c34-822d-61c7aa31a14f'],
}


def configure_local_checkpoints(require=()):
    """Call in the training/validation process before constructing any model.

    The registry's existing download cache is populated only for files with valid
    complete-file receipts. This bypasses the source's separate uvx/HF downloader.
    Bindings last for this process; no absolute path is written into the source.
    """
    from cosmos_oss.checkpoints_predict2 import register_checkpoints
    from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import CheckpointConfig

    register_checkpoints()
    bound = {}
    for name, ids in REGISTRY.items():
        repo, _, revision, directory, files = prepare.JOBS[name]
        complete = True
        for filename in files:
            path = layout.rooted(directory) / filename
            key = hashlib.sha256(f'{repo}/{revision}/{filename}'.encode()).hexdigest()
            receipt = layout.rooted('download-state/verified') / (key + '.json')
            if not path.is_file() or not receipt.is_file():
                complete = False
                break
            s = path.stat()
            record = json.loads(receipt.read_text())
            if record.get('verified') != [s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino]:
                complete = False
                break
        if not complete:
            if name in require:
                raise RuntimeError(f'{name}: missing or unverified assets; resume the download queue first')
            continue
        for identifier in ids:
            checkpoint = CheckpointConfig.from_uri(identifier)
            hf = checkpoint.hf
            if (hf.repository, hf.revision) != (repo, revision):
                raise RuntimeError(f'Checkpoint registry revision changed: {name}')
            filename = getattr(hf, 'filename', None)
            if filename is not None and filename not in files:
                raise RuntimeError(f'Checkpoint registry filename changed: {name}')
            target = layout.rooted(directory)
            if filename is not None:
                target /= filename
            # This private cache attribute is verified against the pinned source.
            hf._path = str(target)
            bound[identifier] = str(layout.relative(target))
    return bound


if __name__ == '__main__':
    print(json.dumps(configure_local_checkpoints(), ensure_ascii=False, indent=2))
