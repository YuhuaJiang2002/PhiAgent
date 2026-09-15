#!/usr/bin/env python3
"""Export a pinned training checkpoint for inference without optimizer state."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export_checkpoint(source, destination):
    import torch
    torch.set_num_threads(1)
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError('Use a fresh inference snapshot path')
    started = time.monotonic()
    saved = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    required = {'schema', 'trainer_version', 'step', 'micro_step', 'trainable', 'config',
                'dataset_hash', 'caption_ledgers', 'world_size',
                'base_revision', 'vae_revision', 'reason_revision'}
    if (not required <= saved.keys() or saved['trainer_version'] not in (3, 4, 5)
            or saved['step'] < 1):
        raise ValueError('Require a completed trainer-version-3/4/5 checkpoint with provenance metadata')
    # Keep every inference/provenance field, including ledgers proving unseen scenes.
    payload = {key: value for key, value in saved.items() if key not in {'optimizer', 'rng'}}
    source_sha = digest_file(source)
    payload['inference_snapshot'] = dict(schema=1, source_sha256=source_sha,
        source_bytes=source.stat().st_size, omitted_fields=sorted(set(saved) & {'optimizer', 'rng'}))
    temporary = destination.with_suffix('.packing.tmp')
    metadata_path = destination.with_suffix('.json')
    if temporary.exists() or metadata_path.exists():
        raise FileExistsError('Inference snapshot workspace is not fresh')
    try:
        torch.save(payload, temporary)
        result = dict(payload['inference_snapshot'], checkpoint_step=saved['step'],
            snapshot_sha256=digest_file(temporary), snapshot_bytes=temporary.stat().st_size,
            tensor_count=len(payload['trainable']), seconds=round(time.monotonic()-started, 3),
            created_at=datetime.now().astimezone().isoformat())
        os.replace(temporary, destination)
        metadata_path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        return result
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.source, args.output)), flush=True)
