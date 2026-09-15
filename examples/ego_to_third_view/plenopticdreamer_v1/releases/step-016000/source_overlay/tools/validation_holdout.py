"""Prove that supplemental scenes are unused by a finite stage-1 training plan."""
import plenoptic_paths as layout
from collections import defaultdict
import hashlib
import json


def training_identity(saved):
    cfg = saved['config']
    # Storage relocation preserves the manifest identity and sampling plan.
    data = layout.canonical_data_config(cfg['data'])
    return dict(dataset_hash=saved['dataset_hash'], world_size=saved['world_size'],
        context_parallel_size=cfg['context_parallel_size'], seed=cfg['seed'],
        gradient_accumulation=cfg['gradient_accumulation'], max_steps=cfg['max_steps'],
        phase=cfg.get('phase'), manifests=data['manifests'],
        generated_manifest=data.get('generated_manifest'))


def identity_hash(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def planned_and_used_scenes(saved, scenes, permutation=None):
    """Match DistributedSampler and the trainer's optimizer/micro-step accounting.

    The extra scenes remain in the source manifest, but are never drawn by this
    specific finite training schedule. A changed recipe or a seen scene must
    invalidate this cohort; it is not a permanently excluded training split.
    """
    cfg = saved['config']
    if cfg.get('phase') != 'stage1' or cfg['data'].get('generated_manifest'):
        raise ValueError('The supplemental cohort is valid only for its stage-1 training plan')
    world, cp = saved['world_size'], cfg['context_parallel_size']
    if world % cp or saved['micro_step'] != saved['step'] * cfg['gradient_accumulation']:
        raise ValueError('Training topology or sample accounting is inconsistent')
    if not 0 <= saved['step'] <= cfg['max_steps']:
        raise ValueError('Checkpoint exceeds the protected training horizon')
    ledgers = saved.get('caption_ledgers')
    if not isinstance(ledgers, list) or len(ledgers) != world:
        raise ValueError('Cannot prove unseen scenes without all checkpoint caption ledgers')
    used_paths = {path for ledger in ledgers if ledger for path in ledger}
    used = {(s['dataset'], s['scene_id']) for s in scenes
            if used_paths.intersection(s['videos'].values())}
    items = [(s['dataset'], s['scene_id']) for s in scenes for _ in sorted(s['videos'])]
    if not items:
        raise ValueError('Training manifest is empty')
    dp = world // cp
    per_epoch = (len(items) + dp - 1) // dp
    micro_steps = cfg['max_steps'] * cfg['gradient_accumulation']
    if micro_steps >= per_epoch:
        return set(items), used  # A complete epoch visits every scene.
    if permutation is None:
        import torch
        generator = torch.Generator().manual_seed(cfg['seed'])
        permutation = torch.randperm(len(items), generator=generator).tolist()
    if sorted(permutation) != list(range(len(items))):
        raise ValueError('Invalid sampler permutation')
    # The union of the first m samples from every DP rank is the first m*DP
    # elements of the common permutation (no padding occurs before epoch end).
    planned = {items[i] for i in permutation[:micro_steps * dp]}
    return planned, used


def check_protected_scenes(saved, scenes, protection, selected, permutation=None):
    identity = training_identity(saved)
    if identity_hash(identity) != protection['training_identity_sha256']:
        raise ValueError('Training plan changed; supplemental validation scenes must be requalified')
    planned, used = planned_and_used_scenes(saved, scenes, permutation=permutation)
    selected = set(selected)
    available = {(s['dataset'], s['scene_id']) for s in scenes}
    if not selected or not selected <= available:
        raise ValueError('Supplemental validation scene is missing from the qualified manifest')
    if selected & used:
        raise ValueError('Supplemental validation scene has entered training: '+str(sorted(selected & used)))
    if selected & planned:
        raise ValueError('Supplemental validation scene is scheduled for training: '+str(sorted(selected & planned)))
    return dict(status='passed', checkpoint_step=saved['step'],
        protected_until_step=identity['max_steps'], scenes=len(selected),
        training_identity_sha256=identity_hash(identity),
        definition='Unseen in checkpoint ledgers and unsampled throughout this fixed stage-1 plan; '
                   'these scenes are still members of the original training manifest.')
