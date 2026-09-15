"""Fixed calibrated validation inputs and a checked, finite unseen-scene cohort."""
import plenoptic_paths as layout
import json
import re

import prepare_plenoptic as prepare
from validation_holdout import check_protected_scenes
from validation_suite import digest_file, digest_json

SUITE = layout.rooted('configs/plenoptic/fixed_validation_v2.json')


def load_suite(checkpoint):
    suite = json.loads(SUITE.read_text())
    if suite.get('schema') != 2 or suite.get('context_parallel_size') != 4:
        raise ValueError('Validation v2 requires its pinned four-GPU protocol')
    if (suite['frames'], suite['height'], suite['width']) != (81, 432, 768):
        raise ValueError('Fixed validation clip geometry changed')
    if any(not 0 < sigma < 1 for sigma in suite['noise_levels']):
        raise ValueError('Invalid fixed noise levels')
    rows = []
    for dataset in ('syncam', 'multicam'):
        manifest = layout.rooted(f'datasets/manifests/{dataset}_scenes.jsonl')
        rows.extend(json.loads(line) for line in manifest.read_text().splitlines() if line.strip())
    lookup = {(s['dataset'], s['scene_id']): s for s in rows if s['structurally_complete']}
    # Reconstruct the exact manifest order used by the checkpoint.
    train = []
    for name in checkpoint['config']['data']['manifests']:
        manifest = layout.rooted(name)
        train.extend(s for s in map(json.loads, manifest.read_text().splitlines())
                     if s['split'] == 'train' and s['structurally_complete'])
    scenes, protected, identifiers, targets, scene_protocols = [], set(), set(), set(), {}
    fingerprints = {}
    for case in suite['cases']:
        name = case['id']
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', name) or name in identifiers:
            raise ValueError('Validation case IDs must be unique safe names')
        identifiers.add(name)
        key = case['dataset'], case['scene_id']
        target_key = (*key, case['target_camera'])
        if target_key in targets:
            raise ValueError('Duplicate scene/target would bias validation averages')
        targets.add(target_key)
        scene = lookup[key]
        if case['partition'] == 'official_val':
            if scene['split'] != 'val':
                raise ValueError('Official validation scene is present in the training split')
        elif case['partition'] == 'stage1_unsampled':
            if scene['split'] != 'train':
                raise ValueError('Supplemental scene provenance differs from its qualified source')
            protected.add(key)
        else:
            raise ValueError('Unknown validation partition')
        views = case['source_cameras'] + [case['target_camera']]
        if len(views) != 5 or len(set(views)) != 5:
            raise ValueError('Require four fixed source slots and a distinct target')
        if case['seed'] != suite['generation_seed']:
            raise ValueError('All target cameras must use the pinned generation seed')
        fixed = case['source_cameras'], case['seed'], case.get('prompt')
        if key in scene_protocols and scene_protocols[key] != fixed:
            raise ValueError('Source cameras, seed and caption must stay identical within each scene')
        scene_protocols[key] = fixed
        required = {scene['videos'][camera] for camera in views} | {scene['extrinsics']}
        if set(case['input_sha256']) != required:
            raise ValueError('Every validation video and camera file must be fingerprinted')
        for relative, expected in case['input_sha256'].items():
            path = (layout.rooted(relative)).resolve()
            layout.require_workspace_path(path)
            if relative not in fingerprints:
                fingerprints[relative] = digest_file(path)
            if fingerprints[relative] != expected:
                raise ValueError('Fixed validation input changed: '+relative)
        scenes.append(scene)
    if len(scene_protocols) < 3 or not protected:
        raise ValueError('The expanded protocol requires at least three distinct scenes')
    proof = check_protected_scenes(checkpoint, train, suite['protection'], protected)
    return suite, scenes, proof
