"""Fixed SynCam validation cases and their input fingerprints."""
import plenoptic_paths as layout
import hashlib
import json
from pathlib import Path

import prepare_plenoptic as prepare

SUITE = layout.rooted('configs/plenoptic/fixed_validation.json')


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def load_suite():
    """Read only SynCam; preserve existing cases, captions, seeds and hashes."""
    path = layout.rooted('datasets/manifests/syncam_scenes.jsonl')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    training_files = {name for row in rows if row['split'] == 'train'
                      for name in row['videos'].values()}
    if SUITE.exists():
        suite = json.loads(SUITE.read_text())
    else:
        cases = []
        # The release has one SynCam val scene. Its 24 mm focal length is known.
        available = [r for r in rows if r['dataset'] == 'syncam'
                     and r['split'] == 'val' and r['structurally_complete']]
        if len(available) != 1:
            raise ValueError('Expected the released single SynCam validation scene')
        scene = available[0]
        for target in ('cam07','cam08','cam09','cam10'):
            cases.append(dict(id=f'syncam-{target}', dataset='syncam',
                scene_id=scene['scene_id'], source_cameras=['cam01','cam02','cam03','cam04'],
                target_camera=target, seed=20260910+len(cases)*100, prompt=None))
        suite = dict(schema=1, name='basic-fixed-val-v1', split='val', cases=cases,
            frames=81, height=432, width=768, noise_levels=[.1,.35,.65,.9],
            denoising_steps=35, shift=5., guidance=1.5,
            text_dropout=0., video_dropout=0., overlap_probability=0.,
            scope='Four fixed target views of the single released SynCam validation scene. '
                  'Validation paths are excluded from training. This is a small monitoring suite, '
                  'not a broad generalization test or the full paper benchmark.')
    if suite.get('schema') != 1 or suite.get('split') != 'val' or len(suite['cases']) != 4:
        raise ValueError('Unexpected validation suite schema/split/size')
    if (suite['frames'],suite['height'],suite['width']) != (81,432,768):
        raise ValueError('Fixed suite must retain full training clip geometry')
    if any(not 0 < t < 1 for t in suite['noise_levels']):
        raise ValueError('Noise levels must be strictly between 0 and 1')
    scenes = []
    for case in suite['cases']:
        if case['dataset'] != 'syncam':
            raise ValueError('This validation suite supports SynCam cases only')
        matches = [r for r in rows if r['dataset'] == 'syncam' and r['scene_id'] == case['scene_id']
                   and r['split'] == 'val' and r['structurally_complete']]
        if len(matches) != 1:
            raise ValueError(f'Cannot locate a unique held-out scene for {case["id"]}')
        scene = matches[0]
        views = case['source_cameras']+[case['target_camera']]
        if len(views) != 5 or len(set(views)) != 5:
            raise ValueError('Require four distinct sources and a separate target')
        paths = [scene['videos'][camera] for camera in views]+[scene['extrinsics']]
        if training_files.intersection(paths):
            raise ValueError('Validation input path is also present in the training manifest')
        fingerprints = {name: digest_file(layout.rooted(name)) for name in paths}
        if 'input_sha256' in case and case['input_sha256'] != fingerprints:
            raise ValueError(f'Fixed validation inputs changed for {case["id"]}')
        case['input_sha256'] = fingerprints
        scenes.append(scene)
    if len({(c['dataset'],c['scene_id'],c['target_camera']) for c in suite['cases']}) != 4:
        raise ValueError('Validation scene/target pairs must be distinct')
    prepare.save_json(SUITE, suite)
    return suite, scenes
