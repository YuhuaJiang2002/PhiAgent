"""Optional user-video cases, with explicit camera provenance and no target truth."""
import plenoptic_paths as layout
import json
from pathlib import Path
import re

import prepare_plenoptic as prepare
from validation_suite import digest_file, digest_json

CUSTOM_SUITE = layout.rooted('configs/plenoptic/custom_validation.json')


def camera_arrays(scene, frames):
    import numpy as np
    path = (layout.rooted(scene['camera_file'])).resolve()
    layout.require_workspace_path(path)
    with np.load(path, allow_pickle=False) as saved:
        c2w = saved['c2w'].copy()
        intrinsics = saved['intrinsics'].copy()
    if c2w.shape != (2, frames, 4, 4) or intrinsics.shape != (2, frames, 3, 3):
        raise ValueError('Custom camera arrays require source/target and one matrix per frame')
    if not np.isfinite(c2w).all() or not np.isfinite(intrinsics).all():
        raise ValueError('Non-finite custom camera parameters')
    rotations = c2w[..., :3, :3]
    if (not np.allclose(c2w[..., 3, :], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=2e-4)
            or not np.allclose(np.linalg.det(rotations), 1., atol=2e-4)):
        raise ValueError('Invalid custom camera rotations or homogeneous rows')
    if (np.any(intrinsics[..., 0, 0] <= 0) or np.any(intrinsics[..., 1, 1] <= 0)
            or not np.allclose(intrinsics[..., 2, :], [0, 0, 1], atol=1e-5)):
        raise ValueError('Invalid custom camera intrinsics')
    return c2w, intrinsics


def load_custom_suite(base_suite, path=CUSTOM_SUITE):
    path = Path(path)
    if not path.is_file():
        return None, [], []
    suite = json.loads(path.read_text())
    if suite.get('schema') != 1:
        raise ValueError('Unknown custom validation schema')
    cases, scenes, ids = [], [], set()
    for case in suite['cases']:
        name = case['id']
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', name) or name in ids:
            raise ValueError('Custom validation case IDs must be unique safe names')
        ids.add(name)
        if not case.get('qualitative_only') or case.get('has_target_reference') is not False:
            raise ValueError('User-video cases must explicitly have no target reference or loss')
        scene = case['scene']
        if (scene.get('dataset') != 'custom' or scene.get('split') != 'val'
                or scene.get('camera_format') != 'opencv_npz'
                or set(scene['videos']) != {'source'}):
            raise ValueError('Unexpected custom scene or target pixels in the source-only scene')
        if any(scene[key] != base_suite[key] for key in ('frames', 'height', 'width')):
            raise ValueError('Custom clip geometry must match the fixed validation')
        if scene['fps'] != 15 or not case.get('prompt', '').strip():
            raise ValueError('Custom validation requires 15 fps and a pinned source description')
        for relative, expected in case['input_sha256'].items():
            path = (layout.rooted(relative)).resolve()
            layout.require_workspace_path(path)
            if digest_file(path) != expected:
                raise ValueError(f'Custom validation input changed: {relative}')
        required = {scene['videos']['source'], scene['camera_file'], scene['original_video']}
        if not required <= set(case['input_sha256']):
            raise ValueError('Custom source video and camera file must be fingerprinted')
        camera_arrays(scene, scene['frames'])
        cases.append(case)
        scenes.append(scene)
    return suite, cases, scenes


def make_job(case, scene, k):
    sources = ['source']*k if case.get('qualitative_only') else case['source_cameras'][:k]
    return dict(scene=scene, source_cameras=sources, target_camera=case['target_camera'],
                prompt=case.get('prompt') or '', seed=case['seed'])


def custom_input_sample(job, plan):
    import numpy as np
    import torch
    from plenoptic_data import decode_video
    scene = job['scene']
    if job['source_cameras'] != ['source']*plan['k'] or job['target_camera'] != 'target':
        raise ValueError('Custom inference uses one real source replicated for the context size')
    hw = (plan['height'], plan['width'])
    frames = list(range(plan['frames']))
    source = decode_video(scene['videos']['source'], frames, hw, calibrated_crop=True)
    videos = torch.stack([source]*plan['k']+[torch.zeros_like(source)])
    c2w, intrinsics = camera_arrays(scene, plan['frames'])
    selection = [0]*plan['k']+[1]
    poses = np.linalg.inv(c2w[0, 0]) @ c2w[selection]
    return dict(videos=videos, extrinsics=torch.from_numpy(np.linalg.inv(poses)).float(),
                intrinsics=torch.from_numpy(intrinsics[selection]).float(),
                caption=job['prompt'], image_size=torch.tensor(hw))


def aggregate_supervised(records):
    supervised = [r for r in records if not r.get('qualitative_only')]
    if not supervised:
        raise ValueError('Fixed validation must retain its supervised cases')
    return {key: sum(r[key] for r in supervised)/len(supervised)
            for key in ('target_mse', 'training_normalized_mse')}
