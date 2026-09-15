"""Portable SynCam/MultiCam dataset. Camera poses follow the dataset authors.

Conventions: right-handed relative c2w, metres; exported extrinsics are w2c.
Video and poses share frame indices. Views are [conditions..., target].
"""
import plenoptic_paths as layout
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, center_crop

from prepare_plenoptic import ROOT


def rooted(path):
    return layout.rooted(path)


def caption_key(video):
    return hashlib.sha256(str(video).encode()).hexdigest()


def read_scenes(manifests, split='train'):
    scenes = []
    for manifest in manifests:
        with rooted(manifest).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['split'] == split and row['structurally_complete']:
                    scenes.append(row)
    keys = [(r['dataset'], r['scene_id']) for r in scenes]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate scene in input manifests')
    if not scenes:
        raise ValueError(f'No complete {split} scenes in {manifests}')
    return scenes


def resize_geometry(source_hw, output_hw):
    h, w = source_hw
    oh, ow = output_hw
    scale = max(oh / h, ow / w)
    rh, rw = round(h * scale), round(w * scale)
    # Match torchvision center_crop (including its rounding convention).
    top, left = int(round((rh - oh) / 2)), int(round((rw - ow) / 2))
    return (rh, rw), (top, left)


def intrinsics_for_scene(scene, output_hw, source_hw=(1280, 1280)):
    match = re.search(r'/f(18|24|35|50)_', '/' + scene['scene_id'])
    if match is None and scene.get('dataset') != 'syncam':
        raise ValueError(f"Unknown focal length: {scene['scene_id']}")
    focal = float(match[1]) if match else 24.
    h, w = source_hw
    (rh, rw), (top, left) = resize_geometry(source_hw, output_hw)
    # Dataset card: square 23.76 mm sensor. Principal point at image centre.
    k = torch.tensor([[focal / 23.76 * w, 0., w / 2],
                      [0., focal / 23.76 * h, h / 2], [0., 0., 1.]], dtype=torch.float64)
    transform = torch.tensor([[rw / w, 0., -left], [0., rh / h, -top], [0., 0., 1.]],
                             dtype=torch.float64)
    return (transform @ k).float()


def parse_author_c2w(value):
    if isinstance(value, str):
        numbers = np.fromstring(value.replace('[', ' ').replace(']', ' '), sep=' ')
        if numbers.size != 16:
            raise ValueError('Camera matrix must have 16 numbers')
        pose = numbers.reshape(4, 4).T.copy()
    else:
        raise ValueError('Expected the released dataset matrix-string format')
    if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError('Invalid homogeneous camera matrix')
    # Author reference: KwaiVGI/ReCamMaster train_recammaster.py, pinned in docs.
    pose = pose[:, [1, 2, 0, 3]]
    pose[:3, 1] *= -1
    pose[:3, 3] /= 100.
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-4):
        raise ValueError('Non-orthogonal camera rotation')
    return pose


def camera_sequence(scene, views, frame_indices, output_hw):
    metadata = json.loads(rooted(scene['extrinsics']).read_text())
    poses = np.stack([[parse_author_c2w(metadata[f'frame{int(f)}'][view])
                       for f in frame_indices] for view in views])
    # A single scene-wide anchor preserves all inter-view and temporal motion.
    relative = np.linalg.inv(poses[0, 0]) @ poses
    if not np.allclose(np.linalg.det(relative[..., :3, :3]), 1., atol=4e-4):
        raise ValueError('Relative rotation is not right handed')
    c2w = torch.from_numpy(relative).float()
    w2c = torch.linalg.inv(c2w)
    k = intrinsics_for_scene(scene, output_hw).expand(len(views), len(frame_indices), 3, 3).clone()
    return c2w, w2c, k


def decode_video(video, frame_indices, output_hw, calibrated_crop=False):
    import decord
    reader = decord.VideoReader(str(rooted(video)), ctx=decord.cpu(0), num_threads=2)
    if max(frame_indices) >= len(reader):
        raise ValueError(f'Video too short: {video}, {len(reader)} frames')
    frames = reader.get_batch(list(frame_indices)).asnumpy()
    expected = tuple(output_hw) if calibrated_crop else (1280,1280)
    if tuple(frames.shape[1:3]) != expected:
        raise ValueError(f'Unexpected image dimensions for calibrated dataset: {frames.shape}')
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
    if calibrated_crop:
        return tensor.permute(1,0,2,3).contiguous()
    resized, _ = resize_geometry(frames.shape[1:3], output_hw)
    tensor = center_crop(resize(tensor, list(resized), InterpolationMode.BILINEAR, antialias=True),
                         list(output_hw))
    return tensor.permute(1, 0, 2, 3).contiguous()  # C,T,H,W uint8


class PlenopticDataset(Dataset):
    def __init__(self, manifests=('datasets/manifests/syncam_scenes.jsonl',),
                 captions='datasets/captions/reason1', split='train', k=1,
                 frames=81, height=432, width=768, seed=2026,
                 captioned_only=False, max_scenes=None, generated_manifest=None,
                 allow_diagnostic_generated=False,caption_mode='offline'):
        if k not in range(1, 5) or frames not in range(1, 82, 4):
            raise ValueError('Require k=1..4 and 4n+1 frames up to 81')
        if height % 16 or width % 16:
            raise ValueError('Image dimensions must be multiples of 16')
        self.scenes = read_scenes(manifests, split)
        self.generated = {}
        self.generated_hash = None
        if generated_manifest:
            raw = rooted(generated_manifest).read_bytes()
            synthetic = json.loads(raw)
            if synthetic['status']!='verified' or (synthetic['diagnostic'] and not allow_diagnostic_generated):
                raise ValueError('Stage 2 requires verified production synthetic videos')
            if [synthetic['height'],synthetic['width'],synthetic['frames']] != [height,width,frames]:
                raise ValueError('Synthetic geometry differs from training geometry')
            self.generated_hash = hashlib.sha256(raw).hexdigest()
            for row in synthetic['videos']:
                key = (row['dataset'],row['scene_id'])
                if row['camera'] in self.generated.setdefault(key,{}):
                    raise ValueError('Duplicate synthetic camera')
                if not rooted(row['generated_video']).is_file():
                    raise FileNotFoundError(row['generated_video'])
                self.generated[key][row['camera']] = row
            self.scenes = [s for s in self.scenes if (s['dataset'],s['scene_id']) in self.generated]
        if max_scenes is not None:
            self.scenes = self.scenes[:max_scenes]
        self.captions = rooted(captions)
        if caption_mode not in ('offline','online'):
            raise ValueError('Caption mode must be offline or online')
        if caption_mode=='online' and captioned_only:
            raise ValueError('Online captioning must sample the full dataset')
        self.caption_mode=caption_mode
        self.k, self.frames, self.output_hw, self.seed = k, frames, (height, width), seed
        self.epoch = 0
        self.items = []
        missing = []
        for scene_index, scene in enumerate(self.scenes):
            for camera, video in sorted(scene['videos'].items()):
                if self.generated:
                    available = self.generated[(scene['dataset'],scene['scene_id'])]
                    if not set(available)-{camera}:
                        continue
                    for view,row in available.items():
                        if scene['videos'].get(view)!=row['original_video']:
                            raise ValueError('Synthetic camera does not match original scene')
                if caption_mode=='online':
                    # Stable identity independent of when the cache fills.
                    self.items.append((scene_index,camera,video))
                    continue
                record = self.captions / (caption_key(video) + '.json')
                if record.is_file():
                    data = json.loads(record.read_text())
                    if data.get('video') != video or not data.get('caption', '').strip():
                        raise ValueError(f'Invalid caption record: {record}')
                    if data.get('split') != split:
                        raise ValueError(f'Caption split mismatch: {record}')
                    self.items.append((scene_index, camera, data['caption']))
                else:
                    missing.append(video)
        self.missing_captions = len(missing)
        if missing and not captioned_only:
            raise ValueError(f'{len(missing)} captions missing; run caption_plenoptic.py. First: {missing[0]}')
        if not self.items:
            raise ValueError('No captioned target videos; captions are mandatory')

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        scene_index, target, annotation = self.items[index]
        caption=annotation if self.caption_mode=='offline' else None
        scene = self.scenes[scene_index]
        # Independent of worker count; explicit epoch supports reproducible resume.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        generated = self.generated.get((scene['dataset'],scene['scene_id']))
        candidates = sorted(set(generated or scene['videos']) - {target})
        conditions = rng.choice(candidates, self.k, replace=len(candidates)<self.k).tolist()
        views = conditions + [target]
        start = 0 if generated else int(rng.integers(0, 82 - self.frames))
        indices = list(range(start, start + self.frames))
        paths = [generated[v]['generated_video'] if generated and i<self.k else scene['videos'][v]
                 for i,v in enumerate(views)]
        videos = torch.stack([decode_video(path,indices,self.output_hw,calibrated_crop=bool(generated and i<self.k))
                              for i,path in enumerate(paths)])
        c2w, w2c, k = camera_sequence(scene, views, indices, self.output_hw)
        return dict(videos=videos, caption=caption, c2w=c2w, extrinsics=w2c, intrinsics=k,
                    frame_indices=torch.tensor(indices), latent_frame_indices=torch.tensor(indices[::4]),
                    fps=torch.tensor(15.), image_size=torch.tensor(self.output_hw),
                    num_frames=torch.tensor(self.frames), num_conditions=torch.tensor(self.k),
                    dataset=scene['dataset'], scene_id=scene['scene_id'], camera_ids=views,
                    video_paths=paths, split=scene['split'],
                    caption_video=scene['videos'][target],
                    generated_conditions=bool(generated),target_is_ground_truth=True)
