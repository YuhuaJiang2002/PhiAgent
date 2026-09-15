#!/usr/bin/env python3
"""Generate only explicitly selected custom cases with the fixed CP4 protocol."""
import plenoptic_paths as layout
import argparse
from datetime import datetime, timedelta
import json
import os
import socket
import time

import torch
import torch.distributed as dist

import prepare_plenoptic as prepare
from custom_validation import load_custom_suite, make_job
from infer_stage1 import input_sample, sample_video, save_video
from local_checkpoints import configure_local_checkpoints
from plenoptic_data import rooted
from plenoptic_distributed import make_groups
from plenoptic_model import build_model
from train_plenoptic import Preprocessor, context_at_step, load_trainable
from validation_suite import digest_file, digest_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] != 'h20-1':
        raise RuntimeError('Custom validation runs only on h20-1')
    rank, cp, local = (int(os.getenv(key, default)) for key, default in
                       [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    if not args.check_only and (cp != 4 or int(os.getenv('LOCAL_WORLD_SIZE', cp)) != cp):
        raise ValueError('Use exactly four GPUs on one node')
    torch.set_num_threads(4)
    checkpoint, output = rooted(args.checkpoint), rooted(args.output)
    layout.require_workspace_path(output)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if saved.get('trainer_version') != 3 or saved.get('step', 0) < 1:
        raise ValueError('Require a completed trainer-version-3 checkpoint')
    for name in ('base', 'vae', 'reason'):
        if saved[name+'_revision'] != prepare.JOBS[name][2]:
            raise ValueError('Frozen checkpoint revision differs')
    step = saved['step']
    base = json.loads((layout.rooted('configs/plenoptic/fixed_validation_v2.json')).read_text())
    plan = dict(base, k=context_at_step(saved['config'], step-1))
    suite, cases, scenes = load_custom_suite(base, rooted(args.suite))
    if not cases or any(not case['qualitative_only'] for case in cases):
        raise ValueError('Only qualitative user cases are allowed here')
    if args.check_only:
        proof = []
        for case, scene in zip(cases, scenes):
            sample = input_sample(make_job(case, scene, plan['k']), plan, None)
            if (sample['videos'][-1].count_nonzero().item()
                    or not torch.isfinite(sample['extrinsics']).all()
                    or not torch.isfinite(sample['intrinsics']).all()):
                raise ValueError('Invalid cameras or nonzero target input')
            proof.append(dict(case_id=case['id'], video_shape=list(sample['videos'].shape),
                target_input_zero=True, camera_provenance=scene['camera_provenance']))
        report = dict(status='preflight_passed', checkpoint_step=step, k=plan['k'],
            context_parallel_size=4, cases=proof)
        prepare.save_json(output/'preflight.json', report)
        print(json.dumps(report), flush=True)
        return
    if any((output/case['id']/'generated.mp4').exists() for case in cases):
        raise FileExistsError('Choose fresh output; prior videos are preserved')
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(local)
    torch.manual_seed(20260910)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    configure_local_checkpoints(require=['base', 'vae', 'reason'])
    model = build_model(False).cuda().eval()
    load_trainable(model, saved)
    model.requires_grad_(False)
    del saved
    preprocessor = Preprocessor(model) if rank == 0 else None
    if preprocessor is not None:
        preprocessor.encoder.model.eval()
    dist.init_process_group('gloo', timeout=timedelta(minutes=15))
    group, _ = make_groups(cp, cp, rank)
    model.enable_context_parallel(group)
    checksum = [digest_file(checkpoint) if rank == 0 else None]
    dist.broadcast_object_list(checksum, src=0)
    started = time.monotonic()
    try:
        for case, scene in zip(cases, scenes):
            torch.manual_seed(case['seed'])
            job = make_job(case, scene, plan['k'])
            case_started = time.monotonic()
            if rank == 0:
                print(json.dumps(dict(event='case_started', case_id=case['id'], checkpoint_step=step)), flush=True)
            frames = sample_video(model, preprocessor, job, plan, None, rank, cp, group)
            if rank == 0:
                directory = output/case['id']
                directory.mkdir(parents=True, exist_ok=True)
                save_video(directory/'generated.mp4', frames)
                report = dict(status='generated', checkpoint=str(layout.relative(checkpoint)),
                    checkpoint_step=step, checkpoint_sha256=checksum[0], context_parallel_size=cp,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), dataset='custom', split='val',
                    case_id=case['id'], scene_id=case['scene_id'], custom_scene=scene,
                    source_cameras=job['source_cameras'], target_camera=job['target_camera'],
                    prompt=job['prompt'], seed=job['seed'], input_sha256=case['input_sha256'],
                    qualitative_only=True, has_target_reference=False,
                    target_reference_used_during_generation=False, context_policy='repeat_single_source',
                    suite_sha256=digest_json(suite), video=str(layout.relative(directory/'generated.mp4')),
                    seconds=round(time.monotonic()-case_started, 3),
                    **{key: plan[key] for key in ('k', 'frames', 'height', 'width', 'denoising_steps', 'guidance', 'shift')},
                    created_at=datetime.now().astimezone().isoformat())
                prepare.save_json(directory/'inference.json', report)
                print(json.dumps(dict(event='case_complete', case_id=case['id'], seconds=report['seconds'])), flush=True)
            dist.barrier()
        peak = dict(rank=rank, allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        peaks = [None]*cp
        dist.all_gather_object(peaks, peak)
        if rank == 0:
            prepare.save_json(output/'generation.json', dict(status='complete', checkpoint_step=step,
                checkpoint_sha256=checksum[0], context_parallel_size=cp, cases=[c['id'] for c in cases],
                gpu_memory_peaks=peaks, seconds=round(time.monotonic()-started, 3)))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
