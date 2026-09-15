#!/usr/bin/env python3
"""Fixed held-out flow loss and videos, evaluated without optimizer updates."""
import plenoptic_paths as layout
import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import socket
import time
import sys
import numpy as np

import torch
import torch.distributed as dist

import prepare_plenoptic as prepare
from infer_stage1 import input_sample, sample_video, sample_video_batch, save_video
from local_checkpoints import configure_local_checkpoints
from plenoptic_data import decode_video, rooted
from plenoptic_distributed import make_groups
from plenoptic_model import build_model
from train_plenoptic import Preprocessor, broadcast_tensor, context_at_step, load_trainable
from validation_suite_v2 import SUITE, digest_file, digest_json, load_suite
from custom_validation import CUSTOM_SUITE, aggregate_supervised, load_custom_suite, make_job
from video_pipeline import sampling_plan, compatible_batches
from custom_validation_comparison import probe_video, side_by_side, concatenate_segments
from refinement_inputs import verify_request
from validation_artifacts import read
from validation_live import atomic_json, sha256
import validation_live as live


@torch.no_grad()
def fixed_loss(model, preprocessor, job, plan, levels, rank, cp, group):
    k = plan['k']
    length = (plan['frames']-1)//4+1
    total_t, h, w = length*(k+1), plan['height']//8, plan['width']//8
    x0 = camera = text = mask = None
    if rank == 0:
        sample = input_sample(job, plan, None)
        # The target truth is used only for teacher-forced loss, never generation.
        sample['videos'][-1] = decode_video(job['scene']['videos'][job['target_camera']],
                                           list(range(plan['frames'])), (plan['height'],plan['width']))
        x0, camera, text, mask = preprocessor(sample, dict(data=dict(k=k), text_dropout=0.,
                                                overlap_probability=0., text_cache_entries=16))
    x0 = broadcast_tensor(x0, (1,16,total_t,h,w), torch.float32, 0, group, cp)
    camera = broadcast_tensor(camera, (1,total_t,h//2,w//2,1536), torch.bfloat16, 0, group, cp)
    text = broadcast_tensor(text, (1,512,1024), torch.bfloat16, 0, group, cp)
    mask = broadcast_tensor(mask, (1,1,total_t,h,w), torch.float32, 0, group, cp)
    generator = torch.Generator(device='cuda').manual_seed(job['seed']+1)
    noise = torch.randn(x0.shape, device='cuda', generator=generator)
    target = (noise-x0).chunk(cp, dim=-1)[rank]
    local_mask = mask.chunk(cp, dim=-1)[rank]
    results = []
    for sigma in levels:
        xt = ((1-sigma)*x0+sigma*noise)*(1-mask)+x0*mask
        frame_mask = mask.mean(dim=(1,3,4))
        times = sigma*1000*(1-frame_mask)+.1*frame_mask
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(x_B_C_T_H_W=xt, timesteps_B_T=times, camera=camera,
                crossattn_emb=text, crossattn_projected=True,
                condition_video_input_mask_B_C_T_H_W=mask,
                padding_mask=torch.zeros(1,h,w,device='cuda'), fps=torch.tensor([15.],device='cuda')).float()
        error = ((prediction-target)*(1-local_mask)).square()
        statistics = torch.stack([error.sum(dtype=torch.float64),
            (1-local_mask).expand_as(error).sum(dtype=torch.float64),
            torch.tensor(error.numel(),dtype=torch.float64,device='cuda')])
        if cp > 1:
            dist.all_reduce(statistics, group=group)
        if not torch.isfinite(statistics).all() or statistics[1] <= 0:
            raise RuntimeError('Non-finite validation error or no supervised target elements')
        results.append(dict(noise_level=sigma,
            target_mse=(statistics[0]/statistics[1]).item(),
            training_normalized_mse=(statistics[0]/statistics[2]).item()))
    return results


def evaluation_batches(custom_cases, custom_scenes, fixed_cases, fixed_scenes, batch_size):
    """Interleave custom video groups; retain one fixed case per batch."""
    if type(batch_size) is not int or batch_size not in (1, 2):
        raise ValueError('Video batch size must be 1 or 2')
    if len(custom_cases) != len(custom_scenes) or len(fixed_cases) != len(fixed_scenes):
        raise ValueError('Validation cases and scenes must match')
    if (any(not case.get('qualitative_only') for case in custom_cases)
            or any(case.get('qualitative_only') for case in fixed_cases)):
        raise ValueError('Only custom qualitative cases may enter video batches')
    identifiers = [case['id'] for case in custom_cases+fixed_cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('Fixed and custom case IDs must be distinct')
    pairs = list(zip(custom_cases, custom_scenes))
    if batch_size > 1:
        groups = {}
        for case, scene in pairs:
            groups.setdefault(case.get('video_group') or case['id'], []).append((case, scene))
        for items in groups.values():
            items.sort(key=lambda pair: pair[0].get('segment', {}).get('index', 0))
        pairs = [items[index]
                 for index in range(max((len(items) for items in groups.values()), default=0))
                 for items in groups.values() if index < len(items)]
    batches = [pairs[start:start+batch_size] for start in range(0, len(pairs), batch_size)]
    batches.extend([pair] for pair in zip(fixed_cases, fixed_scenes))
    return compatible_batches(batches)


def rank_zero(action, rank, cp):
    """Publish rank-zero failures before peers enter the next collective."""
    value, error = None, None
    if rank == 0:
        try:
            value = action()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    payload = [error]
    if cp > 1:
        dist.broadcast_object_list(payload, src=0)
    if payload[0]:
        raise RuntimeError(payload[0])
    return value


def initial_rgb(job):
    return decode_video(job['initial_video'], job['initial_frame_indices'],
                        (432,768), calibrated_crop=True).permute(1,2,3,0).numpy()


def validate_inputs(request, jobs):
    verify_request(request)
    if not jobs or len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('Refinement jobs must be nonempty and unique')
    sources = {video['id']:video for video in request['videos']}
    totals = dict.fromkeys(sources, 0)
    for job in jobs:
        source = sources[job['group']]
        if (job['initial_video'] != source['generated']
                or job['initial_video_sha256'] != source['generated_sha256']
                or job['source_start_frame'] != totals[job['group']]):
            raise ValueError('Refinement must read the selected stage-one video in temporal order')
        valid, overlap = job['valid_frames'], job['history_overlap_frames']
        if not 1 <= valid <= 80 or overlap not in (0,1) or overlap and not totals[job['group']]:
            raise ValueError('Invalid refinement frame coverage')
        expected = np.minimum(np.arange(job['source_start_frame']-overlap,
                                       job['source_start_frame']-overlap+81),
                              job['source_start_frame']+valid-1).tolist()
        if job['initial_frame_indices'] != expected or job['plan']['frames'] != 81:
            raise ValueError('Refinement initial RGB indices are misaligned')
        originals = {entry['inference_sha256']:entry['inference'] for entry in source['segments']}
        original = originals.get(job['stage1_inference_sha256'])
        expected_plan = dict(k=(request['parameters'].get('k') or original['k']) if original else None, frames=81, height=432, width=768,
            **{key:request['parameters'][key] for key in ('denoising_steps','guidance','shift')})
        if original is None or job['plan'] != expected_plan or job['prompt'] != original['prompt']:
            raise ValueError('Refinement plan differs from its pinned source and request')
        totals[job['group']] += valid
        for name, expected_sha in job['input_sha256'].items():
            if sha256(name) != expected_sha:
                raise ValueError('Refinement prepared input changed: '+name)
        sample = input_sample(job, job['plan'], None)
        if sample['videos'][-1].count_nonzero() or not torch.isfinite(sample['extrinsics']).all():
            raise ValueError('Refinement ego input has target leakage or invalid cameras')
        prior = initial_rgb(job)
        if prior.shape != (81,432,768,3) or prior.dtype != np.uint8:
            raise ValueError('Stage-one RGB decoding failed')
    for name,total in totals.items():
        if total != sources[name]['group']['total_frames']:
            raise ValueError('Refinement does not cover its complete stage-one input')


def assemble_ready(run, request, jobs, records):
    published = live.refinement_directory(run.name)
    current = read(published/'manifest.json')
    ready = {row['case_id'] for row in records}
    done = {row['case_id'] for row in current['groups']}
    for video in request['videos']:
        selected = [job for job in jobs if job['group'] == video['id']]
        if video['id'] in done or any(job['id'] not in ready for job in selected):
            continue
        folder = run/'full'/video['id']
        folder.mkdir(parents=True)
        count = video['group']['total_frames']
        concatenate_segments(folder/'generated.mp4', [dict(video=published/'clips'/job['id']/'generated.mp4',
                               valid_frames=job['valid_frames']) for job in selected])
        probe_video(folder/'generated.mp4', frames=count, width=768, height=432)
        side_by_side(layout.rooted(video['generated']), folder/'generated.mp4', folder/'comparison.mp4', count)
        probe_video(folder/'comparison.mp4', frames=count, width=1536, height=432)
        info = dict(stage='stage2', status='generated', run_id=run.name, source_run_id=request['source_run_id'],
            checkpoint_step=request['checkpoint_step'], checkpoint_sha256=request['checkpoint_sha256'],
            source_video=video['generated'], source_video_sha256=video['generated_sha256'],
            assembled_full_video=True, frames=count, fps=15, width=768, height=432,
            segments=[job['id'] for job in selected], uses_actual_stage1_rgb=True,
            real_target_ground_truth_used=False, parameters=request['parameters'],
            quality_status='pending_visual_review', source_video_replaced=False)
        live.publish_refinement(run, video['id'], folder, info, full_video=True)


def refine_main():
    if socket.gethostname().split('.')[0] != 'h20-1':
        raise RuntimeError('Refinement evaluation is permitted only on h20-1')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--jobs', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    run = layout.rooted(args.output)
    layout.require_workspace_path(run)
    request, jobs = read(args.request), read(args.jobs)
    rank, cp = int(os.getenv('RANK','0')), int(os.getenv('WORLD_SIZE','1'))
    local = int(os.getenv('LOCAL_RANK','0'))
    if not args.check_only and (cp != 4 or int(os.getenv('LOCAL_WORLD_SIZE',str(cp))) != cp):
        raise ValueError('Refinement requires one four-GPU context-parallel group')
    torch.set_num_threads(4)
    saved = torch.load(layout.rooted(args.checkpoint), map_location='cpu', weights_only=True, mmap=True)
    if saved.get('trainer_version') != 3 or saved.get('step') != request['checkpoint_step']:
        raise ValueError('Refinement checkpoint differs from stage one')
    for name in ('base','vae','reason'):
        if saved[name+'_revision'] != prepare.JOBS[name][2]:
            raise ValueError('Frozen model revision differs from stage one')
    if rank == 0:
        if sha256(args.checkpoint) != request['checkpoint_sha256']:
            raise ValueError('Refinement must use exactly the stage-one checkpoint bytes')
        validate_inputs(request, jobs)
    if args.check_only:
        proof = dict(status='preflight_passed', stage='stage2', checkpoint_step=request['checkpoint_step'],
            checkpoint_sha256=request['checkpoint_sha256'], cases=len(jobs), groups=len(request['videos']),
            uses_actual_stage1_rgb=True, real_target_ground_truth_used=False, context_parallel_size=4)
        atomic_json(run/'preflight.json', proof)
        print(json.dumps(proof), flush=True)
        return
    torch.cuda.set_device(local)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(20260913)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    configure_local_checkpoints(require=['base','vae','reason'])
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
    from infer_stage1 import refine_video
    records, previous = [], {}
    try:
        for job in jobs:
            started = time.monotonic()
            prior = rank_zero(lambda: initial_rgb(job), rank, cp)
            prefix = previous.get(job['group']) if rank == 0 and job['history_overlap_frames'] else None
            if job['history_overlap_frames']:
                rank_zero(lambda: None if prefix is not None else (_ for _ in ()).throw(
                    ValueError('Missing same-time refinement history')), rank, cp)
            frames = refine_video(model, preprocessor, job, job['plan'], rank, cp, group,
                initial_frames=prior, strength=request['parameters']['strength'], prefix=prefix,
                progress=lambda step,total: print(json.dumps(dict(event='refining',case_id=job['id'],step=step,total=total)),flush=True))
            def publish_clip():
                overlap = job['history_overlap_frames']
                valid = frames[overlap:overlap+job['valid_frames']]
                before = prior[overlap:overlap+job['valid_frames']]
                folder = run/'clips'/job['id']
                folder.mkdir(parents=True)
                save_video(folder/'generated.mp4', valid)
                save_video(folder/'stage1.mp4', before)
                side_by_side(folder/'stage1.mp4', folder/'generated.mp4', folder/'comparison.mp4', len(valid))
                probe_video(folder/'generated.mp4', frames=len(valid), width=768, height=432)
                probe_video(folder/'comparison.mp4', frames=len(valid), width=1536, height=432)
                info = dict(stage='stage2', status='generated', run_id=run.name,
                    source_run_id=request['source_run_id'], checkpoint_step=request['checkpoint_step'],
                    checkpoint_sha256=request['checkpoint_sha256'], source_video=job['initial_video'],
                    source_video_sha256=job['initial_video_sha256'], initial_frame_indices=job['initial_frame_indices'],
                    source_start_frame=job['source_start_frame'], history_overlap_frames=overlap,
                    frames=len(valid), fps=15, width=768, height=432, uses_actual_stage1_rgb=True,
                    original_ego_conditioning=True, real_target_ground_truth_used=False,
                    plan=job['plan'], parameters=request['parameters'], prompt=job['prompt'], seed=job['seed'],
                    seconds=round(time.monotonic()-started,3), input_sha256=job['input_sha256'],
                    stage1_inference_sha256=job['stage1_inference_sha256'], quality_status='pending_visual_review')
                entry = live.publish_refinement(run, job['id'], folder, info)
                previous[job['group']] = valid[-1].copy()
                records.append(entry)
                atomic_json(run/'progress.json',dict(cases=records,total_cases=len(jobs)))
                assemble_ready(run, request, jobs, records)
            rank_zero(publish_clip, rank, cp)
        rank_zero(lambda: atomic_json(run/'refinement.json',dict(status='complete',cases=records,
                   source_run_id=request['source_run_id'], uses_actual_stage1_rgb=True)), rank, cp)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()



def main():
    if sys.argv[1:2] == ['--refine']:
        del sys.argv[1]
        return refine_main()
    if socket.gethostname().split('.')[0] != 'h20-1':
        raise RuntimeError('This validation evaluator runs only on h20-1')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='outputs/basic_stage1_24gpu/latest.pt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--loss-only', action='store_true')
    parser.add_argument('--skip-standard-videos', action='store_true',
                        help='Keep fixed validation losses; generate only custom videos')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--publish-progress', action='store_true',
                        help='Archive each completed case and full source video immediately')
    parser.add_argument('--custom-suite', default=str(CUSTOM_SUITE))
    parser.add_argument('--video-batch-size', type=int, choices=(1, 2), default=1,
                        help='Custom clips generated together; fixed validation stays batch 1')
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = rooted(args.output)
    layout.require_workspace_path(output)
    rank, cp = int(os.getenv('RANK','0')), int(os.getenv('WORLD_SIZE','1'))
    local = int(os.getenv('LOCAL_RANK','0'))
    if not args.check_only and cp != 4:
        raise ValueError('Fixed validation v2 always uses context parallel size 4')
    if int(os.getenv('LOCAL_WORLD_SIZE',str(cp))) != cp or 48 % cp:
        raise ValueError('Evaluation CP must stay on one node and divide 48 spatial patches')
    checkpoint = rooted(args.checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if saved.get('trainer_version') != 3 or saved.get('step',0) < 1:
        raise ValueError('Require a completed trainer-version-3 checkpoint')
    for name in ('base','vae','reason'):
        if saved[name+'_revision'] != prepare.JOBS[name][2]:
            raise ValueError(f'Frozen {name} revision differs from the checkpoint')
    step = saved['step']
    source_sha = saved.get('inference_snapshot', {}).get('source_sha256')
    k = context_at_step(saved['config'], step-1)
    if rank == 0:
        suite, scenes, holdout_proof = load_suite(saved)
        custom_suite, custom_cases, custom_scenes = load_custom_suite(
            suite, rooted(args.custom_suite))
        if args.loss_only:
            custom_cases, custom_scenes = [], []
        if args.check_only:
            batches = evaluation_batches(custom_cases, custom_scenes, suite['cases'], scenes,
                                         args.video_batch_size)
            for case, scene in (pair for batch in batches for pair in batch):
                # Check real decoding and camera geometry without loading any GPU model.
                case_plan = sampling_plan(dict(suite,k=k), [case])
                job = make_job(case, scene, case_plan['k'])
                sample = input_sample(job, case_plan, None)
                if (sample['videos'][-1].count_nonzero()
                        or not torch.isfinite(sample['extrinsics']).all()
                        or not torch.isfinite(sample['intrinsics']).all()):
                    raise ValueError('Invalid inference input or target leakage')
                if not case.get('qualitative_only'):
                    decode_video(scene['videos'][case['target_camera']],list(range(suite['frames'])),
                                 (suite['height'],suite['width']))
            proof = dict(status='preflight_passed', checkpoint_step=step, k=k,
                cases=len(suite['cases'])+len(custom_cases), supervised_cases=len(suite['cases']),
                qualitative_cases=len(custom_cases), split='fixed_validation_v2', execution_host='h20-1',
                suite=str(layout.relative(SUITE)),
                context_parallel_size=4, holdout_proof=holdout_proof,
                video_batch_size=args.video_batch_size, fixed_case_batch_size=1,
                skip_standard_videos=args.skip_standard_videos,
                planned_generation_batches=0 if args.loss_only else sum(
                    bool(batch[0][0].get('qualitative_only')) or not args.skip_standard_videos
                    for batch in batches),
                checkpoint_source_sha256=source_sha,
                generation_seed=suite['generation_seed'])
            prepare.save_json(output/'preflight.json',proof)
            print(json.dumps(proof),flush=True)
            return
    if args.check_only:
        raise ValueError('Use single-process CPU preflight')
    if (output/'validation.json').exists():
        raise FileExistsError('Validation output already exists')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(local)
    torch.manual_seed(20260910)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    configure_local_checkpoints(require=['base','vae','reason'])
    model = build_model(False).cuda().eval()
    load_trainable(model, saved)
    model.requires_grad_(False)
    del saved
    preprocessor = Preprocessor(model) if rank == 0 else None
    if preprocessor is not None:
        preprocessor.encoder.model.eval()
    if rank == 0:
        if any(not case['prompt'] for case in suite['cases']):
            from online_caption import OnlineCaptioner
            captioner = OnlineCaptioner('datasets/captions/fixed_validation')
            for case, scene in zip(suite['cases'],scenes):
                if not case['prompt']:
                    prompt = captioner(dict(caption_video=scene['videos'][case['source_cameras'][0]],
                        scene_id=scene['scene_id'],dataset=case['dataset'],split=scene['split']))
                    for matching in suite['cases']:
                        if (matching['dataset'],matching['scene_id']) == (case['dataset'],case['scene_id']):
                            matching['prompt'] = prompt
                    # Persist every target of this scene together, including on
                    # an interrupted first run, so the fixed-input guard holds.
                    prepare.save_json(SUITE,suite)
            del captioner
            torch.cuda.empty_cache()
        output.mkdir(parents=True,exist_ok=True)
        prepare.save_json(output/'suite.json',suite)
        if custom_suite is not None:
            prepare.save_json(output/'custom_suite.json',custom_suite)
        checkpoint_sha = digest_file(checkpoint)
    group = None
    if cp > 1:
        dist.init_process_group('gloo',timeout=timedelta(minutes=15))
        group, _ = make_groups(cp,cp,rank)
        model.enable_context_parallel(group)
        payload = [(suite,scenes,checkpoint_sha,custom_suite,custom_cases,custom_scenes) if rank == 0 else None]
        dist.broadcast_object_list(payload,src=0)
        suite, scenes, checkpoint_sha, custom_suite, custom_cases, custom_scenes = payload[0]
    plan = dict(suite,k=k)
    started = time.monotonic()
    records = []
    generation_batches = []
    try:
        registration_error = [None]
        if rank == 0 and args.publish_progress:
            try:
                from validation_live import register_inputs
                register_inputs(output)
            except Exception as exc:
                registration_error[0] = f'{type(exc).__name__}: {exc}'
        if cp > 1:
            dist.broadcast_object_list(registration_error, src=0)
        if registration_error[0]:
            raise RuntimeError('Validation archive initialization failed: '+registration_error[0])
        batches = evaluation_batches(custom_cases, custom_scenes, suite['cases'], scenes,
                                     args.video_batch_size)
        total_cases = len(custom_cases)+len(suite['cases'])
        for batch_index, batch in enumerate(batches, start=1):
            batch_started = time.monotonic()
            qualitative = bool(batch[0][0].get('qualitative_only'))
            generate = not args.loss_only and (qualitative or not args.skip_standard_videos)
            generation_plan = sampling_plan(plan, [case for case, _ in batch])
            jobs = [make_job(case, scene, generation_plan['k']) for case, scene in batch]
            batch_records = []
            for (case, scene), job in zip(batch, jobs):
                torch.manual_seed(case['seed'])
                losses = [] if qualitative else fixed_loss(
                    model,preprocessor,job,plan,suite['noise_levels'],rank,cp,group)
                record = dict(case_id=case['id'],scene_id=case['scene_id'],dataset=case['dataset'],
                    partition=case.get('partition','qualitative'),
                    qualitative_only=qualitative, has_target_reference=not qualitative,
                    generation_skipped=not generate,
                    losses=losses,target_mse=sum(row['target_mse'] for row in losses)/len(losses) if losses else None,
                    training_normalized_mse=sum(row['training_normalized_mse'] for row in losses)/len(losses) if losses else None)
                batch_records.append(record)
                if rank == 0:
                    print(json.dumps(dict(event='qualitative_case' if qualitative else 'case_loss',**record)),flush=True)
            videos = None
            if generate:
                if rank == 0:
                    print(json.dumps(dict(event='video_batch_start',batch_index=batch_index,
                        batch_size=len(batch),case_ids=[case['id'] for case, _ in batch],
                        qualitative_only=qualitative)),flush=True)
                # Fixed cases retain the original sampler and the RNG state after fixed_loss.
                if len(batch) == 1:
                    frames = sample_video(model,preprocessor,jobs[0],generation_plan,None,rank,cp,group)
                    videos = [frames] if rank == 0 else None
                else:
                    videos = sample_video_batch(model,preprocessor,jobs,generation_plan,None,rank,cp,group)
                if rank == 0 and (videos is None or len(videos) != len(batch)):
                    raise RuntimeError('Generated video count differs from the batch')
            shared_seconds = (time.monotonic()-batch_started)/len(batch)
            for index, ((case, scene), job, record) in enumerate(zip(batch, jobs, batch_records)):
                save_started = time.monotonic()
                if rank == 0 and generate:
                    case_output = output/case['id']
                    case_output.mkdir(parents=True,exist_ok=True)
                    save_video(case_output/'generated.mp4',videos[index])
                    prepare.save_json(case_output/'inference.json',dict(status='generated',
                        checkpoint=str(layout.relative(checkpoint)),checkpoint_step=step,
                        checkpoint_sha256=checkpoint_sha,checkpoint_source_sha256=source_sha,
                        dataset=case['dataset'],split=scene['split'],
                        evaluation_partition=case.get('partition','qualitative'),
                        case_id=case['id'],context_parallel_size=cp,
                        suite_name=suite['name'],suite_sha256=digest_json(suite),
                        scene_id=case['scene_id'],source_cameras=job['source_cameras'],
                        target_camera=job['target_camera'],seed=job['seed'],prompt=job['prompt'],
                        **{key:generation_plan[key] for key in ('k','frames','height','width','denoising_steps','shift','guidance')},
                        video=str(layout.relative(case_output/'generated.mp4')),
                        qualitative_only=qualitative, has_target_reference=not qualitative,
                        custom_scene=scene if qualitative else None,
                        context_policy=('single_real_source' if generation_plan['k'] == 1 else 'repeat_single_source') if qualitative else 'distinct_real_views',
                        checkpoint_trained_contexts=k,
                        input_sha256=case.get('input_sha256'),
                        segment=case.get('segment'), video_group=case.get('video_group'),
                        pipeline_profile_sha256=case.get('pipeline_profile_sha256'),
                        target_reference_used_during_generation=False,
                        created_at=datetime.now().astimezone().isoformat()))
                    record['video_directory'] = case['id']
                # Divide shared compute time so full-video assembly does not double-count it.
                record['seconds'] = round(shared_seconds+time.monotonic()-save_started,3)
                records.append(record)
                if rank == 0:
                    prepare.save_json(output/'progress.json',dict(checkpoint_step=step,k=k,
                        cases=records,total_cases=total_cases,video_batch_size=args.video_batch_size))
                    print(json.dumps(dict(event='case_complete',case_id=case['id'],seconds=record['seconds'])),flush=True)
                publication_error = [None]
                if rank == 0 and generate and args.publish_progress:
                    try:
                        from validation_live import publish_case, publish_ready_groups
                        publish_case(output, record)
                        publish_ready_groups(output, records)
                        prepare.save_json(output/'progress.json',dict(checkpoint_step=step,k=k,
                            cases=records,total_cases=total_cases,video_batch_size=args.video_batch_size))
                    except Exception as exc:
                        publication_error[0] = f'{type(exc).__name__}: {exc}'
                if cp > 1:
                    dist.broadcast_object_list(publication_error, src=0)
                if publication_error[0]:
                    raise RuntimeError('Completed video publication failed: ' + publication_error[0])
            if rank == 0 and generate:
                summary = dict(batch_index=batch_index,batch_size=len(batch),
                    case_ids=[case['id'] for case, _ in batch],qualitative_only=qualitative,
                    seconds=round(time.monotonic()-batch_started,3))
                generation_batches.append(summary)
                print(json.dumps(dict(event='video_batch_complete',**summary)),flush=True)
            del videos
            if cp > 1:
                dist.barrier()
        peak = dict(rank=rank, allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        peaks = [None]*cp
        dist.all_gather_object(peaks, peak)
        if rank == 0:
            measured = [r for r in records if not r['qualitative_only']]
            qualitative = [r for r in records if r['qualitative_only']]
            report = dict(status='evaluated', checkpoint_step=step,checkpoint_sha256=checkpoint_sha,k=k,
                checkpoint_source_sha256=source_sha,
                execution_host=socket.gethostname(),context_parallel_size=cp,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                suite_name=suite['name'],suite_sha256=digest_json(suite),cases=measured,
                qualitative_cases=qualitative,total_cases=len(records),
                qualitative_suite_sha256=digest_json(custom_suite) if custom_suite else None,
                holdout_proof=holdout_proof,
                generation_seed=suite['generation_seed'],
                gpu_memory_peaks=peaks,
                video_batch_size=args.video_batch_size, fixed_case_batch_size=1,
                generation_batches=generation_batches,
                case_timing_policy='Shared batch compute time divided equally, plus per-case video saving time.',
                scene_count=len({(c['dataset'],c['scene_id']) for c in suite['cases']}),
                **aggregate_supervised(records),
                loss_only=args.loss_only,skip_standard_videos=args.skip_standard_videos,
                seconds=round(time.monotonic()-started,3),
                created_at=datetime.now().astimezone().isoformat(),
                scope=suite['scope'],
                metric_definition='Equal average over the fixed supervised cases and noise levels; '
                    'target_mse divides only by target elements; training_normalized_mse also includes '
                    'zeroed source elements in its denominator. No condition dropout or overlap. '
                    'Compare checkpoints with the same suite hash, k and CP=4; these are monitoring metrics, not paper benchmark scores. '
                    'Custom user-video cases have no target truth and are excluded from all loss aggregates.')
            prepare.save_json(output/'validation.json',report)
            print(json.dumps(report),flush=True)
        if cp > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
