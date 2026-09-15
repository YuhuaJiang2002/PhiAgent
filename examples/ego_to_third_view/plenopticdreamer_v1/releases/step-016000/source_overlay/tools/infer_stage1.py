#!/usr/bin/env python3
"""Single-clip inference validation using a local stage-1 checkpoint and camera data."""
import plenoptic_paths as layout
import argparse
from datetime import datetime,timedelta
import hashlib
import json
import math
import numpy as np
import os
from pathlib import Path
import subprocess
import socket
import time
import torch
import torch.distributed as dist
import prepare_plenoptic as prepare
from local_checkpoints import configure_local_checkpoints
from hoi_checkpoint import load_adaptation, model_options
from plenoptic_data import camera_sequence,decode_video,read_scenes,rooted
from plenoptic_model import build_model
from plenoptic_distributed import make_groups
from train_plenoptic import Preprocessor,broadcast_tensor,conditioning_mask,context_at_step


def input_sample(job,plan,captions):
    scene=job['scene']
    if scene.get('camera_format') == 'opencv_npz':
        from custom_validation import custom_input_sample
        return custom_input_sample(job, plan)
    frames=list(range(plan['frames']))
    hw=(plan['height'],plan['width'])
    views=job['source_cameras']+[job['target_camera']]
    contexts=[decode_video(scene['videos'][camera],frames,hw) for camera in job['source_cameras']]
    # Only source videos are read. Target inputs are zeros; target poses are allowed.
    videos=torch.stack(contexts+[torch.zeros_like(contexts[0])])
    _,w2c,intrinsics=camera_sequence(scene,views,frames,hw)
    return dict(videos=videos,extrinsics=w2c,intrinsics=intrinsics,
                caption=job['prompt'],image_size=torch.tensor(hw))


def save_video(path,frames):
    # uint8 T,H,W,C, exact 15 fps and explicit frame count.
    temporary=path.with_suffix('.tmp.mp4')
    h,w=frames.shape[1:3]
    subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{w}x{h}','-r','15','-i','pipe:0','-an','-c:v','libx264','-crf','18',
        '-pix_fmt','yuv420p','-threads','4',str(temporary)],input=frames.tobytes(),check=True)
    os.replace(temporary,path)


@torch.no_grad()
def sample_video(model,preprocessor,job,plan,captions,rank,cp,group):
    from cosmos_transfer2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
    k=plan['k']
    length=(plan['frames']-1)//4+1
    t=length*(k+1)
    h,w=plan['height']//8,plan['width']//8
    cfg=dict(data=dict(k=k),text_dropout=0.,overlap_probability=0.,text_cache_entries=16)
    x0=camera=text=mask=untext=None
    if rank==0:
        data=input_sample(job,plan,captions)
        x0,camera,text,mask=preprocessor(data,cfg)
        # Same frozen text projection and zero raw embedding as training CFG.
        key=(True,None)
        if key not in preprocessor.cache:
            empty=torch.zeros(1,512,100352,device='cuda',dtype=torch.bfloat16)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                preprocessor.cache[key]=model.crossattn_proj(empty).detach().cpu()
            del empty
        untext=preprocessor.cache[key].cuda()
    x0=broadcast_tensor(x0,(1,16,t,h,w),torch.float32,0,group,cp)
    camera=broadcast_tensor(camera,(1,t,h//2,w//2,1536),torch.bfloat16,0,group,cp)
    text=broadcast_tensor(text,(1,512,1024),torch.bfloat16,0,group,cp)
    untext=broadcast_tensor(untext,(1,512,1024),torch.bfloat16,0,group,cp)
    mask=conditioning_mask(x0,k,False)
    generator=torch.Generator(device='cuda').manual_seed(job['seed'])
    latents=torch.randn(x0.shape,device='cuda',generator=generator)* (1-mask)+x0*mask
    scheduler=FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1,use_dynamic_shifting=False)
    scheduler.set_timesteps(plan['denoising_steps'],device='cuda',shift=plan['shift'])
    for timestamp in scheduler.timesteps:
        times=timestamp.expand(1,t)*(1-mask.mean(dim=(1,3,4)))+.1*mask.mean(dim=(1,3,4))
        predictions=[]
        for conditioned,embedding in ((True,text),(False,untext)):
            value=latents*(1-mask)+(x0*mask if conditioned else 0)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                local=model(x_B_C_T_H_W=value,timesteps_B_T=times,camera=camera,
                    crossattn_emb=embedding,crossattn_projected=True,
                    condition_video_input_mask_B_C_T_H_W=mask,
                    padding_mask=torch.zeros(1,h,w,device='cuda'),fps=torch.tensor([15.],device='cuda')).float()
            if cp>1:
                parts=[torch.empty_like(local) for _ in range(cp)]
                dist.all_gather(parts,local.contiguous(),group=group)
                predictions.append(torch.cat(parts,dim=-1))
            else:
                predictions.append(local)
        velocity=predictions[0]+plan['guidance']*(predictions[0]-predictions[1])
        latents=scheduler.step(velocity,timestamp,latents,return_dict=False,generator=generator)[0]
        latents=latents*(1-mask)+x0*mask
    target=latents[:,:,k//2*length:(k//2+1)*length]
    if not torch.isfinite(target).all():
        raise RuntimeError('Non-finite inference latent')
    if rank==0:
        decoded=preprocessor.tokenizer.decode(target.to(torch.bfloat16))
        if not torch.isfinite(decoded).all():
            raise RuntimeError('Non-finite decoded inference video')
        if list(decoded.shape)!=[1,3,plan['frames'],plan['height'],plan['width']]:
            raise ValueError(f'Unexpected decoded inference shape: {decoded.shape}')
        return decoded[0].add(1).mul(127.5).round().clamp(0,255).byte().permute(1,2,3,0).cpu().numpy()


@torch.no_grad()
def sample_video_batch(model,preprocessor,jobs,plan,captions,rank,cp,group):
    """Batch independent clips in one CP group; preserve one seed/solver per clip."""
    if not jobs:
        raise ValueError('A video batch must contain at least one job')
    if len(jobs)==1:
        torch.manual_seed(jobs[0]['seed'])
        frames=sample_video(model,preprocessor,jobs[0],plan,captions,rank,cp,group)
        return [frames] if rank==0 else None
    from cosmos_transfer2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
    batch=len(jobs)
    k=plan['k']
    length=(plan['frames']-1)//4+1
    t=length*(k+1)
    h,w=plan['height']//8,plan['width']//8
    cfg=dict(data=dict(k=k),text_dropout=0.,overlap_probability=0.,text_cache_entries=16)
    x0=camera=text=untext=None
    if rank==0:
        prepared=[]
        for job in jobs:
            torch.manual_seed(job['seed'])
            data=input_sample(job,plan,captions)
            prepared.append(preprocessor(data,cfg)[:3])
            del data
        x0,camera,text=[
            torch.cat([row[column] for row in prepared],dim=0)
            for column in range(3)
        ]
        del prepared
        key=(True,None)
        if key not in preprocessor.cache:
            empty=torch.zeros(1,512,100352,device='cuda',dtype=torch.bfloat16)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                preprocessor.cache[key]=model.crossattn_proj(empty).detach().cpu()
            del empty
        untext=preprocessor.cache[key].cuda().expand(batch,-1,-1).contiguous()
    x0=broadcast_tensor(x0,(batch,16,t,h,w),torch.float32,0,group,cp)
    camera=broadcast_tensor(camera,(batch,t,h//2,w//2,1536),torch.bfloat16,0,group,cp)
    text=broadcast_tensor(text,(batch,512,1024),torch.bfloat16,0,group,cp)
    untext=broadcast_tensor(untext,(batch,512,1024),torch.bfloat16,0,group,cp)
    mask=conditioning_mask(x0,k,False)
    generators=[torch.Generator(device='cuda').manual_seed(job['seed']) for job in jobs]
    noise=torch.cat([
        torch.randn((1,16,t,h,w),device='cuda',generator=generator)
        for generator in generators
    ],dim=0)
    latents=noise*(1-mask)+x0*mask
    del noise
    schedulers=[]
    for job in jobs:
        scheduler=FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,shift=1,use_dynamic_shifting=False)
        scheduler.set_timesteps(plan['denoising_steps'],device='cuda',shift=plan['shift'])
        schedulers.append(scheduler)
    frame_mask=mask.mean(dim=(1,3,4))
    # The existing model expands B,1,H,W into B,1,T,H,W.
    padding=torch.zeros(batch,1,h,w,device='cuda')
    fps=torch.full((batch,),15.,device='cuda')
    for timestamp in schedulers[0].timesteps:
        times=timestamp.expand(batch,t)*(1-frame_mask)+.1*frame_mask
        predictions=[]
        for conditioned,embedding in ((True,text),(False,untext)):
            value=latents*(1-mask)+(x0*mask if conditioned else 0)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                local=model(x_B_C_T_H_W=value,timesteps_B_T=times,camera=camera,
                    crossattn_emb=embedding,crossattn_projected=True,
                    condition_video_input_mask_B_C_T_H_W=mask,
                    padding_mask=padding,fps=fps).float()
            if cp>1:
                parts=[torch.empty_like(local) for _ in range(cp)]
                dist.all_gather(parts,local.contiguous(),group=group)
                predictions.append(torch.cat(parts,dim=-1))
            else:
                predictions.append(local)
        velocity=predictions[0]+plan['guidance']*(predictions[0]-predictions[1])
        latents=torch.cat([
            scheduler.step(velocity[index:index+1],timestamp,
                latents[index:index+1],return_dict=False,generator=generators[index])[0]
            for index,scheduler in enumerate(schedulers)
        ],dim=0)
        latents=latents*(1-mask)+x0*mask
    target=latents[:,:,k//2*length:(k//2+1)*length]
    if not torch.isfinite(target).all():
        raise RuntimeError('Non-finite batched inference latent')
    if rank==0:
        results=[]
        # Keep VAE decoding at one clip per call to limit its peak memory.
        for index in range(batch):
            decoded=preprocessor.tokenizer.decode(target[index:index+1].to(torch.bfloat16))
            if not torch.isfinite(decoded).all():
                raise RuntimeError('Non-finite decoded video in batch')
            if list(decoded.shape)!=[1,3,plan['frames'],plan['height'],plan['width']]:
                raise ValueError(f'Unexpected decoded inference shape: {decoded.shape}')
            results.append(decoded[0].add(1).mul(127.5).round().clamp(0,255)
                           .byte().permute(1,2,3,0).cpu().numpy())
        return results


def refinement_sigmas(strength, steps, shift):
    """Invert the scheduler's shift so strength is the actual starting noise sigma."""
    if (not math.isfinite(strength) or not 0 < strength <= .95
            or type(steps) is not int or not 1 <= steps <= 100
            or not math.isfinite(shift) or not 0 < shift <= 10
            or steps > max(1, int(strength*1000))):
        raise ValueError('Invalid refinement noise schedule')
    physical = np.linspace(strength, 0., steps+1, dtype=np.float64)[:-1]
    return physical / (shift - (shift-1) * physical)


def initialize_latents(encoded, noise, mask, sigma):
    if encoded.shape != noise.shape or not 0 <= sigma <= 1:
        raise ValueError('Refinement noise must match the encoded stage-one video')
    return ((1-sigma) * encoded + sigma * noise) * (1-mask) + encoded * mask


@torch.no_grad()
def refine_video(model, preprocessor, job, plan, rank, cp, group, *, initial_frames=None,
                 strength=.35, prefix=None, progress=None):
    """The entire stage-one target video initializes the unknown target latents."""
    from cosmos_transfer2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
    if not 0 < strength <= .95:
        raise ValueError('GPU refinement strength must be in (0, .95]')
    k = plan['k']
    length = (plan['frames']-1)//4+1
    total, h, w = length*(k+1), plan['height']//8, plan['width']//8
    start = k//2 * length
    encoded = camera = text = untext = None
    prefix_flag = [bool(prefix is not None) if rank == 0 else None]
    if cp > 1:
        dist.broadcast_object_list(prefix_flag, src=0)
    has_prefix = prefix_flag[0]
    preparation_error = [None]
    try:
        if rank == 0:
            if (not isinstance(initial_frames, np.ndarray) or initial_frames.dtype != np.uint8
                    or initial_frames.shape != (plan['frames'], plan['height'], plan['width'], 3)):
                raise ValueError('Refinement requires the full aligned RGB output from stage one')
            sample = input_sample(job, plan, None)
            # Target RGB comes only from stage one, never from real target truth.
            sample['videos'][-1] = torch.from_numpy(initial_frames.copy()).permute(3, 0, 1, 2)
            if has_prefix:
                if prefix.dtype != np.uint8 or prefix.shape != initial_frames[0].shape:
                    raise ValueError('Invalid previous refined overlap frame')
                sample['videos'][-1, :, 0] = torch.from_numpy(prefix.copy()).permute(2, 0, 1)
            encoded, camera, text, _ = preprocessor(sample, dict(data=dict(k=k), text_dropout=0.,
                overlap_probability=0., text_cache_entries=16))
            key = (True, None)
            if key not in preprocessor.cache:
                empty = torch.zeros(1, 512, 100352, device='cuda', dtype=torch.bfloat16)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    preprocessor.cache[key] = model.crossattn_proj(empty).detach().cpu()
                del empty
            untext = preprocessor.cache[key].cuda()
    except Exception as exc:
        preparation_error[0] = f'{type(exc).__name__}: {exc}'
    if cp > 1:
        dist.broadcast_object_list(preparation_error, src=0)
    if preparation_error[0]:
        raise RuntimeError('Refinement input encoding failed: '+preparation_error[0])
    encoded = broadcast_tensor(encoded, (1, 16, total, h, w), torch.float32, 0, group, cp)
    camera = broadcast_tensor(camera, (1, total, h//2, w//2, 1536), torch.bfloat16, 0, group, cp)
    text = broadcast_tensor(text, (1, 512, 1024), torch.bfloat16, 0, group, cp)
    untext = broadcast_tensor(untext, (1, 512, 1024), torch.bfloat16, 0, group, cp)
    mask = conditioning_mask(encoded, k, False)
    retained = None
    if has_prefix:
        # The causal VAE's first latent contains only this same-time overlap frame.
        mask[:, :, start:start+1] = 1
        reference_mask = torch.zeros_like(mask)
        reference_mask[:, :, start:start+1] = 1
        retained = encoded * reference_mask
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(plan['denoising_steps'], device='cuda', shift=plan['shift'],
                            sigmas=refinement_sigmas(strength, plan['denoising_steps'], plan['shift']))
    sigma = float(scheduler.sigmas[0])
    if not math.isclose(sigma, strength, abs_tol=1e-6):
        raise RuntimeError('Refinement scheduler strength differs from the requested sigma')
    generator = torch.Generator(device='cuda').manual_seed(job['seed'])
    noise = torch.randn(encoded.shape, device='cuda', generator=generator)
    latents = initialize_latents(encoded, noise, mask, sigma)
    del noise
    for index, timestamp in enumerate(scheduler.timesteps, 1):
        times = timestamp.expand(1, total)*(1-mask.mean(dim=(1, 3, 4))) + .1*mask.mean(dim=(1, 3, 4))
        predictions = []
        for conditioned, embedding in ((True, text), (False, untext)):
            known = encoded * mask if conditioned else retained if retained is not None else 0
            value = latents*(1-mask) + known
            with torch.autocast('cuda', dtype=torch.bfloat16):
                local = model(x_B_C_T_H_W=value, timesteps_B_T=times, camera=camera,
                    crossattn_emb=embedding, crossattn_projected=True,
                    condition_video_input_mask_B_C_T_H_W=mask,
                    padding_mask=torch.zeros(1, h, w, device='cuda'), fps=torch.tensor([15.], device='cuda')).float()
            if cp > 1:
                parts = [torch.empty_like(local) for _ in range(cp)]
                dist.all_gather(parts, local.contiguous(), group=group)
                predictions.append(torch.cat(parts, dim=-1))
            else:
                predictions.append(local)
        velocity = predictions[0] + plan['guidance']*(predictions[0]-predictions[1])
        latents = scheduler.step(velocity, timestamp, latents, return_dict=False, generator=generator)[0]
        latents = latents*(1-mask) + encoded*mask
        if rank == 0 and progress is not None and (index % 5 == 0 or index == len(scheduler.timesteps)):
            progress(index, len(scheduler.timesteps))
    target = latents[:, :, start:start+length]
    if not torch.isfinite(target).all():
        raise RuntimeError('Non-finite refined target latent')
    if rank == 0:
        decoded = preprocessor.tokenizer.decode(target.to(torch.bfloat16))
        if not torch.isfinite(decoded).all() or list(decoded.shape) != [1, 3, plan['frames'], plan['height'], plan['width']]:
            raise RuntimeError('Invalid decoded refinement output')
        return decoded[0].add(1).mul(127.5).round().clamp(0,255).byte().permute(1,2,3,0).cpu().numpy()



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='outputs/basic_stage1_24gpu/latest.pt')
    p.add_argument('--dataset',choices=['syncam','multicam'],default='syncam')
    p.add_argument('--split',choices=['train','val'],default='val')
    p.add_argument('--scene-index',type=int,default=0)
    p.add_argument('--source-cameras',help='Comma-separated distinct cameras; default selects k cameras')
    p.add_argument('--target-camera',default='cam10')
    p.add_argument('--prompt',help='English scene description; otherwise caption the first source video')
    p.add_argument('--steps',type=int,default=35)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--frames',type=int,default=81)
    p.add_argument('--height',type=int,default=432)
    p.add_argument('--width',type=int,default=768)
    p.add_argument('--output')
    p.add_argument('--check-only',action='store_true',help='CPU checkpoint/input check, without generating a video')
    args=p.parse_args()
    if args.frames not in range(1,82,4) or args.height%16 or args.width%16 or args.steps<1:
        p.error('Require 4n+1 frames <=81, dimensions divisible by 16, and steps>=1')
    checkpoint=rooted(args.checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if saved.get('trainer_version') not in (3,4) or saved.get('step',0)<1:
        raise ValueError('Require a completed update from trainer version 3 or 4')
    if saved['base_revision']!=prepare.JOBS['base'][2]:
        raise ValueError('Checkpoint frozen base revision differs')
    track_config, trainable_components = model_options(saved)
    if track_config and track_config.get('enabled'):
        raise ValueError('Track-enabled HOI checkpoints require the tabletop HOI evaluator')
    k=context_at_step(saved['config'],saved['step']-1)
    scenes=read_scenes([f'datasets/manifests/{args.dataset}_scenes.jsonl'],args.split)
    if not 0<=args.scene_index<len(scenes):p.error('Scene index outside the selected split')
    scene=scenes[args.scene_index]
    cameras=sorted(scene['videos'])
    sources=args.source_cameras.split(',') if args.source_cameras else [c for c in cameras if c!=args.target_camera][:k]
    if args.target_camera not in cameras or len(sources)!=k or len(set(sources))!=k or any(c not in cameras or c==args.target_camera for c in sources):
        p.error(f'Checkpoint expects {k} distinct source cameras and a different target camera')
    plan=dict(k=k,frames=args.frames,height=args.height,width=args.width,denoising_steps=args.steps,shift=5.,guidance=1.5)
    job=dict(scene=scene,source_cameras=sources,target_camera=args.target_camera,seed=args.seed,prompt=args.prompt or '')
    metadata=dict(checkpoint=str(layout.relative(checkpoint)),checkpoint_step=saved['step'],
        dataset=args.dataset,split=args.split,scene_id=scene['scene_id'],source_cameras=sources,
        target_camera=args.target_camera,seed=args.seed,**plan)
    if args.check_only:
        sample=input_sample(job,plan,None)
        assert list(sample['videos'].shape)==[k+1,3,args.frames,args.height,args.width]
        assert sample['videos'][-1].count_nonzero().item()==0
        assert torch.isfinite(sample['extrinsics']).all() and torch.isfinite(sample['intrinsics']).all()
        print(json.dumps(dict(status='input_check_passed',generated_video=False,**metadata)),flush=True)
        return
    if socket.gethostname().split('.')[0]!='h20-1':
        raise RuntimeError('Single-clip GPU inference is permitted only on h20-1')
    caption_state=layout.rooted('download-state/caption-job.json')
    from background_caption import active as caption_active
    if caption_state.exists() and caption_active(json.loads(caption_state.read_text())):
        raise RuntimeError('Caption workers still occupy this node')
    rank,cp=int(os.getenv('RANK','0')),int(os.getenv('WORLD_SIZE','1'))
    local=int(os.getenv('LOCAL_RANK','0'))
    if int(os.getenv('LOCAL_WORLD_SIZE',str(cp)))!=cp or (args.width//16)%cp:
        raise ValueError('Inference CP must stay on one node and divide the spatial patch width')
    output=rooted(args.output or f"outputs/inference_stage1/step-{saved['step']:06d}-{args.dataset}-{args.split}-{args.scene_index}-seed{args.seed}")
    layout.require_workspace_path(output)
    if (output/'generated.mp4').exists():raise FileExistsError('Choose a new --output to preserve the existing video')
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    configure_local_checkpoints(require=['base','vae','reason'])
    model=build_model(False,track_config,trainable_components).cuda().eval()
    load_adaptation(model,saved,required_components=('self_attn','cam_encoder'))
    model.requires_grad_(False)
    del saved
    preprocessor=Preprocessor(model) if rank==0 else None
    if rank==0 and not job['prompt']:
        from online_caption import OnlineCaptioner
        captioner=OnlineCaptioner('datasets/captions/inference_sources')
        job['prompt']=captioner(dict(caption_video=scene['videos'][sources[0]],scene_id=scene['scene_id'],dataset=args.dataset,split=args.split))
        del captioner
    group=None
    if cp>1:
        dist.init_process_group('gloo',timeout=timedelta(minutes=15))
        group,_=make_groups(cp,cp,rank)
        model.enable_context_parallel(group)
    try:
        started=time.monotonic()
        frames=sample_video(model,preprocessor,job,plan,None,rank,cp,group)
        if rank==0:
            output.mkdir(parents=True,exist_ok=True)
            video=output/'generated.mp4'
            save_video(video,frames)
            digest=hashlib.sha256()
            with checkpoint.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
            report=dict(status='generated',**metadata,prompt=job['prompt'],
                checkpoint_sha256=digest.hexdigest(),video=str(layout.relative(video)),
                seconds=round(time.monotonic()-started,3),created_at=datetime.now().astimezone().isoformat())
            prepare.save_json(output/'inference.json',report)
            print(json.dumps(report),flush=True)
        if cp>1:dist.barrier()
    finally:
        if cp>1:dist.destroy_process_group()


if __name__=='__main__':main()
