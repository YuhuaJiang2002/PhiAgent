#!/usr/bin/env python3
"""Public Predict2.5 -> Plenoptic supervised training, CP within nodes + DP.

Batch size is one scene per DP replica. Effective batch = DP * accumulation.
No private checkpoints or internal dataset registries are used.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import prepare_plenoptic as prepare
from local_checkpoints import configure_local_checkpoints
from plenoptic_data import PlenopticDataset, rooted
from plenoptic_model import build_model, trainable_name
from plenoptic_distributed import make_groups, verify_initial_parameters, synchronize_gradients


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def parameter_key(name):
    return name.replace('._checkpoint_wrapped_module','')


def collate_scene(items):
    if len(items) != 1:
        raise ValueError('This trainer uses batch_size=1 per DP replica')
    return items[0]


def view_order(k):
    order = list(range(k))
    order.insert(k//2, k)
    return order


def conditioning_mask(x0, k, overlap):
    length = x0.shape[2]//(k+1)
    start = (k//2)*length
    mask = torch.ones_like(x0[:,:1])
    mask[:,:,start:start+length] = 0
    if overlap:
        # Keep at least one supervised latent in short validation clips.
        mask[:,:,start:start+min(6,length-1)] = 1
    return mask


def broadcast_tensor(tensor, shape, dtype, source, group, cp):
    if cp == 1:
        return tensor
    if tensor is None:
        tensor = torch.empty(shape, device='cuda', dtype=dtype)
    dist.broadcast(tensor,src=source,group=group)
    return tensor


class Preprocessor:
    """Only CP leaders own the frozen VAE and Reason encoder.

    Construct before process-group initialization because upstream constructors
    include global barriers. Features are computed once per CP group and shared.
    """
    def __init__(self, net, caption_directory=None):
        from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        from cosmos_transfer2._src.predict2.text_encoders.text_encoder import TextEncoder, TextEncoderConfig
        from cosmos_transfer2._src.predict2.camera.utils import convert_camera_to_plucker_rays
        assert not dist.is_initialized()
        self.net = net
        self.rays = convert_camera_to_plucker_rays
        self.tokenizer = Wan2pt1VAEInterface(
            chunk_duration=81, load_mean_std=False, temporal_window=16,
            vae_pth=str(prepare.ROOT/prepare.JOBS['vae'][3]/prepare.JOBS['vae'][4][0]))
        self.encoder = TextEncoder(TextEncoderConfig(compute_online=True,
            ckpt_path=str(prepare.ROOT/prepare.JOBS['reason'][3]), embedding_concat_strategy='full_concat'))
        # Upstream cp_mesh/tp_mesh properties consult world_mesh once any
        # process group exists. This leader-only encoder has no parallel axes;
        # describe that explicitly without creating extra collective groups.
        from types import SimpleNamespace
        self.encoder.model.world_mesh = SimpleNamespace(mesh_dim_names=None)
        self.encoder.model.requires_grad_(False)
        self.cache = {}  # Bounded projected-text memory cache, ~1 MiB per entry.
        self.captioner=None
        if caption_directory:
            from online_caption import OnlineCaptioner
            self.captioner=OnlineCaptioner(caption_directory)

    @torch.no_grad()
    def __call__(self, sample, cfg):
        k = cfg['data']['k']
        order = view_order(k)
        videos = sample['videos']
        caption = self.captioner(sample) if self.captioner else sample['caption']
        latents = []
        for v in order:
            video = videos[v:v+1].cuda().to(torch.bfloat16)/127.5-1
            latent = self.tokenizer.encode(video).float()
            assert torch.isfinite(latent).all(), 'Non-finite VAE latent'
            latents.append(latent)
        x0 = torch.cat(latents,dim=2)
        w2c = sample['extrinsics'][order,::4].flatten(0,1).unsqueeze(0).cuda()
        intrinsics = sample['intrinsics'][order,::4].flatten(0,1).unsqueeze(0).cuda()
        camera = self.rays(w2c,intrinsics,sample['image_size'],patch_spatial=16,out_dtype=torch.bfloat16)
        assert camera.shape[1] == x0.shape[2]
        drop_text = bool(torch.rand(()) < cfg['text_dropout'])
        key = (drop_text, None if drop_text else caption)
        if key not in self.cache:
            if drop_text:
                text = torch.zeros(1,512,100352,device='cuda',dtype=torch.bfloat16)
            else:
                text = self.encoder.compute_text_embeddings_online({'caption':[caption]},'caption')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                projected = self.net.crossattn_proj(text).detach()
            if len(self.cache) >= cfg['text_cache_entries']:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = projected.cpu()
        text = self.cache[key].cuda()
        overlap = bool(torch.rand(()) < cfg['overlap_probability'])
        mask = conditioning_mask(x0,k,overlap)
        return x0, camera, text, mask


def rng_state():
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())


def save_checkpoint(path, model, optimizer, step, micro_step, cfg, dataset_hash, rank, world,caption_ledger=None):
    state = rng_state()
    states = [None]*world
    if world > 1:
        dist.all_gather_object(states,state)
    else:
        states[0] = state
    ledgers=[None]*world
    if world>1: dist.all_gather_object(ledgers,caption_ledger)
    else: ledgers[0]=caption_ledger
    if rank == 0:
        payload = dict(schema=1, trainer_version=3, step=step, micro_step=micro_step,
            trainable={k:v.detach().cpu() for k,v in model.state_dict().items()
                       if k in {parameter_key(n) for n,p in model.named_parameters() if p.requires_grad}},
            optimizer=optimizer.state_dict(), rng=states, config=cfg, dataset_hash=dataset_hash,
            caption_ledgers=ledgers,
            base_revision=prepare.JOBS['base'][2], vae_revision=prepare.JOBS['vae'][2],
            reason_revision=prepare.JOBS['reason'][2], world_size=world)
        temporary = path.with_suffix('.tmp')
        torch.save(payload,temporary)
        os.replace(temporary,path)
        prepare.save_json(path.with_suffix('.json'),dict(step=step,micro_step=micro_step,
            path=str(path.relative_to(prepare.ROOT)),world_size=world,base_revision=payload['base_revision']))
    if world > 1:
        dist.barrier()


def load_trainable(model, saved):
    expected = {parameter_key(n) for n,p in model.named_parameters() if p.requires_grad}
    if set(saved['trainable']) != expected:
        raise RuntimeError('Resume trainable keys do not match the explicit freeze rule')
    if saved['base_revision'] != prepare.JOBS['base'][2]:
        raise RuntimeError('Resume uses a different frozen base checkpoint')
    state = model.state_dict()
    for k,v in saved['trainable'].items():
        if state[k].shape != v.shape:
            raise RuntimeError(f'Resume tensor shape mismatch: {k}')
        state[k] = v
    model.load_state_dict(state,strict=True)


def context_at_step(cfg,step):
    schedule=cfg.get('context_schedule')
    if not schedule:
        return cfg['data']['k']
    previous=0
    for stage in schedule:
        if stage['until_step']<=previous or stage['k'] not in range(1,5):
            raise ValueError('Context schedule must have increasing boundaries and k=1..4')
        previous=stage['until_step']
    for stage in schedule:
        if step<stage['until_step']:
            return stage['k']
    return schedule[-1]['k']


def run(cfg, resume=None, weights_from=None):
    cfg.setdefault('dp_backend','gloo')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from cosmos_transfer2._src.imaginaire.flags import SMOKE
    from cosmos_transfer2._src.predict2.schedulers.rectified_flow import RectifiedFlow
    assert not SMOKE, 'Never use COSMOS smoke weights for this trainer'
    rank,world = int(os.getenv('RANK','0')),int(os.getenv('WORLD_SIZE','1'))
    local = int(os.getenv('LOCAL_RANK','0'))
    cp = cfg['context_parallel_size']
    if world != cfg['world_size'] or world % cp or (cfg['data']['width']//16)%cp:
        raise ValueError('Launch topology or spatial width does not match config')
    local_world = int(os.getenv('LOCAL_WORLD_SIZE',str(world)))
    if local_world % cp:
        raise ValueError('CP groups must fit wholly within each node')
    dp, dp_rank, cp_rank = world//cp,rank//cp,rank%cp
    source = dp_rank*cp
    torch.cuda.set_device(local)
    torch.set_num_threads(cfg.get('cpu_threads',4))
    configure_local_checkpoints(require=['base','vae','reason'])
    output = rooted(cfg['output'])
    output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(cfg['seed'])
    model = build_model(cfg['activation_checkpointing']).cuda().train()
    saved = torch.load(rooted(resume or weights_from),map_location='cpu',weights_only=True,mmap=True) if (resume or weights_from) else None
    if saved is not None:
        load_trainable(model,saved)
    dataset = PlenopticDataset(**cfg['data']) if cp_rank == 0 else None
    data_identity = dict(items=dataset.items,scenes=[(s['dataset'],s['scene_id']) for s in dataset.scenes]) if dataset else None
    if dataset and dataset.generated_hash:
        data_identity['generated_hash'] = dataset.generated_hash
    if dataset and cfg['data'].get('caption_mode')=='online':
        from caption_plenoptic import caption_spec
        data_identity['caption_spec_hash']=caption_spec()[1]
    data_hash = signature(data_identity) if dataset else None
    caption_directory=cfg['data']['captions'] if cfg['data'].get('caption_mode')=='online' else None
    preprocessor = Preprocessor(model,caption_directory) if cp_rank == 0 else None
    if world > 1:
        dist.init_process_group('gloo',timeout=timedelta(minutes=5))
    group,leaders = make_groups(world,cp,rank,cfg['dp_backend']) if world>1 else (None,None)
    if group is not None:
        model.enable_context_parallel(group)
    if world > 1:
        hashes = [data_hash]
        dist.broadcast_object_list(hashes,src=0)
        if data_hash is not None and data_hash != hashes[0]:
            raise ValueError('Dataset/caption snapshots differ between nodes')
        data_hash = hashes[0]
    parameters = [p for p in model.parameters() if p.requires_grad]
    if world>1:
        verify_initial_parameters(parameters,world)
    training_model=model
    optimizer = torch.optim.AdamW(parameters,lr=cfg['learning_rate'],betas=tuple(cfg['betas']),
                                  weight_decay=cfg['weight_decay'],eps=1e-8,fused=True)
    step,micro_step = 0,0
    torch.manual_seed(cfg['seed']+dp_rank+100)
    if resume:
        if saved.get('trainer_version')!=3:
            raise ValueError('Exact resume requires trainer version 3; use --weights-from for older validation checkpoints')
        if saved['config'].get('dp_backend','gloo')!=cfg['dp_backend']:
            raise ValueError('Exact resume requires the same gradient backend; use --weights-from for a new stage')
        compare = ('context_parallel_size','data','learning_rate','betas','weight_decay','seed',
                   'gradient_accumulation','text_dropout','video_dropout','overlap_probability',
                   'context_schedule','gradient_clip')
        if saved['world_size'] != world or any(saved['config'].get(k) != cfg.get(k) for k in compare):
            raise ValueError('Exact resume requires the same data/topology/optimizer; use --weights-from for a new stage')
        if saved['dataset_hash'] != data_hash:
            raise ValueError('Dataset/caption snapshot changed since checkpoint')
        optimizer.load_state_dict(saved['optimizer'])
        step,micro_step = saved['step'],saved['micro_step']
        torch.set_rng_state(saved['rng'][rank]['cpu'])
        torch.cuda.set_rng_state(saved['rng'][rank]['cuda'])
        if preprocessor and preprocessor.captioner:
            preprocessor.captioner.restore(saved['caption_ledgers'][rank])
    del saved
    if rank == 0:
        prepare.save_json(output/'resolved_config.json',{**cfg,'effective_batch':dp*cfg['gradient_accumulation'],
            'dataset_hash':data_hash,'cp_axis':'spatial_width','base_revision':prepare.JOBS['base'][2]})
    # Each CP leader decodes once; other ranks receive encoded features.
    iterator = None
    if dataset:
        dataset.k=context_at_step(cfg,step)
        sampler = DistributedSampler(dataset,num_replicas=dp,rank=dp_rank,shuffle=True,seed=cfg['seed'],drop_last=False)
        per_epoch = len(sampler)
        epoch, offset = divmod(micro_step,per_epoch)
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        loader = DataLoader(dataset,batch_size=1,sampler=sampler,num_workers=0,collate_fn=collate_scene,
                            generator=torch.Generator().manual_seed(cfg['seed']))
        if offset:
            # Skip sampler indices without decoding already completed videos.
            # Share the loader RNG so later epoch initialization is unchanged.
            first_loader = DataLoader(dataset,batch_size=1,sampler=list(sampler)[offset:],
                                      num_workers=0,collate_fn=collate_scene,generator=loader.generator)
            iterator = iter(first_loader)
        else:
            iterator = iter(loader)
    flow = RectifiedFlow(None,train_time_distribution='logitnormal',shift=5,device='cuda')
    h,w = cfg['data']['height']//8,cfg['data']['width']//8
    tensor_kwargs = dict(device='cuda',dtype=torch.float32)
    first = True
    last_k=None
    if rank == 0:
        print(json.dumps(dict(event='training_start',step=step,world_size=world,cp=cp,dp=dp,
                             effective_batch=dp*cfg['gradient_accumulation'])),flush=True)
    while step < cfg['max_steps']:
        current_k=context_at_step(cfg,step)
        stage_cfg={**cfg,'data':{**cfg['data'],'k':current_k}}
        if dataset: dataset.k=current_k
        t=((cfg['data']['frames']-1)//4+1)*(current_k+1)
        if current_k!=last_k:
            first=True
            last_k=current_k
            if rank==0:
                print(json.dumps(dict(event='context_stage',step=step,k=current_k)),flush=True)
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((),device='cuda')
        probe_names = [next(n for n,p in model.named_parameters() if p.requires_grad and term in n)
                       for term in ('.cam_encoder.','.self_attn.q_proj.')]
        probes = {n:p.detach().clone() for n,p in model.named_parameters() if first and n in probe_names}
        frozen_probes = {n:p.detach().clone() for n,p in model.named_parameters()
                         if first and parameter_key(n).endswith('blocks.0.cross_attn.q_proj.weight')}
        for accumulation in range(cfg['gradient_accumulation']):
            x0=camera=text=mask=None
            if cp_rank == 0:
                try:
                    sample = next(iterator)
                except StopIteration:
                    epoch += 1
                    dataset.set_epoch(epoch)
                    sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    sample = next(iterator)
                x0,camera,text,mask = preprocessor(sample,stage_cfg)
            x0 = broadcast_tensor(x0,(1,16,t,h,w),torch.float32,source,group,cp)
            camera = broadcast_tensor(camera,(1,t,h//2,w//2,1536),torch.bfloat16,source,group,cp)
            text = broadcast_tensor(text,(1,512,1024),torch.bfloat16,source,group,cp)
            mask = broadcast_tensor(mask,(1,1,t,h,w),torch.float32,source,group,cp)
            noise = torch.randn_like(x0) if cp_rank == 0 else None
            noise = broadcast_tensor(noise,x0.shape,torch.float32,source,group,cp)
            times = flow.get_discrete_timestamp(flow.sample_train_time(1),tensor_kwargs) if cp_rank==0 else None
            times = broadcast_tensor(times,(1,),torch.float32,source,group,cp)
            dropout = torch.tensor([float(torch.rand(()) < cfg['video_dropout'])],device='cuda') if cp_rank==0 else None
            dropout = broadcast_tensor(dropout,(1,),torch.float32,source,group,cp)
            xt,target = flow.get_interpolation(noise,x0,flow.get_sigmas(times,tensor_kwargs))
            xt = xt*(1-mask) + x0*mask*(1-dropout)
            frame_times = times[:,None]*(1-mask.mean(dim=(1,3,4))) + .1*mask.mean(dim=(1,3,4))
            # Ordered gradient averaging follows completed CP backward work.
            sync = nullcontext()
            with sync:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction = training_model(x_B_C_T_H_W=xt,timesteps_B_T=frame_times,
                        crossattn_emb=text,crossattn_projected=True,camera=camera,
                        condition_video_input_mask_B_C_T_H_W=mask,
                        padding_mask=torch.zeros(1,h,w,device='cuda'),fps=torch.tensor([15.],device='cuda')).float()
                target = target.chunk(cp,dim=-1)[cp_rank]
                local_mask = mask.chunk(cp,dim=-1)[cp_rank]
                # Matches the public RF loss: clean-condition errors are zero,
                # followed by mean over all elements and global CP/DP averaging.
                loss = (((prediction-target)*(1-local_mask))**2).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                (loss/cfg['gradient_accumulation']).backward()
            total_loss += loss.detach()/cfg['gradient_accumulation']
            micro_step += 1
        synchronize_gradients(parameters,world,cp,group,leaders,dp_backend=cfg['dp_backend'])
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip'],error_if_nonfinite=True)
        if first:
            bad_frozen = [n for n,p in model.named_parameters() if not p.requires_grad and p.grad is not None]
            bad_trainable = [n for n,p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
            if bad_frozen or bad_trainable:
                raise RuntimeError(f'Gradient audit failed: frozen={bad_frozen}, trainable={bad_trainable}')
        optimizer.step()
        step += 1
        total_loss = total_loss.cpu()
        if world > 1:
            dist.all_reduce(total_loss)
            total_loss /= world
        memory = torch.tensor([torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()])
        if world>1: dist.all_reduce(memory,op=dist.ReduceOp.MAX)
        if first:
            params = dict(model.named_parameters())
            updates = {n:(params[n].detach()-v).float().abs().max().item() for n,v in probes.items()}
            if not updates or not all(v > 0 for v in updates.values()):
                raise RuntimeError(f'Parameters did not update: {updates}')
            if any(not torch.equal(params[n],v) for n,v in frozen_probes.items()):
                raise RuntimeError('Frozen cross-attention parameter changed')
            if rank == 0:
                proof=dict(status='passed',scope='pretrained 2B, real video+caption+VAE, forward/backward/update',
                         step=step,resumed=bool(resume),world_size=world,cp=cp,config=cfg,
                         k=current_k,
                         finite_loss=total_loss.item(),trainable_max_updates=updates,
                         frozen_gradient_violations=bad_frozen,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
                prepare.save_json(prepare.ROOT/'download-state/h20-training-check.json',proof)
                prepare.save_json(output/f'update-check-k{current_k}.json',proof)
            first = False
        if rank == 0:
            row = dict(step=step,k=current_k,phase=cfg.get('phase','diagnostic'),
                       loss=total_loss.item(),gradient_norm=gradient_norm.item(),
                       seconds=round(time.monotonic()-started,3),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                       max_rank_allocated_gib=memory[0].item()/2**30,max_rank_reserved_gib=memory[1].item()/2**30,
                       updated_at=datetime.now().astimezone().isoformat())
            print(json.dumps(row),flush=True)
            with (output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
        # A shared stop request is coordinated at the optimizer boundary so
        # every rank participates in saving the same completed step.
        stop = torch.tensor([int(rank == 0 and (output/'STOP').exists())])
        if world > 1:
            dist.broadcast(stop,src=0)
        stopping = bool(stop.item())
        milestone=any(stage['until_step']==step for stage in cfg.get('context_schedule',[]))
        if step%cfg['save_every']==0 or step==cfg['max_steps'] or stopping or milestone:
            ledger=preprocessor.captioner.used if preprocessor and preprocessor.captioner else None
            save_checkpoint(output/'latest.pt',model,optimizer,step,micro_step,cfg,data_hash,rank,world,ledger)
            if rank==0 and (milestone or step==cfg['max_steps']):
                link=output/f'step-{step:06d}.pt'
                if not link.exists(): os.link(output/'latest.pt',link)
        if stopping:
            if rank == 0:
                (output/'STOP').unlink(missing_ok=True)
                print(json.dumps(dict(event='stopped_after_checkpoint',step=step)),flush=True)
            break
    if world>1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/plenoptic/smoke_1gpu.json')
    parser.add_argument('--resume')
    parser.add_argument('--weights-from')
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--output')
    args = parser.parse_args()
    manifest = os.getenv('PLENOPTIC_RUN_MANIFEST')
    if not manifest or not Path(manifest).is_file():
        parser.error('GPU training must be started through tools/launch.py so provenance is recorded')
    if args.resume and args.weights_from:
        parser.error('Choose exact --resume or --weights-from for a new stage')
    cfg = json.loads(rooted(args.config).read_text())
    if args.max_steps is not None:
        cfg['max_steps'] = args.max_steps
    if args.output:
        cfg['output'] = args.output
    run(cfg,args.resume,args.weights_from)


if __name__ == '__main__':
    main()
