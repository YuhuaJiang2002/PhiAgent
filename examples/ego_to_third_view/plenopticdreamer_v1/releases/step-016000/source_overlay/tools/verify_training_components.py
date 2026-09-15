#!/usr/bin/env python3
"""Check CP+DP gradient scaling, accumulation, freeze and checkpoint restore.

Small randomly initialized Camera DiT: this is NOT a pretrained training test.
"""
import plenoptic_paths as layout
from contextlib import nullcontext
import copy
import json
import os
import torch
import torch.distributed as dist

import prepare_plenoptic as prepare
from plenoptic_model import SpatialCameraDIT, NET_2B, SACConfig, CheckpointMode, freeze_for_plenoptic
from train_plenoptic import conditioning_mask, view_order, parameter_key, save_checkpoint, load_trainable, synchronize_gradients
from plenoptic_distributed import make_groups, verify_initial_parameters


def main():
    rank,world = int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    dist.init_process_group('gloo')
    cp = int(os.environ.get('VERIFY_CP','2'))
    group,leaders = make_groups(world,cp,rank)
    torch.manual_seed(57)
    kwargs = {**NET_2B,'model_channels':256,'num_blocks':2,'num_heads':8,
              'crossattn_emb_channels':64,'use_crossattn_projection':False,
              'sac_config':SACConfig(mode=CheckpointMode.NONE)}
    reference = SpatialCameraDIT(**kwargs).cuda()
    freeze_for_plenoptic(reference)
    actual = copy.deepcopy(reference)
    actual.enable_selective_checkpoint(SACConfig(mode=CheckpointMode.BLOCK_WISE),actual.blocks)
    actual.enable_context_parallel(group)
    verify_initial_parameters([p for p in actual.parameters() if p.requires_grad],world)
    opt = torch.optim.AdamW((p for p in actual.parameters() if p.requires_grad),lr=2e-5,fused=True)
    refopt = torch.optim.AdamW((p for p in reference.parameters() if p.requires_grad),lr=2e-5,fused=True)
    path = layout.rooted('outputs/component_validation/latest.pt')
    path.parent.mkdir(parents=True,exist_ok=True)
    worst_grad = 0.
    try:
        assert view_order(4) == [0,1,4,2,3]
        for k in range(1,5):
            m=conditioning_mask(torch.zeros(1,16,21*(k+1),2,4,device='cuda'),k,True)
            assert (m==0).sum().item()==15*2*4
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            refopt.zero_grad(set_to_none=True)
            for micro in range(3):
                torch.manual_seed(1000+100*step+micro+10*(rank//cp))
                x=torch.randn(1,16,42,4,16,device='cuda')
                camera=torch.randn(1,42,2,8,1536,device='cuda')
                text=torch.randn(1,12,64,device='cuda')
                mask=conditioning_mask(x,1,True)
                target=torch.randn_like(x)
                inputs=dict(x_B_C_T_H_W=x,timesteps_B_T=torch.full((1,42),234.,device='cuda'),
                    camera=camera,crossattn_emb=text,condition_video_input_mask_B_C_T_H_W=mask,
                    padding_mask=torch.zeros(1,4,16,device='cuda'),fps=torch.tensor([15.],device='cuda'))
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    ref=reference(**inputs).float()
                (((ref-target)*(1-mask))**2).mean().div(3).backward()
                with nullcontext():
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        out=actual(**inputs).float()
                    local_target=target.chunk(cp,dim=-1)[rank%cp]
                    local_mask=mask.chunk(cp,dim=-1)[rank%cp]
                    (((out-local_target)*(1-local_mask))**2).mean().div(3).backward()
            synchronize_gradients([p for p in actual.parameters() if p.requires_grad],world,cp,group,leaders)
            ref_params=dict(reference.named_parameters())
            for n,p in actual.named_parameters():
                r=ref_params[parameter_key(n)]
                if not p.requires_grad:
                    assert p.grad is None and torch.equal(p,r)
                    continue
                assert p.grad is not None and torch.isfinite(p.grad).all()
                cpu_grad=r.grad.cpu()
                dist.all_reduce(cpu_grad)
                r.grad.copy_(cpu_grad/world)
                error=((p.grad-r.grad).float().norm()/r.grad.float().norm().clamp_min(1e-7)).item()
                worst_grad=max(worst_grad,error)
                assert error<.06,(n,error)
            opt.step();refopt.step()
            save_checkpoint(path,actual,opt,step+1,(step+1)*3,{'fixture':'small random model'},'fixture',rank,world)
            # Only rank 0 has a checkpoint file on a non-shared filesystem.
            objects=[torch.load(path,map_location='cpu',weights_only=True,mmap=True) if rank==0 else None]
            dist.broadcast_object_list(objects,src=0)
            saved=objects[0]
            expected={n:p.detach().clone() for n,p in actual.named_parameters() if p.requires_grad}
            with torch.no_grad():
                for p in actual.parameters():
                    if p.requires_grad:
                        p.add_(1.)
            load_trainable(actual,saved)
            opt.load_state_dict(saved['optimizer'])
            for n,p in actual.named_parameters():
                if p.requires_grad:
                    assert torch.equal(p,expected[n]),n
            next_value=torch.rand(4)
            torch.set_rng_state(saved['rng'][rank]['cpu'])
            assert torch.equal(torch.rand(4),next_value),'RNG restore differs'
        if rank==0:
            result=dict(status='passed',scope='small random Camera DiT; not pretrained 2B training',
                        world_size=world,cp=cp,dp=world//cp,gradient_accumulation=3,
                        gradient_sync='nccl_cp_gloo_dp',
                        max_gradient_relative_l2=worst_grad,optimizer_steps=2,
                        freeze_verified=True,checkpoint_weights_optimizer_rng_restored=True)
            prepare.save_json(layout.rooted('download-state/h20-training-components-check.json'),result)
            print(json.dumps(result),flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
