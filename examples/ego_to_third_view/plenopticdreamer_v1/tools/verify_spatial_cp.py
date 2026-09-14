#!/usr/bin/env python3
"""Compare CP=1 and spatial CP forward/backward for 81-frame k=1..4."""
import copy
import json
import os
import torch
import torch.distributed as dist
from plenoptic_model import SpatialCameraDIT, NET_2B, SACConfig, CheckpointMode, freeze_for_plenoptic
import prepare_plenoptic as prepare


def relative_error(actual, expected):
    return ((actual.float()-expected.float()).norm() / expected.float().norm().clamp_min(1e-7)).item()


def main():
    manifest = os.getenv('PLENOPTIC_RUN_MANIFEST')
    if not manifest or not os.path.isfile(manifest):
        raise RuntimeError('Launch this GPU check through tools/launch.py')
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(2)
    dist.init_process_group('nccl', device_id=torch.device('cuda', int(os.environ['LOCAL_RANK'])))
    world = dist.get_world_size()
    records = []
    try:
        for k in range(1, 5):
            torch.manual_seed(73)
            config = {**NET_2B, 'model_channels':256, 'num_blocks':2, 'num_heads':8,
                      'crossattn_emb_channels':64, 'use_crossattn_projection':False,
                      'sac_config':SACConfig(mode=CheckpointMode.NONE)}
            ref = SpatialCameraDIT(**config).cuda()
            freeze_for_plenoptic(ref)
            cp = copy.deepcopy(ref)
            if k == 4:
                cp.enable_selective_checkpoint(SACConfig(mode=CheckpointMode.BLOCK_WISE), cp.blocks)
            cp.enable_context_parallel(dist.group.WORLD)
            t, h, w = 21*(k+1), 4, 16
            torch.manual_seed(91+k)
            x = torch.randn(1,16,t,h,w,device='cuda',requires_grad=True)
            xc = x.detach().clone().requires_grad_()
            camera = torch.randn(1,t,h//2,w//2,1536,device='cuda')
            mask = torch.zeros(1,1,t,h,w,device='cuda')
            mask[:,:,:21*k+6] = 1
            time = torch.full((1,t),345.,device='cuda')
            time[:,:21*k+6] = .1
            context = torch.randn(1,12,64,device='cuda')
            kwargs = dict(timesteps_B_T=time, crossattn_emb=context, camera=camera,
                          condition_video_input_mask_B_C_T_H_W=mask,
                          padding_mask=torch.zeros(1,h,w,device='cuda'), fps=torch.tensor([15.],device='cuda'))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                expected = ref(x_B_C_T_H_W=x, **kwargs).float()
            target = torch.randn_like(expected)
            loss = (((expected-target)*(1-mask))**2).sum() / expected.numel()
            loss.backward()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                actual = cp(x_B_C_T_H_W=xc, **kwargs).float()
            target_local = target.chunk(world,dim=-1)[rank].contiguous()
            mask_local = mask.chunk(world,dim=-1)[rank]
            local_loss = (((actual-target_local)*(1-mask_local))**2).sum() / expected.numel()
            local_loss.backward()
            parts = [torch.empty_like(actual) for _ in range(world)]
            dist.all_gather(parts,actual.contiguous())
            forward_error = relative_error(torch.cat(parts,dim=-1),expected)
            dist.all_reduce(xc.grad)
            input_error = relative_error(xc.grad,x.grad)
            gradients = {}
            expected_params = dict(ref.named_parameters())
            for name,p in cp.named_parameters():
                canonical = name.replace('._checkpoint_wrapped_module','')
                if p.requires_grad:
                    assert p.grad is not None and torch.isfinite(p.grad).all(), canonical
                    dist.all_reduce(p.grad)
                    gradients[canonical] = relative_error(p.grad,expected_params[canonical].grad)
                else:
                    assert p.grad is None, canonical
            camera_nonzero = sum(p.grad.float().norm().item() for n,p in ref.named_parameters() if '.cam_encoder.' in n)
            assert camera_nonzero > 0, 'Degenerate zero-gradient test'
            max_param_error = max(gradients.values())
            row = dict(k=k, latent_frames=t, cp=world, forward_relative_l2=forward_error,
                       input_gradient_relative_l2=input_error, max_parameter_gradient_relative_l2=max_param_error,
                       camera_gradient_norm=camera_nonzero, checkpointing=(k==4))
            records.append(row)
            assert forward_error < .02 and input_error < .05 and max_param_error < .08, (row,gradients)
            if rank == 0:
                print(json.dumps(row),flush=True)
            del ref,cp,x,xc,expected,actual
            torch.cuda.empty_cache()
        if rank == 0:
            prepare.save_json(prepare.ROOT/'download-state/h20-spatial-cp-check.json',
                dict(status='passed', scope='small Camera DiT; real 21-frame latent length, k=1..4; not pretrained training', cases=records))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
