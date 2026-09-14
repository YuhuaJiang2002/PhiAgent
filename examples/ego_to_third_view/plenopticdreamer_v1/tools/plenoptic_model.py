"""Audited public-base initialization and spatial CP for released Camera DiT.

Spatial CP keeps all temporal positions (81 frames -> 21 latents per view).
The pinned dense attention kernel is invariant to the rank-major spatial order.
Sparse attention and per-block absolute embeddings are intentionally rejected.
"""
import json
import os
import torch
import torch.distributed as dist
from einops import rearrange
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

import prepare_plenoptic as prepare
from local_checkpoints import configure_local_checkpoints
from cosmos_transfer2._src.predict2.camera.networks.dit_multiview_camera_ar import (
    CameraARMiniTrainDITwithConditionalMask, SACConfig, CheckpointMode,
)

# Resolved from the pinned public camera experiment; see reports/config audit.
NET_2B = dict(max_img_h=240, max_img_w=240, max_frames=128,
    in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
    model_channels=2048, num_blocks=28, num_heads=16, concat_padding_mask=True,
    pos_emb_cls='rope3d', pos_emb_learnable=True, pos_emb_interpolation='crop',
    use_adaln_lora=True, adaln_lora_dim=256, atten_backend='minimal_a2a',
    extra_per_block_abs_pos_emb=False, rope_h_extrapolation_ratio=3.,
    rope_w_extrapolation_ratio=3., rope_t_extrapolation_ratio=1.,
    rope_enable_fps_modulation=False, use_crossattn_projection=True,
    crossattn_proj_in_channels=100352, crossattn_emb_channels=1024,
    timestep_scale=0.001, use_wan_fp32_strategy=True)


class SpatialCameraDIT(CameraARMiniTrainDITwithConditionalMask):
    def __init__(self, *args, **kwargs):
        if kwargs.get('extra_per_block_abs_pos_emb', False) or kwargs.get('n_dense_blocks', -1) != -1:
            raise ValueError('Spatial CP currently requires dense attention and no absolute per-block positions')
        super().__init__(*args, **kwargs)
        self.spatial_cp_size, self.spatial_cp_rank = 1, 0

    def enable_context_parallel(self, process_group=None):
        if self.atten_backend != 'minimal_a2a':
            raise ValueError('Spatial CP requires the validated minimal_a2a attention backend')
        self.spatial_cp_size = dist.get_world_size(process_group)
        self.spatial_cp_rank = dist.get_rank(process_group)
        if self.num_heads % self.spatial_cp_size:
            raise ValueError('Number of attention heads must be divisible by CP size')
        super().enable_context_parallel(process_group)

    def disable_context_parallel(self):
        super().disable_context_parallel()
        self.spatial_cp_size, self.spatial_cp_rank = 1, 0

    def prepare_embedded_sequence(self, x, fps=None, padding_mask=None):
        if self.spatial_cp_size == 1:
            return super().prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)
        if self.concat_padding_mask:
            padding_mask = resize(padding_mask, list(x.shape[-2:]), interpolation=InterpolationMode.NEAREST)
            x = torch.cat([x, padding_mask.unsqueeze(1).repeat(1,1,x.shape[2],1,1)], dim=1)
        embedded = self.x_embedder(x)
        b, t, h, w, d = embedded.shape
        if w % self.spatial_cp_size:
            raise ValueError(f'Patch-grid width {w} must be divisible by CP={self.spatial_cp_size}')
        # Bypass upstream temporal CP expansion and generate exact global 3D RoPE.
        rope = self.pos_embedder.generate_embeddings(embedded.shape, fps=fps)
        rope = rearrange(rope, '(t h w) a b d -> t h w a b d', t=t,h=h,w=w)
        rope = rope.chunk(self.spatial_cp_size, dim=2)[self.spatial_cp_rank].contiguous()
        rope = rearrange(rope, 't h w a b d -> (t h w) a b d')
        local = embedded.chunk(self.spatial_cp_size, dim=3)[self.spatial_cp_rank].contiguous()
        return local, rope, None

    def forward(self, *args, camera=None, crossattn_projected=False, **kwargs):
        if self.spatial_cp_size > 1:
            camera = camera.chunk(self.spatial_cp_size, dim=3)[self.spatial_cp_rank].contiguous()
        if not crossattn_projected:
            return super().forward(*args, camera=camera, **kwargs)
        if kwargs['crossattn_emb'].shape[-1] != self.crossattn_proj[0].out_features:
            raise ValueError('Projected text has the wrong feature dimension')
        if any(p.requires_grad for p in self.crossattn_proj.parameters()):
            raise ValueError('Preprojected text requires a frozen text projection')
        previous = self.use_crossattn_projection
        self.use_crossattn_projection = False
        try:
            return super().forward(*args, camera=camera, **kwargs)
        finally:
            self.use_crossattn_projection = previous


def trainable_name(name):
    # Do not accidentally train cross-attention or all AdaLN modules by substring.
    return '.self_attn.' in name or '.cam_encoder.' in name


def freeze_for_plenoptic(model):
    model.requires_grad_(False)
    names = []
    for name, parameter in model.named_parameters():
        if trainable_name(name):
            parameter.requires_grad_(True)
            names.append(name)
    if not names or not any('.cam_encoder.' in n for n in names):
        raise RuntimeError('No camera/self-attention parameters selected')
    return dict(rule='blocks.*.self_attn.* and blocks.*.cam_encoder.*', names=names,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad))


def checkpoint_state(path):
    raw = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    container = None
    for key in ('state_dict', 'model'):
        if key in raw and isinstance(raw[key], dict):
            raw, container = raw[key], key
            break
    # The pinned downloadable file is EMA. Accept a bare or namespaced DiT.
    prefix = 'net_ema.' if any(k.startswith('net_ema.') for k in raw) else (
             'net.' if any(k.startswith('net.') for k in raw) else '')
    selected = {k[len(prefix):]: v for k,v in raw.items() if k.startswith(prefix)}
    other = sorted(k for k in raw if not k.startswith(prefix))
    if other and not all(k.startswith('net.') for k in other):
        raise ValueError(f'Unexpected checkpoint sections: {other[:10]}')
    return selected, dict(container=container, selected_prefix=prefix, ignored_other_namespace_keys=other)


def audit_and_load_base(model, report_path=None):
    configure_local_checkpoints(require=['base'])
    path = prepare.ROOT / prepare.JOBS['base'][3] / prepare.JOBS['base'][4][0]
    state, selection = checkpoint_state(path)
    target = model.state_dict()
    source_shapes = {k:list(v.shape) for k,v in state.items() if isinstance(v, torch.Tensor) and not k.endswith('._extra_state')}
    target_shapes = {k:list(v.shape) for k,v in target.items() if isinstance(v, torch.Tensor) and not k.endswith('._extra_state')}
    missing = sorted(target_shapes.keys() - source_shapes.keys())
    allowed = sorted(n for n in target_shapes if '.cam_encoder.' in n)
    unexpected = sorted(source_shapes.keys() - target_shapes.keys())
    mismatch = {k:[source_shapes[k],target_shapes[k]] for k in target_shapes.keys() & source_shapes.keys()
                if source_shapes[k] != target_shapes[k]}
    report = dict(status='running', checkpoint=str(path.relative_to(prepare.ROOT)),
        revision=prepare.JOBS['base'][2], selection=selection,
        source_tensors=len(source_shapes), target_tensors=len(target_shapes),
        missing=missing, allowed_new_camera=allowed, unexpected=unexpected, shape_mismatch=mismatch,
        source_shapes=source_shapes, target_shapes=target_shapes,
        camera_initialization='public truncated-normal initialization; trained Plenoptic weights are not released')
    okay = missing == allowed and not unexpected and not mismatch
    report['status'] = 'passed' if okay else 'failed'
    if report_path is not None:
        prepare.save_json(report_path, report)
    if not okay:
        raise RuntimeError('Base checkpoint audit failed; inspect the report before any training')
    selected = {k:v for k,v in state.items() if k in target and not k.endswith('._extra_state')}
    result = model.load_state_dict(selected, strict=False)
    if sorted(k for k in result.missing_keys if not k.endswith('._extra_state')) != allowed or result.unexpected_keys:
        raise RuntimeError(f'Load result differed from key audit: {result}')
    return report


def build_model(activation_checkpointing=True):
    config = dict(NET_2B)
    config['sac_config'] = SACConfig(mode=CheckpointMode.NONE)
    # Construction on CPU retains exact FP32 master parameters for AdamW.
    model = SpatialCameraDIT(**config)
    primary = int(os.getenv('RANK','0')) == 0
    audit_and_load_base(model, prepare.ROOT/'download-state/h20-base-audit.json' if primary else None)
    frozen = freeze_for_plenoptic(model)
    if primary:
        prepare.save_json(prepare.ROOT/'download-state/h20-freeze-audit.json', frozen)
    if activation_checkpointing:
        model.enable_selective_checkpoint(SACConfig(mode=CheckpointMode.BLOCK_WISE), model.blocks)
    return model
