"""Audited public-base initialization and spatial CP for released Camera DiT.

Spatial CP keeps all temporal positions (81 frames -> 21 latents per view).
The pinned dense attention kernel is invariant to the rank-major spatial order.
Sparse attention and per-block absolute embeddings are intentionally rejected.
"""
import plenoptic_paths as layout
import json
import os
import torch
import torch.distributed as dist
from torch import nn
from einops import rearrange
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from paired_track_conditioner import PairedTrackConditioner

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


ADAPTATION_COMPONENTS = {
    'self_attn': lambda name: '.self_attn.' in name,
    'cam_encoder': lambda name: '.cam_encoder.' in name,
    'track': lambda name: name.startswith('track_conditioner.'),
}


class SpatialCameraDIT(CameraARMiniTrainDITwithConditionalMask):
    def __init__(self, *args, **kwargs):
        track_config = kwargs.pop('track_conditioner', None)
        if kwargs.get('extra_per_block_abs_pos_emb', False) or kwargs.get('n_dense_blocks', -1) != -1:
            raise ValueError('Spatial CP currently requires dense attention and no absolute per-block positions')
        super().__init__(*args, **kwargs)
        self.track_conditioner = None
        if track_config and track_config.get('enabled'):
            unknown = set(track_config) - {'enabled', 'max_tracks', 'hidden_channels',
                                            'temporal_layers', 'num_heads',
                                            'fourier_frequencies'}
            if unknown:
                raise ValueError('Unknown track-conditioner options: ' + str(sorted(unknown)))
            options = {k:v for k,v in track_config.items() if k != 'enabled'}
            self.track_conditioner = PairedTrackConditioner(
                out_channels=self.model_channels, **options)
        self._active_track_condition = None
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
            embedded, rope, extra = super().prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)
            if self._active_track_condition is not None:
                if self.track_conditioner is None:
                    raise ValueError('Track input was provided without a paired-track conditioner')
                embedded = embedded + self.track_conditioner(
                    embedded, self._active_track_condition).type_as(embedded)
            return embedded, rope, extra
        if self.concat_padding_mask:
            padding_mask = resize(padding_mask, list(x.shape[-2:]), interpolation=InterpolationMode.NEAREST)
            x = torch.cat([x, padding_mask.unsqueeze(1).repeat(1,1,x.shape[2],1,1)], dim=1)
        embedded = self.x_embedder(x)
        b, t, h, w, d = embedded.shape
        track_residual = None
        if self._active_track_condition is not None:
            if self.track_conditioner is None:
                raise ValueError('Track input was provided without a paired-track conditioner')
            track_residual = self.track_conditioner(embedded, self._active_track_condition)
        if w % self.spatial_cp_size:
            raise ValueError(f'Patch-grid width {w} must be divisible by CP={self.spatial_cp_size}')
        # Bypass upstream temporal CP expansion and generate exact global 3D RoPE.
        rope = self.pos_embedder.generate_embeddings(embedded.shape, fps=fps)
        rope = rearrange(rope, '(t h w) a b d -> t h w a b d', t=t,h=h,w=w)
        rope = rope.chunk(self.spatial_cp_size, dim=2)[self.spatial_cp_rank].contiguous()
        rope = rearrange(rope, 't h w a b d -> (t h w) a b d')
        local = embedded.chunk(self.spatial_cp_size, dim=3)[self.spatial_cp_rank].contiguous()
        if track_residual is not None:
            if track_residual.shape != embedded.shape:
                raise ValueError(f'Track grid {track_residual.shape} differs from DiT grid {embedded.shape}')
            local = local + track_residual.chunk(self.spatial_cp_size, dim=3)[self.spatial_cp_rank].contiguous().type_as(local)
        return local, rope, None

    def forward(self, *args, camera=None, crossattn_projected=False,
                track_condition=None, intermediate_feature_ids=None, **kwargs):
        if track_condition is not None and self.track_conditioner is None:
            raise ValueError('Track input requires an enabled track conditioner')
        self._active_track_condition = track_condition
        try:
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
                if intermediate_feature_ids:
                    # The released conditional-mask wrapper accepts **kwargs but
                    # discards them, including intermediate_feature_ids. Reapply
                    # its one-channel mask/timestep transformation here and call
                    # CameraARMiniTrainDIT.forward directly so the requested
                    # block features are actually returned for correspondence SFT.
                    if args or 'x_B_C_T_H_W' not in kwargs:
                        raise ValueError('Intermediate features require named model inputs')
                    condition = kwargs.pop('condition_video_input_mask_B_C_T_H_W', None)
                    if condition is None:
                        raise ValueError('Intermediate features require a conditioning mask')
                    value = kwargs.pop('x_B_C_T_H_W')
                    value = torch.cat([value, condition.type_as(value)], dim=1)
                    kwargs['timesteps_B_T'] = kwargs['timesteps_B_T'] * self.timestep_scale
                    return super(CameraARMiniTrainDITwithConditionalMask, self).forward(
                        x_B_C_T_H_W=value, camera=camera,
                        intermediate_feature_ids=intermediate_feature_ids, **kwargs)
                return super().forward(*args, camera=camera, **kwargs)
            finally:
                self.use_crossattn_projection = previous
        finally:
            self._active_track_condition = None


def trainable_name(name):
    # Do not accidentally train cross-attention or all AdaLN modules by substring.
    return ADAPTATION_COMPONENTS['self_attn'](name) or ADAPTATION_COMPONENTS['cam_encoder'](name)


def adaptation_name(name):
    return any(match(name) for match in ADAPTATION_COMPONENTS.values())


def component_for_name(name):
    matched = [component for component, match in ADAPTATION_COMPONENTS.items() if match(name)]
    if len(matched) > 1:
        raise RuntimeError('Adaptation parameter matched multiple component rules: ' + name)
    return matched[0] if matched else None


def freeze_for_plenoptic(model, components=('self_attn', 'cam_encoder')):
    components = tuple(components)
    if not components or len(set(components)) != len(components) or set(components) - set(ADAPTATION_COMPONENTS):
        raise ValueError('Unknown or duplicate trainable model component')
    if 'track' in components and model.track_conditioner is None:
        raise ValueError('The track component was selected without constructing the module')
    model.requires_grad_(False)
    names = []
    for name, parameter in model.named_parameters():
        if component_for_name(name) in components:
            parameter.requires_grad_(True)
            names.append(name)
    missing = sorted(component for component in components
                     if not any(component_for_name(name) == component for name in names))
    if not names or missing:
        raise RuntimeError('No parameters selected for components: ' + str(missing))
    return dict(components=list(components), names=names,
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
    path = layout.rooted(prepare.JOBS['base'][3]) / prepare.JOBS['base'][4][0]
    state, selection = checkpoint_state(path)
    target = model.state_dict()
    source_shapes = {k:list(v.shape) for k,v in state.items() if isinstance(v, torch.Tensor) and not k.endswith('._extra_state')}
    target_shapes = {k:list(v.shape) for k,v in target.items() if isinstance(v, torch.Tensor) and not k.endswith('._extra_state')}
    missing = sorted(target_shapes.keys() - source_shapes.keys())
    allowed = sorted(n for n in target_shapes if '.cam_encoder.' in n or n.startswith('track_conditioner.'))
    unexpected = sorted(source_shapes.keys() - target_shapes.keys())
    mismatch = {k:[source_shapes[k],target_shapes[k]] for k in target_shapes.keys() & source_shapes.keys()
                if source_shapes[k] != target_shapes[k]}
    report = dict(status='running', checkpoint=str(layout.relative(path)),
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


def build_model(activation_checkpointing=True, track_config=None,
                trainable_components=('self_attn', 'cam_encoder')):
    config = dict(NET_2B)
    config['sac_config'] = SACConfig(mode=CheckpointMode.NONE)
    if track_config:
        config['track_conditioner'] = {key:value for key, value in track_config.items()
                                       if key in {'enabled', 'max_tracks', 'hidden_channels',
                                                  'temporal_layers', 'num_heads',
                                                  'fourier_frequencies'}}
    # Construction on CPU retains exact FP32 master parameters for AdamW.
    model = SpatialCameraDIT(**config)
    primary = int(os.getenv('RANK','0')) == 0
    audit_and_load_base(model, layout.rooted('download-state/h20-base-audit.json') if primary else None)
    frozen = freeze_for_plenoptic(model, trainable_components)
    if primary:
        prepare.save_json(layout.rooted('download-state/h20-freeze-audit.json'), frozen)
    if activation_checkpointing:
        model.enable_selective_checkpoint(SACConfig(mode=CheckpointMode.BLOCK_WISE), model.blocks)
    return model
