"""Lightweight dual-view conditioner for sparse paired hand/object tracks."""
import math

import torch
from einops import rearrange
from torch import nn


def _fourier(values, frequencies):
    scales = (2. ** torch.arange(frequencies, device=values.device,
                                 dtype=values.dtype)) * math.pi
    phase = values.unsqueeze(-1) * scales
    return torch.cat((values.unsqueeze(-1), phase.sin(), phase.cos()), dim=-1).flatten(-2)


def _bilinear_sample(grid, coords):
    """Sample B,L,H,W,D at normalized B,N,L,(y,x) coordinates."""
    batch, length, height, width, channels = grid.shape
    if coords.shape[0] != batch or coords.shape[2] != length or coords.shape[-1] != 2:
        raise ValueError('Paired source coordinates do not match the source token grid')
    points = coords.clamp(0, 1).permute(0, 2, 1, 3)
    y = points[..., 0] * (height - 1)
    x = points[..., 1] * (width - 1)
    y0, x0 = y.floor().long(), x.floor().long()
    y1, x1 = (y0 + 1).clamp(max=height - 1), (x0 + 1).clamp(max=width - 1)
    wy, wx = y - y0, x - x0
    flat = grid.reshape(batch, length, height * width, channels)

    def gather(yy, xx):
        index = (yy * width + xx).unsqueeze(-1).expand(-1, -1, -1, channels)
        return torch.gather(flat, 2, index)

    sampled = (gather(y0, x0) * ((1 - wy) * (1 - wx)).unsqueeze(-1)
               + gather(y0, x1) * ((1 - wy) * wx).unsqueeze(-1)
               + gather(y1, x0) * (wy * (1 - wx)).unsqueeze(-1)
               + gather(y1, x1) * (wy * wx).unsqueeze(-1))
    return sampled.permute(0, 2, 1, 3)


def _bilinear_scatter(features, coords, confidence, height, width):
    """Scatter B,N,L,D to B,L,H,W,D and confidence-normalize overlaps."""
    batch, points, length, channels = features.shape
    if coords.shape != (batch, points, length, 2) or confidence.shape != (batch, points, length):
        raise ValueError('Paired tracks and sparse features have different shapes')
    positions = coords.clamp(0, 1)
    y = positions[..., 0] * (height - 1)
    x = positions[..., 1] * (width - 1)
    y0, x0 = y.floor().long(), x.floor().long()
    y1, x1 = (y0 + 1).clamp(max=height - 1), (x0 + 1).clamp(max=width - 1)
    wy, wx = y - y0, x - x0
    dense = features.new_zeros(batch, length, height * width, channels)
    weights = features.new_zeros(batch, length, height * width, 1)
    values = features.permute(0, 2, 1, 3)
    valid = confidence.permute(0, 2, 1).to(features.dtype)

    for yy, xx, bilinear in ((y0, x0, (1 - wy) * (1 - wx)),
                             (y0, x1, (1 - wy) * wx),
                             (y1, x0, wy * (1 - wx)),
                             (y1, x1, wy * wx)):
        weight = (bilinear * confidence).permute(0, 2, 1).to(features.dtype) * valid.gt(0)
        index = (yy * width + xx).permute(0, 2, 1)
        dense.scatter_add_(2, index.unsqueeze(-1).expand(-1, -1, -1, channels),
                           values * weight.unsqueeze(-1))
        weights.scatter_add_(2, index.unsqueeze(-1), weight.unsqueeze(-1))
    dense = dense / weights.clamp_min(1e-6)
    return dense.reshape(batch, length, height, width, channels)


class PairedTrackConditioner(nn.Module):
    """Transfer source-token context along paired source/target 3-D tracks."""
    VERSION = 2

    def __init__(self, max_tracks=128, hidden_channels=256, temporal_layers=2,
                 num_heads=8, fourier_frequencies=6, out_channels=2048):
        super().__init__()
        if (max_tracks < 1 or hidden_channels < 32 or temporal_layers < 1
                or num_heads < 1 or hidden_channels % num_heads
                or fourier_frequencies < 1):
            raise ValueError('Invalid paired-track conditioner dimensions')
        self.max_tracks = int(max_tracks)
        self.hidden_channels = int(hidden_channels)
        self.fourier_frequencies = int(fourier_frequencies)
        # Per observation: y/x position, normalized inverse depth, confidence,
        # y/x velocity, inverse-depth velocity, and normalized time.
        metadata_channels = 8 * (1 + 2 * fourier_frequencies)
        self.source_projection = nn.Linear(out_channels, hidden_channels)
        self.metadata_projection = nn.Sequential(
            nn.Linear(metadata_channels, hidden_channels), nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels))
        self.entity_embedding = nn.Embedding(4, hidden_channels)
        self.instance_embedding = nn.Embedding(max_tracks, hidden_channels)
        self.point_embedding = nn.Embedding(256, hidden_channels)
        layer = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=num_heads,
            dim_feedforward=2 * hidden_channels, dropout=0., activation='gelu',
            batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_layers,
                                              enable_nested_tensor=False)
        self.temporal_norm = nn.LayerNorm(hidden_channels)
        self.zero_out = nn.Conv3d(hidden_channels, out_channels, 1)
        nn.init.zeros_(self.zero_out.weight)
        nn.init.zeros_(self.zero_out.bias)

    @staticmethod
    def _normalized_inverse_depth(depth, confidence):
        valid = confidence > 0
        inverse = torch.where(valid, depth.clamp_min(1e-4).reciprocal(),
                              torch.zeros_like(depth))
        weight = confidence.clamp_min(0)
        denominator = weight.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        mean = (inverse * weight).sum(dim=(1, 2, 3), keepdim=True) / denominator
        variance = ((inverse - mean).square() * weight).sum(
            dim=(1, 2, 3), keepdim=True) / denominator
        scale = variance.sqrt().clamp_min(1e-2)
        return torch.where(valid, ((inverse - mean) / scale).clamp(-5, 5),
                           torch.zeros_like(inverse))

    def _metadata(self, coords, inverse_depth, confidence,
                  coordinate_velocity, depth_velocity, time):
        values = torch.cat((coords.mul(2).sub(1), inverse_depth.unsqueeze(-1),
                            confidence.unsqueeze(-1), coordinate_velocity.mul(2),
                            depth_velocity.unsqueeze(-1), time), dim=-1)
        return self.metadata_projection(_fourier(values, self.fourier_frequencies))

    def forward(self, embedded, tracks):
        if embedded.ndim != 5:
            raise ValueError('Embedded video tokens must be B,T,H,W,D')
        required = {'coords', 'depth', 'confidence', 'entity', 'instance', 'point'}
        if not isinstance(tracks, dict) or set(tracks) != required:
            raise ValueError('Paired-track condition is missing geometry or stable identity')
        coords, depth = tracks['coords'], tracks['depth']
        confidence, entity = tracks['confidence'], tracks['entity']
        instance, point = tracks['instance'], tracks['point']
        batch, total_t, height, width, channels = embedded.shape
        if (coords.shape[:3] != (batch, self.max_tracks, 2)
                or coords.shape[-1] != 2 or depth.shape != coords.shape[:-1]
                or confidence.shape != depth.shape
                or entity.shape != (batch, self.max_tracks)
                or instance.shape != entity.shape or point.shape != entity.shape
                or total_t != 2 * coords.shape[3]):
            raise ValueError('Paired tracks do not match the dual-view token sequence')
        length = coords.shape[3]
        confidence = confidence.clamp(0, 1)
        inverse_depth = self._normalized_inverse_depth(depth, confidence)
        coordinate_velocity = torch.zeros_like(coords)
        depth_velocity = torch.zeros_like(inverse_depth)
        velocity_valid = ((confidence[:, :, :, 1:] > 0)
                          & (confidence[:, :, :, :-1] > 0))
        coordinate_velocity[:, :, :, 1:] = (
            coords[:, :, :, 1:] - coords[:, :, :, :-1]
        ) * velocity_valid.unsqueeze(-1).to(coords.dtype)
        depth_velocity[:, :, :, 1:] = (
            inverse_depth[:, :, :, 1:] - inverse_depth[:, :, :, :-1]
        ) * velocity_valid.to(inverse_depth.dtype)
        depth_velocity.clamp_(-5, 5)
        time = torch.linspace(-1, 1, length, device=coords.device,
                              dtype=coords.dtype).view(1, 1, length, 1)
        time = time.expand(batch, self.max_tracks, -1, -1)

        # The project stores target tokens first and clean source tokens second.
        sampled = _bilinear_sample(embedded[:, length:], coords[:, :, 0])
        hidden = (self.source_projection(sampled)
                  + self._metadata(coords[:, :, 0], inverse_depth[:, :, 0],
                                   confidence[:, :, 0], coordinate_velocity[:, :, 0],
                                   depth_velocity[:, :, 0], time)
                  + self.entity_embedding(entity.clamp(0, 3))[:, :, None]
                  + self.instance_embedding(
                      instance.clamp(0, self.max_tracks - 1))[:, :, None]
                  + self.point_embedding(point.clamp(0, 255))[:, :, None])
        source_valid = confidence[:, :, 0] > 0
        active = source_valid.any(dim=-1)
        hidden = hidden * source_valid.unsqueeze(-1).to(hidden.dtype)
        padding = ~source_valid.reshape(batch * self.max_tracks, length)
        # Transformer attention cannot accept an all-masked row. Such padded
        # tracks are zeroed again after temporal aggregation.
        padding = padding.clone()
        padding[~active.reshape(-1), 0] = False
        hidden = self.temporal(hidden.reshape(batch * self.max_tracks, length, -1),
                               src_key_padding_mask=padding)
        hidden = self.temporal_norm(hidden).reshape(batch, self.max_tracks, length, -1)
        hidden = hidden * active[:, :, None, None].to(hidden.dtype)

        dense = []
        # Cache view order is source,target; model time order is target,source.
        for view in (1, 0):
            view_features = hidden + self._metadata(
                coords[:, :, view], inverse_depth[:, :, view], confidence[:, :, view],
                coordinate_velocity[:, :, view], depth_velocity[:, :, view], time)
            dense.append(_bilinear_scatter(view_features, coords[:, :, view],
                                           confidence[:, :, view], height, width))
        grid = torch.cat(dense, dim=1)
        residual = self.zero_out(rearrange(grid, 'b t h w d -> b d t h w'))
        return rearrange(residual, 'b d t h w -> b t h w d')
