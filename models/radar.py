"""Radar tensor feature field and spatial sampling."""

import torch
from torch import nn
from torch.nn import functional as F


def radar_indices(rpc_xyz):
    return (
        rpc_xyz + rpc_xyz.new_tensor([0.0, 0.0, 1.5]) - rpc_xyz.new_tensor([-3.0, 0.5, 0.0])
    ) / rpc_xyz.new_tensor([0.05, 0.05, 0.1])


class RadarField(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.ModuleList([nn.Conv3d(a, b, 3, 2, 1) for a, b in [(4, 64), (64, 128), (128, 256)]])
        self.projections = nn.ModuleList([nn.Conv3d(c, 128, 1) for c in [64, 128, 256]])

    def forward(self, crop):
        if tuple(crop.shape[1:]) != (4, 64, 64, 32):
            raise ValueError("expected a B x 4 x 64 x 64 x 32 radar crop")
        values = []
        x = crop
        for i, layer in enumerate(self.trunk):
            x = F.relu(layer(x))
            y = self.projections[i](x)
            if i:
                axes = [torch.arange(n, device=x.device, dtype=x.dtype) / 2**i for n in (32, 32, 16)]
                grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)
                grid = (2 * grid / (grid.new_tensor(y.shape[2:]) - 1) - 1).flip(-1)
                y = F.grid_sample(
                    y, grid[None].expand(len(y), -1, -1, -1, -1), padding_mode="border", align_corners=True
                )
            values.append(y)
        return torch.stack(values).sum(0)


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 128))
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.norm1, self.norm2 = (nn.LayerNorm(128), nn.LayerNorm(128))
        self.ffn = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))

    def forward(self, field, xyz, relative, feature, origin):
        native = radar_indices(xyz)
        coord = native - origin[:, None, None]
        valid = ((native >= 0) & (native <= native.new_tensor([120, 110, 30]))).all(-1)
        valid &= ((coord >= 0) & (coord <= coord.new_tensor([62, 62, 30]))).all(-1)
        grid = (coord / coord.new_tensor([31, 31, 15]) - 1).flip(-1)
        sampled = F.grid_sample(field, grid[:, :, :, None], align_corners=True)
        sampled = sampled.squeeze(-1).permute(0, 2, 3, 1)
        memory = (sampled + self.position(relative)).flatten(0, 1)
        mask = ~valid.flatten(0, 1)
        empty = mask.all(-1)
        mask = mask.clone()
        mask[:, 0] &= ~empty
        query = self.norm1(feature).flatten(0, 1)[:, None]
        evidence = self.attention(query, memory, memory, key_padding_mask=mask, need_weights=False)[0][:, 0]
        evidence = evidence + self.ffn(self.norm2(evidence))
        evidence = torch.where(empty[:, None], torch.zeros_like(evidence), evidence)
        return (evidence.reshape_as(feature), valid)


def sample_field(field, xyz, origin):
    native = radar_indices(xyz)
    shape = (len(origin),) + (1,) * (xyz.ndim - 2) + (3,)
    coord = native - origin.reshape(shape)
    valid = ((native >= 0) & (native <= native.new_tensor([120, 110, 30]))).all(-1)
    valid &= ((coord >= 0) & (coord <= coord.new_tensor([62, 62, 30]))).all(-1)
    grid = (coord / coord.new_tensor([31, 31, 15]) - 1).flip(-1)
    sampled = F.grid_sample(field, grid.reshape(len(field), -1, 1, 1, 3), align_corners=True)
    sampled = sampled[:, :, :, 0, 0].transpose(1, 2).reshape(*xyz.shape[:-1], field.shape[1])
    return (sampled, valid)
