"""Prepared point-cloud convolution, preserving native point-query order."""

import importlib
import math

import torch
from torch import nn

from models.geometry import (
    PARENTS,
    RestSkeleton,
    State,
    geometry_vector,
    mlp,
    named_init,
    solve_kinematics,
)


def spatial_add_max(feature, displacement):
    return (feature + displacement).max(-1).values


class PreparedP4DConv(nn.Module):
    """Four-frame point convolution with the native FPS and radius-query order."""

    def __init__(self, reference, compiled_pool=False):
        super().__init__()
        self.r, self.k = (reference.r, reference.k)
        self.spatial_stride = reference.spatial_stride
        for name, module in reference.named_children():
            self.add_module(name, module)
        self.pool = (
            torch.compile(spatial_add_max, dynamic=False, fullgraph=True)
            if compiled_pool
            else spatial_add_max
        )

    def forward(self, xyzs, features):
        ops = importlib.import_module("pointnet2_utils")
        frames = [x.contiguous() for x in xyzs.unbind(1)]
        feats = [x.contiguous() for x in features.unbind(1)]
        frames, feats = ([frames[0]] + frames, [feats[0]] + feats)
        flipped = [x.transpose(1, 2).contiguous() for x in frames]
        anchors, outputs = ([], [])
        for t in range(1, len(frames) - 1):
            index = ops.furthest_point_sample(frames[t], xyzs.shape[2] // self.spatial_stride)
            anchor_flipped = ops.gather_operation(flipped[t], index)
            anchor = anchor_flipped.transpose(1, 2).contiguous()
            neighbors = []
            for i in range(t - 1, t + 2):
                index = ops.ball_query(self.r, self.k, frames[i], anchor)
                grouped = ops.grouping_operation(flipped[i], index)
                delta = grouped - anchor_flipped.unsqueeze(3)
                dt = torch.full(
                    (delta.shape[0], 1, delta.shape[2], delta.shape[3]),
                    float(i - t),
                    device=delta.device,
                    dtype=torch.float32,
                )
                displacement = self.conv_d(torch.cat((delta, dt), 1))
                feature = self.conv_f(ops.grouping_operation(feats[i], index))
                neighbors.append(self.pool(feature, displacement))
            outputs.append(torch.stack(neighbors, 1).max(1).values)
            anchors.append(anchor)
        return (torch.stack(anchors, 1), torch.stack(outputs, 1))


class LocalObservationEncoder(nn.Module):
    """Local point features with spatial and temporal position embeddings."""

    def __init__(self, loaded_encoder):
        super().__init__()
        self.tube_embedding = loaded_encoder.tube_embedding
        self.pos_embedding = loaded_encoder.pos_embedding
        self.emb_relu = loaded_encoder.emb_relu

    def forward(self, radar):
        xyz, features = self.tube_embedding(radar[..., :3], radar[..., 3:].permute(0, 1, 3, 2))
        times = torch.arange(1, xyz.shape[1] + 1, device=xyz.device, dtype=torch.float32)
        times = times[None, :, None, None].expand(*xyz.shape[:-1], 1)
        xyzt = torch.cat((xyz, times), -1).flatten(1, 2)
        h = features.permute(0, 1, 3, 2).flatten(1, 2)
        h = h + self.pos_embedding(xyzt.transpose(1, 2)).transpose(1, 2)
        if self.emb_relu:
            h = self.emb_relu(h)
        return h, xyz.flatten(1, 2)


class Initializer(nn.Module):
    def __init__(self, encoder, geometry, seed=42):
        super().__init__()
        self.rpc_encoder = LocalObservationEncoder(encoder)
        self.memory_projection = nn.Sequential(nn.Linear(1024, 128), nn.LayerNorm(128))
        self.queries = nn.Parameter(torch.randn(1, 23, 128) * 0.02)
        layer = nn.TransformerDecoderLayer(
            128, 4, 512, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.semantic_decoder = nn.TransformerDecoder(layer, 2, norm=nn.LayerNorm(128))
        self.proposal_head = mlp(128, 3)
        self.part_formation = mlp(263, 128)
        self.joint_position = mlp(3, 128)
        self.incident = nn.Linear(128, 128)
        incidence = torch.zeros(22, 21)
        for c, p in enumerate(PARENTS[1:], 1):
            incidence[p, c - 1] = incidence[c, c - 1] = 1
        self.register_buffer("incidence", incidence / incidence.sum(-1, keepdim=True))
        self.beta_head = mlp(384, 10)
        self.gender_head = mlp(128, 1)
        self.twist_head = mlp(128, 1)
        self.terminal_head = mlp(128, 6)
        self.rest = RestSkeleton(geometry)
        for name, module in self.named_children():
            if name not in ("rpc_encoder", "rest"):
                named_init(module, "initializer." + name, seed=seed)
        nn.init.zeros_(self.twist_head[-1].weight)
        nn.init.zeros_(self.twist_head[-1].bias)
        nn.init.zeros_(self.terminal_head[-1].weight)
        with torch.no_grad():
            self.terminal_head[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, rpc):
        tokens, coordinates = self.rpc_encoder(rpc)
        if tokens.shape[1:] != (93, 1024):
            raise ValueError(f"expected 93 point tokens with 1024 features, got {tokens.shape}")
        memory = self.memory_projection(tokens)
        decoded = self.semantic_decoder(self.queries.expand(len(rpc), -1, -1), memory)
        joint, glob = (decoded[:, :22], decoded[:, 22:])
        a = self.proposal_head(joint)
        part = self.part_formation(
            torch.cat((joint[:, list(PARENTS[1:])], joint[:, 1:], geometry_vector(a)), -1)
        )
        h = torch.cat((joint, part, glob), 1)
        context = (
            joint + self.joint_position(a) + self.incident(torch.einsum("jp,bpd->bjd", self.incidence, part))
        )
        beta = self.beta_head(torch.cat((glob[:, 0], context.mean(1), part.mean(1)), -1))
        gender = self.gender_head(glob[:, 0]).sigmoid()
        rest = self.rest(beta, gender)
        rotation = solve_kinematics(
            rest, a, math.pi * self.twist_head(context).tanh(), self.terminal_head(context)
        )
        z = State(beta, a[:, 0] - rest[:, 0], rotation, gender)
        return z, h, a, memory, coordinates
