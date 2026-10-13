"""Body-conditioned global, part and surface observations."""

import torch
from torch import nn

from models.geometry import JOINT_NAMES, PARENTS, geometry_vector, mlp
from models.radar import Reader, sample_field


def relations():
    flags = torch.zeros(44, 44, 6)
    adjacency = torch.zeros(43, 43, dtype=torch.bool)
    for child, parent in enumerate(PARENTS[1:], 1):
        segment = 21 + child
        for x, y in ((child, parent), (segment, child), (segment, parent)):
            adjacency[x, y] = adjacency[y, x] = True
    edges = [{p, c} for c, p in enumerate(PARENTS[1:], 1)]
    for i in range(21):
        for j in range(i + 1, 21):
            if edges[i] & edges[j]:
                adjacency[22 + i, 22 + j] = adjacency[22 + j, 22 + i] = True
    flags[:43, :43, 1] = adjacency
    two = (adjacency.float() @ adjacency.float() > 0) & ~adjacency
    two.fill_diagonal_(False)
    flags[:43, :43, 2] = two
    for i, name in enumerate(JOINT_NAMES):
        if name.startswith("left_"):
            j = JOINT_NAMES.index(name.replace("left_", "right_"))
            flags[i, j, 3] = flags[j, i, 3] = 1
            flags[21 + i, 21 + j, 3] = flags[21 + j, 21 + i, 3] = 1
    flags[43, :, 4] = flags[:, 43, 4] = 1
    flags[torch.arange(44), torch.arange(44), 0] = 1
    flags[..., 5] = flags[..., :5].sum(-1) == 0
    return flags


class BodyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = (nn.LayerNorm(128), nn.LayerNorm(128))
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128))
        self.bias = nn.Parameter(torch.zeros(4, 6))
        self.register_buffer("relations", relations())

    def forward(self, x):
        y = self.norm1(x)
        mask = torch.einsum("ijr,hr->hij", self.relations, self.bias).repeat(len(x), 1, 1)
        x = x + self.attention(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class AnchorReader(Reader):
    def forward(self, field, xyz, relative, feature, origin):
        sampled, valid = sample_field(field, xyz, origin)
        memory = (sampled + self.position(relative)).reshape(-1, 7, 128)
        mask = ~valid.reshape(-1, 7)
        empty = mask.all(-1)
        mask = mask.clone()
        mask[:, 0] &= ~empty
        query = self.norm1(feature).reshape(-1, 1, 128)
        evidence = self.attention(query, memory, memory, key_padding_mask=mask, need_weights=False)[0][:, 0]
        evidence = evidence + self.ffn(self.norm2(evidence))
        evidence = torch.where(empty[:, None], torch.zeros_like(evidence), evidence)
        return (evidence.reshape_as(feature), valid)


class Observation(nn.Module):
    def __init__(self):
        super().__init__()
        self.joint_reader = Reader()
        self.surface_reader = AnchorReader()
        self.joint_geometry = mlp(3, 128)
        self.part_geometry = mlp(7, 128)
        self.anchor_geometry = mlp(6, 128)
        self.joint_fusion = nn.Sequential(nn.Linear(512, 128), nn.LayerNorm(128))
        self.part_fusion = nn.Sequential(nn.Linear(512, 128), nn.LayerNorm(128))
        self.global_fusion = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128))
        self.surface_fusion = nn.Sequential(nn.Linear(384, 128), nn.LayerNorm(128))
        self.reasoning = nn.Sequential(BodyBlock(), BodyBlock())
        self.register_buffer(
            "joint_offsets",
            torch.stack(torch.meshgrid(*[torch.linspace(-0.25, 0.25, 3)] * 3, indexing="ij"), -1).reshape(
                27, 3
            ),
        )
        self.register_buffer(
            "surface_offsets", torch.cat((torch.zeros(1, 3), torch.eye(3) * 0.1, -torch.eye(3) * 0.1))
        )
        self.anchor_id = nn.Parameter(torch.randn(1, 22, 8, 128) * 0.02)

    def forward(self, g, h, field, origin):
        jx = g["joints"][:, :, None] + self.joint_offsets
        sx = (
            g["anchors"][:, :, :, None]
            + torch.einsum("bjcd,od->bjoc", g["orientation"], self.surface_offsets)[:, :, None]
        )
        relative = torch.einsum("bjdc,bjnod->bjnoc", g["orientation"], sx - g["anchors"][:, :, :, None])
        local_anchor = torch.einsum(
            "bjdc,bjnd->bjnc", g["orientation"], g["anchors"] - g["center"][:, :, None]
        )
        ag = self.anchor_geometry(torch.cat((g["anchors"], local_anchor), -1))
        aq = h[:, :22, None] + ag + self.anchor_id
        je, _ = self.joint_reader(field, jx, self.joint_offsets.expand_as(jx), h[:, :22], origin)
        se, _ = self.surface_reader(field, sx, relative, aq, origin)
        return self.fuse(g, h, je, se, ag, field.mean((2, 3, 4)))

    def fuse(self, g, h, je, se, ag, global_memory):
        region = se.mean(2)
        segment_e = (region[:, 1:] + region[:, list(PARENTS[1:])]) * 0.5
        segment_j = (je[:, 1:] + je[:, list(PARENTS[1:])]) * 0.5
        joint = h[:, :22] + self.joint_fusion(
            torch.cat((h[:, :22], je, region, self.joint_geometry(g["joints"])), -1)
        )
        part = h[:, 22:43] + self.part_fusion(
            torch.cat(
                (h[:, 22:43], segment_e, segment_j, self.part_geometry(geometry_vector(g["joints"]))), -1
            )
        )
        glob = h[:, -1] + self.global_fusion(torch.cat((h[:, -1], global_memory), -1))
        hbar = self.reasoning(torch.cat((joint, part, glob[:, None]), 1))
        surface = se + self.surface_fusion(
            torch.cat((se, ag + self.anchor_id, hbar[:, :22, None].expand_as(se)), -1)
        )
        evidence = dict(global_=hbar[:, -1], part=hbar[:, :43], surface=surface)
        evidence["global"] = evidence.pop("global_")
        return (evidence, hbar)


class SparseReader(nn.Module):
    """Attend to the 16 nearest RPC tokens in the current body frame."""

    def __init__(self):
        super().__init__()
        self.position = mlp(3, 128)
        self.query = nn.Linear(128, 128)
        self.key = nn.Linear(128, 128)
        self.value = nn.Linear(128, 128)
        self.norm = nn.LayerNorm(128)
        self.ffn = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))

    def forward(self, xyz, feature, rotation, memory, coordinates):
        distance = torch.cdist(xyz, coordinates)
        indices = distance.topk(min(16, coordinates.shape[1]), largest=False).indices
        batch = torch.arange(len(xyz), device=xyz.device)[:, None, None]
        neighbor = coordinates[batch, indices]
        relative = torch.einsum("bqdc,bqnd->bqnc", rotation, neighbor - xyz[:, :, None])
        tokens = memory[batch, indices] + self.position(relative)
        logits = (self.query(feature)[:, :, None] * self.key(tokens)).sum(-1) / (128**0.5)
        result = (logits.softmax(-1)[:, :, :, None] * self.value(tokens)).sum(2)
        return result + self.ffn(self.norm(result))


class RPCObservation(Observation):
    def __init__(self):
        super().__init__()
        self.joint_reader = SparseReader()
        self.surface_reader = SparseReader()

    def forward(self, g, h, memory, coordinates):
        joints, anchors = g["joints"], g["anchors"]
        local_anchor = torch.einsum("bjdc,bjnd->bjnc", g["orientation"], anchors - g["center"][:, :, None])
        ag = self.anchor_geometry(torch.cat((anchors, local_anchor), -1))
        aq = h[:, :22, None] + ag + self.anchor_id
        je = self.joint_reader(joints, h[:, :22], g["orientation"], memory, coordinates)
        se = self.surface_reader(
            anchors.flatten(1, 2),
            aq.flatten(1, 2),
            g["orientation"][:, :, None].expand(-1, -1, 8, -1, -1).flatten(1, 2),
            memory,
            coordinates,
        ).reshape(len(joints), 22, 8, 128)
        return self.fuse(g, h, je, se, ag, memory.mean(1))
