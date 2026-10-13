"""Body skeleton, rotation maps and geometric initialization."""

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 128), nn.GELU(), nn.Linear(128, outputs))


def named_init(module, prefix, seed=42):
    for name, p in module.named_parameters():
        name = prefix + "." + name
        parameter_seed = (
            int.from_bytes(hashlib.sha256((str(seed) + ":" + name).encode()).digest()[:8], "little") % 2**63
        )
        g = torch.Generator().manual_seed(parameter_seed)
        if "bias" in name:
            nn.init.zeros_(p)
        elif p.ndim < 2:
            nn.init.ones_(p)
        else:
            nn.init.xavier_uniform_(p, generator=g)


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-06)


def skew(v):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)


def axial(axis, angle):
    k = skew(unit(axis))
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype)
    return eye + angle.sin()[..., None] * k + (1 - angle.cos())[..., None] * (k @ k)


def swing(source, target):
    a, b = (unit(source), unit(target))
    cross = torch.cross(a, b, dim=-1)
    cosine = (a * b).sum(-1, keepdim=True).clamp(-1, 1)
    eye = torch.eye(3, device=a.device, dtype=a.dtype).expand(*a.shape[:-1], 3, 3)
    q = torch.cat((1 + cosine, cross), -1)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-08)
    w, v = (q[..., :1], q[..., 1:])
    regular = (
        (w.square() - v.square().sum(-1, keepdim=True))[..., None] * eye
        + 2 * v[..., :, None] * v[..., None, :]
        + 2 * w[..., None] * skew(v)
    )
    basis = F.one_hot(a.abs().argmin(-1), 3).to(a.dtype)
    perpendicular = unit(torch.cross(a, basis, dim=-1))
    opposite = 2 * perpendicular[..., :, None] * perpendicular[..., None, :] - eye
    result = torch.where((cosine < -1 + 1e-05)[..., None], opposite, regular)
    valid = (source.norm(dim=-1) > 1e-06) & (target.norm(dim=-1) > 1e-06)
    return torch.where(valid[..., None, None], result, eye)


def frame(primary, secondary):
    first = unit(primary)
    residual = secondary - (secondary * first).sum(-1, keepdim=True) * first
    second = unit(residual)
    third = torch.cross(first, second, dim=-1)
    valid = (primary.norm(dim=-1) > 1e-06) & (residual.norm(dim=-1) > 1e-06)
    return (torch.stack((first, second, third), -1), valid)


def matrix_to_axis_angle(matrix):
    """Largest quaternion component, canonical hemisphere; finite branch guards."""
    m = matrix
    qabs = (
        torch.stack(
            (
                1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2],
                1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
                1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
                1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
            ),
            -1,
        )
        .clamp_min(1e-12)
        .sqrt()
    )
    candidates = torch.stack(
        (
            torch.stack(
                (
                    qabs[..., 0].square(),
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 2, 1] - m[..., 1, 2],
                    qabs[..., 1].square(),
                    m[..., 1, 0] + m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] + m[..., 0, 1],
                    qabs[..., 2].square(),
                    m[..., 2, 1] + m[..., 1, 2],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 2, 0] + m[..., 0, 2],
                    m[..., 2, 1] + m[..., 1, 2],
                    qabs[..., 3].square(),
                ),
                -1,
            ),
        ),
        -2,
    )
    candidates = candidates / (2 * qabs[..., :, None]).clamp_min(1e-06)
    q = (candidates * F.one_hot(qabs.argmax(-1), 4)[..., :, None]).sum(-2)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-06)
    q = torch.where(q[..., :1] < 0, -q, q)
    length = q[..., 1:].norm(dim=-1, keepdim=True)
    factor = 2 * torch.atan2(length, q[..., :1]) / length.clamp_min(1e-08)
    factor = torch.where(length > 1e-08, factor, torch.full_like(factor, 2.0))
    return q[..., 1:] * factor


def solve_kinematics(rest, anchors, twist, terminal):
    """Column-vector rotations; world target directions moved into parent frame."""
    local, world = ([], [])
    for joint, parent in enumerate(PARENTS):
        children = [c for c in range(1, 22) if PARENTS[c] == joint]
        parent_rotation = (
            torch.eye(3, device=rest.device, dtype=rest.dtype).expand(len(rest), 3, 3)
            if parent < 0
            else world[parent]
        )
        if not children:
            q, valid = frame(terminal[:, joint, :3], terminal[:, joint, 3:])
            rotation = torch.where(
                valid[:, None, None], q, torch.eye(3, device=rest.device, dtype=rest.dtype)
            )
        else:
            child = 3 if joint == 0 else 12 if joint == 9 else children[0]
            source = rest[:, child] - rest[:, joint]
            target = (parent_rotation.transpose(-1, -2) @ (anchors[:, child] - anchors[:, joint])[..., None])[
                ..., 0
            ]
            rotation = swing(source, target) @ axial(source, twist[:, joint])
            if joint in (0, 9):
                left, right = (1, 2) if joint == 0 else (13, 14)
                reference, rvalid = frame(source, rest[:, right] - rest[:, left])
                lateral = (
                    parent_rotation.transpose(-1, -2) @ (anchors[:, right] - anchors[:, left])[..., None]
                )[..., 0]
                observed, ovalid = frame(target, lateral)
                rotation = torch.where(
                    (rvalid & ovalid)[:, None, None], observed @ reference.transpose(-1, -2), rotation
                )
        local.append(rotation)
        world.append(parent_rotation @ rotation)
    return torch.stack(local, 1)


class RestSkeleton(nn.Module):
    def __init__(self, geometry):
        super().__init__()
        for name in ("male", "female"):
            layer = getattr(geometry, "smplx_layer_" + name)
            regressor = layer.J_regressor[:22]
            self.register_buffer(name + "_base", regressor @ layer.v_template)
            self.register_buffer(
                name + "_shape", torch.einsum("jv,vck->jck", regressor, layer.shapedirs[..., :10])
            )

    def forward(self, beta, gender):
        male = self.male_base + torch.einsum("bk,jck->bjc", beta, self.male_shape)
        female = self.female_base + torch.einsum("bk,jck->bjc", beta, self.female_shape)
        return torch.where((gender > 0.5)[:, :, None], male, female)


def part_geometry(a):
    delta = a[:, 1:] - a[:, list(PARENTS[1:])]
    length = delta.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(delta)
    fallback[..., 0] = 1
    u = torch.where(length > 1e-06, delta / length.clamp_min(1e-06), fallback)
    mu = (a[:, 1:] + a[:, list(PARENTS[1:])]) / 2
    return (mu, u, length)


def geometry_vector(a):
    mu, u, length = part_geometry(a)
    return torch.cat((mu, u, torch.log(length + 1e-06)), -1)


@dataclass
class State:
    beta: torch.Tensor
    translation: torch.Tensor
    rotation: torch.Tensor
    gender: torch.Tensor

    def detach(self):
        return State(*(x.detach() for x in (self.beta, self.translation, self.rotation, self.gender)))


def rotation_exp(omega):
    return torch.matrix_exp(skew(omega))


def retract(z, xi, capped=False):
    beta, trans, omega = (xi[:, :10], 0.1 * xi[:, 10:13], 0.2 * xi[:, 13:].reshape(-1, 22, 3))
    caps = []
    if capped:
        values = []
        for value, limit in ((beta, 2.0), (trans, 0.5), (omega, math.pi / 3)):
            norm = value.norm(dim=-1, keepdim=True)
            caps.append((norm > limit).detach())
            values.append(value * (limit / norm.clamp_min(1e-12)).clamp_max(1))
        beta, trans, omega = values
    rotation = rotation_exp(omega) @ z.rotation
    return (State(z.beta + beta, z.translation + trans, rotation, z.gender), caps)


def relative(z, target):
    return torch.cat(
        (
            target.beta - z.beta,
            (target.translation - z.translation) / 0.1,
            (matrix_to_axis_angle(target.rotation @ z.rotation.transpose(-1, -2)) / 0.2).flatten(1),
        ),
        1,
    )


def worlds(rotation):
    chain = []
    for j, p in enumerate(PARENTS):
        chain.append(rotation[:, j] if p < 0 else chain[p] @ rotation[:, j])
    return torch.stack(chain, 1)


class Body(nn.Module):
    def __init__(self, geometry, anchor_ids):
        super().__init__()
        self.male = geometry.smplx_layer_male
        self.female = geometry.smplx_layer_female
        self.requires_grad_(False)
        self.register_buffer("anchor_ids", torch.as_tensor(anchor_ids, dtype=torch.long).reshape(22, 8))
        for name in ("male", "female"):
            layer = getattr(self, name)
            selected = self.anchor_ids.flatten()
            columns = (selected[:, None] * 3 + torch.arange(3, device=selected.device)).flatten()
            buffers = dict(
                rest_base=layer.J_regressor @ layer.v_template,
                rest_shape=torch.einsum("jv,vck->jck", layer.J_regressor, layer.shapedirs[..., :10]),
                template=layer.v_template[selected],
                shape=layer.shapedirs[selected, :, :10],
                pose=layer.posedirs[:, columns],
                weights=layer.lbs_weights[selected],
            )
            for key, value in buffers.items():
                self.register_buffer(name + "_" + key, value.detach().clone())

    def forward(self, z, compact=False):
        vertices = joints = None
        for flag, layer in ((True, self.male), (False, self.female)):
            ids = torch.where((z.gender.reshape(-1) > 0.5) == flag)[0]
            if not len(ids):
                continue
            if compact:
                v, j = self.compact_layer(layer, z, ids)
            else:
                out = layer(
                    betas=z.beta[ids],
                    body_pose=z.rotation[ids, 1:],
                    global_orient=z.rotation[ids, :1],
                    transl=z.translation[ids],
                )
                v, j = (out.vertices, out.joints[:, :22])
            if vertices is None:
                vertices = v.new_zeros(len(z.beta), v.shape[1], 3)
                joints = j.new_zeros(len(z.beta), 22, 3)
            vertices = vertices.index_copy(0, ids, v)
            joints = joints.index_copy(0, ids, j)
        anchors = vertices.reshape(len(z.beta), 22, 8, 3) if compact else vertices[:, self.anchor_ids]
        q = worlds(z.rotation)
        center = anchors.mean(2)
        local = torch.einsum("bjdc,bjnd->bjnc", q, anchors - center[:, :, None])
        extents = (local.square().mean(2) + 1e-12).sqrt()
        size = (local.square().sum(-1).mean(2) + 1e-12).sqrt()
        return dict(
            joints=joints,
            vertices=vertices,
            anchors=anchors,
            center=center,
            orientation=q,
            extents=extents,
            size=size,
            segments=part_geometry(joints),
        )

    def canonical(self, z, compact=False):
        return self(
            State(
                z.beta,
                torch.zeros_like(z.translation),
                torch.eye(3, device=z.beta.device, dtype=z.beta.dtype).expand_as(z.rotation),
                z.gender,
            ),
            compact=compact,
        )

    def anchors_only(self, z):
        """Same compact LBS, without unconsumed derived region geometry."""
        vertices = None
        for flag, layer in ((True, self.male), (False, self.female)):
            ids = torch.where((z.gender.reshape(-1) > 0.5) == flag)[0]
            if not len(ids):
                continue
            v, _ = self.compact_layer(layer, z, ids)
            if vertices is None:
                vertices = v.new_zeros(len(z.beta), 176, 3)
            vertices = vertices.index_copy(0, ids, v)
        return vertices.reshape(len(z.beta), 22, 8, 3)

    def compact_layer(self, layer, z, ids):
        """Exact selected-vertex LBS, avoiding full-vertex tangent expansion."""
        from smplx.lbs import batch_rigid_transform

        beta = z.beta[ids]
        n = len(ids)
        name = "male" if layer is self.male else "female"
        rest_base = getattr(self, name + "_rest_base")
        rest_shape = getattr(self, name + "_rest_shape")
        rest = rest_base + torch.einsum("bk,jck->bjc", beta, rest_shape)
        shaped = getattr(self, name + "_template") + torch.einsum(
            "bk,vck->bvc", beta, getattr(self, name + "_shape")
        )
        eye = torch.eye(3, device=beta.device, dtype=beta.dtype)
        rotations = torch.cat((z.rotation[ids], eye.expand(n, 33, 3, 3)), 1)
        pose = (rotations[:, 1:] - eye).flatten(1)
        posed = shaped + (pose @ getattr(self, name + "_pose")).reshape(n, -1, 3)
        joints, transforms = batch_rigid_transform(rotations, rest, layer.parents, dtype=beta.dtype)
        skin = (getattr(self, name + "_weights")[None] @ transforms.reshape(n, 55, 16)).reshape(n, -1, 4, 4)
        homogeneous = torch.cat((posed, torch.ones_like(posed[..., :1])), -1)
        vertices = (skin @ homogeneous[..., None])[..., :3, 0] + z.translation[ids, None]
        return (vertices, joints[:, :22] + z.translation[ids, None])


@torch.no_grad()
def select_anchors(layer):
    """Male zero-beta reference, ancestor-collapse weights, deterministic FPS.

    Start at smallest vertex ID, then farthest squared distance with lowest-ID
    ties. Both templates share topology; never inspect targets or outcomes.
    """
    parents = layer.parents.cpu().tolist()
    weights = layer.lbs_weights.detach().cpu()
    collapsed = weights.new_zeros(weights.shape[0], 22)
    for j in range(weights.shape[1]):
        ancestor = j
        while ancestor >= 22:
            ancestor = parents[ancestor]
        if ancestor < 0:
            raise ValueError("fine joint lacks body22 ancestor")
        collapsed[:, ancestor] += weights[:, j]
    region = collapsed.argmax(-1)
    xyz = layer.v_template.detach().cpu()
    result = []
    for j in range(22):
        ids = torch.where(region == j)[0]
        if len(ids) < 8:
            raise ValueError(f"region {j} has fewer than eight vertices")
        distance = torch.full((len(ids),), float("inf"))
        chosen = []
        next_id = 0
        for _ in range(8):
            chosen.append(int(ids[next_id]))
            distance = torch.minimum(distance, (xyz[ids] - xyz[ids[next_id]]).square().sum(-1))
            distance[next_id] = -1
            next_id = int(distance.argmax())
        if len(set(chosen)) != 8:
            raise ValueError("anchor coverage duplicate")
        result.append(chosen)
    return result
