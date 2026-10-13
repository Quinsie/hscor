"""Geometry-mediated state correction and learned acceptance."""

import math

import torch
from torch import nn

from models.geometry import PARENTS, State, retract, skew


def mlp(inputs, outputs, hidden=256):
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, outputs),
    )


def state_features(z):
    return torch.cat((z.beta, z.translation / 0.1, z.rotation.flatten(1)), -1)


def normalized_geometry_jacobian(z, body):
    """B x (66+528) x 79 exact derivative at the zero tangent.

    The common tangent probe is valid because body samples are independent.
    Linearized rotations have the exact dExp(0); finite updates use SO(3) Exp.
    """
    base = z.detach()

    def measure(xi):
        rotation = (
            torch.eye(3, device=xi.device, dtype=xi.dtype) + skew(0.2 * xi[13:].reshape(22, 3))
        ) @ base.rotation
        probe = State(base.beta + xi[:10], base.translation + 0.1 * xi[10:13], rotation, base.gender)
        g = body(probe, compact=True)
        return torch.cat((g["joints"].flatten(1), g["anchors"].flatten(1)), 1) / 0.1

    with torch.enable_grad():
        jacobian = torch.func.jacfwd(measure)(z.beta.new_zeros(79))
    return jacobian.detach()


def geometry_solve(jacobian, displacement, rotation_prior=None, damping=0.01):
    """argmin_x mean_points ||Jx-d||^2 + damping ||x-q||^2.

    q is zero in shape/translation and learned only in rotation coordinates.
    Equal isotropic point weights are fixed, never learned confidence or U.
    """
    count = displacement.shape[1]
    if jacobian.shape[1:] != (count * 3, 79):
        raise ValueError("point Jacobian and displacement support differ")
    jt = jacobian.transpose(1, 2) / count
    normal = jt @ jacobian
    matrix = normal + damping * torch.eye(79, device=jacobian.device, dtype=jacobian.dtype)
    rhs = (jt @ displacement.flatten(1).unsqueeze(-1)).squeeze(-1)
    if rotation_prior is not None:
        prior = torch.cat((rhs.new_zeros(len(rhs), 13), rotation_prior.flatten(1)), 1)
        rhs = rhs + damping * prior
    delta = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)
    return delta


class Refinement(nn.Module):
    """Three geometry experts with soft step-role conditioning."""

    damping = 0.01

    def __init__(self, seed=42):
        super().__init__()
        self.context = mlp(3 * 128 + 211, 128)
        incidence = (torch.arange(22)[:, None] == torch.tensor(PARENTS[1:])[None]).float()
        self.register_buffer("parent_incidence", incidence, persistent=False)
        self.global_expert = mlp(256, 6)
        self.part_expert = mlp(256, 6)
        self.surface_expert = mlp(256, 3)
        self.gate = mlp(128 + 3, 3, hidden=128)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed * 1000 + 36)
            self.acceptance = nn.Sequential(nn.Linear(131, 32), nn.GELU(), nn.Linear(32, 1))
            nn.init.normal_(self.acceptance[-1].weight, std=0.001)
            nn.init.constant_(self.acceptance[-1].bias, math.log(99.0))

    def forward(self, z, evidence, body, step):
        if step not in (0, 1, 2):
            raise ValueError("step is zero-based and must be 0, 1, or 2")
        eg, ep, es = (evidence["global"], evidence["part"], evidence["surface"])
        batch = len(z.beta)
        if eg.shape != (batch, 128) or ep.shape != (batch, 43, 128) or es.shape != (batch, 22, 8, 128):
            raise ValueError("expected global B128, part B43x128, surface B22x8x128")
        context = self.context(torch.cat((eg, ep.mean(1), es.mean((1, 2)), state_features(z)), -1))

        def joined(tokens):
            return torch.cat((tokens, context[:, None].expand(-1, tokens.shape[1], -1)), -1)

        segments = ep[:, 22:]
        incoming = torch.cat((torch.zeros_like(segments[:, :1]), segments), 1)
        incidence = self.parent_incidence.to(segments)
        outgoing = incidence @ segments
        counts = incidence.sum(-1).clamp_min(1)
        part = ep[:, :22] + incoming + outgoing / counts[None, :, None]
        global_prediction = self.global_expert(joined(ep[:, :22] + eg[:, None]))
        part_prediction = self.part_expert(joined(part))
        surface_prediction = self.surface_expert(joined(es.flatten(1, 2)))
        jacobian = normalized_geometry_jacobian(z, body)
        dg = geometry_solve(jacobian[:, :66], global_prediction[..., :3], global_prediction[..., 3:])
        dp = geometry_solve(jacobian[:, :66], part_prediction[..., :3], part_prediction[..., 3:])
        ds = geometry_solve(jacobian[:, 66:], surface_prediction)
        proposals = torch.stack((dg, dp, ds), 1)
        role = context.new_zeros(batch, 3)
        role[:, step] = 1
        prior = role * torch.log(context.new_tensor(3.0))
        logits = self.gate(torch.cat((context, role), -1))
        alpha = torch.softmax(logits + prior, -1)
        contributions = alpha[..., None] * proposals
        delta = contributions.sum(1)
        delta = self.accept_correction(context, step, delta)
        new_state, _ = retract(z, delta, capped=True)
        return new_state

    def accept_correction(self, context, step, delta):
        role = context.new_zeros(len(context), 3)
        role[:, step] = 1
        acceptance = torch.sigmoid(self.acceptance(torch.cat((context, role), -1)))
        applied = acceptance * delta
        return applied
