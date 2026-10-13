"""Per-state geometry supervision and body-joint geodesic loss."""

import math

import torch
from torch.nn import functional as F

from models.geometry import State, skew


def target_state(target):
    aa = torch.cat((target["root"][:, None], target["pose"].reshape(-1, 21, 3)), 1)
    return State(
        target["betas"], target["trans"], torch.matrix_exp(skew(aa)), target["gender"].reshape(-1, 1)
    )


def articulation(j):
    angles = []
    for a, b, c in ((16, 18, 20), (17, 19, 21), (1, 4, 7), (2, 5, 8)):
        v = j[:, a] - j[:, b]
        w = j[:, c] - j[:, b]
        cross = torch.cross(v, w, dim=-1).norm(dim=-1)
        dot = (v * w).sum(-1)
        angles.append((math.pi - torch.atan2(cross, dot)) / (math.pi / 6))
    return torch.stack(
        angles + [(j[:, 7] - j[:, 8]).norm(dim=-1) / 0.1, (j[:, 20] - j[:, 21]).norm(dim=-1) / 0.1], 1
    )


def state_loss(z, target, body, truth, canonical_truth):
    g = body(z)
    canonical = body.canonical(z)
    terms = dict(
        mesh=F.huber_loss(g["vertices"] / 0.1, truth["vertices"] / 0.1),
        joint=0.5 * F.huber_loss(g["joints"] / 0.1, truth["joints"] / 0.1),
        rotation=rotation_loss(z.rotation, target.rotation),
        shape=0.2 * F.huber_loss(canonical["vertices"] / 0.05, canonical_truth["vertices"] / 0.05),
        beta=0.1 * F.huber_loss(z.beta, target.beta),
        translation=0.5 * F.huber_loss(z.translation / 0.1, target.translation / 0.1),
        articulation=0.5 * F.huber_loss(articulation(g["joints"]), articulation(truth["joints"])),
    )
    return (sum(terms.values()), terms)


def geodesic(pred, target):
    """SO(3) angle in radians; finite subgradient at identity, no acos floor."""
    r = target @ pred.transpose(-1, -2)
    vee = (
        torch.stack(
            (r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]), -1
        )
        * 0.5
    )
    sine = torch.linalg.vector_norm(vee, dim=-1)
    cosine = (r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5
    return torch.atan2(sine, cosine)


def rotation_loss(pred, target, coefficient=1.025866004889126):
    squared = (pred - target).square().sum((-1, -2))
    root = 0.5 / 22 * squared[:, 0].mean()
    geo = geodesic(pred[:, 1:], target[:, 1:]).mean()
    weighted = coefficient * geo
    return root + weighted


def objective(model, result, target):
    gt = target_state(target)
    with torch.no_grad():
        truth = model.body(gt)
        canonical = model.body.canonical(gt)
    total = gt.beta.new_zeros(())
    parts = {}
    weights = (1.0,) if model.warmup else (0.05, 0.15, 0.3, 1.0)
    for k, (z, weight) in enumerate(zip(result["states"], weights)):
        value, terms = state_loss(z, gt, model.body, truth, canonical)
        total = total + weight * value
        parts.update({f"state{k}_{name}": v for name, v in terms.items()})
    gender = 0.1 * torch.nn.functional.binary_cross_entropy(result["states"][0].gender, gt.gender)
    total = total + gender
    parts["gender"] = gender
    if model.warmup:
        proposal = 0.1 * torch.nn.functional.huber_loss(result["proposal"] / 0.1, truth["joints"] / 0.1)
        total = total + proposal
        parts["proposal"] = proposal
    parts["total"] = total
    return (total, parts)
