"""Radar-only inference with three connected human-state refinements."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from utils.general_utils import seed_everything, setup_source

from models.geometry import PARENTS, Body, named_init, select_anchors
from models.initializer import Initializer, PreparedP4DConv
from models.observation import Observation, RPCObservation
from models.radar import RadarField, radar_indices
from models.refinement import Refinement


class Model(nn.Module):
    def __init__(self, encoder, geometry, anchor_ids, seed=42, modality="rpc_rt"):
        super().__init__()
        if modality not in ("rpc_rt", "rpc_only"):
            raise ValueError("modality must be rpc_rt or rpc_only")
        self.rpc_only = modality == "rpc_only"
        self.initializer = Initializer(encoder, geometry, seed=seed)
        self.body = Body(geometry, anchor_ids)
        if not self.rpc_only:
            self.rt_field = RadarField()
        self.observation = RPCObservation() if self.rpc_only else Observation()
        named_init(self.observation, "e2ehss.common.observation", seed=seed)
        if not self.rpc_only:
            named_init(self.rt_field.projections, "e2ehss.common.rt_projection", seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed * 1000 + 3)
            self.family = Refinement(seed=seed)
        self.configure(False)

    def configure(self, warmup=False):
        self.warmup = warmup
        for name, p in self.named_parameters():
            p.requires_grad_(not name.startswith("body.") and (not warmup or name.startswith("initializer.")))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.body.eval()
        if getattr(self, "warmup", False) and not self.rpc_only:
            self.rt_field.trunk.eval()
        return self

    def forward(self, rpc, rt=None):
        if self.rpc_only and rt is not None:
            raise ValueError("RPC-only inference does not accept RT input")
        z, h, proposal, memory, coordinates = self.initializer(rpc)
        g = self.body(z, compact=True)
        if self.rpc_only:
            if coordinates.shape[1:] != (93, 3) or not torch.isfinite(coordinates).all():
                raise ValueError("expected 93 finite RPC coordinates")
            field, origin = memory, coordinates
        else:
            center = torch.round(radar_indices(g["joints"][:, 0])).long()
            origin = torch.minimum(
                (center - center.new_tensor([32, 32, 16])).clamp_min(0), center.new_tensor([57, 47, 0])
            )
            field = None if self.warmup else self.rt_field(rt(origin))
        initial_geometry = g
        states = [z]
        if not self.warmup:
            for step in range(3):
                if step:
                    g = self.body(z, compact=True)
                evidence, h = self.observation(g, h, field, origin)
                z = self.family(z, evidence, self.body, step)
                states.append(z)
        return dict(
            states=states,
            proposal=proposal,
            initial_geometry=initial_geometry,
            origin=None if self.rpc_only else origin,
            coordinates=coordinates,
        )


def build_model(config, device):
    """Random sensing initialization; only licensed body geometry is loaded."""
    setup_source(config["m4human_source"])
    from smplx import SMPLXLayer

    root = Path(config["smplx_root"])
    paths = {g: str(root / f"SMPLX_{g.upper()}.npz") for g in ("male", "female")}
    for path in paths.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    convolution = importlib.import_module("mmwave_models.Point_models.p4trans.point_4d_convolution").P4DConv
    seed = config["seed"]
    seed_everything(seed * 10000 + 202)
    tube = convolution(
        in_planes=1,
        mlp_planes=[1024],
        mlp_batch_norm=[False],
        mlp_activation=[False],
        spatial_kernel_size=[0.3, 32],
        spatial_stride=32,
        temporal_kernel_size=3,
        temporal_stride=1,
        temporal_padding=[1, 0],
        operator="+",
        spatial_pooling="max",
        temporal_pooling="max",
    )
    encoder = torch.nn.Module()
    encoder.tube_embedding = PreparedP4DConv(tube, compiled_pool=config["compile_pool"])
    encoder.pos_embedding = nn.Conv1d(4, 1024, 1)
    encoder.emb_relu = False
    geometry = SimpleNamespace(
        **{
            "smplx_layer_" + g: SMPLXLayer(model_path=paths[g], gender=g, use_pca=False)
            for g in ("male", "female")
        }
    )
    for layer in (geometry.smplx_layer_male, geometry.smplx_layer_female):
        layer.requires_grad_(False)
        if tuple(layer.parents[:22].tolist()) != PARENTS:
            raise ValueError("SMPL-X body22 skeleton mismatch")
    anchors = select_anchors(geometry.smplx_layer_male)
    seed_everything(seed)
    model = Model(encoder, geometry, anchors, seed=seed, modality=config["modality"])
    return model.to(device)
