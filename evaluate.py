"""Full-partition evaluation without checkpoint or subset selection."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from utils.metric_utils import (
    batched_joints,
    batched_smplx,
    mean_joint_rotation_error_deg,
)

from datasets.m4human import stage
from models.geometry import matrix_to_axis_angle
from models.model import build_model

METRICS = ("mve_mm", "mje_mm", "body_mre_deg", "translation_mm", "root_relative_mje_mm")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    geometry = SimpleNamespace(smplx_layer_male=model.body.male, smplx_layer_female=model.body.female)
    total = torch.zeros(len(METRICS), dtype=torch.float64, device=device)
    count = 0
    for batch in loader:
        rpc, rt, target = stage(batch, device)
        state = model(rpc, rt)["states"][-1]
        angle = matrix_to_axis_angle(state.rotation)
        pose = angle[:, 1:].flatten(1)
        pv = batched_smplx(geometry, state.beta, pose, angle[:, 0], state.translation, state.gender)
        gv = batched_smplx(
            geometry, target["betas"], target["pose"], target["root"], target["trans"], target["gender"]
        )
        pj = batched_joints(geometry, pv, target["gender"])
        gj = batched_joints(geometry, gv, target["gender"])
        values = torch.stack(
            (
                (pv - gv).norm(dim=-1).mean(-1),
                (pj - gj).norm(dim=-1).mean(-1),
                mean_joint_rotation_error_deg(pose, target["pose"]),
                (state.translation - target["trans"]).norm(dim=-1) * 1000,
                (pj - pj[:, :1] - (gj - gj[:, :1])).norm(dim=-1).mean(-1),
            ),
            -1,
        )
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite evaluation metric")
        total += values.double().sum(0)
        count += len(rpc)
    if count != len(loader.dataset) or count == 0:
        raise ValueError("evaluation population mismatch")
    return dict(samples=count, **dict(zip(METRICS, (total / count).tolist())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--partition", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--smplx-root")
    parser.add_argument("--m4human-source")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.workers is not None and args.workers < 0:
        parser.error("workers must be nonnegative")
    import torch
    from utils.general_utils import write_json

    from datasets.m4human import M4Human, make_loader

    output = Path(args.output)
    if output.exists():
        parser.error(f"output already exists: {output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["phase"] != "main":
        parser.error("expected a main-training checkpoint")
    config = dict(checkpoint["config"])
    if args.workers is not None:
        config["workers"] = args.workers
    for key in ("dataset_root", "smplx_root", "m4human_source"):
        if getattr(args, key):
            config[key] = getattr(args, key)
    model = build_model(config, args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = M4Human(config, args.partition)
    metrics = evaluate(model, make_loader(dataset, config), args.device)
    result = dict(
        checkpoint=str(Path(args.checkpoint).resolve()),
        epoch=checkpoint["epoch"],
        split=config["split"],
        partition=args.partition,
        metrics=metrics,
    )
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
