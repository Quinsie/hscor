"""TRAIN-only bootstrap, fixed-budget optimization and VAL selection."""

import argparse
import json
from pathlib import Path

import torch
from evaluate import evaluate
from torch.utils.data import Subset
from utils.general_utils import optimizer_for, restore_rng, save_checkpoint, write_json
from utils.loss_utils import objective
from utils.sampling import TrainingBatches

from datasets.m4human import M4Human, make_loader, stage
from models.geometry import PARENTS
from models.model import build_model
from models.radar import radar_indices


def train_epoch(model, dataset, optimizer, config, epoch, device):
    """One forward/backward pass per batch, followed by one optimizer update."""
    plan = TrainingBatches(len(dataset), config["seed"], epoch)
    model.train()
    total, count = 0.0, 0
    for update, batch in enumerate(make_loader(dataset, config, plan=plan), 1):
        optimizer.zero_grad(set_to_none=True)
        rpc, rt, target = stage(batch, device, model.warmup)
        result = model(rpc, rt)
        loss, _ = objective(model, result, target)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(rpc)
        count += len(rpc)
        if update == 1 or update % 50 == 0 or update == len(plan):
            print(
                f"epoch={epoch + 1} update={update}/{len(plan)} loss={float(loss.detach()):.6f}", flush=True
            )
        del result, loss
    return total / count


@torch.no_grad()
def assess_initializer(model, loader, device):
    model.eval()
    states, losses, coverage, errors, bones = ([], [], [], [], [])
    for batch in loader:
        rpc, _, target = stage(batch, device, True)
        result = model(rpc)
        z = result["states"][0]
        loss, _ = objective(model, result, target)
        g, origin = (result["initial_geometry"], result["origin"])
        if model.rpc_only:
            valid = torch.isfinite(result["coordinates"]).all(-1)
            row = [float(valid.float().mean())] * 2
        else:
            joints = g["joints"][:, :, None] + model.observation.joint_offsets
            surface = (
                g["anchors"][:, :, :, None]
                + torch.einsum("bjcd,od->bjoc", g["orientation"], model.observation.surface_offsets)[
                    :, :, None
                ]
            )
            row = []
            for xyz in (joints, surface):
                native = radar_indices(xyz)
                coord = native - origin.reshape((len(origin),) + (1,) * (xyz.ndim - 2) + (3,))
                valid = ((native >= 0) & (native <= native.new_tensor([120, 110, 30]))).all(-1)
                valid &= ((coord >= 0) & (coord <= coord.new_tensor([62, 62, 30]))).all(-1)
                row.append(float(valid.float().mean()))
        value = torch.cat((z.beta, z.translation, z.rotation.flatten(1)), 1)
        if not torch.isfinite(value).all() or not torch.isfinite(loss):
            raise ValueError("nonfinite initializer")
        states.append(value.cpu())
        losses.append((float(loss), len(rpc)))
        coverage.append((row, len(rpc)))
        errors.append(
            max(
                float((z.rotation.transpose(-1, -2) @ z.rotation - torch.eye(3, device=device)).abs().max()),
                float((torch.linalg.det(z.rotation) - 1).abs().max()),
            )
        )
        bones.append(float((g["joints"][:, 1:] - g["joints"][:, list(PARENTS[1:])]).norm(dim=-1).min()))
    values = torch.cat(states)
    count = sum((n for _, n in losses))
    variation = (
        values - values.mean(0, keepdim=True)
    ).square().mean().sqrt() / values.square().mean().sqrt().clamp_min(1e-12)
    return dict(
        samples=count,
        loss=sum((v * n for v, n in losses)) / count,
        coverage=[sum((v[k] * n for v, n in coverage)) / count for k in range(2)],
        rotation_error=max(errors),
        min_bone=min(bones),
        relative_variation=float(variation),
    )


def bootstrap(config, directory, device):
    model = build_model(config, device)
    model.configure(True)
    train = M4Human(config, "train", include_rt=False)
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(42035))[
        : config["bootstrap_probe_samples"]
    ].tolist()
    probe = make_loader(Subset(train, indices), config)
    baseline = assess_initializer(model, probe, device)
    write_json(directory / "bootstrap_baseline.json", dict(indices=indices, measurement=baseline))
    optimizer = optimizer_for(model, config)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, config["scheduler_step"], config["scheduler_gamma"]
    )
    for epoch in range(config["bootstrap_epochs"]):
        train_epoch(model, train, optimizer, config, epoch, device)
        scheduler.step()
        measured = assess_initializer(model, probe, device)
        checks = dict(
            finite_legal=measured["rotation_error"] <= 0.001 and measured["min_bone"] > 1e-06,
            geometry_support=min(measured["coverage"]) >= 0.95,
            sample_conditioned=measured["relative_variation"] > 1e-06,
            learning_progress=measured["loss"] <= 0.7 * baseline["loss"],
        )
        write_json(directory / f"bootstrap_{epoch + 1:02d}.json", dict(measurement=measured, checks=checks))
        if all(checks.values()):
            state = {k: v.detach().cpu().clone() for k, v in model.initializer.state_dict().items()}
            torch.save(state, directory / "initializer.pt")
            return state
    raise RuntimeError("Initializer did not pass its TRAIN-only checks within the bootstrap budget.")


def train(config, output, device, resume=None):
    directory = Path(output)
    if resume is None:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "config.json", config)
        initial = bootstrap(config, directory, device)
        model = build_model(config, device)
        model.initializer.load_state_dict(initial, strict=True)
        checkpoint = None
    else:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint["phase"] != "main" or checkpoint["config"] != config:
            raise ValueError("Resume requires a main-training checkpoint with the same configuration.")
        if Path(resume).resolve().parent != directory.resolve():
            raise ValueError("Resume in the checkpoint output directory.")
        model = build_model(config, device)
        model.load_state_dict(checkpoint["model"], strict=True)
    model.configure(False)
    optimizer = optimizer_for(model, config)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, config["scheduler_step"], config["scheduler_gamma"]
    )
    epoch_start, best = (0, float("inf"))
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch_start, best = (checkpoint["epoch"], checkpoint["best_mve"])
        restore_rng(checkpoint)
    train_data = M4Human(config, "train")
    val_data = M4Human(config, "val")
    val_loader = make_loader(val_data, config)
    for epoch in range(epoch_start, config["epochs"]):
        loss = train_epoch(model, train_data, optimizer, config, epoch, device)
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        improved = metrics["mve_mm"] < best
        if improved:
            best = metrics["mve_mm"]
        write_json(
            directory / f"epoch_{epoch + 1:03d}.json", dict(epoch=epoch + 1, loss=loss, validation=metrics)
        )
        if improved:
            save_checkpoint(
                directory / "best.pt", model, optimizer, scheduler, epoch + 1, best, config, "main"
            )
        save_checkpoint(directory / "latest.pt", model, optimizer, scheduler, epoch + 1, best, config, "main")
        print(f"epoch={epoch + 1} val_mve_mm={metrics['mve_mm']:.4f} best={best:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=("rpc_rt", "rpc_only"), required=True)
    parser.add_argument("--split", choices=("s2", "s3"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--smplx-root", required=True)
    parser.add_argument("--m4human-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resume", help="Resume a main-training epoch checkpoint in its original output directory"
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    config.update(
        modality=args.model,
        split=args.split,
        seed=args.seed,
        workers=args.workers,
        dataset_root=args.dataset_root,
        smplx_root=args.smplx_root,
        m4human_source=args.m4human_source,
    )
    for key in (
        "epochs",
        "bootstrap_epochs",
        "bootstrap_probe_samples",
        "eval_batch_size",
    ):
        if not isinstance(config[key], int) or config[key] < 1:
            parser.error(f"{key} must be a positive integer")
    if config["batch_size"] != 256:
        parser.error("batch_size must be 256")
    if config["workers"] < 0:
        parser.error("workers must be nonnegative")
    train(config, args.output, args.device, args.resume)


if __name__ == "__main__":
    main()
