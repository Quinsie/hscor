"""Portable model construction and checkpoint I/O."""

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

UPSTREAM_COMMIT = "e01b8fed6132306c3553ed6680be912c21594dce"


def setup_source(path):
    source = Path(path).resolve()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"M4Human must be checked out at {UPSTREAM_COMMIT}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def optimizer_for(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        decay = (
            0.0
            if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower()
            else config["weight_decay"]
        )
        groups.setdefault(decay, []).append(parameter)
    return torch.optim.AdamW(
        [dict(params=p, weight_decay=d) for d, p in groups.items()],
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-08,
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_mve, config, phase):
    """Atomic epoch-boundary checkpoint, including all random generators."""
    state = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        epoch=epoch,
        best_mve=best_mve,
        config=config,
        phase=phase,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),
    )
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_rng(checkpoint):
    torch.set_rng_state(checkpoint["torch_rng"].cpu())
    if checkpoint["cuda_rng"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint["cuda_rng"]])
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])
