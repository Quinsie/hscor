"""M4Human processed LMDBs: normalized RPC and independently cropped RT."""

import gzip
import hashlib
import importlib
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from utils.general_utils import setup_source


class M4Human(Dataset):
    def __init__(self, config, partition, include_rt=True):
        if partition not in ("train", "val", "test"):
            raise ValueError(partition)
        if config["split"] not in ("s2", "s3"):
            raise ValueError("split must be s2 or s3")
        self.config = dict(config)
        self.partition = partition
        self.include_rt = include_rt and config["modality"] == "rpc_rt"
        self.module = None
        self.envs = None
        self.owner = None
        self.body_models = None
        setup_source(config["m4human_source"])
        module = importlib.import_module("dataset.m4human_dataset")
        with gzip.open(Path(config["dataset_root"]) / "indeces.pkl.gz", "rb") as stream:
            indicators = pickle.load(stream)["p1"][config["split"]][partition]
        invalid = {tuple(x) for x in module.NON_VALID_INDICATOR}
        self.indicators = [x for x in indicators if tuple(x) not in invalid]
        if not self.indicators:
            raise ValueError("empty partition")

    def __len__(self):
        return len(self.indicators)

    def initialize(self):
        if self.module is None:
            setup_source(self.config["m4human_source"])
            self.module = importlib.import_module("dataset.m4human_dataset")
        paths = {
            name: str(Path(self.config["dataset_root"]) / (name + ".lmdb"))
            for name in ("radar_pc", "params", "calib", "indicator")
        }
        if self.include_rt:
            paths["radar_comp"] = str(Path(self.config["dataset_root"]) / "radar_comp.lmdb")
        self.envs, self.owner = self.module.ensure_read_envs(self.envs, paths, self.owner)
        if self.body_models is None:
            import smplx

            self.body_models = {
                g: smplx.create(
                    model_path=str(Path(self.config["smplx_root"]) / f"SMPLX_{g.upper()}.npz"),
                    model_type="smplx",
                    gender=g,
                    ext="npz",
                )
                for g in ("male", "female")
            }

    def temporal_indices(self, index):
        sid, action, frame = self.indicators[index]
        last, expected, selected = (index, frame, [])
        for i in range(index, -1, -1):
            s, a, f = self.indicators[i]
            if s == sid and a == action and (f == expected):
                last = i
            selected.append(last)
            expected -= 1
            if len(selected) == 4 or s != sid or a != action or (f < frame - 3):
                break
        selected += selected[-1:] * (4 - len(selected))
        return selected[::-1]

    def sparse_rt(self, index):
        unpack = importlib.import_module("dataset.m4human_utils").unpack_ndarray
        shape = (121, 111, 31)
        size = int(np.prod(shape))
        indices, values = ([], [])
        with self.envs["radar_comp"].begin() as transaction:
            for slot, frame in enumerate(self.temporal_indices(index)):
                compressed = unpack(transaction.get(str(self.indicators[frame]).encode()))
                xyz = compressed[:, :3].astype(np.int64)
                for axis, bound in enumerate(shape):
                    if np.any(xyz[:, axis] < -bound) or np.any(xyz[:, axis] >= bound):
                        raise IndexError("RT coordinate outside source shape")
                    xyz[:, axis] %= bound
                flat = (xyz[:, 0] * shape[1] + xyz[:, 1]) * shape[2] + xyz[:, 2]
                value = compressed[:, 3].astype(np.float32)
                if len(flat) > 1 and (not np.all(flat[1:] > flat[:-1])):
                    flat, reverse = np.unique(flat[::-1], return_index=True)
                    value = value[len(value) - 1 - reverse]
                indices.append(torch.from_numpy(flat + slot * size))
                values.append(torch.from_numpy(value))
        return (torch.cat(indices), torch.cat(values))

    def __getitem__(self, key):
        index, epoch, rank, occurrence = (key, 0, 0, key) if isinstance(key, int) else key
        self.initialize()
        m = self.module
        indicator = self.indicators[index]
        with self.envs["params"].begin() as pt, self.envs["calib"].begin() as ct:
            key_bytes = str(indicator).encode()
            parameter = m.unpack_dict_np(pt.get(key_bytes))
            calibration = m.unpack_dict_np(ct.get(key_bytes))
        parameter["gender"] = m.gender_info[f"P{indicator[0]}"]
        parameter = m.calibrate_param_to_radar(parameter, calibration)
        parameter["trans"][..., 2] -= 1.5
        vertices = m.generate_vertices_in_radar_space(self.body_models, parameter)
        joints = m.compute_joints_from_vertices(self.body_models, vertices, parameter["gender"])
        seed = self.config["seed"] if self.partition == "train" else 42
        seed_bytes = f"{seed}:{self.partition}:{epoch}:{rank}:{occurrence}".encode()
        rng = np.random.get_state()
        np.random.seed(int.from_bytes(hashlib.sha256(seed_bytes).digest()[:4], "little"))
        try:
            points = m.find_temporal_radar_pc_lmdb(self.indicators, self.envs, index, T=4)[0]
            for i in range(len(points)):
                points[i][..., 2] -= 1.5
                points[i] = m.pad_radar_pc(points[i], target_points=1000)
        finally:
            np.random.set_state(rng)
        result = m.ToTensor()(
            dict(
                rawImage_XYZ=np.zeros((1,), dtype=np.float32),
                radar_points=np.stack(points),
                parameter=parameter,
                joints_root=joints,
                indicator=indicator,
            )
        )
        result["_dataset_index"] = index
        if self.include_rt:
            result["_sparse"] = self.sparse_rt(index)
        return result


def collate(items):
    sparse = [item.pop("_sparse") for item in items if "_sparse" in item]
    result = default_collate(items)
    if sparse:
        if len(sparse) != len(items):
            raise ValueError("mixed RT availability within a batch")
        size = 4 * 121 * 111 * 31
        result["_rt_indices"] = torch.cat([x[0] + i * size for i, x in enumerate(sparse)])
        result["_rt_values"] = torch.cat([x[1] for x in sparse])
    return result


class RadarCrop:
    def __init__(self, batch, device):
        self.indices = batch["_rt_indices"].to(device, non_blocking=True)
        self.values = batch["_rt_values"].to(device, non_blocking=True)
        self.batch_size = len(batch["radar_points"])

    def __call__(self, origin):
        i = self.indices
        z, y, x = (i % 31, i // 31 % 111, i // (31 * 111) % 121)
        slot = i // (31 * 111 * 121)
        xyz = torch.stack((x, y, z), -1) - origin[slot // 4]
        inside = ((xyz >= 0) & (xyz < xyz.new_tensor([64, 64, 31]))).all(-1)
        q = xyz[inside]
        target = ((slot[inside] * 64 + q[:, 0]) * 64 + q[:, 1]) * 32 + q[:, 2]
        crop = self.values.new_zeros(self.batch_size, 4, 64, 64, 32)
        crop.view(-1).index_copy_(0, target, self.values[inside])
        return crop


def stage(batch, device, warmup=False):
    p = batch["parameter"]
    targets = {
        name: p[key].to(device, non_blocking=True)
        for name, key in [
            ("betas", "betas"),
            ("pose", "pose_body"),
            ("root", "root_orient"),
            ("trans", "trans"),
            ("gender", "gender"),
        ]
    }
    targets["betas"] = targets["betas"][:, :10]
    rpc = batch["radar_points"].to(device, non_blocking=True)
    rt = RadarCrop(batch, device) if not warmup and "_rt_indices" in batch else None
    return rpc, rt, targets


def make_loader(dataset, config, *, plan=None):
    options = dict(
        num_workers=config["workers"],
        pin_memory=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(8600 + config["seed"]),
    )
    if config["workers"]:
        options.update(multiprocessing_context="spawn", persistent_workers=True)
    if plan is not None:
        return DataLoader(dataset, batch_sampler=plan, **options)
    return DataLoader(dataset, batch_size=config["eval_batch_size"], shuffle=False, **options)
