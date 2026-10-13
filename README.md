<h1 align="center">

  *Human Mesh Reconstruction from mmWave Radar*
  </h1>

<p align="center">
  <a href="https://quinsie.github.io/">Jiho Pyo</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Department of Computer Science &amp; Artificial Intelligence,
Jeonbuk National University, Jeonju, Republic of Korea
</p>

This implementation reconstructs a 3D human mesh from radar point clouds, with optional radar tensors. Both RPC+RT and RPC-only models form an initial parametric human state, apply three geometry-mediated corrections, and directly generate the final SMPL-X mesh.

## Setup

### Installation

Use Linux with an NVIDIA GPU and the CUDA 12.8 toolkit for the point-cloud operators.

```bash
git clone https://github.com/Quinsie/hscor.git
cd hscor
conda env create -f environment.yml
conda activate radar

git clone https://github.com/FanJunqiao/M4Human.git external/M4Human
git -C external/M4Human checkout e01b8fed6132306c3553ed6680be912c21594dce
pip install --no-build-isolation "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#subdirectory=pointnet2_ops_lib"
```

### Data Preparation

Download the processed radar release from [M4Human](https://github.com/FanJunqiao/M4Human) and the male/female NPZ models from [SMPL-X](https://smpl-x.is.tue.mpg.de/). Arrange the files as follows:

```text
data/
├── M4Human/
│   ├── indeces.pkl.gz
│   ├── radar_pc.lmdb
│   ├── radar_comp.lmdb
│   ├── params.lmdb
│   ├── calib.lmdb
│   └── indicator.lmdb
└── SMPL-X/
    ├── SMPLX_MALE.npz
    └── SMPLX_FEMALE.npz
```

Set the data paths in `scripts/run.sh` if the files are stored elsewhere. RPC-only does not require `radar_comp.lmdb`. The loader reads the saved `p1` split, uses four radar frames, and applies the same 1.5 m vertical normalization to point clouds and targets.

## Running

Activate the `radar` environment and edit the experiment settings at the top of `scripts/run.sh`: GPU, model (`rpc_rt` or `rpc_only`), split, seed, workers, data paths, output directory, and checkpoints. Training hyperparameters are defined in `configs/train.json`.

Then run:

```bash
bash scripts/run.sh
```

This trains the selected model, then evaluates the validation-selected `best.pt` on the selected partition. A failed training process stops the pipeline. Relative paths are resolved from the repository root. Fresh training requires a new output directory; evaluation requires a new result file.

### Training

To train without the final evaluation:

```bash
bash scripts/run.sh train
```

Training starts from random sensing weights. A TRAIN-only initializer bootstrap runs for up to five epochs, followed by 30 joint-training epochs. Both models use batch size 256 on a single GPU, with one forward/backward pass per batch and no gradient accumulation.

Approximate training requirements on an NVIDIA RTX 4090:

| Model | VRAM | Training time |
| --- | --- | --- |
| RPC+RT | 17.5 GiB | 22 hours |
| RPC-only | 13.0 GiB | 15 hours |

The default split is S3. Set `split="s2"` in `scripts/run.sh` for S2. Training selects `best.pt` by full-validation mean vertex error and saves `latest.pt` after each epoch.

Resume main training from the last completed epoch:

```bash
bash scripts/run.sh resume
```

Keep the original experiment settings and training hyperparameters when resuming. This resumes training only; run evaluation separately after completion.

### Evaluation

```bash
bash scripts/run.sh evaluate
```

Evaluation restores the model and split saved in the checkpoint and reports mean vertex error, regressed-joint error, root-relative joint error, and translation error in millimetres, plus body-joint rotation error in degrees. Set `partition="val"` in `scripts/run.sh` to evaluate validation data.

## Code Structure

```text
configs/       training configuration
datasets/      M4Human loader and radar crops
models/        human state, radar encoders, and refinement
scripts/       training and evaluation scripts
utils/         losses, metrics, sampling, and checkpoints
train.py       initializer bootstrap and training
evaluate.py    fixed-checkpoint evaluation
```

## Acknowledgements

This implementation uses [M4Human](https://github.com/FanJunqiao/M4Human), [SMPL-X](https://github.com/vchoutas/smplx), and the CUDA operators from [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch).
