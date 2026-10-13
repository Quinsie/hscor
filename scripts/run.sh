#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Edit the run settings here.
cuda=0
model="rpc_rt"                    # rpc_rt or rpc_only
split="s3"                        # s2 or s3
seed=42
workers=2
dataset_root="data/M4Human"
smplx_root="data/SMPL-X"
m4human_source="external/M4Human"
config="configs/train.json"
output="outputs/$model/$split/seed$seed"
partition="test"                  # test or val
resume="$output/latest.pt"
checkpoint="$output/best.pt"
result="$output/$partition.json"

export CUDA_VISIBLE_DEVICES="$cuda"
mode="${1:-all}"

case "$model" in
    rpc_rt|rpc_only) ;;
    *) echo "model must be rpc_rt or rpc_only." >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
    echo "Usage: bash scripts/run.sh [all|train|resume|evaluate]" >&2
    exit 2
fi
case "$mode" in
    all|train|resume|evaluate) ;;
    *) echo "Usage: bash scripts/run.sh [all|train|resume|evaluate]" >&2; exit 2 ;;
esac
case "$partition" in
    val|test) ;;
    *) echo "partition must be val or test." >&2; exit 2 ;;
esac

if [[ "$mode" == all || "$mode" == train || "$mode" == resume ]]; then
    resume_args=()
    if [[ "$mode" == resume ]]; then
        resume_args=(--resume "$resume")
    fi
    python -u train.py --config "$config" --model "$model" --split "$split" \
        --seed "$seed" --workers "$workers" --output "$output" --device cuda:0 \
        --dataset-root "$dataset_root" --smplx-root "$smplx_root" \
        --m4human-source "$m4human_source" "${resume_args[@]}"
fi

if [[ "$mode" == all || "$mode" == evaluate ]]; then
    python -u evaluate.py --checkpoint "$checkpoint" --device cuda:0 \
        --partition "$partition" --output "$result" --workers "$workers" \
        --dataset-root "$dataset_root" --smplx-root "$smplx_root" \
        --m4human-source "$m4human_source"
fi
