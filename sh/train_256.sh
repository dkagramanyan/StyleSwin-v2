#!/bin/bash
# Train StyleSwin at 256px. Runs on a workstation (`bash sh/train_256.sh`) or a
# cluster (`sbatch --account=<proj> --partition=rocky --gpus=2 sh/train_256.sh`) --
# SLURM options (proj/partition/host) are passed at submission time, never written in this file.
set -euo pipefail

# Self-locate the repo root (walk up to pyproject.toml) so the script runs from anywhere.
cd "$(dirname "$0")/.."
while [[ ! -f pyproject.toml && "$PWD" != / ]]; do cd ..; done

conda activate styleswin-v2

# System CUDA toolkit provides nvcc to JIT-build op/; derive CUDA_HOME from it.
command -v module >/dev/null 2>&1 && module load CUDA/13.1 || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
# TORCH_CUDA_ARCH_LIST: build the JIT `op/` extensions for the GPUs actually present
# (a 3090 is sm_86, not the sm_90 this used to assume); falls back to Hopper when
# nvidia-smi is unavailable, e.g. on a login node. An explicit value always wins.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd';' -)}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

# Offline cluster: combra metric backbones are prefetched once on a login node via
# `styleswin-download-models`; force HF offline so the CLIP/DINOv2 load reads the cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

styleswin-train --outdir=./runs/wc-cv \
    --cfg styleswin-256 \
    --data=./datasets/imagenet_9to4_1024x1024_256x256.zip \
    --gpus="${GPUS:-2}" \
    --cond True \
    --combra-metrics True \
    --snapshot-keep-last 3 \
    --kimg 25000 \
    --snap 50
