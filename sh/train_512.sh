#!/bin/bash
# Train StyleSwin at 512px. Runs on a workstation (`bash sh/train_512.sh`) or a
# cluster (`sbatch --account=<proj> --partition=rocky --gpus=2 sh/train_512.sh`) --
# SLURM options (proj/partition/host) are passed at submission time, never written in this file.
set -euo pipefail

# Self-locate the repo root (walk up to pyproject.toml) so the script runs from anywhere.
cd "$(dirname "$0")/.."
while [[ ! -f pyproject.toml && "$PWD" != / ]]; do cd ..; done

conda activate styleswin

# System CUDA toolkit provides nvcc to JIT-build op/; derive CUDA_HOME from it. TORCH_CUDA_ARCH_LIST
# defaults to Hopper (sm_90); override for other GPUs.
command -v module >/dev/null 2>&1 && module load CUDA/13.1 || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

# Offline cluster: combra metric backbones are prefetched once on a login node via
# `styleswin-download-models`; force HF offline so the CLIP/DINOv2 load reads the cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

styleswin-train --outdir=./runs/wc-cv \
    --cfg styleswin-512 \
    --data=./datasets/imagenet_9to4_1024x1024_512x512.zip \
    --gpus="${GPUS:-2}" \
    --cond True \
    --combra-metrics True \
    --snapshot-keep-last 3 \
    --kimg 25000 \
    --snap 50
