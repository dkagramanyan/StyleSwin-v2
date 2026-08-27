#!/bin/bash
# Generate 512px samples (hdf5, angle-pipeline input). Set NETWORK=<snapshot-inference.pt>.
# Workstation: `NETWORK=... bash sh/generate_512.sh`; cluster: add SLURM options at submit.
set -euo pipefail

cd "$(dirname "$0")/.."
while [[ ! -f pyproject.toml && "$PWD" != / ]]; do cd ..; done

conda activate styleswin-v2

command -v module >/dev/null 2>&1 && module load CUDA/13.1 || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
# TORCH_CUDA_ARCH_LIST: build the JIT `op/` extensions for the GPUs actually present
# (a 3090 is sm_86, not the sm_90 this used to assume); falls back to Hopper when
# nvidia-smi is unavailable, e.g. on a login node. An explicit value always wins.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd';' -)}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

: "${NETWORK:?set NETWORK=path/to/styleswin-snapshot-<kimg>-inference.pt}"

styleswin-gen-images \
    --network="${NETWORK}" \
    --outdir=./generated/512 \
    --save-mode hdf5 \
    --classes 0,1,2 \
    --samples-per-class "${SAMPLES_PER_CLASS:-10000}" \
    --trunc 0.7 \
    --gpus="${GPUS:-2}" \
    --batch-gpu 32
