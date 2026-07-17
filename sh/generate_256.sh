#!/bin/bash
# Generate 256px samples (hdf5, angle-pipeline input). Set NETWORK=<snapshot-inference.pt>.
# Workstation: `NETWORK=... bash sh/generate_256.sh`; cluster: add SLURM options at submit.
set -euo pipefail

cd "$(dirname "$0")/.."
while [[ ! -f pyproject.toml && "$PWD" != / ]]; do cd ..; done

conda activate styleswin

command -v module >/dev/null 2>&1 && module load CUDA/13.1 || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

: "${NETWORK:?set NETWORK=path/to/network-snapshot-<kimg>-inference.pt}"

styleswin-gen-images \
    --network="${NETWORK}" \
    --outdir=./generated/256 \
    --save-mode hdf5 \
    --classes 0,1,2 \
    --samples-per-class "${SAMPLES_PER_CLASS:-10000}" \
    --trunc 0.7 \
    --gpus="${GPUS:-2}" \
    --batch-gpu 32
