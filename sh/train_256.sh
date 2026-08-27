#!/usr/bin/env bash
# StyleSwin -- train at 256x256.
#
# Workstation:  bash sh/train_256.sh    (DATA=<zip> GPUS=<n> ... to override)
# SLURM:        sbatch --account=<proj> --partition=<part> --nodes=1 --gpus=2 --cpus-per-task=8 --time=3-0:0 sh/train_256.sh
#
# Defaults target the production allocation: 2x H200 (sm_90), 8 CPUs, fixed seed 42.
#
# Every knob is an env var with a default; anything after the script name is appended to
# the command (e.g. `... --kimg 200 --snap 2` for a smoke run). No user homes, --nodelist
# or account IDs live here -- SLURM specifics come from the sbatch line (spec §9).
set -euo pipefail

# --- Environment -------------------------------------------------------------
# Repo root: under SLURM the script runs from a spool copy, so walk up from the submit
# dir there and from this file's own location on a workstation.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
while [[ ! -f "$REPO_DIR/pyproject.toml" && "$REPO_DIR" != / ]]; do REPO_DIR="$(dirname "$REPO_DIR")"; done
[[ -f "$REPO_DIR/pyproject.toml" ]] || { echo "cannot find the repo root -- submit from inside the repo" >&2; exit 1; }
cd "$REPO_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-styleswin-v2}"   # env name = repo name
# Custom CUDA ops JIT-compile on first import against the system CUDA toolkit (module
# name overridable via CUDA_MODULE); H200 (sm_90) arch by default.
command -v module >/dev/null 2>&1 && module load "${CUDA_MODULE:-CUDA/13.1}" || true
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"   # H200 = sm_90
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${HOME}/.cache/cuda_cache}"
# Offline-cluster contract: backbones are prefetched once on a login node
# (styleswin-download-models); compute nodes never reach the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"      # CLIP (CMMD) weights
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"      # torch.hub DINOv2 + Inception weights

# GPUs / CPUs: 2x H200 and 8 CPUs. SLURM sets CUDA_VISIBLE_DEVICES itself; the default
# only applies on a workstation. 8 CPUs / 2 ranks -> 4 threads per rank, 3 loader
# workers per rank (WORKERS) so the two main processes keep a core each.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# Determinism / logging: PYTHONHASHSEED pins Python hashing alongside --seed; NCCL
# surfaces a dead rank as an error instead of a hang; Python output is unbuffered so
# the SLURM log follows the run.
export PYTHONHASHSEED=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

# --- One console-command call ------------------------------------------------
styleswin-train \
    --outdir "${OUTDIR:-./runs}" \
    --cfg "${CFG:-styleswin-256}" \
    --data "${DATA:-./datasets/imagenet_9to4_1024x1024_256x256.zip}" \
    --gpus "${GPUS:-2}" \
    ${BATCH_GPU:+--batch-gpu "$BATCH_GPU"} \
    --cond True --mirror False \
    --kimg "${KIMG:-25000}" --snap "${SNAP:-50}" --snapshot-keep-last "${KEEP_LAST:-1}" \
    --combra-metrics True --num-fid-samples "${NUM_FID_SAMPLES:-10000}" \
    --seed "${SEED:-42}" --workers "${WORKERS:-3}" \
    "$@"
