#!/bin/bash
#SBATCH --job-name=styleswin-gen-256
#SBATCH --partition=rocky
#SBATCH --account=proj_1661
#SBATCH --nodelist=cn-049
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0:0
#SBATCH --gres-flags=enforce-binding

# Hop to the repo root so the script can be submitted from the sbatch folder,
# e.g. `cd sbatch && sbatch generate_256x256.sbatch`.
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
while [[ ! -f gen_images.py && "$PWD" != / ]]; do cd ..; done

conda activate styleswin

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# H200 (Hopper sm_90) + persistent kernel cache for fast startup.
# Use the system CUDA toolkit (provides nvcc to JIT-build op/); derive CUDA_HOME
# from nvcc's location so it tracks the loaded module.
module load CUDA/13.1
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR="${HOME}/.cache/torch_extensions"
export CUDA_CACHE_PATH="${HOME}/.cache/cuda_cache"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1

nvidia-smi
nvcc --version

python gen_images.py \
  --network=./runs/wc-cv_h200/00000-styleswin-imagenet_9to4_256x256-gpus2-batch32-cond/best_model.pt \
  --outdir=./generated/256x256 \
  --trunc=0.7 \
  --classes 0,1,2 \
  --samples-per-class 100000 \
  --gpus 2 \
  --batch-gpu 32
