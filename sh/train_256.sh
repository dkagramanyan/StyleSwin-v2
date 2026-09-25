#!/usr/bin/env bash
# StyleSwin -- train at 256x256.
#
# Workstation:  bash sh/train_256.sh                (detaches, prints the log path)
#               FOREGROUND=1 bash sh/train_256.sh   (stays attached; output still goes to the log)
# SLURM:        sbatch --account=<proj> --partition=<part> --nodes=1 --gpus=2 --cpus-per-task=8 --time=3-0:0 sh/train_256.sh
#
# Defaults target the production allocation: 2x H200 (sm_90), 8 CPUs, fixed seed 42.
#
# Every knob is in the run settings block below: edit it there, or override one for a
# single launch with an env var (DATA=<zip> GPUS=<n> bash sh/train_256.sh). Anything
# after the script name is appended to the command (e.g. `... --kimg 200 --snap 2` for a
# smoke run). No user homes, --nodelist or account IDs live here -- SLURM specifics come
# from the sbatch line (spec §9).
set -euo pipefail

# --- Run settings --------------------------------------------------------------
# NAME="${NAME:-default}": the default is used unless NAME is set in the environment.
CONDA_ENV="${CONDA_ENV:-styleswin-v2}"   # env name = repo name
CUDA_MODULE="${CUDA_MODULE:-CUDA/13.1}"  # system CUDA toolkit for the JIT-compiled ops
OUTDIR="${OUTDIR:-./runs}"
CFG="${CFG:-styleswin-256}"
DATA="${DATA:-./datasets/imagenet_9to4_1024x1024_256x256.zip}"
GPUS="${GPUS:-2}"
BATCH_GPU="${BATCH_GPU:-}"            # per GPU; empty = the --cfg preset's value
KIMG="${KIMG:-25000}"
SNAP="${SNAP:-50}"                    # ticks per snapshot + combra eval
KEEP_LAST="${KEEP_LAST:-1}"           # newest snapshots kept (bests are never pruned)
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-10000}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-3}"               # data-loader workers per rank

# --- Environment -------------------------------------------------------------
# Repo root: under SLURM the script runs from a spool copy, so walk up from the submit
# dir there and from this file's own location on a workstation.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
while [[ ! -f "$REPO_DIR/pyproject.toml" && "$REPO_DIR" != / ]]; do REPO_DIR="$(dirname "$REPO_DIR")"; done
[[ -f "$REPO_DIR/pyproject.toml" ]] || { echo "cannot find the repo root -- submit from inside the repo" >&2; exit 1; }
SELF="$(realpath "${BASH_SOURCE[0]}")"   # this file, for the detached re-launch below
cd "$REPO_DIR"

# --- Launch: detach and log ----------------------------------------------------
# On a workstation the script re-launches itself in its own session (setsid nohup) and
# returns at once: the run survives closing the terminal, and everything it prints
# goes to logs/<name>-<date>.log (with a .pid file beside it). FOREGROUND=1 keeps it
# attached; the output is still copied to the log. Under SLURM it never detaches (the
# job already runs unattended); the output goes both to the slurm .out and to the log.
RUN_NAME=styleswin-train_256
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
if [[ -z "${RUN_LOG:-}" ]]; then
    mkdir -p "$LOG_DIR"
    RUN_LOG="$LOG_DIR/$RUN_NAME-$(date +%Y%m%d-%H%M%S).log"
    export RUN_LOG
    if [[ -z "${SLURM_JOB_ID:-}" && "${FOREGROUND:-0}" != 1 ]]; then
        setsid nohup bash "$SELF" "$@" > "$RUN_LOG" 2>&1 < /dev/null &
        pid=$!
        echo "$pid" > "${RUN_LOG%.log}.pid"
        echo "Started $RUN_NAME in the background (pid $pid, its own process group)."
        echo "  log:    $RUN_LOG"
        echo "  follow: tail -f $RUN_LOG"
        echo "  stop:   kill -- -$pid"
        exit 0
    fi
    exec > >(tee -a "$RUN_LOG") 2>&1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
# Custom CUDA ops JIT-compile on first import against the system CUDA toolkit (module
# CUDA_MODULE); H200 (sm_90) arch by default.
command -v module >/dev/null 2>&1 && module load "$CUDA_MODULE" || true
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"   # H200 = sm_90
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${HOME}/.cache/cuda_cache}"
# Offline-cluster contract: backbones are prefetched once on a login node
# (bash download_models.sh); compute nodes never reach the network.
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
# Snapshots kept: the KEEP_LAST newest (default 1) plus the best by each of combra_fid,
# combra_fd_dinov2 and combra_cmmd -- at most 4 files with the default; bests are never pruned.
CMD=(
    styleswin-train
    --outdir "$OUTDIR"
    --cfg "$CFG"
    --data "$DATA"
    --gpus "$GPUS"
    ${BATCH_GPU:+--batch-gpu "$BATCH_GPU"}
    --cond True
    --kimg "$KIMG" --snap "$SNAP" --snapshot-keep-last "$KEEP_LAST"
    --combra-metrics True --num-fid-samples "$NUM_FID_SAMPLES"
    --seed "$SEED" --workers "$WORKERS"
    "$@"
)

# The log alone reproduces the run: settings, code version, machine, command.
commit="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
[[ -z "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null)" ]] || commit+=-dirty
echo "Run settings:"
for name in CONDA_ENV CUDA_MODULE OUTDIR CFG DATA GPUS BATCH_GPU KIMG SNAP KEEP_LAST \
            NUM_FID_SAMPLES SEED WORKERS; do
    echo "  $name=${!name}"
done
echo "  commit=$commit"
echo "  host=$HOSTNAME"
echo "  date=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  command=$(printf '%q ' "${CMD[@]}")"

"${CMD[@]}"
