#!/bin/bash
# StyleSwin-v2 smoke driver — exercises the full CLI contract end-to-end on the GPU:
#   prepare-data  ->  train (tiny, 1 kimg)  ->  gen-images (hdf5 + dir)  ->  verify outputs
#
# This is the agent path documented in ../run-styleswin/SKILL.md. It reproduces exactly
# what was run in-container on an RTX 3090. Every step is a real console-script invocation,
# not a mock. Run from anywhere; it self-locates the repo root.
#
# Subcommands:
#   check   env + deps + `--help` (no GPU compute; fast)
#   smoke   full prepare->train->gen->verify  (default; ~3 min on a 3090)
#
# Override via env: ENVNAME, SRC (image source dir with class subfolders), WORK (scratch dir),
# CUDA_ROOT, ARCH (TORCH_CUDA_ARCH_LIST, e.g. 8.6 for a 3090 / Ampere, 9.0 for Hopper).
set -euo pipefail

# --- locate repo root (walk up to pyproject.toml) --------------------------------
cd "$(dirname "${BASH_SOURCE[0]}")"
while [[ ! -f pyproject.toml && "$PWD" != / ]]; do cd ..; done
REPO="$PWD"
[[ -f "$REPO/pyproject.toml" ]] || { echo "FATAL: repo root not found"; exit 1; }

# --- config (overridable) --------------------------------------------------------
ENVNAME="${ENVNAME:-styleswin}"
WORK="${WORK:-${TMPDIR:-/tmp}/styleswin_smoke}"
SRC="${SRC:-/home/david/mnt/ssd_2_sata/phd/datasets/san/o_bc_left_4x_1536_1024x1024_256x256_rgb_N360}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-13.0}"
ARCH="${ARCH:-8.6}"   # RTX 3090 = sm_86. NOT the sh/ scripts' default of 9.0 (Hopper).

# --- environment -----------------------------------------------------------------
source /home/david/anaconda3/etc/profile.d/conda.sh
conda activate "$ENVNAME"
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="$ARCH"
# The editable install maps ONLY the console-script modules; the sibling package dirs
# (dnnlib, op, models, training, dataset, ...) resolve only with the repo root on PYTHONPATH.
export PYTHONPATH="$REPO"
cd "$REPO"

say(){ echo; echo "=== $* ==="; }

cmd_check(){
  say "env"
  python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
  python -c "import numpy; assert numpy.__version__[0]=='1', 'numpy must be <2 (see SKILL Gotchas): '+numpy.__version__; print('numpy', numpy.__version__)"
  nvcc --version | tail -1
  say "styleswin-train --help (imports model -> op JIT-compiles on first run)"
  styleswin-train --help | tail -3
}

cmd_smoke(){
  rm -rf "$WORK"; mkdir -p "$WORK"
  say "1/4 prepare-data -> $WORK/smoke_256.zip"
  styleswin-prepare-data convert --source "$SRC" --dest "$WORK/smoke_256.zip" \
      --transform center-crop --resolution 256x256
  python -c "import zipfile,json; d=json.loads(zipfile.ZipFile('$WORK/smoke_256.zip').read('dataset.json')); print('class_names', d['class_names'], '| n', len(d['labels']))"

  say "2/4 train (1 kimg, batch 4, 1 GPU, combra off) -> checkpoint"
  styleswin-train --outdir="$WORK/runs" --cfg styleswin-256 --data="$WORK/smoke_256.zip" \
      --gpus 1 --cond True --combra-metrics False --batch-gpu 4 \
      --kimg 1 --tick 1 --snap 1 --workers 1
  CKPT="$(ls "$WORK"/runs/*/network-snapshot-000001-inference.pt)"
  echo "checkpoint: $CKPT"
  ls "$(dirname "$CKPT")"

  say "3/4 gen-images hdf5 (3 classes x 6)"
  styleswin-gen-images --network="$CKPT" --outdir="$WORK/generated" --save-mode hdf5 \
      --classes 0,1,2 --samples-per-class 6 --trunc 0.7 --gpus 1 --batch-gpu 4

  say "4/4 gen-images dir (class names, 2 each) -> viewable PNGs"
  styleswin-gen-images --network="$CKPT" --outdir="$WORK/generated_dir" --save-mode dir \
      --classes Ultra_Co11,Ultra_Co25,Ultra_Co6_2 --samples-per-class 2 --trunc 0.7 --gpus 1 --batch-gpu 4

  say "verify"
  python -c "
import h5py
f=h5py.File('$WORK/generated/network-snapshot-000001-inference.h5','r')
assert f.attrs['missing_count']==0, 'incomplete shards'
for c in ('class_0','class_1','class_2'):
    assert f[c+'/images'].shape==(6,256,256,3), f[c].name
print('h5 OK: 3 classes x (6,256,256,3), missing_count=0, class_names=', list(f.attrs['class_names']))
"
  echo "PASS. Sample grid: $(ls "$WORK"/runs/*/fakes000001.png) | PNGs: $WORK/generated_dir/class_*/"
}

case "${1:-smoke}" in
  check) cmd_check ;;
  smoke) cmd_check; cmd_smoke ;;
  *) echo "usage: driver.sh [check|smoke]"; exit 2 ;;
esac
