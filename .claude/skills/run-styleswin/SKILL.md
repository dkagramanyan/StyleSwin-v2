---
name: run-styleswin
description: Build, run, and drive StyleSwin-v2 (Swin-transformer GAN for WC-Co microstructures). Use when asked to run, start, smoke-test, build, or train StyleSwin, generate images from a checkpoint, prepare its dataset, or confirm a StyleSwin change works end-to-end on the GPU.
---

StyleSwin-v2 is a class-conditional **Swin-transformer StyleGAN** (no GUI) driven by a
`click` CLI (`styleswin-*` console scripts). Drive it with the committed
**`.claude/skills/run-styleswin/driver.sh`**, which runs the whole contract end-to-end on
the GPU: `prepare-data` → `train` (tiny, 1 kimg) → `gen-images` (hdf5 + dir) → verify.

All paths below are relative to `StyleSwin/`. Everything here was run in this container on
an **RTX 3090 (24 GB)** with the `styleswin` conda env.

## Run (agent path) — the driver

```bash
# fast: env + deps + `--help` (no GPU compute)
.claude/skills/run-styleswin/driver.sh check

# full end-to-end (~2m15s on a 3090): prepare-data -> train -> gen-images -> verify
.claude/skills/run-styleswin/driver.sh smoke
```

`smoke` ends with `PASS.` and prints the sample grid + generated-PNG paths. Artifacts land
in `$WORK` (default `/tmp/styleswin_smoke/`): `runs/*/network-snapshot-000001-inference.pt`
(+ `fakes000001.png`, `stats.jsonl`), `generated/*.h5`, `generated_dir/class_*/*.png`.

Override via env vars (see the header of `driver.sh`): `ENVNAME`, `SRC` (image source dir
with class subfolders), `WORK`, `CUDA_ROOT`, `ARCH`.

**Look at the output** — the trained grid is a real image but expect smooth gray blur: 1 kimg
is essentially untrained. It confirms the pipeline, not image quality.

```bash
# view a generated sample (or runs/*/fakes000001.png)
ls /tmp/styleswin_smoke/generated_dir/class_0/
```

## Prerequisites

- **GPU** — an NVIDIA GPU (verified RTX 3090, `sm_86`).
- **System CUDA toolkit with `nvcc`** — the `op/` ops (`fused_act`, `upfirdn2d`) JIT-compile
  on first model import. `nvcc` is at `/usr/local/cuda-13.0/bin` here (torch wheel is cu128;
  the version mismatch is fine for the JIT build). No `apt-get` packages needed.

## Setup

The `styleswin` conda env already exists on this box (torch 2.9.1+cu128 + ninja + CUDA,
cloned from the sibling `san` env). It has the repo installed editable and all deps. Verify:

```bash
source /home/david/anaconda3/etc/profile.d/conda.sh
conda activate styleswin
python -c "import torch,numpy; print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'| numpy',numpy.__version__)"
# -> torch 2.9.1+cu128 | cuda True | numpy 1.26.4
```

**Rebuilding the env from scratch** (network-restricted here — pip retries then succeeds):

```bash
conda create --clone san -n styleswin -y          # fast: reuses a working torch+ninja+CUDA
conda activate styleswin
pip install -e . --no-build-isolation --no-deps   # console scripts (skip PyPI build isolation)
pip install lmdb einops tensorboard wandb 'numpy<2'  # missing from the clone; numpy<2 is REQUIRED
pip install -U timm                                # clone had timm 0.4.12; code needs timm.layers (>=0.9)
```

The README's from-clean-machine recipe (`conda create -n styleswin python=3.12`,
`pip3 install torch torchvision --index-url .../cu132`, `pip install -e .`) was **not**
exercised — cloning `san` is the working path in this multi-env repo.

## Test

```bash
source /home/david/anaconda3/etc/profile.d/conda.sh && conda activate styleswin
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TORCH_CUDA_ARCH_LIST=8.6
cd StyleSwin && PYTHONPATH=$PWD pytest -q
# -> 7 passed  (dataset_tool + RankH5 + CLI-contract smoke; smoke tests JIT-compile op/)
```

## Run (human path)

The `sh/` scripts wrap one console call each for the cluster/workstation. On a 3090, run the
driver instead — the `sh/` scripts hardcode `TORCH_CUDA_ARCH_LIST=9.0` (Hopper) and
`--gpus 2`. To use one directly, override: `GPUS=1 TORCH_CUDA_ARCH_LIST=8.6 bash sh/train_256.sh`.

## Gotchas

- **`PYTHONPATH=$PWD` is mandatory.** The editable install maps only the 5 console-script
  modules (`train`, `gen_images`, …); the sibling package dirs (`dnnlib`, `op`, `models`,
  `training`, `dataset`) resolve only with the repo root on `PYTHONPATH`. Without it:
  `ModuleNotFoundError: No module named 'dnnlib'`. The driver sets it.
- **numpy must be < 2.** With numpy 2.x the startup self-check `_assert_norm_roundtrip`
  (`training/training_loop.py:132`) dies: `OverflowError: Python integer 256 out of bounds
  for uint8` (`np.arange(..., dtype=np.uint8) % 256`). Pin `numpy<2` (1.26.4 works). This is
  a latent bug for any fresh `pip install` that pulls numpy 2 — worth fixing upstream.
- **`ARCH`/`TORCH_CUDA_ARCH_LIST` = 8.6 for the 3090.** The `sh/` scripts default to `9.0`
  (Hopper); wrong arch makes the JIT `op` build target the wrong SM. Driver default is `8.6`.
- **`combra` is a private repo and is NOT installed here.** So run training with
  `--combra-metrics False`, and **`styleswin-eval` does not work** (`ModuleNotFoundError:
  No module named 'combra'` from `_combra_precompute_reference`). `--combra-metrics True` /
  eval need `pip install -e '.[combra]'` with GitHub auth.
- **`lmdb` is imported at module load** (`dataset/dataset.py:6`), not just for `--lmdb` — it
  must be installed even for the zip-dataset path.
- **`--max-images` on `prepare-data` fills classes in alphabetical order**, so a small cap
  yields only the first class. Omit it (or cap high) to get all classes into `class_names`.
- **Legacy on-disk `imagenet_9to4_*` zips have no `class_names`** and carry SAN's swapped
  label order. Build fresh zips with `styleswin-prepare-data` (writes `class_names`) for
  self-describing artifacts; the driver does this.

## Troubleshooting

- **`No module named 'dnnlib'`** — set `PYTHONPATH=$PWD` (see Gotchas).
- **`OverflowError: Python integer 256 out of bounds for uint8`** at train startup —
  numpy 2.x; `pip install 'numpy<2'`.
- **`No module named 'timm.layers'`** — old timm from the env clone; `pip install -U timm`.
- **`RuntimeError` / nvcc-not-found during first import** — `nvcc` not on PATH; export
  `CUDA_HOME=/usr/local/cuda-13.0` and prepend `$CUDA_HOME/bin` to PATH.
- **pip `ConnectionResetError` / `Could not find setuptools`** — network is flaky here; pip
  retries and usually succeeds. Use `--no-build-isolation` to reuse the env's setuptools.
