# StyleSwin-v2 (WC-Co fork)

[![CI](https://github.com/dkagramanyan/StyleSwin-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/dkagramanyan/StyleSwin-v2/actions/workflows/ci.yml)

This fork (`StyleSwin-v2`) specialises Microsoft's
[StyleSwin](https://github.com/microsoft/StyleSwin) (a Swin-transformer StyleGAN) for
generating **WC-Co microstructure SEM images** (the `imagenet_9to4` dataset, three grain
classes). It adopts the shared generative-model API convention ("v2 convention") so command
names, flags, checkpoint format and generated-artifact layout match the sibling repos
([san-v2](https://github.com/dkagramanyan/san-v2),
[DiffiT-v2](https://github.com/dkagramanyan/DiffiT-v2),
[edm2-v2](https://github.com/dkagramanyan/edm2-v2)), and its output feeds the
[wc_cv](https://github.com/dkagramanyan/wc_cv) angle pipeline with zero conversion.

Relative to upstream StyleSwin it adds **class-conditional generation** over the 3 grain
classes, a `click` CLI, StyleGAN-style kimg/tick logging, and **[combra](https://github.com/dkagramanyan/combra)**
generative-quality metrics (FID / CMMD / FD-DINOv2 + angle-distribution) sharded across GPU
ranks each snapshot tick. The losses and regularizers (logistic GAN loss, R1, bCR) are
upstream's; every other difference is listed in
[Differences from upstream StyleSwin](#differences-from-upstream-styleswin). Full API notes
live on the wc_cv docs site (the `models_api` and `styleswin` pages).

## Conditioning

Enable with `--cond True`; `n_classes` and `class_names` are read from the dataset's
`dataset.json` (`n_classes = 0` keeps the unconditional path):

- **Generator — san-v2 mapping conditioning.** The one-hot label is embedded to `style_dim`,
  2nd-moment-normalised alongside `z`, concatenated, and fed to the mapping MLP so the style
  `w` becomes class-dependent (`models/generator.py`).
- **Discriminator — projection (Miyato & Koyama).** The label embedding is projected onto the
  pre-logit feature and added to the logit (`models/discriminator.py`), inside StyleSwin's
  unchanged logistic loss. Fake labels default to the empirical class distribution
  (`--fake-label-sampling empirical`).

## Differences from upstream StyleSwin

Audited 2026-09-25 against [microsoft/StyleSwin](https://github.com/microsoft/StyleSwin)
`main` (`train_styleswin.py`, `models/`, README training commands). Each entry is marked
**improvement** (a deliberate change we consider better), **contract** (required by the
shared four-repo model API), or **adaptation** (forced by our data, hardware or software
stack).

| Area | Upstream | This fork | Kind |
|---|---|---|---|
| Class conditioning, G | unconditional | one-hot label → `EqualLinear` (lr multiplier 1.0) → PixelNorm, concatenated with PixelNorm(`z`); the first mapping layer takes 1024 → 512 inputs. The StyleGAN2-ADA `MappingNetwork.embed` scheme, as in san-v2 (`models/generator.py`) | adaptation (3 grain classes) |
| Class conditioning, D | unconditional | projection term (Miyato & Koyama 2018): ⟨embed(`c`), `h`⟩ / √C added to the logit, C the pre-logit feature width (`models/discriminator.py`) | adaptation |
| Batch | total 32 at 256 (8 GPUs × 4), 16 at 1024 (8 × 2) | 64 / 32 / 8 per GPU at 256 / 512 / 1024 (total 128 / 64 / 16 on 2 GPUs) | adaptation (2 × H200) |
| G channel multiplier at 256 | 2 | 1 (2 runs out of memory at 64 images per GPU) | adaptation |
| G EMA | fixed decay `0.5 ** (32 / 10k)` ≈ 0.9978 per step, whatever the batch | `0.5 ** (batch / 10k)`: a 10 kimg half-life at any batch (StyleGAN2-ADA style) | improvement |
| lr decay | G lr falls by a fixed step per iteration to reach 0 at `--iter`; D is reset to 4 × G under `--ttur` (dropping D's lazy-regularization factor, a ~6 % step up at the decay start), and without `--ttur` D subtracts G's decrement, so an unequal D lr never reaches 0 | both lrs are scaled by the same factor, so both reach 0 and keep their ratio; start given as a fraction of `--kimg` | improvement |
| 512 recipe | none (256 and 1024 only) | interpolated: decay start 0.7625, no bCR, as at 1024 | adaptation |
| D spectral norm | `--D_sn` in every published recipe | on in every preset (`--d-sn`), as upstream | — (same) |
| TF32 | not set (torch defaults: matmul TF32 off, cuDNN TF32 on) | on (`--tf32 True`) | improvement (speed) |
| Data sampler | `DistributedSampler.set_epoch` never called: every epoch repeats the same order | `set_epoch` called each pass | improvement (bug fix) |
| Gradient accumulation / mixed precision | none | opt-in `--grad-accum`, `--precision fp16/bf16` (defaults 1 / fp32 train as upstream) | contract |
| Real-image augmentation | opt-in `--use_flip` horizontal flip, used only in the LSUN Church recipe | `--augment True` (default): each training item gets a uniformly random dihedral transform (rot90 by 0/90/180/270° × horizontal flip with p = 0.5) on the raw uint8 image in the loader; the combra reference is built with `dihedral=True` to match | adaptation (SEM microstructures have no preferred orientation; the dataset stores 1080 originals, not 8 stored orientations each) |
| Library compatibility | `torch.meshgrid` without `indexing`, `timm.models.layers`, `torch.cuda.amp.custom_fwd` | `indexing='ij'`, `timm.layers`, `torch.amp.custom_fwd(device_type='cuda')` — same results on current torch / timm | adaptation |
| Evaluation and logging | in-loop FID (`utils/fid_score.py`) against a folder of real images; wandb / TensorBoard losses; argparse CLI, resume from `--ckpt` | combra FID / CMMD / FD-DINOv2 + angle metrics, sharded over ranks; kimg/tick logging, `stats.jsonl` + TensorBoard, self-describing inference snapshots (spec §3–§7); `click` CLI, no resume | contract |

The bCR consistency-regularization flips (`utils/CRDiffAug.py`) are part of bCR's
augmentation set and unchanged from upstream.

## Installation

The `op/` CUDA ops (`fused_act`, `upfirdn2d`) are JIT-compiled by torch on first import, so the
env needs `nvcc` and `ninja`. `nvcc` comes from the system CUDA module; `ninja` — torch's build
backend — from conda (a pip `ninja` conflicts with conda's). torch comes from the CUDA wheel index:

```bash
conda create -n styleswin-v2 python=3.12 -y && conda activate styleswin-v2
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
conda install anaconda::ninja -y         # torch's JIT build backend
pip install -e .                         # base deps (console scripts, reads pyproject.toml)
pip install -e '.[combra]'               # optional: combra in-training metrics
```

**combra is a private repo**, so the `.[combra]` extra clones it over `git+https` and only
succeeds when authenticated to GitHub — sign in once with `gh auth login` (github.com → HTTPS)
and `pip` inherits its credential helper. The extra requests `combra[metrics]`, not bare
`combra`: since combra 0.5.0 the torch / `pytorch-fid` / `open-clip-torch` stack lives
behind that extra, and without it `combra_fid`, `combra_cmmd` and `combra_fd_dinov2` come
back `nan`. combra also floors Python at **3.12**, which is why this package does too.
The metrics pull InceptionV3 / CLIP / DINOv2 backbones on first use;
`bash download_models.sh` prefetches and caches them for offline nodes (wget/curl + git,
no Python; `MODEL_CACHE=/path bash download_models.sh` caches somewhere other than
`~/.cache`, then point the jobs at it with `TORCH_HOME=$MODEL_CACHE/torch` and
`HF_HOME=$MODEL_CACHE/huggingface`). StyleSwin itself has no pretrained weights.

## Data preparation

Build one ImageNet-style zip per resolution with `styleswin-prepare-data`. The class is the
top-level subfolder of each image, and the integer label is that class's **alphabetical** index;
class names are written into `dataset.json` (`class_names`), and grayscale sources are converted
to RGB at build time:

```bash
styleswin-prepare-data convert --source /path/to/wc_co_source \
    --dest ./datasets/imagenet_9to4_1024x1024_256x256.zip --transform center-crop --resolution 256x256
```

The training sets are `imagenet_9to4_1024x1024_<r>x<r>.zip` (r = 256, 512, 1024): **1080 unique
WC-Co crops**, 360 per class (`class_names` `['Ultra_Co25', 'Ultra_Co11', 'Ultra_Co6_2']`).
They replace the earlier 8640-image archives, which stored each crop in all 8 dihedral
orientations; that augmentation is now applied on the fly (`--augment`, see Training). One
epoch is therefore 1080 images. With the default 2 GPUs the `DistributedSampler` gives each
rank 540 images and `drop_last` keeps whole batches: 8 steps per epoch at 256 (64 per GPU),
16 at 512 (32), 67 at 1024 (8); the few dropped images differ every epoch because the
shuffle is re-seeded per epoch.

## Training

```bash
# conditional, 2 GPUs, combra metrics on every snapshot tick
styleswin-train --outdir=./runs/wc-cv \
    --cfg styleswin-256 --data=./datasets/imagenet_9to4_1024x1024_256x256.zip \
    --gpus=2 --cond True --combra-metrics True --kimg 25000 --snap 50
```

`--cfg styleswin-{256,512,1024}` selects a per-resolution preset (each resolution is trained
independently); `--precision {fp32,fp16,bf16}`, `--tf32/--bench` and `--grad-accum` follow
the shared CLI.

`--augment True` (default) applies a uniformly random element of the dihedral group to
every training image: a rotation by k × 90° (k = 0–3) and a horizontal flip with
probability 0.5, on the raw uint8 image in the training loader, before ImageNet
normalization. The draws come from the per-rank torch RNG seeded from `--seed`, so a run
is reproducible for a fixed `--seed` / `--gpus` / `--workers`. Images must be square. Only
the training loader augments: the combra reference, the `reals.png` grid and
`styleswin-eval` read the stored images, and the reference is precomputed with
`dihedral=True` so the metrics compare against the augmented distribution the generator
learns. `--augment False` feeds the reals as stored. Not available with `--lmdb`. The bCR
augmentations (`--bcr`) are separate and unchanged.

The presets take their optimizer recipe from the upstream StyleSwin FFHQ runs (paper
arXiv:2112.10762 appendix A / table 7, and the upstream README commands): G lr 5e-5, D lr
2e-4, R1 10 every 16 steps, and a linear decay of both learning rates to 0 over the last
part of `--kimg` (from 77.5 % at 256, 76.25 % at 512, 75 % at 1024; `--lr-decay`,
`--lr-decay-start`), and spectral norm in D (`--d-sn`). bCR is on at 256 only, as upstream.
Where they differ from upstream (batch, G channel multiplier, EMA, lr-decay details) is
listed in [Differences from upstream StyleSwin](#differences-from-upstream-styleswin).
The lr decay only takes effect if the run reaches the decay start, so size `--kimg` to the
job's time limit.

On the cluster, run the `sh/` scripts. Every setting of a run sits in the block at the top
of the train script: edit the block, or override one value for a single launch with an env
var (`KIMG=200 SNAP=2 bash sh/train_256.sh`).

```bash
sbatch --account=<proj> --partition=rocky --gpus=2 sh/train_256.sh   # or 512 / 1024
bash sh/train_256.sh                                                 # workstation: detaches, prints the log path
FOREGROUND=1 bash sh/train_256.sh                                    # workstation, stays attached
```

On a workstation the train script re-launches itself in its own session and returns at
once, so the run survives closing the terminal. Everything it prints goes to
`logs/styleswin-train_256-<date>-<time>.log`, with a `.pid` file beside it: follow the run
with `tail -f <log>`, stop it (every rank) with `kill -- -<pid>`. `FOREGROUND=1` and SLURM
jobs stay attached and copy the output to the same log. The log opens with a
`Run settings:` block — every setting, the git commit, host, date, `CUDA_VISIBLE_DEVICES`
and the full command.

Each run writes to `runs/.../NNNNN-<cfg>-gpus<G>-batch<B>[-desc]/` with `<runname>.log`,
`stats.jsonl`, TensorBoard events, `reals.png` / `fakes<kimg>.png` grids, and — the single
checkpoint kind — `styleswin-snapshot-<kimg>-inference.pt` (EMA-only weights + self-describing
metadata), written atomically every snapshot tick and always at the last tick. Retention
keeps the `--snapshot-keep-last` newest (default 1; `0` keeps all) plus the best snapshot by
each of `combra_fid`, `combra_fd_dinov2` and `combra_cmmd` (lower is better; ties keep the
earlier one, `nan` is skipped). One file can be best at several metrics, so the default
keeps at most 4 files; best snapshots are never pruned, and each snapshot tick logs a
`Best snapshots: ...` line naming them. There is **no resume**: size `--kimg` (or split
stages) to fit the job's time limit.

## Metrics (combra)

With `--combra-metrics` (default), every snapshot tick scores `G_ema` against the whole training
set — **sharded across ranks**: each rank generates its slice of a fixed `--num-fid-samples`
(default 10 000) sample and extracts FID / CMMD / FD-DINOv2 features + pooled vertex angles,
gathered to rank 0 for the distances. Results are logged to TensorBoard **and** `stats.jsonl`
under `Metrics/combra_*` — `combra_fid`, `combra_cmmd`, `combra_fd_dinov2`,
`combra_fid_best`, the angle-density metrics, and `combra_num_fid_samples` recording the
sample count the run actually used. (Keys used to carry a literal `10k` suffix that never
tracked `--num-fid-samples`; they no longer do.) `styleswin-eval` scores a checkpoint
standalone and reproduces the training metrics: it reads the `augment` flag recorded in the
snapshot and builds the reference with the same `dihedral` setting (snapshots without the
flag predate `--augment` and were trained without it, so they get `dihedral=False`). It
also defaults `--seed` to the training `--seed` stored in the snapshot (0 for snapshots
that predate the key), since the eval latents, labels and capped reference subset derive
from it. The eval labels follow the reference's class mix (360 / 360 / 360 on the current archives). combra is optional; if missing, training warns at startup and continues.

## Generation

```bash
styleswin-gen-images --network=./runs/.../styleswin-snapshot-000500-inference.pt \
    --outdir=./generated --save-mode hdf5 \
    --classes Ultra_Co11,Ultra_Co25,Ultra_Co6_2 --samples-per-class 1000 \
    --trunc 0.7 --gpus 2 --batch-gpu 32
```

`--save-mode hdf5` (default) writes per-rank shards merged into `<desc>.h5` in the RankH5Writer
layout the wc_cv angle pipeline consumes (the merge hard-fails on any incomplete shard);
`--save-mode dir` writes `class_<c>/idx_<i>_seed_<s>.png` + a `classes.json` manifest. `--classes`
accepts names or indices. The `sh/generate_{256,512,1024}.sh` scripts wrap this per resolution.

### Class index → grain class

StyleSwin consumes the same `imagenet_9to4_*` archives as the sibling repos and takes their
labels **verbatim**. Under the alphabetical convention the indices map `0 → Ultra_Co11`,
`1 → Ultra_Co25`, `2 → Ultra_Co6_2`. **The legacy on-disk archives carry SAN's swapped order**
(`0 → Ultra_Co25`, `1 → Ultra_Co11`), so classify each checkpoint by the dataset path in its
`training_options.json` before assuming either convention. Newly built zips (via
`styleswin-prepare-data`) record `class_names`, which travel into every checkpoint and generated
h5, so new artifacts are self-describing.

## Development

Install the dev extra (ruff + pytest) and run the same checks as CI
(`.github/workflows/ci.yml` — a ruff lint job + CPU smoke tests on Python 3.11):

```bash
pip install -e '.[dev]'
ruff check .
pytest
```

The test suite is CPU-only: the `dataset_tool` and `RankH5` tests (`tests/`) run everywhere;
the CLI-contract smoke tests self-skip where there is no CUDA toolchain, since importing the
training/generation stack JIT-compiles the `op` extension.

## Citing StyleSwin

```
@misc{zhang2021styleswin,
      title={StyleSwin: Transformer-based GAN for High-resolution Image Generation},
      author={Bowen Zhang and Shuyang Gu and Bo Zhang and Jianmin Bao and Dong Chen and Fang Wen and Yong Wang and Baining Guo},
      year={2021},
      eprint={2112.10762},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

This code borrows heavily from [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)
and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer). We also thank the
contributors of [Positional Encoding in GANs](https://github.com/open-mmlab/mmgeneration),
[DiffAug](https://github.com/mit-han-lab/data-efficient-gans),
[StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN) and
[GIQA](https://github.com/cientgu/GIQA).

## License

The code in this repository is under the MIT license as specified by the LICENSE file.
