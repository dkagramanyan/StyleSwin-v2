# StyleSwin-v2 (WC-Co fork)

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
ranks each snapshot tick. The generator/discriminator **update math is unchanged** from
upstream — the conditioning and tooling wrap around it. Full API notes live on the wc_cv docs
site (the `models_api` and `styleswin` pages).

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

## Installation

The `op/` CUDA ops (`fused_act`, `upfirdn2d`) are JIT-compiled by torch on first import, so the
env needs `nvcc` and `ninja`. `nvcc` comes from the system CUDA module; `ninja` — torch's build
backend — from conda (a pip `ninja` conflicts with conda's). torch comes from the CUDA wheel index:

```bash
conda create -n styleswin python=3.12 -y && conda activate styleswin
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
conda install anaconda::ninja -y         # torch's JIT build backend
pip install -e .                         # base deps (console scripts, reads pyproject.toml)
pip install -e '.[combra]'               # optional: combra in-training metrics
```

**combra is a private repo**, so the `.[combra]` extra clones it over `git+https` and only
succeeds when authenticated to GitHub — sign in once with `gh auth login` (github.com → HTTPS)
and `pip` inherits its credential helper. The combra metrics pull InceptionV3 / CLIP / DINOv2
backbones on first use; `styleswin-download-models` prefetches and caches them for offline nodes.

## Data preparation

Build one ImageNet-style zip per resolution with `styleswin-prepare-data`. The class is the
top-level subfolder of each image, and the integer label is that class's **alphabetical** index;
class names are written into `dataset.json` (`class_names`), and grayscale sources are converted
to RGB at build time:

```bash
styleswin-prepare-data convert --source /path/to/wc_co_source \
    --dest ./datasets/imagenet_9to4_256x256.zip --transform center-crop --resolution 256x256
```

## Training

```bash
# conditional, 2 GPUs, combra metrics on every snapshot tick
styleswin-train --outdir=./runs/wc-cv \
    --cfg styleswin-256 --data=./datasets/imagenet_9to4_256x256.zip \
    --gpus=2 --cond True --combra-metrics True --kimg 25000 --snap 50
```

`--cfg styleswin-{256,512,1024}` selects a per-resolution preset (each resolution is trained
independently); `--precision {fp32,fp16,bf16}`, `--tf32/--bench`, `--grad-accum` and the single
`--mirror` loader-level flip follow the shared CLI. On the cluster, run the `sh/` scripts:

```bash
sbatch --account=<proj> --partition=rocky --gpus=2 sh/train_256.sh   # or 512 / 1024
bash sh/train_256.sh                                                 # same script on a workstation
```

Each run writes to `runs/.../NNNNN-<cfg>-gpus<G>-batch<B>[-desc]/` with `<runname>.log`,
`stats.jsonl`, TensorBoard events, `reals.png` / `fakes<kimg>.png` grids, and — the single
checkpoint kind — `network-snapshot-<kimg>-inference.pt` (EMA-only weights + self-describing
metadata), written atomically every snapshot tick and always at the last tick, pruned to
`--snapshot-keep-last`. There is **no resume**: size `--kimg` (or split stages) to fit the job's
time limit. Pick the best snapshot post-hoc from `stats.jsonl` (`Metrics/combra_fid10k`).

## Metrics (combra)

With `--combra-metrics` (default), every snapshot tick scores `G_ema` against the whole training
set — **sharded across ranks**: each rank generates its slice of a fixed `--num-fid-samples`
(default 10 000) sample and extracts FID / CMMD / FD-DINOv2 features + pooled vertex angles,
gathered to rank 0 for the distances. Results are logged to TensorBoard **and** `stats.jsonl`
under `Metrics/combra_*`. `styleswin-eval` scores a checkpoint standalone. combra is optional; if
missing, training warns at startup and continues.

## Generation

```bash
styleswin-gen-images --network=./runs/.../network-snapshot-000500-inference.pt \
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
