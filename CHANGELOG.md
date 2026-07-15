# Changelog

All notable changes to this fork (`StyleSwin-v2`, the WC-Co specialisation of StyleSwin)
are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Class-conditional generation (3 grain classes).** StyleSwin is unconditional
  upstream; this fork makes it conditional with two standard techniques, enabled by
  `--cond True` (`n_classes` read from the dataset's `dataset.json`; `n_classes = 0` keeps
  the unconditional path):
  - **Generator:** san-v2 mapping conditioning — the one-hot label is embedded, 2nd-moment
    normalised alongside `z`, concatenated, and fed to the mapping MLP (`models/generator.py`).
  - **Discriminator:** Miyato & Koyama projection — the label embedding is projected onto
    the pre-logit feature and added to the logit (`models/discriminator.py`), on StyleSwin's
    unchanged logistic loss (no SAN objective).
  - `--fake-label-sampling {empirical,uniform}` (default `empirical`) samples fake labels
    from the empirical class distribution so imbalanced classes are not over-represented.
- **san-v2-style `click` CLI** (`train.py`) with `--outdir/--data/--gpus/--batch-gpu/--cond/
  --kimg/--tick/--snap/--combra-metrics/--save-inference-only/--resume/--dry-run` plus
  StyleSwin's own model flags.
- **StyleGAN-style logging** via a **kimg/tick** loop (`training/training_loop.py`): per-run
  `log.txt`, `stats.jsonl`, TensorBoard events, and the `tick … kimg … sec/tick …` status
  line — matching san-v2. Vendored `dnnlib/` and `torch_utils/{training_stats,misc}.py`.
- **combra generative-quality metrics** each snapshot tick, **sharded across all GPU ranks**
  (FID / CMMD / FD-DINOv2 + angle-Wasserstein / bimodal-Gaussian), ported from san-v2. Best
  model selected by `combra_fid10k`. combra is an optional dependency; a startup warning is
  emitted if it is requested but missing.
- **ImageNet-style zip datasets** (`dataset/imagenet_dataset.py`, `dataset_tool.py`,
  `dataset_tool_for_imagenet.py`) yielding `(uint8 image, one-hot label)` — same format and
  label convention as san-v2.
- `gen_images.py` generate script (per-class output, multi-GPU sharding, truncation) and
  `sbatch/` train + generate scripts for **256 / 512 / 1024**.
- **Full metric/loss coverage in TensorBoard.** Beyond the losses/timing/resources and the combra
  suite, the loop now also logs the effective learning rates (`LearningRate/G`, `LearningRate/D`),
  the running best FID (`Metrics/combra_fid10k_best`), and the `G_ema` sample grid each snapshot
  (image tag `Fakes`, alongside the on-disk `fakes<kimg>.png`).
- **Per-resolution `--cfg` presets** (`styleswin-256/512/1024`) — a `RESOLUTION_CONFIGS` dict in
  `train.py` bundling the per-resolution knobs (batch size, `enable_full_resolution`, channel
  multipliers, lr, R1). `--batch-gpu` is now optional when a preset supplies it; explicit CLI
  flags still override the preset, and `--cfg` cross-checks its resolution against the `--data`
  zip.

### Fixed
- **`g_ema` is now synchronised across ranks.** It was initialised from each rank's own random
  weights *before* the DDP broadcast; because combra generation is sharded per rank over each
  rank's `g_ema`, the metric image set mixed divergent EMAs early in training. It is now copied
  from the post-broadcast weights (`training/training_loop.py`).
- **`DistributedSampler.set_epoch()` is now called each epoch** (`sample_data`), so the shard
  ordering varies epoch-to-epoch instead of repeating epoch 0's order forever.
- **EMA decay now scales with the real batch size** (`0.5 ** (batch_size / 10000)`); it was
  hardcoded to batch 32, mis-calibrating the EMA whenever total batch ≠ 32 (e.g. 512/1024).
- **The combra gate stays rank-uniform on reference-precompute failure** — a success flag is
  all-reduced so combra is disabled on all ranks together (avoids a divergent-gate deadlock and
  repeated per-tick failures).

### Changed
- `--save-inference-only True` now writes **only** the G_ema-only
  `network-snapshot-<kimg>-inference.pt` each snapshot tick and **skips** the full
  `network-snapshot-<kimg>.pt`; previously it wrote both. `best_model.pt` is still a full
  checkpoint (`training/training_loop.py`).
- `sbatch/train_*.sbatch` select the resolution via `--cfg styleswin-<res>` and reference the
  dataset by its real name `./datasets/imagenet_9to4_1024x1024_<res>.zip`.
- PyTorch install target is the CUDA 13.2 wheel index; `requirements.txt` drops the pinned
  `tensorflow==1.15.0` and `torch>=1.6.0`, replaces `sklearn` with the maintained deps, and
  adds `click`, `tensorboard`, `psutil`, `pillow`, `requests`, `pyspng`.
- Custom CUDA op decorators updated for the latest PyTorch: `torch.cuda.amp.custom_fwd/bwd`
  → `torch.amp.custom_fwd/bwd(device_type='cuda')` (`op/fused_act.py`).
- `torchvision.utils.save_image(..., range=...)` → `value_range=...` (`train_styleswin.py`).

### Preserved
- The generator/discriminator **update math is unchanged** from upstream StyleSwin
  (logistic + R1 losses, EMA accumulate) — conditioning and tooling wrap around it. The
  Swin blocks, window/shift logic, sinusoidal positional encoding, ToRGB and the wavelet
  discriminator are untouched. The original `train_styleswin.py` entry point still works
  (unconditional) for backward compatibility.
