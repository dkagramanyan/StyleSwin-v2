# Changelog

All notable changes to this fork (`StyleSwin-v2`, the WC-Co specialisation of StyleSwin)
are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0] - 2026-07-17

Adopt the shared generative-model API convention ("v2 convention"). Command names,
flags, checkpoint format and generated-artifact layout now match the sibling repos, so
anything learned on one transfers unchanged, and StyleSwin output feeds the wc_cv angle
pipeline with zero conversion.

### Added
- **Console scripts**: `styleswin-train`, `styleswin-gen-images`, `styleswin-eval`,
  `styleswin-prepare-data`, `styleswin-download-models` (`pyproject.toml`).
- **Generation contract (§4)** — `gen_images.py` gains `--save-mode hdf5` (default): per-rank
  `shards/rank_<NNN>.h5` in the RankH5Writer layout (`class_<c>/images|seeds`, uint8 NHWC,
  `format="generated_images_shard"`/`schema_version=1`, per-sample `written` mask +
  `missing_count`) merged into `<desc>.h5`; the merge **hard-fails on any incomplete shard**.
  `--classes` accepts names or indices (validated against the checkpoint); `dir` mode writes
  `class_<c>/idx_<i>_seed_<s>.png` + a `classes.json` manifest (`utils/rank_h5.py`).
- **Standalone evaluator** `styleswin-eval` (`eval.py`) + a startup `combra_smoke_test`.
- **Precision scheme (§2)** — `--precision {fp32,fp16,bf16}` (autocast; GradScaler for fp16),
  `--tf32 True/False` (default `True`), `--bench True/False` (default `True`).
- **`--grad-accum`** (default 1): total batch = `batch_gpu × gpus × grad_accum`.
- **`--num-fid-samples` / `--combra-ref-count`** eval knobs; a capped reference is a **seeded
  random subset**, never the first N.
- **Label contract (§5)** — the dataset tools derive labels from the **alphabetical** class
  order and write `class_names`; names travel into every checkpoint (`arch` + metadata) and
  every generated h5 / `classes.json`; the dataset exposes `class_names`.
- **Grid contract (§7)** — `reals.png` + `fakes_init.png` and a class-sorted, resolution-adaptive
  fixed-latent sample grid.
- **Infrastructure (§10)** — `.github/workflows/ci.yml` (ruff + CPU smoke tests), `tests/`,
  ruff/pytest config in `pyproject.toml`, full `.gitignore` template, `h5py`/`imageio` deps,
  `.[dev]` extra.
- **`sh/` launch scripts** (`train_{256,512,1024}.sh`, `generate_*.sh`): environment + one
  console call, no hardcoded home paths / hosts / account IDs, HF-offline set.

### Changed (breaking)
- **Checkpoint contract (§3)** — exactly one artifact kind: `network-snapshot-<kimg>-inference.pt`
  (EMA-only weights + self-describing metadata `{n_classes, resolution, class_names, cur_nimg,
  arch}`), written **atomically** (`tmp` + `os.replace`) every snapshot tick **and always at the
  last tick**, pruned to `--snapshot-keep-last`. Removed `--resume`, `--save-inference-only`, the
  rolling `network-snapshot-latest.pt`, and `best_model.pt` / `best_nimg.txt`. Interrupted runs
  can no longer be continued — size `--kimg` (or split stages) to fit the job's time limit.
- **`--use-flip` merged into `--mirror`**, now a **loader-level** stochastic per-item flip; the
  dataset (and thus the combra reference) is never flip-doubled.
- Run-dir name is `<id>-<cfg>-gpus<G>-batch<B>[-desc]` (total batch `B`, no dataset name spliced in).
- TensorBoard global step is `cur_nimg` (was kimg); `stats.jsonl` now mirrors `Metrics/combra_*`.
- Checkpoint metadata key `size` → `resolution`; `.pt` state dicts only.
- `requires-python` drops the `<3.14` cap, floors `>=3.10`.

### Fixed
- combra `Metrics/*` are mirrored into `stats.jsonl` (were TensorBoard-only), enabling post-hoc
  best-snapshot selection; the metrics row is **cleared when an eval tick fails** so a failed
  eval never re-logs the previous tick's values at the new step.
- Build-time grayscale→RGB in the dataset tools + a runtime 3-channel assert in the dataset class
  (a grayscale zip no longer trains as a silently tinted RGB image).
- The eval/grid latents derive from `--seed` alone (not `seed × gpus`), and the
  `DistributedSampler` is seeded from `--seed`, so the same command+seed is reproducible at any
  `--gpus`.

### Removed
- Fork leftovers: `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md`, `imgs/`, the legacy
  `train_styleswin.py` entry point, the `sbatch/` scripts, the reserved `--metrics` stub and the
  unused `restart_every` config key.

## [0.1.0]

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
- **Bounded checkpoint storage** so long runs no longer fill the disk (`training/training_loop.py`,
  `train.py`, `sbatch/train_*.sbatch`):
  - Each snapshot tick writes a small `network-snapshot-<kimg>-inference.pt` (G_ema-only — the
    part used for inference); `--snapshot-keep-last N` (default `3`) keeps only the most recent
    `N` and deletes the rest (`0` = keep all).
  - The full checkpoint is now a single `network-snapshot-latest.pt` **overwritten each tick**
    (never accumulates) instead of one `network-snapshot-<kimg>.pt` per tick; `--save-inference-only
    True` skips it entirely. `best_model.pt` is still a full checkpoint.
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
