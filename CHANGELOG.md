# Changelog

All notable changes to this fork (`StyleSwin-v2`, the WC-Co specialisation of StyleSwin)
are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed
- **`sh/train_{256,512,1024}.sh` and `sh/generate_{256,512,1024}.sh` rewritten to
  the §9 launch-script shape shared by all four model repos.** SLURM-spool-safe
  repo-root discovery, `conda.sh` sourced before `conda activate` (a bare
  `conda activate` fails in a non-interactive job shell), the system-CUDA JIT
  toolchain with the module name overridable via `CUDA_MODULE`, the offline-hub
  contract, and one console-command call whose every knob is an env var with a
  default (`DATA`, `OUTDIR`, `GPUS`, `BATCH_GPU`, `NETWORK`, `CLASSES`,
  `SAMPLES_PER_CLASS`, `SEED`, …) plus `"$@"` passthrough. The dataset default is
  the shared `imagenet_9to4_1024x1024_<res>x<res>.zip` name; the per-resolution
  preset's `batch_gpu` applies unless `BATCH_GPU` is set.

### Removed
- **Dead upstream utilities**: `utils/fid_score.py` and `utils/inception.py` (the
  pytorch-fid port, superseded by combra's `fid_features`), `utils/visualizer.py`,
  `utils/distributed.py`, and `torch_utils.misc.get_ckpt_path` (which still named
  `network-snapshot.pkl`). Nothing imported any of them.

## [0.4.0] - 2026-08-27

### Fixed
- **Snapshots are named `styleswin-snapshot-<kimg:06d>-inference.pt`** (was
  `network-snapshot-…`), matching the `<model>-snapshot-…` pattern of the other
  three repos and this repo's own `gen_images` docstring. The prune glob, README,
  `eval.py`, `sh/generate_*.sh` and the run skill follow. Old `network-snapshot-*`
  files still load (the loader takes a path) but are no longer pruned by new runs.

- **The training loop fabricated `['0', '1', …]` class names into checkpoints**
  when the dataset zip carried none, so a conditional run produced snapshots whose
  fabricated names downstream code could not tell from real ones — one stage
  upstream of the `gen_images` fallback removed earlier. The loop now stores what
  `dataset.json` reports and the launcher refuses a conditional run whose zip has
  no `class_names` (rebuild it with `styleswin-prepare-data`); unconditional runs
  store `None`, which `gen_images` maps to the single `['0']` pseudo-class.

- **`tests/test_combra_contract.py` asserted combra symbols the training loop no
  longer imports.** Its `REQUIRED` list still named the eight feature / angle
  functions from before the sharded harness moved into combra, and never named
  `combra.metrics.distributed`'s `all_ranks_ok` / `distributed_metrics` /
  `gather_generated` / `precompute_reference` — the four symbols the loop actually
  depends on. That is the exact blind spot the test exists to close (combra 0.5.0
  removing three functions hid for a release the same way). It now pins
  `(module, name)` pairs for every combra import in the repo and the unguarded
  import block mirrors the loop's real imports.

- **A conditional checkpoint with no `class_names` is refused instead of getting
  fabricated numeric names.** `gen_images._load_checkpoint` fell back to
  `['0', '1', ...]` whenever the checkpoint carried no (or an empty) `class_names`
  list, which made the writer's mandatory-`class_names` `ValueError` unreachable
  from the CLI and stamped fabricated names into the h5 — defeating downstream
  name-based class matching (label contract, §5) at scoring time instead of
  failing before generation. The policy now lives in
  `utils.rank_h5.resolve_checkpoint_class_names`: real names pass through,
  a conditional checkpoint without them raises, and an unconditional checkpoint
  keeps the single-entry `['0']` pseudo-class.

- **`Timing/eval_sec` was reported on rank 0 only, desyncing the stats reduction.**
  `training_stats.report0` registers the counter *name* on whichever rank calls it —
  it discards non-zero ranks' values, not the registration — and `Collector.update()`
  all-reduces over the registered set, which the module's own docstring says must
  match across processes. The call sat inside `if rank == 0 and combra_results is not
  None`, so rank 0 carried a name no other rank had and the reduction disagreed on
  shape from the first snapshot tick of any multi-GPU run. It is now called by every
  rank, outside the guard.

- **Loading the reference slice happened before combra's rank handshake.** A decode
  error or `MemoryError` while stacking this rank's images raised outside
  `precompute_reference`, stranding every other rank in the precompute's `all_reduce`.
  The stack is now guarded and agreed through `all_ranks_ok` before any collective.

- **Eval failures printed on rank 0 only, hiding the rank that actually failed.** The
  common case is a non-zero rank OOMing, which produced a run reporting a failure with
  no error attached. Every rank now prints, tagged with its rank.

- **The merge trusted a `missing_count` attr that a crashed writer never stamps.**
  `merge_shards` read the attr with a default of 0, so a shard whose process died
  before `close()` — the one case the gate exists for — sailed through, and the
  merged `written` mask is synthesized all-True on top. An absent attr now refuses
  the merge outright, and every shard group's actual `written` mask is verified
  during the read, so the attr can no longer outrank the data.
- **The merged h5's row order depended on `--gpus`.** `merge_shards` concatenated each
  class's rows in shard-path order (rank-grouped), so the same command produced a
  differently-ordered `<desc>.h5` at every world size. Shards now carry a per-row
  `indices` dataset (the sample's global index within its class, threaded through from
  `gen_images`' strided split) and the merge sorts each class by it (stable argsort,
  `indices` dropped from the merged file) — the merged file is now byte-identical
  regardless of `--gpus`. Same pattern as edm2's `training/h5_writer.py`.
- **`styleswin-prepare-data --max-images N` filled classes alphabetically.** A cap
  truncated a sorted (therefore class-grouped) file list, so it took every image from
  the first class and `class_names` ended up with one entry. `stratified_subset` now
  spreads the cap round-robin across classes and warns when the cap is below the class
  count; applied to the folder, zip and ImageNet openers. The identical bug in edm2
  was fixed in the same pass.
- **`sh/*.sh` defaulted `TORCH_CUDA_ARCH_LIST` to 9.0 (Hopper)**, so the JIT `op`
  build targeted the wrong SM on any other GPU. All six scripts now derive it from
  `nvidia-smi --query-gpu=compute_cap`, falling back to 9.0 when nvidia-smi is absent
  (a login node) and still yielding to an explicit value.
- **combra is pinned to a tag (`@v0.10.0`) instead of tracking `main`.** Unpinned, every
  fresh env resolved whatever combra `main` was that day, so the FID / CMMD / FD-DINOv2 /
  angle numbers a run is judged on could change with no signal and no record. combra
  0.8.0 also stamps `combra/version` into this run's TensorBoard HPARAMS, so the metric
  code behind a run is now recoverable from its log. Local development is unaffected --
  the env's editable combra install shadows the URL.
- **3 of the 5 console scripts were unusable outside the repo root.** `pyproject.toml`
  declared the five `py-modules` but had no `[tool.setuptools.packages.find]`, so the
  sibling package dirs the scripts import were never installed: `styleswin-train` died
  on `No module named 'dnnlib'`, `styleswin-eval` and `styleswin-gen-images` on
  `No module named 'models'`. The `sh/*.sh` launchers do not set `PYTHONPATH`, so they
  hit it too. The find block now mirrors san-v2 and edm2.
- **Console scripts are now covered by a packaging test** (`tests/test_entry_points.py`).
  It launches every entry point declared in `[project.scripts]` with `--help` from a
  temp cwd, which is the only way to see this class of bug: pytest runs with the repo
  root on `sys.path`, so an in-repo test passes while the installed script is broken.
  Confirmed to fail against the pre-fix packaging before being kept.
- **`stats.jsonl` rows are built by a testable function**, and a new
  `tests/test_stats_contract.py` feeds a real row to `combra.metrics.load_fid_by_kimg`.
  The reader was only ever tested against a synthetic flat row, so nothing checked the
  producer.
- **The §7 logging contract is now asserted** (`tests/test_logging_contract.py`).
  Thirteen scalar keys had drifted across the four repos; nothing failed because
  nothing checked. See below for this repo's share.

### Removed
- **`todo.md`.** Every item in it was closed, so the file said nothing a reader
  needed; the fixes are described in this changelog instead.

### Changed
- **Generated-h5 metadata parity with san-v2 / DiffiT-v2** (`utils/rank_h5.py`): the
  shard and merged roots now stamp `image_shape_hwc` and `samples_per_class`, and every
  `class_<c>` group stamps `class_idx`, `samples_per_class` and `image_shape_hwc` —
  the exact attribute names the sibling repos write.
- **`class_names` is mandatory** in `RankH5Writer` and `merge_shards`: a missing or
  too-short name list raises `ValueError` instead of silently degrading to an empty
  `class_names` attribute / `str(c)` group names. Unconditional checkpoints still work:
  `gen_images` derives a single-entry list for pseudo-class 0 from the checkpoint
  metadata (falling back to `['0']` when the checkpoint predates `class_names`).
- **The sharded eval harness moved into combra** (`combra.metrics.distributed`). This
  repo kept only what is model-specific: producing a shard of generated images and the
  float->uint8 denormalisation. The four private copies had drifted three ways --
  `all_gather` vs `gather`, a failure flag or none, and a different
  `precompute_reference` signature in each.
- **The combra startup check is `self_test(image_metrics=True, strict=True, images=...)`.**
  A missing CLIP download previously surfaced only as a whole run logging `nan`.
- **Hyperparameters reach TensorBoard.** The resolved config is read back from
  `training_options.json` at the end of training and written to the HPARAMS tab with
  the run's final `Metrics/combra_fid_best`, so runs are comparable by configuration
  and not only by curve shape. Nothing logged them before.
- **§7 keys:** `Timing/eval_sec` added; the `filename_suffix` format now matches the
  other three (`.<run-name>`).
- **The `op/` CUDA extensions JIT-build on first use, not at import.** Building at
  import meant `styleswin-train --help` could not run without ninja/nvcc, which is why
  three CLI-contract tests failed on a CPU-only runner. They pass now.

- **The combra contract test fed a unimodal sample to a bimodal-fit metric.**
  `test_angle_metrics_run_on_pooled_angles` drew two near-identical normals
  (mu 120 and 126), so the second Gaussian had no mode to sit on. combra now
  reports that as `nan` rather than dividing by the phantom, which turned the
  assertion red. The fixture is now genuinely bimodal (a 70/30 mixture at
  100 deg and 240 deg), which is what a WC-Co vertex-angle distribution
  actually looks like.
- **`scipy.linalg.sqrtm(..., disp=False)` raises under SciPy >= 1.18**, which
  removed the `disp` parameter. Fixed in `utils/fid_score.py`. Calling `sqrtm(X)` without `disp` returns
  the matrix alone on every SciPy version, so the fix is version-agnostic. This
  surfaced when the environment moved to SciPy 1.18 (see below); before that the
  call would have failed at runtime the moment anyone upgraded.

### Changed
- **The conda environment is now `styleswin-v2`** (Python 3.12, torch 2.13+cu130,
  numpy 2.5, SciPy 1.18), rebuilt alongside the previous `styleswin` env rather
  than replacing it. `requires-python` has said `>=3.12` since the v2 convention
  landed, but the working env was still 3.11 — so `pip install -e .` could not
  succeed, which is why the console scripts were missing and combra was absent.
  README and `sh/` launch scripts point at the new name.
- **CI installs combra and arms the contract test.** `tests/test_combra_contract.py`
  is entirely `skipif(not combra_installed)`, and no CI job installed combra, so the
  file could go green by doing nothing. CI now installs combra when a `COMBRA_TOKEN`
  secret is present and sets `COMBRA_REQUIRED=1`; a new always-on test fails if
  combra is missing under that flag.

## [0.3.0] - 2026-08-18

Repairs the combra integration, fixes two crashes on the modern Python/numpy floor,
and closes the remaining logging-contract gaps.

### Fixed
- **combra metrics were silently disabled.** `_combra_eval_distributed` imported
  `angle_density_metrics_from_pooled`, `fid_from_features` and
  `fd_dinov2_from_features`, which combra removed in 0.5.0, plus `combra_smoke_test`,
  which it renamed to `self_test`. The `except` around the per-tick eval swallowed the
  `ImportError` and printed "metric evaluation failed", so `--combra-metrics True`
  produced nothing at all. Now imports `frechet_from_features` (one helper for both
  Fréchet metrics) and `self_test`; combra >= 0.7.0 restores
  `angle_density_metrics_from_pooled`.
- **`[combra]` installed a combra with no metric backends.** The extra pulled bare
  `combra`; since combra 0.5.0 the torch / `pytorch-fid` / `open-clip-torch` stack is
  behind `combra[metrics]`, so FID / CMMD / FD-DINOv2 would have returned `nan` even
  after the import fix. Now `combra[metrics] @ git+…`.
- **numpy >= 2 crashed training at startup.** `_assert_norm_roundtrip` built its
  fixture with `np.arange(48, dtype=np.uint8) % 256`, and numpy 2 raises
  `OverflowError` on the out-of-range `uint8` literal. `arange(48)` never exceeds 255,
  so the modulo was dead; it is gone. This also removes the `numpy<2` pin that was
  under consideration, which combra (`numpy>=2.4`) could never have satisfied.
- **`distutils` import broke on Python 3.12+.** `dnnlib/util.py` imported `strtobool`
  from `distutils`, removed from the standard library in 3.12 — the floor this release
  moves to — and available only through setuptools' own deprecated shim. Replaced with
  a local `_strtobool`.
- **Stale metric rows.** `stats_metrics` was cleared only inside the snapshot branch,
  so the ticks between snapshots re-emitted the previous evaluation's values at a new
  step. It is now cleared every tick.

### Changed
- **Metric keys lost the literal `10k`.** `Metrics/combra_fid10k` was emitted whatever
  `--num-fid-samples` said, so any chart built from it was mislabelled. Keys are now
  bare — `Metrics/combra_fid`, `combra_cmmd`, `combra_fd_dinov2`, `combra_fid_best` —
  and the count is logged once as `Metrics/combra_num_fid_samples`.
- **`stats.jsonl` carries `wall_time` and `datetime`**, as the logging contract
  requires; it previously wrote only `timestamp`.
- **Angle-extraction workers scale with the rank count** (`cpu_count // gpus`, capped
  at 32). Every rank asking for `min(32, cpu_count)` oversubscribed an 8-GPU node
  eightfold.
- `requires-python` raised to **3.12** to match combra.
- `timm` floored at **0.9**: `models/generator.py` imports `timm.layers`, which does
  not exist in older releases, so an unpinned resolve could install a timm that fails
  at import.

### Removed
- `einops` from `dependencies` — declared but imported nowhere in the tree.

### Added
- `tests/test_combra_contract.py` — asserts every combra symbol this repo imports
  actually exists. CPU-only, no GPU/dataset/network, so it runs in every CI job. This
  is the check whose absence let the breakage above survive a whole release.

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
