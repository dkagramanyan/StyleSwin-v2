# TODO

Problems found while driving the CLI end-to-end on an RTX 3090 (2026-07-21). These are
real repo/packaging issues, not local env quirks. Env-setup notes live in
`.claude/skills/run-styleswin/SKILL.md`.

## Bugs

- [x] **numpy 2.x crashes training at startup.** Resolved 2026-08-18: the redundant
  `% 256` is gone (and no `numpy<2` pin was added -- combra requires `numpy>=2.4`).

  Original report: `_assert_norm_roundtrip`
  (`training/training_loop.py:132`) does `np.arange(3*4*4, dtype=np.uint8) % 256`, which
  raises `OverflowError: Python integer 256 out of bounds for uint8` on numpy >= 2. A fresh
  `pip install` resolves `numpy>=1.20` to numpy 2, so training is broken out of the box.
  Fix: drop the redundant `% 256` (`arange(48)` never exceeds 255), or use a fitting dtype.
  Pin `numpy<2` in `pyproject.toml` as a stopgap.

- [x] **Console scripts fail standalone: `No module named 'dnnlib'`.** Resolved
  2026-08-20: `pyproject.toml` gained a `[tool.setuptools.packages.find]` block, and
  `tests/test_entry_points.py` launches every entry point from a temp cwd so it cannot
  regress silently.

  Original report: The editable install
  maps only the 5 `py-modules` (`train`, `gen_images`, …); the sibling package dirs
  (`dnnlib`, `op`, `models`, `training`, `dataset`, `torch_utils`, `utils`) are not exposed,
  so `styleswin-train` etc. fail on `import dnnlib` unless the repo root is on `PYTHONPATH`.
  The `sh/*.sh` launch scripts run `styleswin-train` from repo root but do NOT set
  `PYTHONPATH`, so they hit this too. Fix: make those dirs importable (packages + `find`
  config, or add repo root to the install), or set `PYTHONPATH` in the `sh/` scripts.

## Packaging

- [x] **`timm` is unpinned but the code needs `timm.layers`** Resolved 2026-08-18:
  `timm>=0.9` is declared.

  Original report: (`models/generator.py:8`),
  which only exists in timm >= 0.9. An older resolved timm (0.4.12) gives
  `No module named 'timm.layers'`. Add `timm>=0.9` to `pyproject.toml`.

- [x] **`einops` is declared in `dependencies` but never imported** Resolved
  2026-08-18: removed.

  Original report: anywhere in the tree —
  dead dependency; remove it, or use it.

## UX

- [x] **`styleswin-prepare-data --max-images N` fills classes alphabetically.** Resolved
  2026-08-20: `dataset_tool.stratified_subset` spreads the cap round-robin across
  classes, warns when the cap is smaller than the class count, and leaves a
  single-class source alone. Applied to the folder, zip and ImageNet openers.
  **The same bug was in edm2** (`edm2/dataset_tool.py`) and was fixed there too.
  `tests/test_dataset_tool.py` pins it; confirmed failing against the pre-fix code.

  Original report: so a small
  cap yields images from only the first class (and `class_names` ends up with one entry).
  Consider sampling across classes, or document that `--max-images` is per-run-order.

- [x] **`sh/*.sh` defaulted `TORCH_CUDA_ARCH_LIST` to 9.0** (Hopper). Resolved 2026-08-20:
  all six scripts derive it from `nvidia-smi --query-gpu=compute_cap` (deduplicated,
  `;`-joined for multi-GPU), falling back to 9.0 when nvidia-smi is absent — e.g. on a
  login node. An explicit `TORCH_CUDA_ARCH_LIST` still wins. Verified all three paths on
  a 3090: derives `8.6`, falls back to `9.0` with nvidia-smi off `PATH`, honours `7.5`.

  Original report: On other GPUs (e.g. a 3090,
  `sm_86`) the JIT `op` build targets the wrong SM unless the caller overrides it. Consider
  deriving the arch from the detected device, or documenting the override.
