# TODO

Problems found while driving the CLI end-to-end on an RTX 3090 (2026-07-21). These are
real repo/packaging issues, not local env quirks. Env-setup notes live in
`.claude/skills/run-styleswin/SKILL.md`.

## Bugs

- [ ] **numpy 2.x crashes training at startup.** `_assert_norm_roundtrip`
  (`training/training_loop.py:132`) does `np.arange(3*4*4, dtype=np.uint8) % 256`, which
  raises `OverflowError: Python integer 256 out of bounds for uint8` on numpy >= 2. A fresh
  `pip install` resolves `numpy>=1.20` to numpy 2, so training is broken out of the box.
  Fix: drop the redundant `% 256` (`arange(48)` never exceeds 255), or use a fitting dtype.
  Pin `numpy<2` in `pyproject.toml` as a stopgap.

- [ ] **Console scripts fail standalone: `No module named 'dnnlib'`.** The editable install
  maps only the 5 `py-modules` (`train`, `gen_images`, …); the sibling package dirs
  (`dnnlib`, `op`, `models`, `training`, `dataset`, `torch_utils`, `utils`) are not exposed,
  so `styleswin-train` etc. fail on `import dnnlib` unless the repo root is on `PYTHONPATH`.
  The `sh/*.sh` launch scripts run `styleswin-train` from repo root but do NOT set
  `PYTHONPATH`, so they hit this too. Fix: make those dirs importable (packages + `find`
  config, or add repo root to the install), or set `PYTHONPATH` in the `sh/` scripts.

## Packaging

- [ ] **`timm` is unpinned but the code needs `timm.layers`** (`models/generator.py:8`),
  which only exists in timm >= 0.9. An older resolved timm (0.4.12) gives
  `No module named 'timm.layers'`. Add `timm>=0.9` to `pyproject.toml`.

- [ ] **`einops` is declared in `dependencies` but never imported** anywhere in the tree —
  dead dependency; remove it, or use it.

## UX

- [ ] **`styleswin-prepare-data --max-images N` fills classes alphabetically**, so a small
  cap yields images from only the first class (and `class_names` ends up with one entry).
  Consider sampling across classes, or document that `--max-images` is per-run-order.

- [ ] **`sh/*.sh` hardcode `TORCH_CUDA_ARCH_LIST=9.0`** (Hopper). On other GPUs (e.g. a 3090,
  `sm_86`) the JIT `op` build targets the wrong SM unless the caller overrides it. Consider
  deriving the arch from the detected device, or documenting the override.
