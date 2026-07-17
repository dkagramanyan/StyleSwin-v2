# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared HDF5 generation layout (``RankH5Writer``) + shard merge.

This is the on-disk contract the wc_cv angle pipeline consumes, identical across the
generative-model repos (the "generation contract"):

* Per-rank shard ``shards/rank_<NNN>.h5`` with, per class, a ``class_<c>`` group holding
  ``images`` (uint8 **NHWC**), ``seeds`` (int64) and a per-sample boolean ``written`` mask.
* Root attributes ``format = "generated_images_shard"`` and ``schema_version = 1`` so any
  model's output is sniffed identically; ``class_names`` is stamped as a root attribute and
  each ``class_<c>`` group carries its own ``class_name``.
* Every shard records a ``missing_count`` attribute; the merge into ``<desc>.h5``
  **hard-fails** while any shard's ``missing_count`` is nonzero, so a crashed generation
  run can never feed zero-filled (black) slots downstream.
"""

import glob
import os

import h5py
import numpy as np

H5_FORMAT = "generated_images_shard"
H5_SCHEMA_VERSION = 1

_STR_DT = h5py.string_dtype(encoding="utf-8")


def _class_name(class_names, c):
    return class_names[c] if class_names is not None and c < len(class_names) else str(c)


class RankH5Writer:
    """Preallocated per-rank shard writer.

    ``plan`` maps ``class_id -> planned_count`` for THIS rank. Datasets are preallocated
    and the ``written`` mask starts all-False, so an interrupted run closes with a nonzero
    ``missing_count`` and the merge refuses it.
    """

    def __init__(self, path, plan, resolution, class_names):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.f = h5py.File(path, "w")
        self.f.attrs["format"] = H5_FORMAT
        self.f.attrs["schema_version"] = H5_SCHEMA_VERSION
        self.f.attrs["class_names"] = np.array(list(class_names or []), dtype=_STR_DT)
        self._groups = {}
        self._pos = {}
        for c, count in plan.items():
            g = self.f.create_group(f"class_{c}")
            g.attrs["class_name"] = _class_name(class_names, c)
            g.create_dataset("images", shape=(count, resolution, resolution, 3), dtype=np.uint8)
            g.create_dataset("seeds", shape=(count,), dtype=np.int64)
            g.create_dataset("written", shape=(count,), dtype=bool, data=np.zeros(count, dtype=bool))
            self._groups[c] = g
            self._pos[c] = 0

    def write(self, class_id, images_nhwc_u8, seeds):
        g = self._groups[class_id]
        p = self._pos[class_id]
        k = len(seeds)
        g["images"][p:p + k] = images_nhwc_u8
        g["seeds"][p:p + k] = np.asarray(seeds, dtype=np.int64)
        g["written"][p:p + k] = True
        self._pos[class_id] = p + k

    def close(self):
        total_missing = 0
        for c, g in self._groups.items():
            missing = int((~g["written"][()]).sum())
            g.attrs["missing_count"] = missing
            total_missing += missing
        self.f.attrs["missing_count"] = total_missing
        self.f.close()
        return total_missing


def merge_shards(shard_dir, out_path, class_names):
    """Merge every ``rank_*.h5`` in ``shard_dir`` into ``out_path``.

    Hard-fails (RuntimeError) if any shard reports a nonzero ``missing_count``.
    """
    shard_paths = sorted(glob.glob(os.path.join(shard_dir, "rank_*.h5")))
    if not shard_paths:
        raise RuntimeError(f"no shards found in {shard_dir}")

    per_class_images = {}
    per_class_seeds = {}
    for sp in shard_paths:
        with h5py.File(sp, "r") as f:
            mc = int(f.attrs.get("missing_count", 0))
            if mc != 0:
                raise RuntimeError(
                    f"shard {os.path.basename(sp)} has missing_count={mc}; refusing to merge an "
                    "incomplete generation run")
            for key in f:
                if not key.startswith("class_"):
                    continue
                c = int(key.split("_")[1])
                per_class_images.setdefault(c, []).append(f[key]["images"][()])
                per_class_seeds.setdefault(c, []).append(f[key]["seeds"][()])

    with h5py.File(out_path, "w") as out:
        out.attrs["format"] = H5_FORMAT
        out.attrs["schema_version"] = H5_SCHEMA_VERSION
        out.attrs["class_names"] = np.array(list(class_names or []), dtype=_STR_DT)
        out.attrs["missing_count"] = 0
        for c in sorted(per_class_images):
            images = np.concatenate(per_class_images[c], axis=0)
            seeds = np.concatenate(per_class_seeds[c], axis=0)
            g = out.create_group(f"class_{c}")
            g.attrs["class_name"] = _class_name(class_names, c)
            g.attrs["missing_count"] = 0
            g.create_dataset("images", data=images)
            g.create_dataset("seeds", data=seeds)
            g.create_dataset("written", data=np.ones(len(seeds), dtype=bool))
    return out_path
