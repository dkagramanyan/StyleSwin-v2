# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared HDF5 generation layout (``RankH5Writer``) + shard merge.

This is the on-disk contract the wc_cv angle pipeline consumes, identical across the
generative-model repos (the "generation contract"):

* Per-rank shard ``shards/rank_<NNN>.h5`` with, per class, a ``class_<c>`` group holding
  ``images`` (uint8 **NHWC**), ``seeds`` (int64), ``indices`` (int64, each row's global
  sample index within its class) and a per-sample boolean ``written`` mask. The merge sorts
  each class by ``indices`` (and drops the dataset), so the merged file is identical at any
  world size.
* Root attributes ``format = "generated_images_shard"`` and ``schema_version = 1`` so any
  model's output is sniffed identically; ``class_names`` (mandatory), ``image_shape_hwc``
  and ``samples_per_class`` are stamped as root attributes and each ``class_<c>`` group
  carries its own ``class_name``, ``class_idx``, ``samples_per_class`` and
  ``image_shape_hwc``.
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


def _check_class_names(class_names, classes):
    if class_names is None:
        raise ValueError("class_names is required (got None)")
    for c in classes:
        if c >= len(class_names):
            raise ValueError(
                f"class_names has {len(class_names)} entries; missing a name for class {c}")


def resolve_checkpoint_class_names(class_names, n_classes, source):
    """Class names a checkpoint's outputs are stamped with (the §5 label contract).

    A conditional checkpoint (``n_classes > 0``) must carry real grain-class names —
    fabricating ``['0', '1', ...]`` would defeat downstream name-based matching, so it
    raises instead. An unconditional checkpoint has one anonymous pseudo-class and gets
    the single-entry ``['0']``.
    """
    if class_names:
        return list(class_names)
    if n_classes > 0:
        raise ValueError(
            f"checkpoint {source!r} carries no class_names metadata; refusing to fabricate "
            "numeric names for a conditional model (label contract, §5). Re-export the "
            "snapshot with class_names.")
    return ["0"]


class RankH5Writer:
    """Preallocated per-rank shard writer.

    ``plan`` maps ``class_id -> planned_count`` for THIS rank. Datasets are preallocated
    and the ``written`` mask starts all-False, so an interrupted run closes with a nonzero
    ``missing_count`` and the merge refuses it.
    """

    def __init__(self, path, plan, resolution, class_names, samples_per_class):
        _check_class_names(class_names, plan)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        image_shape_hwc = (int(resolution), int(resolution), 3)
        self.f = h5py.File(path, "w")
        self.f.attrs["format"] = H5_FORMAT
        self.f.attrs["schema_version"] = H5_SCHEMA_VERSION
        self.f.attrs["class_names"] = np.array(list(class_names), dtype=_STR_DT)
        self.f.attrs["image_shape_hwc"] = image_shape_hwc
        self.f.attrs["samples_per_class"] = int(samples_per_class)
        self._groups = {}
        self._pos = {}
        for c, count in plan.items():
            g = self.f.create_group(f"class_{c}")
            g.attrs["class_name"] = class_names[c]
            g.attrs["class_idx"] = int(c)
            g.attrs["samples_per_class"] = int(samples_per_class)
            g.attrs["image_shape_hwc"] = image_shape_hwc
            g.create_dataset("images", shape=(count, resolution, resolution, 3), dtype=np.uint8)
            g.create_dataset("seeds", shape=(count,), dtype=np.int64)
            g.create_dataset("indices", shape=(count,), dtype=np.int64)
            g.create_dataset("written", shape=(count,), dtype=bool, data=np.zeros(count, dtype=bool))
            self._groups[c] = g
            self._pos[c] = 0

    def write(self, class_id, images_nhwc_u8, seeds, indices):
        g = self._groups[class_id]
        p = self._pos[class_id]
        k = len(seeds)
        g["images"][p:p + k] = images_nhwc_u8
        g["seeds"][p:p + k] = np.asarray(seeds, dtype=np.int64)
        g["indices"][p:p + k] = np.asarray(indices, dtype=np.int64)
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

    Each class's rows are sorted by their global ``indices`` (dropped from the output),
    so the merged file is identical regardless of how generation was sharded.
    Hard-fails (RuntimeError) if any shard reports a nonzero ``missing_count``.
    """
    shard_paths = sorted(glob.glob(os.path.join(shard_dir, "rank_*.h5")))
    if not shard_paths:
        raise RuntimeError(f"no shards found in {shard_dir}")

    per_class_images = {}
    per_class_seeds = {}
    per_class_indices = {}
    image_shape_hwc = None
    samples_per_class = None
    for sp in shard_paths:
        with h5py.File(sp, "r") as f:
            # An absent attr means the writer never reached close() — the shard
            # is from a crashed run, not a complete one, so `.get(..., 0)` would
            # wave it through. The mask check below is the ground truth anyway,
            # since the merged `written` is synthesized all-True.
            if "missing_count" not in f.attrs:
                raise RuntimeError(
                    f"shard {os.path.basename(sp)} has no missing_count attr (writer never "
                    "closed); refusing to merge an incomplete generation run")
            mc = int(f.attrs["missing_count"])
            if mc != 0:
                raise RuntimeError(
                    f"shard {os.path.basename(sp)} has missing_count={mc}; refusing to merge an "
                    "incomplete generation run")
            image_shape_hwc = tuple(int(x) for x in f.attrs["image_shape_hwc"])
            samples_per_class = int(f.attrs["samples_per_class"])
            for key in f:
                if not key.startswith("class_"):
                    continue
                c = int(key.split("_")[1])
                unwritten = int(np.count_nonzero(~f[key]["written"][()]))
                if unwritten:
                    raise RuntimeError(
                        f"shard {os.path.basename(sp)} {key}: {unwritten} slot(s) never "
                        "written; refusing to merge an incomplete generation run")
                per_class_images.setdefault(c, []).append(f[key]["images"][()])
                per_class_seeds.setdefault(c, []).append(f[key]["seeds"][()])
                per_class_indices.setdefault(c, []).append(f[key]["indices"][()])

    _check_class_names(class_names, per_class_images)
    with h5py.File(out_path, "w") as out:
        out.attrs["format"] = H5_FORMAT
        out.attrs["schema_version"] = H5_SCHEMA_VERSION
        out.attrs["class_names"] = np.array(list(class_names), dtype=_STR_DT)
        out.attrs["image_shape_hwc"] = image_shape_hwc
        out.attrs["samples_per_class"] = samples_per_class
        out.attrs["missing_count"] = 0
        for c in sorted(per_class_images):
            images = np.concatenate(per_class_images[c], axis=0)
            seeds = np.concatenate(per_class_seeds[c], axis=0)
            indices = np.concatenate(per_class_indices[c], axis=0)
            order = np.argsort(indices, kind="stable")
            g = out.create_group(f"class_{c}")
            g.attrs["class_name"] = class_names[c]
            g.attrs["class_idx"] = int(c)
            g.attrs["samples_per_class"] = samples_per_class
            g.attrs["image_shape_hwc"] = image_shape_hwc
            g.attrs["missing_count"] = 0
            g.create_dataset("images", data=images[order])
            g.create_dataset("seeds", data=seeds[order])
            g.create_dataset("written", data=np.ones(len(seeds), dtype=bool))
    return out_path
