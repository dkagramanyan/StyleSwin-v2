# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""RankH5Writer / merge_shards contract (torch-free)."""

import os

import h5py
import numpy as np
import pytest

from utils.rank_h5 import H5_FORMAT, H5_SCHEMA_VERSION, RankH5Writer, merge_shards

CLASS_NAMES = ['Ultra_Co11', 'Ultra_Co25', 'Ultra_Co6_2']
RES = 8


def _write_shard(shard_dir, rank, plan, fill=True):
    w = RankH5Writer(os.path.join(shard_dir, f'rank_{rank:03d}.h5'), plan, RES, CLASS_NAMES)
    for c, n in plan.items():
        if fill:
            imgs = np.full((n, RES, RES, 3), c + 1, np.uint8)
            w.write(c, imgs, list(range(rank * 100, rank * 100 + n)))
    return w.close()


def test_merge_roundtrip(tmp_path):
    sh = tmp_path / 'shards'
    sh.mkdir()
    assert _write_shard(str(sh), 0, {0: 2, 1: 1}) == 0
    assert _write_shard(str(sh), 1, {1: 1, 2: 3}) == 0
    out = merge_shards(str(sh), str(tmp_path / 'merged.h5'), CLASS_NAMES)
    with h5py.File(out) as f:
        assert f.attrs['format'] == H5_FORMAT
        assert int(f.attrs['schema_version']) == H5_SCHEMA_VERSION
        assert int(f.attrs['missing_count']) == 0
        assert list(f.attrs['class_names']) == CLASS_NAMES
        # class counts: c0=2, c1=1+1=2, c2=3
        assert f['class_0']['images'].shape == (2, RES, RES, 3)
        assert f['class_1']['images'].shape == (2, RES, RES, 3)
        assert f['class_2']['images'].shape == (3, RES, RES, 3)
        assert f['class_0'].attrs['class_name'] == 'Ultra_Co11'
        assert bool(f['class_2']['written'][()].all())


def test_incomplete_shard_hard_fails(tmp_path):
    sh = tmp_path / 'shards'
    sh.mkdir()
    w = RankH5Writer(str(sh / 'rank_000.h5'), {0: 3}, RES, CLASS_NAMES)
    w.write(0, np.zeros((1, RES, RES, 3), np.uint8), [7])  # only 1 of 3
    assert w.close() == 2
    with pytest.raises(RuntimeError, match='missing_count'):
        merge_shards(str(sh), str(tmp_path / 'bad.h5'), CLASS_NAMES)
