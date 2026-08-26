# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""RankH5Writer / merge_shards contract (torch-free)."""

import os

import h5py
import numpy as np
import pytest

from utils.rank_h5 import (
    H5_FORMAT,
    H5_SCHEMA_VERSION,
    RankH5Writer,
    merge_shards,
    resolve_checkpoint_class_names,
)

CLASS_NAMES = ['Ultra_Co11', 'Ultra_Co25', 'Ultra_Co6_2']
RES = 8
SPC = 5


def _write_shard(shard_dir, rank, plan, fill=True):
    w = RankH5Writer(os.path.join(shard_dir, f'rank_{rank:03d}.h5'), plan, RES, CLASS_NAMES, SPC)
    for c, n in plan.items():
        if fill:
            imgs = np.full((n, RES, RES, 3), c + 1, np.uint8)
            w.write(c, imgs, list(range(rank * 100, rank * 100 + n)), list(range(n)))
    return w.close()


def _write_strided_shards(shard_dir, items, num_ranks):
    # Mimic gen_images: items = [(class, global_idx), ...], rank r gets items[r::num_ranks].
    for rank in range(num_ranks):
        my_items = items[rank::num_ranks]
        plan = {}
        for cl, _j in my_items:
            plan[cl] = plan.get(cl, 0) + 1
        w = RankH5Writer(os.path.join(shard_dir, f'rank_{rank:03d}.h5'), plan, RES, CLASS_NAMES, SPC)
        for cl, j in my_items:
            img = np.full((1, RES, RES, 3), cl + 1, np.uint8)
            w.write(cl, img, [1000 * cl + j], [j])  # seed encodes (class, index)
        assert w.close() == 0


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
        assert 'indices' not in f['class_0']  # working dataset, dropped from the merge


def test_incomplete_shard_hard_fails(tmp_path):
    sh = tmp_path / 'shards'
    sh.mkdir()
    w = RankH5Writer(str(sh / 'rank_000.h5'), {0: 3}, RES, CLASS_NAMES, SPC)
    w.write(0, np.zeros((1, RES, RES, 3), np.uint8), [7], [0])  # only 1 of 3
    assert w.close() == 2
    with pytest.raises(RuntimeError, match='missing_count'):
        merge_shards(str(sh), str(tmp_path / 'bad.h5'), CLASS_NAMES)


def test_parity_attrs_on_shard_and_merged(tmp_path):
    sh = tmp_path / 'shards'
    sh.mkdir()
    assert _write_shard(str(sh), 0, {0: 2, 1: 1}) == 0
    out = merge_shards(str(sh), str(tmp_path / 'merged.h5'), CLASS_NAMES)
    for path in (str(sh / 'rank_000.h5'), out):
        with h5py.File(path) as f:
            assert list(f.attrs['image_shape_hwc']) == [RES, RES, 3]
            assert int(f.attrs['samples_per_class']) == SPC
            for c in (0, 1):
                g = f[f'class_{c}']
                assert int(g.attrs['class_idx']) == c
                assert int(g.attrs['samples_per_class']) == SPC
                assert list(g.attrs['image_shape_hwc']) == [RES, RES, 3]


def test_merged_order_independent_of_world_size(tmp_path):
    # gen_images work list for 2 classes x SPC samples; rank r gets items[r::num_ranks].
    items = [(cl, j) for cl in (0, 1) for j in range(SPC)]
    merged = {}
    for num_ranks in (1, 2):
        sh = tmp_path / f'shards_{num_ranks}'
        sh.mkdir()
        _write_strided_shards(str(sh), items, num_ranks)
        merged[num_ranks] = merge_shards(str(sh), str(tmp_path / f'merged_{num_ranks}.h5'),
                                         CLASS_NAMES)
    with h5py.File(merged[1]) as f1, h5py.File(merged[2]) as f2:
        for c in (0, 1):
            np.testing.assert_array_equal(f1[f'class_{c}']['seeds'][()],
                                          f2[f'class_{c}']['seeds'][()])
            # 1-rank merge is already in global order, so both must be 0..SPC-1.
            np.testing.assert_array_equal(f1[f'class_{c}']['seeds'][()],
                                          1000 * c + np.arange(SPC))


def test_unclosed_or_lying_shard_hard_fails(tmp_path):
    # A shard whose writer never reached close() carries no missing_count attr;
    # and an attr claiming completeness must not outrank the written mask.
    sh = tmp_path / 'shards'
    sh.mkdir()
    assert _write_shard(str(sh), 0, {0: 2}) == 0
    with h5py.File(sh / 'rank_000.h5', 'r+') as f:
        del f.attrs['missing_count']
    with pytest.raises(RuntimeError, match='no missing_count'):
        merge_shards(str(sh), str(tmp_path / 'bad.h5'), CLASS_NAMES)
    with h5py.File(sh / 'rank_000.h5', 'r+') as f:
        f.attrs['missing_count'] = 0
        f['class_0']['written'][1] = False
    with pytest.raises(RuntimeError, match='never written'):
        merge_shards(str(sh), str(tmp_path / 'bad.h5'), CLASS_NAMES)


def test_missing_class_names_raises(tmp_path):
    with pytest.raises(ValueError, match='class_names'):
        RankH5Writer(str(tmp_path / 'rank_000.h5'), {0: 1}, RES, None, SPC)
    with pytest.raises(ValueError, match='class_names'):
        RankH5Writer(str(tmp_path / 'rank_001.h5'), {2: 1}, RES, ['only_one'], SPC)
    sh = tmp_path / 'shards'
    sh.mkdir()
    assert _write_shard(str(sh), 0, {0: 1}) == 0
    with pytest.raises(ValueError, match='class_names'):
        merge_shards(str(sh), str(tmp_path / 'merged.h5'), None)


def test_conditional_checkpoint_without_names_is_refused():
    # A conditional checkpoint with no class_names must raise, never fabricate
    # '0','1',... — the fallback made the writer's mandatory raise unreachable.
    with pytest.raises(ValueError, match='no class_names'):
        resolve_checkpoint_class_names(None, 3, 'ckpt.pt')
    with pytest.raises(ValueError, match='no class_names'):
        resolve_checkpoint_class_names([], 3, 'ckpt.pt')
    # Unconditional keeps the single-entry pseudo-class; real names pass through.
    assert resolve_checkpoint_class_names(None, 0, 'ckpt.pt') == ['0']
    assert resolve_checkpoint_class_names(CLASS_NAMES, 3, 'ckpt.pt') == CLASS_NAMES
