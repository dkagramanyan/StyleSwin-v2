"""--augment: the training-loader dihedral transform and its combra / eval plumbing."""

import inspect
import sys
import types

import numpy as np
import pytest

pytest.importorskip("torch")  # the training-loop module imports torch at module level


class _FakeSet:
    """Minimal ImageFolderDataset stand-in: uint8 CHW images + one-hot labels."""

    def __init__(self, n=4, res=5, shape=None):
        rng = np.random.RandomState(0)
        self.image_shape = list(shape or (3, res, res))
        self.images = [rng.randint(0, 256, self.image_shape, dtype=np.uint8) for _ in range(n)]
        self.labels = [np.eye(3, dtype=np.float32)[i % 3] for i in range(n)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx].copy(), self.labels[idx].copy()


def _d4(image):
    """The 8 dihedral transforms of a CHW image, written independently of _dihedral."""
    out = []
    for k in range(4):
        r = np.rot90(image, k, axes=(1, 2))
        out += [r, np.flip(r, axis=2)]
    return out


def _which(image, orig):
    matches = [i for i, t in enumerate(_d4(orig)) if np.array_equal(image, t)]
    assert len(matches) == 1, 'not exactly one dihedral transform of the original'
    return matches[0]


def test_eight_distinct_transforms():
    from training.training_loop import _dihedral

    img = _FakeSet().images[0]
    got = [_dihedral(img, k, f) for k in range(4) for f in (False, True)]
    assert len({t.tobytes() for t in got}) == 8
    assert sorted(_which(t, img) for t in got) == list(range(8))


def test_items_are_dihedral_and_keep_dtype_shape_label():
    import torch

    from training.training_loop import DihedralAugment

    base = _FakeSet()
    aug = DihedralAugment(base)
    assert len(aug) == len(base)
    torch.manual_seed(0)
    for idx in range(len(base)):
        for _ in range(20):
            image, label = aug[idx]
            assert image.dtype == np.uint8 and list(image.shape) == base.image_shape
            assert image.flags['C_CONTIGUOUS']
            _which(image, base.images[idx])
            np.testing.assert_array_equal(label, base.labels[idx])


def test_roughly_uniform_and_seeded():
    import torch

    from training.training_loop import DihedralAugment

    base = _FakeSet(n=1)
    aug = DihedralAugment(base)

    def draws(seed, n):
        torch.manual_seed(seed)
        return [_which(aug[0][0], base.images[0]) for _ in range(n)]

    n = 8000
    counts = np.bincount(draws(0, n), minlength=8)
    assert counts.min() > 0.85 * n / 8 and counts.max() < 1.15 * n / 8, counts
    assert draws(1, 50) == draws(1, 50)
    assert draws(1, 50) != draws(2, 50)


def test_non_square_refused():
    from training.training_loop import DihedralAugment

    with pytest.raises(ValueError, match='square'):
        DihedralAugment(_FakeSet(shape=(3, 4, 6)))


def test_loader_only_augments_when_enabled():
    from training import training_loop

    src = inspect.getsource(training_loop.training_loop)
    assert 'loader_set = DihedralAugment(training_set) if augment else training_set' in src
    # The reference / labels / reals grid read the unaugmented set.
    assert 'reference_u8_set = training_set' in src
    assert inspect.signature(training_loop.training_loop).parameters['augment'].default is False
    assert 'dihedral=augment' in src


@pytest.mark.parametrize('dihedral', [False, True])
def test_reference_call_passes_dihedral(monkeypatch, dihedral):
    import torch

    from training.training_loop import _combra_precompute_reference

    calls = []
    fake = types.ModuleType('combra.metrics.distributed')
    fake.all_ranks_ok = lambda ok, device, n: ok
    fake.precompute_reference = lambda u8, device, rank, ws, **kw: calls.append((u8.shape, kw)) or ({}, True)
    monkeypatch.setitem(sys.modules, 'combra.metrics.distributed', fake)

    ref_set = _FakeSet()
    out = _combra_precompute_reference(ref_set, [0, 1, 2, 3], torch.device('cpu'), 0, 1, dihedral=dihedral)
    assert out == ({}, True)
    assert calls == [((4, 3, 5, 5), {'dihedral': dihedral})]


@pytest.mark.parametrize('stored, expected', [({'augment': True}, True),
                                              ({'augment': False}, False),
                                              ({}, False)])  # pre---augment snapshot
def test_eval_reads_augment_from_snapshot(monkeypatch, tmp_path, stored, expected):
    from click.testing import CliRunner

    import dataset.imagenet_dataset
    import eval as eval_mod

    ckpt = dict(stored, n_classes=3, resolution=5)
    arch = dict(style_dim=4)
    seen = {}

    class _RefSet(_FakeSet):
        resolution = 5

        def __init__(self, path, use_labels):
            super().__init__(n=6)

        def _get_raw_labels(self):
            return np.array([0, 1, 2, 0, 1, 2])

    def fake_ref(ref_set, ref_indices, device, rank, n, dihedral):
        seen['dihedral'] = dihedral
        return {}, True

    monkeypatch.setattr(eval_mod, '_load_checkpoint', lambda p: (ckpt, 3, 5, ['a', 'b', 'c'], arch))
    monkeypatch.setattr(eval_mod, '_build_generator', lambda *a: None)
    monkeypatch.setattr(dataset.imagenet_dataset, 'ImageFolderDataset', _RefSet)
    monkeypatch.setattr(eval_mod, '_combra_precompute_reference', fake_ref)
    monkeypatch.setattr(eval_mod, '_combra_eval_distributed', lambda *a: {'fid': 1.0})
    monkeypatch.setattr(eval_mod.torch.cuda, 'is_available', lambda: False)

    r = CliRunner().invoke(eval_mod.main, ['--network', 'x.pt', '--data', str(tmp_path),
                                           '--num-fid-samples', '6'])
    assert r.exit_code == 0, r.output
    assert seen['dihedral'] is expected


def test_combra_precompute_reference_accepts_dihedral():
    combra_dist = pytest.importorskip('combra.metrics.distributed')
    params = inspect.signature(combra_dist.precompute_reference).parameters
    assert 'dihedral' in params, 'combra too old: precompute_reference has no dihedral= (need v0.19.0)'
