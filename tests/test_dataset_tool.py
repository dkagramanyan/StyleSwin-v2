# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""prepare-data convert: alphabetical labels + class_names + build-time RGB (torch-free)."""

import io
import json
import zipfile

import numpy as np
import PIL.Image
from click.testing import CliRunner

import dataset_tool


def _make_source(root):
    # Classes deliberately out of alphabetical order on disk; one grayscale image.
    for cls, n in [('Ultra_Co25', 2), ('Ultra_Co11', 2), ('Ultra_Co6_2', 1)]:
        (root / cls).mkdir(parents=True)
        for i in range(n):
            arr = (np.random.rand(64, 64, 3) * 255).astype('uint8')
            PIL.Image.fromarray(arr, 'RGB').save(root / cls / f'{i}.png')
    gray = (np.random.rand(64, 64) * 255).astype('uint8')
    PIL.Image.fromarray(gray, 'L').save(root / 'Ultra_Co11' / 'gray.png')


def test_convert_alphabetical_labels_and_rgb(tmp_path):
    src = tmp_path / 'src'
    _make_source(src)
    dest = tmp_path / 'out.zip'
    res = CliRunner().invoke(dataset_tool.prepare_data,
                             ['convert', '--source', str(src), '--dest', str(dest)])
    assert res.exit_code == 0, res.output

    with zipfile.ZipFile(dest) as z:
        meta = json.loads(z.read('dataset.json'))
        assert meta['class_names'] == ['Ultra_Co11', 'Ultra_Co25', 'Ultra_Co6_2']
        assert sorted({lab for _f, lab in meta['labels']}) == [0, 1, 2]
        assert len(meta['labels']) == 6
        # Every stored image is 3-channel RGB (grayscale converted at build time).
        img = np.array(PIL.Image.open(io.BytesIO(z.read(meta['labels'][0][0]))))
        assert img.ndim == 3 and img.shape[2] == 3


def test_flat_source_is_unconditional(tmp_path):
    src = tmp_path / 'flat'
    src.mkdir()
    for i in range(3):
        PIL.Image.fromarray((np.random.rand(64, 64, 3) * 255).astype('uint8'), 'RGB').save(src / f'{i}.png')
    dest = tmp_path / 'out.zip'
    res = CliRunner().invoke(dataset_tool.prepare_data,
                             ['convert', '--source', str(src), '--dest', str(dest)])
    assert res.exit_code == 0, res.output
    with zipfile.ZipFile(dest) as z:
        meta = json.loads(z.read('dataset.json'))
        assert meta['labels'] is None and meta['class_names'] is None


def test_max_images_covers_every_class(tmp_path):
    # A cap smaller than the source used to take images in sorted() order, so all
    # six came from Ultra_Co11 and class_names had one entry -- a single-class
    # dataset from a three-class source.
    src = tmp_path / 'src'
    for cls in ('Ultra_Co11', 'Ultra_Co25', 'Ultra_Co6_2'):
        (src / cls).mkdir(parents=True)
        for i in range(4):
            arr = (np.random.rand(64, 64, 3) * 255).astype('uint8')
            PIL.Image.fromarray(arr, 'RGB').save(src / cls / f'{i}.png')
    dest = tmp_path / 'out.zip'
    res = CliRunner().invoke(dataset_tool.prepare_data,
                             ['convert', '--source', str(src), '--dest', str(dest),
                              '--max-images', '6'])
    assert res.exit_code == 0, res.output

    with zipfile.ZipFile(dest) as z:
        meta = json.loads(z.read('dataset.json'))
        assert meta['class_names'] == ['Ultra_Co11', 'Ultra_Co25', 'Ultra_Co6_2']
        assert len(meta['labels']) == 6
        assert sorted(lab for _f, lab in meta['labels']) == [0, 0, 1, 1, 2, 2]


def test_max_images_below_class_count_warns(tmp_path):
    src = tmp_path / 'src'
    for cls in ('a', 'b', 'c'):
        (src / cls).mkdir(parents=True)
        PIL.Image.fromarray((np.random.rand(64, 64, 3) * 255).astype('uint8'), 'RGB').save(src / cls / '0.png')
    dest = tmp_path / 'out.zip'
    res = CliRunner().invoke(dataset_tool.prepare_data,
                             ['convert', '--source', str(src), '--dest', str(dest),
                              '--max-images', '2'])
    assert res.exit_code == 0, res.output
    assert 'smaller than the 3 classes' in res.output
