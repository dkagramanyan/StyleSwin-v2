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
