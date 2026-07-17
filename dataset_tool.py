# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Build ZIP/PNG datasets for StyleSwin training.

Exposed as the ``styleswin-prepare-data`` console command, a click *group* whose
``convert`` subcommand turns a folder (or archive) of images into the training
zip.  It follows the shared model-API dataset contract:

* **Labels are the alphabetical index of the class subfolder** (Rule 1): an image
  at ``<class>/img.png`` gets the label ``sorted(class_names).index(<class>)``.  Any
  ``dataset.json`` in the *source* is ignored — the folder layout is the source of
  truth, so a rebuilt zip never inherits a stale/swapped label order.
* **Class names travel with the artifact** (Rule 2): the written ``dataset.json``
  carries ``"class_names"`` index-aligned with the integer labels, alongside the
  ``"labels"`` list.
* **RGB is fixed at build time**: grayscale (and RGBA) sources are converted to
  3-channel RGB here, once, so the training dataset class can assert 3 channels
  instead of converting per item at runtime.
"""

import functools
import gzip
import io
import json
import os
import pickle
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import click
import imageio.v2 as imageio
import numpy as np
import PIL.Image
from tqdm import tqdm

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def parse_tuple(s: str) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore


def to_rgb(arr: np.ndarray) -> np.ndarray:
    """Force any image array to 3-channel uint8 RGB (Rule: RGB fixed at build time).

    Grayscale (HW / HW1) is broadcast to 3 channels; RGBA is alpha-composited onto a
    white background; RGB passes through. This is the single grayscale/alpha conversion
    point — the training dataset class asserts 3 channels rather than converting again.
    """
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    if arr.shape[2] == 4:
        png = PIL.Image.fromarray(arr)
        png.load()  # required for png.split()
        background = PIL.Image.new('RGB', png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
        return np.array(background)
    return arr

#----------------------------------------------------------------------------

def _class_from_archpath(arch_fname: str) -> Optional[str]:
    """The top-level directory of an archive path is the class name; flat = unlabeled."""
    parts = arch_fname.replace('\\', '/').split('/')
    return parts[0] if len(parts) > 1 else None

#----------------------------------------------------------------------------

def open_image_folder(source_dir, *, max_images: Optional[int]):
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, source_dir).replace('\\', '/')
            img = imageio.imread(fname)
            yield dict(img=img, cls_name=_class_from_archpath(arch_fname))
            if idx >= max_idx - 1:
                break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]):
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = np.array(PIL.Image.open(file))  # type: ignore
                yield dict(img=img, cls_name=_class_from_archpath(fname))
                if idx >= max_idx - 1:
                    break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_lmdb(lmdb_dir: str, *, max_images: Optional[int]):
    import cv2  # pip install opencv-python # pylint: disable=import-error
    import lmdb  # pip install lmdb # pylint: disable=import-error

    with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
        max_idx = maybe_min(txn.stat()['entries'], max_images)

    def iterate_images():
        with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
            for idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.array(PIL.Image.open(io.BytesIO(value)))
                    yield dict(img=img, cls_name=None)
                    if idx >= max_idx-1:
                        break
                except:
                    print(sys.exc_info()[1])

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_cifar10(tarball: str, *, max_images: Optional[int]):
    images = []
    labels = []

    with tarfile.open(tarball, 'r:gz') as tar:
        for batch in range(1, 6):
            member = tar.getmember(f'cifar-10-batches-py/data_batch_{batch}')
            with tar.extractfile(member) as file:
                data = pickle.load(file, encoding='latin1')
            images.append(data['data'].reshape(-1, 3, 32, 32))
            labels.append(data['labels'])

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    images = images.transpose([0, 2, 3, 1]) # NCHW -> NHWC
    assert images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, cls_name=str(int(labels[idx])))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_mnist(images_gz: str, *, max_images: Optional[int]):
    labels_gz = images_gz.replace('-images-idx3-ubyte.gz', '-labels-idx1-ubyte.gz')
    assert labels_gz != images_gz
    images = []
    labels = []

    with gzip.open(images_gz, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    with gzip.open(labels_gz, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, cls_name=str(int(labels[idx])))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        return np.array(img)


    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    def center_crop_dhariwal(width, height, img):
        # ADM / guided-diffusion preprocessing (Dhariwal & Nichol): repeatedly BOX-halve
        # while the smaller side stays >= 2x the target, then BICUBIC to the target scale
        # and center-crop. Produces the crop used by the shared eval protocol.
        pil = PIL.Image.fromarray(img, 'RGB')
        while min(pil.size) >= 2 * min(width, height):
            pil = pil.resize((pil.size[0] // 2, pil.size[1] // 2), resample=PIL.Image.BOX)
        scale = min(width, height) / min(pil.size)
        pil = pil.resize((round(pil.size[0] * scale), round(pil.size[1] * scale)), resample=PIL.Image.BICUBIC)
        arr = np.array(pil)
        cy = (arr.shape[0] - height) // 2
        cx = (arr.shape[1] - width) // 2
        return arr[cy : cy + height, cx : cx + width]

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    need_res = {'center-crop': center_crop, 'center-crop-wide': center_crop_wide,
                'center-crop-dhariwal': center_crop_dhariwal}
    if transform in need_res:
        if (output_width is None) or (output_height is None):
            error('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(need_res[transform], output_width, output_height)
    assert False, 'unknown transform'

#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        if source.rstrip('/').endswith('_lmdb'):
            return open_lmdb(source, max_images=max_images)
        else:
            return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if os.path.basename(source) == 'cifar-10-python.tar.gz':
            return open_cifar10(source, max_images=max_images)
        elif os.path.basename(source) == 'train-images-idx3-ubyte.gz':
            return open_mnist(source, max_images=max_images)
        elif file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            assert False, 'unknown archive type'
    else:
        error(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

@click.group()
def prepare_data():
    """StyleSwin dataset tools (``styleswin-prepare-data``)."""

#----------------------------------------------------------------------------

@prepare_data.command('convert')
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--max-images', help='Output only up to `max-images` images', type=int, default=None)
@click.option('--transform', help='Input crop/resize mode',
              type=click.Choice(['center-crop', 'center-crop-wide', 'center-crop-dhariwal']))
@click.option('--resolution', help='Output resolution (e.g., \'512x512\')', metavar='WxH', type=parse_tuple)
def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]]
):
    """Convert an image dataset into a StyleSwin training archive.

    The input dataset format is guessed from the --source argument:

    \b
    --source *_lmdb/                    Load LSUN dataset (unconditional)
    --source cifar-10-python.tar.gz     Load CIFAR-10 dataset
    --source train-images-idx3-ubyte.gz Load MNIST dataset
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Recursively load all images from dataset.zip

    For directory / zip sources, the **class is the top-level subfolder** of each
    image, and the integer label is that class's index in alphabetical order. Class
    names are written into the output ``dataset.json`` as ``"class_names"``. A flat
    source (no subfolders) produces an unconditional dataset.

    Grayscale and RGBA sources are converted to 3-channel RGB here at build time.

    Use --transform=center-crop / center-crop-wide / center-crop-dhariwal (with
    --resolution=WxH) to crop; output images must be square, power-of-two.
    """

    PIL.Image.init() # type: ignore

    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)

    if resolution is None:
        resolution = (None, None)
    transform_image = make_transform(transform, *resolution)

    dataset_attrs = None

    # (archive_fname, cls_name) per kept image; labels are assigned after the sweep so
    # the alphabetical class ordering is known before any index is written.
    entries = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        img = to_rgb(np.asarray(image['img']))
        img = transform_image(img)

        # Transform may drop images.
        if img is None:
            continue

        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {'width': img.shape[1], 'height': img.shape[0], 'channels': channels}
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if dataset_attrs['channels'] != 3:
                error('Output images must be 3-channel RGB after to_rgb()')
            if width != 2 ** int(np.floor(np.log2(width))):
                error('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
            error(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an uncompressed PNG.
        img = PIL.Image.fromarray(img, 'RGB')
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        entries.append((archive_fname, image['cls_name']))

    # Assign labels from the alphabetical class order (Rule 1) and record class_names (Rule 2).
    cls_names = sorted({c for _f, c in entries if c is not None})
    metadata = {'labels': None, 'class_names': None}
    if cls_names and all(c is not None for _f, c in entries):
        cls_to_idx = {name: i for i, name in enumerate(cls_names)}
        metadata['labels'] = [[f, cls_to_idx[c]] for f, c in entries]
        metadata['class_names'] = cls_names
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    prepare_data() # pylint: disable=no-value-for-parameter
