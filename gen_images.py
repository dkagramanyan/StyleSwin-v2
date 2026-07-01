# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate images from a trained StyleSwin checkpoint.

Mirrors san-v2's ``gen_images.py`` API. For a conditional model the images are written
per class into ``class_<id>/<name>_<index>.png``; for an unconditional model they are
written flat. Pass ``--gpus`` to shard generation across GPUs (each process writes its
own slice of the work -- no cross-rank communication needed).

Example (conditional, 2 GPUs):

    python gen_images.py --network=./runs/00000-.../best_model.pt --outdir=./generated \\
        --trunc=0.7 --classes 0,1,2 --samples-per-class 1000 --gpus 2 --batch-gpu 32
"""

import os

import click
import numpy as np
import torch
import torchvision
from torch.nn import functional as F

from models.generator import Generator

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

#----------------------------------------------------------------------------

def parse_range(s):
    """'0,1,4-6' -> [0, 1, 4, 5, 6]."""
    if isinstance(s, (list, tuple)):
        return list(s)
    ranges = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            ranges.extend(range(int(a), int(b) + 1))
        else:
            ranges.append(int(part))
    return ranges


def _build_generator(ckpt, device):
    n_classes = int(ckpt.get('n_classes', 0))
    size = int(ckpt['size'])
    # Model hyperparameters that are not architecture-defining default to StyleSwin's
    # standard values; the checkpoint's state_dict pins the actual widths.
    G = Generator(size, style_dim=512, n_mlp=8, n_classes=n_classes).to(device)
    G.load_state_dict(ckpt['g_ema'])
    G.eval()
    return G, n_classes, size


@torch.no_grad()
def _mean_latent(G, n_classes, device, n=10000, batch=1000):
    ws = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        z = torch.randn(b, G.style_dim, device=device)
        if n_classes > 0:
            idx = torch.randint(0, n_classes, (b,), device=device)
            c = F.one_hot(idx, n_classes).float()
            zc = G.style[0](z)
            yc = G.style[0](G.class_embed(c))
            w = G.style[1:](torch.cat([zc, yc], dim=1))
        else:
            w = G.style(z)
        ws.append(w)
    return torch.cat(ws, dim=0).mean(0, keepdim=True)


def _denorm(img):
    mean = torch.tensor(_IMAGENET_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=img.device).view(1, 3, 1, 1)
    return (img * std + mean).clamp(0, 1)


@torch.no_grad()
def _generate_worker(rank, num_gpus, network, outdir, trunc, seed, classes,
                     samples_per_class, samples, batch_gpu):
    device = torch.device('cuda', rank)
    torch.cuda.set_device(rank)
    ckpt = torch.load(network, map_location=lambda s, loc: s)
    G, n_classes, _size = _build_generator(ckpt, device)

    trunc_latent = _mean_latent(G, n_classes, device) if trunc < 1 else None

    # Build the global work list of (class_or_None, index) items, then take this rank's
    # stride. Seeds are deterministic per item so reruns reproduce the same images.
    items = []
    if n_classes > 0:
        assert classes is not None, 'conditional model requires --classes'
        for cl in classes:
            for j in range(samples_per_class):
                items.append((cl, j))
    else:
        items = [(None, j) for j in range(samples)]

    my_items = items[rank::num_gpus]
    for start in range(0, len(my_items), batch_gpu):
        chunk = my_items[start:start + batch_gpu]
        gseeds = [seed + (cl or 0) * (samples_per_class or 0) + j for (cl, j) in chunk]
        z = torch.stack([torch.from_numpy(np.random.RandomState(s).randn(G.style_dim).astype(np.float32))
                         for s in gseeds]).to(device)
        if n_classes > 0:
            c = F.one_hot(torch.tensor([cl for (cl, _j) in chunk], device=device), n_classes).float()
            img = G(z, c, truncation=trunc, truncation_latent=trunc_latent)[0]
        else:
            img = G(z, truncation=trunc, truncation_latent=trunc_latent)[0]
        img = _denorm(img)
        for k, (cl, j) in enumerate(chunk):
            if cl is not None:
                d = os.path.join(outdir, f'class_{cl}')
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, f'class_{cl}_{j:06d}.png')
            else:
                path = os.path.join(outdir, f'{j:06d}.png')
            torchvision.utils.save_image(img[k], path, padding=0)
    if rank == 0:
        print(f'Done. Wrote images to {outdir}', flush=True)

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network', help='Checkpoint .pt (with g_ema)', metavar='PATH', required=True)
@click.option('--outdir', help='Where to write the generated images', metavar='DIR', required=True)
@click.option('--trunc', 'trunc', help='Truncation psi', type=float, default=1.0, show_default=True)
@click.option('--seed', help='Base random seed', type=int, default=42, show_default=True)
@click.option('--classes', help='Class indices for a conditional model, e.g. "0,1,2"', type=parse_range, default=None)
@click.option('--samples-per-class', help='Images per class (conditional)', type=int, default=1000, show_default=True)
@click.option('--samples', help='Total images (unconditional)', type=int, default=1000, show_default=True)
@click.option('--batch-gpu', help='Batch size per GPU', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--gpus', help='Number of GPUs', type=click.IntRange(min=1), default=1, show_default=True)
def main(network, outdir, trunc, seed, classes, samples_per_class, samples, batch_gpu, gpus):
    os.makedirs(outdir, exist_ok=True)
    args = (gpus, network, outdir, trunc, seed, classes, samples_per_class, samples, batch_gpu)
    if gpus == 1:
        _generate_worker(0, *args)
    else:
        torch.multiprocessing.set_start_method('spawn')
        torch.multiprocessing.spawn(fn=_generate_worker, args=args, nprocs=gpus)


if __name__ == '__main__':
    main()

#----------------------------------------------------------------------------
