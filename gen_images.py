# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate images from a trained StyleSwin checkpoint (the generation contract).

Two output modes (``--save-mode``):

* ``hdf5`` (default): per-rank shards ``shards/rank_<NNN>.h5`` in the shared RankH5Writer
  layout, merged by rank 0 into ``<outdir>/<desc>.h5`` — exactly what the wc_cv angle
  pipeline consumes. The merge hard-fails if any shard is incomplete.
* ``dir``: ``class_<c>/idx_<i:06d>_seed_<s>.png`` plus a ``classes.json`` manifest.

``--classes`` accepts indices *or* class names (validated against the checkpoint's
``n_classes`` / ``class_names`` metadata). Per-image seeds are deterministic
(``seed = base + class*samples_per_class + idx``), so any subset reproduces in isolation.
``--gpus N`` self-spawns one worker per GPU; ``--batch-gpu`` is the only batching knob.

Example (conditional, 2 GPUs, hdf5):

    python gen_images.py --network=./runs/00000-.../styleswin-snapshot-000500-inference.pt \\
        --outdir=./generated --classes Ultra_Co11,Ultra_Co25 --samples-per-class 1000 \\
        --trunc 0.7 --gpus 2 --batch-gpu 32
"""

import json
import os

import click
import numpy as np
import torch
from torch.nn import functional as F

from models.generator import Generator
from utils.rank_h5 import RankH5Writer, merge_shards

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Architecture defaults for legacy checkpoints that predate the self-describing `arch`
# metadata block (§3). New checkpoints carry the exact values used at training time.
_ARCH_DEFAULTS = dict(style_dim=512, n_mlp=8, channel_multiplier=1, lr_mlp=0.01,
                      enable_full_resolution=8)

#----------------------------------------------------------------------------

def parse_classes(s, n_classes, class_names):
    """'0,1,4-6' or 'Ultra_Co11,Ultra_Co25' -> [validated int indices]."""
    if s is None:
        return None
    name_to_idx = {name: i for i, name in enumerate(class_names or [])}
    out = []

    def _resolve(token):
        token = token.strip()
        if token in name_to_idx:
            return name_to_idx[token]
        try:
            idx = int(token)
        except ValueError:
            raise click.ClickException(f'--classes: unknown class name {token!r}; '
                                       f'known names: {list(name_to_idx)}')
        if not (0 <= idx < n_classes):
            raise click.ClickException(f'--classes: index {idx} out of range for n_classes={n_classes}')
        return idx

    for part in str(s).split(','):
        if '-' in part and all(p.strip().isdigit() for p in part.split('-')):
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(_resolve(part))
    for idx in out:
        if not (0 <= idx < n_classes):
            raise click.ClickException(f'--classes: index {idx} out of range for n_classes={n_classes}')
    return out


def _load_checkpoint(network):
    ckpt = torch.load(network, map_location=lambda s, loc: s)
    n_classes = int(ckpt.get('n_classes', 0))
    resolution = int(ckpt.get('resolution', ckpt.get('size')))
    class_names = ckpt.get('class_names') or [str(i) for i in range(max(n_classes, 1))]
    arch = dict(_ARCH_DEFAULTS)
    arch.update(ckpt.get('arch', {}))
    return ckpt, n_classes, resolution, list(class_names), arch


def _build_generator(ckpt, n_classes, resolution, arch, device):
    G = Generator(resolution, arch['style_dim'], arch['n_mlp'],
                  channel_multiplier=arch['channel_multiplier'], lr_mlp=arch['lr_mlp'],
                  enable_full_resolution=arch['enable_full_resolution'],
                  n_classes=n_classes).to(device)
    G.load_state_dict(ckpt['g_ema'])
    G.eval()
    return G


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


def _denorm_to_uint8_nhwc(img_norm):
    # normalized float NCHW tensor -> uint8 NHWC numpy [0,255] (reverse of the ImageNet
    # preprocessing), the uint8 boundary format for both the h5 and the PNG writers.
    mean = torch.tensor(_IMAGENET_MEAN, device=img_norm.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=img_norm.device).view(1, 3, 1, 1)
    x = (img_norm * std + mean).clamp(0, 1)
    x = (x * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()  # NHWC


def _build_items(n_classes, classes, samples_per_class, samples):
    # (class_id, index) work list. Unconditional collapses onto pseudo-class 0.
    if n_classes > 0:
        return [(cl, j) for cl in classes for j in range(samples_per_class)]
    return [(0, j) for j in range(samples)]


def _seed_for(base, cl, j, samples_per_class):
    return base + cl * (samples_per_class or 0) + j

#----------------------------------------------------------------------------

@torch.no_grad()
def _generate_worker(rank, num_gpus, network, outdir, trunc, seed, classes,
                     samples_per_class, samples, batch_gpu, save_mode):
    device = torch.device('cuda', rank)
    torch.cuda.set_device(rank)
    ckpt, n_classes, resolution, class_names, arch = _load_checkpoint(network)
    G = _build_generator(ckpt, n_classes, resolution, arch, device)
    trunc_latent = _mean_latent(G, n_classes, device) if trunc < 1 else None

    items = _build_items(n_classes, classes, samples_per_class, samples)
    my_items = items[rank::num_gpus]

    writer = None
    if save_mode == 'hdf5':
        plan = {}
        for cl, _j in my_items:
            plan[cl] = plan.get(cl, 0) + 1
        writer = RankH5Writer(os.path.join(outdir, 'shards', f'rank_{rank:03d}.h5'),
                              plan, resolution, class_names)

    for start in range(0, len(my_items), batch_gpu):
        chunk = my_items[start:start + batch_gpu]
        gseeds = [_seed_for(seed, cl, j, samples_per_class) for (cl, j) in chunk]
        z = torch.stack([torch.from_numpy(np.random.RandomState(s).randn(G.style_dim).astype(np.float32))
                         for s in gseeds]).to(device)
        if n_classes > 0:
            c = F.one_hot(torch.tensor([cl for (cl, _j) in chunk], device=device), n_classes).float()
            img = G(z, c, truncation=trunc, truncation_latent=trunc_latent)[0]
        else:
            img = G(z, truncation=trunc, truncation_latent=trunc_latent)[0]
        img_u8 = _denorm_to_uint8_nhwc(img)  # NHWC uint8

        if save_mode == 'hdf5':
            # Group this (possibly class-straddling) chunk by class before writing.
            for cl in sorted(set(c for c, _j in chunk)):
                sel = [k for k, (cc, _j) in enumerate(chunk) if cc == cl]
                writer.write(cl, img_u8[sel], [gseeds[k] for k in sel])
        else:
            for k, (cl, j) in enumerate(chunk):
                d = os.path.join(outdir, f'class_{cl}')
                os.makedirs(d, exist_ok=True)
                from PIL import Image
                Image.fromarray(img_u8[k], 'RGB').save(
                    os.path.join(d, f'idx_{j:06d}_seed_{gseeds[k]}.png'))

    if writer is not None:
        missing = writer.close()
        if missing and rank == 0:
            print(f'[warn] rank {rank} shard has {missing} unwritten slots', flush=True)
    if rank == 0:
        print(f'Done generating on {num_gpus} GPU(s).', flush=True)

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network', help='Checkpoint .pt (with g_ema)', metavar='PATH', required=True)
@click.option('--outdir', help='Where to write the generated images', metavar='DIR', required=True)
@click.option('--save-mode', type=click.Choice(['hdf5', 'dir']), default='hdf5', show_default=True,
              help='hdf5: merged <desc>.h5 (angle-pipeline input); dir: per-class PNG folders')
@click.option('--desc', help='Merged h5 basename (default: the checkpoint stem)', type=str, default=None)
@click.option('--trunc', 'trunc', help='Truncation psi', type=float, default=1.0, show_default=True)
@click.option('--seed', help='Base random seed', type=int, default=42, show_default=True)
@click.option('--classes', help='Class indices or names, e.g. "0,1,2" or "Ultra_Co11,Ultra_Co25"',
              type=str, default=None)
@click.option('--samples-per-class', help='Images per class (conditional)', type=int, default=1000, show_default=True)
@click.option('--samples', help='Total images (unconditional)', type=int, default=1000, show_default=True)
@click.option('--batch-gpu', help='Batch size per GPU', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--gpus', help='Number of GPUs (self-spawns one worker per GPU)', type=click.IntRange(min=1),
              default=1, show_default=True)
def main(network, outdir, save_mode, desc, trunc, seed, classes, samples_per_class, samples, batch_gpu, gpus):
    os.makedirs(outdir, exist_ok=True)

    # Resolve class selection against the checkpoint metadata up front (fail fast, and so
    # every worker generates the same set).
    _ckpt, n_classes, _res, class_names, _arch = _load_checkpoint(network)
    class_idx = parse_classes(classes, n_classes, class_names)
    if n_classes > 0 and class_idx is None:
        raise click.ClickException('conditional model requires --classes')

    args = (gpus, network, outdir, trunc, seed, class_idx, samples_per_class, samples, batch_gpu, save_mode)
    if gpus == 1:
        _generate_worker(0, *args)
    else:
        torch.multiprocessing.set_start_method('spawn')
        torch.multiprocessing.spawn(fn=_generate_worker, args=args, nprocs=gpus)

    if save_mode == 'hdf5':
        # Rank 0 (this process, after all workers joined) merges the shards. Hard-fails on
        # any incomplete shard.
        if desc is None:
            desc = os.path.splitext(os.path.basename(network))[0]
        out_path = os.path.join(outdir, f'{desc}.h5')
        merge_shards(os.path.join(outdir, 'shards'), out_path, class_names)
        print(f'Merged shards -> {out_path}', flush=True)
    else:
        # dir mode: write the class manifest next to the class_<c>/ folders.
        manifest = {str(i): name for i, name in enumerate(class_names)}
        with open(os.path.join(outdir, 'classes.json'), 'w') as f:
            json.dump(manifest, f, indent=2)


if __name__ == '__main__':
    main()

#----------------------------------------------------------------------------
