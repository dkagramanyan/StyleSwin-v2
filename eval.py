# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standalone combra evaluator for a trained StyleSwin checkpoint (``styleswin-eval``).

Scores a checkpoint's ``g_ema`` against a dataset with the same sharded combra split-API
metrics used during training (angle-density + fid / cmmd / fd_dinov2), reusing the training
loop's helpers on the single-process path. Prints the metrics and, with ``--out``, writes
them as JSON.

    styleswin-eval --network run/styleswin-snapshot-000500-inference.pt \\
        --data datasets/imagenet_9to4_256x256.zip --num-fid-samples 10000
"""

import json

import click
import numpy as np
import torch

from gen_images import _build_generator, _load_checkpoint
from training.training_loop import _combra_eval_distributed, _combra_eval_labels, _combra_precompute_reference


@click.command()
@click.option('--network', help='Checkpoint .pt (with g_ema)', metavar='PATH', required=True)
@click.option('--data', help='Reference dataset (ImageNet-style zip/dir)', metavar='[ZIP|DIR]', required=True)
@click.option('--num-fid-samples', help='Fakes generated for the image metrics', type=click.IntRange(min=1),
              default=10000, show_default=True)
@click.option('--combra-ref-count', help='Cap the reference to a seeded random subset (0 = whole set)',
              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--batch-gpu', help='Batch size per forward', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--seed', help='Seed for the eval latents / reference subset', type=int, default=0, show_default=True)
@click.option('--out', help='Optional path to write the metrics JSON', metavar='PATH', default=None)
def main(network, data, num_fid_samples, combra_ref_count, batch_gpu, seed, out):
    from dataset.imagenet_dataset import ImageFolderDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt, n_classes, resolution, class_names, arch = _load_checkpoint(network)
    G = _build_generator(ckpt, n_classes, resolution, arch, device)

    ref_set = ImageFolderDataset(path=data, use_labels=(n_classes > 0))
    if ref_set.resolution != resolution:
        raise click.ClickException(
            f'--data is {ref_set.resolution}px but the checkpoint generates {resolution}px images')
    n_ref = len(ref_set)
    if combra_ref_count and combra_ref_count < n_ref:
        ref_indices = np.sort(np.random.RandomState(seed).permutation(n_ref)[:combra_ref_count]).tolist()
    else:
        ref_indices = list(range(n_ref))
    combra_ref, ok = _combra_precompute_reference(ref_set, ref_indices, device, 0, 1)
    if not ok:
        raise click.ClickException('combra reference precompute failed (see log above)')

    if device.type != 'cuda':
        # Training draws the eval latents from a CUDA generator; a CPU generator with the
        # same seed yields different latents, so these scores are not comparable with the
        # in-training combra metrics of the same seed.
        print('Warning: no CUDA device -- the eval latents differ from the ones training '
              'drew for the same --seed; metrics are not comparable with the training log.', flush=True)
    z = torch.randn([num_fid_samples, arch['style_dim']], device=device,
                    generator=torch.Generator(device=device).manual_seed(seed + 1))
    c = None
    if n_classes > 0:
        c = _combra_eval_labels(ref_set, num_fid_samples, n_classes, device, seed)

    metrics = _combra_eval_distributed(G, z, c, batch_gpu, 1, 0, device, combra_ref)
    metrics = {k: float(v) for k, v in metrics.items()}
    print(json.dumps(metrics, indent=2))
    if out is not None:
        with open(out, 'w') as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
