# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Train StyleSwin with the shared model-API click CLI (``styleswin-train``).

Mirrors the cross-model training convention -- ``--outdir/--data/--gpus/--batch-gpu/--cfg``,
kimg/ticks (``--kimg/--tick/--snap``), the ``--precision/--tf32/--bench`` scheme, the single
``--mirror`` loader-level flip, and the ``--grad-accum`` batch formula -- while keeping
StyleSwin's own model flags. The kimg/tick loop, logging, checkpoint contract and sharded
combra metrics live in ``training/training_loop.py``; the generator/discriminator update
math is unchanged from upstream StyleSwin.

Class conditioning uses the san-v2 generator technique (embed the one-hot label into the
mapping network) and a Miyato & Koyama projection discriminator. Pass ``--cond True`` to
enable it; ``n_classes`` and ``class_names`` are read from the dataset's ``dataset.json``.

Example (single stage, conditional, 2 GPUs):

    styleswin-train --outdir=./runs --data=./datasets/imagenet_9to4_256x256.zip \\
        --gpus=2 --batch-gpu=16 --cond True --combra-metrics True \\
        --kimg 25000 --snap 50
"""

import json
import os
import re
import tempfile

import click
import torch

import dnnlib
from torch_utils import training_stats
from training import training_loop

#----------------------------------------------------------------------------
# Per-resolution presets, selected with --cfg. The StyleSwin generator/discriminator are
# resolution-parametric (built from the dataset resolution), so today these presets differ
# only in the memory-bound batch size; the remaining knobs are bundled here so each
# resolution has one place to tune. Any explicit CLI flag overrides the preset. Keys are
# click parameter names.

RESOLUTION_CONFIGS = {
    'styleswin-256':  dict(size=256,  batch_gpu=64, enable_full_resolution=8,
                           g_channel_multiplier=1, d_channel_multiplier=2,
                           glr=0.0002, dlr=0.0002, r1=10.0, d_reg_every=16, style_dim=512),
    'styleswin-512':  dict(size=512,  batch_gpu=32,  enable_full_resolution=8,
                           g_channel_multiplier=1, d_channel_multiplier=2,
                           glr=0.0002, dlr=0.0002, r1=10.0, d_reg_every=16, style_dim=512),
    'styleswin-1024': dict(size=1024, batch_gpu=4,  enable_full_resolution=8,
                           g_channel_multiplier=1, d_channel_multiplier=2,
                           glr=0.0002, dlr=0.0002, r1=10.0, d_reg_every=16, style_dim=512),
}

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    # Rank-0-only run log, named after the run directory (§7). Other ranks stay on stdout;
    # under SLURM their output lands in slurm-<jobid>.out.
    if rank == 0:
        log_name = os.path.basename(c.run_dir) + '.log'
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, log_name), file_mode='a', should_flush=True)

    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        init_method = f'file://{init_file}'
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)

    training_loop.training_loop(rank=rank, num_gpus=c.num_gpus, run_dir=c.run_dir, **c.loop)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # A fresh run id is always allocated -- existing directories are never reused (§2).
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.loop.batch_gpu * c.num_gpus * c.loop.grad_accum} images')
    print(f'Training duration:   {c.loop.total_kimg} kimg')
    print(f'Dataset path:        {c.loop.data_path}')
    print(f'Dataset resolution:  {c.loop.resolution}')
    print(f'Num classes:         {c.loop.n_classes}')
    print()

    if dry_run:
        print('Dry run; exiting.')
        return

    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def _dataset_info(data_path, lmdb, size, cond):
    """Return (resolution, n_classes, class_names, name) for the dataset."""
    if lmdb:
        return size, 0, None, os.path.splitext(os.path.basename(data_path.rstrip('/')))[0]
    from dataset.imagenet_dataset import ImageFolderDataset
    try:
        ds = ImageFolderDataset(path=data_path, use_labels=cond)
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    n_classes = ds.label_dim if (cond and ds.has_labels) else 0
    class_names = ds.class_names if (cond and ds.has_labels) else None
    return ds.resolution, n_classes, class_names, ds.name

#----------------------------------------------------------------------------

@click.command()
# Required.
@click.option('--outdir',      help='Where to save the results', metavar='DIR',       required=True)
@click.option('--data',        help='Training data', metavar='[ZIP|DIR]',             type=str, required=True)
@click.option('--gpus',        help='Number of GPUs to use', metavar='INT',           type=click.IntRange(min=1), required=True)
@click.option('--batch-gpu',   help='Batch size per GPU (total = batch-gpu * gpus * grad-accum); from --cfg if omitted', metavar='INT', type=click.IntRange(min=1), default=None)
@click.option('--cfg',         help='Per-resolution preset', type=click.Choice(list(RESOLUTION_CONFIGS)), default=None, show_default=True)
@click.option('--grad-accum',  help='Gradient-accumulation micro-steps per optimizer step', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
# Conditioning / dataset.
@click.option('--cond',        help='Train class-conditional model', metavar='BOOL',  type=bool, default=False, show_default=True)
@click.option('--mirror',      help='Stochastic per-item horizontal flip in the training loader', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--lmdb',        help='Use a legacy LMDB dataset (unconditional)', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--size',        help='Image resolution (lmdb only; else read from data)', metavar='INT', type=click.IntRange(min=4), default=256, show_default=True)
@click.option('--fake-label-sampling', help='Fake-label distribution', type=click.Choice(['empirical', 'uniform']), default='empirical', show_default=True)
# Duration / logging.
@click.option('--kimg',        help='Total training duration', metavar='KIMG',        type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',        help='How often to print progress', metavar='KIMG',    type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',        help='How often to snapshot/eval', metavar='TICKS',    type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--combra-metrics', help='Compute combra metrics each snapshot tick', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--num-fid-samples', help='Fakes generated for the combra image metrics (0 disables eval)', metavar='INT', type=click.IntRange(min=0), default=10000, show_default=True)
@click.option('--combra-ref-count', help='Cap the combra reference to a seeded random subset (0 = whole set)', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--snapshot-keep-last', help='Keep only the most recent N inference snapshots (0 = keep all)', metavar='INT', type=click.IntRange(min=0), default=3, show_default=True)
@click.option('--seed',        help='Random seed', metavar='INT',                     type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--workers',     help='DataLoader worker processes', metavar='INT',     type=click.IntRange(min=1), default=3, show_default=True)
@click.option('--desc',        help='String to include in the run dir name', metavar='STR', type=str)
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)
# Precision.
@click.option('--precision',   help='Training precision', type=click.Choice(['fp32', 'fp16', 'bf16']), default='fp32', show_default=True)
@click.option('--tf32',        help='Allow TF32 matmul/cuDNN', metavar='BOOL',        type=bool, default=True, show_default=True)
@click.option('--bench',       help='Enable cuDNN autotune (benchmark)', metavar='BOOL', type=bool, default=True, show_default=True)
# StyleSwin model / optimizer hyperparameters.
@click.option('--style-dim',   help='Style (latent) dimension', metavar='INT',        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--n-mlp',       help='Mapping-network depth', metavar='INT',           type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--lr-mlp',      help='LR multiplier for the mapping MLP', metavar='FLOAT', type=float, default=0.01, show_default=True)
@click.option('--enable-full-resolution', help='Full-attention resolution index', metavar='INT', type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--g-channel-multiplier', help='Generator channel multiplier', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--d-channel-multiplier', help='Discriminator channel multiplier', metavar='INT', type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--glr',         help='G learning rate', metavar='FLOAT',               type=click.FloatRange(min=0), default=0.0002, show_default=True)
@click.option('--dlr',         help='D learning rate', metavar='FLOAT',               type=click.FloatRange(min=0), default=0.0002, show_default=True)
@click.option('--r1',          help='R1 regularization weight', metavar='FLOAT',      type=float, default=10.0, show_default=True)
@click.option('--d-reg-every', help='Apply R1 every N steps', metavar='INT',          type=click.IntRange(min=1), default=16, show_default=True)
@click.option('--gan-weight',  help='GAN loss weight', metavar='FLOAT',               type=float, default=1.0, show_default=True)
@click.option('--ttur',        help='Use TTUR (G_lr = D_lr / 4)', metavar='BOOL',     type=bool, default=False, show_default=True)
@click.option('--bcr',         help='Enable bCR consistency regularization', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--d-sn',        help='Spectral norm in D', metavar='BOOL',             type=bool, default=False, show_default=True)
@click.option('--use-checkpoint', help='Gradient checkpointing in G', metavar='BOOL', type=bool, default=False, show_default=True)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)

    # Apply a per-resolution preset (--cfg). Explicit CLI flags take precedence over it.
    if opts.cfg is not None:
        ctx = click.get_current_context()
        for key, val in RESOLUTION_CONFIGS[opts.cfg].items():
            if ctx.get_parameter_source(key) != click.core.ParameterSource.COMMANDLINE:
                opts[key] = val

    if opts.batch_gpu is None:
        raise click.ClickException('Provide --batch-gpu, or a --cfg preset that sets it.')

    resolution, n_classes, class_names, name = _dataset_info(opts.data, opts.lmdb, opts.size, opts.cond)

    if opts.cfg is not None and not opts.lmdb and resolution != RESOLUTION_CONFIGS[opts.cfg]['size']:
        raise click.ClickException(
            f'--cfg {opts.cfg} expects {RESOLUTION_CONFIGS[opts.cfg]["size"]}px data, '
            f'but --data is {resolution}px ({name}).')

    c = dnnlib.EasyDict()
    c.num_gpus = opts.gpus
    c.loop = dnnlib.EasyDict(
        data_path=opts.data,
        resolution=resolution,
        lmdb=opts.lmdb,
        n_classes=n_classes,
        class_names=class_names,
        batch_gpu=opts.batch_gpu,
        grad_accum=opts.grad_accum,
        total_kimg=opts.kimg,
        kimg_per_tick=opts.tick,
        snap_ticks=opts.snap,
        random_seed=opts.seed,
        workers=opts.workers,
        combra_metrics=opts.combra_metrics,
        num_fid_samples=opts.num_fid_samples,
        combra_ref_count=opts.combra_ref_count,
        snapshot_keep_last=opts.snapshot_keep_last,
        fake_label_sampling=opts.fake_label_sampling,
        mirror=opts.mirror,
        precision=opts.precision,
        tf32=opts.tf32,
        bench=opts.bench,
        style_dim=opts.style_dim,
        n_mlp=opts.n_mlp,
        lr_mlp=opts.lr_mlp,
        enable_full_resolution=opts.enable_full_resolution,
        g_channel_multiplier=opts.g_channel_multiplier,
        d_channel_multiplier=opts.d_channel_multiplier,
        g_lr=opts.glr,
        d_lr=opts.dlr,
        r1=opts.r1,
        d_reg_every=opts.d_reg_every,
        gan_weight=opts.gan_weight,
        ttur=opts.ttur,
        bcr=opts.bcr,
        use_checkpoint=opts.use_checkpoint,
        D_sn=opts.d_sn,
    )

    # Run dir: <id>-<cfg>-gpus<G>-batch<B>[-desc], B the total batch, no dataset name (§2).
    total_batch = opts.batch_gpu * opts.gpus * opts.grad_accum
    cfg_name = opts.cfg or 'styleswin'
    desc = f'{cfg_name}-gpus{opts.gpus:d}-batch{total_batch:d}'
    suffix = []
    if opts.cond:
        suffix.append('cond')
    if opts.desc is not None:
        suffix.append(opts.desc)
    if suffix:
        desc += '-' + '-'.join(suffix)

    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    main()

#----------------------------------------------------------------------------
