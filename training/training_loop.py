# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""StyleSwin training loop, refactored to mirror san-v2's logging and metrics.

The generator/discriminator update steps are kept byte-for-byte identical to the
original ``train_styleswin.py`` (D step + logistic loss, R1 regularization, G step +
non-saturating loss, EMA accumulate). Only the *surrounding* machinery changes:

* progress is accounted in **kimg/ticks** (not raw iterations) and printed with the
  same status line as san-v2 (``tick kimg time sec/tick ...``);
* logs are written to ``run_dir/log.txt`` (stdout capture), ``stats.jsonl`` and
  TensorBoard events, exactly like san-v2;
* combra generative-quality metrics are computed each snapshot tick, sharded across
  all GPU ranks (ported from ``san-v2/training/training_loop.py``).

Class conditioning is threaded through every G/D call: the generator embeds the
one-hot label into its mapping network (san-v2 technique) and the discriminator adds
a projection term (Miyato & Koyama). ``n_classes == 0`` keeps the unconditional path.
"""

import importlib.util
import json
import math
import os
import time

import numpy as np
import torch
import torchvision
from torch import autograd, nn, optim
from torch.nn import functional as F
from torch.utils import data

import dnnlib
from torch_utils import training_stats

from dataset.dataset import MultiResolutionDataset
from dataset.imagenet_dataset import ImageFolderDataset
from models.discriminator import Discriminator
from models.generator import Generator
from utils.CRDiffAug import CR_DiffAug

# ImageNet normalization -- StyleSwin's original real-image preprocessing. Reals are
# fed to D in this normalized space; G learns to match it. combra needs uint8 [0,255],
# so the reverse transform is applied before scoring generated samples.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Number of fake images generated each snapshot for the combra image metrics
# (combra_fid10k / combra_cmmd10k / combra_fd_dinov2_10k), scored against the whole
# training set as the fixed reference -- mirrors san-v2.
COMBRA_NUM_GEN = 10000

#----------------------------------------------------------------------------
# Small helpers, copied verbatim from the original train_styleswin.py so the update
# math is unchanged.

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    return data.SequentialSampler(dataset)


def d_logistic_loss(real_pred, fake_pred):
    assert type(real_pred) is type(fake_pred), "real_pred must be the same type as fake_pred"
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    return F.softplus(-fake_pred).mean()

#----------------------------------------------------------------------------
# Normalization helpers between the D training space (ImageNet-normalized float) and
# combra's uint8 space.

def _norm_stats(device):
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return mean, std


def _normalize_real(img_u8, device):
    # uint8 NCHW tensor [0,255] -> ImageNet-normalized float, identical to the old
    # transforms.ToTensor()+Normalize() pipeline.
    mean, std = _norm_stats(device)
    x = img_u8.to(device).float().div_(255.0)
    return (x - mean) / std


def _denorm_to_uint8(images_norm):
    # normalized float NCHW numpy -> uint8 NCHW numpy in [0,255] (reverse of the real
    # preprocessing), so generated samples are on the same scale as the uint8 reals.
    mean = np.asarray(_IMAGENET_MEAN, np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(_IMAGENET_STD, np.float32).reshape(1, 3, 1, 1)
    x = images_norm * std + mean
    return np.rint(x * 255.0).clip(0, 255).astype(np.uint8)

#----------------------------------------------------------------------------
# Distributed combra metrics: shard ALL the per-image extraction across ranks instead
# of running it on rank 0. Ported from san-v2; adapted to StyleSwin's generator call
# convention (G_ema(z, c)[0], or G_ema(z)[0] when unconditional) and its normalized
# output space. Every rank runs the same collectives, so the caller MUST invoke the
# eval/precompute on all ranks (gated by a rank-uniform flag) or the gathers deadlock.

@torch.no_grad()
def _combra_generate_local_shard(G_ema, grid_z, grid_c, batch_gpu, num_gpus, rank):
    # Rank r generates the fakes at indices [r, r+num_gpus, ...]; concatenating every
    # rank's shard reproduces the full set exactly. Returns a normalized float NCHW numpy.
    n = grid_z.shape[0]
    idx = torch.arange(rank, n, num_gpus, device=grid_z.device)
    z = grid_z.index_select(0, idx)
    if grid_c is not None:
        c = grid_c.index_select(0, idx)
        images = torch.cat([G_ema(zz, cc)[0]
                            for zz, cc in zip(z.split(batch_gpu), c.split(batch_gpu))], dim=0)
    else:
        images = torch.cat([G_ema(zz)[0] for zz in z.split(batch_gpu)], dim=0)
    return images.cpu().numpy()


def _combra_gather_to_rank0(local, device, rank, num_gpus):
    # Gather per-rank arrays [k, ...] to rank 0, concatenated in rank order (None on
    # other ranks). all_gather works on both gloo and nccl; ranks may hold different k,
    # so each block is padded to the max along axis 0 and trimmed back.
    if num_gpus == 1:
        return local
    t = torch.from_numpy(np.ascontiguousarray(local)).to(device)
    count = torch.tensor([t.shape[0]], device=device, dtype=torch.long)
    counts = [torch.zeros_like(count) for _ in range(num_gpus)]
    torch.distributed.all_gather(counts, count)
    max_count = max(int(c.item()) for c in counts)
    if t.shape[0] < max_count:
        pad = torch.zeros(max_count - t.shape[0], *t.shape[1:], device=device, dtype=t.dtype)
        t = torch.cat([t, pad], dim=0)
    gathered = [torch.empty_like(t) for _ in range(num_gpus)]
    torch.distributed.all_gather(gathered, t)
    if rank != 0:
        return None
    rows = [gathered[i][:int(counts[i].item())].cpu().numpy() for i in range(num_gpus)]
    return np.concatenate(rows, axis=0)


def _combra_gather_pooled_angles(images_u8, device, rank, num_gpus):
    # Extract this rank's pooled vertex angles and gather the 1-D arrays to rank 0.
    from combra.metrics import images_to_pooled_angles
    pooled = np.asarray(
        images_to_pooled_angles(images_u8, workers=min(32, os.cpu_count() or 1)),
        np.float32).reshape(-1, 1)
    gathered = _combra_gather_to_rank0(pooled, device, rank, num_gpus)
    return gathered.reshape(-1) if gathered is not None else None


def _combra_precompute_reference(reference_u8_set, device, rank, num_gpus):
    # All ranks extract pooled angles + the three feature sets from this rank's
    # deterministic slice of the reference (the whole training set, as uint8) and gather
    # them to rank 0. Called once before the loop so no reference work recurs per tick.
    from combra.metrics import cmmd_features, fd_dinov2_features, fid_features
    n = len(reference_u8_set)
    idx = range(rank, n, num_gpus)
    local_u8 = np.stack([reference_u8_set[i][0] for i in idx])  # NCHW uint8
    angles = _combra_gather_pooled_angles(local_u8, device, rank, num_gpus)
    extractors = (('fid', fid_features), ('cmmd', cmmd_features), ('fd_dinov2', fd_dinov2_features))
    feat = {}
    for name, fn in extractors:
        feats = fn(local_u8, device=device).astype(np.float32)
        feat[name] = _combra_gather_to_rank0(feats, device, rank, num_gpus)
    if rank != 0:
        return None
    return {'angles': angles, 'feat': feat}


def _combra_eval_distributed(G_ema, grid_z, grid_c, batch_gpu, num_gpus, rank, device, combra_ref):
    # Returns the combra metrics dict on rank 0, None on other ranks. Every rank runs the
    # same collectives, so the caller MUST invoke this on all ranks.
    from combra.metrics import (angle_density_metrics_from_pooled, cmmd_features,
        cmmd_from_features, fd_dinov2_features, fd_dinov2_from_features, fid_features,
        fid_from_features)

    # 1. Generate this rank's shard and denormalize to uint8 (combra's angle path is
    #    scale-sensitive, so both sides must be uint8 on the same [0,255] scale).
    local = _combra_generate_local_shard(G_ema, grid_z, grid_c, batch_gpu, num_gpus, rank)
    local_u8 = _denorm_to_uint8(local)

    # 2. Extract image features on the local shard; gather feature rows to rank 0.
    extractors = (('fid', fid_features), ('cmmd', cmmd_features), ('fd_dinov2', fd_dinov2_features))
    gen_feats = {}
    for name, fn in extractors:
        feats = fn(local_u8, device=device).astype(np.float32)
        gen_feats[name] = _combra_gather_to_rank0(feats, device, rank, num_gpus)

    # 3. Pool the vertex angles on the local shard; gather the 1-D arrays to rank 0.
    gen_angles = _combra_gather_pooled_angles(local_u8, device, rank, num_gpus)

    if rank != 0:
        return None

    # 4. Rank 0: angle / Gaussian-fit metrics + image-feature distances vs the cached reference.
    metrics = dict(angle_density_metrics_from_pooled(combra_ref['angles'], gen_angles))
    combiners = {'fid': fid_from_features, 'cmmd': cmmd_from_features, 'fd_dinov2': fd_dinov2_from_features}
    for name in ('fid', 'cmmd', 'fd_dinov2'):
        metrics[name] = combiners[name](combra_ref['feat'][name], gen_feats[name])
    return metrics

#----------------------------------------------------------------------------

def _sample_labels(class_probs, n, n_classes, device, generator=None):
    # Sample n one-hot labels [n, n_classes] from the empirical class distribution
    # (avoids over-representing rare classes for imbalanced data).
    idx = torch.multinomial(class_probs, n, replacement=True, generator=generator)
    return F.one_hot(idx, n_classes).float().to(device)

#----------------------------------------------------------------------------

def training_loop(
    rank                    = 0,
    num_gpus                = 1,
    run_dir                 = '.',
    data_path               = None,
    resolution              = 256,
    lmdb                    = False,
    n_classes               = 0,
    batch_gpu               = 4,
    total_kimg              = 25000,
    kimg_per_tick           = 4,
    snap_ticks              = 50,
    random_seed             = 0,
    workers                 = 3,
    combra_metrics          = True,
    save_inference_only     = False,
    fake_label_sampling     = 'empirical',
    resume                  = None,
    # Model / optimizer hyperparameters (StyleSwin defaults).
    style_dim               = 512,
    n_mlp                   = 8,
    lr_mlp                  = 0.01,
    enable_full_resolution  = 8,
    g_channel_multiplier    = 1,
    d_channel_multiplier    = 2,
    g_lr                    = 0.0002,
    d_lr                    = 0.0002,
    beta1                   = 0.0,
    beta2                   = 0.99,
    r1                      = 10.0,
    d_reg_every             = 16,
    gan_weight              = 1.0,
    ttur                    = False,
    bcr                     = False,
    bcr_fake_lambda         = 10.0,
    bcr_real_lambda         = 10.0,
    use_checkpoint          = False,
    use_flip                = False,
    D_sn                    = False,
):
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = True

    def stage(msg):
        if rank == 0:
            print(f'[Stage] {msg}', flush=True)

    # ------------------------------------------------------------------ Dataset.
    if lmdb:
        from torchvision import transforms
        tfm = [transforms.Resize((resolution, resolution))]
        if use_flip:
            tfm.append(transforms.RandomHorizontalFlip())
        tfm += [transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)]
        training_set = MultiResolutionDataset(data_path, transforms.Compose(tfm), resolution)
        reference_u8_set = None
    else:
        # ImageNet-style zip/dir (uint8 CHW + one-hot label) -- same format as san-v2.
        training_set = ImageFolderDataset(path=data_path, use_labels=(n_classes > 0), xflip=use_flip)
        reference_u8_set = training_set  # already uint8; used as the fixed combra reference

    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', getattr(training_set, 'image_shape', [3, resolution, resolution]))
        print('Num classes:', n_classes)
        print()

    sampler = data_sampler(training_set, shuffle=True, distributed=(num_gpus > 1))
    loader = sample_data(data.DataLoader(training_set, batch_size=batch_gpu, sampler=sampler,
                                         num_workers=workers, drop_last=True, pin_memory=True))

    # Empirical class distribution for fake-label sampling.
    class_probs = None
    if n_classes > 0:
        if fake_label_sampling == 'uniform':
            class_probs = torch.ones(n_classes, device=device)
        else:
            raw = training_set._get_raw_labels()
            counts = np.bincount(raw.astype(np.int64), minlength=n_classes).astype(np.float32)
            class_probs = torch.tensor(counts, device=device).clamp_min(1.0)
        if rank == 0:
            print('Class distribution (fake-label sampling):',
                  (class_probs / class_probs.sum()).cpu().numpy())

    # ------------------------------------------------------------------ Networks.
    def make_G():
        return Generator(resolution, style_dim, n_mlp, channel_multiplier=g_channel_multiplier,
                         lr_mlp=lr_mlp, enable_full_resolution=enable_full_resolution,
                         use_checkpoint=use_checkpoint, n_classes=n_classes).to(device)

    generator = make_G()
    discriminator = Discriminator(resolution, channel_multiplier=d_channel_multiplier,
                                  sn=D_sn, n_classes=n_classes).to(device)
    g_ema = make_G()
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = 1.0                       # StyleSwin: no G regularization
    d_reg_ratio = d_reg_every / (d_reg_every + 1)

    start_nimg = 0
    if resume is not None:
        stage(f'Resuming from "{resume}"')
        ckpt = torch.load(resume, map_location=lambda s, loc: s)
        generator.load_state_dict(ckpt['g'])
        g_ema.load_state_dict(ckpt['g_ema'])
        try:
            discriminator.load_state_dict(ckpt['d'])
        except Exception:
            if rank == 0:
                print("We don't load D.")
        start_nimg = int(ckpt.get('cur_nimg', 0))

    g_module = generator
    d_module = discriminator
    if num_gpus > 1:
        generator = nn.parallel.DistributedDataParallel(
            generator, device_ids=[rank], output_device=rank, broadcast_buffers=False)
        discriminator = nn.parallel.DistributedDataParallel(
            discriminator, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    g_lr_eff = (d_lr / 4 if ttur else g_lr) * g_reg_ratio
    g_optim = optim.Adam(generator.parameters(), lr=g_lr_eff,
                         betas=(beta1 ** g_reg_ratio, beta2 ** g_reg_ratio))
    d_optim = optim.Adam(discriminator.parameters(), lr=d_lr * d_reg_ratio,
                         betas=(beta1 ** d_reg_ratio, beta2 ** d_reg_ratio))
    if resume is not None:
        try:
            g_optim.load_state_dict(ckpt['g_optim'])
            d_optim.load_state_dict(ckpt['d_optim'])
        except Exception:
            if rank == 0:
                print("We don't load optimizers.")

    # ------------------------------------------------------------------ combra reference.
    combra_ref = None
    if combra_metrics and (importlib.util.find_spec('combra') is not None) and (reference_u8_set is not None):
        stage('Precomputing combra reference (sharded over ranks)')
        try:
            combra_ref = _combra_precompute_reference(reference_u8_set, device, rank, num_gpus)
        except Exception as e:
            combra_ref = None
            if rank == 0:
                print(f'[combra] reference precompute failed, disabling combra metrics: {e}', flush=True)
    if combra_metrics and (rank == 0) and (importlib.util.find_spec('combra') is None):
        print("Warning: combra_metrics=True but the `combra` package is not installed -- "
              "combra metrics will be skipped. Install it (e.g. `pip install -e ../combra`) "
              "to enable them, or pass --combra-metrics=false to silence this warning.", flush=True)
    if combra_metrics and lmdb and rank == 0:
        print('Warning: combra metrics require the ImageNet-style dataset (uint8 reference); '
              'skipping combra metrics on the --lmdb path.', flush=True)

    # Fixed combra generation set (latents + labels), seeded identically on every rank.
    combra_active = combra_metrics and (importlib.util.find_spec('combra') is not None) and (reference_u8_set is not None)
    combra_z = combra_c = None
    if combra_active:
        combra_z = torch.randn([COMBRA_NUM_GEN, style_dim], device=device,
                               generator=torch.Generator(device=device).manual_seed(random_seed * num_gpus + 1))
        if n_classes > 0:
            gcpu = torch.Generator().manual_seed(random_seed)
            probs = class_probs.cpu() if class_probs is not None else torch.ones(n_classes)
            combra_c = _sample_labels(probs, COMBRA_NUM_GEN, n_classes, device, generator=gcpu)

    # ------------------------------------------------------------------ Logging.
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # ------------------------------------------------------------------ Training loop.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = start_nimg
    batch_size = batch_gpu * num_gpus
    accum = 0.5 ** (32 / (10 * 1000))
    best_fid = float('inf')
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    start_time = tick_start_time
    maintenance_time = 0.0
    batch_idx = 0

    while True:
        # ---------------------------------------------- one training iteration (unchanged math).
        generator.train()
        real_batch = next(loader)
        if n_classes > 0 and not lmdb:
            real_u8, real_labels = real_batch
            real_img = _normalize_real(real_u8, device)
            real_labels = real_labels.to(device)
        elif not lmdb:
            real_u8, _ = real_batch
            real_img = _normalize_real(real_u8, device)
            real_labels = None
        else:
            real_img = real_batch.to(device)
            real_labels = None

        # Train D
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        noise = torch.randn(batch_gpu, style_dim, device=device)
        fake_labels = _sample_labels(class_probs, batch_gpu, n_classes, device) if n_classes > 0 else None

        fake_img, _ = generator(noise, fake_labels)
        fake_pred = discriminator(fake_img, fake_labels)
        real_pred = discriminator(real_img, real_labels)
        d_loss = d_logistic_loss(real_pred, fake_pred) * gan_weight

        if bcr:
            real_img_cr_aug = CR_DiffAug(real_img)
            fake_img_cr_aug = CR_DiffAug(fake_img)
            fake_pred_aug = discriminator(fake_img_cr_aug, fake_labels)
            real_pred_aug = discriminator(real_img_cr_aug, real_labels)
            d_loss = d_loss + bcr_fake_lambda * F.mse_loss(fake_pred_aug, fake_pred) \
                + bcr_real_lambda * F.mse_loss(real_pred_aug, real_pred)

        training_stats.report('Loss/D/loss', d_loss)
        discriminator.zero_grad()
        d_loss.backward()
        nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
        d_optim.step()

        if batch_idx % d_reg_every == 0:
            real_img.requires_grad = True
            real_pred = discriminator(real_img, real_labels)
            r1_loss = d_r1_loss(real_pred, real_img)
            discriminator.zero_grad()
            (gan_weight * (r1 / 2 * r1_loss * d_reg_every + 0 * real_pred[0])).backward()
            d_optim.step()
            training_stats.report('Loss/r1', r1_loss)

        # Train G
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        noise = torch.randn(batch_gpu, style_dim, device=device)
        fake_labels = _sample_labels(class_probs, batch_gpu, n_classes, device) if n_classes > 0 else None
        fake_img, _ = generator(noise, fake_labels)
        fake_pred = discriminator(fake_img, fake_labels)
        g_loss = g_nonsaturating_loss(fake_pred) * gan_weight
        training_stats.report('Loss/G/loss', g_loss)
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)

        cur_nimg += batch_size
        batch_idx += 1

        # ---------------------------------------------- tick maintenance.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', _cpu_mem_gb()):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        if rank == 0:
            print(' '.join(fields))

        # Snapshot: metrics + checkpoints. Skipped at tick 0 (untrained G_ema), like san-v2.
        snapshot = (done or (cur_tick > 0 and cur_tick % snap_ticks == 0))
        if snapshot:
            g_ema.eval()

            if combra_active:
                stage('Evaluating combra metrics')
                try:
                    combra_results = _combra_eval_distributed(
                        g_ema, combra_z, combra_c, batch_gpu, num_gpus, rank, device, combra_ref)
                except Exception as e:
                    combra_results = None
                    if rank == 0:
                        print(f'[combra] metric evaluation failed: {e}', flush=True)
                if rank == 0 and combra_results is not None:
                    combra_image_rename = {'fid': 'fid10k', 'cmmd': 'cmmd10k', 'fd_dinov2': 'fd_dinov2_10k'}
                    for name, value in combra_results.items():
                        key = combra_image_rename.get(name, name)
                        stats_metrics[f'combra_{key}'] = float(value)
                    print('combra metrics: ' + ', '.join(
                        f'{k}={v:.4f}' for k, v in combra_results.items()), flush=True)

            if rank == 0:
                # Save an image snapshot grid and the checkpoints.
                _save_image_snapshot(g_ema, combra_z, combra_c, batch_gpu, n_classes, run_dir, cur_nimg, device)
                ckpt = {
                    'g': g_module.state_dict(), 'd': d_module.state_dict(),
                    'g_ema': g_ema.state_dict(), 'g_optim': g_optim.state_dict(),
                    'd_optim': d_optim.state_dict(), 'cur_nimg': cur_nimg,
                    'n_classes': n_classes, 'size': resolution,
                }
                torch.save(ckpt, os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pt'))
                if save_inference_only:
                    torch.save({'g_ema': g_ema.state_dict(), 'n_classes': n_classes, 'size': resolution},
                               os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}-inference.pt'))

                fid_key = 'combra_fid10k' if 'combra_fid10k' in stats_metrics else None
                if fid_key is not None and stats_metrics[fid_key] < best_fid:
                    best_fid = stats_metrics[fid_key]
                    torch.save(ckpt, os.path.join(run_dir, 'best_model.pt'))
                    with open(os.path.join(run_dir, 'best_nimg.txt'), 'w') as f:
                        f.write(str(cur_nimg))

        # Update logs.
        timestamp = time.time()
        stats_collector.update()
        stats_dict = stats_collector.as_dict()
        if stats_jsonl is not None:
            stats_jsonl.write(json.dumps(dict(stats_dict, timestamp=timestamp)) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            gstep = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=gstep, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=gstep, walltime=walltime)
            stats_tfevents.flush()

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    if rank == 0:
        stage('Exiting')
        if stats_jsonl is not None:
            stats_jsonl.close()

#----------------------------------------------------------------------------

def _cpu_mem_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**30
    except Exception:
        return 0.0


@torch.no_grad()
def _save_image_snapshot(g_ema, combra_z, combra_c, batch_gpu, n_classes, run_dir, cur_nimg, device):
    # A small grid of G_ema samples, denormalized to [0,1] for viewing.
    n = min(16, batch_gpu if combra_z is None else combra_z.shape[0])
    z = torch.randn(n, g_ema.style_dim, device=device) if combra_z is None else combra_z[:n]
    c = combra_c[:n] if (combra_c is not None) else None
    img = g_ema(z, c)[0] if n_classes > 0 else g_ema(z)[0]
    mean, std = _norm_stats(device)
    img = (img * std + mean).clamp(0, 1)
    torchvision.utils.save_image(img, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'),
                                 nrow=int(math.sqrt(n)), padding=0)

#----------------------------------------------------------------------------
