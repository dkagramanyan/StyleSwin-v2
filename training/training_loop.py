# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""StyleSwin training loop, following the shared model-API convention.

The generator/discriminator update steps are the same StyleSwin math (D step + logistic
loss, R1 regularization, G step + non-saturating loss, EMA accumulate); the surrounding
machinery follows the cross-model contract:

* progress in **kimg/ticks**, the san-v2 status line, rank-0 ``stats.jsonl`` + TensorBoard;
* the **checkpoint contract (§3)**: exactly one artifact kind --
  ``network-snapshot-<kimg:06d>-inference.pt`` (EMA-only weights + self-describing
  metadata), written atomically every snapshot tick **and always at the last tick**,
  history pruned to ``--snapshot-keep-last``. No resume, no rolling ``latest``, no
  ``best_model.*``;
* combra generative-quality metrics each snapshot tick, sharded across ranks, mirrored
  into both TensorBoard (``Metrics/combra_*``) and ``stats.jsonl``;
* the **normalization contract (§5)**: uint8 at every boundary, one normalize/denormalize
  pair (ImageNet mean/std) asserted to round-trip; ``--mirror`` is a loader-level flip that
  never touches the combra reference.
"""

import glob
import importlib.util
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torchvision
from torch import autograd, nn, optim
from torch.nn import functional as F
from torch.utils import data

import dnnlib
from dataset.dataset import MultiResolutionDataset
from dataset.imagenet_dataset import ImageFolderDataset
from models.discriminator import Discriminator
from models.generator import Generator
from torch_utils import training_stats
from utils.CRDiffAug import CR_DiffAug

# ImageNet normalization -- StyleSwin's real-image preprocessing (the per-family float
# training space). Reals are fed to D in this normalized space; the exact inverse
# recovers the uint8 [0,255] boundary format used for combra and every saved artifact.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_AMP_DTYPES = {'fp32': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}

#----------------------------------------------------------------------------
# Small helpers (StyleSwin update math, unchanged).

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader, sampler=None):
    # Re-seed the DistributedSampler each epoch so the shard ordering varies epoch-to-epoch
    # (without set_epoch it stays at epoch 0 forever and repeats the same order every pass).
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def data_sampler(dataset, shuffle, distributed, seed):
    if distributed:
        # Seed the distributed sampler from --seed so multi-GPU data order is reproducible
        # and controlled by the seed (a bare DistributedSampler defaults to seed 0).
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle, seed=seed)
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
# The single normalize/denormalize pair (§5), asserted to round-trip.

def _norm_stats(device):
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return mean, std


def _normalize_real(img_u8, device):
    # uint8 NCHW [0,255] -> ImageNet-normalized float (== transforms.ToTensor()+Normalize()).
    mean, std = _norm_stats(device)
    x = img_u8.to(device).float().div_(255.0)
    return (x - mean) / std


def _denorm_to_uint8(images_norm):
    # normalized float NCHW numpy -> uint8 NCHW numpy [0,255] (exact inverse of the real
    # preprocessing), so generated samples cross the boundary on the same uint8 scale.
    mean = np.asarray(_IMAGENET_MEAN, np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(_IMAGENET_STD, np.float32).reshape(1, 3, 1, 1)
    x = images_norm * std + mean
    return np.rint(x * 255.0).clip(0, 255).astype(np.uint8)


def _assert_norm_roundtrip(device):
    # uint8 -> normalize -> denormalize must recover the exact uint8 bytes.
    # arange(48) never exceeds 255, so no wrap is needed. The old `% 256` raised
    # OverflowError on numpy >= 2 (256 is out of range for uint8), which broke
    # training at startup on any fresh install.
    u8 = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(1, 3, 4, 4)
    back = _denorm_to_uint8(_normalize_real(torch.from_numpy(u8), device).cpu().numpy())
    assert np.array_equal(u8, back), 'normalize/denormalize pair does not round-trip'

#----------------------------------------------------------------------------
# Distributed combra metrics: shard ALL per-image extraction across ranks. Every rank runs
# the same collectives, so the caller MUST invoke the eval/precompute on all ranks.

@torch.no_grad()
def _combra_generate_local_shard(G_ema, grid_z, grid_c, batch_gpu, num_gpus, rank):
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


def _combra_precompute_reference(reference_u8_set, ref_indices, device, rank, num_gpus):
    """This rank's slice of the raw uint8 reference, extracted and gathered by combra.

    Returns ``(reference, ok)``; ``ok`` is rank-uniform, so the caller can gate the
    per-tick eval on it (``reference`` is None on every non-zero rank regardless).
    """
    from combra.metrics.distributed import all_ranks_ok, precompute_reference

    my = ref_indices[rank::num_gpus]
    # Loading this rank's slice happens BEFORE combra's own handshake, so it needs its
    # own: a decode error or MemoryError on one rank would otherwise raise there while
    # every other rank was already blocked in the precompute's all_reduce.
    local_u8, ok = None, True
    try:
        local_u8 = np.stack([reference_u8_set[i][0] for i in my])  # NCHW uint8
    except Exception as e:  # noqa: BLE001 -- agreed across ranks below
        ok = False
        print(f'[combra][rank {rank}] reference load failed: {e}', flush=True)
    if not all_ranks_ok(ok, device, num_gpus):
        return None, False
    return precompute_reference(local_u8, device, rank, num_gpus)


def _combra_eval_distributed(G_ema, grid_z, grid_c, batch_gpu, num_gpus, rank, device, combra_ref):
    """combra metrics on rank 0, None elsewhere. Every rank MUST call this: the
    gathers inside are collectives, so a rank that skips them hangs the others."""
    from combra.metrics.distributed import distributed_metrics, gather_generated

    local_u8 = _denorm_to_uint8(
        _combra_generate_local_shard(G_ema, grid_z, grid_c, batch_gpu, num_gpus, rank))
    gen_feats, gen_angles = gather_generated(local_u8, device, rank, num_gpus)
    if rank != 0:
        return None
    # device= so the CMMD reduction runs where the features were extracted.
    return distributed_metrics(combra_ref, gen_angles, gen_feats, device=device)


def _write_run_hparams(run_dir, writer, metrics):
    """Record this run's configuration in TensorBoard's HPARAMS tab (§7).

    The config is read back from ``training_options.json``, which the launcher has
    already written, so no plumbing through the training-loop signature is needed.
    Paired with the run's final metrics, this is what makes two runs comparable in
    TensorBoard: without it the curves are there but nothing says which
    configuration produced them, and comparing runs means diffing two JSON files.
    """
    import json
    import os

    try:
        from combra.io import write_hparams
    except ImportError:
        return  # combra is optional; the curves are still logged
    path = os.path.join(run_dir, 'training_options.json')
    if not os.path.isfile(path):
        return
    with open(path) as fh:
        config = json.load(fh)
    write_hparams(writer, config, metrics)


def build_stats_row(stats_dict, stats_metrics, timestamp, start_time):
    """One scalar JSON row for ``stats.jsonl`` (§7).

    Collector values are flattened to their means. The collector nests every
    scalar as ``{num, mean, std}``, and ``combra.metrics.load_fid_by_kimg`` reads
    ``Progress/kimg`` through ``int()`` -- a nested value makes it shape-filter the
    whole record away, so every run returned ``{}`` with no error and post-hoc
    snapshot selection silently found nothing. TensorBoard already receives
    ``value.mean``, so flattening also makes the file carry the same keys as the
    tags, which is what §7 asks for.

    Kept as a plain function so tests can exercise the real row without running a
    training loop.
    """
    row = {name: value.mean for name, value in stats_dict.items()}
    for name, value in stats_metrics.items():
        row[f'Metrics/{name}'] = float(value)
    row['timestamp'] = timestamp
    row['wall_time'] = timestamp - start_time
    row['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return row


def _sample_labels(class_probs, n, n_classes, device, generator=None):
    idx = torch.multinomial(class_probs, n, replacement=True, generator=generator)
    return F.one_hot(idx, n_classes).float().to(device)

#----------------------------------------------------------------------------

def _startup_header(num_gpus):
    parts = [f'torch {torch.__version__}', f'cuda {torch.version.cuda}', f'gpus {num_gpus}']
    try:
        parts.append('device ' + torch.cuda.get_device_name(0))
    except Exception:
        pass
    for v in ('CUDA_VISIBLE_DEVICES', 'TORCH_CUDA_ARCH_LIST', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        if os.environ.get(v):
            parts.append(f'{v}={os.environ[v]}')
    return ' | '.join(parts)

#----------------------------------------------------------------------------

def training_loop(
    rank                    = 0,
    num_gpus                = 1,
    run_dir                 = '.',
    data_path               = None,
    resolution              = 256,
    lmdb                    = False,
    n_classes               = 0,
    class_names             = None,
    batch_gpu               = 4,
    grad_accum              = 1,
    total_kimg              = 25000,
    kimg_per_tick           = 4,
    snap_ticks              = 50,
    random_seed             = 0,
    workers                 = 3,
    combra_metrics          = True,
    num_fid_samples         = 10000,
    combra_ref_count        = 0,
    snapshot_keep_last      = 3,
    fake_label_sampling     = 'empirical',
    mirror                  = False,
    precision               = 'fp32',
    tf32                    = True,
    bench                   = True,
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
    D_sn                    = False,
):
    device = torch.device('cuda', rank)
    # Per-rank RNG streams (dropout / noise) differ, but the eval/grid latents below derive
    # from --seed alone so the metric image set is identical at any --gpus.
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = bench
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    amp_dtype = _AMP_DTYPES[precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == 'fp16'))
    _assert_norm_roundtrip(device)

    if class_names is None:
        class_names = [str(i) for i in range(max(n_classes, 1))]
    arch = dict(style_dim=style_dim, n_mlp=n_mlp, channel_multiplier=g_channel_multiplier,
                lr_mlp=lr_mlp, enable_full_resolution=enable_full_resolution)

    def stage(msg):
        if rank == 0:
            print(f'[Stage] {msg}', flush=True)

    if rank == 0:
        print('[startup] ' + _startup_header(num_gpus), flush=True)

    # ------------------------------------------------------------------ Dataset.
    if lmdb:
        from torchvision import transforms
        tfm = transforms.Compose([transforms.Resize((resolution, resolution)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)])
        training_set = MultiResolutionDataset(data_path, tfm, resolution)
        reference_u8_set = None
    else:
        # ImageNet-style zip/dir (uint8 CHW + one-hot label). xflip is never applied here:
        # --mirror is a loader-level augmentation (below), so the dataset -- and thus the
        # combra reference -- is never flip-doubled.
        training_set = ImageFolderDataset(path=data_path, use_labels=(n_classes > 0), xflip=False)
        reference_u8_set = training_set  # raw uint8; the fixed combra reference

    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', getattr(training_set, 'image_shape', [3, resolution, resolution]))
        print('Num classes:', n_classes)
        print()

    sampler = data_sampler(training_set, shuffle=True, distributed=(num_gpus > 1), seed=random_seed)
    loader = sample_data(data.DataLoader(training_set, batch_size=batch_gpu, sampler=sampler,
                                         num_workers=workers, drop_last=True, pin_memory=True), sampler)

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

    g_reg_ratio = 1.0                       # StyleSwin: no G regularization
    d_reg_ratio = d_reg_every / (d_reg_every + 1)

    g_module = generator
    if num_gpus > 1:
        generator = nn.parallel.DistributedDataParallel(
            generator, device_ids=[rank], output_device=rank, broadcast_buffers=False)
        discriminator = nn.parallel.DistributedDataParallel(
            discriminator, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # Initialise G_ema from the post-broadcast weights so it is identical on every rank
    # (DDP broadcasts rank 0's parameters at construction). This must happen *after* the DDP
    # wrap: doing it before seeds g_ema from each rank's own random init, and since combra
    # generation is sharded per rank over each rank's g_ema, the metric image set would be a
    # mix of divergent EMAs early in training.
    accumulate(g_ema, g_module, 0)

    g_lr_eff = (d_lr / 4 if ttur else g_lr) * g_reg_ratio
    g_optim = optim.Adam(generator.parameters(), lr=g_lr_eff,
                         betas=(beta1 ** g_reg_ratio, beta2 ** g_reg_ratio))
    d_optim = optim.Adam(discriminator.parameters(), lr=d_lr * d_reg_ratio,
                         betas=(beta1 ** d_reg_ratio, beta2 ** d_reg_ratio))

    # ------------------------------------------------------------------ combra reference.
    combra_metrics = combra_metrics and num_fid_samples > 0   # --num-fid-samples 0 disables eval
    combra_installed = importlib.util.find_spec('combra') is not None
    if combra_metrics and combra_installed and rank == 0 and reference_u8_set is not None:
        # strict=True is the point: the plain self_test() this used to call defaults to
        # image_metrics=False, so it never touched InceptionV3 / CLIP / DINOv2 and a
        # missing CLIP download surfaced only as a whole run logging nan for
        # combra_fid / combra_cmmd / combra_fd_dinov2. images= scores the real
        # reference slice rather than synthetic grains.
        try:
            from combra.metrics import self_test
            sample = np.stack([reference_u8_set[i][0]
                               for i in range(min(4, len(reference_u8_set)))])
            self_test(images=sample, device=device, image_metrics=True, strict=True)
        except Exception as e:
            print(f'[combra] smoke test failed: {e}', flush=True)

    combra_ref = None
    combra_ref_ok = True
    if combra_metrics and combra_installed and (reference_u8_set is not None):
        stage('Precomputing combra reference (sharded over ranks)')
        n_ref = len(reference_u8_set)
        if combra_ref_count and combra_ref_count < n_ref:
            # A capped reference is a SEEDED RANDOM subset -- never the first N (the zip is
            # class-sorted, so a first-N slice is class-biased).
            ref_indices = np.sort(np.random.RandomState(random_seed).permutation(n_ref)[:combra_ref_count]).tolist()
        else:
            ref_indices = list(range(n_ref))
        combra_ref, combra_ref_ok = _combra_precompute_reference(
            reference_u8_set, ref_indices, device, rank, num_gpus)
    if combra_metrics and (rank == 0) and not combra_installed:
        print("Warning: combra_metrics=True but the `combra` package is not installed -- "
              "combra metrics will be skipped. Install it (`pip install -e '.[combra]'`) to "
              "enable them, or pass --combra-metrics False to silence this warning.", flush=True)
    if combra_metrics and lmdb and rank == 0:
        print('Warning: combra metrics require the ImageNet-style dataset (uint8 reference); '
              'skipping combra metrics on the --lmdb path.', flush=True)

    combra_active = combra_metrics and combra_installed and (reference_u8_set is not None) and combra_ref_ok

    # Fixed combra generation set (latents + labels), seeded from --seed alone.
    combra_z = combra_c = None
    if combra_active:
        combra_z = torch.randn([num_fid_samples, style_dim], device=device,
                               generator=torch.Generator(device=device).manual_seed(random_seed + 1))
        if n_classes > 0:
            gcpu = torch.Generator().manual_seed(random_seed)
            probs = class_probs.cpu() if class_probs is not None else torch.ones(n_classes)
            combra_c = _sample_labels(probs, num_fid_samples, n_classes, device, generator=gcpu)

    # Fixed sample-grid latents (class-sorted, resolution-adaptive), seeded from --seed alone.
    grid_z, grid_c, grid_ncol = _make_grid_latents(resolution, n_classes, style_dim, random_seed, device)

    # ------------------------------------------------------------------ Logging.
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            # filename_suffix stamps the event file with the run name so a copied tfevents
            # file stays self-identifying (§7).
            stats_tfevents = tensorboard.SummaryWriter(run_dir, filename_suffix=f'.{os.path.basename(run_dir)}')
        except ImportError as err:
            print('Skipping tfevents export:', err)
        # reals.png (once) + fakes_init.png (untrained G_ema) sample grids.
        _save_reals_grid(reference_u8_set, grid_c, grid_ncol, n_classes, run_dir)
        init_grid = _render_fakes_grid(g_ema, grid_z, grid_c, grid_ncol, batch_gpu, n_classes, device)
        torchvision.utils.save_image(init_grid, os.path.join(run_dir, 'fakes_init.png'))
        if stats_tfevents is not None:
            stats_tfevents.add_image('Fakes', init_grid, global_step=0)

    # ------------------------------------------------------------------ Training loop.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    batch_size = batch_gpu * num_gpus * grad_accum          # the total batch (§2 formula)
    accum = 0.5 ** (batch_size / (10 * 1000))               # EMA half-life ~10 kimg
    best_fid = float('inf')
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    start_time = tick_start_time
    maintenance_time = 0.0
    batch_idx = 0

    def next_real():
        # One micro-batch of ImageNet-normalized reals (+ optional loader-level mirror flip),
        # and the matching labels (or None). combra never sees this flip -- it reads the raw set.
        real_batch = next(loader)
        if lmdb:
            real_img = real_batch.to(device)
            real_labels = None
        elif n_classes > 0:
            real_u8, real_labels = real_batch
            real_img = _normalize_real(real_u8, device)
            real_labels = real_labels.to(device)
        else:
            real_u8, _ = real_batch
            real_img = _normalize_real(real_u8, device)
            real_labels = None
        if mirror:
            flip = torch.rand(real_img.shape[0], device=device) < 0.5
            if flip.any():
                real_img[flip] = real_img[flip].flip(-1)
        return real_img, real_labels

    while True:
        generator.train()

        # -------------------------------------------------- Train D (grad accumulation).
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        discriminator.zero_grad()
        last_real_img = last_real_labels = None
        for _micro in range(grad_accum):
            real_img, real_labels = next_real()
            noise = torch.randn(batch_gpu, style_dim, device=device)
            fake_labels = _sample_labels(class_probs, batch_gpu, n_classes, device) if n_classes > 0 else None
            with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
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
            scaler.scale(d_loss / grad_accum).backward()
            last_real_img, last_real_labels = real_img, real_labels
        scaler.unscale_(d_optim)
        nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
        scaler.step(d_optim)

        # R1 regularization (fp32, no autocast; the gradient penalty needs full precision).
        if batch_idx % d_reg_every == 0:
            last_real_img.requires_grad = True
            real_pred = discriminator(last_real_img, last_real_labels)
            r1_loss = d_r1_loss(real_pred, last_real_img)
            discriminator.zero_grad()
            (gan_weight * (r1 / 2 * r1_loss * d_reg_every + 0 * real_pred[0])).backward()
            d_optim.step()
            training_stats.report('Loss/r1', r1_loss)

        # -------------------------------------------------- Train G (grad accumulation).
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        generator.zero_grad()
        for _micro in range(grad_accum):
            noise = torch.randn(batch_gpu, style_dim, device=device)
            fake_labels = _sample_labels(class_probs, batch_gpu, n_classes, device) if n_classes > 0 else None
            with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
                fake_img, _ = generator(noise, fake_labels)
                fake_pred = discriminator(fake_img, fake_labels)
                g_loss = g_nonsaturating_loss(fake_pred) * gan_weight
            training_stats.report('Loss/G/loss', g_loss)
            scaler.scale(g_loss / grad_accum).backward()
        scaler.step(g_optim)
        scaler.update()

        accumulate(g_ema, g_module, accum)

        cur_nimg += batch_size
        batch_idx += 1

        # ---------------------------------------------- tick maintenance.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        tick_end_time = time.time()
        fields = []
        fields += [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"]
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
        training_stats.report0('LearningRate/G', g_optim.param_groups[0]['lr'])
        training_stats.report0('LearningRate/D', d_optim.param_groups[0]['lr'])
        if rank == 0:
            print(' '.join(fields))

        # Snapshot: metrics + checkpoint. Skipped at tick 0 (untrained G_ema); always at the
        # last tick so the newest snapshot IS the final model (§3).
        # Clear the combra row first: the metrics are only recomputed at snapshot
        # ticks, so a dict that persists across ticks re-emits the previous tick's
        # values at a new step -- turning the metric curves into step functions and
        # letting post-hoc snapshot selection resolve to a kimg never evaluated.
        stats_metrics = {}
        snapshot = (done or (cur_tick > 0 and cur_tick % snap_ticks == 0))
        if snapshot:
            g_ema.eval()

            if combra_active:
                stage('Evaluating combra metrics')
                eval_start = time.time()
                try:
                    combra_results = _combra_eval_distributed(
                        g_ema, combra_z, combra_c, batch_gpu, num_gpus, rank, device, combra_ref)
                except Exception as e:
                    combra_results = None
                    # Every rank prints: the rank that fails is rarely rank 0, and a
                    # rank-0-only print left the actual error invisible while the run
                    # reported only that "combra metrics failed" somewhere.
                    print(f'[combra][rank {rank}] metric evaluation failed: {e}', flush=True)
                # Outside the rank guard on purpose. report0 registers the counter NAME
                # on whichever rank calls it (before it discards non-zero ranks' values),
                # and Collector.update() all_reduces over the registered set -- so a name
                # only rank 0 ever reported makes that reduction disagree on shape.
                training_stats.report0('Timing/eval_sec', time.time() - eval_start)
                if rank == 0 and combra_results is not None:
                    # Bare keys (combra_fid, not combra_fid10k): the old `10k` suffix
                    # was a literal that stayed 10k whatever --num-fid-samples said, so
                    # every chart built from it was mislabelled. The count is logged
                    # once, as its own scalar.
                    for name, value in combra_results.items():
                        stats_metrics[f'combra_{name}'] = float(value)
                    stats_metrics['combra_num_fid_samples'] = float(num_fid_samples)
                    if 'combra_fid' in stats_metrics:
                        best_fid = min(best_fid, stats_metrics['combra_fid'])
                        stats_metrics['combra_fid_best'] = best_fid
                    print('combra metrics: ' + ', '.join(
                        f'{k}={v:.4f}' for k, v in combra_results.items()), flush=True)

            if rank == 0:
                grid = _render_fakes_grid(g_ema, grid_z, grid_c, grid_ncol, batch_gpu, n_classes, device)
                torchvision.utils.save_image(grid, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'))
                if stats_tfevents is not None:
                    stats_tfevents.add_image('Fakes', grid, global_step=cur_nimg)
                # The single artifact kind: EMA-only weights + self-describing metadata,
                # written atomically, pruned to --snapshot-keep-last.
                snapshot_data = {
                    'g_ema': g_ema.state_dict(),
                    'n_classes': n_classes, 'resolution': resolution,
                    'class_names': class_names, 'cur_nimg': cur_nimg, 'arch': arch,
                }
                _atomic_save(snapshot_data,
                             os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}-inference.pt'),
                             run_dir)
                if snapshot_keep_last > 0:
                    old_snaps = sorted(glob.glob(os.path.join(run_dir, 'network-snapshot-*-inference.pt')))
                    for old in old_snaps[:-snapshot_keep_last]:
                        os.remove(old)

        # Update logs.
        timestamp = time.time()
        stats_collector.update()
        # NOTE: stats_metrics was cleared before the snapshot block above, so a tick
        # with no combra eval writes no combra columns rather than repeating the
        # previous tick's values at a new step.
        stats_dict = stats_collector.as_dict()
        if stats_jsonl is not None:
            row = build_stats_row(stats_dict, stats_metrics, timestamp, start_time)
            stats_jsonl.write(json.dumps(row) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            gstep = cur_nimg                          # global step = cur_nimg (§7)
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
            if stats_tfevents is not None:
                _write_run_hparams(run_dir, stats_tfevents,
                                   {'Metrics/combra_fid_best': float(best_fid)})
            stats_jsonl.close()

#----------------------------------------------------------------------------

def _atomic_save(obj, path, run_dir):
    # Write to a temp file in the same directory, then os.replace into place, so a snapshot
    # present under its final name is always complete (§3 atomic-write MUST).
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=run_dir, suffix='.tmp')
    os.close(fd)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _cpu_mem_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**30
    except Exception:
        return 0.0


def _make_grid_latents(resolution, n_classes, style_dim, seed, device):
    # Fixed, class-sorted, resolution-adaptive grid: fewer images at higher resolution. For a
    # labeled model each row is one class (class-sorted rows); unconditional is a square grid.
    ncol = 8 if resolution <= 256 else 4 if resolution <= 512 else 2
    nrow = max(n_classes, 1) if n_classes > 0 else ncol
    n = nrow * ncol
    z = torch.randn([n, style_dim], device=device,
                    generator=torch.Generator(device=device).manual_seed(seed + 2))
    c = None
    if n_classes > 0:
        idx = torch.arange(n, device=device) // ncol % n_classes   # row r -> class r
        c = F.one_hot(idx, n_classes).float()
    return z, c, ncol


@torch.no_grad()
def _render_fakes_grid(g_ema, grid_z, grid_c, ncol, batch_gpu, n_classes, device):
    n = grid_z.shape[0]
    nb = (n + batch_gpu - 1) // batch_gpu
    c_splits = grid_c.split(batch_gpu) if grid_c is not None else [None] * nb
    imgs = [g_ema(zz, cc)[0] if n_classes > 0 else g_ema(zz)[0]
            for zz, cc in zip(grid_z.split(batch_gpu), c_splits)]
    img = torch.cat(imgs, dim=0)
    mean, std = _norm_stats(device)
    img = (img * std + mean).clamp(0, 1)
    return torchvision.utils.make_grid(img, nrow=ncol, padding=0)


def _save_reals_grid(reference_u8_set, grid_c, ncol, n_classes, run_dir):
    # Build reals.png once from raw dataset samples, class-sorted to mirror the fakes grid.
    if reference_u8_set is None:
        return
    try:
        if n_classes > 0:
            raw = np.asarray(reference_u8_set._get_raw_labels()).astype(np.int64)
            classes = grid_c.argmax(dim=1).cpu().numpy()
            rng = np.random.RandomState(0)
            picks = []
            for cl in classes:
                where = np.where(raw == cl)[0]
                picks.append(int(rng.choice(where)) if len(where) else 0)
        else:
            picks = list(range(min(ncol * ncol, len(reference_u8_set))))
        imgs = np.stack([reference_u8_set[i][0] for i in picks]).astype(np.float32) / 255.0
        grid = torchvision.utils.make_grid(torch.from_numpy(imgs), nrow=ncol, padding=0)
        torchvision.utils.save_image(grid, os.path.join(run_dir, 'reals.png'))
    except Exception as e:
        print(f'[warn] could not write reals.png: {e}', flush=True)

#----------------------------------------------------------------------------
