# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pre-download the combra image-metric backbones for offline / cluster nodes:
InceptionV3 (FID), CLIP (CMMD) and DINOv2 (FD-DINOv2). Run once on a networked
node so the training-time combra eval finds them cached.

StyleSwin itself has no separate weights to fetch -- it is a GAN trained from
scratch -- so this only warms the combra backbones (the analog of edm2's
``download_models.py``, minus the latent-diffusion VAE).

Installed by ``pip install -e .`` as the ``styleswin-download-models`` command:

    styleswin-download-models
"""

import click
import numpy as np
import torch


@click.command()
def main():
    """Download and cache the combra metric backbones (InceptionV3 / CLIP / DINOv2)."""
    try:
        from combra.metrics import cmmd_features, fd_dinov2_features, fid_features
    except ImportError:
        print("combra not installed; nothing to fetch. "
              "Install it with `pip install -e '.[combra]'`.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # A tiny NHWC uint8 batch; each extractor loads (and caches) its backbone on
    # first call, which is all we want here.
    dummy = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    print('Fetching combra metric backbones (InceptionV3 / CLIP / DINOv2) ...')
    fid_features(dummy, device=device)
    cmmd_features(dummy, device=device)
    fd_dinov2_features(dummy, device=device)
    print('  done')


if __name__ == '__main__':
    main()
