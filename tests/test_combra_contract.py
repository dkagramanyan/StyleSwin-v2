"""Assert every combra symbol this repo imports actually exists.

The eval path is deliberately fault-tolerant: a missing or broken combra must
never abort a training run. That tolerance is also how a real breakage hid for a
whole release. combra 0.5.0 removed ``angle_density_metrics_from_pooled``,
``fid_from_features`` and ``fd_dinov2_from_features`` and renamed
``combra_smoke_test``; this repo kept importing all four, the ``except`` around
the eval swallowed the ImportError every tick, and ``--combra-metrics true``
quietly produced nothing for months.

Nothing caught it because nothing asserted the symbols. This does — CPU-only, no
GPU, no dataset, no network, so it runs in every CI job. It is skipped only when
combra is genuinely absent, which is a different (and visible) condition.
"""

import importlib.util

import pytest

# Every name this repo imports from combra, in one place. Adding a combra call
# anywhere in the repo means adding its name here.
REQUIRED = [
    "angle_density_metrics_from_pooled",
    "cmmd_features",
    "cmmd_from_features",
    "fd_dinov2_features",
    "fid_features",
    "frechet_from_features",
    "images_to_pooled_angles",
    "self_test",
]

combra_installed = importlib.util.find_spec("combra") is not None
requires_combra = pytest.mark.skipif(not combra_installed, reason="combra is not installed")


@requires_combra
@pytest.mark.parametrize("name", REQUIRED)
def test_combra_exports_symbol(name):
    import combra.metrics

    assert hasattr(combra.metrics, name), (
        f"combra.metrics.{name} is missing. This repo's eval path imports it; without it "
        "every combra metric silently disappears. Check combra's CHANGELOG for a rename."
    )


@requires_combra
def test_combra_import_block_resolves():
    # The exact import the training loop performs. Guarded there, unguarded here.
    from combra.metrics import (  # noqa: F401
        angle_density_metrics_from_pooled,
        cmmd_features,
        cmmd_from_features,
        fd_dinov2_features,
        fid_features,
        frechet_from_features,
    )


@requires_combra
def test_angle_metrics_run_on_pooled_angles():
    # Not just importable -- callable, and returning the keys the loop logs.
    import numpy as np
    from combra.metrics import angle_density_metrics_from_pooled

    rng = np.random.default_rng(0)
    out = angle_density_metrics_from_pooled(
        rng.normal(120, 25, 4000) % 360, rng.normal(126, 27, 4000) % 360
    )
    for key in ("w1", "w2", "circular_w1", "circular_w2", "mu1", "sigma1", "amp1"):
        assert np.isfinite(out[key]), f"{key} is not finite"
