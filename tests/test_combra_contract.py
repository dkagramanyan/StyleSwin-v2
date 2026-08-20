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
import os

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


def test_combra_is_installed_when_required():
    """CI sets COMBRA_REQUIRED=1 once it has installed combra; from then on an
    absent combra is a FAILURE, not a skip. Every test below is skipif-guarded, so
    without this one the whole file can go green by doing nothing -- which is the
    exact failure mode it exists to prevent."""
    if os.environ.get("COMBRA_REQUIRED") == "1":
        assert combra_installed, "COMBRA_REQUIRED=1 but combra is not importable"


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

    # The sample must be genuinely BIMODAL. These are WC-Co vertex angles -- a
    # convex mode and a reflex one -- and the gauss half of this metric fits two
    # Gaussians to them. A single normal (which is what this fixture used to pass)
    # leaves the second Gaussian with nothing to sit on, so combra reports the
    # relative errors as nan rather than dividing by a phantom mode.
    rng = np.random.default_rng(0)

    def angles(mu1, sigma1, mu2, sigma2, n=4000, share=0.7):
        k = int(n * share)
        both = [rng.normal(mu1, sigma1, k), rng.normal(mu2, sigma2, n - k)]
        return np.concatenate(both) % 360

    out = angle_density_metrics_from_pooled(
        angles(100, 20, 240, 25), angles(104, 21, 236, 26)
    )
    for key in ("w1", "w2", "circular_w1", "circular_w2", "mu1", "sigma1", "amp1"):
        assert np.isfinite(out[key]), f"{key} is not finite"
