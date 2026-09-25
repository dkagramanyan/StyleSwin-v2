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

# Every (module, name) this repo imports from combra, in one place. Adding a
# combra call anywhere in the repo means adding its name here. The training loop
# reaches the feature / angle / Gaussian-fit functions only through
# combra.metrics.distributed, so those four symbols are the load-bearing ones.
REQUIRED = [
    ("combra.metrics.distributed", "all_ranks_ok"),
    ("combra.metrics.distributed", "distributed_metrics"),
    ("combra.metrics.distributed", "gather_generated"),
    ("combra.metrics.distributed", "precompute_reference"),
    ("combra.metrics", "self_test"),
    ("combra.io", "write_hparams"),
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
@pytest.mark.parametrize("module, name", REQUIRED)
def test_combra_exports_symbol(module, name):
    mod = importlib.import_module(module)

    assert hasattr(mod, name), (
        f"{module}.{name} is missing. This repo imports it; without it the combra "
        "metrics silently disappear. Check combra's CHANGELOG for a rename."
    )


@requires_combra
def test_combra_import_block_resolves():
    # The exact imports the training loop performs. Guarded there, unguarded here.
    from combra.io import write_hparams  # noqa: F401
    from combra.metrics import self_test  # noqa: F401
    from combra.metrics.distributed import (  # noqa: F401
        all_ranks_ok,
        distributed_metrics,
        gather_generated,
        precompute_reference,
    )


@requires_combra
def test_write_hparams_accepts_step():
    # combra >= 0.15.3: step= writes the hparams into the run's own event file at the
    # final cur_nimg, which is how the training loop calls it.
    import inspect

    from combra.io import write_hparams

    assert "step" in inspect.signature(write_hparams).parameters


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
    for key in ("w1", "w2", "circular_w1", "circular_w2", "mu1", "sigma1", "pi"):
        assert np.isfinite(out[key]), f"{key} is not finite"


@pytest.mark.parametrize("conditional", [False, True])
def test_generate_local_shard_matches_per_batch_denorm(conditional):
    # The shard is filled in place into one preallocated uint8 array; it must hold
    # exactly this rank's strided slice, batch by batch, in order.
    import numpy as np
    torch = pytest.importorskip("torch")

    from training.training_loop import _combra_generate_local_shard, _denorm_to_uint8

    def G(z, c=None):
        x = z[:, :1, None, None].expand(-1, 3, 2, 2) * 0.1
        if c is not None:
            x = x + c.argmax(1)[:, None, None, None] * 0.01
        return (x,)

    z = torch.randn(7, 4)
    c = torch.eye(3)[torch.arange(7) % 3] if conditional else None
    out = _combra_generate_local_shard(G, z, c, batch_gpu=2, num_gpus=2, rank=1)
    idx = torch.arange(1, 7, 2)
    ref = _denorm_to_uint8(G(z[idx], c[idx] if conditional else None)[0].numpy())
    assert out.dtype == np.uint8 and out.shape == (3, 3, 2, 2)
    assert np.array_equal(out, ref)
