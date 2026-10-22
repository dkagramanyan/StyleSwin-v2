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
    ("combra.io", "write_hparams"),
]

combra_installed = importlib.util.find_spec("combra") is not None
requires_combra = pytest.mark.skipif(not combra_installed, reason="combra is not installed")

# The angle keys of each angle version (combra >= 0.25): four Wasserstein distances plus
# the bimodal fit's relative errors -- two amplitudes for v1, one mass share `pi` for v2.
_WASS = ("w1", "w2", "circular_w1", "circular_w2")
ANGLE_KEYS = {
    "v1": (*_WASS, "mu1", "mu2", "sigma1", "sigma2", "amp1", "amp2"),
    "v2": (*_WASS, "mu1", "mu2", "sigma1", "sigma2", "pi"),
}


def _bimodal_angles(rng, mu1, sigma1, mu2, sigma2, n=4000, share=0.7):
    # WC-Co vertex angles: a convex mode and a reflex one.
    import numpy as np

    k = int(n * share)
    both = [rng.normal(mu1, sigma1, k), rng.normal(mu2, sigma2, n - k)]
    return np.concatenate(both) % 360


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
    out = angle_density_metrics_from_pooled(
        _bimodal_angles(rng, 100, 20, 240, 25), _bimodal_angles(rng, 104, 21, 236, 26)
    )
    for key in ANGLE_KEYS["v1"]:
        assert np.isfinite(out[key]), f"{key} is not finite"


@requires_combra
def test_distributed_metrics_keys_map_to_three_sections():
    # The real combra key set, through the one key -> tag mapping every sink uses:
    # image metrics under Metrics/, each angle version under its own Metrics_vN/.
    import numpy as np
    pytest.importorskip("torch")
    from combra.metrics.distributed import distributed_metrics

    from training.training_loop import build_stats_row, metric_tag, metrics_log_lines

    rng = np.random.default_rng(0)

    def side():
        angles = {m: _bimodal_angles(rng, 100, 20, 240, 25) for m in ("v1", "v2")}
        feat = {n: rng.normal(size=(8, 4)).astype(np.float32) for n in ("fid", "cmmd", "fd_dinov2")}
        return angles, feat

    (ref_angles, ref_feat), (gen_angles, gen_feat) = side(), side()
    out = distributed_metrics({"angles": ref_angles, "feat": ref_feat}, gen_angles, gen_feat, device="cpu")
    # As the training loop builds its row.
    stats_metrics = {f"combra_{k}": float(v) for k, v in out.items()}
    stats_metrics["combra_num_fid_samples"] = 64.0
    stats_metrics["combra_fid_best"] = stats_metrics["combra_fid"]

    sections = {}
    for name in stats_metrics:
        section, key = metric_tag(name).split("/")
        sections.setdefault(section, []).append(key)
    assert sections == {
        "Metrics": ["combra_fid", "combra_cmmd", "combra_fd_dinov2", "combra_num_fid_samples", "combra_fid_best"],
        **{f"Metrics_{m}": [f"combra_{k}" for k in keys] for m, keys in ANGLE_KEYS.items()},
    }

    row = build_stats_row({}, stats_metrics, 1000.0, 900.0)
    assert {k for k in row if "/" in k} == {metric_tag(n) for n in stats_metrics}

    lines = metrics_log_lines(stats_metrics)
    assert [line.split(": ")[0] for line in lines] == ["Metrics", "Metrics v1", "Metrics v2"]
    assert lines[0].startswith("Metrics: combra_fid ") and "combra_fid_best" in lines[0]
    for line, (m, keys) in zip(lines[1:], ANGLE_KEYS.items()):
        assert line.startswith(f"Metrics {m}: combra_w1 ")
        assert line.endswith(f"combra_{keys[-1]} {out[f'{m}_{keys[-1]}']:.4f}")


def test_startup_header_records_combra_version():
    pytest.importorskip("torch")
    from training.training_loop import _startup_header

    # A walltime-killed run never reaches the end-of-run HPARAMS, so the startup line is
    # its only record of which combra (angle method) produced its numbers.
    expected = importlib.import_module("combra").__version__ if combra_installed else "not installed"
    assert f"combra {expected}" in _startup_header(1).split(" | ")


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
