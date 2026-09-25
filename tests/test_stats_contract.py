"""The row this repo writes to ``stats.jsonl`` must be readable by combra.

``combra.metrics.load_fid_by_kimg`` reads ``Metrics/combra_fid`` and
``Progress/kimg`` from the same JSON line, and shape-filters away any record whose
values are not plain scalars -- silently, returning ``{}``. That is exactly how
san-v2 and StyleSwin runs produced an unreadable metric history while every
combra-side test passed: the reader was tested against a synthetic *flat* row, and
nothing tested the producer.

This is the producer half. It builds the real row through the training loop's own
function, so a change to the row shape fails here instead of silently emptying the
analysis layer.
"""

import importlib.util
import json

import pytest

pytest.importorskip("torch")  # the training-loop module imports torch at module level

requires_combra = pytest.mark.skipif(
    importlib.util.find_spec("combra") is None, reason="combra is not installed"
)


def _row():
    from types import SimpleNamespace

    from training.training_loop import build_stats_row

    # The collector nests every scalar as {num, mean, std}; build_stats_row must
    # flatten it to the mean, which is what TensorBoard receives too.
    stats_dict = {
        "Progress/kimg": SimpleNamespace(mean=403.2),
        "Loss/G/loss": SimpleNamespace(mean=1.25),
    }
    return build_stats_row(stats_dict, {"combra_fid": 12.5}, 1000.0, 900.0)



def test_row_contains_only_json_scalars():
    for key, value in _row().items():
        assert value is None or isinstance(value, (int, float, str)), (
            f"{key} is {type(value).__name__}, not a JSON scalar -- "
            "load_fid_by_kimg will shape-filter this record away"
        )


@requires_combra
def test_row_round_trips_through_load_fid_by_kimg(tmp_path):
    from combra.metrics import load_fid_by_kimg

    path = tmp_path / "stats.jsonl"
    path.write_text(json.dumps(_row()) + "\n")
    assert load_fid_by_kimg(str(path)) == {"000403": 12.5}


def test_non_finite_values_are_written_as_null():
    # A bare NaN token is not valid JSON; the row writes null, which load_fid_by_kimg skips.
    from types import SimpleNamespace

    from training.training_loop import build_stats_row

    stats_dict = {
        "Progress/kimg": SimpleNamespace(mean=403.2),
        "Progress/tick": SimpleNamespace(mean=7.0),
        "Loss/r1": SimpleNamespace(mean=float("nan")),
    }
    row = build_stats_row(stats_dict, {"combra_fid": float("inf")}, 1000.0, 900.0)
    back = json.loads(json.dumps(row, allow_nan=False))
    assert back["Loss/r1"] is None and back["Metrics/combra_fid"] is None
    assert back["Progress/tick"] == 7 and isinstance(back["Progress/tick"], int)


def test_collector_drops_scalars_not_reported_this_tick():
    # §7: a scalar not reported this tick is left out, not repeated. R1 runs every
    # --d-reg-every iterations, so a short tick can report no Loss/r1 at all.
    from torch_utils import training_stats
    from training.training_loop import reported_this_tick

    collector = training_stats.Collector(regex="Test/.*", keep_previous=False)
    training_stats.report("Test/every_iter", 1.0)
    training_stats.report("Test/r1", 3.0)
    collector.update()
    first = reported_this_tick(collector.as_dict())
    assert first["Test/r1"].mean == 3.0 and first["Test/every_iter"].mean == 1.0

    training_stats.report("Test/every_iter", 2.0)
    collector.update()
    second = reported_this_tick(collector.as_dict())
    assert "Test/r1" not in second
    assert second["Test/every_iter"].mean == 2.0


def test_training_loop_collector_does_not_keep_previous():
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "training/training_loop.py").read_text()
    assert "Collector(regex='.*', keep_previous=False)" in src
    assert "reported_this_tick(stats_collector.as_dict())" in src
