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
        assert isinstance(value, (int, float, str)), (
            f"{key} is {type(value).__name__}, not a JSON scalar -- "
            "load_fid_by_kimg will shape-filter this record away"
        )


@requires_combra
def test_row_round_trips_through_load_fid_by_kimg(tmp_path):
    from combra.metrics import load_fid_by_kimg

    path = tmp_path / "stats.jsonl"
    path.write_text(json.dumps(_row()) + "\n")
    assert load_fid_by_kimg(str(path)) == {"000403": 12.5}
