"""Snapshot retention: the N newest plus the best by each of fid / fd_dinov2 / cmmd."""

import math

import pytest

pytest.importorskip("torch")  # the training-loop module imports torch at module level


def _name(kimg):
    return f"styleswin-snapshot-{kimg:06d}-inference.pt"


def _run(metrics_per_snap, keep_last=1):
    """Replay the loop's save -> update-best -> prune sequence over a list of snapshots."""
    from training.training_loop import _snapshots_to_prune, _update_best_snapshots

    on_disk, best = set(), {}
    for kimg, metrics in metrics_per_snap:
        on_disk.add(_name(kimg))
        _update_best_snapshots(best, metrics, _name(kimg))
        protected = {n for _, n in best.values()}
        for old in _snapshots_to_prune(on_disk, keep_last, protected):
            on_disk.remove(old)
        assert protected <= on_disk  # a best is never pruned
    return on_disk, best


def _m(fid, dino, cmmd):
    return {"combra_fid": fid, "combra_fd_dinov2": dino, "combra_cmmd": cmmd}


def test_keeps_last_plus_one_best_per_metric():
    on_disk, best = _run([
        (100, _m(50.0, 900.0, 2.0)),
        (200, _m(30.0, 950.0, 3.0)),   # best fid
        (300, _m(40.0, 700.0, 3.5)),   # best dino
        (400, _m(45.0, 800.0, 1.0)),   # best cmmd
        (500, _m(60.0, 990.0, 4.0)),   # last, best at nothing
    ])
    assert on_disk == {_name(200), _name(300), _name(400), _name(500)}
    assert best == {"combra_fid": (30.0, _name(200)),
                    "combra_fd_dinov2": (700.0, _name(300)),
                    "combra_cmmd": (1.0, _name(400))}


def test_one_file_can_serve_several_roles():
    on_disk, _ = _run([
        (100, _m(50.0, 900.0, 2.0)),
        (200, _m(10.0, 100.0, 0.5)),   # best at everything
        (300, _m(20.0, 200.0, 0.9)),
    ])
    assert on_disk == {_name(200), _name(300)}


def test_nan_is_ignored_and_missing_metrics_keep_last_only():
    on_disk, best = _run([
        (100, _m(math.nan, math.nan, math.nan)),
        (200, _m(20.0, math.nan, math.inf)),
        (300, {}),                      # combra off / failed this tick
        (400, _m(25.0, math.nan, math.nan)),
    ])
    assert best == {"combra_fid": (20.0, _name(200))}
    assert on_disk == {_name(200), _name(400)}


def test_at_most_four_files_over_a_long_run():
    import random

    rng = random.Random(0)
    snaps = [(k, _m(rng.random(), rng.random(), rng.random())) for k in range(1, 200)]
    on_disk, _ = _run(snaps)
    assert len(on_disk) <= 4 and _name(199) in on_disk


def test_keep_last_zero_keeps_all_and_n_keeps_n_newest():
    from training.training_loop import _snapshots_to_prune

    snaps = [_name(k) for k in (300, 100, 200, 400)]
    assert _snapshots_to_prune(snaps, 0, set()) == []
    assert _snapshots_to_prune(snaps, 2, set()) == [_name(100), _name(200)]
    assert _snapshots_to_prune(snaps, 2, {_name(100)}) == [_name(200)]
