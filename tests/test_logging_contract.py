"""The §7 TensorBoard / stats.jsonl contract, asserted against the training loop.

These namespaces are the contract the wc_cv analysis layer and every cross-repo
comparison read. They drifted apart once already -- thirteen keys differed across the
four repos: two logged the global step in kimg and two in images, one spelled the
GPU-memory keys differently, one filed learning rate under ``Loss/``, one logged no
sample grid at all. Nothing failed, because nothing checked.

This is a source-level check on purpose: the tags are emitted deep inside a training
loop that needs GPUs and a dataset to run, so asserting on the source is what can
actually run in CI.
"""

import pathlib

import pytest

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "training/training_loop.py"

REQUIRED_TAGS = [
    "Progress/kimg",
    "Progress/tick",
    "Timing/sec_per_tick",
    "Timing/sec_per_kimg",
    "Timing/total_sec",
    "Timing/maintenance_sec",
    "Timing/eval_sec",
    "Resources/cpu_mem_gb",
    "Resources/peak_gpu_mem_gb",
    "Resources/peak_gpu_mem_reserved_gb",
    "LearningRate/",
    "Metrics/",
]

RETIRED_TAGS = [
    "Resources/gpu_mem_gb",          # -> Resources/peak_gpu_mem_gb
    "Resources/gpu_reserved_gb",     # -> Resources/peak_gpu_mem_reserved_gb
    "Loss/learning_rate",            # -> LearningRate/lr
    "Timing/total_hours",            # san-v2-only, just total_sec rescaled
    "Timing/total_days",
]


@pytest.fixture(scope="module")
def source():
    return SOURCE.read_text()


@pytest.mark.parametrize("tag", REQUIRED_TAGS)
def test_required_tag_is_logged(source, tag):
    assert tag in source, f"{tag} is missing from {SOURCE.name} (§7 logging contract)"


@pytest.mark.parametrize("tag", RETIRED_TAGS)
def test_retired_tag_is_gone(source, tag):
    assert tag not in source, f"{tag} was renamed; {SOURCE.name} still emits it"


def test_sample_grid_reaches_tensorboard(source):
    # §7 lists `Fakes` as an image every snapshot tick. san-v2 wrote the grid to disk
    # only, so the repo the proposal calls the reference implementation was the one
    # place the grid never reached TensorBoard.
    assert "add_image" in source and "Fakes" in source


def test_event_file_is_self_identifying(source):
    # §7: the run name is carried as a filename_suffix so a tfevents file copied out of
    # its run directory stays identifiable -- which is how wc_cv/ml/ stores them.
    assert "filename_suffix" in source


def test_hyperparameters_are_recorded(source):
    # Without this the HPARAMS tab is empty and two runs can be compared by their
    # curves but not by the configuration that produced them.
    assert "_write_run_hparams" in source
