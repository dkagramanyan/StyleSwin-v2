# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU smoke tests for the CLI contract (§2/§4).

These parse ``--help`` for each console entry point and assert the contract flags exist and
the removed ones are gone. They need torch (imported by the training/generation modules), so
they self-skip where torch is not installed (e.g. the lint-only CI runner).
"""

import importlib.util
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec('torch') is None,
                                reason='torch not installed (CPU-only runner)')


def _help(script):
    r = subprocess.run([sys.executable, script, '--help'], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_train_cli_contract():
    out = _help('train.py')
    for flag in ('--precision', '--tf32', '--bench', '--grad-accum', '--mirror',
                 '--num-fid-samples', '--combra-ref-count', '--snapshot-keep-last', '--cfg'):
        assert flag in out, f'missing {flag}'
    for gone in ('--resume', '--save-inference-only', '--metrics', '--use-flip'):
        assert gone not in out, f'{gone} should have been removed'


def test_gen_images_cli_contract():
    out = _help('gen_images.py')
    for flag in ('--save-mode', '--network', '--classes', '--samples-per-class',
                 '--batch-gpu', '--gpus', '--desc'):
        assert flag in out, f'missing {flag}'


def test_eval_cli_contract():
    out = _help('eval.py')
    for flag in ('--network', '--data', '--num-fid-samples', '--combra-ref-count'):
        assert flag in out, f'missing {flag}'
