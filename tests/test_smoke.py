# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Smoke tests for the CLI contract (§2/§4).

These parse ``--help`` for each console entry point and assert the contract flags exist and
the removed ones are gone. Importing the training/generation modules pulls in the ``op`` CUDA
extension, which JIT-compiles on import and needs a CUDA toolchain, so these self-skip when
torch is missing or no CUDA device is available (e.g. the CPU CI runner).
"""

import importlib.util
import subprocess
import sys

import pytest


def _cuda_unavailable():
    if importlib.util.find_spec('torch') is None:
        return True
    import torch
    return not torch.cuda.is_available()


pytestmark = pytest.mark.skipif(_cuda_unavailable(),
                                reason='no CUDA toolchain (the op extension compiles on import)')


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
