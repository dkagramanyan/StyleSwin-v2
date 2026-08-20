"""Every declared console script must answer ``--help`` from a foreign cwd.

This is a *packaging* test, not a CLI test. It catches the case where an entry
point is declared in ``[project.scripts]`` but the sibling package dirs it imports
are not installed, so the script only runs from the repo root. StyleSwin shipped
exactly that way -- 3 of its 5 scripts died on ``No module named 'dnnlib'`` /
``'models'`` -- and no in-repo test could see it, because pytest puts the repo root
on ``sys.path`` and masks the missing install.

The same sweep also catches import-time landmines: san-v2's ``san-eval`` was
unreachable because the vendored CLIP tokenizer was built at module import and its
vocab file is not in the repo, and again because ``dill`` was used but undeclared.
Both are invisible unless the script is actually launched.
"""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _declared_scripts():
    cfg = _ROOT / "pyproject.toml"
    if not cfg.is_file():
        return []
    with cfg.open("rb") as fh:
        return sorted(tomllib.load(fh).get("project", {}).get("scripts", {}))


SCRIPTS = _declared_scripts()


@pytest.mark.parametrize("name", SCRIPTS)
def test_console_script_runs_from_foreign_cwd(name, tmp_path):
    exe = Path(sys.executable).parent / name
    if not exe.is_file():
        pytest.skip(f"{name} not installed in this env (needs `pip install -e .`)")

    # PATH carries the env's bin/ so JIT-compiled CUDA ops can find ninja; cwd is
    # deliberately NOT the repo root -- that is the whole point of the check.
    env = dict(os.environ, PATH=f"{exe.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    proc = subprocess.run(
        [str(exe), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"`{name} --help` failed from {tmp_path}:\n"
        f"{(proc.stderr or proc.stdout).strip()[-1500:]}"
    )


def test_every_declared_script_is_named_for_its_repo():
    """The v2 convention: `<model>-<verb>`, so the four repos never collide on PATH."""
    assert SCRIPTS, "no [project.scripts] declared"
    prefixes = {name.split("-", 1)[0] for name in SCRIPTS}
    assert len(prefixes) == 1, f"mixed script prefixes: {sorted(prefixes)}"
