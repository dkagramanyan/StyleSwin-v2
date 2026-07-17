# Ensure the repo root is importable when the tests run under the bare `pytest`
# launcher (which, unlike `python -m pytest`, does not put the CWD on sys.path), so the
# in-tree `utils` / `dataset_tool` modules resolve without installing the whole tree.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
