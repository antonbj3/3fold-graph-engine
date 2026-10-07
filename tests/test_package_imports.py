"""Local imports must work without pytest's sibling-directory path additions."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", [
    "graph_engine.oed_probe_selection_engine_v2_instance",
    "graph_engine.paper_graph.unified_graph_engine",
    "graph_engine.tools.context_loader_eval",
])
def test_package_import_with_only_src_on_path(module, tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_compat_demo_as_direct_script(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/graph_engine/paper_graph/unified_graph_engine.py")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
