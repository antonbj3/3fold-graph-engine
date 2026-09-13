"""Tools and examples whose selftest is a CLI entry point."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
TOOLS = SRC / "graph_engine" / "tools"
EXAMPLES = REPO_ROOT / "examples"


def _env():
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), str(TOOLS), env.get("PYTHONPATH", "")])
    return env


@pytest.mark.parametrize("script", ["atoms_lint.py", "atoms_validation.py", "dataset_profiler.py"])
def test_tool_cli_selftest(script, tmp_path):
    env = _env()
    env["ATOMS_VALIDATION_OUT"] = str(tmp_path)
    env["FOLD_GATE_OUT"] = str(tmp_path)
    r = subprocess.run([sys.executable, str(TOOLS / script), "--selftest"],
                       capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


@pytest.mark.parametrize("script", ["fold_gate.py", "fold_gate_v2.py"])
def test_fold_gate_cli_selftest(script, tmp_path):
    env = _env()
    env["FOLD_GATE_OUT"] = str(tmp_path)
    r = subprocess.run([sys.executable, str(TOOLS / script), "--selftest"],
                       capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def test_graph_hole_engine_tests():
    r = subprocess.run([sys.executable, str(SRC / "graph_engine" / "graph_hole_engine" / "tests.py")],
                       capture_output=True, text=True, env=_env())
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "passed" in r.stdout


def test_inverse_design_spine_regression_suite():
    r = subprocess.run([sys.executable, str(EXAMPLES / "inverse_design" / "spine_regression_suite.py")],
                       capture_output=True, text=True, env=_env())
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]


@pytest.mark.parametrize("cell", [
    "vibrating_multitone_OED_bundle_FDT_lockin_reads_fisher_spectrum_model_free.py",
    "multimodal_dither_lifts_null_space_active_two_kinds_acquisition_fisher_adds.py",
    "fleet_oed_vibrate_the_fleet_read_collective_soft_mode_and_next_best_acquisition.py",
])
def test_oed_probe_cells(cell, tmp_path):
    r = subprocess.run([sys.executable, str(EXAMPLES / "oed_probes" / cell)],
                       capture_output=True, text=True, env=_env(), cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def test_oed_stopping_rule_cell(tmp_path):
    env = _env()
    env["OED_STOPPING_RULE_OUT"] = str(tmp_path / "stopping_rule.json")
    r = subprocess.run([sys.executable, str(EXAMPLES / "oed_probes" / "a7_recalibrate_oed_stopping_rule.py")],
                       capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
