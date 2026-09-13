"""Layer B: the tools' own selftests, plus the example graph/ledger under examples/mini_graph/."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "src" / "graph_engine" / "tools"
EX = REPO_ROOT / "examples" / "mini_graph"


@pytest.mark.parametrize("script", ["anchor_graph_tools.py", "fold_ledger_tools.py", "tool_find.py"])
def test_selftest_cli(script):
    r = subprocess.run([sys.executable, str(TOOLS / script), "--selftest"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_example_graph_validates():
    r = subprocess.run([sys.executable, str(TOOLS / "anchor_graph_tools.py"), "validate",
                        "--graph", str(EX / "ANCHOR_GRAPH.json"),
                        "--ledger", str(EX / "FOLD_LEDGER.jsonl")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads(r.stdout)
    assert res["ok"] and res["error_count"] == 0


def test_example_ledger_resolves_supersedes():
    import fold_ledger_tools
    rows = fold_ledger_tools.load_ledger(EX / "FOLD_LEDGER.jsonl")
    res = fold_ledger_tools.resolve_active(rows)
    assert "MEAS-A1" in res["superseded_ids"]
    assert "MEAS-A2" in res["active_ids"]


def test_graph_store_queries_the_example():
    import graph_store
    gs = graph_store.GraphStore(str(EX / "GRAPH_STORE.json"))
    assert "n.fusion" in gs.nodes
    assert gs.neighbors("n.sensor", depth=1)
    assert [h["id"] for h in gs.holes()] == ["n.orphan"]


def test_preflight_runs_on_the_example():
    r = subprocess.run([sys.executable, str(TOOLS / "preflight.py"),
                        "--graph", str(EX / "ANCHOR_GRAPH.json"),
                        "--ledger", str(EX / "FOLD_LEDGER.jsonl")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "GRAPH ENTRY" in r.stdout
