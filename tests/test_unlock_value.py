"""unlock_value: the ordering theorems checked by brute force against every feasible order."""
import json, sys
from itertools import permutations
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "graph_engine" / "tools"))
from graph_engine.unlock_value import rank, expected_cost_of_order  # noqa: E402


def _graph(p, c, deps=None):
    deps = deps or {}
    nodes = [{"id": f"n{k}", "claim": "", "type": "EMPIRICAL", "status": "OPEN", "evidence": [], "depends_on": list(deps.get(f"n{k}", [])), "load": 1,
              "risk": "MED", "regime_note": "n/a", "cost": c[k]} for k in range(len(p))]
    nodes.append({"id": "G", "claim": "", "type": "GOAL", "status": "OPEN", "evidence": [], "depends_on": [n["id"] for n in nodes],
                  "load": 0, "risk": "HIGH", "regime_note": "n/a", "cost": 1})
    return {"_meta": {}, "nodes": nodes}


def _best(ids, P, C, deps):
    ok = lambda o: all(o.index(d) < o.index(i) for i, ds in deps.items() for d in ds)
    return min(expected_cost_of_order(list(o), P, C) for o in permutations(ids) if ok(o))


def test_no_precedence_order_is_the_brute_force_optimum():
    rng = np.random.default_rng(0)
    for _ in range(40):
        k = 6; p = rng.uniform(0.2, 0.98, k); c = rng.uniform(0.5, 6.0, k); ids = [f"n{i}" for i in range(k)]; P, C = dict(zip(ids, p)), dict(zip(ids, c))
        mine = [r["id"] for r in rank(_graph(p, c), p_holds=P) if r["id"] != "G"]
        assert np.isclose(expected_cost_of_order(mine, P, C), _best(ids, P, C, {}))


def test_chain_precedence_order_is_the_brute_force_optimum():
    """Review finding: with dependencies between open nodes the single-node rule lost up to 1.81×. Two and three chains."""
    rng = np.random.default_rng(1)
    for trial in range(120):
        lens = [2, 2] if trial % 3 == 0 else [3, 2] if trial % 3 == 1 else [2, 2, 2]
        k = sum(lens); p = rng.uniform(0.2, 0.98, k); c = rng.uniform(0.5, 6.0, k); ids = [f"n{i}" for i in range(k)]; P, C = dict(zip(ids, p)), dict(zip(ids, c))
        deps, start = {}, 0
        for L in lens:
            for t in range(start + 1, start + L): deps[f"n{t}"] = [f"n{t - 1}"]
            start += L
        mine = [r["id"] for r in rank(_graph(p, c, deps), p_holds=P) if r["id"] != "G"]
        assert all(mine.index(d) < mine.index(i) for i, ds in deps.items() for d in ds)
        assert np.isclose(expected_cost_of_order(mine, P, C), _best(ids, P, C, deps)), (trial, lens)


def test_existing_priority_order_costs_more_when_nodes_can_fail():
    import anchor_graph_tools as agt
    rng = np.random.default_rng(1); ratio = []
    for _ in range(200):
        k = 8; p = rng.uniform(0.3, 0.98, k); c = rng.integers(1, 7, k).astype(float); ids = [f"n{i}" for i in range(k)]; P, C = dict(zip(ids, p)), dict(zip(ids, c)); g = _graph(p, c)
        old = [r["id"] for r in agt.priority(g) if r["id"] != "G"]; new = [r["id"] for r in rank(g, p_holds=P) if r["id"] != "G"]
        ratio.append(expected_cost_of_order(old, P, C) / expected_cost_of_order(new, P, C))
    assert min(ratio) >= 1 - 1e-12 and np.mean(ratio) > 1.10


def test_shipped_mini_graph_ranks_and_keeps_readiness():
    g = json.load(open(ROOT / "examples" / "mini_graph" / "ANCHOR_GRAPH.json")); r = rank(g); ids = [x["id"] for x in r]
    assert r[0]["actionable"] and ids.index("FUSION-GOAL") > ids.index("COVERAGE") and ids.index("FUSION-GOAL") > ids.index("FUSION-V1")
