#!/usr/bin/env python3
"""E11: order of work under one goal when nodes can fail. Expected spend until the goal is reached or the first node
fails, relative to the optimal order under the TRUE p. Orders: existing anchor_graph_tools.priority · unlock_value.rank
with the true p · with p mis-estimated (logit noise sd) · with only a three-level risk class (LOW/MED/HIGH prior) · random.
SIMULATION: 8 nodes, p ~ U(0.3, 0.98), integer cost 1..6, 3 000 graphs per row."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src/graph_engine/tools")); sys.path.insert(0, str(ROOT / "tests"))
import anchor_graph_tools as agt
from graph_engine.unlock_value import rank, expected_cost_of_order, RISK_PRIOR
from test_unlock_value import _graph
rng = np.random.default_rng(0); rows = {k: [] for k in ["existing_priority", "true_p", "p_noise_0.5", "p_noise_1.0", "p_noise_2.0", "risk_class_only", "random"]}
for _ in range(3000):
    k = 8; p = rng.uniform(0.3, 0.98, k); c = rng.integers(1, 7, k).astype(float); ids = [f"n{i}" for i in range(k)]
    P, C = dict(zip(ids, p)), dict(zip(ids, c)); g = _graph(p, c); e = lambda o: expected_cost_of_order(o, P, C)
    best = e([r["id"] for r in rank(g, p_holds=P) if r["id"] != "G"])
    rows["existing_priority"].append(e([r["id"] for r in agt.priority(g) if r["id"] != "G"]) / best); rows["true_p"].append(1.0)
    for sd in (0.5, 1.0, 2.0):
        lo = np.log(p / (1 - p)) + sd * rng.standard_normal(k); ph = dict(zip(ids, 1 / (1 + np.exp(-lo))))
        rows[f"p_noise_{sd}"].append(e([r["id"] for r in rank(g, p_holds=ph) if r["id"] != "G"]) / best)
    for n, pi in zip(g["nodes"][:-1], p): n["risk"] = "LOW" if pi > 0.8 else "MED" if pi > 0.6 else "HIGH"
    rows["risk_class_only"].append(e([r["id"] for r in rank(g) if r["id"] != "G"]) / best)
    rows["random"].append(e(list(rng.permutation(ids))) / best)
out = {k: {"mean_cost_vs_optimal": round(float(np.mean(v)), 3), "p95": round(float(np.quantile(v, 0.95)), 3)} for k, v in rows.items()}
json.dump(out, open(Path(__file__).parent / "e11_results.json", "w"), indent=1); [print(f"{k:20s}", v) for k, v in out.items()]

# ---- nodes that do NOT fail independently: a shared cause hits a susceptible half of the nodes with probability s.
# The rule is applied with the MARGINAL p_i only; the reference is the brute-force optimum under the true joint law (6 nodes).
from itertools import permutations
dep = {}
for s_shock in (0.0, 0.1, 0.3):
    ratios, old = [], []
    for _ in range(300):
        k = 6; own = rng.uniform(0.5, 0.98, k); c = rng.integers(1, 7, k).astype(float); sus = rng.random(k) < 0.5; ids = [f"n{i}" for i in range(k)]
        p1 = dict(zip(ids, np.where(sus, 0.0, own))); p0 = dict(zip(ids, own)); C = dict(zip(ids, c))
        marg = dict(zip(ids, (1 - s_shock) * own + s_shock * np.where(sus, 0.0, own)))
        e = lambda o: s_shock * expected_cost_of_order(o, p1, C) + (1 - s_shock) * expected_cost_of_order(o, p0, C)
        g = _graph(np.array([marg[i] for i in ids]), c); best = min(e(list(o)) for o in permutations(ids))
        ratios.append(e([r["id"] for r in rank(g, p_holds=marg) if r["id"] != "G"]) / best); old.append(e([r["id"] for r in agt.priority(g) if r["id"] != "G"]) / best)
    dep[f"shared_cause_prob={s_shock}"] = {"new_rule_with_marginal_p_vs_true_optimum": round(float(np.mean(ratios)), 4), "p95": round(float(np.quantile(ratios, 0.95)), 4),
                                          "existing_priority_vs_true_optimum": round(float(np.mean(old)), 4)}
    print(f"shared cause {s_shock}:", dep[f"shared_cause_prob={s_shock}"])
out["dependent_failures"] = dep; json.dump(out, open(Path(__file__).parent / "e11_results.json", "w"), indent=1)
