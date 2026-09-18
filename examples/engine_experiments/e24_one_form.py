#!/usr/bin/env python3
"""E24: one Gaussian precision form behind three rankings. The numbers the tests assert, written out.

T1 effective resistance = contrast variance of J⁺ (exact pinv, random weighted graphs).
T2 GLS posterior mean/variance = margin_net m̂, s² (including a singular Σ from declared copies).
T3 Sherman–Morrison on the covariance = recomputed pinv when h ∈ range(J), and the null-space branch.
T4 value = ½log₂(1 + hᵀCh/σ²) (matrix determinant lemma); restricted to quantities of interest it is
   smaller; for a Bernoulli belief it ranks exactly as regime_posterior's Fisher/Jensen form and only
   approximately as the exact two-outcome entropy drop.
T5 set value ½log₂det(I + D⁻¹HCHᵀD⁻¹): chain rule, monotone, submodular; greedy against brute force;
   what the best triple of contrast probes actually looks like.
T6 one currency (bits per cost) over structure probes, new reports and belief probes.

The scripts imports the test module so that the reported numbers are the asserted ones.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
from graph_engine.precision_form import (  # noqa: E402
    Candidate, PrecisionForm, bernoulli_exact_bits, bernoulli_fisher_bits, bernoulli_probe,
)
from graph_engine.resistance_sketch import ResistanceSketch, laplacian_from_edges  # noqa: E402
import test_precision_form as T  # noqa: E402

out = {}

# -- T1 --------------------------------------------------------------------------------------------
errs, rel_sketch = [], []
for seed in range(5):
    n = 40
    edges, w = T.random_connected_graph(n, 60, seed)
    L, _, _ = laplacian_from_edges(n, edges, w)
    Lp = np.linalg.pinv(L.toarray(), hermitian=True)
    f = T.form_from_graph(edges, w, n)
    sk = ResistanceSketch.build(n, edges, w, k=4096, seed=seed)
    rng = np.random.default_rng(100 + seed)
    ii = rng.integers(0, n, 50); jj = (ii + 1 + rng.integers(0, n - 1, 50)) % n
    exact = np.array([Lp[i, i] + Lp[j, j] - 2 * Lp[i, j] for i, j in zip(ii, jj)])
    mine = np.array([f.resistance(int(i), int(j)) for i, j in zip(ii, jj)])
    errs.append(float(np.abs(mine - exact).max())); rel_sketch.append(float(np.abs(sk.resistance(ii, jj) / exact - 1).max()))
out["T1_resistance"] = {"graphs": 5, "pairs_per_graph": 50, "max_abs_error_vs_exact_pinv": max(errs),
                        "max_rel_error_of_the_k4096_sketch": max(rel_sketch), "exact": max(errs) < 1e-9}

# -- T2 --------------------------------------------------------------------------------------------
net, reports = T.build_margin_net()
est = net.estimate("e1"); Sigma = T.sigma_from_lineage(net, "e1", reports)
y = np.array([r["margin"] for r in reports])
f = PrecisionForm.zeros(1).add_measurement(np.ones((len(y), 1)), y, Sigma)
out["T2_margins"] = {"n_reports": len(y), "rank_Sigma": int(np.linalg.matrix_rank(Sigma, tol=1e-10)),
                     "margin_net_mhat": est.m, "form_mean": float(f.mean()[0]),
                     "abs_error_mean": abs(float(f.mean()[0]) - est.m),
                     "margin_net_s2": est.s ** 2, "form_var": float(f.cov()[0, 0]),
                     "abs_error_var": abs(float(f.cov()[0, 0]) - est.s ** 2)}

# -- T3 --------------------------------------------------------------------------------------------
n = 25; edges, w = T.random_connected_graph(n, 35, 3); rng = np.random.default_rng(7)
f = T.form_from_graph(edges, w, n); worst = 0.0
for _ in range(20):
    h = rng.standard_normal(n); h -= h.mean(); sigma = float(rng.uniform(0.2, 2.0))
    ref = np.linalg.pinv(f.J + np.outer(h, h) / sigma ** 2, hermitian=True)
    used = f.observe(h, sigma)
    worst = max(worst, float(np.abs(f.cov() - ref).max()))
g = T.form_from_graph(edges, w, n); C = g.cov(); h1 = np.ones(n)
Ch = C @ h1; sm = C - np.outer(Ch, Ch) / (1.0 + float(h1 @ Ch))
used_null = g.observe(h1, 1.0)
out["T3_sherman_morrison"] = {"updates_in_range": 20, "max_abs_error_vs_recomputed_pinv": worst,
                              "null_space_h_used_sherman_morrison": bool(used_null),
                              "error_if_it_had_been_used": float(np.abs(g.cov() - sm).max()),
                              "fallback_matches_pinv": float(np.abs(g.cov() - np.linalg.pinv(g.J, hermitian=True)).max())}

# -- T4 --------------------------------------------------------------------------------------------
n = 18; edges, w = T.random_connected_graph(n, 25, 11); rng = np.random.default_rng(5); worst = 0.0
for _ in range(25):
    f = T.form_from_graph(edges, w, n)
    h = rng.standard_normal(n); h -= h.mean(); sigma = float(rng.uniform(0.3, 2.0))
    before = T._logpdet(f.cov()) / (2 * np.log(2)); v = f.value_bits(h, sigma); f.observe(h, sigma)
    worst = max(worst, abs((before - T._logpdet(f.cov()) / (2 * np.log(2))) - v))
ps = np.linspace(0.2, 0.8, 13); rels = [0.6, 0.7, 0.8, 0.9, 0.97]
gauss, fisher, exact_b = [], [], []
for p in ps:
    for r in rels:
        hb, sb = bernoulli_probe(float(p), r)
        gauss.append(PrecisionForm.zeros(1).add_bernoulli(0, float(p)).value_bits([hb], sb))
        fisher.append(bernoulli_fisher_bits(float(p), r)); exact_b.append(bernoulli_exact_bits(float(p), r))
gauss, fisher, exact_b = map(np.array, (gauss, fisher, exact_b))
out["T4_value"] = {"max_abs_error_determinant_lemma_bits": worst,
                   "n_bernoulli_candidates": int(gauss.size), "p_range": [0.2, 0.8], "reliabilities": rels,
                   "spearman_gaussian_vs_regime_fisher_form": float(spearmanr(gauss, fisher).statistic),
                   "spearman_gaussian_vs_exact_entropy_drop": float(spearmanr(gauss, exact_b).statistic),
                   "max_rel_gap_gaussian_vs_exact": float(np.abs(gauss / exact_b - 1).max()),
                   "median_rel_gap_gaussian_vs_exact": float(np.median(np.abs(gauss / exact_b - 1))),
                   "note": "the Bernoulli block is a LOCAL Gaussian image: the order is kept, the value is not"}

# -- T5 --------------------------------------------------------------------------------------------
n = 16; edges, w = T.random_connected_graph(n, 22, 4); base = T.form_from_graph(edges, w, n)
rng = np.random.default_rng(13); cands = T.random_candidates(n, rng, 10)
ff = base.copy(); acc = 0.0
for c in cands[:5]:
    acc += ff.value_bits(c.h, c.sigma); ff.observe(c.h, c.sigma)
n_mono = n_sub = 0; worst_sub = 0.0
for _ in range(200):
    k = int(rng.integers(0, 6)); idx = rng.choice(len(cands), k + 2, replace=False)
    S = [cands[i] for i in idx[:k]]; a, b = cands[idx[k]], cands[idx[k + 1]]
    vS, vSa, vSb, vSab = base.value_of(S), base.value_of(S + [a]), base.value_of(S + [b]), base.value_of(S + [a, b])
    n_mono += int(vSa >= vS - 1e-12 and vSab >= vSb - 1e-12)
    d = (vSa - vS) - (vSab - vSb); n_sub += int(d >= -1e-9); worst_sub = min(worst_sub, d)
rng = np.random.default_rng(21); ratios = []
for t in range(100):
    e12, w12 = T.random_connected_graph(12, 14, 1000 + t); bs = T.form_from_graph(e12, w12, 12)
    cs = [Candidate(f"c{i}", hh.h, float(rng.uniform(0.4, 1.6))) for i, hh in enumerate(T.random_candidates(12, rng, 8))]
    ratios.append(bs.value_of(bs.best_set(cs, 3)) / bs.value_of(bs.brute_best_set(cs, 3)))
ratios = np.array(ratios)
comp = [T.set_composition(s, 300) for s in range(6)]
out["T5_set"] = {"chain_rule_abs_error_bits": abs(acc - base.value_of(cands[:5])),
                 "checks": 200, "monotone_ok": n_mono, "submodular_ok": n_sub,
                 "worst_diminishing_returns_violation_bits": float(worst_sub),
                 "greedy_vs_brute": {"instances": 100, "candidates": 8, "k": 3, "min_ratio": float(ratios.min()),
                                     "mean_ratio": float(ratios.mean()), "n_optimal": int((ratios > 1 - 1e-12).sum()),
                                     "bound_1_minus_1_over_e": float(1 - 1 / np.e)},
                 "composition_two_cluster_graph": {
                     "instances": len(comp), "n_far_in_best": [c["best_n_far"] for c in comp],
                     "best_mean_R": [round(c["best_mean_R"], 3) for c in comp],
                     "top3_by_resistance_mean_R": [round(c["top3_by_resistance_mean_R"], 3) for c in comp],
                     "best_mean_coherence": [round(c["best_mean_coherence"], 3) for c in comp],
                     "top3_mean_coherence": [round(c["top3_mean_coherence"], 3) for c in comp],
                     "random_mean_coherence": [round(c["random_mean_coherence"], 3) for c in comp],
                     "best_value_bits": [round(c["best_value_bits"], 3) for c in comp],
                     "top3_by_resistance_value_bits": [round(c["top3_by_resistance_value_bits"], 3) for c in comp],
                     "random_value_bits": [round(c["random_value_bits"], 3) for c in comp],
                     "finding": "not a tight core plus one far element: the best triple is usually all far, "
                                "but gives up resistance and mutual coherence against the top-3 by resistance"}}

# -- T6 --------------------------------------------------------------------------------------------
f6, cands6, meta6, margins6 = T.unified_instance()
ranked = f6.rank(cands6)
by_res = max((c for c in cands6 if c.kind == "structure"), key=lambda c: meta6[c.id]["resistance"]).id
by_z = min((c for c in cands6 if c.kind == "margin"), key=lambda c: abs(meta6[c.id]["z"])).id
by_reg = max((c for c in cands6 if c.kind == "regime"), key=lambda c: meta6[c.id]["exact_bits"]).id
out["T6_one_currency"] = {
    "top5": [{"id": i, "kind": k, "bits": round(v, 4), "bits_per_cost": round(vc, 4),
              "cost": meta6[i]["cost"]} for i, v, vc, k in ranked[:5]],
    "n_candidates": len(cands6),
    "top1_by_resistance_alone": by_res, "top1_by_margin_z_alone": by_z, "top1_by_regime_value_alone": by_reg,
    "joint_top1": ranked[0][0], "gap_top1_top2_bits_per_cost": ranked[0][2] - ranked[1][2],
    "margin_z": {k: round(v.z, 3) for k, v in margins6.items()},
    "note": "each part ranks only its own kind; bits per cost ranks all three and disagrees with "
            "resistance even inside the structure kind, because cost and probe noise are invisible to R"}

Path(__file__).with_name("e24_results.json").write_text(json.dumps(out, indent=1, sort_keys=False))
print(json.dumps(out, indent=1))
