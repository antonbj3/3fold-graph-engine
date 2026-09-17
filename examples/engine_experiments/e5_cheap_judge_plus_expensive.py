#!/usr/bin/env python3
"""E5: what a very cheap deterministic typed judge changes, at EQUAL spend. SIMULATION with ASSUMED judge
parameters (no real judge was called): reliability r_c, cost 1, deterministic — asking the same point twice
returns the same answer — and errors correlated along the condition axis with length ell (a thresholded
Gaussian field per pair). The expensive probe: reliability 0.95, independent errors, cost 200.
Budget = 60 expensive probes = 12 000 units. Arms:
  expensive_only            60 greedy probes (E4)
  cheap_sweep_naive         judge on an m-point grid of every pair, each answer taken as independent
  cheap_sweep_tempered      same answers, weight N_eff/m with N_eff = m/(1+(m-1)rho), rho measured on held-out pairs
  hybrid_*                  sweep first, the remaining units on greedy expensive probes
Reported: share of right signs (undecided = 0.5) and Brier score of P(+) (calibration)."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import e4_regime_posterior as e4
M, C_EXP, BUDGET = 10, 200, 12000
XS = (np.arange(M) + 0.5) / M

def error_field(rng, r, ell):
    if ell <= 0: return rng.random(M) > r
    K = np.exp(-0.5 * ((XS[:, None] - XS[None, :]) / ell) ** 2) + 1e-6 * np.eye(M)
    return np.linalg.cholesky(K) @ rng.standard_normal(M) > norm.ppf(r)

def rho_of(r, ell, rng, n=4000):
    E = np.array([error_field(rng, r, ell) for _ in range(n)], float)
    Cm = np.corrcoef(E.T); return float(np.nanmean(Cm[np.triu_indices(M, 1)]))

def run(a):
    seed, r, ell, rho = a
    fed, truth = e4.build(seed, 0.15, 0.3, 0.0); rng = np.random.default_rng(seed + 11)
    base, _ = e4.posteriors(fed); pairs = list(base)
    answers = {}
    for pr in pairs:
        err = error_field(rng, r, ell); answers[pr] = [(-1 if err[k] else 1) * e4.tsign(truth, pr, XS[k]) for k in range(M)]
    def brier(rp):
        tot = 0.0
        for pr in pairs:
            for T in e4.GRID: tot += (rp[pr].p_plus(float(T)) - (e4.tsign(truth, pr, T) > 0)) ** 2
        return tot / (len(pairs) * len(e4.GRID))
    def sweep(weight):
        rp, _ = e4.posteriors(fed)
        for pr in pairs:
            for k in range(M): rp[pr].add_probe(XS[k], answers[pr][k], r, weight)
        return rp
    def expensive(rp, n):
        best = {pr: rp[pr].best_probe() for pr in pairs}
        for _ in range(n):
            pr = max(best, key=lambda k: best[k][1]); x = best[pr][0]; s = e4.tsign(truth, pr, x)
            rp[pr].add_probe(x, -s if rng.random() < 0.05 else s); best[pr] = rp[pr].best_probe()
        return rp
    sc = lambda rp: e4.score(lambda q, T: rp[q].p_plus(T), truth, pairs)
    n_left = (BUDGET - len(pairs) * M) // C_EXP; w = 1.0 / (1 + (M - 1) * rho)
    out = {}
    for name, rp in [("no_probes", base), ("expensive_only", expensive(e4.posteriors(fed)[0], BUDGET // C_EXP)),
                     ("cheap_sweep_naive", sweep(1.0)), ("cheap_sweep_tempered", sweep(w)),
                     ("hybrid_naive", expensive(sweep(1.0), n_left)), ("hybrid_tempered", expensive(sweep(w), n_left))]:
        out[name] = [sc(rp), brier(rp)]
    out["expensive_probes_left_after_sweep"] = [float(n_left), float(len(pairs) * M)]
    return out

if __name__ == "__main__":
    res = {}; rng = np.random.default_rng(0)
    for r, ell in [(0.8, 0.0), (0.8, 0.15), (0.8, 0.4), (0.7, 0.15), (0.9, 0.15)]:
        rho = max(rho_of(r, ell, rng), 0.0)
        with Pool(8) as pool: runs = pool.map(run, [(s, r, ell, rho) for s in range(8)])
        agg = {k: np.round(np.mean([x[k] for x in runs], 0), 4).tolist() for k in runs[0]}
        res[f"judge r={r}, error length={ell}, measured rho={rho:.2f}"] = agg
        print(f"r={r} ell={ell} rho={rho:.2f}  [accuracy, Brier]"); [print("   ", k, v) for k, v in agg.items()]
    json.dump(res, open(Path(__file__).parent / "e5_results.json", "w"), indent=1)
