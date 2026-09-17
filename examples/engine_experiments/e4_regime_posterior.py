#!/usr/bin/env python3
"""E4: joint reading of all claims on a pair (regime_posterior) against the pointwise vote of
claim_federation, on the E2 TEST-GRAPHS. Also a world that BREAKS the declared assumption (some pairs
have two transitions) to measure what the collision probe catches.
Score = share of (pair, T) grid points with the right sign; an undecided point counts 0.5."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.claim_federation import Federation
from graph_engine.regime_posterior import RegimePosterior
import e2_federation_benchmark as e2
PROBES = 60

GRID = np.linspace(0.025, 0.975, 19)

def tsign(truth, pair, T):
    sg, th = truth[pair]
    if th is None: return sg
    if isinstance(th, tuple): return sg if (T < th[0] or T >= th[1]) else -sg      # two transitions
    return sg if T < th else -sg
e2.true_sign = tsign                                                                # observe() uses it

def build(seed, err, echo, two_frac):
    rng = np.random.default_rng(seed); A, B, S, truth = e2.world(rng)
    for k, (sg, th) in list(truth.items()):
        if th is not None and rng.random() < two_frac:
            a = float(rng.uniform(0.2, 0.45)); truth[k] = (sg, (a, a + float(rng.uniform(0.25, 0.4))))
    # observe() draws boxes on one side of a single threshold; for two-transition pairs draw free boxes
    def obs(dom, gid):
        g = e2.observe(rng, {k: (v[0], None if isinstance(v[1], tuple) else v[1]) for k, v in truth.items()}, dom, gid, err, echo)
        for c in g["claims"]:
            k = (c["subject"], c["object"])
            if isinstance(truth[k][1], tuple):
                w = rng.uniform(0.08, 0.2); a = rng.uniform(0, 1 - w); c["validity"] = {"T": [a, a + w]}
                s = tsign(truth, k, a + w / 2); c["sign"] = -s if rng.random() < err else s
        return g
    fed = Federation(); fed.add_graph(obs(set(A + S), "A")); fed.add_graph(obs(set(B + S), "B"))
    return fed, truth

def posteriors(fed):
    rp, org = {}, {}
    for c in fed.claims.values():
        p = rp.setdefault(c["pair"], RegimePosterior(0.0, 1.0)); a, b = c["box"]["T"]
        p.add_claim(a, b, c["sign"], fed.n_eff(c["evidence"]))
        org.setdefault(c["pair"], []).append(frozenset().union(*[fed.roots(s) for s in c["evidence"]]))
    return rp, org

def score(fn, truth, pairs):
    tot = 0.0
    for pair in pairs:
        for T in GRID:
            p = fn(pair, float(T)); s = tsign(truth, pair, T)
            tot += 0.5 if abs(p - 0.5) < 1e-9 else float((p > 0.5) == (s > 0))
    return tot / (len(pairs) * len(GRID))

def run(a):
    seed, err, echo, two = a
    fed, truth = build(seed, err, echo, two); rp, org = posteriors(fed); pairs = list(rp)
    out = {"vote": score(lambda pr, T: fed.belief(pr, {"T": T})[0], truth, pairs),
           "regime_posterior": score(lambda pr, T: rp[pr].p_plus(T), truth, pairs)}
    dbl = {pr: isinstance(truth[pr][1], tuple) for pr in pairs}
    def coll(rpx, tag):
        flags = {pr: rpx[pr].collision()["flag"] for pr in pairs}; tp = sum(flags[p] and dbl[p] for p in pairs)
        out[f"collision_flagged{tag}"] = sum(flags.values()); out[f"collision_prec{tag}"] = tp / sum(flags.values()) if sum(flags.values()) else float("nan")
        out[f"collision_rec{tag}"] = tp / sum(dbl.values()) if sum(dbl.values()) else float("nan")
    coll(rp, "_passive")
    rp0 = {pr: RegimePosterior(0.0, 1.0, p_two=0.0) for pr in pairs}
    for pr in pairs:
        for cl in rp[pr].claims: rp0[pr].add_claim(*cl)
    out["regime_posterior_assuming_no_collision(p_two=0)"] = score(lambda pr, T: rp0[pr].p_plus(T), truth, pairs)
    # probing, 60 probes, answers right with prob 0.95
    prng = np.random.default_rng(seed + 7)
    for mode in ["greedy_entropy_potential", "greedy_error_potential", "random"]:
        rp2, _ = posteriors(fed); curve = []
        for r_ in rp2.values(): r_.potential = "error" if mode == "greedy_error_potential" else "entropy"
        best = {pr: rp2[pr].best_probe() for pr in pairs}
        for step in range(PROBES):
            if mode == "random": pr = pairs[int(prng.integers(len(pairs)))]; x = float(prng.uniform(0, 1))
            else: pr = max(best, key=lambda k: best[k][1]); x = best[pr][0]
            s = tsign(truth, pr, x); rp2[pr].add_probe(x, -s if prng.random() < 0.05 else s); best[pr] = rp2[pr].best_probe()
            if step % 20 == 19: curve.append(score(lambda q, T: rp2[q].p_plus(T), truth, pairs))
        out[mode] = curve
        if mode == "greedy_entropy_potential": coll(rp2, "_after_60_greedy_probes")
    f2 = Federation(); f2.claims = dict(fed.claims); f2._parents = fed._parents; curve = []
    for step in range(60):
        c = f2.next_experiments(include_inferred=False)
        if not c: break
        pr = c[0]; s = tsign(truth, pr.pair, pr.point.get("T", .5))
        for r in range(2): f2.record(pr, -s if prng.random() < 0.05 else s, f"m{step}{r}")
        if step % 20 == 19: curve.append(score(lambda q, T: f2.belief(q, {"T": T})[0], truth, pairs))
    out["federation_default(2 answers per probe)"] = curve
    return out

if __name__ == "__main__":
    res = {}
    for err, echo, two in [(0.15, 0.3, 0.0), (0.3, 0.3, 0.0), (0.15, 0.6, 0.0), (0.15, 0.3, 0.4)]:
        with Pool(8) as pool: runs = pool.map(run, [(s, err, echo, two) for s in range(8)])
        agg = {k: (np.round(np.nanmean([r[k] for r in runs], 0), 4).tolist()) for k in runs[0]}
        res[f"err={err},copied={echo},two_transition_share={two}"] = agg; print(err, echo, two, agg, flush=True)
    json.dump(res, open(Path(__file__).parent / "e4_results.json", "w"), indent=1)
