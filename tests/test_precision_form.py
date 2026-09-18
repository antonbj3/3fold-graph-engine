"""precision_form: every claim in the module docstring, as a number.

T1 resistance = contrast variance of J⁺; T2 GLS mean/variance = margin_net m̂, s²; T3 Sherman–Morrison
= recomputed pinv (and the null-space branch where it is NOT); T4 matrix determinant lemma, the
restricted-Q inequality, the Bernoulli reductions; T5 chain rule, monotonicity, submodularity,
greedy vs brute force, set composition; T6 one currency across three kinds.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "graph_engine" / "tools"))
from graph_engine.margin_net import MarginNet  # noqa: E402
from graph_engine.precision_form import (  # noqa: E402
    Candidate, PrecisionForm, bernoulli_exact_bits, bernoulli_fisher_bits, bernoulli_probe,
)
from graph_engine.resistance_sketch import ResistanceSketch, laplacian_from_edges  # noqa: E402


# -- helpers ---------------------------------------------------------------------------------------
def random_connected_graph(n, extra, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    edges = [(int(perm[i]), int(perm[i + 1])) for i in range(n - 1)]          # a random spanning tree
    seen = {frozenset(e) for e in edges}
    while len(edges) < n - 1 + extra:
        i, j = rng.integers(0, n, 2)
        if i != j and frozenset((int(i), int(j))) not in seen:
            seen.add(frozenset((int(i), int(j)))); edges.append((int(i), int(j)))
    w = rng.uniform(0.3, 3.0, len(edges))
    return np.array(edges), w


def two_cluster_graph(per=12, seed=0):
    rng = np.random.default_rng(seed)
    edges = []
    for base in (0, per):
        for i in range(per):
            for j in range(i + 1, per):
                if rng.random() < 0.45:
                    edges.append((base + i, base + j))
    edges += [(0, per), (1, per + 1)]                                         # two bridges
    return np.array(edges), 2 * per


def form_from_graph(edges, w, n):
    return PrecisionForm.zeros(n).add_laplacian(edges, w)


# -- T1 --------------------------------------------------------------------------------------------
def test_t1_contrast_variance_is_effective_resistance():
    errs, sketch_rel = [], []
    for seed in range(5):
        n = 40
        edges, w = random_connected_graph(n, 60, seed)
        L, _, _ = laplacian_from_edges(n, edges, w)
        Lp = np.linalg.pinv(L.toarray(), hermitian=True)
        f = form_from_graph(edges, w, n)
        sk = ResistanceSketch.build(n, edges, w, k=4096, seed=seed)
        rng = np.random.default_rng(100 + seed)
        ii = rng.integers(0, n, 50); jj = (ii + 1 + rng.integers(0, n - 1, 50)) % n
        exact = np.array([Lp[i, i] + Lp[j, j] - 2 * Lp[i, j] for i, j in zip(ii, jj)])
        mine = np.array([f.resistance(int(i), int(j)) for i, j in zip(ii, jj)])
        errs.append(np.abs(mine - exact).max())
        sketch_rel.append(np.abs(sk.resistance(ii, jj) / exact - 1).max())
    assert max(errs) < 1e-9, max(errs)
    assert max(sketch_rel) < 0.2                      # the sketch agrees only to its own O(1/sqrt(k))


# -- T2 --------------------------------------------------------------------------------------------
def build_margin_net():
    net = MarginNet(default_sigma=0.1)
    net.add_sources([{"id": "a"}, {"id": "b"}, {"id": "c1", "derives_from": ["a"]},
                     {"id": "c2", "derives_from": ["a"]}, {"id": "d", "derives_from": ["a", "b"]}])
    reports = [{"margin": 0.9, "sigma": 0.20, "sources": ["c1"]},
               {"margin": 0.4, "sigma": 0.15, "sources": ["c2"]},
               {"margin": 0.7, "sigma": 0.30, "sources": ["d"]},
               {"margin": 0.6, "sigma": 0.25, "sources": ["b"]}]
    net.add_edge("e1", ["u", "v"], reports)
    return net, reports


def sigma_from_lineage(net, edge_id, reports):
    rs = [frozenset().union(*[net.roots(x) for x in r["sources"]]) for r in reports]
    allr = sorted(set().union(*rs))
    M = np.array([[1.0 if x in r else 0.0 for x in allr] for r in rs]); M /= M.sum(1, keepdims=True)
    B = np.array([r["sigma"] for r in reports])[:, None] * M
    return B @ B.T


def test_t2_gls_posterior_equals_margin_net():
    net, reports = build_margin_net()
    est = net.estimate("e1")
    Sigma = sigma_from_lineage(net, "e1", reports)
    y = np.array([r["margin"] for r in reports])
    f = PrecisionForm.zeros(1).add_measurement(np.ones((len(y), 1)), y, Sigma)
    assert abs(float(f.mean()[0]) - est.m) < 1e-12
    assert abs(float(f.cov()[0, 0]) - est.s ** 2) < 1e-12
    # singular Σ (declared copies of one root): same pseudo-inverse on both sides, still exact
    net2 = MarginNet(default_sigma=0.1)
    net2.add_sources([{"id": "a"}, {"id": "k1", "derives_from": ["a"]}, {"id": "k2", "derives_from": ["a"]}])
    rep2 = [{"margin": 0.5, "sigma": 0.1, "sources": ["k1"]}, {"margin": 0.55, "sigma": 0.2, "sources": ["k2"]}]
    net2.add_edge("e", ["u", "v"], rep2)
    S2 = sigma_from_lineage(net2, "e", rep2)
    assert np.linalg.matrix_rank(S2, tol=1e-10) == 1
    e2 = net2.estimate("e")
    f2 = PrecisionForm.zeros(1).add_measurement(np.ones((2, 1)), [r["margin"] for r in rep2], S2)
    assert abs(float(f2.mean()[0]) - e2.m) < 1e-12 and abs(float(f2.cov()[0, 0]) - e2.s ** 2) < 1e-12


# -- T3 --------------------------------------------------------------------------------------------
def test_t3_sherman_morrison_matches_pinv_and_flags_the_null_space():
    n = 25
    edges, w = random_connected_graph(n, 35, 3)
    rng = np.random.default_rng(7)
    f = form_from_graph(edges, w, n)
    worst = 0.0
    for _ in range(20):                                    # h ⟂ 1: inside range(L), SM must be exact
        h = rng.standard_normal(n); h -= h.mean()
        sigma = float(rng.uniform(0.2, 2.0))
        ref = np.linalg.pinv(f.J + np.outer(h, h) / sigma ** 2, hermitian=True)
        assert f.observe(h, sigma) is True
        worst = max(worst, np.abs(f.cov() - ref).max())
    assert worst < 1e-8, worst
    g = form_from_graph(edges, w, n)
    C = g.cov(); h = np.ones(n)                            # h in the null space: SM would be wrong
    assert g.in_range(h) is False
    Ch = C @ h
    sm = C - np.outer(Ch, Ch) / (1.0 + float(h @ Ch))      # = C, no change at all
    assert g.observe(h, 1.0) is False
    assert np.abs(g.cov() - sm).max() > 1e-3               # the fallback is not cosmetic
    assert np.abs(g.cov() - np.linalg.pinv(g.J, hermitian=True)).max() < 1e-10


# -- T4 --------------------------------------------------------------------------------------------
def _logpdet(C, tol=1e-9):
    ev = np.linalg.eigvalsh(C)
    return float(np.log(ev[ev > tol * ev.max()]).sum())


def test_t4_value_is_the_matrix_determinant_lemma():
    n = 18
    edges, w = random_connected_graph(n, 25, 11)
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(25):
        f = form_from_graph(edges, w, n)
        h = rng.standard_normal(n); h -= h.mean()
        sigma = float(rng.uniform(0.3, 2.0))
        before = _logpdet(f.cov()) / (2 * np.log(2))
        v = f.value_bits(h, sigma)
        f.observe(h, sigma)
        worst = max(worst, abs((before - _logpdet(f.cov()) / (2 * np.log(2))) - v))
    assert worst < 1e-8, worst


def test_t4_restricted_value_is_exact_and_never_larger():
    n = 14
    edges, w = random_connected_graph(n, 20, 2)
    rng = np.random.default_rng(9)
    worst, gaps = 0.0, []
    for _ in range(20):
        f = form_from_graph(edges, w, n)
        Q = np.zeros((3, n))
        for r in range(3):
            i, j = rng.choice(n, 2, replace=False)
            Q[r, i], Q[r, j] = 1.0, -1.0
        h = rng.standard_normal(n); h -= h.mean(); sigma = float(rng.uniform(0.3, 2.0))
        full, restricted = f.value_bits(h, sigma), f.value_bits(h, sigma, Q=Q)
        before = _logpdet(Q @ f.cov() @ Q.T) / (2 * np.log(2))
        f.observe(h, sigma)
        worst = max(worst, abs((before - _logpdet(Q @ f.cov() @ Q.T) / (2 * np.log(2))) - restricted))
        gaps.append(full - restricted)
        assert restricted <= full + 1e-12
    assert worst < 1e-8, worst
    assert max(gaps) > 1e-3                                 # marginalization really does lose bits


def test_t4_bernoulli_reduces_to_the_regime_posterior_fisher_form():
    ps = np.linspace(0.2, 0.8, 13)
    rels = [0.6, 0.7, 0.8, 0.9, 0.97]
    gauss, fisher, exact = [], [], []
    for p in ps:
        for r in rels:
            h, s = bernoulli_probe(float(p), r)
            f = PrecisionForm.zeros(1).add_bernoulli(0, float(p))
            assert abs(f.cov()[0, 0] - p * (1 - p)) < 1e-12
            gauss.append(f.value_bits([h], s))
            fisher.append(bernoulli_fisher_bits(float(p), r))
            exact.append(bernoulli_exact_bits(float(p), r))
    gauss, fisher, exact = map(np.array, (gauss, fisher, exact))
    # the Gaussian value and the Jensen/Fisher form are both strictly increasing in t: same order
    assert spearmanr(gauss, fisher).statistic > 0.999
    # against the EXACT two-outcome entropy drop the local Gaussian is an approximation, not an identity
    rho = spearmanr(gauss, exact).statistic
    assert rho > 0.9, rho
    assert np.abs(gauss / exact - 1).max() > 0.1            # stated as approximate, and it is


# -- T5 --------------------------------------------------------------------------------------------
def random_candidates(n, rng, count, sigma=1.0):
    out = []
    for c in range(count):
        h = rng.standard_normal(n); h -= h.mean()
        out.append(Candidate(f"c{c}", h, sigma))
    return out


def test_t5_set_value_chain_rule_monotone_and_submodular():
    n = 16
    edges, w = random_connected_graph(n, 22, 4)
    base = form_from_graph(edges, w, n)
    rng = np.random.default_rng(13)
    cands = random_candidates(n, rng, 10)
    # chain rule: the set value telescopes into sequential single-observation values
    f = base.copy(); acc = 0.0
    for c in cands[:5]:
        acc += f.value_bits(c.h, c.sigma); f.observe(c.h, c.sigma)
    assert abs(acc - base.value_of(cands[:5])) < 1e-9
    n_mono = n_sub = 0
    worst_sub = 0.0
    for _ in range(200):
        k = int(rng.integers(0, 6))
        idx = rng.choice(len(cands), k + 2, replace=False)
        S = [cands[i] for i in idx[:k]]; a, b = cands[idx[k]], cands[idx[k + 1]]
        vS, vSa, vSb, vSab = (base.value_of(S), base.value_of(S + [a]),
                              base.value_of(S + [b]), base.value_of(S + [a, b]))
        n_mono += int(vSa >= vS - 1e-12 and vSab >= vSb - 1e-12)
        diff = (vSa - vS) - (vSab - vSb)
        n_sub += int(diff >= -1e-9)
        worst_sub = min(worst_sub, diff)
    assert n_mono == 200 and n_sub == 200, (n_mono, n_sub, worst_sub)


def test_t5_greedy_is_close_to_brute_force():
    rng = np.random.default_rng(21)
    ratios = []
    for t in range(100):
        n = 12
        edges, w = random_connected_graph(n, 14, 1000 + t)
        base = form_from_graph(edges, w, n)
        cands = [Candidate(f"c{i}", h.h, float(rng.uniform(0.4, 1.6)))
                 for i, h in enumerate(random_candidates(n, rng, 8))]
        g = base.value_of(base.best_set(cands, 3))
        b = base.value_of(base.brute_best_set(cands, 3))
        ratios.append(g / b)
    ratios = np.array(ratios)
    assert ratios.min() > 1 - 1 / np.e                     # the submodular guarantee, realized
    assert ratios.mean() > 0.99


def set_composition(seed=0, n_random=500):
    """The best triple of contrast probes on a two-cluster graph (weak bridges): near, far, mixed?
    Reported against the three highest-resistance candidates — the naive extension of a pair score."""
    edges, n = two_cluster_graph(12, seed)
    w = np.ones(len(edges)); w[-2:] = 0.2
    base = form_from_graph(edges, w, n)
    rng = np.random.default_rng(seed)
    cands, far_flag = [], []
    for c in range(20):
        if c % 2:
            i, j = int(rng.integers(0, 12)), int(rng.integers(12, n))       # cross-cluster: far
        else:
            i, j = rng.choice(12 if c % 4 == 0 else np.arange(12, n), 2, replace=False)
        h = np.zeros(n); h[int(i)], h[int(j)] = 1.0, -1.0
        cands.append(Candidate(f"p{c}", h, 1.0)); far_flag.append(c % 2 == 1)
    far_flag = np.array(far_flag)
    R = np.array([base.contrast_var(c.h) for c in cands])
    best = base.brute_best_set(cands, 3)
    bidx = [cands.index(x) for x in best]
    coh = lambda idx: float(np.mean([abs(cands[a].h @ base.cov() @ cands[b].h) / np.sqrt(R[a] * R[b])
                                     for a in idx for b in idx if a < b]))
    rnd = [rng.choice(len(cands), 3, replace=False) for _ in range(n_random)]
    top3 = list(np.argsort(-R)[:3])
    return {"best_ids": [c.id for c in best], "best_mean_R": float(R[bidx].mean()),
            "random_mean_R": float(np.mean([R[i].mean() for i in rnd])),
            "top3_by_resistance_mean_R": float(R[top3].mean()),
            "best_value_bits": base.value_of(best),
            "top3_by_resistance_value_bits": base.value_of([cands[i] for i in top3]),
            "random_value_bits": float(np.mean([base.value_of([cands[i] for i in idx]) for idx in rnd])),
            "best_mean_coherence": coh(bidx), "top3_mean_coherence": coh(top3),
            "random_mean_coherence": float(np.mean([coh(i) for i in rnd])),
            "best_n_far": int(far_flag[bidx].sum()), "best_is_mixed": bool(0 < far_flag[bidx].sum() < 3),
            "n_far_candidates": int(far_flag.sum())}


def test_t5_best_set_trades_resistance_for_independence():
    """Measured, against the expectation in throws.py: the best triple is NOT "a tight core plus one
    far element" here — with far pairs dominating the candidate pool it is usually three far pairs.
    What it does do is give up resistance for independence: strictly lower mean R than the three
    highest-R candidates, strictly lower mutual coherence, strictly more bits."""
    out = [set_composition(s, 200) for s in range(6)]
    assert all(o["best_mean_R"] < o["top3_by_resistance_mean_R"] for o in out)
    assert all(o["best_mean_coherence"] < o["top3_mean_coherence"] for o in out)
    assert all(o["best_value_bits"] > o["top3_by_resistance_value_bits"] for o in out)
    assert all(o["best_value_bits"] > o["random_value_bits"] for o in out)
    assert not all(o["best_is_mixed"] for o in out)          # the near/far mix is not what happens


# -- T6 --------------------------------------------------------------------------------------------
def unified_instance():
    """One form: 12 graph nodes, 2 margin parameters (reports with lineage), 1 Bernoulli belief."""
    edges, w = random_connected_graph(12, 8, 77)
    n = 12
    d = n + 3
    f = PrecisionForm.zeros(d).add_laplacian(edges, w, index=np.arange(n))
    margins = {}
    for k, (mid, sig, vals) in enumerate([("m_tight", 0.05, [0.62, 0.55, 0.60]),
                                          ("m_loose", 0.30, [0.20, -0.10, 0.25])]):
        net = MarginNet(default_sigma=sig)
        net.add_sources([{"id": f"{mid}_a"}, {"id": f"{mid}_b"}, {"id": f"{mid}_c", "derives_from": [f"{mid}_a"]}])
        reps = [{"margin": v, "sigma": sig, "sources": [s]}
                for v, s in zip(vals, [f"{mid}_a", f"{mid}_b", f"{mid}_c"])]
        net.add_edge(mid, ["x", "y"], reps)
        Sigma = sigma_from_lineage(net, mid, reps)
        A = np.zeros((3, d)); A[:, n + k] = 1.0
        f.add_measurement(A, [r["margin"] for r in reps], Sigma)
        margins[mid] = net.estimate(mid)
    p_belief = 0.45
    f.add_bernoulli(d - 1, p_belief)
    cands, meta = [], {}
    rng = np.random.default_rng(3)
    for c in range(4):                                       # structure probes
        i, j = rng.choice(n, 2, replace=False)
        h = np.zeros(d); h[int(i)], h[int(j)] = 1.0, -1.0
        cands.append(Candidate(f"edge_{i}_{j}", h, 0.5, 1.0, "structure"))
        meta[cands[-1].id] = {"resistance": f.contrast_var(h), "cost": 1.0}
    far = max(cands, key=lambda c: meta[c.id]["resistance"])  # the weakest-connected pair is dear to probe
    far.cost = meta[far.id]["cost"] = 5.0
    for k, mid in enumerate(margins):
        h = np.zeros(d); h[n + k] = 1.0
        cands.append(Candidate(f"report_{mid}", h, 0.12, 1.0, "margin"))
        meta[cands[-1].id] = {"z": margins[mid].z, "s": margins[mid].s, "cost": 1.0}
    for r in (0.75, 0.95):
        hb, sb = bernoulli_probe(p_belief, r)
        h = np.zeros(d); h[d - 1] = hb
        cands.append(Candidate(f"probe_r{r}", h, sb, 1.0, "regime"))
        meta[cands[-1].id] = {"exact_bits": bernoulli_exact_bits(p_belief, r), "cost": 1.0}
    return f, cands, meta, margins


def test_t6_one_currency_decides_what_the_parts_cannot():
    f, cands, meta, margins = unified_instance()
    ranked = f.rank(cands)
    assert all(np.isfinite(v) and v > 0 for _, v, _, _ in ranked)
    assert ranked[0][2] > ranked[1][2] + 1e-6               # well defined: no tie at the top
    by_resistance = max((c for c in cands if c.kind == "structure"), key=lambda c: meta[c.id]["resistance"]).id
    by_z = min((c for c in cands if c.kind == "margin"), key=lambda c: abs(meta[c.id]["z"])).id
    by_regime = max((c for c in cands if c.kind == "regime"), key=lambda c: meta[c.id]["exact_bits"]).id
    assert len({by_resistance, by_z, by_regime}) == 3       # three parts, three different nominations
    joint = ranked[0][0]
    assert joint != by_resistance                           # cost and noise are invisible to resistance
    assert sum(joint == x for x in (by_z, by_regime)) <= 1
    # within one kind the joint order is not the part's order either
    struct = [r for r in ranked if r[3] == "structure"]
    assert struct[0][0] != by_resistance
