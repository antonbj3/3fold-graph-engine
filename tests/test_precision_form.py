"""precision_form: every claim in the module docstring, as a number.

T1 resistance = contrast variance of J⁺; T2 GLS mean/variance = margin_net m̂, s²; T3 Sherman–Morrison
= recomputed pinv (and the null-space branch where it is NOT); T4 matrix determinant lemma, the
restricted-Q inequality, the Bernoulli reductions; T5 chain rule, monotonicity, submodularity,
greedy vs brute force, set composition; T6 one currency across three kinds.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
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


# ==================================================================================================
# T7–T11: the five readings of one covariance — each thin method against the module it duplicates
# ==================================================================================================
from graph_engine import resistance_sketch as rs  # noqa: E402
from graph_engine.throws import dpp_inclusion_probabilities, draw_set_dpp  # noqa: E402


def _laplacian_form(n, extra, seed):
    edges, w = random_connected_graph(n, extra, seed)
    f = PrecisionForm.zeros(n).add_laplacian(edges, w)
    L = laplacian_from_edges(n, edges, w)[0].toarray()
    return f, edges, w, L


def _half_logpdet(J, tol=1e-8):
    """½ log of the PSEUDO-determinant of J (product of the eigenvalues above the tolerance)."""
    ev = np.linalg.eigvalsh((J + J.T) / 2)
    ev = ev[ev > tol * max(ev.max(), 1.0)]
    return 0.5 * float(np.log(ev).sum())


def test_t7_hole_field_is_diag_of_the_covariance_and_the_sketch_hole_field():
    n = 40
    f, edges, w, L = _laplacian_form(n, 30, 7)
    exact = np.diag(np.linalg.pinv(L, hermitian=True))
    assert np.abs(f.hole_field() - exact).max() < 1e-10                       # same matrix, read twice
    sk = ResistanceSketch.build(n, edges, w, k=256, seed=0)
    rel = np.abs(sk.hole_field() - f.hole_field()) / f.hole_field()
    assert rel.max() < 0.35 and rel.mean() < 0.12                            # sketch: its own √(2/k) error
    rel_ref = np.abs(sk.refined_hole_field(laplacian_from_edges(n, edges, w)[0]) - f.hole_field()) / f.hole_field()
    assert rel_ref.max() < rel.max()
    assert np.abs(f.hole_field(index=[3, 5]) - exact[[3, 5]]).max() < 1e-10


def test_t8_edge_leverage_from_the_form_sums_to_n_minus_one():
    n = 45
    f, edges, w, L = _laplacian_form(n, 40, 8)
    lev = f.edge_leverage(edges, w)
    assert abs(lev.sum() - (n - 1)) < 1e-8                                    # exact: spanning-tree count
    assert lev.min() > 0 and lev.max() <= 1 + 1e-9
    sk = ResistanceSketch.build(n, edges, w, k=256, seed=1)
    lev_sk = rs.edge_leverage(sk, edges, w)
    assert np.abs(lev_sk - lev).max() < 0.25                                  # sketch error only
    assert abs(lev_sk.sum() - (n - 1)) < 0.6 * (n - 1)
    r_exact = np.array([L.shape[0] and float((np.eye(n)[a] - np.eye(n)[b]) @ np.linalg.pinv(L, hermitian=True) @ (np.eye(n)[a] - np.eye(n)[b])) for a, b in edges])
    assert np.abs(lev / w - r_exact).max() < 1e-10


def test_t9_throw_set_is_the_same_dpp_as_throws_draw_set_dpp():
    n, sigma = 24, 0.7
    f, edges, w, L = _laplacian_form(n, 18, 9)
    cand = np.arange(n)
    C = f.cov()
    lam, V = np.linalg.eigh(C); lam = np.clip(lam, 0, None)
    Z = V * np.sqrt(lam)                                                      # Z Zᵀ = C
    assert np.abs(Z @ Z.T - C).max() < 1e-10
    quality = np.sqrt(np.diag(C)) / sigma                                     # absorbs _dpp_kernel's row normalization
    pi_throws = dpp_inclusion_probabilities(Z, quality)
    pi_form = f.throw_inclusion(cand, sigma)
    assert np.abs(pi_form - pi_throws).max() < 1e-12                          # identical marginals
    assert abs(pi_form.sum() - np.diag(C @ np.linalg.inv(np.eye(n) * sigma ** 2 + C)).sum()) < 1e-10
    same = 0
    for s in range(30):
        a = f.throw_set(cand, sigma, seed=s)[0]
        b = draw_set_dpp(Z, quality, seed=s)[0]
        same += int(len(a) == len(b) and np.array_equal(a, b))
    assert same >= 27                                                         # same algorithm, same seed
    # and the sampler's own marginals: Monte-Carlo frequency ≈ the exact inclusion probabilities
    cnt = np.zeros(n)
    for s in range(600):
        cnt[f.throw_set(cand, sigma, seed=1000 + s)[0]] += 1
    assert np.abs(cnt / 600 - pi_form).max() < 4 * np.sqrt(0.25 / 600) + 0.02
    m, p = f.throw_set(cand, sigma, seed=3)
    assert np.abs(p - pi_form[m]).max() < 1e-15 and len(set(m.tolist())) == len(m)


def test_t10_link_update_entropy_drop_and_kirchhoff_drop_are_two_different_potentials():
    n, w_new = 30, 1.7
    f, edges, w, L = _laplacian_form(n, 22, 10)
    pairs = [(0, 7), (3, 19), (11, 25)]
    logdet_bits, kirch_pred, kirch_real, logdet_real = [], [], [], []
    for i, j in pairs:
        g = f.copy()
        R = g.resistance(i, j)
        pred_bits = float(np.log1p(w_new * R) / (2 * np.log(2)))
        kirch_pred.append(g.kirchhoff_drop(i, j, w_new))
        tr_before = n * float(np.trace(g.cov()))
        J0 = g.J.copy()
        bits = g.observe_link(i, j, w_new)
        assert abs(bits - pred_bits) < 1e-12                                  # ½log₂(1 + w R_ij)
        # realized drop of ½log det C from a RECOMPUTED pseudo-inverse / pseudo-determinant
        realized = (_half_logpdet(g.J) - _half_logpdet(J0)) / np.log(2)
        logdet_bits.append(pred_bits); logdet_real.append(realized)
        kirch_real.append(tr_before - n * float(np.trace(np.linalg.pinv(g.J, hermitian=True))))
        assert np.abs(g.cov() - np.linalg.pinv(g.J, hermitian=True)).max() < 1e-9   # Sherman–Morrison exact
    assert max(abs(a - b) for a, b in zip(logdet_bits, logdet_real)) < 1e-9
    assert max(abs(a - b) for a, b in zip(kirch_pred, kirch_real)) < 1e-8
    # the sketched Kirchhoff drop of resistance_sketch predicts the same number (its own error)
    sk = ResistanceSketch.build(n, edges, w, k=512, seed=2, second_order=True)
    ii = np.array([p[0] for p in pairs]); jj = np.array([p[1] for p in pairs])
    sk_pred = sk.kirchhoff_drop(ii, jj, w_new)
    assert (np.abs(sk_pred - np.array(kirch_pred)) / np.array(kirch_pred)).max() < 0.5
    # the two potentials RANK links differently: not the same object
    cands = [(a, b) for a in range(n) for b in range(a + 1, n)][:120]
    bits = np.array([float(np.log1p(w_new * f.resistance(a, b)) / (2 * np.log(2))) for a, b in cands])
    kd = np.array([f.kirchhoff_drop(a, b, w_new) for a, b in cands])
    assert cands[int(np.argmax(bits))] != cands[int(np.argmax(kd))]
    assert spearmanr(bits, kd).statistic < 0.98


def test_t11_sparsify_from_the_form_keeps_resistances_like_the_sketch_version():
    n, q = 30, 6000
    f, edges, w, L = _laplacian_form(n, 45, 11)
    Cex = np.linalg.pinv(L, hermitian=True)
    iu = np.triu_indices(n, 1)
    R0 = np.array([Cex[a, a] + Cex[b, b] - 2 * Cex[a, b] for a, b in zip(*iu)])

    def worst_rel(e2, w2):
        L2 = laplacian_from_edges(n, e2, w2)[0].toarray()
        C2 = np.linalg.pinv(L2, hermitian=True)
        R2 = np.array([C2[a, a] + C2[b, b] - 2 * C2[a, b] for a, b in zip(*iu)])
        return float(np.abs(R2 / R0 - 1).max())

    err_f, err_s, sizes = [], [], []
    for s in range(6):
        e_f, w_f, lev_f = f.sparsify(edges, w, q=q, seed=s)
        e_s, w_s, lev_s = rs.sparsify(n, edges, w, q=q, k=256, seed=s)
        assert abs(lev_f.sum() - (n - 1)) < 1e-8                # exact leverages, sketch's are not
        assert np.abs(lev_s - lev_f).max() < 0.25
        err_f.append(worst_rel(e_f, w_f)); err_s.append(worst_rel(e_s, w_s)); sizes.append(len(e_f))
        assert n - 1 <= len(e_f) <= len(edges)
    assert max(err_f) < 0.20 and max(err_s) < 0.20              # same tolerance band, both directions
    assert np.mean(err_f) < 0.16 and np.mean(err_s) < 0.16
    assert np.mean(err_f) <= np.mean(err_s) + 0.02              # exact leverages never worse in the mean


def test_t11b_sparsify_actually_drops_edges_at_the_same_error_as_the_sketch():
    """A denser graph and a smaller sample: edges are removed, and the exact-leverage version and the
    sketched-leverage version land in the same error band (the guarantee is O(√(n log n / q)))."""
    n, q = 40, 1200
    f, edges, w, L = _laplacian_form(n, 160, 12)
    Cex = np.linalg.pinv(L, hermitian=True)
    iu = np.triu_indices(n, 1)
    R0 = np.array([Cex[a, a] + Cex[b, b] - 2 * Cex[a, b] for a, b in zip(*iu)])
    err_f, err_s, kept = [], [], []
    for s in range(4):
        e_f, w_f, _ = f.sparsify(edges, w, q=q, seed=s)
        e_s, w_s, _ = rs.sparsify(n, edges, w, q=q, k=256, seed=s)
        for e2, w2, acc in ((e_f, w_f, err_f), (e_s, w_s, err_s)):
            C2 = np.linalg.pinv(laplacian_from_edges(n, e2, w2)[0].toarray(), hermitian=True)
            R2 = np.array([C2[a, a] + C2[b, b] - 2 * C2[a, b] for a, b in zip(*iu)])
            acc.append(float(np.abs(R2 / R0 - 1).max()))
        kept.append(len(e_f))
    assert min(kept) < len(edges)                               # edges really are dropped here
    assert np.mean(err_f) < 0.55 and np.mean(err_s) < 0.55
    assert np.mean(err_f) <= np.mean(err_s) + 0.10



# ---------------------------------------------------------------------------------------------------
# T12–T15: coupled criticality (the joint form of two glued graphs) and the Marchenko–Pastur floor of
# every small-eigenvalue read-out taken from a k-dimensional sketch.
# ---------------------------------------------------------------------------------------------------
from graph_engine.precision_form import eigen_readout, joint_criticality, mp_floor  # noqa: E402
from graph_engine.precision_form import sequential_rank_tiebreak  # noqa: E402


def _glue_part(n_interior, w_in, c_shared, seed):
    """A healthy part: a strongly connected interior (coords 0..n−1) plus ONE shared concept (coord n)
    attached to every interior node, total conductance c_shared. Nothing measures the shared concept."""
    f = PrecisionForm.zeros(n_interior + 1)
    inside = [(i, j) for i in range(n_interior) for j in range(i + 1, n_interior)]
    f.add_laplacian(inside, [w_in] * len(inside))
    f.add_laplacian([(i, n_interior) for i in range(n_interior)], [c_shared / n_interior] * n_interior)
    return f


def test_t12_criticality_is_the_algebraic_connectivity_with_the_gauge_deflated():
    """The smallest NON-TRIVIAL eigenvalue: on a connected Laplacian-only form exactly λ₂ (Fiedler),
    with the one constant direction deflated; per connected block when there are several; and no
    deflation at all once a measurement makes the constant direction informative."""
    n = 40
    f, edges, w, L = _laplacian_form(n, 70, 5)
    lam = np.linalg.eigvalsh(L)
    r = f.criticality()
    assert r["n_blocks"] == 1 and r["null_dim"] == 1
    assert abs(r["sigma_min"] - lam[1]) < 1e-9                    # = algebraic connectivity, exactly
    v = r["direction"]
    assert abs(float(v @ np.ones(n))) < 1e-8                      # orthogonal to the deflated gauge
    assert abs(float(v @ L @ v) - lam[1]) < 1e-9
    # the energy shares are per-node shares of vᵀJv counted at both endpoints of every edge
    assert 1.99 < float(r["energy_share"].sum()) < 2.01
    # two disconnected blocks: two trivial null vectors, the read-out is the 3rd smallest eigenvalue
    g = PrecisionForm.zeros(2 * n)
    g.add_laplacian(edges, w); g.add_laplacian(edges + n, w)
    r2 = g.criticality()
    assert r2["n_blocks"] == 2 and r2["null_dim"] == 2
    assert abs(r2["sigma_min"] - np.linalg.eigvalsh(g.J)[2]) < 1e-9
    # one measurement on the constant direction: nothing is trivial any more
    h = np.ones(n) / np.sqrt(n)
    f2 = f.copy(); f2.observe(h, 1.0)
    r3 = f2.criticality()
    assert r3["null_dim"] == 0
    assert abs(r3["sigma_min"] - np.linalg.eigvalsh(f2.J)[0]) < 1e-9


def test_t13_joint_criticality_of_two_healthy_parts_glued_through_one_shared_variable():
    """Two parts, each σ_min ≈ 1.05, glued through one shared concept neither of them measures: the
    JOINT σ_min is 0.05 — 21× below either part. The weak direction's AMPLITUDE on the shared variable
    is ~0 (it sits at the neutral point of the mode); its ENERGY share is 1.0 against ≤ 0.025 for every
    other node, which is the read that actually finds the mediator.
    The parts are INDEPENDENTLY built (each owns its edges to the seam), so this is
    mode="federate": both seam couplings exist physically."""
    nA = nB = 20
    A, B = _glue_part(nA, 5.0, 1.0, 0), _glue_part(nB, 5.0, 1.0, 1)
    r = joint_criticality(A, B, {nA: nB}, mode="federate")
    assert r["sigma_min_a"] > 1.0 and r["sigma_min_b"] > 1.0          # each part healthy
    assert abs(r["sigma_min_joint"] - 0.05) < 1e-6
    assert r["ratio"] < 0.05                                          # 21× collapse on gluing
    s = r["shared"][0]
    assert s["energy_share"] > 0.999 and abs(s["amplitude"]) < 1e-6
    other = np.delete(r["report"]["energy_share"], s["joint"])
    assert other.max() < 0.03                                         # the mediator is unmistakable
    # WHY a per-part check cannot see it: the joint form restricted to one part's coordinates is that
    # part's form PLUS the other's contribution, so every direction living inside one part has Rayleigh
    # quotient ≥ that part's σ_min. The weak mode must be cross-part — and it is.
    J = r["joint"].J
    rng = np.random.default_rng(0)
    for coords, part_sigma in ((r["a_map"], r["sigma_min_a"]), (r["b_map"], r["sigma_min_b"])):
        for _ in range(200):
            u = np.zeros(len(J))
            x = rng.standard_normal(len(coords))
            u[coords] = x - x.mean()                                  # inside the part, gauge deflated
            u /= np.linalg.norm(u)
            assert float(u @ J @ u) >= part_sigma - 1e-9              # never below the part's own σ_min
    v = r["report"]["direction"]
    mass_a = float((v[r["a_map"]] ** 2).sum())
    assert 0.3 < mass_a < 0.7                                         # mass in BOTH interiors


def test_t13b_the_repair_is_a_contrast_not_a_level_and_sherman_morrison_is_exact_there():
    """What raises σ_min back, measured both ways. A rank-1 observation only lifts the weak mode where
    the mode has amplitude: h = e_s (the LEVEL of the shared variable) is orthogonal to it and does not
    raise σ_min at all — it lowers it, because measuring an absolute level also spends the gauge and
    creates a new soft direction 1/(σ²d). The observation that does work is the CONTRAST h = e_a − e_b
    across the seam (one cross link): σ_min 0.050 → 0.148. That h lies in range(J) of a connected
    Laplacian, so `observe` takes the Sherman–Morrison branch and it is exact to 1e-12 against the
    recomputed pseudo-inverse. Independently built parts: mode="federate"."""
    nA = nB = 20
    A, B = _glue_part(nA, 5.0, 1.0, 0), _glue_part(nB, 5.0, 1.0, 1)
    r = joint_criticality(A, B, {nA: nB}, mode="federate")
    J0, s = r["joint"], r["shared"][0]["joint"]
    base = r["sigma_min_joint"]

    lvl = {}
    for sigma in (1.0, 0.05):
        f = J0.copy(); h = np.zeros(f.d); h[s] = 1.0
        assert f.observe(h, sigma) is False                           # e_s ∉ range(J): null space shrinks
        lvl[sigma] = f.criticality()["sigma_min"]
    assert lvl[1.0] < base and lvl[0.05] < base                       # never "raises σ_min back"
    assert lvl[0.05] / base > 0.99                                    # a perfect pin only approaches it

    f = J0.copy()
    h = np.zeros(f.d); h[0] = 1.0; h[int(r["b_map"][0])] = -1.0
    assert f.in_range(h)
    bits = f.value_bits(h, 1.0)
    assert f.observe(h, 1.0) is True                                  # Sherman–Morrison branch
    C_sm = f.cov().copy()
    assert np.abs(C_sm - np.linalg.pinv(f.J, rcond=f.tol, hermitian=True)).max() < 1e-12
    after = f.criticality()["sigma_min"]
    assert after > 2.9 * base and bits > 0
    # and it is the seam that is repaired: the mediator's energy share drops
    assert f.criticality()["energy_share"][s] < 0.999


def test_t14_marchenko_pastur_floor_flags_a_pure_noise_sketch_and_spares_a_planted_signal():
    """The floor itself: for G = (1/k)ZZᵀ with Z an n×k i.i.d. Gaussian sketch (σ = 1.3), the whole
    spectrum is inside the bulk and the top eigenvalue matches the MP edge σ²(1+√(n/k))² to 1.4 %.
    With a spike planted 10× above the edge, exactly the signal eigenvalue is not flagged."""
    rng = np.random.default_rng(0)
    n, k, s = 200, 800, 1.3
    Z = rng.standard_normal((n, k)) * s
    G = Z @ Z.T / k
    r = eigen_readout(G, k, sigma2=s ** 2)
    assert abs(r["floor"] - mp_floor(n, k, s ** 2)) < 1e-12
    assert r["share_below"] >= 0.95 and r["n_above"] == 0
    assert abs(r["top"] / r["floor"] - 1) < 0.10                      # measured −1.4 %
    u = rng.standard_normal(n); u /= np.linalg.norm(u)
    theta = 10 * mp_floor(n, k, s ** 2)
    Z2 = Z + np.sqrt(theta) * np.outer(u, rng.standard_normal(k))
    r2 = eigen_readout(Z2 @ Z2.T / k, k, sigma2=s ** 2)
    assert r2["n_above"] == 1 and not r2["below_floor"][0]            # the signal survives the floor
    assert r2["eigenvalues"][0] > 5 * r2["floor"]
    assert abs(float(np.linalg.eigh(Z2 @ Z2.T / k)[1][:, -1] @ u)) > 0.95
    assert r2["share_below"] >= 0.99                                  # everything else is still bulk


def test_t15_how_much_of_a_sketched_dpp_kernel_is_sketch_noise():
    """Applied where the engine actually reads small eigenvalues off a sketch: the DPP kernel
    K = C_S/σ² of all 300 nodes of a graph, with C from a k-column resistance sketch instead of the
    dense pinv. At k = 64, 296 of 300 eigenvalues are inside the MP bulk (298 of 300 after
    diagonal normalisation) — the kernel is almost entirely sketch noise, and its top eigenvalue sits
    at the MP edge, which is what a pure-noise Gram does. The EXACT kernel's top eigenvalue is far
    BELOW the k = 64 floor: at that sketch width no eigenvalue of this kernel is a finding at all.
    Widening the sketch lowers the floor monotonically, which is the only way to buy the read-out."""
    n = 300
    edges, w = random_connected_graph(n, 600, 3)
    f = PrecisionForm.zeros(n).add_laplacian(edges, w)
    K_exact = f._dpp_kernel(np.arange(n))

    def corr(M):
        d = np.sqrt(np.clip(np.diag(M), 1e-300, None))
        return M / np.outer(d, d)

    below, below_norm = {}, {}
    for k in (16, 64, 256):
        Z = ResistanceSketch.build(n, edges, w, k=k, seed=1).Z
        K = Z @ Z.T                                                   # the sketch's C, the DPP kernel's
        r = eigen_readout(K, k)                                       # plug-in null σ̂² = mean diag
        rn = eigen_readout(corr(K), k, sigma2=1.0)                    # unit-diagonal convention
        below[k], below_norm[k] = r["n_below"], rn["n_below"]
        assert 0.95 < rn["top"] / rn["floor"] < 1.35                  # the top hugs the MP edge
        if k == 64:
            assert r["n_below"] == 296 and rn["n_below"] == 298
            assert eigen_readout(corr(K_exact), k, sigma2=1.0)["n_below"] == n
    assert below[16] >= below[64] >= below[256]                       # more columns, lower floor
    assert below_norm[16] >= below_norm[64] >= below_norm[256]
    assert below[256] < n                                             # at k = 256 something clears it


# -- e35: the EXACT set value of Bernoulli probes (no Gaussian image) --------------------------------
from graph_engine.precision_form import (  # noqa: E402
    exact_bernoulli_chain, exact_bernoulli_one_belief, exact_bernoulli_set_value, p_of_theta,
    sphere_error_grid, sphere_set_value, theta_of_p,
)


def _ent_bits(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def brute_joint_bernoulli_value(beliefs, probes):
    """Independent reference for the exact set value: the full 2^m truths × 2^k answers table,
    H(X) − Σ_y P(y) H(X|y). No structure assumed — not additivity, not the chain rule."""
    m, k = len(beliefs), len(probes)
    x = ((np.arange(1 << m)[:, None] >> np.arange(m)[None, :]) & 1).astype(float)
    p = np.asarray(beliefs, float)
    prior = np.prod(np.where(x > 0, p, 1 - p), axis=1)
    y = ((np.arange(1 << k)[:, None] >> np.arange(k)[None, :]) & 1).astype(float)
    lik = np.ones((1 << m, 1 << k))
    for j, (b, r) in enumerate(probes):
        lik *= np.where(x[:, None, int(b)] == y[None, :, j], float(r), 1.0 - float(r))
    joint = prior[:, None] * lik
    Py = joint.sum(0)
    post = joint / np.where(Py > 0, Py, 1.0)
    return float(_ent_bits(prior) - float(np.sum(Py * np.array([_ent_bits(post[:, c]) for c in range(1 << k)]))))


def test_e35_independent_beliefs_are_additive():
    """(a) Independent beliefs, one probe each: the joint posterior factorizes, so the exact set value
    is the SUM of the single exact values — against a brute enumeration over 2^m × 2^k, to 1e-12."""
    rng = np.random.default_rng(0)
    for _ in range(60):
        m = int(rng.integers(2, 5))
        beliefs = list(rng.uniform(0.05, 0.95, m))
        probes = [(b, float(rng.uniform(0.55, 0.99))) for b in range(m)]
        s = sum(bernoulli_exact_bits(beliefs[b], r) for b, r in probes)
        assert abs(exact_bernoulli_set_value(beliefs, probes) - s) < 1e-12
        assert abs(brute_joint_bernoulli_value(beliefs, probes) - s) < 1e-12


def test_e35_independent_beliefs_monte_carlo():
    """The same identity against Monte Carlo (sample truth, sample answers, average posterior entropy)."""
    rng = np.random.default_rng(4)
    beliefs = [0.3, 0.55, 0.8]
    probes = [(0, 0.7), (1, 0.85), (2, 0.9)]
    ex = exact_bernoulli_set_value(beliefs, probes)
    n = 200_000
    x = (rng.random((n, 3)) < np.array(beliefs)).astype(int)
    prior_h = sum(_hb_prior(p) for p in beliefs)
    post_h = np.zeros(n)
    for b, r in probes:                                 # the joint posterior factorizes: entropies add
        truth = x[:, b]
        y = np.where(rng.random(n) > r, 1 - truth, truth)
        p = beliefs[b]
        qy = p * r + (1 - p) * (1 - r)
        post = np.where(y == 1, p * r / qy, p * (1 - r) / (1 - qy))
        post_h += -(post * np.log2(post) + (1 - post) * np.log2(1 - post))
    assert abs((prior_h - float(post_h.mean())) - ex) < 5e-3


def _hb_prior(p):
    return _ent_bits([p, 1 - p])


def test_e35_chain_rule_on_one_belief():
    """(b) Several probes on ONE belief: enumeration over 2^k outcomes = the chain rule
    Σ_k E[value of probe k | answers 1..k−1], in every order, and = the brute joint table."""
    rng = np.random.default_rng(1)
    for _ in range(40):
        k = int(rng.integers(1, 7))
        p = float(rng.uniform(0.05, 0.95))
        rels = list(rng.uniform(0.55, 0.99, k))
        ex = exact_bernoulli_one_belief(p, rels)
        total, terms = exact_bernoulli_chain(p, rels)
        assert abs(total - ex) < 1e-12 and abs(sum(terms) - ex) < 1e-12
        assert abs(brute_joint_bernoulli_value([p], [(0, r) for r in rels]) - ex) < 1e-12
        idx = rng.permutation(k)
        assert abs(exact_bernoulli_chain(p, [rels[i] for i in idx])[0] - ex) < 1e-12
    rels12 = list(np.random.default_rng(2).uniform(0.55, 0.95, 12))          # k = 12 still enumerable
    ex12 = exact_bernoulli_one_belief(0.37, rels12)
    assert abs(exact_bernoulli_chain(0.37, rels12)[0] - ex12) < 1e-12
    assert ex12 < sum(bernoulli_exact_bits(0.37, r) for r in rels12)          # probes on one belief overlap
    assert ex12 < _hb_prior(0.37)                                            # and can never exceed h(p)


def test_e35_mixed_case_sums_over_beliefs():
    """(c) Several beliefs each with several probes = sum over beliefs of (b), against the brute table."""
    beliefs = [0.2, 0.5, 0.8]
    probes = [(0, 0.7), (0, 0.9), (1, 0.6), (1, 0.6), (1, 0.85), (2, 0.95)]
    per = [exact_bernoulli_one_belief(beliefs[b], [r for bb, r in probes if bb == b]) for b in range(3)]
    v = exact_bernoulli_set_value(beliefs, probes)
    assert abs(v - sum(per)) < 1e-12
    assert abs(v - brute_joint_bernoulli_value(beliefs, probes)) < 1e-12
    assert v < sum(bernoulli_exact_bits(beliefs[b], r) for b, r in probes)


def test_e35_sphere_coordinate_and_second_order_error():
    """The flat Fisher coordinate θ = 2 arcsin√p (ds = dθ) and the second-order value Var(Δθ)/(2 ln 2):
    exact in the small-step limit, and measurably wrong for informative probes."""
    ps = np.linspace(0.02, 0.98, 25)
    assert np.allclose(p_of_theta(theta_of_p(ps)), ps, atol=1e-12)
    d = 1e-6                                                        # ds/dp = 1/sqrt(p(1-p))
    num = (theta_of_p(ps + d) - theta_of_p(ps - d)) / (2 * d)
    assert np.max(np.abs(num - 1 / np.sqrt(ps * (1 - ps)))) < 1e-6
    tiny = abs(sphere_set_value([0.5], [(0, 0.5 + 1e-4)]) / exact_bernoulli_set_value([0.5], [(0, 0.5 + 1e-4)]) - 1)
    assert tiny < 1e-6                                              # second order is exact as the step → 0
    g = sphere_error_grid(np.linspace(0.02, 0.98, 49), [0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99])
    assert 0.40 < g["max_rel_error"] < 0.55                         # what the local/Gaussian read pays
    assert g["median_rel_error"] < 0.10
    assert g["argmax_r"] == 0.99                                    # the error is in the LARGE steps
    assert sphere_set_value([0.5], [(0, 0.99)]) > exact_bernoulli_set_value([0.5], [(0, 0.99)])  # overshoots


def test_e35_exact_route_in_set_value_bits_on_t6():
    """(d) The Bernoulli block's value is replaced by the exact form when every row of the set is a
    probe on an independent registered belief. e24's T6 instance: the ranking of the Bernoulli
    candidates is unchanged, and their bits now equal the exact two-outcome values."""
    f, cands, meta, _ = unified_instance()
    old = f.rank(cands)
    new = f.rank(cands, exact_bernoulli=True)
    bern = [c.id for c in cands if c.kind == "regime"]
    assert [i for i, _, _, _ in old if i in bern] == [i for i, _, _, _ in new if i in bern]
    nb = {i: v for i, v, _, _ in new}
    ob = {i: v for i, v, _, _ in old}
    for c in cands:
        if c.kind == "regime":
            assert abs(nb[c.id] - meta[c.id]["exact_bits"]) < 1e-12          # exact, not the image
            assert abs(ob[c.id] / meta[c.id]["exact_bits"] - 1) > 0.1        # the image was 14-40 % off
        else:
            assert abs(nb[c.id] - ob[c.id]) < 1e-12                          # nothing else moves
    single = [c for c in cands if c.id == "probe_r0.95"][0]
    assert abs(f.set_value_bits(np.atleast_2d(single.h), single.sigma) - meta["probe_r0.95"]["exact_bits"]) < 1e-12
    assert abs(f.set_value_bits(np.atleast_2d(single.h), single.sigma, exact_bernoulli=False)
               - f.value_bits(single.h, single.sigma)) < 1e-12               # the image is still reachable
    # a MIXED set keeps the Gaussian image, and the residual is the belief row's own gap
    mix = [c for c in cands if c.kind == "structure"][:1] + [c for c in cands if c.kind == "margin"][:1] + [single]
    H = np.array([c.h for c in mix]); s = np.array([c.sigma for c in mix])
    assert f.bernoulli_rows(H) is None
    gauss = f.set_value_bits(H, s)
    hybrid = f.set_value_bits(H[:2], s[:2]) + meta["probe_r0.95"]["exact_bits"]
    assert gauss < hybrid and 0.1 < abs(gauss / hybrid - 1) < 0.2


def test_e35_exact_route_gate_is_conservative():
    """The exact route fires only where the discrete model is the whole story: not on unregistered
    coordinates, not on multi-coordinate rows, and not once the belief is coupled or updated."""
    f = PrecisionForm.zeros(2).add_bernoulli(0, 0.4)
    h, s = bernoulli_probe(0.4, 0.8)
    assert f.bernoulli_rows(np.array([[h, 0.0]])) is not None
    assert f.bernoulli_rows(np.array([[h, h]])) is None                      # two coordinates: not a probe
    assert f.bernoulli_rows(np.array([[0.0, h]])) is None                    # coordinate 1 is not a belief
    g = PrecisionForm.zeros(2).add_bernoulli(0, 0.4).add_bernoulli(1, 0.6)
    two = exact_bernoulli_set_value([0.4, 0.6], [(0, 0.8), (1, 0.9)])
    h2, s2 = bernoulli_probe(0.6, 0.9)
    assert abs(g.set_value_bits(np.array([[h, 0.0], [0.0, h2]]), np.array([s, s2])) - two) < 1e-12
    same = exact_bernoulli_one_belief(0.4, [0.8, 0.8])                       # two probes on ONE belief
    assert abs(g.set_value_bits(np.array([[h, 0.0], [h, 0.0]]), np.array([s, s])) - same) < 1e-12
    assert same < 2 * bernoulli_exact_bits(0.4, 0.8)
    g.observe(np.array([h, 0.0]), s)                                         # the belief is no longer at its prior
    assert g.bernoulli_rows(np.array([[h, 0.0]])) is None


def test_e35_exact_set_value_is_submodular_on_one_belief():
    """(e) Diminishing returns of the exact set value, 500 random (p, reliabilities, A ⊆ B, e):
    I(X;Y_S) is monotone submodular for observations conditionally independent given X."""
    rng = np.random.default_rng(35)
    viol, strict = 0, 0
    for _ in range(500):
        p = float(rng.uniform(0.02, 0.98))
        n = int(rng.integers(2, 8))
        rel = list(rng.uniform(0.5, 0.999, n))
        idx = list(rng.permutation(n))
        e, rest = idx[0], idx[1:]
        nb = int(rng.integers(0, len(rest) + 1)); B = rest[:nb]
        na = int(rng.integers(0, nb + 1)); A = B[:na]
        gA = exact_bernoulli_one_belief(p, [rel[i] for i in A + [e]]) - exact_bernoulli_one_belief(p, [rel[i] for i in A])
        gB = exact_bernoulli_one_belief(p, [rel[i] for i in B + [e]]) - exact_bernoulli_one_belief(p, [rel[i] for i in B])
        assert gB >= -1e-12                                                  # monotone
        viol += int(gA - gB < -1e-12)
        strict += int(na < nb)
    assert viol == 0 and strict > 100
    gains = [exact_bernoulli_one_belief(0.4, [0.8] * k) - exact_bernoulli_one_belief(0.4, [0.8] * (k - 1))
             for k in range(1, 9)]
    assert all(gains[i] > gains[i + 1] for i in range(len(gains) - 1))        # strictly diminishing


# ---------------------------------------------------------------------------------------------------
# T16: the energy read beside the amplitude read, on a measured chain.
#
# A contact-solver study formed the Delassus operator G = J M⁻¹ Jᵀ of the chain
# ground – A0 A1 A2 – [mediator] – B0 B1 B2 explicitly: seven contacts, and the mediating body's
# INVERSE MASS is the conductance of the contact-space edge (3,4), so the two directions of an
# unhealthy mediator land at opposite ends of the spectrum. The form below is that contact graph —
# a 7-node chain, unit conductances except edge (3,4) = 1/m, and the far end anchored — and it
# reproduces that study's σ_min, σ_max, amplitude and energy_share row by row.
# ---------------------------------------------------------------------------------------------------

def _mediator_chain(m_mediator):
    """The measured contact-space form: conductance 1/m on the mediating edge (3,4), B-side anchored."""
    f = PrecisionForm.zeros(7)
    w = [1.0] * 6
    w[3] = 1.0 / m_mediator
    f.add_laplacian([(i, i + 1) for i in range(6)], w)
    f.J[6, 6] += 1.0
    return f


# measured: m_mediator, σ_min(G), σ_max(G), node carrying the largest energy share
MEDIATOR_SWEEP = ((1e-3, 5.0949e-2, 2.0010e+3, 6), (1e-2, 5.0876e-2, 2.0101e+2, 6),
                  (1e-1, 5.0154e-2, 2.1053e+1, 6), (1e+0, 4.3705e-2, 3.8271e+0, 6),
                  (1e+1, 1.7643e-2, 3.4227e+0, 4), (1e+2, 2.4059e-3, 3.4150e+0, 4),
                  (1e+3, 2.4903e-4, 3.4143e+0, 4), (1e+4, 2.4990e-5, 3.4142e+0, 4))


def test_t16_criticality_reproduces_the_measured_sweep():
    """σ_min and σ_max of the eight measured rows, to four digits. A HEAVY mediator collapses σ_min by
    175× (4.3705e-2 → 2.4903e-4) — that is the coupling criticality this read is for. A LIGHT one moves
    σ_max instead (2.0010e+3 at m = 1e-3 against 3.8271e+0 at m = 1) while σ_min moves by 16 %, which is
    the documented blind spot: the classic mass-ratio bottleneck is not a σ_min finding."""
    for m, sigma_min, sigma_max, _ in MEDIATOR_SWEEP:
        f = _mediator_chain(m)
        r = f.criticality()
        assert r["sigma_min"] == pytest.approx(sigma_min, rel=1e-4), f"m = {m:g}"
        assert float(np.linalg.eigvalsh(f.J)[-1]) == pytest.approx(sigma_max, rel=1e-4), f"m = {m:g}"
    light, heavy = _mediator_chain(1e-3).criticality(), _mediator_chain(1e+0).criticality()
    assert abs(light["sigma_min"] / heavy["sigma_min"] - 1) < 0.17          # σ_min barely moves …
    assert np.linalg.eigvalsh(_mediator_chain(1e-3).J)[-1] / np.linalg.eigvalsh(_mediator_chain(1e+0).J)[-1] > 500


def test_t16_energy_share_names_the_mediator_that_the_amplitude_argmax_misses():
    """The measured mode at m = 1e3: |amplitude| [0.5003 0.5002 0.4999 0.4996 0.0015 0.0010 0.0005],
    energy_share [0.0001 0.0003 0.0008 0.9967 0.9971 0.0020 0.0020]. The amplitude argmax is node 0,
    which mediates nothing; the energy read names both mediating contacts (3 and 4) at 0.997. Both reads
    are returned, and the disagreement is a field of the result rather than something a caller has to
    notice."""
    r = _mediator_chain(1e3).criticality()
    assert np.allclose(np.abs(r["direction"]), [0.5003, 0.5002, 0.4999, 0.4996, 0.0015, 0.0010, 0.0005],
                       atol=5e-5)
    assert np.allclose(r["energy_share"], [0.0001, 0.0003, 0.0008, 0.9967, 0.9971, 0.0020, 0.0020], atol=5e-5)
    assert r["amplitude_argmax"] == 0 and r["energy_argmax"] == 4
    assert not r["amplitude_finds_the_load"]
    assert [n for n, _ in r["energy_loading"][:2]] == [4, 3]                # both mediating contacts, first
    assert all(s > 0.99 for _, s in r["energy_loading"][:2])
    assert [n for n, _ in r["loading"][:2]] == [0, 1]                       # what amplitude would have said
    # every EDGE is counted at both of its endpoints, so the shares sum to 2 up to the anchor's own term,
    # which belongs to one node only
    assert float(np.sum(r["energy_share"])) == pytest.approx(2.0, rel=1e-3)


def test_t16_the_two_reads_agree_where_there_is_nothing_to_mediate():
    """Where the soft direction is one weakly measured coordinate and nothing mediates it, both reads name
    that coordinate — the disagreement flag is a finding, not a constant."""
    r = PrecisionForm(np.diag([0.01, 1.0, 1.0]), np.zeros(3)).criticality()
    assert r["sigma_min"] == pytest.approx(0.01)
    assert r["amplitude_finds_the_load"] and r["amplitude_argmax"] == r["energy_argmax"] == 0
    assert r["energy_share"][0] == pytest.approx(1.0)


# -- T17: the share sum is a measurement, and the share is a share -------------------------

def test_t17_share_sum_is_two_on_a_pure_laplacian_and_a_fraction_there():
    """On a pure Laplacian (r = 0) the edge-counted shares sum to exactly 2, the unit-sum
    fraction sums to 1, no conductance is negative, and the load-bearing read is valid."""
    rng = np.random.default_rng(7)
    n = 12
    edges = [(i, int(rng.integers(0, n))) for i in range(n) for _ in range(2)]
    edges = [(a, b) for a, b in edges if a != b]
    f = PrecisionForm.zeros(n).add_laplacian(edges, [1.0] * len(edges))
    r = f.criticality()
    assert r["energy_share_sum"] == pytest.approx(2.0, rel=1e-9)
    assert float(r["energy_fraction"].sum()) == pytest.approx(1.0, rel=1e-9)
    assert r["negative_conductances"] == 0
    assert r["load_bearing_valid"] is True


def test_t17b_share_sum_follows_the_identity_with_diagonal_mass():
    """Laplacian + diagonal mass m (measurement blocks): the correct sum is
    2 − (Σ r_i v_i²)/(vᵀJv) to the digit, the fraction still sums to 1, and the
    load-bearing flag is off. Same check as on the contact-solver measurements, where the
    sum was −2.115112 on the contact-space operator and 53.004463 on a Fisher form, both
    against this identity to 9e-16."""
    rng = np.random.default_rng(11)
    n = 10
    edges = [(i, (i + 1) % n) for i in range(n)] + [(i, (i + 3) % n) for i in range(n)]
    f = PrecisionForm.zeros(n).add_laplacian(edges, [1.0] * len(edges))
    m = np.linspace(0.5, 2.0, n)
    f.J = f.J + np.diag(m)
    f._dirty()
    r = f.criticality()
    v = r["direction"]
    W = -f.J.copy(); np.fill_diagonal(W, 0.0)
    rr = np.diag(f.J) - W.sum(1)
    expect = 2.0 - float(rr @ (v ** 2)) / float(v @ f.J @ v)
    assert r["energy_share_sum"] == pytest.approx(expect, rel=1e-9)
    assert abs(expect - 2.0) > 0.1                       # off the Laplacian, measurably
    assert float(r["energy_fraction"].sum()) == pytest.approx(1.0, rel=1e-9)
    assert r["load_bearing_valid"] is False


def test_t17c_signed_conductances_are_counted_and_refuse_the_read():
    """One negative off-diagonal conductance: counted, fraction still unit-sum, flag off."""
    J = np.array([[2.0, -1.0, -0.5], [-1.0, 2.0, 0.3], [-0.5, 0.3, 1.2]])
    J = 0.5 * (J + J.T) + np.eye(3) * 0.5
    r = PrecisionForm(J, np.zeros(3)).criticality()
    assert r["negative_conductances"] >= 1
    assert r["load_bearing_valid"] is False
    assert float(r["energy_fraction"].sum()) == pytest.approx(1.0, rel=1e-9)


# -- T18: shared rows assembled once --------------------------------------------------------

def _split_true_form():
    """A known 5x5 coupled form of the shape this function states: two parts that touch ONLY
    through the shared coordinate 2 (no direct interior-A/interior-B block, exactly what was
    measured on the contact-solver operator, where that block was 0.0), split into the two
    principal submatrices [0,1,2] and [2,3,4]."""
    rng = np.random.default_rng(3)
    PA = rng.standard_normal((3, 3)); PA = PA @ PA.T + np.eye(3)
    PB = rng.standard_normal((3, 3)); PB = PB @ PB.T + np.eye(3)
    A_set, B_set = [0, 1, 2], [2, 3, 4]
    J = np.zeros((5, 5))
    J[np.ix_(A_set, A_set)] += PA
    J[np.ix_(B_set, B_set)] += PB
    fa = PrecisionForm(J[np.ix_(A_set, A_set)].copy(), np.zeros(3))
    fb = PrecisionForm(J[np.ix_(B_set, B_set)].copy(), np.zeros(3))
    return J, fa, fb


def test_t18_split_mode_recovers_the_true_coupled_form():
    """Sides that are principal submatrices of ONE matrix: split mode assembles the shared
    block once and returns the true coupled matrix, entry for entry. On contact-solver
    measurements (9 contacts, 4 shared) that reconstruction matched the coupled operator to
    0.0 absolute and its sigma_min to 5.546751e-01, against the seam-doubled 8.496287e-01 —
    the 1.5318x upper bound this fixes."""
    J, fa, fb = _split_true_form()
    r = joint_criticality(fa, fb, {2: 0}, mode="split")
    assert r["mode"] == "split"
    assert np.abs(r["joint"].J - J).max() == 0.0
    assert r["sigma_min_joint"] == pytest.approx(float(np.linalg.eigvalsh(J)[0]), rel=1e-9)


def test_t18b_federate_mode_keeps_both_seam_couplings():
    """Independently built parts keep both copies (the T13 phenomenon lives here, unchanged);
    on a split pair federate doubles the seam block and lands ABOVE the true sigma_min —
    an upper bound, which is exactly the old default behavior."""
    J, fa, fb = _split_true_form()
    r = joint_criticality(fa, fb, {2: 0}, mode="federate")
    true = float(np.linalg.eigvalsh(J)[0])
    assert r["sigma_min_joint"] > true
    assert r["joint"].J[2, 2] == pytest.approx(2.0 * J[2, 2])


def test_t18c_split_mode_fails_closed_on_disagreeing_seams():
    """Split mode with sides that are NOT submatrices of one matrix raises instead of
    halving or doubling a seam."""
    _, fa, fb = _split_true_form()
    fb2 = PrecisionForm(fb.J.copy(), fb.b.copy(), fb.tol)
    fb2.J[0, 0] += 1.0
    with pytest.raises(ValueError):
        joint_criticality(fa, fb2, {2: 0}, mode="split")
    fb3 = PrecisionForm(fb.J.copy(), fb.b.copy(), fb.tol)
    fb3.b[0] += 1.0                                   # the seam's b must agree too
    with pytest.raises(ValueError):
        joint_criticality(fa, fb3, {2: 0}, mode="split")
    with pytest.raises(ValueError):
        joint_criticality(fa, fb, {2: 0}, mode="bogus")
    # the two Fisher forms of independent observation channels: information ADDS at the
    # seam, so their shared blocks differ by construction and split must refuse them
    ga = PrecisionForm(np.diag([1.0, 1.0, 5.256006e5]), np.zeros(3))
    gb = PrecisionForm(np.diag([5.482346e3, 1.0, 1.0]), np.zeros(3))
    with pytest.raises(ValueError):
        joint_criticality(ga, gb, {2: 0}, mode="split")
    assert joint_criticality(ga, gb, {2: 0}, mode="federate")["joint"].J[2, 2] == \
        pytest.approx(5.256006e5 + 5.482346e3)


# -- T19: the plateau cycles instead of sticking ---------------------------------------------

def test_t19_plateau_cycles_round_robin_and_covers_first():
    """Four contacts, set values tied at 2e-6 — the measured post-coverage plateau, where every
    belief has saturated at 1.0 and the value is blind — with the four measured per-contact
    torques as the price. Every pass over the pool takes all four contacts, cheapest first
    (3, 2, 1, 0 for these torques), instead of sticking to one: the measured stuck rule spent
    probes 6, 7, 8 on one contact. The pass constraint is the whole repair; nothing else on
    the plateau carries a gradient."""
    costs = [5.2135, 5.1670, 4.4467, 4.4070]     # |tau|_2 (N m) at probes k = 1..4
    vals = [[2e-6] * 4 for _ in range(8)]
    gains = [[1] * 4 for _ in range(8)]
    picks = sequential_rank_tiebreak(vals, gains, costs)
    assert picks == [3, 2, 1, 0, 3, 2, 1, 0]
    assert set(picks[:4]) == {0, 1, 2, 3} and set(picks[4:]) == {0, 1, 2, 3}
    stuck = [min(range(4), key=lambda i: costs[i])] * 8      # value-blind argmin, no pass rule
    assert len(set(stuck)) == 1 and len(set(picks)) == 4


def test_t19b_coverage_gain_beats_cost_before_the_plateau():
    """While a candidate is still uncovered (gain 1 vs 0) the gain decides, whatever the
    price: the expensive uncovered candidate is picked before the cheap covered ones."""
    picks = sequential_rank_tiebreak(
        [[2e-6] * 4], [[0, 0, 0, 1]], [1.0, 1.0, 1.0, 100.0])
    assert picks == [3]


def test_t19c_round_robin_beats_the_stuck_rule_on_the_measured_numbers():
    """The recorded outcome of the four-contact probing study: cycling 0,1,2,3,0,1,2,3
    reached sigma_min 1.831658e6 (max CRB 7.389e-4), the value-blind rule that stuck,
    0,1,2,3,1,0,0,0, reached 1.060295e6 (max CRB 9.712e-4) — 1.73x at 1.5 % less actuation.
    sigma_min = 1/maxCRB^2 holds on both recorded final CRBs, so the inequality is
    arithmetic on the measured rows, not a re-simulation."""
    rr, circ = 1.831658e6, 1.060295e6
    assert 1.0 / 7.389e-4 ** 2 == pytest.approx(rr, rel=1e-3)
    assert 1.0 / 9.712e-4 ** 2 == pytest.approx(circ, rel=1e-3)
    assert rr / circ == pytest.approx(1.727, rel=1e-2)
