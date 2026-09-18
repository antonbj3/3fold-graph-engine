"""Tests for resistance_sketch, claim_federation and replay_policy. Each identity the modules rely on
is checked numerically against the dense/exact computation."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
pytest.importorskip("scipy")

from graph_engine.resistance_sketch import (QuantizedSketch, ResistanceSketch, laplacian_from_edges,  # noqa: E402
                                            lloyd_max_gaussian)
from graph_engine.claim_federation import Federation, box_gap_point, box_intersection  # noqa: E402


def _graph(n=120, seed=0):
    rng = np.random.default_rng(seed)
    e = [(i, i + 1) for i in range(n - 21)] + [(int(a), int(b)) for a, b in rng.integers(0, n - 20, (300, 2)) if a != b]
    e += [(p, int(rng.integers(n - 20))) for p in range(n - 20, n)]          # 20 pendants
    e = np.unique(np.sort(np.array(e), 1), axis=0)
    return n, e


# ---------------------------------------------------------------- resistance sketch
def test_sketch_identity_L_pinv_is_a_gram_matrix():
    n, e = _graph()
    L, B, w = laplacian_from_edges(n, e)
    Lp = np.linalg.pinv(L.toarray())
    assert np.allclose(Lp @ (B.T @ B).toarray() @ Lp, Lp, atol=1e-9)         # L⁺BᵀWBL⁺ = L⁺ (exact)
    G = np.mean([(lambda Z: Z @ Z.T)(ResistanceSketch.build(n, e, k=64, seed=s).Z) for s in range(40)], 0)
    assert np.abs(G - Lp).max() < 0.08 * np.abs(Lp).max()                    # E[ZZᵀ] = L⁺


def test_refinement_identity_is_exact_and_reduces_error():
    n, e = _graph()
    L, _, _ = laplacian_from_edges(n, e)
    Lp = np.linalg.pinv(L.toarray()); d = L.diagonal(); A = np.diag(d) - L.toarray()
    assert np.allclose((1 - 1 / n + np.einsum("ij,ij->i", A, Lp)) / d, np.diag(Lp), atol=1e-9)
    plain, refined = [], []
    for s in range(10):
        sk = ResistanceSketch.build(n, e, k=32, seed=s)
        plain.append(np.median(np.abs(sk.hole_field() - np.diag(Lp)) / np.diag(Lp)))
        refined.append(np.median(np.abs(sk.refined_hole_field(L) - np.diag(Lp)) / np.diag(Lp)))
    assert np.mean(refined) < 0.6 * np.mean(plain)


def test_kirchhoff_drop_is_sherman_morrison():
    n, e = _graph(60)
    L, _, _ = laplacian_from_edges(n, e)
    Lp = np.linalg.pinv(L.toarray()); i, j = 3, 55
    L2, _, _ = laplacian_from_edges(n, np.vstack([e, [[i, j]]]))
    exact = n * (np.trace(Lp) - np.trace(np.linalg.pinv(L2.toarray())))
    u = Lp[:, i] - Lp[:, j]
    assert np.isclose(exact, n * (u @ u) / (1 + Lp[i, i] + Lp[j, j] - 2 * Lp[i, j]), rtol=1e-8)
    est = np.mean([ResistanceSketch.build(n, e, k=256, seed=s, second_order=True).kirchhoff_drop(np.array([i]), np.array([j]))[0] for s in range(8)])
    assert abs(est - exact) / exact < 0.15


def test_lloyd_max_centroid_condition_and_4bit_cost():
    lev, edges = lloyd_max_gaussian(4)
    x = np.random.default_rng(0).standard_normal(400_000); q = lev[np.searchsorted(edges, x)]
    assert abs((x * q).mean() - (q * q).mean()) < 5e-3                      # E[x·x̂] = E[x̂²]
    assert 0.009 < ((x - q) ** 2).mean() < 0.0105                           # 16-level Lloyd–Max MSE ≈ 0.0095
    n, e = _graph()
    sk = ResistanceSketch.build(n, e, k=64, seed=1); qs = QuantizedSketch.build(sk.Z, bits=4)
    assert np.median(np.abs(qs.hole_field() - sk.hole_field()) / sk.hole_field()) < 0.03
    assert qs.nbytes() == n * (64 * 4 // 8 + 4)
    ip = qs.inner(sk.Z[7]); exact = sk.Z @ sk.Z[7]
    assert np.corrcoef(ip, exact)[0, 1] > 0.98


def test_hole_engine_sketch_opt_in_matches_dense_ranking():
    from graph_engine.graph_hole_engine.engine import DomainAdapter, GraphHoleEngine
    n, e = _graph()
    L = laplacian_from_edges(n, e)[0].toarray()

    class A(DomainAdapter):
        def nodes(self): return list(range(n))
        def governing_operator(self): return L
        def c_obs(self, node): return 0.0
        def c_pred(self, node): return 0.0
        def confirmer_matrix(self, node): return np.zeros((2, 1))
    dense, sk = GraphHoleEngine(A()).hole_field(), GraphHoleEngine(A(), sketch_k=64).hole_field()
    assert len(set(np.argsort(-dense)[:20]) & set(np.argsort(-sk)[:20])) >= 15
    assert set(np.argsort(-dense)[:20]) <= set(range(n - 20, n))             # the holes are the pendants


# ---------------------------------------------------------------- federation
def _fed(**kw):
    f = Federation(**kw)
    for g in json.load(open(REPO_ROOT / "examples" / "engine_experiments" / "curated_two_graphs.json"))["graphs"]:
        f.add_graph(g)
    return f


def test_n_eff_independent_copies_and_shared_origin():
    f = Federation()
    f.add_graph({"graph_id": "T", "sources": [{"id": "r"}, {"id": "c1", "derives_from": ["r"]}, {"id": "c2", "derives_from": ["c1"]}], "claims": []})
    assert np.isclose(f.n_eff(["a", "b", "c"]), 3) and np.isclose(f.n_eff(["r", "c1", "c2"]), 1)
    assert np.isclose(f.n_eff(["r", "c1", "x"]), 2)
    g = _fed()
    assert np.isclose(g.n_eff(["Hall1951", "Petch1953", "Textbook-HP", "ProcessHandbook"]), 2)
    assert np.isclose(g.n_eff(["Hall1951", "ProcessHandbook"]), 1)          # shared origin ACROSS the two graphs
    assert Federation(use_lineage=False).n_eff(["r", "c1", "c2"]) == 3       # the naive count it replaces


def test_box_geometry():
    a, b = {"T": (0, 1), "p": (0, 5)}, {"T": (2, 3)}
    assert box_intersection(a, b) is None and box_intersection(a, {"T": (0.5, 9)}) == {"T": (0.5, 1), "p": (0, 5)}
    g = box_gap_point(a, b)
    assert g["T"] == 1.5 and g == box_gap_point(b, a)
    assert np.isclose(box_gap_point({"d": (1, 10)}, {"d": (1e4, 1e5)}, frozenset({"d"}))["d"], np.sqrt(10 * 1e4))   # decades → geometric


def test_curated_conf_boundary_and_inferred_links_stay_separate():
    f = _fed()
    kinds = {(p.kind, round(p.point["grain_size_nm"], 1)) for p in f.stress_points()}
    assert ("BOUNDARY", 14.1) in kinds and ("CONTRADICTION", 15.8) in kinds       # geometric: grain size is declared log
    links, _ = f.inferred_links()
    assert links and all(l.status == "INFERRED" and f.claims[l.legs[0]]["graph"] != f.claims[l.legs[1]]["graph"] for l in links)
    assert ("cooling rate", "yield strength") not in {c["pair"] for c in f.claims.values()}
    signs = {(l.sign, l.validity["grain_size_nm"]) for l in links}
    assert (1, (25.0, 100000.0)) in signs and (-1, (2.0, 8.0)) in signs      # the inferred relation inherits the regime split
    ranked = f.next_experiments()
    assert ranked[0].pair == ("grain size", "yield strength") and ranked[0].evpi >= 0.45   # open points first
    assert any(p.kind == "INFERRED" and p.pair == ("cooling rate", "yield strength") for p in ranked)
    assert all(8 <= p.point["grain_size_nm"] <= 25 for p in ranked if p.kind in ("BOUNDARY", "CONTRADICTION"))
    assert all(0 <= p.extent <= 1 for p in ranked)
    touching = [p for p in ranked if p.kind == "CONTRADICTION" and p.point["grain_size_nm"] == 25.0]
    assert touching and touching[0].value == 0                              # boxes that only touch decide nothing


def test_inferred_link_rejected_when_validity_boxes_are_disjoint():
    f = Federation()
    f.add_graph({"graph_id": "A", "claims": [{"id": "1", "subject": "s", "object": "x", "sign": 1, "validity": {"T": [0, 1]}, "evidence": ["a"]}]})
    f.add_graph({"graph_id": "B", "claims": [{"id": "1", "subject": "x", "object": "o", "sign": 1, "validity": {"T": [2, 3]}, "evidence": ["b"]}]})
    links, rejected = f.inferred_links()
    assert not links and rejected[0]["legs"] == ("A:1", "B:1")


# ---------------------------------------------------------------- replay policy
def test_episode_features_do_not_see_the_future():
    pytest.importorskip("sklearn")
    from graph_engine.replay_policy import build_episode
    rng = np.random.default_rng(0); n = 600
    month = np.sort(rng.integers(0, 60, n)); edges = []
    for i in range(1, n):
        for j in rng.choice(i, min(i, 4), replace=False): edges.append((i, int(j)))
    edges = np.array(edges)
    a = build_episode(edges, month, cutoff=36, horizon=12, n_rows=200, n_communities=3)
    keep = month[edges[:, 0]] <= 36
    fut = edges[~keep].copy(); fut[:, 1] = rng.permutation(fut[:, 1]) % np.maximum(fut[:, 0], 1)   # rewrite the future
    b = build_episode(np.vstack([edges[keep], fut]), month, cutoff=36, horizon=12, n_rows=200, n_communities=3)
    assert np.array_equal(a.pairs, b.pairs) and np.allclose(a.X, b.X)
    assert not np.array_equal(a.y, b.y)


# ---------------------------------------------------------------- regime posterior (declared: ≤ 1 transition; collision family kept)
def _rp(claims, **kw):
    from graph_engine.regime_posterior import RegimePosterior
    rp = RegimePosterior(0.0, 1.0, **kw)
    for a, b, s in claims:
        rp.add_claim(a, b, s)
    return rp


def test_closure_decides_the_span_between_same_sign_claims_without_a_probe():
    rp = _rp([(0.1, 0.2, 1), (0.7, 0.8, 1)])
    assert rp.p_plus(0.45) > 0.8                                             # the pointwise vote says 0.5 here
    f = Federation()
    f.add_graph({"graph_id": "T", "claims": [{"id": str(k), "subject": "s", "object": "o", "sign": 1, "validity": {"T": [a, b]}, "evidence": [f"r{k}"]}
                                             for k, (a, b) in enumerate([(0.1, 0.2), (0.7, 0.8)])]})
    assert f.belief(("s", "o"), {"T": 0.45})[0] == 0.5


def test_probe_placement_one_sign_then_gap():
    from graph_engine.regime_posterior import RegimePosterior
    strong = RegimePosterior(0.0, 1.0); strong.add_claim(0.4, 0.5, 1, 5.0)
    x1, g1 = strong.best_probe(); strong.add_probe(x1, 1); x2, g2 = strong.best_probe()
    assert 0.75 < x1 < 1.0 and x2 < 0.4 and g1 > g2 > 0                      # deep into the larger unknown span, then the other side
    xq, _ = _rp([(0.0, 0.2, 1), (0.8, 1.0, -1)]).best_probe()
    assert 0.35 < xq < 0.65                                                  # a gap between opposite signs is halved


def test_linear_potential_gives_zero_value_when_no_single_answer_flips_a_decision():
    from graph_engine.regime_posterior import RegimePosterior
    lin = RegimePosterior(0.0, 1.0, potential="error"); lin.add_claim(0.0, 1.0, 1, 8.0)
    ent = RegimePosterior(0.0, 1.0); ent.add_claim(0.0, 1.0, 1, 8.0)
    assert abs(lin.best_probe()[1]) < 1e-9 and ent.best_probe()[1] > 1e-4


def test_order_contradiction_between_disjoint_boxes():
    calm, torn = _rp([(0.0, 0.1, 1), (0.9, 1.0, 1)]), _rp([(0.0, 0.1, 1), (0.45, 0.55, -1), (0.9, 1.0, 1)])
    assert torn.expected_error() > 1.3 * calm.expected_error()               # + − + cannot all be right under one transition
    assert torn.collision()["p_two_transitions"] > 1.3 * calm.collision()["p_two_transitions"]   # measured 1.44×: weak, passive
    assert not torn.collision()["flag"]                                      # one claim is cheaper to doubt than the assumption


def test_collision_needs_probes_and_is_then_declared():
    rp = _rp([(0.0, 0.1, 1), (0.9, 1.0, 1)])
    for x, s in [(0.5, -1), (0.5, -1), (0.45, -1), (0.55, -1), (0.2, 1), (0.8, 1)]:
        rp.add_probe(x, s)
    c = rp.collision()
    assert c["flag"] and c["bayes_factor_two_vs_one"] > 19
    one = _rp([(0.0, 0.1, 1), (0.9, 1.0, -1)])
    for x, s in [(0.5, -1), (0.3, 1), (0.4, 1), (0.45, -1)]:
        one.add_probe(x, s)
    assert not one.collision()["flag"] and one.collision()["p_two_transitions"] < 0.2
    assert _rp([(0.0, 0.1, 1), (0.9, 1.0, 1)], p_two=0.0).collision()["p_two_transitions"] == 0.0   # assumed away = can never be seen


def test_greedy_probe_is_within_the_adaptive_submodular_bound():
    """Near-noiseless answers, two probes: greedy expected-error drop ≥ (1 − 1/e) × the best adaptive two-probe plan."""
    import copy
    base = _rp([(0.0, 0.05, 1), (0.3, 0.35, 1), (0.95, 1.0, -1)], p_two=0.0, potential="error")
    e0 = base.expected_error(); rel = 0.999

    def after(rp, x):                                   # [(P(outcome), posterior)]
        out = []
        for s in (1, -1):
            p = rp.p_plus(x); pout = p * rel + (1 - p) * (1 - rel) if s > 0 else (1 - p) * rel + p * (1 - rel)
            q = copy.deepcopy(rp); q.add_probe(x, s, rel); out.append((pout, q))
        return out
    xs = list(np.linspace(0.02, 0.98, 25))
    greedy = sum(p1 * sum(p2 * r2.expected_error() for p2, r2 in after(r1, r1.best_probe(rel)[0])) for p1, r1 in after(base, base.best_probe(rel)[0]))
    best = min(sum(p1 * min(sum(p2 * r2.expected_error() for p2, r2 in after(r1, x2)) for x2 in xs) for p1, r1 in after(base, x1)) for x1 in xs)
    assert (e0 - greedy) >= (1 - 1 / np.e) * (e0 - best) - 1e-9


# ---------------------------------------------------------------- graph interface (federation through shared concepts only)
def test_cross_graph_resistance_from_interfaces_equals_the_glued_graph():
    import scipy.sparse as sp
    from graph_engine.graph_interface import GraphInterface, cross_resistance
    rng = np.random.default_rng(3); nA, nB, nS = 60, 50, 6

    def lap(n):
        e = [(i, i + 1) for i in range(n - 1)] + [tuple(x) for x in rng.integers(0, n, (3 * n, 2)) if x[0] != x[1]]
        e = np.unique(np.sort(np.array(e), 1), axis=0)
        return laplacian_from_edges(n, e, rng.uniform(0.5, 2.0, len(e)))[0]
    LA, LB = lap(nA), lap(nB); sA, sB = np.arange(nS), np.arange(nS)          # first nS nodes of each are the shared concepts
    A, B = GraphInterface.export(LA, sA), GraphInterface.export(LB, sB)
    assert np.allclose(A.H.sum(1), 1) and (A.H > -1e-12).all()                 # harmonic measure: a point on the simplex
    # glued graph: shared 0..nS-1, then A interior, then B interior
    n = nS + (nA - nS) + (nB - nS); G = np.zeros((n, n))
    ia = np.r_[np.arange(nS), nS + np.arange(nA - nS)]; ib = np.r_[np.arange(nS), nS + (nA - nS) + np.arange(nB - nS)]
    G[np.ix_(ia, ia)] += LA.toarray(); G[np.ix_(ib, ib)] += LB.toarray()
    Gp = np.linalg.pinv(G); a = rng.integers(0, nA - nS, 40); b = rng.integers(0, nB - nS, 40)
    ga, gb = nS + a, nS + (nA - nS) + b
    exact = Gp[ga, ga] + Gp[gb, gb] - 2 * Gp[ga, gb]
    assert np.allclose(cross_resistance(A, B, a, b), exact, rtol=1e-9, atol=1e-10)
    assert A.nbytes() + B.nbytes() < 0.25 * 8 * n * n                          # what crosses the boundary vs the dense glued L⁺


def test_interfaces_nest_a_node_is_a_collapsed_subgraph():
    """Reducing onto S1 and then onto S2 ⊂ S1 equals reducing onto S2 directly (quotient property of Schur
    complements). Opening a node into its subgraph, or collapsing it again, leaves the rest of the graph unchanged."""
    from graph_engine.graph_interface import GraphInterface
    n, e = _graph(80, seed=5); L = laplacian_from_edges(n, e)[0]
    S1, S2 = np.arange(20), np.arange(6)
    once = GraphInterface.export(L, S2).S
    twice = GraphInterface.export(GraphInterface.export(L, S1).S, S2).S
    assert np.allclose(once, twice, atol=1e-9)


def test_probe_value_is_squared_travel_in_fisher_coordinates():
    H = lambda p: -(p * np.log2(p) + (1 - p) * np.log2(1 - p)); th = lambda p: 2 * np.arcsin(np.sqrt(p))
    def both(p, rel):
        po = p * rel + (1 - p) * (1 - rel); q1, q0 = p * rel / po, p * (1 - rel) / (1 - po)
        return H(p) - (po * H(q1) + (1 - po) * H(q0)), (po * (th(q1) - th(p)) ** 2 + (1 - po) * (th(q0) - th(p)) ** 2) / (2 * np.log(2))
    for p in (0.5, 0.7, 0.9):
        a, b = both(p, 0.6); assert abs(a / b - 1) < 0.01                      # weak answers: the two are the same number
    a, b = both(0.5, 0.95); assert 0.75 < a / b < 0.85                         # a near-certain answer: second order no longer enough
    m = lambda q: min(q, 1 - q); p, rel = 0.9, 0.75; po = p * rel + (1 - p) * (1 - rel)
    assert abs(m(p) - (po * m(p * rel / po) + (1 - po) * m(p * (1 - rel) / (1 - po)))) < 1e-12   # linear potential: exactly 0


def test_grow_span_adds_the_planted_interaction_and_improves_held_out_ranking():
    pytest.importorskip("sklearn")
    from graph_engine.replay_policy import ReplayPolicy, grow_span, average_precision, FEATURES
    rng = np.random.default_rng(0); n = 6000; X = np.abs(rng.standard_normal((n, len(FEATURES)))) * 3
    Z = np.log1p(X); Z = (Z - Z.mean(0)) / Z.std(0)
    y = rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 0.8 * Z[:, 0] + 1.6 * Z[:, 3] * Z[:, 8])))
    tr, te = slice(0, 4000), slice(4000, n)
    base = ReplayPolicy().fit(X[tr], y[tr]); grown, picked = grow_span(X[tr], y[tr], rounds=3)
    assert (3, 8) in picked[:2]
    assert average_precision(y[te], grown.score(X[te])) > average_precision(y[te], base.score(X[te])) + 0.05


def test_independence_rule_agrees_with_the_existing_paper_pipeline():
    """paper_graph/pipeline.confidence_from_legs counts distinct lineage keys. claim_federation.n_eff generalizes it to a
    lineage tree; with one root per source the two must give the same number."""
    sys.path.insert(0, str(REPO_ROOT / "src" / "graph_engine" / "paper_graph"))
    import pipeline as pp
    legs = [(0.35, "external:lab-A"), (0.35, "external:lab-B"), (0.35, "external:lab-A"), (0.35, "external:lab-C"), (0.35, "external:lab-B")]
    _, n_existing = pp.confidence_from_legs(0.2, legs)
    f = Federation(); f.add_graph({"graph_id": "T", "claims": [], "sources": [{"id": f"paper{k}", "derives_from": [lin]} for k, (_, lin) in enumerate(legs)]})
    assert n_existing == 3.0 and np.isclose(f.n_eff([f"paper{k}" for k in range(len(legs))]), n_existing)
    # what the tree adds: a review that rests on two labs is not a third origin
    f.add_graph({"graph_id": "U", "claims": [], "sources": [{"id": "review", "derives_from": ["external:lab-A", "external:lab-B"]}]})
    assert np.isclose(f.n_eff(["paper0", "paper1", "review"]), 2.0)


def test_lineage_information_is_kish_times_one_report_and_matches_neff_form():
    """Places the new count inside the repository's N_eff families (neff_form.py): variance reduction, not rank."""
    from graph_engine.claim_federation import lineage_information
    from graph_engine.neff_form import neff_form
    from graph_engine.lens_pooling import effective_lenses, participation_ratio
    for K in (2, 3, 6):
        # each report rests on its own root and on one root shared by all: error correlation 0.5, report variance 0.5
        info = lineage_information([{f"own{k}", "shared"} for k in range(K)])
        kish = neff_form(K, 0.5, goal="variance_reduction")["n_eff"]
        assert np.isclose(info, kish / 0.5, atol=1e-3)
    C = np.full((4, 4), 0.3); np.fill_diagonal(C, 1.0)
    assert np.isclose(effective_lenses(C), neff_form(4, 0.3, "variance_reduction")["kish"], atol=1e-3)
    assert np.isclose(participation_ratio(C), neff_form(4, 0.3)["participation_ratio"], atol=1e-3)


def test_three_estimators_of_a_diagonal_inverse_agree():
    from graph_engine.graph_hole_engine.kernel import resolvent_leverage
    from graph_engine.graph_interface import _diag_inv_hutchinson
    import scipy.sparse as sp, scipy.sparse.linalg as spla
    n, e = _graph(); L = laplacian_from_edges(n, e)[0]
    exact = resolvent_leverage(L.toarray(), np.eye(n))
    assert np.median(np.abs(ResistanceSketch.build(n, e, k=128, seed=0).refined_hole_field(L) - exact) / exact) < 0.05
    A = L[10:, 10:].tocsc(); d = np.diag(np.linalg.inv(A.toarray()))
    assert np.median(np.abs(_diag_inv_hutchinson(spla.splu(A), A.shape[0], k=2048) - d) / d) < 0.08


def test_fixed_hypothesis_space_probe_value_is_what_is_realized_and_never_negative():
    """Review finding: when probes added cells, the expected potential could rise and best_probe mispredicted."""
    import copy
    for claims, pot in [([(0.343, 0.393, 1), (0.015, 0.075, 1), (0.257, 0.337, 1)], "error"), ([(0.4, 0.5, 1)], "entropy"), ([(0.0, 0.2, 1), (0.8, 1.0, -1)], "entropy")]:
        rp = _rp(claims, potential=pot); e0 = rp.potential_value()
        for x in np.linspace(0.02, 0.98, 13):
            p = rp.p_plus(x); rel = 0.95; after = 0.0
            for s, po in ((1, p * rel + (1 - p) * (1 - rel)), (-1, (1 - p) * rel + p * (1 - rel))):
                q = copy.deepcopy(rp); q.add_probe(x, s, rel); after += po * q.potential_value()
            assert e0 - after > -1e-9                                         # Jensen: a probe cannot raise it in expectation
        xb, predicted = rp.best_probe(0.95); p = rp.p_plus(xb); after = 0.0
        for s, po in ((1, p * 0.95 + (1 - p) * 0.05), (-1, (1 - p) * 0.95 + p * 0.05)):
            q = copy.deepcopy(rp); q.add_probe(xb, s, 0.95); after += po * q.potential_value()
        assert np.isclose(predicted, e0 - after, atol=1e-9)                   # predicted value == realized expectation


def test_guards_found_in_review():
    from graph_engine.lens_pooling import effective_lenses
    from graph_engine.graph_interface import GraphInterface
    assert effective_lenses(np.array([[1.0, -1.0], [-1.0, 1.0]])) == float("inf")
    n, e = _graph(60, seed=2); L = laplacian_from_edges(n, e)[0].tolil(); L[5, 5] += 2.0
    with pytest.raises(ValueError):
        ResistanceSketch.from_laplacian(L.tocsr())
    L0 = laplacian_from_edges(n, e)[0]; exact = GraphInterface.export(L0, np.arange(6)).g
    assert np.allclose(exact, np.diag(np.linalg.inv(L0.toarray()[6:, 6:])), atol=1e-9)             # exact by default
    est = GraphInterface.export(L0, np.arange(6), estimate_g_with=4096).g
    assert 0.0 < np.median(np.abs(est - exact) / exact) < 0.1                                       # an estimate only on request


def test_model_check_probe_targets_the_gap_the_value_rule_ignores():
    """Two + claims at both ends: the sign potential is nearly settled by closure, but a − in between would mean two
    transitions. The model-check probe goes between the claims and its realized information about the family is the
    expectation it predicted (exact Bayes on the fixed partition)."""
    from graph_engine.regime_posterior import RegimePosterior
    rp = RegimePosterior(0.0, 1.0); rp.add_claim(0.05, 0.25, 1, 3); rp.add_claim(0.75, 0.95, 1, 3)
    x, gain = rp.model_check_probe(0.95)
    assert 0.25 < x < 0.75 and gain > 0
    h = lambda p: -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    before = h(rp.collision()["p_two_transitions"]); realized = 0.0
    for sg in (1, -1):
        cells, _, F, post = rp._with_probes(); c = min(int(np.searchsorted(cells[:, 1], x, side="left")), len(cells) - 1)
        f = F[:, c]; like = (f if sg > 0 else 1 - f) * 0.95 + (1 - (f if sg > 0 else 1 - f)) * 0.05; pout = float(post @ like)
        trial = RegimePosterior(0.0, 1.0); trial.claims = list(rp.claims); trial.add_probe(x, sg, 0.95)
        realized += pout * h(trial.collision()["p_two_transitions"])
    assert abs((before - realized) - gain) < 1e-9


def test_per_claim_reliability_changes_the_reading():
    from graph_engine.regime_posterior import RegimePosterior
    a, b = RegimePosterior(0.0, 1.0), RegimePosterior(0.0, 1.0)
    a.add_claim(0.2, 0.4, 1, 1, reliability=0.55); b.add_claim(0.2, 0.4, 1, 1, reliability=0.95)
    assert a.p_plus(0.3) < b.p_plus(0.3)
    try:
        a.add_claim(0.2, 0.4, 1, 1, reliability=0.3); assert False
    except ValueError:
        pass


def test_majority_claim_model_is_less_sure_at_the_far_end_of_a_wide_box():
    """A + claim over [0, 0.9]: read pointwise, every point is r-accurate; read as a majority report, a transition inside the
    box is still compatible with the report, so P₊ at the far end is lower (the e21 over-confidence)."""
    from graph_engine.regime_posterior import RegimePosterior
    a, b = RegimePosterior(0.0, 1.0), RegimePosterior(0.0, 1.0, claim_model="majority")
    a.add_claim(0.0, 0.9, 1, 3); b.add_claim(0.0, 0.9, 1, 3)
    assert b.p_plus(0.85) < a.p_plus(0.85) and b.p_plus(0.1) > 0.7


def test_model_check_pair_straddles_the_gap_and_predicts_its_own_realized_drop():
    """Two + claims at the ends of a wide gap: one probe rarely settles the family, so the pair is bought together.
    The two points land on opposite sides of the middle (or both inside the gap), and the expected drop over the four
    joint outcomes is exactly what the realized posteriors give (exact Bayes on the fixed partition)."""
    from graph_engine.regime_posterior import RegimePosterior
    rp = RegimePosterior(0.0, 1.0); rp.add_claim(0.05, 0.25, 1, 3); rp.add_claim(0.75, 0.95, 1, 3)
    (x1, x2), gain = rp.model_check_pair(0.95)
    assert x1 < x2 and gain > 0
    assert (x1 < 0.5 < x2) or (0.25 < x1 and x2 < 0.75)
    assert gain >= rp.model_check_probe(0.95)[1] - 1e-12                 # two probes are worth at least one
    h = lambda p: 0.0 if p <= 0 or p >= 1 else -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    before = h(rp.collision()["p_two_transitions"]); realized = 0.0
    for s1 in (1, -1):
        for s2 in (1, -1):
            trial = RegimePosterior(0.0, 1.0); trial.claims = list(rp.claims)
            cells, _, F, post = trial._with_probes()
            c1 = min(int(np.searchsorted(cells[:, 1], x1, side="left")), len(cells) - 1)
            c2 = min(int(np.searchsorted(cells[:, 1], x2, side="left")), len(cells) - 1)
            f1 = F[:, c1] if s1 > 0 else 1 - F[:, c1]; l1 = f1 * 0.95 + (1 - f1) * 0.05
            p1 = float(post @ l1); q1 = post * l1 / p1
            f2 = F[:, c2] if s2 > 0 else 1 - F[:, c2]; l2 = f2 * 0.95 + (1 - f2) * 0.05
            p2 = float(q1 @ l2)
            trial.add_probe(x1, s1, 0.95); trial.add_probe(x2, s2, 0.95)
            realized += p1 * p2 * h(trial.collision()["p_two_transitions"])
    assert abs((before - realized) - gain) < 1e-9


def test_edge_leverage_sums_to_n_minus_one_and_sparsifier_keeps_resistances():
    """Kirchhoff: Σ_e w_e R_e = n − 1 exactly (a spanning tree has n − 1 edges). Spielman–Srivastava: sampling edges by leverage
    keeps every resistance within a few percent with a fraction of the edges."""
    from graph_engine.resistance_sketch import ResistanceSketch, edge_leverage, sparsify, laplacian_from_edges
    rng = np.random.default_rng(0); n = 300
    edges = np.unique(np.sort(rng.integers(0, n, (2400, 2)), 1), axis=0); edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.vstack([edges, np.stack([np.arange(n - 1), np.arange(1, n)], 1)])       # a path keeps it connected
    L = laplacian_from_edges(n, edges)[0].toarray(); Lp = np.linalg.pinv(L)
    exact = np.array([Lp[i, i] + Lp[j, j] - 2 * Lp[i, j] for i, j in edges])
    assert abs(exact.sum() - (n - 1)) < 1e-6
    sk = ResistanceSketch.build(n, edges, k=256, seed=0)
    assert abs(edge_leverage(sk, edges).sum() - (n - 1)) / (n - 1) < 0.1
    e2, w2, _ = sparsify(n, edges, q=16 * n, k=256, seed=1)
    L2 = laplacian_from_edges(n, e2, w2)[0].toarray(); Lp2 = np.linalg.pinv(L2)
    pairs = rng.integers(0, n, (200, 2)); pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    r1 = np.array([Lp[i, i] + Lp[j, j] - 2 * Lp[i, j] for i, j in pairs]); r2 = np.array([Lp2[i, i] + Lp2[j, j] - 2 * Lp2[i, j] for i, j in pairs])
    assert len(e2) < 0.9 * len(edges) and np.median(np.abs(r2 / r1 - 1)) < 0.15, (len(e2), np.median(np.abs(r2 / r1 - 1)))


def test_e_optimal_probe_is_flat_in_the_exact_max_min_and_decides_on_the_tied_weakest_set():
    """regime_posterior.weakest_direction_probe, the E-optimal (max-min) rule, against a hand recomputation.

    Three things are pinned. (1) The discrimination matrix is exactly the Jeffreys divergence between the two-outcome
    emissions of the hypotheses that carry mass, summed over the probes made: the chosen x and the reported value
    agree to 1e-12 with an independent recomputation from the probe list. (2) The EXACT one-step max-min gain is ZERO on this hypothesis space, for
    every candidate, as long as two hypotheses that carry mass agree on every cell probed so far: their
    discrimination is 0 and one probe lifts only the pairs that straddle its own cell, so min D does not move. Here it is
    flat on the first 3 of 5 steps and decisive on the last 2. That is the structural degeneracy of greedy E-optimality,
    and it is why the rule falls back to the mean lift of the tied weakest set — which is what the returned value is,
    verified against the recomputation. (3) The mixed rule's λ is fixed on the first step so that the entropy term and
    the discrimination term have exactly equal scale there: λ · max-gain = max expected entropy drop = best_probe's.
    """
    from graph_engine.regime_posterior import RegimePosterior

    def jeffreys(Fk, c, r, wt=1.0):
        a = np.clip(r * Fk[:, c] + (1 - r) * (1 - Fk[:, c]), 1e-12, 1 - 1e-12)
        la, lb = np.log(a), np.log(1 - a)
        return wt * (a[:, None] - a[None, :]) * ((la[:, None] - la[None, :]) - (lb[:, None] - lb[None, :]))

    def dmat(rp, eps=0.01, n_max=30):
        cells, _w, F, post = rp._with_probes()
        keep = np.where(post >= eps)[0]
        if len(keep) > n_max:
            keep = keep[np.argsort(post[keep])[::-1][:n_max]]
        if len(keep) < 2:
            keep = np.argsort(post)[::-1][: min(n_max, len(post))]
        Fk = F[keep]
        D = np.zeros((len(keep), len(keep)))
        for x, _sg, r, wt in rp.probes:
            c = min(int(np.searchsorted(cells[:, 1], x, side="left")), len(cells) - 1)
            D += jeffreys(Fk, c, r, wt)
        return cells, F, post, keep, D

    rp = RegimePosterior(0.0, 1.0, n_grid=8)
    rp.add_claim(0.05, 0.25, 1, 3); rp.add_claim(0.75, 0.95, 1, 3)
    flat = []
    for _ in range(5):
        cells, F, post, keep, D = dmat(rp)
        iu = np.triu_indices(len(keep), 1)
        now_min = float(D[iu].min())
        assert now_min == 0.0                                   # (2) an unseparated pair always exists early
        xs = list(cells.mean(1)) + [rp.lo, rp.hi]
        cs = list(range(len(cells))) + [0, len(cells) - 1]
        weak = D[iu] <= now_min + 1e-9
        exact = [float((D[iu] + jeffreys(F[keep], c, 0.95)[iu]).min()) - now_min for c in cs]
        ties = [float(jeffreys(F[keep], c, 0.95)[iu][weak].mean()) for c in cs]
        flat.append(max(exact) <= 1e-12)                        # is max-min flat for every candidate?
        x, gain = rp.weakest_direction_probe(0.95)
        score = ties if flat[-1] else exact
        k = int(np.argmax(score))
        assert abs(x - xs[k]) < 1e-12 and abs(gain - score[k]) < 1e-12, (x, xs[k], gain, score[k])
        rp.add_probe(float(x), 1 if 0.3 < x < 0.7 else -1, 0.95)
    assert flat[0] and sum(flat) >= 3, flat                     # flat while the kept pairs are mostly unseparated; the
    assert not all(flat), flat                                  # exact rule only becomes decisive once they are not

    d_star = rp.best_probe(0.95)[1]                             # (3) equal scale on the first mixed step
    g_star = rp.weakest_direction_probe(0.95)[1]
    twin = RegimePosterior(0.0, 1.0, n_grid=8); twin.claims = list(rp.claims); twin.probes = list(rp.probes)
    twin.weakest_direction_probe(0.95, mix=True)
    assert abs(twin._eopt_lambda * g_star - d_star) < 1e-12, (twin._eopt_lambda, g_star, d_star)


def test_total_value_probe_is_at_least_the_sign_value_and_exact():
    from graph_engine.regime_posterior import RegimePosterior
    rp = RegimePosterior(0.0, 1.0); rp.add_claim(0.05, 0.25, 1, 3); rp.add_claim(0.75, 0.95, 1, 3)
    xs, gs = rp.best_probe(0.95); xm, gm = rp.model_check_probe(0.95); xt, gt = rp.total_value_probe(0.95)
    assert gt >= max(gs, gm) - 1e-12 and gt <= gs + gm + 1e-12


def test_bundle_value_prices_a_pair_exactly_and_between_its_bounds():
    """regime_posterior.bundle_value on the two-claims-far-apart case: the PAIR priced in the same bits as the single.

    Three things. (1) The pair is worth at least the best single (a second probe cannot have negative value: the
    posterior is a martingale on the fixed partition and both terms of the total potential are concave in it). (2) It is
    worth at most the two best singles plus the family term that is on the table at all — H(P(two transitions)) can only
    be driven to 0, so no bundle can gain more than the sum of the one-step sign values plus the whole family entropy.
    Together these bracket the increasing returns e33 exhibited: the pair is allowed to beat the sum of the parts inside
    the family term (which is where a second transition lives) and nowhere else. (3) The predicted gain IS the realized
    expectation of the total potential over the four joint outcomes, probe 1 then probe 2 with the posterior updated in
    between — exact Bayes on the fixed partition, to 1e-9, the same pin as model_check_pair."""
    from graph_engine.regime_posterior import RegimePosterior
    lam = 1.0
    rp = RegimePosterior(0.0, 1.0); rp.add_claim(0.05, 0.25, 1, 3); rp.add_claim(0.75, 0.95, 1, 3)
    bv = rp.bundle_value(0.95, k=2, lam=lam, top=12)
    (xs, gs, ns) = bv["single"]; ((x1, x2), gp, np_) = bv["pair"]
    assert (ns, np_) == (1.0, 2.0) and x1 < x2 and gp > 0
    assert abs(gs - rp.total_value_probe(0.95, lam=lam)[1]) < 1e-12          # the single IS the one-step rule
    assert gp >= gs - 1e-12                                                  # (1)
    h = lambda p: 0.0 if p <= 0 or p >= 1 else -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    fam_now = h(rp.collision()["p_two_transitions"])
    g_two = sum(g for _x, g in bv["top_singles"][:2])
    assert gp <= g_two + lam * fam_now + 1e-12, (gp, g_two, fam_now)         # (2)

    cells, w, F, post = rp._with_probes()                                    # (3) realized expectation, four outcomes
    two = np.zeros(len(post)); two[rp._n_one:] = 1.0
    cell_of = lambda x: min(int(np.searchsorted(cells[:, 1], x, side="left")), len(cells) - 1)
    c1, c2 = cell_of(x1), cell_of(x2)
    now = float(rp._u(post @ F) @ w) + lam * h(float(post @ two))
    assert abs(now - bv["now"]) < 1e-12
    realized = 0.0
    for s1 in (1, -1):
        f1 = F[:, c1] if s1 > 0 else 1 - F[:, c1]; l1 = f1 * 0.95 + (1 - f1) * 0.05
        p1 = float(post @ l1); q1 = post * l1 / p1
        for s2 in (1, -1):
            f2 = F[:, c2] if s2 > 0 else 1 - F[:, c2]; l2 = f2 * 0.95 + (1 - f2) * 0.05
            p2 = float(q1 @ l2); q2 = q1 * l2 / p2
            trial = RegimePosterior(0.0, 1.0); trial.claims = list(rp.claims)
            trial.add_probe(x1, s1, 0.95); trial.add_probe(x2, s2, 0.95)
            _c, _w, _F, qq = trial._with_probes()
            assert np.max(np.abs(qq - q2)) < 1e-12                           # the two-step update is the joint update
            realized += p1 * p2 * (float(trial._u(qq @ _F) @ _w) + lam * h(float(qq @ two)))
    assert abs((now - realized) - gp) < 1e-9, (now - realized, gp)
