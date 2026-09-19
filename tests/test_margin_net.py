"""margin_net: every statement in the module docstring that can be checked numerically is checked here."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from graph_engine.margin_net import MarginNet  # noqa: E402
from graph_engine.claim_federation import Federation  # noqa: E402


def _net(**kw):
    n = MarginNet(**kw)
    n.add_sources([{"id": "r"}, {"id": "c1", "derives_from": ["r"]}, {"id": "c2", "derives_from": ["c1"]}])
    return n


def test_copies_do_not_narrow_the_estimate_independent_origins_do():
    n = _net()
    n.add_edge("one", ["a", "b"], [{"margin": 0.2, "sigma": 0.1, "sources": ["r"]}])
    n.add_edge("copies", ["a", "b"], [{"margin": 0.2, "sigma": 0.1, "sources": [x]} for x in ("r", "c1", "c2")])
    n.add_edge("indep", ["a", "b"], [{"margin": 0.2, "sigma": 0.1, "sources": [x]} for x in ("x", "y", "z", "w")])
    assert np.isclose(n.estimate("copies").s, n.estimate("one").s) and np.isclose(n.estimate("copies").n_eff, 1)
    assert np.isclose(n.estimate("indep").s, 0.1 / 2) and np.isclose(n.estimate("indep").n_eff, 4)
    f = Federation(); f._parents.update(n._parents)
    assert np.isclose(n.estimate("copies").n_eff, f.n_eff(["r", "c1", "c2"]))           # one rule for signs and margins


def test_estimate_is_calibrated_with_shared_origins():
    """z-scores of (m̂ − truth)/s are standard normal when reports share roots as declared."""
    rng = np.random.default_rng(0); zs = []
    for _ in range(3000):
        eps = rng.standard_normal(3); sig = np.array([0.1, 0.2, 0.15, 0.1])
        M = np.array([[1, 0, 0], [1, 0, 0], [.5, .5, 0], [0, 0, 1.0]]); m = 0.3 + sig * (M @ eps)
        n = MarginNet(); n.add_sources([{"id": "t", "derives_from": ["a", "b"]}])
        n.add_edge("e", ["u", "v"], [{"margin": m[0], "sigma": .1, "sources": ["a"]}, {"margin": m[1], "sigma": .2, "sources": ["a"]},
                                      {"margin": m[2], "sigma": .15, "sources": ["t"]}, {"margin": m[3], "sigma": .1, "sources": ["c"]}])
        est = n.estimate("e"); zs.append((est.m - 0.3) / est.s)
    zs = np.array(zs)
    assert abs(zs.mean()) < 0.06 and 0.94 < zs.std() < 1.06


def test_stress_is_distance_in_uncertainty_units_not_raw_margin():
    n = MarginNet()
    n.add_edge("small_but_sure", ["a", "b"], [{"margin": 0.05, "sigma": 0.01, "sources": [s]} for s in "pq"])
    n.add_edge("large_but_vague", ["a", "c"], [{"margin": 0.30, "sigma": 0.40, "sources": ["u"]}])
    order = [x.id for x in n.estimates()]
    assert order[0] == "large_but_vague" and n.estimate("small_but_sure").p_violated < 1e-6 < n.estimate("large_but_vague").p_violated


def test_disagreement_is_contradiction_or_regime_boundary():
    n = MarginNet()
    n.add_edge("conf", ["a", "b"], [{"margin": 0.4, "sigma": 0.05, "sources": ["p"], "validity": {"T": [0, 50]}},
                                    {"margin": -0.3, "sigma": 0.05, "sources": ["q"], "validity": {"T": [20, 80]}}])
    n.add_edge("regime", ["a", "b"], [{"margin": 0.4, "sigma": 0.05, "sources": ["p"], "validity": {"T": [0, 20]}},
                                      {"margin": -0.3, "sigma": 0.05, "sources": ["q"], "validity": {"T": [60, 80]}}])
    n.add_edge("agree", ["a", "b"], [{"margin": 0.40, "sigma": 0.05, "sources": ["p"]}, {"margin": 0.43, "sigma": 0.05, "sources": ["q"]}])
    assert n.estimate("conf").kind == "CONTRADICTION" and n.estimate("regime").kind == "REGIME-BOUNDARY" and n.estimate("agree").kind == "OK"
    rng = np.random.default_rng(1); ps = []
    for _ in range(2000):                                   # Q is χ²(dof): p-values uniform when the reports DO agree
        m = MarginNet(); m.add_edge("e", ["a", "b"], [{"margin": 0.2 + 0.1 * rng.standard_normal(), "sigma": 0.1, "sources": [f"s{k}"]} for k in range(4)])
        ps.append(m.estimate("e").p_agree)
    assert abs(np.mean(np.array(ps) < 0.05) - 0.05) < 0.015


def test_measurement_value_peaks_at_the_boundary_and_matches_simulation():
    vals = []
    for m0 in (-0.6, -0.2, 0.0, 0.2, 0.6):
        n = MarginNet(); n.add_edge("e", ["a", "b"], [{"margin": m0, "sigma": 0.2, "sources": ["p"]}]); vals.append(n.measurement_value("e", 0.2))
    assert vals[2] == max(vals) and vals[0] < vals[1] < vals[2] > vals[3] > vals[4] and abs(vals[1] - vals[3]) < 1e-9
    n = MarginNet(); n.add_edge("e", ["a", "b"], [{"margin": 0.1, "sigma": 0.2, "sources": ["p"]}])
    rng = np.random.default_rng(2); truth = 0.1 + 0.2 * rng.standard_normal(40000); y = truth + 0.2 * rng.standard_normal(40000)
    from scipy.stats import norm
    post_m, post_s = (0.1 / 0.04 + y / 0.04) / (2 / 0.04), (2 / 0.04) ** -0.5
    h = lambda p: -(p * np.log2(p) + (1 - p) * np.log2(1 - p)); p1 = np.clip(norm.cdf(-post_m / post_s), 1e-12, 1 - 1e-12)
    assert abs(n.measurement_value("e", 0.2) - (h(norm.cdf(-0.5)) - h(p1).mean())) < 0.01


def test_shared_source_raises_both_all_hold_and_many_fail_together():
    shared, split = MarginNet(), MarginNet()
    for k in range(10):
        shared.add_edge(f"e{k}", ["a", f"v{k}"], [{"margin": 0.12, "sigma": 0.1, "sources": ["one_file"]}])
        split.add_edge(f"e{k}", ["a", f"v{k}"], [{"margin": 0.12, "sigma": 0.1, "sources": [f"file{k}"]}])
    a, b = shared.failure_counts(seed=3), split.failure_counts(seed=3)
    assert abs(a["with_shared_sources"]["mean"] - b["with_shared_sources"]["mean"]) < 0.03          # same expected number of violations
    assert a["with_shared_sources"]["p_all_hold"] > b["with_shared_sources"]["p_all_hold"] + 0.3    # Slepian direction
    # and the far tail is fatter: with one file behind all ten, "half of them fail at once" is as likely as one failing
    assert a["with_shared_sources"]["p_at_least_half"] > 0.1 > 0.005 > b["with_shared_sources"]["p_at_least_half"]      # binomial(10, 0.115): 0.0031
    lev = shared.source_leverage()
    assert lev[0][0] == "one_file" and lev[0][2] == 10 and np.isclose(lev[0][1], 10.0)


def test_declared_copies_that_disagree_are_a_contradiction():
    """Review finding: Q is identically 0 for reports sharing all their roots, whatever their values."""
    n = _net()
    n.add_edge("torn", ["a", "b"], [{"margin": v, "sigma": 0.05, "sources": [s]} for v, s in ((-0.50, "r"), (0.50, "c1"), (0.52, "c2"))])
    n.add_edge("same", ["a", "b"], [{"margin": v, "sigma": 0.05, "sources": [s]} for v, s in ((0.50, "r"), (0.51, "c1"), (0.52, "c2"))])
    assert n.estimate("torn").kind == "CONTRADICTION" and n.estimate("same").kind == "OK"
    with pytest.raises(ValueError):
        m = MarginNet(); m.add_edge("z", ["a", "b"], [{"margin": 0.1, "sigma": 0.0, "sources": ["p"]}]); m.estimate("z")


def test_partially_shared_error_sits_between_independent_and_copy_and_is_not_a_copy():
    """Three reports from one collaboration with shares rho = 0.5: s is between the independent (1/√3) and the copy (1)
    case, they may differ without firing the copy check, and with no group declared N_eff equals lineage_information."""
    def net(sources, reports):
        n = MarginNet(default_sigma=0.1); n.add_sources(sources); n.add_edge("e", ["a", "b"], reports); return n.estimate("e")
    reps = [{"margin": 0.30, "sources": ["p1"]}, {"margin": 0.42, "sources": ["p2"]}, {"margin": 0.20, "sources": ["p3"]}]
    ind = net([{"id": p} for p in ("p1", "p2", "p3")], reps)
    shared = net([{"id": p, "shares": {"CDF": 0.5}} for p in ("p1", "p2", "p3")], reps)
    copies = net([{"id": "p1"}, {"id": "p2", "derives_from": ["p1"]}, {"id": "p3", "derives_from": ["p1"]}], reps)
    assert ind.s < shared.s < 0.1 <= copies.s + 1e-9
    assert 1.0 < shared.n_eff < 3.0 and abs(ind.n_eff - 3.0) < 1e-9
    assert shared.kind in ("OK", "STRESSED") and copies.kind == "CONTRADICTION"      # copies must agree; shared may differ
    assert shared.p_agree > 0.01


# ---------------------------------------------------------------------------------------------------
# the three lineage states as ONE object: a tree of error components  (e36)
# ---------------------------------------------------------------------------------------------------
from graph_engine.margin_net import tree_gls, tree_covariance, tree_from_shares  # noqa: E402
from graph_engine.claim_federation import lineage_information  # noqa: E402


def _pinv_gls(nodes, sig, m, tree):
    """the reference route: form Sigma = D T Tt D and go through the pseudo-inverse, exactly as MarginNet.estimate."""
    S = tree_covariance(nodes, sig, tree); Sp = np.linalg.pinv(S, rcond=1e-10)
    one = np.ones(len(m)); info = float(one @ Sp @ one); mh = float(one @ Sp @ m) / info
    q = max(float(m @ Sp @ m) - mh * mh * info, 0.0)
    off = (np.eye(len(m)) - S @ Sp) @ (m - mh)
    return mh, (1.0 / info) ** 0.5, info, q, off


def _random_tree(rng, n_reports, depth, copies=False):
    """a random tree of `depth` levels with shares drawn from a Dirichlet (they sum to 1 along every path)."""
    par = {"root": None}; layers = [["root"]]
    for d in range(depth):
        new = []
        for p in layers[-1]:
            for c in range(int(rng.integers(1, 3))):
                v = f"n{d}_{len(new)}"; par[v] = p; new.append(v)
        layers.append(new)
    sh = rng.dirichlet(np.ones(depth + 1))
    if copies:                                           # the deepest level carries no own variance -> exact copies
        sh[-2] += sh[-1]; sh[-1] = 0.0
    tree = [{"id": v, "parent": par[v], "share": float(sh[d])} for d, layer in enumerate(layers) for v in layer]
    nodes = [layers[-1][int(rng.integers(len(layers[-1])))] for _ in range(n_reports)]
    return tree, nodes


def test_tree_gls_equals_the_pinv_estimate_including_exact_copies():
    """O(n*depth) recursion == the pinv route to 1e-10 on random trees. Where Sigma is singular (exact copies) the
    recursion collapses each copy class to ONE observation weighted by sigma, with design coefficient
    (sum sigma)/(sum sigma^2) instead of 1 — that is what the Moore-Penrose inverse does there, not 1/sigma^2 weights."""
    rng = np.random.default_rng(0); worst = 0.0; n_singular = 0
    for trial in range(120):
        tree, nodes = _random_tree(rng, int(rng.integers(3, 12)), int(rng.integers(1, 4)), copies=bool(trial % 2))
        sig = rng.uniform(0.05, 0.5, len(nodes)); m = rng.normal(0.3, 0.2, len(nodes))
        r = tree_gls([{"margin": m[k], "sigma": sig[k], "node": nodes[k]} for k in range(len(nodes))], tree)
        a, b, info, q, off = _pinv_gls(nodes, sig, m, tree)
        n_singular += int(np.linalg.matrix_rank(tree_covariance(nodes, sig, tree), tol=1e-10) < len(nodes))
        worst = max(worst, abs(r["m"] - a), abs(r["s"] - b), abs(r["q"] - q) / max(1.0, abs(q)),
                    float(np.max(np.abs(np.array(r["off"]) - off))))
    assert worst < 1e-10 and n_singular > 20
    # the closed form for two exact copies with different sigma (1 is OUTSIDE range(Sigma))
    tree = [{"id": "o", "parent": None, "share": 1.0}, {"id": "a", "parent": "o", "share": 0.0},
            {"id": "b", "parent": "o", "share": 0.0}]
    r = tree_gls([{"margin": 0.4, "sigma": 0.1, "node": "a"}, {"margin": 0.6, "sigma": 0.2, "node": "b"}], tree)
    assert abs(r["m"] - (0.1 * 0.4 + 0.2 * 0.6) / 0.3) < 1e-12 and abs(r["s"] - (0.01 + 0.04) / 0.3) < 1e-12


def test_the_three_lineage_states_are_three_trees_and_marginnet_reads_them():
    """independent = own leaf theta 1; copy = theta 0 under one node; shares rho = ancestor rho + leaf 1-rho.
    MarginNet.add_sources takes the tree records and returns exactly what tree_gls and the old machinery return."""
    vals = [0.30, 0.42, 0.20]
    trees = {"independent": [{"id": f"p{k}", "parent": None, "share": 1.0} for k in range(3)],
             "copy": [{"id": "o", "parent": None, "share": 1.0}] + [{"id": f"p{k}", "parent": "o", "share": 0.0} for k in range(3)],
             "shared": [{"id": "CDF", "parent": None, "share": 0.5}] + [{"id": f"p{k}", "parent": "CDF", "share": 0.5} for k in range(3)]}
    out = {}
    for name, tree in trees.items():
        n = MarginNet(default_sigma=0.1); n.add_sources(tree)
        n.add_edge("e", ["a", "b"], [{"margin": v, "sources": [f"p{k}"]} for k, v in enumerate(vals)])
        est = n.estimate("e"); g = tree_gls([{"margin": v, "sigma": 0.1, "node": f"p{k}"} for k, v in enumerate(vals)], tree)
        assert abs(est.m - g["m"]) < 1e-12 and abs(est.s - g["s"]) < 1e-12 and abs(est.n_eff - g["n_eff"]) < 1e-12
        out[name] = est
    assert abs(out["independent"].s - 0.1 / 3 ** 0.5) < 1e-12 and abs(out["copy"].s - 0.1) < 1e-12
    assert out["independent"].s < out["shared"].s < out["copy"].s
    assert out["copy"].kind == "CONTRADICTION" and out["shared"].kind in ("OK", "STRESSED")
    # the same three states through the OLD records give the same numbers (the old contract is untouched)
    old = MarginNet(default_sigma=0.1)
    old.add_sources([{"id": "p0"}, {"id": "p1", "derives_from": ["p0"]}, {"id": "p2", "derives_from": ["p0"]}])
    old.add_edge("e", ["a", "b"], [{"margin": v, "sources": [f"p{k}"]} for k, v in enumerate(vals)])
    assert abs(old.estimate("e").s - out["copy"].s) < 1e-12
    sh = MarginNet(default_sigma=0.1); sh.add_sources([{"id": f"p{k}", "shares": {"CDF": 0.5}} for k in range(3)])
    sh.add_edge("e", ["a", "b"], [{"margin": v, "sources": [f"p{k}"]} for k, v in enumerate(vals)])
    assert abs(sh.estimate("e").s - out["shared"].s) < 1e-12 and abs(sh.estimate("e").n_eff - out["shared"].n_eff) < 1e-12


def test_tree_neff_is_lineage_information_for_copies_and_kish_for_one_shared_ancestor():
    """(a) copies only (theta_leaf in {0,1}) and equal sigma: N_eff = ||M+1||^2 = number of distinct origins.
       (b) one ancestor with rho over m reports: N_eff = m/(1+(m-1)rho) (Kish = neff_form's variance-reduction form)."""
    from graph_engine.neff_form import neff_form
    rng = np.random.default_rng(1)
    for origins, per in ((3, 2), (4, 1), (2, 5), (5, 3)):
        tree = [{"id": f"o{j}", "parent": None, "share": 1.0} for j in range(origins)]
        nodes = []
        for j in range(origins):
            for c in range(per):
                tree.append({"id": f"c{j}_{c}", "parent": f"o{j}", "share": 0.0}); nodes.append(f"c{j}_{c}")
        m = rng.normal(0.2, 0.05, len(nodes))
        r = tree_gls([{"margin": m[k], "sigma": 0.1, "node": nodes[k]} for k in range(len(nodes))], tree)
        li = lineage_information([frozenset([n.split("_")[0].replace("c", "o")]) for n in nodes])
        assert abs(r["n_eff"] - li) < 1e-10 and abs(r["n_eff"] - origins) < 1e-10
        assert abs(r["s"] - 0.1 / origins ** 0.5) < 1e-10
    for m_rep in (2, 3, 10, 50):
        for rho in (0.05, 0.2, 0.5, 0.9):
            tree, nodes = tree_from_shares(["g"] * m_rep, rho)
            r = tree_gls([{"margin": 0.3, "sigma": 0.2, "node": n} for n in nodes], tree)
            kish = m_rep / (1 + (m_rep - 1) * rho)
            assert abs(r["n_eff"] - kish) < 1e-9
            assert abs(r["n_eff"] - neff_form(m_rep, rho, goal="variance_reduction")["n_eff"]) < 1e-3
            assert abs(r["s"] - 0.2 * (kish ** -0.5)) < 1e-12


def test_rho_interpolates_the_two_old_states_continuously():
    """rho -> 0 is the independent case, rho -> 1 the copy case, and s is strictly increasing in between."""
    m_rep = 6; sig = 0.15; ss = []
    for rho in np.linspace(0.0, 1.0, 21):
        tree, nodes = tree_from_shares(["g"] * m_rep, float(rho))
        ss.append(tree_gls([{"margin": 0.3, "sigma": sig, "node": n} for n in nodes], tree)["s"])
    ss = np.array(ss)
    assert abs(ss[0] - sig / m_rep ** 0.5) < 1e-12 and abs(ss[-1] - sig) < 1e-12
    assert (np.diff(ss) > 0).all() and abs(ss[10] - sig * ((1 + 5 * 0.5) / 6) ** 0.5) < 1e-12


# -- units are not evidence ------------------------------------------------------------------------
#
# Measured on a five-channel cross-solver edge (one compiled reference implementation plus four
# configurations of a second solver, agreeing on a penetration requirement to ~1e-7 in the margin's own
# units): under an ABSOLUTE rank tolerance the edge reported Q = 3.77614e6 with dof = -1, p_agree = 1,
# "OK" — a negative dof is not a statistic, and the chi^2 disagreement test could not fire at all —
# while the same edge with every margin and sigma multiplied by 1e6 reported dof = 4, p_agree = 0,
# CONTRADICTION. The reconstruction below carries the measured margins and sigmas of that edge.

CUBE_MARGINS = (0.99768, 1 - 7.5405e-8 / 0.5, 1 - 1.892e-10 / 0.5, 1 - 2.71917e-9 / 0.5, 1 - 2.77556e-16 / 0.5)
CUBE_SIGMAS = (1.192e-6, 1.281e-7, 1.281e-7, 1.281e-7, 1.281e-7)
CUBE_CHANNELS = ("ref", "b_exact_gpu", "b_convex_gpu", "b_exact_cpu", "b_convex_cpu")


def _cross_solver_edge(scale):
    n = MarginNet()
    n.add_sources([{"id": c} for c in CUBE_CHANNELS])
    n.add_edge("penetration", ["a", "b"],
               [{"margin": m * scale, "sigma": s * scale, "sources": [c]}
                for c, m, s in zip(CUBE_CHANNELS, CUBE_MARGINS, CUBE_SIGMAS)])
    return n.estimate("penetration")


def test_estimate_is_invariant_to_the_units_of_the_margin():
    """A change of units is not a change of evidence: z, Q, dof, p_agree and kind are identical over twelve
    orders of magnitude of scale."""
    base = _cross_solver_edge(1.0)
    for scale in (1e-6, 1e-3, 1e3, 1e6):
        e = _cross_solver_edge(scale)
        assert (e.dof, e.kind) == (base.dof, base.kind), f"scale {scale:g}"
        assert e.z == pytest.approx(base.z, rel=1e-6) and e.q == pytest.approx(base.q, rel=1e-6)
        assert e.p_agree == pytest.approx(base.p_agree, abs=1e-12)
        assert e.m == pytest.approx(base.m * scale, rel=1e-9)
        assert e.s == pytest.approx(base.s * scale, rel=1e-6)


def test_a_disagreement_at_small_sigma_still_fires():
    """The measured edge: five channels whose spread is ~1e4 times their declared sigma. dof = 4 (five
    reports estimating one number), Q = 3.777e6, and the verdict is CONTRADICTION at every scale — not
    the "OK" an absolute rank tolerance produced when Sigma's entries fell to ~1e-14."""
    e = _cross_solver_edge(1.0)
    assert e.dof == 4
    assert e.q == pytest.approx(3.777e6, rel=1e-3)
    assert e.p_agree == 0.0 and e.kind == "CONTRADICTION"
    assert np.linalg.matrix_rank(np.diag([s ** 2 for s in CUBE_SIGMAS]), tol=1e-10) == 0   # the absolute read
