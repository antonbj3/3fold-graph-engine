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
