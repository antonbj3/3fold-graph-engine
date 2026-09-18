"""decision_cert: certificates relative to a DECISION, and the measurement that can flip one.

D1 a pair whose transition point is wide open still certifies a decision on a box it cannot reach — and the same
pair fails a decision whose box straddles the transition (decision-relativity, not a property of the node);
D2 p_flip equals a brute-force enumeration over the posterior on a small partition; D3 the attribution lands on the
pair the decision reads even when another pair has far more entropy to give; D4 a decision certified far inside its
α has only zero-value flips; D5 the same candidate's P(flip) is relative to α; D6 the margin certificate is
Φ(−(z − z_req)) exactly and D7 its flip probability matches a Monte-Carlo of the pre-posterior; D8 a chain decision
is the complement of the product under declared independence, and its attribution is exact over the two answers;
D9 malformed decisions raise; D10 the certificate declares its own scope and never a global threshold.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "graph_engine" / "tools"))
from graph_engine.decision_cert import Decision, certify, flip_actions, flip_attribution  # noqa: E402
from graph_engine.margin_net import MarginNet  # noqa: E402
from graph_engine.next_actions import EngineState, Instrument  # noqa: E402
from graph_engine.regime_posterior import RegimePosterior  # noqa: E402

PAIR = ("grain_size", "yield_strength")
OTHER = ("dose", "response")


def _post(n_grid=8, p_two=0.0):
    p = RegimePosterior(0.0, 1.0, reliability=0.9, p_flip=0.5, p_two=p_two, n_grid=n_grid)
    p.add_claim(0.0, 0.25, +1, n_eff=6.0)
    return p


def _state(**kw):
    base = dict(regimes={PAIR: _post()}, regime_variables={PAIR: "grain_size_nm"},
                probe_instruments=[Instrument("judge", 1.0, 0.9)])
    base.update(kw)
    return EngineState(**base)


# -- D1, D10 ----------------------------------------------------------------------------------------
def test_decision_relative_certificate_on_a_pair_with_a_wide_open_transition():
    """The pair's transition point is unknown over most of the domain (the posterior is spread over cells), yet
    the decision on the claimed box certifies. The SAME pair fails a decision on the rest of the domain."""
    st = _state()
    post = st.regimes[PAIR]
    assert post.potential_value() > 0.3                     # the pair is far from settled globally

    near = certify(Decision("sign_near", "sign", alpha=0.1, pair=PAIR, box=(0.0, 0.25), sign=+1), st)
    far = certify(Decision("sign_far", "sign", alpha=0.1, pair=PAIR, box=(0.5, 1.0), sign=+1), st)
    assert near["holds"] and near["p_flip"] < 0.1
    assert not far["holds"] and far["p_flip"] > 0.1         # same pair, same state, opposite verdicts
    assert near["kind"] == far["kind"] == "decision-relative"
    for c in (near, far):
        assert "says nothing about the target globally" in c["scope"]
        assert c["alpha"] == 0.1 and "hypotheses" in c["credible_set_checked"]


# -- D2 ---------------------------------------------------------------------------------------------
def test_p_flip_equals_brute_force_enumeration_over_the_posterior():
    """Rebuild the hypothesis space by hand on a small partition, evaluate the predicate by dense sampling,
    and sum the posterior mass of the hypotheses that fail it."""
    post = _post(n_grid=8, p_two=0.0)
    post.add_probe(0.6, -1, reliability=0.9)
    st = _state(regimes={PAIR: post})
    a, b = 0.0, 0.25                                        # box aligned to cell edges (0.125 grid, claim edge 0.25)
    cert = certify(Decision("d", "sign", alpha=0.05, pair=PAIR, box=(a, b), sign=+1), st)

    cells, _w, _F, p = post._with_probes()
    thr = cells.mean(1)
    xs = np.linspace(a, b, 501)
    sign_funcs = [lambda x: np.ones_like(x), lambda x: -np.ones_like(x)]
    sign_funcs += [(lambda x, t=t: np.where(x < t, 1.0, -1.0)) for t in thr]          # + → −
    sign_funcs += [(lambda x, t=t: np.where(x < t, -1.0, 1.0)) for t in thr]          # − → +
    assert len(sign_funcs) == len(p)
    brute = sum(float(w) for w, f in zip(p, sign_funcs) if not np.all(f(xs) > 0))
    assert cert["p_flip"] == pytest.approx(brute, abs=1e-9)
    assert cert["n_hypotheses"] == len(p)


# -- D3 ---------------------------------------------------------------------------------------------
def test_attribution_is_on_the_pair_the_decision_reads_not_the_pair_with_the_most_entropy():
    empty = RegimePosterior(0.0, 1.0, reliability=0.9, p_two=0.0, n_grid=8)          # no claims: maximal entropy
    st = _state(regimes={PAIR: _post(), OTHER: empty},
                regime_variables={PAIR: "grain_size_nm", OTHER: "dose_mg"})
    assert empty.best_probe(0.9)[1] > st.regimes[PAIR].best_probe(0.9)[1]            # OTHER offers more bits

    d = Decision("sign_near", "sign", alpha=0.1, pair=PAIR, box=(0.0, 0.25), sign=+1)
    cert = certify(d, st)
    d = Decision("sign_near", "sign", alpha=cert["p_flip"] * 1.05, pair=PAIR, box=(0.0, 0.25), sign=+1)
    rows = flip_attribution(d, st)
    assert rows and {r["target"] for r in rows} == {PAIR}                            # never the other pair
    assert rows[0]["p_flip_status"] > 0.0 and rows[0]["flipping_outcome"] == {"sign": -1}
    # the flipping probe need not sit inside the box: a − answer far to the right also lifts the "− everywhere"
    # and "− → + late" hypotheses, which are exactly the ones that fail the predicate ON the box.
    assert 0.0 < rows[0]["p_flip_status"] <= 1.0
    assert [r["value_per_cost"] for r in rows] == sorted((r["value_per_cost"] for r in rows), reverse=True)

    from graph_engine.next_actions import next_actions
    acts = next_actions(st, k=6, decisions=[d])
    top_bits = acts[0]
    assert top_bits.target == OTHER and top_bits.value_unit == "bits"                # the entropy channel disagrees
    flips = [a for a in acts.other if a.kind == "flip"]
    assert flips and all(a.target == PAIR and a.value_unit == "p_flip" for a in flips)
    assert all("sign_near" in a.how for a in flips)
    assert all(a.value_unit != "p_flip" for a in acts)                               # never inside the bits ranking


# -- D4 ---------------------------------------------------------------------------------------------
def test_a_decision_certified_far_inside_alpha_has_only_zero_value_flips():
    post = RegimePosterior(0.0, 1.0, reliability=0.95, p_flip=0.5, p_two=0.0, n_grid=8)
    post.add_claim(0.0, 0.25, +1, n_eff=40.0)                                        # overwhelming evidence
    st = _state(regimes={PAIR: post})
    d = Decision("settled", "sign", alpha=0.5, pair=PAIR, box=(0.0, 0.25), sign=+1)
    cert = certify(d, st)
    assert cert["holds"] and cert["p_flip"] < 0.5 * 1e-2
    rows = flip_attribution(d, st)
    assert rows and all(r["p_flip_status"] == 0.0 and r["value_per_cost"] == 0.0 for r in rows)
    assert flip_actions(st, [d]) == []                                               # nothing worth buying


# -- D5 ---------------------------------------------------------------------------------------------
def test_flip_probability_is_relative_to_alpha():
    st = _state()
    base = Decision("d", "sign", alpha=0.1, pair=PAIR, box=(0.0, 0.25), sign=+1)
    p0 = certify(base, st)["p_flip"]
    tight = Decision("d", "sign", alpha=p0 * 1.05, pair=PAIR, box=(0.0, 0.25), sign=+1)
    loose = Decision("d", "sign", alpha=0.999, pair=PAIR, box=(0.0, 0.25), sign=+1)
    assert certify(tight, st)["holds"] and certify(loose, st)["holds"]
    best_tight = max(r["p_flip_status"] for r in flip_attribution(tight, st))
    best_loose = max(r["p_flip_status"] for r in flip_attribution(loose, st))
    assert best_tight > 0.0 and best_loose == 0.0          # same beliefs, same probes, different decision level


# -- D6 ---------------------------------------------------------------------------------------------
def _net():
    net = MarginNet(default_sigma=0.1)
    net.add_sources([{"id": "S1"}])
    net.add_edge("e.beam", ["capacity", "demand"], [{"margin": 0.05, "sigma": 0.1, "sources": ["S1"]}])
    return net


def test_margin_certificate_is_the_exact_gaussian_tail():
    net = _net()
    st = _state(margins=net)
    est = net.estimate("e.beam")
    for z_req in (0.0, 1.0, 2.0):
        c = certify(Decision("m", "margin", alpha=0.05, edge="e.beam", z_req=z_req), st)
        assert c["p_flip"] == pytest.approx(float(norm.cdf(-(est.z - z_req))), abs=1e-12)
        assert c["holds"] == (c["p_flip"] <= 0.05) and c["channel"] == "margin_net"


# -- D7 ---------------------------------------------------------------------------------------------
def test_margin_flip_probability_matches_a_monte_carlo():
    net = _net()
    st = _state(margins=net, measure_instrument=Instrument("cell", 2.0, 1.0, sigma=0.05))
    est = net.estimate("e.beam")
    for alpha, z_req in ((0.3, 0.0), (0.05, 0.0), (0.6, 1.0)):
        d = Decision("m", "margin", alpha=alpha, edge="e.beam", z_req=z_req)
        row = flip_attribution(d, st)[0]
        now = certify(d, st)["holds"]

        rng = np.random.default_rng(7)
        sn = 0.05
        s1 = 1.0 / np.sqrt(1.0 / est.s ** 2 + 1.0 / sn ** 2)
        y = est.m + rng.standard_normal(400_000) * np.sqrt(est.s ** 2 + sn ** 2)      # predictive next report
        mu = (est.m / est.s ** 2 + y / sn ** 2) * s1 ** 2                             # its posterior mean
        holds_after = norm.cdf(z_req - mu / s1) <= alpha
        assert row["p_flip_status"] == pytest.approx(float((holds_after != now).mean()), abs=3e-3)
        assert row["cost"] == 2.0 and row["value_per_cost"] == pytest.approx(row["p_flip_status"] / 2.0)


# -- D8 ---------------------------------------------------------------------------------------------
def test_chain_decision_under_declared_independence_and_its_attribution():
    st = _state(p_holds={"n1": 0.8, "n2": 0.9}, instruments={"n1": [(1.0, 0.9, "judge")], "n2": [(4.0, 0.9, "cell")]})
    d = Decision("goal_G", "chain", alpha=0.25, nodes=("n1", "n2"), tau=0.9)
    c = certify(d, st)
    assert c["p_flip"] == pytest.approx(1 - 0.8 * 0.9) and not c["holds"]
    assert c["weakest_node"] == "n1" and "independence" in c["credible_set_checked"]

    rows = {r["target"]: r for r in flip_attribution(d, st)}
    q1 = 0.8 * 0.9 + 0.2 * 0.1                                                        # P(instrument says "holds")
    assert rows["n1"]["p_flip_status"] == pytest.approx(q1)                            # only the "holds" answer flips
    assert rows["n1"]["flipping_outcome"] == {"answer": "holds"}
    q2 = 0.9 * 0.9 + 0.1 * 0.1
    assert rows["n2"]["p_flip_status"] == pytest.approx(q2)
    assert rows["n1"]["value_per_cost"] > rows["n2"]["value_per_cost"]                 # per cost, the cheap node wins
    acts = flip_actions(st, [d])
    assert acts and acts[0].target == "n1" and "goal_G" in acts[0].how and acts[0].meta["apply_as"] == "work_node"


# -- D9 ---------------------------------------------------------------------------------------------
def test_malformed_decisions_and_missing_channels_raise():
    st = _state()
    with pytest.raises(ValueError, match="unknown decision kind"):
        Decision("d", "vibe")
    with pytest.raises(ValueError, match="alpha"):
        Decision("d", "sign", alpha=0.0, pair=PAIR)
    with pytest.raises(ValueError, match="needs a pair"):
        Decision("d", "sign")
    with pytest.raises(ValueError, match="sign must be"):
        Decision("d", "sign", pair=PAIR, sign=0)
    with pytest.raises(ValueError, match="b ≥ a"):
        Decision("d", "sign", pair=PAIR, box=(0.5, 0.1))
    with pytest.raises(ValueError, match="needs an edge"):
        Decision("d", "margin")
    with pytest.raises(ValueError, match="needs at least one node"):
        Decision("d", "chain")
    with pytest.raises(ValueError, match="no regime posterior"):
        certify(Decision("d", "sign", pair=OTHER), st)
    with pytest.raises(ValueError, match="no margin edge"):
        certify(Decision("d", "margin", edge="nope"), st)
    with pytest.raises(ValueError, match="no belief for node"):
        certify(Decision("d", "chain", nodes=("ghost",)), st)


# -- D10 --------------------------------------------------------------------------------------------
def test_a_flip_action_is_not_applied_through_apply():
    """The flip action names the underlying channel action; the answer is routed back through that one."""
    from graph_engine.next_actions import apply
    st = _state()
    d = Decision("d", "sign", alpha=certify(Decision("d", "sign", alpha=0.1, pair=PAIR, box=(0.0, 0.25)), st)["p_flip"] * 1.05,
                 pair=PAIR, box=(0.0, 0.25), sign=+1)
    act = flip_actions(st, [d])[0]
    assert act.kind == "flip" and act.meta["apply_as"] == "probe"
    with pytest.raises(ValueError, match="unknown action kind"):
        apply(st, act, {"sign": -1})
