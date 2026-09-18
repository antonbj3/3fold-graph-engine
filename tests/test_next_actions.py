"""next_actions: one list across the channels, and the route back.

N1 every present channel contributes; N2 the bits are the channels' own numbers; N3 ordering is by value per cost;
N4 the guard share reserves model_check slots (and none when there is nothing to guard); N5 unlock work_nodes and the
other non-bits channels never enter the bits ranking; N6 `how` names the target and the instrument; N7 apply routes
each outcome kind; N8 realized = predicted in expectation on the regime channel (martingale-exact); N9 wrong-shaped
outcomes raise.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "graph_engine" / "tools"))
from graph_engine.margin_net import MarginNet  # noqa: E402
from graph_engine.next_actions import Action, EngineState, Instrument, apply, next_actions  # noqa: E402
from graph_engine.precision_form import Candidate, PrecisionForm  # noqa: E402
from graph_engine.profile import EngineProfile  # noqa: E402
from graph_engine.regime_posterior import RegimePosterior  # noqa: E402

PAIR_A = ("grain_size", "yield_strength")
PAIR_B = ("dose", "response")


def _regimes():
    a = RegimePosterior(0.0, 1.0, p_two=0.05)
    a.add_claim(0.0, 0.3, +1, n_eff=2.0)
    a.add_claim(0.6, 1.0, -1, n_eff=1.0)
    b = RegimePosterior(0.0, 1.0, p_two=0.05)
    b.add_claim(0.1, 0.5, +1, n_eff=1.0)
    return {PAIR_A: a, PAIR_B: b}


def _margins():
    net = MarginNet(default_sigma=0.1)
    net.add_sources([{"id": "S1"}, {"id": "S2"}])
    net.add_edge("e.beam", ["capacity", "demand"],
                 [{"margin": 0.05, "sigma": 0.1, "sources": ["S1"]},
                  {"margin": 0.12, "sigma": 0.1, "sources": ["S2"]}], weight=2.0, cost=4.0)
    net.add_edge("e.joint", ["capacity2", "demand2"],
                 [{"margin": 0.30, "sigma": 0.2, "sources": ["S1"]}], weight=1.0, cost=1.0)
    return net


def _form():
    form = PrecisionForm.zeros(4).add_laplacian([(0, 1), (1, 2), (2, 3)])
    form.J = form.J + 1e-3 * np.eye(4)
    cands = [Candidate("c.link02", np.array([1.0, 0.0, -1.0, 0.0]), sigma=0.5, cost=1.0, kind="link"),
             Candidate("c.node3", np.array([0.0, 0.0, 0.0, 1.0]), sigma=1.0, cost=2.0, kind="point")]
    return form, cands


UNLOCK = {"nodes": [
    {"id": "G", "type": "GOAL", "status": "OPEN", "depends_on": ["n1", "n2"], "cost": 1},
    {"id": "n1", "type": "NODE", "status": "OPEN", "depends_on": [], "risk": "HIGH", "cost": 3},
    {"id": "n2", "type": "NODE", "status": "OPEN", "depends_on": [], "risk": "LOW", "cost": 2},
]}

REPORTS = [{"margin": 0.4, "sigma": 0.05, "attributes": {"site": "A", "age": 30}},
           {"margin": 0.42, "sigma": 0.05, "attributes": {"site": "A", "age": 35}},
           {"margin": -0.3, "sigma": 0.05, "attributes": {"site": "B", "age": 70}},
           {"margin": -0.28, "sigma": 0.05, "attributes": {"site": "B", "age": 75}}]


def _state(**kw):
    form, cands = _form()
    base = dict(regimes=_regimes(), regime_variables={PAIR_A: "grain_size_nm", PAIR_B: "dose_mg"},
                margins=_margins(), form=form, candidates=cands,
                probe_instruments=[Instrument("judge", 1.0, 0.8), Instrument("cell", 4.0, 0.99)])
    base.update(kw)
    return EngineState(**base)


# -- N1, N2, N3 -------------------------------------------------------------------------------------
def test_every_channel_contributes_and_bits_are_the_channels_own_numbers():
    st = _state()
    acts = next_actions(st, k=20)
    assert {a.source_channel for a in acts} == {"regime_posterior", "margin_net", "precision_form"}
    assert {a.kind for a in acts} >= {"probe", "measure", "throw", "model_check"}

    # regime: the action's bits are best_probe's own expected drop for the chosen instrument
    probe = next(a for a in acts if a.kind == "probe" and a.target == PAIR_A)
    x, gain = st.regimes[PAIR_A].best_probe(probe.instrument.reliability)
    assert probe.value_bits == pytest.approx(gain) and probe.meta["x"] == pytest.approx(x)
    assert probe.value_per_cost == pytest.approx(gain / probe.instrument.cost)

    # margin_net: measurement_value is bits × weight / cost, the action carries the bits
    meas = next(a for a in acts if a.kind == "measure" and a.target == "e.beam")
    v = st.margins.measurement_value("e.beam", st.margins.default_sigma)
    assert meas.value_bits == pytest.approx(v * 4.0 / 2.0) and meas.cost == 4.0

    # precision_form: rank's own bits and bits per cost
    form, cands = _form()
    ranked = dict((cid, (bits, per)) for cid, bits, per, _ in form.rank(cands))
    for a in acts:
        if a.source_channel == "precision_form":
            assert a.value_bits == pytest.approx(ranked[a.target][0])
            assert a.value_per_cost == pytest.approx(ranked[a.target][1])


def test_ranking_is_by_value_per_cost_and_k_is_respected():
    acts = next_actions(_state(), k=3)
    assert len(acts) == 3
    assert [a.value_per_cost for a in acts] == sorted((a.value_per_cost for a in acts), reverse=True)
    assert all(a.value_unit == "bits" for a in acts)


# -- N4 ---------------------------------------------------------------------------------------------
def test_guard_share_reserves_model_check_slots():
    st = _state()
    all_acts = next_actions(st, k=50)
    guards = [a for a in all_acts if a.kind == "model_check"]
    assert guards, "the two-transition family is on, so model_check candidates exist"
    # without a guard share the cheap high-value channels can crowd them out
    none = next_actions(st, k=4, guard_share=0.0)
    with_guard = next_actions(st, k=4, guard_share=0.5)
    assert len([a for a in with_guard if a.kind == "model_check"]) == 2
    assert len([a for a in with_guard if a.kind == "model_check"]) >= len([a for a in none if a.kind == "model_check"])
    # the share comes from the profile when it is not given
    prof = EngineProfile(guard_share=0.5).validate()
    assert len([a for a in next_actions(st, profile=prof, k=4) if a.kind == "model_check"]) == 2


def test_guard_share_reserves_nothing_when_the_collision_family_is_off():
    post = RegimePosterior(0.0, 1.0, p_two=0.0)
    post.add_claim(0.0, 0.4, +1, n_eff=1.0)
    acts = next_actions(EngineState(regimes={PAIR_A: post}), k=5, guard_share=0.5)
    assert acts and not [a for a in acts if a.kind == "model_check"]


# -- N5 ---------------------------------------------------------------------------------------------
def test_non_bits_channels_never_enter_the_bits_ranking():
    st = _state(unlock_graph=UNLOCK, p_holds={"n1": 0.5, "n2": 0.9},
                reports_by_edge={"e.beam": REPORTS})
    acts = next_actions(st, k=20)
    assert not [a for a in acts if a.kind in ("work_node", "declare_variable")]
    assert all(a.value_unit == "bits" for a in acts)
    work = [a for a in acts.other if a.kind == "work_node"]
    decl = [a for a in acts.other if a.kind == "declare_variable"]
    # every OPEN node unlock_value calls actionable, the goal included (its own output, unfiltered)
    assert {a.target for a in work} == {"G", "n1", "n2"} and all(a.value_unit == "priority" for a in work)
    assert decl and all(a.value_unit == "chi2_drop" for a in decl)
    assert any(a.target == "site" for a in decl)             # the planted split
    assert acts.by_kind("work_node") == work


# -- N6 ---------------------------------------------------------------------------------------------
def test_how_names_the_target_and_the_instrument():
    st = _state(unlock_graph=UNLOCK, p_holds={"n1": 0.5, "n2": 0.9}, reports_by_edge={"e.beam": REPORTS})
    acts = next_actions(st, k=20)
    for a in list(acts) + acts.other:
        assert a.how == a.how.strip() and "\n" not in a.how
        if a.instrument is not None:
            assert f"'{a.instrument.name}'" in a.how
        assert "return " in a.how and "{" in a.how.split("return ", 1)[1]
    probe = next(a for a in acts if a.kind == "probe" and a.target == PAIR_A)
    assert "grain_size_nm" in probe.how and "yield_strength" in probe.how and "sign" in probe.how
    assert f"{probe.meta['x']:g}" in probe.how
    assert "'e.beam'" in next(a for a in acts if a.target == "e.beam").how
    assert "'n1'" in next(a for a in acts.other if a.target == "n1").how


# -- N7 ---------------------------------------------------------------------------------------------
def test_apply_routes_a_probe_and_returns_a_ledger_row():
    st = _state()
    act = next(a for a in next_actions(st, k=20) if a.kind == "probe" and a.target == PAIR_A)
    before = st.regimes[PAIR_A].potential_value()
    row = apply(st, act, {"sign": -1})
    assert len(st.regimes[PAIR_A].probes) == 1
    assert st.regimes[PAIR_A].probes[0][:2] == (act.meta["x"], -1)
    assert row["consumers"] == [str(PAIR_A)] and row["evidence"] == {"sign": -1} and row["supersedes"] == []
    assert row["value_predicted"] == pytest.approx(act.value_bits)
    assert row["value_realized"] == pytest.approx(before - st.regimes[PAIR_A].potential_value())
    assert row["value_unit"] == "bits" and row["instrument"] == act.instrument.name


def test_apply_routes_measure_throw_and_declaration():
    st = _state(reports_by_edge={"e.beam": REPORTS})
    acts = next_actions(st, k=20)

    meas = next(a for a in acts if a.target == "e.beam")
    row = apply(st, meas, {"margin": 0.09, "sigma": 0.1, "sources": ["S3"]})
    assert len(st.margins.edges["e.beam"]["reports"]) == 3
    assert row["value_realized"] is not None and row["consumers"] == ["e.beam"]

    throw = next(a for a in acts if a.kind == "throw" and a.target == "c.link02")
    before = np.linalg.slogdet(st.form.cov())[1]
    row = apply(st, throw, {"confirmed_links": [(0, 2, 1.0)]})
    after = np.linalg.slogdet(st.form.cov())[1]
    assert row["value_realized"] == pytest.approx((before - after) / (2 * np.log(2)), rel=1e-6)

    decl = next(a for a in acts.other if a.kind == "declare_variable" and a.target == "site")
    row = apply(st, decl, {"attribute": "site"})
    assert row["value_realized"] is None and "site" in row["boxes"]
    assert set(row["boxes"]["site"]) == {"left", "right"}


# -- N8 -------------------------------------------------------------------------------------------
def test_regime_realized_drop_equals_predicted_in_expectation():
    """The posterior is a martingale: the predicted drop IS the expectation of the realized drop over the two
    outcomes, with the probe's own predictive weights. Checked for the sign potential and for the guard probe."""
    for kind, potential in (("probe", lambda p: p.potential_value()),
                            ("model_check", lambda p: __import__("graph_engine.next_actions", fromlist=["x"])._family_entropy(p))):
        st = _state()
        act = next(a for a in next_actions(st, k=20, guard_share=0.5) if a.kind == kind and a.target == PAIR_A)
        post = st.regimes[PAIR_A]
        x, r = act.meta["x"], act.instrument.reliability
        pp = post.p_plus(x)
        q_plus = pp * r + (1 - pp) * (1 - r)
        realized = {}
        for sign in (+1, -1):
            s = _state()
            realized[sign] = apply(s, act, {"sign": sign})["value_realized"]
        expected = q_plus * realized[1] + (1 - q_plus) * realized[-1]
        assert expected == pytest.approx(act.value_bits, abs=1e-9)
        assert potential(post) >= 0.0


# -- N9 -------------------------------------------------------------------------------------------
def test_wrong_shaped_outcomes_raise():
    st = _state(unlock_graph=UNLOCK, p_holds={"n1": 0.5, "n2": 0.9}, reports_by_edge={"e.beam": REPORTS})
    acts = next_actions(st, k=20)
    probe = next(a for a in acts if a.kind == "probe")
    meas = next(a for a in acts if a.kind == "measure")
    throw = next(a for a in acts if a.kind == "throw")
    with pytest.raises(ValueError, match="must carry"):
        apply(st, probe, {"margin": 0.1, "sigma": 0.1, "sources": ["S1"]})
    with pytest.raises(ValueError, match="sign must be"):
        apply(st, probe, {"sign": 0})
    with pytest.raises(ValueError, match="must be a dict"):
        apply(st, probe, -1)
    with pytest.raises(ValueError, match="must carry"):
        apply(st, meas, {"margin": 0.1, "sigma": 0.1})
    with pytest.raises(ValueError, match="must be"):
        apply(st, throw, {"confirmed_links": [(0, 2)]})
    with pytest.raises(ValueError, match="work_node"):
        apply(st, next(a for a in acts.other if a.kind == "work_node"), {"sign": 1})
    with pytest.raises(ValueError, match="unknown action kind"):
        apply(st, Action("guess", PAIR_A, None, 1.0, 1.0, 1.0, "nowhere", "how"), {"sign": 1})
    with pytest.raises(ValueError):
        next_actions(st, k=0)
    with pytest.raises(ValueError, match="guard_share"):
        next_actions(st, k=3, guard_share=1.5)


# -- N10 ------------------------------------------------------------------------------------------
def test_decisions_add_flip_actions_named_after_the_decision():
    """`decisions=` adds "flip" actions ranked by P(the outcome flips that decision's certificate) / cost. They
    carry value_unit "p_flip", so they stay in `.other` and out of the bits ranking."""
    from graph_engine.decision_cert import Decision, certify

    st = _state()
    d_pair = Decision("sign_low", "sign", alpha=0.5, pair=PAIR_A, box=(0.0, 0.3), sign=+1)
    d_edge = Decision("beam_margin", "margin", alpha=0.05, edge="e.beam", z_req=1.0)
    plain = next_actions(st, k=8)
    acts = next_actions(st, k=8, decisions=[d_pair, d_edge])

    assert [a.id for a in acts] == [a.id for a in plain]              # the bits ranking is untouched
    flips = [a for a in acts.other if a.kind == "flip"]
    assert flips and acts.by_kind("flip") == flips
    assert all(a.value_unit == "p_flip" and 0.0 < a.value_bits <= 1.0 for a in flips)
    assert all(a.value_per_cost == pytest.approx(a.value_bits / a.cost) for a in flips)
    assert [a.value_per_cost for a in flips] == sorted((a.value_per_cost for a in flips), reverse=True)
    assert {a.meta["decision"] for a in flips} <= {"sign_low", "beam_margin"}
    for a in flips:                                                    # `how` names the decision and its status
        assert a.meta["decision"] in a.how and "p_flip" in a.how
        assert a.meta["holds"] == certify(d_pair if a.meta["decision"] == "sign_low" else d_edge, st)["holds"]
    assert {a.target for a in flips} <= {PAIR_A, "e.beam"}             # only what the decisions read
    assert next_actions(st, k=8, decisions=None).other == plain.other
