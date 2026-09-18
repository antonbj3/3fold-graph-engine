"""next_actions: one list across the channels, and the route back.

N1 every present channel contributes; N2 the bits are the channels' own numbers; N3 ordering is by value per cost;
N4 the guard share reserves model_check slots (and none when there is nothing to guard); N5 unlock work_nodes and the
other non-bits channels never enter the bits ranking; N6 `how` names the target and the instrument; N7 apply routes
each outcome kind; N8 realized = predicted in expectation on the regime channel (martingale-exact); N9 wrong-shaped
outcomes raise; N10 decisions add flip actions outside the bits ranking; N11 BUNDLES: a purchase priced at its own
depth (probe pair, sweep at weight 1/K, chain throw) ranks in the same bits-per-cost list with no reserved share,
its value is the exact joint one (additive across targets, not within one), apply routes the members in order and
realized = predicted in expectation over the joint outcomes, and a malformed bundle raises.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "graph_engine" / "tools"))
from graph_engine.margin_net import MarginNet  # noqa: E402
from graph_engine.next_actions import (Action, EngineState, Instrument, _joint_probe_gain, apply,  # noqa: E402
                                       next_actions)
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


# -- N11: bundles, priced at their own depth -------------------------------------------------------
from graph_engine.next_actions import (bundle_action, bundle_value_bits, chain_throw_bundle,  # noqa: E402
                                       probe_pair_bundle, sweep_bundle)


def _collision_state(p_two=0.3, reliability=0.9, n_eff=8.0):
    """A pair whose single-transition family already explains the claims: one strong claim over the whole domain.
    e33's regime — a single probe is worth almost nothing, a PAIR on both sides of a second transition is not."""
    post = RegimePosterior(0.0, 1.0, reliability=reliability, p_two=p_two)
    post.add_claim(0.0, 1.0, +1, n_eff=n_eff)
    return EngineState(regimes={PAIR_A: post}, probe_instruments=[Instrument("judge", 1.0, 0.9)])


def test_pair_bundle_outranks_two_singles_exactly_when_its_joint_value_per_cost_is_larger():
    """The bundle is in the SAME bits-per-cost list as the singles and wins a slot only by that number. Both cases
    are constructed: increasing returns (the closure-filled pair, where the pair wins) and the ordinary pair with an
    open transition (where a single probe wins, because two probes cost twice and add less than twice)."""
    for st, bundle_should_win in ((_collision_state(), True), (_state(pair_bundles=True), False)):
        acts = next_actions(st, k=20, guard_share=0.0)
        bundles = [a for a in acts if a.kind == "bundle"]
        singles = [a for a in acts if a.kind in ("probe", "model_check") and a.target == PAIR_A]
        assert bundles and singles
        b = next(a for a in bundles if a.target == str(PAIR_A))
        best_single = max(s.value_per_cost for s in singles)
        assert (b.value_per_cost > best_single) is bundle_should_win
        assert (acts[0] is b) is bundle_should_win                       # the one list is ordered by the one number
        assert [a.value_per_cost for a in acts] == sorted((a.value_per_cost for a in acts), reverse=True)
        # the bundle's number is the channel's own exact pair value, its cost the sum of the members' costs
        post, ins = st.regimes[PAIR_A], b.instrument
        assert b.value_bits == pytest.approx(post.bundle_value(ins.reliability, k=2)["pair"][1], rel=1e-9)
        assert b.cost == pytest.approx(sum(m.cost for m in b.meta["member_actions"]))
        assert len(b.meta["member_actions"]) == 2 and sorted(b.meta["x"]) == list(b.meta["x"])
        assert b.value_unit == "bits" and b.meta["exact"]


def test_bundles_take_no_reserved_share_and_can_be_switched_off():
    st = _collision_state()
    assert [a for a in next_actions(st, k=20, guard_share=0.0) if a.kind == "bundle"]
    st2 = _collision_state()
    st2.pair_bundles = False
    assert not [a for a in next_actions(st2, k=20) if a.kind == "bundle"]
    # a guard share still reserves model_check slots only; the bundle competes for the rest on its own number
    with_guard = next_actions(_collision_state(), k=4, guard_share=0.5)
    assert len([a for a in with_guard if a.kind == "model_check"]) >= 1
    assert [a for a in with_guard if a.kind == "bundle"]


def test_sweep_is_bounded_by_its_lineage():
    """A sweep is ONE computation cell answering K points: one lineage root, so each answer enters with weight 1/K
    (disagreement_field.model_probes). Its exact 2^K-outcome value is therefore at most the value of K INDEPENDENT
    probes at the same points, and at least the value of one of its own answers. Measured here: it is also BELOW a
    single full-weight probe — tempering the likelihood disperses the outcome law as well as the update."""
    st = _state()
    post = st.regimes[PAIR_A]
    cell = Instrument("sweep_cell", 3.0, 0.9)
    xs = [0.05, 0.35, 0.65, 0.95]
    sw = sweep_bundle(st, PAIR_A, xs, cell)
    indep, _t, _e = _joint_probe_gain(post, xs, [cell.reliability] * 4, [1.0] * 4)
    assert sw.meta["K"] == 4 and sw.meta["weight"] == pytest.approx(0.25) and sw.meta["exact"]
    assert all(m.meta["weight"] == pytest.approx(0.25) for m in sw.meta["member_actions"])
    assert sw.value_bits <= indep + 1e-12                                # lineage bound: one root, not K roots
    assert sw.value_bits >= max(sw.meta["members"]) - 1e-12              # at least one of its own answers
    assert sw.cost == pytest.approx(cell.cost)                           # the cell's own cost, not K probe costs
    assert sw.value_per_cost == pytest.approx(sw.value_bits / cell.cost)
    # K > 8: the sum of conditionals along sampled outcome paths, flagged as such and close to the exact number
    xs9 = [0.05 + 0.1 * i for i in range(9)]
    big = sweep_bundle(_state(), PAIR_A, xs9, cell, n_sample=1000)
    exact9, _t, ex = _joint_probe_gain(post, xs9, [cell.reliability] * 9, [1.0 / 9] * 9, max_exact=9)
    assert ex and big.meta["exact"] is False and big.meta["subsample"] == 1000
    assert big.value_bits == pytest.approx(exact9, rel=0.15)


def test_bundle_apply_realized_equals_predicted_in_expectation_over_the_joint_outcomes():
    """Exact over the FOUR outcomes of a probe pair: the price is an expectation over the joint outcome law, and
    `apply` routes the members in order and returns ONE ledger row with both numbers."""
    st = _collision_state()
    act = next(a for a in next_actions(st, k=20, guard_share=0.0) if a.kind == "bundle")
    law = act.meta["outcome_law"]
    assert len(law) == 4 and sum(p for _s, p in law) == pytest.approx(1.0)
    expected = 0.0
    for signs, prob in law:
        s = _collision_state()
        row = apply(s, act, {"outcomes": [{"sign": sg} for sg in signs]})
        assert len(s.regimes[PAIR_A].probes) == 2
        assert [pr[0] for pr in s.regimes[PAIR_A].probes] == list(act.meta["x"])
        assert row["action"] == act.id and row["outcome"] == "bundle" and row["supersedes"] == []
        assert row["value_predicted"] == pytest.approx(act.value_bits) and row["value_unit"] == "bits"
        assert row["members"] == [m.id for m in act.meta["member_actions"]]
        expected += prob * row["value_realized"]
    assert expected == pytest.approx(act.value_bits, abs=1e-9)


def test_chain_throw_bundle_value_is_the_set_value_of_its_links():
    """A chain throw (throws.chain_throw: the decoded path) is priced by precision_form.set_value_bits of its links'
    rows — the joint log-det value, which is not the sum of the links' own values."""
    st = _state()
    ch = chain_throw_bundle(st, [0, 1, 2, 3], w=1.0, cost_per_link=0.5)
    H = np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 1.0, -1.0]])
    assert ch.value_bits == pytest.approx(st.form.set_value_bits(H, np.ones(3)), rel=1e-12)
    assert ch.cost == pytest.approx(1.5) and ch.meta["nodes"] == [0, 1, 2, 3]
    assert ch.value_bits < sum(ch.meta["members"])            # the links overlap: submodular, not additive
    row = apply(st, ch, {"outcomes": [{"confirmed_links": [(i, j, 1.0)]} for i, j in ((0, 1), (1, 2), (2, 3))]})
    assert row["value_realized"] == pytest.approx(ch.value_bits, rel=1e-9)   # log-det is exact, not an expectation
    with pytest.raises(ValueError, match="at least two nodes"):
        chain_throw_bundle(st, [2], w=1.0)


def test_a_bundle_across_different_pairs_is_the_sum_of_its_members():
    """Independent beliefs: the joint posterior factorizes, so the exact set value is additive across targets
    (e35 (a)). On ONE pair it is not — that is the whole point of pricing a purchase at its own depth."""
    st = _state()
    a = probe_pair_bundle(st, PAIR_A).meta["member_actions"][0]
    b = probe_pair_bundle(st, PAIR_B).meta["member_actions"][0]
    cross = bundle_action(st, [a, b])
    assert cross.value_bits == pytest.approx(sum(cross.meta["members"]), abs=1e-12)
    assert len(cross.meta["groups"]) == 2 and cross.cost == pytest.approx(a.cost + b.cost)
    same = probe_pair_bundle(st, PAIR_A)
    assert same.value_bits < sum(same.meta["members"]) - 1e-9              # two probes on one pair overlap
    joint, meta = bundle_value_bits(st, same.meta["member_actions"])
    assert joint == pytest.approx(same.value_bits, rel=1e-12) and meta["exact"]


def test_malformed_bundles_raise():
    st = _state()
    good = probe_pair_bundle(st, PAIR_A)
    m = good.meta["member_actions"][0]
    meas = next(a for a in next_actions(st, k=20) if a.kind == "measure")
    with pytest.raises(ValueError, match="non-empty"):
        bundle_action(st, [])
    with pytest.raises(ValueError, match="must be an Action"):
        bundle_action(st, [m, {"kind": "probe"}])
    with pytest.raises(ValueError, match="cannot contain a bundle"):
        bundle_action(st, [m, good])
    with pytest.raises(ValueError, match="must be a probe"):
        bundle_action(st, [m, meas])
    with pytest.raises(ValueError, match="meta\\['x'\\]"):
        bundle_action(st, [Action("probe", PAIR_A, m.instrument, 1.0, 1.0, 1.0, "regime_posterior", "how")])
    with pytest.raises(ValueError, match="no regime posterior"):
        bundle_action(st, [Action("probe", ("no", "pair"), m.instrument, 1.0, 1.0, 1.0, "regime_posterior", "how",
                                  meta={"x": 0.5})])
    with pytest.raises(ValueError, match="cost must be"):
        bundle_action(st, [m], cost=0.0)
    with pytest.raises(ValueError, match="must carry"):
        apply(st, good, {"sign": 1})
    with pytest.raises(ValueError, match="one outcome per member"):
        apply(st, good, {"outcomes": [{"sign": 1}]})
    with pytest.raises(ValueError, match="at least one point"):
        sweep_bundle(st, PAIR_A, [], Instrument("cell", 2.0, 0.9))


def test_a_subsampled_bundle_says_so_in_its_how():
    big = sweep_bundle(_state(), PAIR_A, [0.05 + 0.1 * i for i in range(9)], Instrument("cell", 3.0, 0.9), n_sample=64)
    assert "subsample" in big.how and "return " in big.how and "\n" not in big.how
