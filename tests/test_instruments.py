"""The instrument contract, and the first instance on measured contact-solver numbers.

The fixture is `examples/engine_experiments/e39_contact_solver_data.json`: recorded contact-solver measurements
(a 21-scene router benchmark, 14 rho ladders of 10 points each, 448 scenes through 5 solver channels). Nothing
here runs a solver; the point of the module under test is that the reduction from those numbers to the engine's
claim form is a contract, checked, rather than a script written once per study.
"""
import json
import math
import os
import statistics

import pytest

from graph_engine.instruments import ContactSolverSweep, Instrument, Reading, SweepFamily, load_family
from graph_engine.next_actions import EngineState, apply
from graph_engine.next_actions import Instrument as PricedInstrument
from graph_engine.regime_posterior import RegimePosterior

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "examples", "engine_experiments", "e39_contact_solver_data.json")


@pytest.fixture(scope="module")
def data():
    with open(DATA) as f:
        return json.load(f)


class Fixed(Instrument):
    """The smallest legal instrument: it reports one sign with one probability at one cost."""

    def __init__(self, r_sign=1, r_p=0.9, r_cost=1.0, **kw):
        self._r = Reading(r_sign, r_p, r_cost)
        kw.setdefault("name", "fixed")
        kw.setdefault("cost", 2.0)
        kw.setdefault("reliability", 0.9)
        kw.setdefault("lineage_root", "run:1")
        super().__init__(**kw)

    def measure(self, pair, x):
        return self._r


def _toy_rows():
    """Two scenes where A is ~8x cheaper, two where B is ~11x cheaper: one sign change on the axis."""
    return [SweepFamily("s1", 0.0, 10.0, 100.0, 1.0), SweepFamily("s2", 0.1, 12.0, 90.0, 1.0),
            SweepFamily("s3", 2.0, 500.0, 50.0, 2.0), SweepFamily("s4", 2.1, 600.0, 45.0, 2.0)]


# -- the contract ------------------------------------------------------------------------------------
def test_an_instrument_is_cost_reliability_lineage_root_and_a_reduction():
    ins = Fixed()
    assert (ins.cost, ins.reliability, ins.lineage_root) == (2.0, 0.9, "run:1")
    r = ins.probe(("a", "b"), 0.5)
    assert isinstance(r, tuple) and len(r) == 3                    # probe(pair, x) -> (sign, p, cost)
    sign, p, cost = r
    assert (sign, p, cost) == (1, 0.9, 1.0)
    assert (r.sign, r.p, r.cost) == (sign, p, cost)


def test_reliability_must_be_the_probability_of_the_correct_sign():
    # P(correct sign) below ½ is a broken instrument, not a pessimistic one; 1.0 is a deductive certificate,
    # which is claim_types' business, not a sweep's.
    for bad in (0.49, 1.0, 1.4, -0.1):
        with pytest.raises(ValueError, match="P.correct sign"):
            Fixed(reliability=bad)
    Fixed(reliability=0.5)                                          # a coin is legal; it just buys nothing


def test_cost_and_lineage_root_are_required():
    with pytest.raises(ValueError, match="cost"):
        Fixed(cost=0.0)
    with pytest.raises(ValueError, match="cost"):
        Fixed(cost=float("inf"))
    with pytest.raises(ValueError, match="ONE lineage root"):
        Fixed(lineage_root="")
    with pytest.raises(ValueError, match="name"):
        Fixed(name="")


def test_a_reading_that_breaks_the_claim_form_is_refused_at_the_boundary():
    with pytest.raises(ValueError, match=r"sign must be \+1 or -1"):
        Fixed(r_sign=0).probe(None, 0.0)
    with pytest.raises(ValueError, match="probability"):
        Fixed(r_p=1.7).probe(None, 0.0)
    with pytest.raises(ValueError, match="cost"):
        Fixed(r_cost=0.0).probe(None, 0.0)
    # a reading whose sign disagrees with its own probability is not a reading
    with pytest.raises(ValueError, match="contradicts"):
        Fixed(r_sign=1, r_p=0.2).probe(None, 0.0)

    class Broken(Fixed):
        def measure(self, pair, x):
            return 1                                                # not a triple

    with pytest.raises(TypeError, match=r"must return \(sign, p, cost\)"):
        Broken().probe(None, 0.0)


def test_sources_is_one_root_in_the_graph_contracts_shape():
    ins = Fixed()
    assert ins.sources() == [{"id": "run:1", "derives_from": []}]
    assert Fixed(derives_from=("campaign:x",)).sources() == [{"id": "run:1", "derives_from": ["campaign:x"]}]


def test_priced_is_the_same_instrument_next_actions_ranks_with():
    ins = Fixed()
    pr = ins.priced()
    assert isinstance(pr, PricedInstrument)
    assert (pr.name, pr.cost, pr.reliability) == (ins.name, ins.cost, ins.reliability)


def test_a_sweep_enters_as_one_lineage_root_with_weight_one_over_k():
    ins = Fixed()
    rp = RegimePosterior(0.0, 1.0, n_grid=16)
    recs = ins.enter(rp, ("a", "b"), [0.1, 0.3, 0.5, 0.7, 0.9])
    assert len(recs) == 5
    assert {r["lineage"] for r in recs} == {"run:1"}
    assert all(r["weight"] == pytest.approx(0.2) for r in recs)
    assert sum(r["weight"] for r in recs) == pytest.approx(1.0)
    assert rp.model_probe_lineage == recs
    assert len(rp.probes) == 5


def test_k_answers_at_one_point_equal_one_answer_of_the_same_reliability():
    """model_probes' own invariant: resample a deterministic cell and it still tells you one thing (N_eff = 1)."""
    ins = Fixed(r_sign=1, r_p=0.9)
    many = RegimePosterior(0.0, 1.0, n_grid=16)
    one = RegimePosterior(0.0, 1.0, n_grid=16)
    ins.enter(many, ("a", "b"), [0.4] * 25)
    ins.enter(one, ("a", "b"), [0.4])
    xs = [0.05, 0.25, 0.45, 0.65, 0.85]
    assert [many.p_plus(x) for x in xs] == pytest.approx([one.p_plus(x) for x in xs], abs=1e-12)
    assert many.potential_value() == pytest.approx(one.potential_value(), abs=1e-12)


def test_a_reading_enters_at_its_own_confidence_not_at_the_instruments_average():
    near_coin = Fixed(r_sign=1, r_p=0.52)
    rp = RegimePosterior(0.0, 1.0, n_grid=16)
    rec = near_coin.enter(rp, ("a", "b"), [0.5])[0]
    assert rec["reliability"] == pytest.approx(0.52)                # max(p, 1-p), not the instrument's 0.9
    flat = Fixed(r_sign=-1, r_p=0.02).enter(RegimePosterior(0.0, 1.0, n_grid=16), ("a", "b"), [0.5])[0]
    assert flat["reliability"] == pytest.approx(0.98)


def test_the_purchase_is_priced_as_one_cell_not_as_k_probes():
    state = EngineState(pair_bundles=False, probe_instruments=[])
    pair = ("x", "y")
    state.regimes[pair] = RegimePosterior(0.0, 1.0, n_grid=16)
    state.regime_variables[pair] = "x"
    ins = Fixed()
    act = ins.action(state, pair, [0.1, 0.3, 0.5, 0.7, 0.9])
    assert act.kind == "bundle"
    assert act.cost == pytest.approx(ins.cost)                      # the cell's cost, NOT 5 x a probe cost
    assert act.meta["K"] == 5 and act.meta["weight"] == pytest.approx(0.2)
    assert act.meta["lineage_root"] == "run:1"
    assert act.meta["sources"] == ins.sources()
    assert act.value_per_cost == pytest.approx(act.value_bits / act.cost)


def test_the_outcome_round_trips_through_apply_and_lands_in_the_ledger():
    state = EngineState(pair_bundles=False, probe_instruments=[])
    pair = ("x", "y")
    state.regimes[pair] = RegimePosterior(0.0, 1.0, n_grid=16)
    state.regime_variables[pair] = "x"
    ins = Fixed()
    xs = [0.2, 0.4, 0.6, 0.8]
    act = ins.action(state, pair, xs)
    row = apply(state, act, ins.outcome(pair, xs))
    assert row["action"].startswith("bundle:")
    assert row["value_predicted"] == pytest.approx(act.value_bits)
    assert len(state.regimes[pair].probes) == 4
    assert all(p[3] == pytest.approx(0.25) for p in state.regimes[pair].probes)


def test_an_empty_sweep_is_refused():
    with pytest.raises(ValueError, match="at least one point"):
        Fixed().sweep(("a", "b"), [])


# -- the first instance ------------------------------------------------------------------------------
def test_the_sign_follows_the_measured_ratio_and_flips_along_the_axis():
    ins = ContactSolverSweep("toy", _toy_rows(), "campaign:toy", method_a="A", method_b="B")
    assert ins.probe(None, 0.05).sign == +1                         # A is ~8x cheaper at the low end
    assert ins.probe(None, 2.05).sign == -1                         # B is ~11x cheaper at the high end
    assert math.exp(ins.advantage(0.05)) == pytest.approx(8.66, rel=0.02)
    assert math.exp(-ins.advantage(2.05)) == pytest.approx(11.4, rel=0.02)
    assert ins.cost == pytest.approx(6.0)                           # the family's own recorded cost, summed
    assert "A beats B" in ins.claim(0.05) and "B beats A" in ins.claim(2.05)


def test_the_probability_comes_from_the_spread_so_an_unseparated_window_is_worth_nothing():
    same = [SweepFamily(f"s{i}", 0.0, 100.0, 100.0 * (1.02 if i % 2 else 0.98), 1.0) for i in range(6)]
    ins = ContactSolverSweep("flat", same, "campaign:flat")
    _sign, p, _c = ins.probe(None, 0.0)
    assert abs(p - 0.5) < 0.1                                       # the two methods sit inside each other's spread
    assert ins.reliability < 0.6
    wide = [SweepFamily(f"s{i}", 0.0, 10.0, 1000.0, 1.0) for i in range(6)]
    assert ContactSolverSweep("sep", wide, "campaign:sep").reliability > 0.99


def test_reliability_is_measured_and_cannot_be_talked_up():
    rows = _toy_rows()
    measured = ContactSolverSweep("toy", rows, "campaign:toy").reliability
    assert 0.5 <= measured < 1.0
    with pytest.raises(ValueError):
        ContactSolverSweep("toy", rows, "campaign:toy", reliability=1.0)
    with pytest.raises(ValueError, match="positive"):
        ContactSolverSweep("bad", [SweepFamily("s", 0.0, 0.0, 1.0, 1.0)], "campaign:bad")
    with pytest.raises(ValueError, match="at least one measured scene"):
        ContactSolverSweep("empty", [], "campaign:empty")


def test_load_family_reads_one_table_along_the_axis_it_is_told_to(data):
    rows = load_family(data["router"]["scenes"], x_of=lambda r: math.log10(r["n_c"]),
                       a_of=lambda r: r["pgs_sweeps"], b_of=lambda r: r["admm_sweeps"],
                       cost_of=lambda r: r["router_ms"], label_of=lambda r: r["scene"])
    assert len(rows) == 21
    assert {r.label for r in rows} >= {"cube", "a1_slip", "dem_step", "pack800"}


# -- the measured numbers ----------------------------------------------------------------------------
def _router_rows(data):
    rows = []
    for r in data["router"]["scenes"]:
        ms = [v for v in (r.get("pgs_ms"), r.get("admm_ms")) if v is not None and math.isfinite(v)]
        rows.append(SweepFamily(r["scene"], math.log10(r["n_c"]), min(r["pgs_sweeps"], 30000.0),
                                min(r["admm_sweeps"], 30000.0), sum(ms) if ms else r["router_ms"]))
    return rows


def test_the_router_benchmark_is_the_table_it_was_measured_as(data):
    t = data["router"]
    assert t["n_scenes"] == 21 and t["n_misrouted"] == 4
    assert {r["scene"] for r in t["scenes"] if r["misrouted"]} == {"cube", "a1_slip", "h25_a1_stick",
                                                                   "h25_a1_slip"}
    assert {r["n_c"] for r in t["scenes"] if r["misrouted"]} == {4}      # every misroute sits at n_c = 4


def test_the_sweep_reads_the_small_contact_regime_as_the_one_that_is_open(data):
    """Where the router misroutes, the measured advantage has no stable sign — and the instrument says so."""
    rows = _router_rows(data)
    ins = ContactSolverSweep("router", rows, "campaign:router_benchmark", method_a="plain PGS",
                             method_b="ADMM ladder", quantity="sweep-equivalents", window=0.14)
    assert ins.reliability == pytest.approx(0.7863, abs=0.001)       # measured over the family's own points
    assert ins.cost == pytest.approx(125996.9, rel=1e-4)             # ms of the whole recorded campaign
    # the window holding every misroute: the six scenes at n_c = 4, read at the window's midpoint
    small = [r for r in rows if r.x == pytest.approx(math.log10(4))]
    assert len(small) == 6
    w = ContactSolverSweep("router:w1", small, "campaign:router_benchmark:w1", method_a="plain PGS",
                           method_b="ADMM ladder", window=0.14)
    sign, p, _c = w.probe(None, 0.6995)
    assert sign == +1                                                # PGS wins there on average
    assert p == pytest.approx(0.878, abs=0.005)                      # and only just, across 6 scenes
    assert math.exp(w.advantage(0.6995)) == pytest.approx(4.84, rel=0.01)
    # the same reduction on the windows either side of it reads the opposite sign at p = 1
    for lo, hi in ((0.30, 0.57), (1.36, 1.63)):
        band = [r for r in rows if lo <= r.x < hi]
        v = ContactSolverSweep("band", band, "campaign:router_benchmark:band", window=0.14)
        sg, pp, _ = v.probe(None, 0.5 * (lo + hi))
        assert sg == -1 and max(pp, 1 - pp) > 0.99


def test_the_least_certain_router_window_is_the_one_holding_the_misroutes(data):
    rows = _router_rows(data)
    ins_all = ContactSolverSweep("router", rows, "campaign:router_benchmark", window=0.14)
    lo, hi = min(r.x for r in rows), max(r.x for r in rows)
    conf = {}
    for i in range(12):
        a = lo + i * (hi - lo) / 12
        b = hi if i == 11 else lo + (i + 1) * (hi - lo) / 12
        inside = [r for r in rows if a <= r.x < b or (i == 11 and r.x == hi)]
        if not inside:
            continue
        w = ContactSolverSweep(f"w{i}", inside, f"campaign:router_benchmark:w{i}", window=0.14)
        _s, p, _c = w.probe(None, 0.5 * (a + b))
        conf[i] = max(p, 1 - p)
    assert len(conf) == 9                                            # 9 of 12 windows hold a scene
    assert conf[1] == pytest.approx(0.878, abs=0.005)
    assert min(conf, key=conf.get) == 7                              # the densest ADMM/PGS tie, 3 scenes at n_c 176-256
    assert sorted(conf, key=conf.get)[1] == 1                        # and next to it, the misroute window
    assert ins_all.reliability == pytest.approx(0.7863, abs=0.001)


def test_the_rho_ladder_censors_a_run_that_did_not_converge(data):
    lad = data["rho_ladders"]["dem_step"]
    assert lad["conv_B"] is False                                    # rho_B does not converge on the jammed step
    caps = [p for p in lad["sweep"] if not p["conv"]]
    assert caps, "the jammed ladder has non-converged points, kept rather than dropped"
    rows = [SweepFamily(f"dem_step@{p['rho']:.4g}", math.log10(p["rho"] / lad["rho_B"]),
                        max(float(p["iters"]) if p["conv"] else float(lad["cap"]), 1.0),
                        float(lad["cap"]), max(p["wall"] * 1e3, 1e-3)) for p in lad["sweep"]]
    ins = ContactSolverSweep("rho_dem", rows, "campaign:rho_ladder", method_a="the ladder point", method_b="rho_B",
                             window=0.30)
    best = min((p for p in lad["sweep"] if p["conv"]), key=lambda p: p["iters"])
    assert math.log10(best["rho"] / lad["rho_B"]) == pytest.approx(0.359, abs=0.005)   # 2.29x rho_B
    assert best["iters"] == 1906
    assert ins.probe(None, 0.359).sign == +1                         # that point beats rho_B, which never converged


def test_the_channel_family_carries_its_own_known_answer(data):
    dis = data["disagreement"]
    assert len(dis["scenes"]) == 448 and dis["n_exact_lost"] == 39
    lost_by_decade = {0: 0, 1: 0, 2: 0}
    n_by_decade = {0: 0, 1: 0, 2: 0}
    for s in dis["scenes"]:
        k = min(int(math.log10(max(s["r"], 1.0))), 2)
        n_by_decade[k] += 1
        lost_by_decade[k] += int(s["exact_lost"])
    assert (n_by_decade[0], n_by_decade[1], n_by_decade[2]) == (92, 134, 222)
    assert (lost_by_decade[0], lost_by_decade[1], lost_by_decade[2]) == (24, 8, 7)
    # the losses concentrate at LOW mass ratio, 3.0x the family's own 39/448 rate in the first decade
    assert lost_by_decade[0] / n_by_decade[0] == pytest.approx(0.261, abs=0.005)
    assert (lost_by_decade[0] / n_by_decade[0]) / (39 / 448) == pytest.approx(3.00, abs=0.05)


def test_the_disagreement_score_is_the_one_that_was_measured(data):
    ds = sorted(s["D"] for s in data["disagreement"]["scenes"] if s["src"] == "baseline")
    assert len(ds) == 64
    assert statistics.median(ds) == pytest.approx(16.344, abs=0.001)  # the recorded baseline median
    assert max(ds) == pytest.approx(65.423, abs=0.001)


def test_the_grid_is_the_familys_own_span():
    ins = ContactSolverSweep("toy", _toy_rows(), "campaign:toy")
    g = ins.grid(5)
    assert g[0] == pytest.approx(0.0) and g[-1] == pytest.approx(2.1)
    assert len(g) == 5 and ins.grid(1) == [pytest.approx(1.05)]
    with pytest.raises(ValueError, match="at least one point"):
        ins.grid(0)
