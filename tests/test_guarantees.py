#!/usr/bin/env python3
"""guarantees: every claim in guarantees.py's docstring that is a claim about arithmetic.

The numbers printed by e33 are measurements; the assertions here are the ones that are either proved
in the docstring (and must then hold exactly) or are the specific certificates the docstring quotes.
"""
import math

import numpy as np
import pytest

from graph_engine.guarantees import (adaptive_submodularity_check, allocation_report,
                                     brute_force_adaptive_optimum, bsc_gain, entropy_drop_bound,
                                     error_drop_bound, family_drop_floor, family_drops, family_entropy,
                                     greedy_sequence_value, greedy_vs_optimal, one_step_drops,
                                     potential_of, state_of, update)
from graph_engine.regime_posterior import RegimePosterior

R = 0.95


def small_case(seed, n_grid=None, p_two=0.05):
    rng = np.random.default_rng(seed)
    rp = RegimePosterior(0.0, 1.0, reliability=0.75,
                         n_grid=int(rng.integers(4, 7)) if n_grid is None else n_grid,
                         p_flip=0.3, p_two=p_two)
    for _ in range(int(rng.integers(2, 4))):
        a = float(rng.uniform(0, 0.8))
        b = float(min(1.0, a + rng.uniform(0.1, 0.4)))
        rp.add_claim(a, b, int(rng.choice([-1, 1])), n_eff=float(rng.uniform(0.5, 3.0)),
                     reliability=float(rng.uniform(0.6, 0.95)))
    return rp


# ── the state arithmetic is RegimePosterior's own ──────────────────────────────────────────────
def test_update_is_the_posteriors_own_probe_step():
    """`update` reproduces _with_probes' likelihood step exactly — everything else rests on this."""
    for s in range(20):
        rp = small_case(s)
        st = state_of(rp)
        cells = st.cells
        x = float(cells[2].mean())
        c = min(int(np.searchsorted(cells[:, 1], x, side="left")), len(cells) - 1)
        for sg in (1, -1):
            st2 = update(st, c, sg, R)
            rp2 = small_case(s)
            rp2.add_probe(x, sg, R)
            assert potential_of(st2) == pytest.approx(rp2.potential_value(), abs=1e-12)
            assert np.allclose(st2.post, state_of(rp2).post, atol=1e-12)


def test_one_step_drops_is_best_probes_objective():
    for s in range(20):
        rp = small_case(s)
        st = state_of(rp)
        d = one_step_drops(st, R)
        x, v = rp.best_probe(R)
        assert float(d.max()) == pytest.approx(v, abs=1e-12)
        c = min(int(np.searchsorted(st.cells[:, 1], x, side="left")), len(st.cells) - 1)
        assert d[c] == pytest.approx(v, abs=1e-12)


def test_family_drops_is_model_check_probes_objective():
    for s in range(20):
        rp = small_case(s)
        st = state_of(rp)
        assert float(family_drops(st, R).max()) == pytest.approx(rp.model_check_probe(R)[1], abs=1e-12)


# ── 1. proved: adaptive monotonicity ───────────────────────────────────────────────────────────
def test_adaptive_monotone_always():
    """PROVED in the docstring: the posterior is a martingale on the fixed hypothesis space, so by
    Jensen every cell term of a concave potential drops by ≥ 0 — for BOTH potentials, every cell,
    every history."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for s in range(40):
        rp = small_case(s)
        st0 = state_of(rp)
        for pot in ("entropy", "error"):
            st = st0.with_potential(pot)
            for _ in range(3):                       # a few random histories
                c = int(rng.integers(0, st.n_cells))
                st = update(st, c, int(rng.choice([-1, 1])), R)
                worst = min(worst, float(one_step_drops(st, R).min()))
    assert worst > -1e-12


def test_cell_drop_is_mutual_information():
    """Eq. (1): the expected drop of one cell's entropy term equals I(Z_c;Y) = h₂(E a) − E h₂(a),
    computed from the emission probabilities alone. This identity is what both bounds rest on."""
    h = lambda t: 0.0 if t <= 0 or t >= 1 else -(t * math.log2(t) + (1 - t) * math.log2(1 - t))
    for s in range(10):
        rp = small_case(s)
        st = state_of(rp)
        for c in range(st.n_cells):
            for target in range(st.n_cells):
                # drop of cell `target`'s entropy from a probe in cell c
                now = h(float(st.post @ st.F[:, target]))
                after = 0.0
                for sg in (1, -1):
                    ff = st.F[:, c] if sg > 0 else 1 - st.F[:, c]
                    like = ff * R + (1 - ff) * (1 - R)
                    pout = float(st.post @ like)
                    if pout <= 0:
                        continue
                    q = st.post * like / pout
                    after += pout * h(float(q @ st.F[:, target]))
                drop = now - after
                # the Jensen-gap form: a = P(Y=+|Z_target), the two conditional emission means
                pt = st.F[:, target]
                ac = st.F[:, c] * R + (1 - st.F[:, c]) * (1 - R)
                m1 = float(st.post @ (pt * ac)); w1 = float(st.post @ pt)
                m0 = float(st.post @ ((1 - pt) * ac)); w0 = float(st.post @ (1 - pt))
                if w1 <= 1e-12 or w0 <= 1e-12:
                    continue
                gap = h(m1 + m0) - (w1 * h(m1 / w1) + w0 * h(m0 / w0))
                assert drop == pytest.approx(gap, abs=1e-9)


# ── 1a. NOT adaptive submodular: the mechanism, and the exact certificate ──────────────────────
def test_bsc_gain_increasing_below_half():
    """Eq. (3): γ_r(p) = h₂(a) − h₂(r) is symmetric about ½ and strictly increasing on (0, ½).
    That monotonicity is the mechanism of the counterexample."""
    for r in (0.6, 0.8, 0.95, 0.99):
        ps = np.linspace(0.001, 0.5, 120)
        g = np.array([bsc_gain(float(p), r) for p in ps])
        assert np.all(np.diff(g) > 0)
        for p in (0.02, 0.17, 0.33, 0.49):
            assert bsc_gain(p, r) == pytest.approx(bsc_gain(1 - p, r), abs=1e-12)
        assert bsc_gain(0.5, r) == pytest.approx(1.0 - (-(r * math.log2(r) + (1 - r) * math.log2(1 - r))), abs=1e-12)


def _minimal(pot):
    """The docstring's certificate: 4 grid cells, one claim [0, ½] sign +, weight 3, r_claim 0.9."""
    rp = RegimePosterior(0.0, 1.0, reliability=0.9, n_grid=4, p_flip=0.3, p_two=0.0, potential=pot)
    rp.add_claim(0.0, 0.5, +1, n_eff=3.0)
    return rp


def test_adaptive_submodularity_counterexample():
    """The exact instance quoted in the docstring. A = ∅ ⊆ B = {one − answer}, and the SAME probe is
    worth strictly more after B — the diminishing-returns inequality fails, so the objective is not
    adaptive submodular and the Golovin–Krause 1−1/e bound does not apply to it."""
    st = state_of(_minimal("entropy"))
    d0 = one_step_drops(st, R)
    d1 = one_step_drops(update(st, 1, -1, R), R)
    assert d0[2] == pytest.approx(0.05112323, abs=1e-6)
    assert d1[2] == pytest.approx(0.22849394, abs=1e-6)
    assert d1[2] / d0[2] == pytest.approx(4.4695, rel=1e-3)

    st = state_of(_minimal("error"))
    d0 = one_step_drops(st, R)
    d1 = one_step_drops(update(st, 1, -1, R), R)
    assert d0[2] == pytest.approx(0.00477591, abs=1e-7)
    assert d1[2] == pytest.approx(0.06948769, abs=1e-7)
    # and the infinite-ratio version: a probe worth EXACTLY zero becomes worth something
    d1b = one_step_drops(update(st, 3, -1, R), R)
    assert abs(d0[1]) < 1e-14
    assert d1b[1] == pytest.approx(0.00716951, abs=1e-7)


def test_submodularity_violations_are_a_steady_minority():
    """e33's measurement, pinned: the inequality fails on a minority — but a solid one — of random
    sub-realizations, for both potentials. MEASURED, not proved."""
    rp = small_case(7, n_grid=6)
    res = adaptive_submodularity_check(rp, list(state_of(rp).midpoints()), n_random=200,
                                       reliability=R, seed=1)
    assert res["entropy"]["violation_rate"] == pytest.approx(0.07, abs=1e-9)
    assert res["error"]["violation_rate"] == pytest.approx(0.065, abs=1e-9)
    for pot in ("entropy", "error"):
        assert res[pot]["violation_rate"] > 0.02
        assert res[pot]["worst_ratio"] > 3.0


# ── 1c. greedy against the optimal adaptive policy ─────────────────────────────────────────────
def test_budget_one_greedy_is_optimal():
    """With one probe the greedy choice IS the optimal adaptive policy — a check on the enumeration."""
    for s in range(10):
        rp = small_case(s)
        r_ = greedy_vs_optimal(rp, 1, reliability=R)
        assert r_["greedy_value"] == pytest.approx(r_["optimal_value"], abs=1e-12)


def test_optimal_never_below_greedy_and_ratio_above_one_minus_1_over_e():
    """MEASURED (e33 runs 50 cases; here 12 of the same family). The enumeration is exact, so
    optimal ≥ greedy is arithmetic; the ratio staying above 1−1/e is a measurement — the objective is
    NOT adaptive submodular, so nothing guarantees it."""
    ratios = []
    for s in range(12):
        rp = small_case(500 + s)
        for pot in ("entropy", "error"):
            r_ = greedy_vs_optimal(rp, 2 + (s % 2), reliability=R, potential=pot)
            assert r_["optimal_value"] >= r_["greedy_value"] - 1e-12
            ratios.append(r_["ratio"])
    ratios = np.array(ratios)
    assert ratios.min() > 1 - 1 / math.e
    assert ratios.min() > 0.9
    assert (ratios < 1 - 1e-9).any()          # greedy is genuinely suboptimal somewhere


def test_greedy_value_is_the_realized_expectation():
    """The expected drop the greedy policy is credited with equals the answer-averaged realized drop,
    computed by replaying the tree through RegimePosterior itself (no state shortcut)."""
    rp = small_case(11)
    st = state_of(rp)
    g = greedy_sequence_value(rp, 2, reliability=R)

    def replay(probes):
        p = small_case(11)
        for x, sg in probes:
            p.add_probe(x, sg, R)
        return p

    def rec(probes, k):
        p = replay(probes)
        if k == 0:
            return p.potential_value()
        x, _ = p.best_probe(R)
        s = state_of(p)
        c = min(int(np.searchsorted(s.cells[:, 1], x, side="left")), len(s.cells) - 1)
        tot = 0.0
        for sg in (1, -1):
            ff = s.F[:, c] if sg > 0 else 1 - s.F[:, c]
            like = ff * R + (1 - ff) * (1 - R)
            pout = float(s.post @ like)
            if pout <= 0:
                continue
            tot += pout * rec(probes + [(x, sg)], k - 1)
        return tot

    assert rec([], 2) == pytest.approx(g["greedy_final_U"], abs=1e-10)


# ── 2. the allocation lemma ────────────────────────────────────────────────────────────────────
NG = 12


def _hidden_excursion(seed, strength, gap=2):
    """Two same-sign claims with a closure-filled span between them (the e21 geometry)."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(2, NG - gap - 1))
    rp = RegimePosterior(0.0, 1.0, reliability=0.95, n_grid=NG, p_flip=0.3, p_two=0.05)
    sgn = int(rng.choice([-1, 1]))
    rp.add_claim(0.0, k / NG, sgn, n_eff=strength * float(rng.uniform(0.9, 1.1)))
    rp.add_claim((k + gap) / NG, 1.0, sgn, n_eff=strength * float(rng.uniform(0.9, 1.1)))
    return rp


def test_f_bound_holds():
    """PROVED, eq. (4): no single probe can drop the potential by more than f(δ), for either
    potential. 120 configurations spanning δ from 0.5 to 0.003."""
    worst = 0.0
    for s in range(120):
        strength = float(np.random.default_rng(9000 + s).choice([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]))
        rp = _hidden_excursion(9000 + s, strength)
        rep = allocation_report(rp, R)
        f = entropy_drop_bound(rep["delta"], rep["width"], R)
        assert rep["max_entropy_drop"] <= f + 1e-12
        st = state_of(rp, "error")
        assert float(one_step_drops(st, R).max()) <= error_drop_bound(rep["delta"], rep["width"], R) + 1e-12
        worst = max(worst, rep["max_entropy_drop"] / max(f, 1e-300))
    assert worst < 1.0


def test_g_bound_holds():
    """PROVED, eq. (5): the best cell's family-entropy drop is at least g(q, r, s)."""
    slacks = []
    for s in range(120):
        strength = float(np.random.default_rng(9000 + s).choice([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]))
        rep = allocation_report(_hidden_excursion(9000 + s, strength), R)
        g = family_drop_floor(rep["q_two"], R, rep["separation_s"])
        assert rep["max_family_drop"] >= g - 1e-12
        slacks.append(rep["max_family_drop"] / max(g, 1e-300))
    assert min(slacks) >= 1.0
    assert np.median(slacks) < 4.0          # the floor is within a small factor of the value


def test_delta_is_at_least_q():
    """δ ≥ q whenever the dominant hypothesis is a single-transition one — the reason f and g do not
    by themselves order the two rules, and the reason the separation has to be measured."""
    for s in range(40):
        rep = allocation_report(_hidden_excursion(9000 + s, 4.0), R)
        assert rep["delta"] >= rep["q_two"] - 1e-12


def test_allocation_separation_grows_as_delta_falls():
    """MEASURED (e33's sweep, pinned here): as the single-transition story gets better (δ → 0) the
    value rule's best probe collapses while the model-check probe's value plateaus, so the ratio
    grows monotonically. This is the theorem behind e21's hole."""
    rows = [allocation_report(_hidden_excursion(3, n, gap=2), R) for n in (0.5, 2, 8, 32)]
    d = [r["delta"] for r in rows]
    ru = [r["max_entropy_drop"] for r in rows]
    rt = [r["max_family_drop"] for r in rows]
    ratio = [r["ratio_T_over_U"] for r in rows]
    assert all(d[i] > d[i + 1] for i in range(3))            # δ falls
    assert all(ru[i] > ru[i + 1] for i in range(3))          # the value rule's best probe collapses
    assert all(ratio[i] < ratio[i + 1] for i in range(3))    # and the ratio grows, monotonically
    assert ratio[0] < 0.1 and ratio[-1] > 9.0
    assert ru[-1] < ru[0] / 100                              # 300× in the e33 sweep
    assert rt[-1] > rt[0] / 4                                # the T-value falls by ~2×, not 300×
    # at the concentrated end the value rule names the SAME cell and prices it ~10x too low: what
    # fails is the price, not the location (the 36.5 % branch in the docstring's measurement)
    assert rows[-1]["argmax_entropy_cell"] == rows[-1]["argmax_family_cell"]
    assert rows[-1]["max_entropy_drop"] < 0.11 * rows[-1]["max_family_drop"]


def test_family_entropy_and_bounds_degenerate_cleanly():
    rp = RegimePosterior(0.0, 1.0, n_grid=6, p_two=0.0)
    rp.add_claim(0.0, 0.5, +1, n_eff=2.0)
    st = state_of(rp)
    assert family_entropy(st) == 0.0
    assert float(family_drops(st, R).max()) == 0.0
    assert family_drop_floor(0.0, R, 1.0) == 0.0
    assert entropy_drop_bound(0.0, 1.0, R) == 0.0
    # r = 0.5 is a coin: no probe can move anything, and both bounds vanish
    assert entropy_drop_bound(0.3, 1.0, 0.5) == 0.0
    assert family_drop_floor(0.3, 0.5, 1.0) == 0.0


def test_brute_force_candidate_mapping():
    """`candidates` are x values and are mapped to their cells; the default is every cell midpoint."""
    rp = small_case(4)
    st = state_of(rp)
    a = brute_force_adaptive_optimum(rp, 1, reliability=R)
    b = brute_force_adaptive_optimum(rp, 1, candidates=list(st.midpoints()), reliability=R)
    assert a["optimal_value"] == pytest.approx(b["optimal_value"], abs=1e-12)
    assert a["n_candidates"] == st.n_cells
    c = brute_force_adaptive_optimum(rp, 1, candidates=[float(st.midpoints()[0])], reliability=R)
    assert c["n_candidates"] == 1
    assert c["optimal_value"] <= a["optimal_value"] + 1e-12
