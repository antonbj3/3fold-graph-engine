"""alarm.py — what the false-alarm guarantee is, and what it is not.

The null is generated the only way that makes "correct model" exact: the truth is SAMPLED FROM the posterior's own
belief (one transition at most, the reliability the posterior assumes), so the answers come from precisely the
predictive law the price was computed with. Under that null the residual realized − predicted is a martingale
difference by construction, which is the hypothesis Ville's inequality is applied to.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_engine.alarm import Alarm, drop_outcomes, out_mean          # noqa: E402
from graph_engine.closed_loop import World, run                        # noqa: E402
from graph_engine.regime_posterior import RegimePosterior              # noqa: E402

ALPHA = 0.05
T = 12
N_GRID = 16


def _claims(rp, rng, sign=None, n_eff=3.0, crel=0.95, n=4):
    for _ in range(n):
        a = rng.uniform(0, 0.6)
        b = min(a + rng.uniform(0.2, 0.4), 1.0)
        if sign is None:
            sg = int(rng.choice([-1, 1]))
        else:
            xs = np.linspace(a, b, 33)
            sg = 1 if sum(sign(x) for x in xs) >= 0 else -1
        rp.add_claim(a, b, sg, n_eff=n_eff, reliability=crel)


def null_sequence(seed, alpha=ALPHA, T=T, r=0.95, place="grid", outcomes=True):
    """Correct model: the truth is drawn from the posterior, answers through the assumed reliability."""
    rng = np.random.default_rng(seed)
    rp = RegimePosterior(0.0, 1.0, n_grid=N_GRID, p_two=0.0, reliability=0.95)
    _claims(rp, rng)
    cells, _w, F, post = rp._with_probes()
    h = int(rng.choice(len(post), p=post))                      # the world this belief considers possible
    al = Alarm(alpha=alpha, require_law=outcomes)       # no law supplied ⇒ the bounded path, asked for
    xs = (np.arange(T) + 0.5) / T
    ws = []
    for t in range(T):
        x = float(xs[t]) if place == "grid" else float(rp.best_probe(r)[0])
        out = drop_outcomes(rp, x, r)
        before = rp.potential_value()
        cells, _w, F, _p = rp._with_probes()
        c = int(min(np.searchsorted(cells[:, 1], x, side="left"), len(cells) - 1))
        pplus = F[h, c] * r + (1 - F[h, c]) * (1 - r)
        sg = 1 if rng.random() < pplus else -1
        rp.add_probe(x, sg, r)
        w, _ = al.update(out_mean(out), before - rp.potential_value(), out if outcomes else None)
        ws.append(w)
    return al, ws


def planted_sequence(seed, alpha=ALPHA, T=T, r=0.95, place="grid", sep=0.35):
    """A SECOND transition the reader does not represent (p_two = 0): the model error the loop is blind to."""
    rng = np.random.default_rng(10_000 + seed)
    t1 = rng.uniform(0.2, 0.4)
    t2 = t1 + sep
    s0 = int(rng.choice([-1, 1]))
    sign = lambda x: s0 * (-1) ** ((x > t1) + (x > t2))          # noqa: E731
    rp = RegimePosterior(0.0, 1.0, n_grid=N_GRID, p_two=0.0, reliability=0.95)
    _claims(rp, rng, sign=sign)
    al = Alarm(alpha=alpha)
    xs = (np.arange(T) + 0.5) / T
    first = None
    for t in range(T):
        x = float(xs[t]) if place == "grid" else float(rp.best_probe(r)[0])
        out = drop_outcomes(rp, x, r)
        before = rp.potential_value()
        s = sign(x)
        if rng.random() > r:
            s = -s
        rp.add_probe(x, s, r)
        al.update(out_mean(out), before - rp.potential_value(), out)
        if al.hit and first is None:
            first = t + 1
    return al, first


# -- the guarantee -------------------------------------------------------------------------------------

def test_null_false_alarm_rate_under_alpha():
    """P(the wealth ever reaches 1/alpha) ≤ alpha. 200 sequences, 12 probes each, correct model."""
    hits = sum(null_sequence(s)[0].hit for s in range(200))
    assert hits / 200 <= ALPHA, f"false alarms {hits}/200 above alpha = {ALPHA}"
    assert hits <= 4                                            # measured 1/200; pinned so a regression is visible


def test_wealth_is_a_martingale_under_the_null():
    """Mean wealth over sequences stays at 1 (it is a martingale, hence a supermartingale): no drift up."""
    W = np.array([null_sequence(s)[1] for s in range(200)])
    means = W.mean(0)
    assert means.max() <= 1.35, f"mean wealth rose to {means.max():.2f}"
    se = W[:, -1].std() / np.sqrt(len(W))
    assert means[-1] <= 1.0 + 3 * se, f"final mean wealth {means[-1]:.3f} ± {se:.3f}"


def test_each_factor_has_conditional_mean_one():
    """The exact statement the guarantee rests on: E[exp(λ d̃ − κ(λ))] = 1 under the model's own law of the drop,
    for every bet in the mixture — no clipping, no boundedness, checked on real posteriors."""
    rng = np.random.default_rng(3)
    for s in range(6):
        rp = RegimePosterior(0.0, 1.0, n_grid=N_GRID, p_two=0.05)
        _claims(rp, rng)
        for x in (0.17, 0.5, 0.83):
            out = drop_outcomes(rp, x, 0.9)
            mu = out_mean(out)
            for c in Alarm().bets:
                sd = np.sqrt(sum(q * (d - mu) ** 2 for q, d in out))
                if sd <= 1e-12:
                    continue
                lam = c / sd
                kap = np.log(sum(q * np.exp(lam * (d - mu)) for q, d in out))
                got = sum(q * np.exp(lam * (d - mu) - kap) for q, d in out)
                assert abs(got - 1.0) < 1e-10


# -- power, which is measured and not guaranteed --------------------------------------------------------

def test_planted_second_transition_fires_in_a_majority_within_12_probes():
    res = [planted_sequence(s) for s in range(100)]
    fired = [f for _al, f in res if f is not None]
    assert len(fired) > 50, f"fired in only {len(fired)}/100 sequences"
    assert np.median(fired) <= T


def test_power_needs_the_probes_to_visit_the_defect():
    """The limit, measured: with the 12 probes placed by the engine's own value rule instead of swept over the domain
    the same planted worlds fire far less often — the value rule buys where the belief is uncertain, and a confidently
    wrong region is not that. This is e21b/e28's allocation limit seen from the test side, and it is why the loop-level
    result below is what it is."""
    swept = sum(planted_sequence(s)[0].hit for s in range(60))
    valued = sum(planted_sequence(s, place="best")[0].hit for s in range(60))
    assert valued < swept


def test_bounded_fallback_is_valid_and_much_weaker():
    """Without the outcome list the alarm falls back to a clipped bet (which has to be asked for, see below):
    still no false alarms, but the planted worlds do not move it — a single huge surprise can multiply the
    wealth by at most 1 + lam."""
    hits = sum(null_sequence(s, outcomes=False)[0].hit for s in range(100))
    assert hits == 0
    al, _ = null_sequence(0, outcomes=False)
    assert al.n == T and al.wealth > 0
    assert al.state()["n_bounded"] == T and al.state()["require_law"] is False


def test_latch_turns_on_at_threshold_and_off_below_one():
    al = Alarm(alpha=0.5)                                       # threshold 2
    out = [(0.5, 0.0), (0.5, 1.0)]                              # a fair coin's worth of drop, priced at 0.5
    for _ in range(3):                                          # the answer is always the high one: the price is wrong
        al.update(0.5, 1.0, out)
    assert al.alarm and al.hit and al.n_alarms == 1 and al.wealth >= 2
    while al.wealth >= 1.0:                                     # now the residuals go the other way
        al.update(0.5, 0.0, out)
    assert not al.alarm and al.hit                              # the latch drops, the crossing stays recorded


def test_alarm_rejects_bad_alpha():
    with pytest.raises(ValueError):
        Alarm(alpha=0.0)
    with pytest.raises(ValueError):
        Alarm(alpha=1.0)


# -- the loop flag -------------------------------------------------------------------------------------

def test_loop_flag_spends_nothing_when_no_alarm_fires():
    """+alarm is bit-for-bit its non-alarm counterpart while no alarm is latched: no reserved share, no probe moved."""
    w = World(seed=0)
    a = run(w, "engine", 20, seed=0)
    b = run(w, "engine+alarm", 20, seed=0)
    assert b["alarm_hits"] == 0
    assert b["wrong"] == a["wrong"] and b["spent"] == a["spent"] and b["spent_guard"] == 0.0
    c = run(w, "engine+bundle", 20, seed=0)
    d = run(w, "engine+alarm+bundle", 20, seed=0)
    assert d["alarm_hits"] == 0 and d["wrong"] == c["wrong"]


def test_loop_flag_changes_the_probes_once_an_alarm_latches():
    """With alpha loose enough that the test rejects, the family weight is raised and the trajectory moves — the
    mechanism that replaces the fixed guard share."""
    w = World(seed=1)
    base = run(w, "engine", 20, seed=1)
    hot = run(w, "engine+alarm", 20, seed=1, alarm_alpha=0.9, alarm_lam=8.0)
    assert hot["alarm_hits"] > 0
    assert hot["spent"] == base["spent"]                        # the budget, not a share of it, is what is spent
    assert hot["wrong"] != base["wrong"]


# -- the law is required -------------------------------------------------------------------------------
#
# The α is a statement about the model's own predictive law of the drop. A caller without such a law does
# not get a weaker guarantee of the same kind — it gets none, so `update` refuses by default instead of
# downgrading to the bounded path in silence. What that refusal is worth was measured on a domain that has
# no predictive law at all: an iterative solver's per-sweep residual drop, a deterministic function of the
# iterate. There the exact path could only be run on an INVENTED two-point law; with the convergence rate
# fitted to the sequence itself the wealth never left its peak 1.00 in three scenes, with the rate frozen
# the alarm fired at sweep 247 against a stagnation heuristic's sweep 104 at the SAME final error
# 2.384e-7, and on the two ill-conditioned scenes — where the stagnation rule stops at sweep 6 with
# 99.6 % / 99.7 % relative error — it never fired at all. The sequence below is that shape: a geometric
# residual (rate 0.853554, the measured per-sweep factor of the well-conditioned island) decaying to a
# float32 floor (2.980e-8, the measured floor of that scene).

RATE, RES0, FLOOR = 0.853554, 0.25, 2.980e-8


def _solver_residuals(n=400):
    """[(predicted drop, realized drop)] under the frozen-rate model: predictable until the floor binds."""
    r, out = RES0, []
    for _ in range(n):
        nxt = max(RATE * r, FLOOR)
        out.append((r * (1 - RATE), r - nxt))
        r = nxt
    return out


def test_update_refuses_a_drop_that_comes_with_no_predictive_law():
    al = Alarm()
    with pytest.raises(ValueError, match="predictive law"):
        al.update(0.1, 0.2)
    with pytest.raises(ValueError, match="predictive law"):
        al.update(0.1, 0.2, [])                                   # an empty law is not a law
    assert al.n == 0 and al.wealth == 1.0 and not al.hit          # the refusal leaves no state behind
    assert Alarm().state()["require_law"] is True


def test_the_bounded_path_has_to_be_asked_for_and_is_counted():
    al = Alarm(require_law=False)
    for predicted, realized in _solver_residuals(20):
        al.update(predicted, realized)
    assert al.n == 20 and al.state()["n_bounded"] == 20
    al_exact = Alarm()
    al_exact.update(0.5, 1.0, [(0.5, 0.0), (0.5, 1.0)])
    assert al_exact.state()["n_bounded"] == 0                     # the exact path is not the fallback


def test_a_perfectly_predictable_drop_cannot_be_tested():
    """The null is 'the price is the conditional expectation of the realized drop'. Where the drop is a
    deterministic function of the iterate and the model is fitted to it, the residual is 0 at every step:
    the wealth stays at exactly 1 forever. That is not power, it is the absence of anything to test —
    the mechanism behind the measured peak wealth 1.00."""
    al = Alarm(require_law=False)
    r = RES0
    for _ in range(100):
        nxt = RATE * r                                            # no floor: the frozen rate is exact
        al.update(r * (1 - RATE), r - nxt)
        r = nxt
    assert abs(al.wealth - 1.0) < 1e-12 and al.peak <= 1.0 + 1e-12 and not al.hit


def test_where_the_fallback_does_fire_a_one_line_heuristic_already_stopped():
    """Taken deliberately on the law-free sequence, the alarm crosses 1/alpha only after the residual has
    been sitting on its floor for several sweeps — later than a stagnation rule that reaches the same
    residual. The measured ordering (247 against 104, same final error) reproduces: here 110 against 106."""
    al = Alarm(alpha=ALPHA, require_law=False)
    fired = None
    for i, (predicted, realized) in enumerate(_solver_residuals(), 1):
        al.update(predicted, realized)
        if al.hit and fired is None:
            fired = i
    r, run, stagnated = RES0, 0, None
    for i in range(1, 401):
        nxt = max(RATE * r, FLOOR)
        run = run + 1 if (r - nxt) / r < 1e-3 else 0
        if run >= 5 and stagnated is None:
            stagnated = i
        r = nxt
    assert (stagnated, fired) == (106, 110)
    assert stagnated < fired
    assert r == FLOOR                                             # both stop at the same residual
