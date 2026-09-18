#!/usr/bin/env python3
"""regime_markov: the telegraph reading of the sign, against the hypothesis-enumeration reading it replaces.

The numbers printed here are the ones e28 runs at scale; the assertions are the thresholds those numbers earn.
"""
import copy
import math

import numpy as np
import pytest

from graph_engine.claim_federation import lineage_information
from graph_engine.closed_loop import World
from graph_engine.regime_markov import RegimeMarkov
from graph_engine.regime_posterior import RegimePosterior

K_TEST = 2          # switches enumerated inside a claim block; the tests report tail_mass() for it


def _pair_claims(world: World, pair: int = 0):
    """The world's claims for one pair, grouped as closed_loop groups them (copies are not independent support)."""
    g = {}
    for p, a, b, sg, roots in world.claims:
        if p == pair:
            g.setdefault((round(a, 6), round(b, 6), sg), []).append(roots)
    return [(a, b, sg, lineage_information(rs)) for (a, b, sg), rs in g.items()]


def _readers(world: World, pair: int = 0, **kw):
    rp = RegimePosterior(0.0, 1.0, claim_model="majority", **{k: v for k, v in kw.items() if k != "max_switches_in_block"})
    rm = RegimeMarkov(0.0, 1.0, **kw)
    for a, b, sg, n in _pair_claims(world, pair):
        rp.add_claim(a, b, sg, n_eff=n)
        rm.add_claim(a, b, sg, n_eff=n)
    return rp, rm


def test_lambda_to_zero_leaves_one_probe_speaking_for_the_whole_domain():
    """λ → 0 is the constant-sign limit: one probe of reliability r puts P₊ = r at every x, near and far."""
    for r in (0.6, 0.8, 0.95):
        rm = RegimeMarkov(0.0, 1.0, lam=1e-12)
        rm.add_probe(0.3, 1, r)
        for x in (0.005, 0.3, 0.31, 0.7, 0.999):
            assert abs(rm.p_plus(x) - r) < 1e-9
        rm2 = RegimeMarkov(0.0, 1.0, lam=1e-12)
        rm2.add_probe(0.3, -1, r)
        assert abs(rm2.p_plus(0.9) - (1 - r)) < 1e-9
    # and the switch-count posterior is concentrated on N = 0 there
    assert RegimeMarkov(0.0, 1.0, lam=1e-12).n_transitions_posterior()[0] > 1 - 1e-9


def test_prior_switch_count_is_the_poisson_of_the_rate_it_declares():
    """The discretised chain's prior N is Poisson(λ·(hi−lo)) up to the switch pairs the partition cannot resolve."""
    rm = RegimeMarkov(0.0, 1.0)
    assert rm.lam == pytest.approx(0.4)                    # E[N] = p_flip + 2·p_two = 0.3 + 2·0.05
    pn = rm.n_transitions_posterior()
    pois = np.array([math.exp(-0.4) * 0.4 ** k / math.factorial(k) for k in range(4)])
    # the gap is what the midpoint discretisation cannot see: switches in the two end half-cells (prior mass
    # λ·(hi−lo)/n_grid ≈ 0.008 of a switch) and switch PAIRS between adjacent midpoints (≈ 47·(λd)²/2 ≈ 0.0016).
    assert np.abs(pn[:4] - pois).max() < 8e-3
    assert pn[0] > math.exp(-0.4)                          # and it lands on N = 0, as those two effects predict
    assert abs(pn.sum() - 1.0) < 1e-12


def test_majority_claim_is_not_the_pointwise_product():
    """A + report on [0, 0.5] licenses a path that is − on part of the box, and says nothing beyond it.

    The pointwise reading does neither: it labels every point of the box r-accurately, which is the reading e21
    identified as the loop's over-confidence.
    """
    maj = RegimeMarkov(0.0, 1.0, max_switches_in_block=3)
    pw = RegimeMarkov(0.0, 1.0, claim_model="pointwise")
    for r in (maj, pw):
        r.add_claim(0.0, 0.5, 1, 8.0, 0.9)
    assert maj.p_plus(0.25) > 0.99 and maj.p_plus(0.02) < 0.96        # box edges stay less certain than its middle
    assert pw.p_plus(0.02) > maj.p_plus(0.02) + 0.03                  # pointwise labels the box edge too, and harder
    assert maj.p_plus(0.75) < 0.85 < pw.p_plus(0.75)                  # and outside the box it carries the claim along
    assert maj.tail_mass() < 1e-3


def test_agrees_with_regime_posterior_majority_on_one_transition_worlds():
    """200 random 0/1-transition worlds, the same claims and 10 random probes in both readers.

    The two are different models — RegimePosterior enumerates {0, 1} transitions with prior mass p_flip and softens
    the majority with a logistic (k = 20); the chain keeps every switch count with Poisson weights and takes the
    majority hard — so this is an agreement measurement, not an identity. Measured over 200 worlds:
        expected_error   max |Δ| 0.117, mean 0.015
        p_plus           mean over x of |Δ| per world: max 0.256, mean 0.024
    """
    de, dp = [], []
    xs = np.linspace(0.01, 0.99, 50)
    for s in range(200):
        w = World(n_pairs=1, p_two=0.0, p_transition=0.6, seed=s)
        rp, rm = _readers(w, p_flip=0.6, p_two=0.0, max_switches_in_block=K_TEST)
        rp.p_two = 1e-9                                              # the world has no two-transition pairs
        rng = np.random.default_rng(1000 + s)
        for x in rng.random(10):
            ans = w.probe(0, float(x), 0.9, rng)
            rp.add_probe(float(x), ans, 0.9)
            rm.add_probe(float(x), ans, 0.9)
        de.append(abs(rp.expected_error() - rm.expected_error()))
        dp.append(float(np.mean([abs(rp.p_plus(x) - rm.p_plus(x)) for x in xs])))
    de, dp = np.array(de), np.array(dp)
    print(f"\nvs RegimePosterior(majority), 200 worlds: expected_error max |Δ| {de.max():.4f} mean {de.mean():.4f}; "
          f"p_plus mean|Δ| per world max {dp.max():.4f} mean {dp.mean():.4f}")
    assert de.max() < 0.15 and de.mean() < 0.03                      # stated tolerance
    assert dp.max() < 0.30 and dp.mean() < 0.05


def test_two_transitions_are_detected_after_twelve_of_its_own_probes():
    """Two-transition pairs, 12 probes placed by the chain's own best_probe, both readers fed the same answers.

    P(N ≥ 2) is a moment of one process here, not the mass of a separate family with prior p_two. Measured on 20
    worlds: the chain flags 16, RegimePosterior.collision() on the same evidence flags 16 — the reading is not
    what was missing in e21, the PLACEMENT is, and best_probe on the chain buys it.
    """
    pm, pr = [], []
    for s in range(20):
        w = World(n_pairs=1, p_two=1.0, p_transition=0.0, seed=s)
        assert len(w.transitions[0]) == 2
        rp, rm = _readers(w, max_switches_in_block=K_TEST)
        rng = np.random.default_rng(2000 + s)
        for _ in range(12):
            x, _gain = rm.best_probe(0.95)
            ans = w.probe(0, x, 0.95, rng)
            rm.add_probe(x, ans, 0.95)
            rp.add_probe(x, ans, 0.95)
        pm.append(rm.collision()["p_ge_two"])
        pr.append(rp.collision()["p_two_transitions"])
    pm, pr = np.array(pm), np.array(pr)
    print(f"\ntwo-transition pairs (20): chain P(N≥2) > 0.5 in {int((pm > 0.5).sum())}, mean {pm.mean():.3f}; "
          f"RegimePosterior.collision() > 0.5 in {int((pr > 0.5).sum())}, mean {pr.mean():.3f}")
    assert (pm > 0.5).sum() > 10                                     # a majority of the cases
    assert pm.mean() > 0.6


def test_probe_value_is_the_realized_expectation_and_never_negative():
    """The fixed partition makes the posterior a martingale: best_probe's number is what the two answers realize.

    Same check the RegimePosterior tests run (test_layer_e: fixed_hypothesis_space_probe_value...).
    """
    for claims, pot in [([(0.343, 0.393, 1), (0.015, 0.075, 1), (0.257, 0.337, 1)], "error"),
                        ([(0.4, 0.5, 1)], "entropy"),
                        ([(0.0, 0.2, 1), (0.8, 1.0, -1)], "entropy")]:
        rm = RegimeMarkov(0.0, 1.0, potential=pot, max_switches_in_block=K_TEST)
        for a, b, sg in claims:
            rm.add_claim(a, b, sg, 3.0)
        u0 = rm.potential_value()
        for x in np.linspace(0.02, 0.98, 9):
            after = 0.0
            p = rm.p_plus(x)
            for sg, po in ((1, p * 0.95 + (1 - p) * 0.05), (-1, (1 - p) * 0.95 + p * 0.05)):
                q = copy.deepcopy(rm); q.add_probe(x, sg, 0.95); after += po * q.potential_value()
            assert u0 - after > -1e-9                                 # Jensen: no probe raises it in expectation
        xb, predicted = rm.best_probe(0.95)
        p = rm.p_plus(xb); after = 0.0
        for sg, po in ((1, p * 0.95 + (1 - p) * 0.05), (-1, (1 - p) * 0.95 + p * 0.05)):
            q = copy.deepcopy(rm); q.add_probe(xb, sg, 0.95); after += po * q.potential_value()
        assert np.isclose(predicted, u0 - after, atol=1e-9)
        assert predicted >= 0.0


def test_reads_are_consistent_and_guarded():
    rm = RegimeMarkov(0.0, 1.0, max_switches_in_block=K_TEST)
    rm.add_claim(0.0, 0.4, 1, 3.0)
    rm.add_claim(0.6, 1.0, -1, 3.0)
    pp = rm.p_plus_cells()
    assert pp.min() >= 0.0 and pp.max() <= 1.0
    assert rm.expected_error() <= rm.potential_value()               # min(p,1−p) ≤ binary entropy in bits
    assert abs(rm.n_transitions_posterior().sum() - 1.0) < 1e-12
    assert rm.collision()["bayes_factor_two_vs_one"] > 0
    assert 0.0 <= rm.tail_mass() < 0.02
    with pytest.raises(ValueError):
        rm.add_claim(0.0, 1.0, 1, 1.0, reliability=1.0)
    with pytest.raises(ValueError):
        rm.add_probe(0.5, 1, reliability=0.4)
    with pytest.raises(ValueError):
        RegimeMarkov(1.0, 0.0)
