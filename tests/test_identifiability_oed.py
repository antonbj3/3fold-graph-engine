"""identifiability_oed: the E-optimal plateau, and the selection rule that gets off it.

The fixture is a measured one. A contact-solver friction-identification study probed an 8-interface
stack with micro-slip impulses: 8 contacts x 4 tangential directions = 32 candidate probes for 8
friction parameters, each probe sized so that exactly one contact slips. A contact that does not slip
contributes an identically zero column to the Fisher, so each probe's Fisher is rank-1 in the parameter
of the contact it moves, and the accumulated Fisher is diagonal. Two things are taken from that run:

  * the per-parameter information, reconstructed from the reported Cramer-Rao bounds of the selected
    set (F_cc = 1 / CRB_c^2, three significant digits -- which is why the reproductions below are
    asserted to 1 %); the largest, F_00 = 1.83e7, agrees with the separately reported row norm
    ||F e_0|| = 1.822471e7 to 0.2 %;
  * the per-probe cost, the reported probe impulse in N s per interface.

What that run measured about the two rules, and what these tests re-measure:

  sigma_min-only argmax   rank(F) 3/8 after 12 probes, sigma_min = 0 at every step, 1.25150 N s at k = 8
  rank-first per cost     rank(F) 8/8 at k = 8, sigma_min 0 -> 6.327508e5, 0.64906 N s, selection order
                          7, 6, 5, 4, 3, 2, 0, 1 (ascending cost), trace 1.265502e6 / 1.898252e6 /
                          2.282914e6 / 2.531003e6 at k = 9..12

A second study priced the two tie-break objectives as sequential selection rules against the adaptive
optimum (full decision-tree enumeration on the engine's own entropy potential over the same probe set):
sigma_min-gain per cost reaches 0.29-0.61 of the optimum and falls below 1 - 1/e = 0.6321 in 30 of 50
random Dirichlet priors (worst 0.2031); the same cost-aware greedy on the engine's own priced objective
stays at 0.77-0.92 and never falls below 1 - 1/e. Hence the tie-break here is value_bits per cost.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_engine.identifiability_oed import (  # noqa: E402
    _spectrum, greedy_oed, identifiability, value_bits,
)

# measured, per interface c = 0..7: Cramer-Rao bound on mu_c from the selected set, and probe impulse (N s)
CRB = (2.34e-4, 2.00e-4, 2.21e-4, 2.65e-4, 3.31e-4, 4.42e-4, 6.62e-4, 1.26e-3)
IMPULSE = (0.12612, 0.18675, 0.08869, 0.07319, 0.06931, 0.04641, 0.04481, 0.01378)
ASCENDING_COST = [7, 6, 5, 4, 3, 2, 0, 1]                  # the reported selection order
SIGMIN_TRACE = (6.327508e5, 1.265502e6, 1.898252e6, 2.282914e6, 2.531003e6)   # reported, k = 8..12
N_DIR = 4                                                   # tangential directions per contact


def probe_pool():
    """(fishers, costs, contact_of_probe) — the 32-candidate pool, in contact order as it was enumerated."""
    info = [1.0 / c ** 2 for c in CRB]
    fishers, costs, contact = [], [], []
    for c in range(8):
        F = np.zeros((8, 8)); F[c, c] = info[c]
        for _ in range(N_DIR):                              # four directions at one contact: same information,
            fishers.append(F.copy()); costs.append(IMPULSE[c]); contact.append(c)   # same price, same Fisher
    return fishers, costs, contact


# -- the defect ------------------------------------------------------------------------------------

def test_e_optimal_argmax_has_no_gradient_while_the_fisher_is_rank_deficient():
    """Every candidate scores sigma_min = 0 until the last parameter is covered, so the argmax reduces to
    input order: four redundant copies of contact 0, then of contact 1, then of contact 2 — rank 3 of 8
    after twelve probes, and sigma_min never leaves 0. Measured on the pool above; reproduced here."""
    pool, costs, contact = probe_pool()
    idx, trace = greedy_oed(pool, 12, tie_break="sigma_min")
    assert [contact[i] for i in idx] == [0] * 4 + [1] * 4 + [2] * 4
    assert trace == [0.0] * 12
    sigmin, rank, _ = _spectrum(sum(pool[i] for i in idx))
    assert (rank, sigmin) == (3, 0.0)
    assert sum(costs[i] for i in idx[:8]) == pytest.approx(1.25150, rel=1e-3)    # reported 1.25150 N s


# -- the fix ---------------------------------------------------------------------------------------

def test_rank_first_per_cost_buys_one_probe_per_parameter_in_ascending_cost_order():
    """On the plateau the only term that moves is the rank, and dividing by cost orders the picks by price:
    the reported order 7, 6, 5, 4, 3, 2, 0, 1 for 0.64906 N s, rank 8/8, sigma_min 0 -> 6.33e5."""
    pool, costs, contact = probe_pool()
    idx, trace = greedy_oed(pool, 8, costs=costs)
    assert [contact[i] for i in idx] == ASCENDING_COST
    assert sum(costs[i] for i in idx) == pytest.approx(0.64906, rel=1e-4)
    assert trace[:7] == [0.0] * 7                                  # rank K is where sigma_min can first move
    assert trace[7] == pytest.approx(SIGMIN_TRACE[0], rel=0.01)
    assert _spectrum(sum(pool[i] for i in idx))[1] == 8


def test_the_whole_selected_trace_reproduces_the_measured_one():
    """k = 8..12: 6.33e5, 1.27e6, 1.90e6, 2.28e6, 2.53e6 — the extra probes go to the worst-identified
    parameter first (the cheapest interface is also the least loaded, hence the least informative)."""
    pool, costs, contact = probe_pool()
    idx, trace = greedy_oed(pool, 12, costs=costs)
    for k, measured in zip(range(8, 13), SIGMIN_TRACE):
        assert trace[k - 1] == pytest.approx(measured, rel=0.01), f"k = {k}"
    assert [contact[i] for i in idx[8:]] == [7, 7, 7, 6]
    assert sum(costs[i] for i in idx) < 1.25150                    # still under the sigma_min-only k = 8 spend


def test_rank_first_lifts_sigma_min_with_uniform_costs_too():
    """Without prices the rank term alone is enough to leave the plateau: cover every parameter, then rank
    by the priced objective. The cost information only changes the ORDER of the first K picks."""
    pool, _costs, contact = probe_pool()
    idx, trace = greedy_oed(pool, 8)
    assert sorted(contact[i] for i in idx) == list(range(8))
    assert trace[-1] == pytest.approx(SIGMIN_TRACE[0], rel=0.01)


def test_ranking_is_invariant_to_the_units_of_the_fisher():
    """A Fisher in different units is the same Fisher: the ridge in the priced objective is relative to the
    pool's own scale and the rank tolerance is relative to the largest eigenvalue, so scaling the whole pool
    (and the costs) changes the trace by that factor and nothing else."""
    pool, costs, contact = probe_pool()
    idx, trace = greedy_oed(pool, 12, costs=costs)
    big = [1e12 * F for F in pool]
    idx_b, trace_b = greedy_oed(big, 12, costs=[1e-6 * c for c in costs])
    assert [contact[i] for i in idx_b] == [contact[i] for i in idx]
    assert np.allclose(np.array(trace_b), 1e12 * np.array(trace), rtol=1e-9)


# -- the priced objective --------------------------------------------------------------------------

def test_value_bits_is_the_gaussian_information_gain_of_the_added_fisher():
    """1/2 log2 det(I + (F_acc + prior)^-1 F): one bit for doubling the information about one parameter,
    and additive over independent parameters (precision_form's value(h), matrix-valued)."""
    P = np.eye(2)
    F = np.diag([1.0, 0.0])
    assert value_bits(np.zeros((2, 2)), F, P) == pytest.approx(0.5)              # 1/2 log2(1 + 1/1)
    assert value_bits(np.zeros((2, 2)), np.eye(2), P) == pytest.approx(1.0)      # both parameters
    assert value_bits(np.diag([3.0, 0.0]), F, P) == pytest.approx(0.5 * np.log2(5 / 4))


def test_value_bits_and_the_spectrum_fail_closed_on_a_degenerate_observable():
    bad = np.array([[np.nan, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        value_bits(np.zeros((2, 2)), bad, np.eye(2))
    with pytest.raises(ValueError):
        _spectrum(bad)
    pool = [np.diag([1.0, 0.0]), bad, np.diag([0.0, 1.0])]
    idx, trace = greedy_oed(pool, 2)                                             # excluded, never picked
    assert 1 not in idx and len(idx) == 2
    assert identifiability([pool[i] for i in idx])["sigmin"] == pytest.approx(1.0)


def test_greedy_oed_rejects_a_cost_vector_that_is_not_a_price():
    pool, costs, _ = probe_pool()
    with pytest.raises(ValueError):
        greedy_oed(pool, 4, costs=costs[:-1])
    with pytest.raises(ValueError):
        greedy_oed(pool, 4, costs=[0.0] * len(pool))
    with pytest.raises(ValueError):
        greedy_oed(pool, 4, tie_break="whatever")
