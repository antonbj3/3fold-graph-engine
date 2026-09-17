import math

import numpy as np
import pytest

from graph_engine.pooled_screening import (
    best_pool_size,
    expected_tests_per_item,
    simulate,
    two_stage_sensitivity,
    two_stage_specificity,
)


def _brute_force_best(p, s_max=64):
    best = None
    for s in range(1, s_max + 1):
        t = expected_tests_per_item(p, s)
        if best is None or t < best[1]:
            best = (s, t)
    return best


def test_noiseless_closed_form():
    for p in (0.03, 0.05, 0.1):
        for s in (2, 3, 5, 10, 20):
            expected = 1.0 / s + 1.0 - (1.0 - p) ** s
            got = expected_tests_per_item(p, s, se=1.0, sp=1.0, dilution=0.0)
            assert got == pytest.approx(expected, rel=1e-12)
    # s = 1 is individual testing: exactly 1.
    assert expected_tests_per_item(0.03, 1) == pytest.approx(1.0)


@pytest.mark.parametrize("p", [0.03, 0.05])
def test_best_pool_size_noiseless_matches_brute_force_and_saves(p):
    got_s, got_t = best_pool_size(p, se=1.0, sp=1.0, dilution=0.0, s_max=64)
    bf_s, bf_t = _brute_force_best(p, s_max=64)
    assert got_s == bf_s
    assert got_t == pytest.approx(bf_t, rel=1e-12)
    # Savings vs individual testing (1 test/item) should exceed 50%.
    savings = 1.0 - got_t
    assert savings > 0.5


@pytest.mark.parametrize(
    "p,s,se,sp,dilution",
    [
        (0.04, 8, 0.9, 0.95, 0.0),
        (0.05, 10, 0.85, 0.9, 0.3),
    ],
)
def test_monte_carlo_matches_closed_forms(p, s, se, sp, dilution):
    n_items = 400_000
    result = simulate(p, s, se, sp, dilution, n_items=n_items, seed=12345)

    exp_t = expected_tests_per_item(p, s, se, sp, dilution)
    exp_sens = two_stage_sensitivity(p, s, se, sp, dilution)
    exp_spec = two_stage_specificity(p, s, se, sp, dilution)

    n_pools = math.ceil(n_items / s)
    # tests_per_item variance is small; use a generous fixed tolerance
    # driven by pool-level test count.
    se_t = 1.0 / math.sqrt(n_pools)
    assert result["tests_per_item"] == pytest.approx(exp_t, abs=6 * se_t)

    n_true = n_items * p
    se_sens = math.sqrt(exp_sens * (1 - exp_sens) / max(n_true, 1))
    assert result["sensitivity"] == pytest.approx(exp_sens, abs=6 * se_sens + 1e-3)

    n_false = n_items * (1 - p)
    se_spec = math.sqrt(exp_spec * (1 - exp_spec) / max(n_false, 1))
    assert result["specificity"] == pytest.approx(exp_spec, abs=6 * se_spec + 1e-3)


def test_false_positive_pools_cost_extra_tests_with_tiny_prevalence():
    p = 0.001
    s = 8
    noiseless = expected_tests_per_item(p, s, se=1.0, sp=1.0, dilution=0.0)
    with_fp_pools = expected_tests_per_item(p, s, se=1.0, sp=0.8, dilution=0.0)
    assert with_fp_pools > noiseless


def test_min_sensitivity_constraint_respected_and_can_force_s1():
    p, se, sp, dilution = 0.04, 0.8, 0.95, 0.0
    # Unconstrained optimum should use a pool size > 1.
    s_unconstrained, _ = best_pool_size(p, se, sp, dilution, s_max=64, min_sensitivity=0.0)
    assert s_unconstrained > 1

    # A sensitivity floor between se**2 (max sensitivity for s>1) and se
    # (sensitivity at s=1) cannot be met by any s > 1, forcing s = 1.
    floor = 0.5 * (se * se + se)
    s_forced, t_forced = best_pool_size(p, se, sp, dilution, s_max=64, min_sensitivity=floor)
    assert s_forced == 1
    assert t_forced == pytest.approx(1.0)

    # Every accepted s must actually satisfy the constraint (feasible cases
    # only: infeasible floors above se fall back to the best-effort s=1).
    for min_sens in (0.0, 0.5, se * se + 1e-6):
        s, _ = best_pool_size(p, se, sp, dilution, s_max=64, min_sensitivity=min_sens)
        assert two_stage_sensitivity(p, s, se, sp, dilution) >= min_sens - 1e-9
