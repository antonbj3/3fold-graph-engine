"""Two-stage (Dorfman) pooled screening for candidate-link ("throw") testing.

Setting: n candidate items, each independently true with prevalence p. A
noisy judge can be asked "is at least one of these s items true?" for a
pool of size s (sensitivity se = P(pool test positive | pool contains at
least one true item); specificity sp = P(pool test negative | pool
contains no true item)). Classical two-stage group testing (Dorfman 1943):
partition items into pools of size s, test each pool once; every item in a
pool that tests positive is then tested individually (same noisy test,
applied to a pool of size 1). Items in a pool that tests negative are all
declared negative, with no further testing.

Assumptions (stated explicitly, all required for the closed forms below):
  1. Items are independent Bernoulli(p) draws (i.i.d. truth labels).
  2. Test errors are independent across all pool and individual tests.
  3. "No dilution": the pool-level sensitivity does not depend on how many
     true items k the pool holds (only on the pool size s). This is a
     simplification, not a claim that real dilution effects are absent.
  4. Optional dilution model: se_pool(s) = se * s**(-dilution), dilution
     >= 0, default 0 (no dilution, se_pool(s) = se for all s). This
     multiplies the *pool-level* sensitivity only; individual (stage-2,
     s=1) tests always use se_pool(1) = se, i.e. the plain single-item
     sensitivity. Specificity is not diluted (a pool with zero true items
     is symmetric in its false members, so sp is unaffected).

Derivation of expected_tests_per_item (s > 1):
  Stage 1 costs 1 pool test per s items: 1/s tests per item.
  Stage 2 costs exactly 1 additional individual test for every item whose
  pool tested positive. By exchangeability, an item's own pool tests
  positive with the same (unconditional) probability as any other pool:
    P(pool positive) = P(pool has >=1 true) * se_pool(s)
                      + P(pool has 0 true)  * (1 - sp)
                      = (1 - (1-p)**s) * se_pool(s) + (1-p)**s * (1 - sp)
  so
    E[tests/item] = 1/s + P(pool positive).
  At s = 1 there is no separate pooling stage: exactly one test is run
  per item, so E[tests/item] = 1 by definition (not 1/1 + P(test
  positive), which would double-count).
  Noiseless check (se=sp=1, dilution=0): P(pool positive) = 1-(1-p)**s,
  giving 1/s + 1 - (1-p)**s, the standard noiseless Dorfman formula.

Derivation of two_stage_sensitivity (probability a truly-true item ends
classified positive), s > 1:
  The item's own pool necessarily contains >= 1 true item, so the pool
  test is positive with probability se_pool(s). If positive, the item is
  retested individually and classified positive with probability
  se_pool(1) = se. If the pool test is negative, the item is declared
  negative (no retest) regardless of dilution.
    sensitivity(s) = se_pool(s) * se
  At s = 1: sensitivity = se (the item's own single test).

Derivation of two_stage_specificity (probability a truly-false item i ends
classified negative), s > 1:
  Condition on the other s-1 pool members. Let q = (1-p)**(s-1) be the
  probability all of them are also false (pool truly negative overall).
    If pool truly negative (prob q):
      pool test negative (prob sp) -> item declared negative directly;
      pool test positive (prob 1-sp, false alarm) -> item retested
        individually, classified negative with probability sp.
      P(item negative | pool truly negative) = sp + (1-sp)*sp
    If pool truly positive due to the other members (prob 1-q):
      pool test positive (prob se_pool(s)) -> item retested individually,
        classified negative with probability sp;
      pool test negative (prob 1 - se_pool(s), a missed detection) -> the
        whole pool, including item i, is declared negative.
      P(item negative | pool truly positive) = se_pool(s)*sp + (1-se_pool(s))
    specificity(s) = q*(sp + (1-sp)*sp) + (1-q)*(se_pool(s)*sp + 1-se_pool(s))
  At s = 1: specificity = sp.

Not covered: adaptive / multi-stage (e.g. binary-splitting or Bayesian
adaptive) group-testing schemes; correlated test errors (e.g. a judge that
is systematically worse on some item types); and, most importantly,
whether a noisy judge can in fact answer a pooled "does this set contain a
true item" question with a well-defined, stable (se, sp) at all -- that is
an empirical question about the judge, not addressed by this module.

References:
  Dorfman, R. (1943). "The Detection of Defective Members of Large
  Populations." Annals of Mathematical Statistics, 14(4): 436-440.
  Aldridge, M., Johnson, O., Scarlett, J. (2019). "Group Testing: An
  Information Theory Perspective." Foundations and Trends in
  Communications and Information Theory, 15(3-4): 196-392.
"""

from __future__ import annotations

import numpy as np


def _pool_sensitivity(se: float, s: int, dilution: float) -> float:
    """se_pool(s) = se * s**(-dilution); se_pool(1) = se for any dilution."""
    return se * (s ** (-dilution))


def expected_tests_per_item(
    p: float, s: int, se: float = 1.0, sp: float = 1.0, dilution: float = 0.0
) -> float:
    """Expected number of tests per item under two-stage Dorfman pooling.

    s = 1 means individual testing only: exactly 1 test per item.
    """
    if s < 1:
        raise ValueError("s must be >= 1")
    if s == 1:
        return 1.0
    se_s = _pool_sensitivity(se, s, dilution)
    p_pool_true = 1.0 - (1.0 - p) ** s
    p_pool_positive = p_pool_true * se_s + (1.0 - p_pool_true) * (1.0 - sp)
    return 1.0 / s + p_pool_positive


def two_stage_sensitivity(
    p: float, s: int, se: float = 1.0, sp: float = 1.0, dilution: float = 0.0
) -> float:
    """Probability a truly-true item ends classified positive."""
    if s < 1:
        raise ValueError("s must be >= 1")
    if s == 1:
        return se
    se_s = _pool_sensitivity(se, s, dilution)
    return se_s * se


def two_stage_specificity(
    p: float, s: int, se: float = 1.0, sp: float = 1.0, dilution: float = 0.0
) -> float:
    """Probability a truly-false item ends classified negative."""
    if s < 1:
        raise ValueError("s must be >= 1")
    if s == 1:
        return sp
    se_s = _pool_sensitivity(se, s, dilution)
    q = (1.0 - p) ** (s - 1)
    neg_given_pool_neg = sp + (1.0 - sp) * sp
    neg_given_pool_pos_due_to_others = se_s * sp + (1.0 - se_s)
    return q * neg_given_pool_neg + (1.0 - q) * neg_given_pool_pos_due_to_others


def best_pool_size(
    p: float,
    se: float = 1.0,
    sp: float = 1.0,
    dilution: float = 0.0,
    s_max: int = 64,
    min_sensitivity: float = 0.0,
):
    """Pool size in 1..s_max minimizing expected tests per item.

    Subject to two_stage_sensitivity(p, s, se, sp, dilution) >= min_sensitivity.
    Returns (s, tests_per_item). s = 1 always satisfies the constraint at
    least as well as any s > 1 (sensitivity is maximized at s = 1), so a
    tight enough min_sensitivity forces s = 1.
    """
    if s_max < 1:
        raise ValueError("s_max must be >= 1")
    best = None
    for s in range(1, s_max + 1):
        sens = two_stage_sensitivity(p, s, se, sp, dilution)
        if sens < min_sensitivity - 1e-12:
            continue
        t = expected_tests_per_item(p, s, se, sp, dilution)
        if best is None or t < best[1]:
            best = (s, t)
    if best is None:
        # Nothing meets the constraint (possible only if se itself is
        # below min_sensitivity); fall back to s = 1, the most sensitive
        # option available.
        best = (1, expected_tests_per_item(p, 1, se, sp, dilution))
    return best


def simulate(
    p: float,
    s: int,
    se: float = 1.0,
    sp: float = 1.0,
    dilution: float = 0.0,
    n_items: int = 100_000,
    seed: int = 0,
) -> dict:
    """Monte Carlo simulation of two-stage Dorfman pooling.

    Returns a dict with empirical 'tests_per_item', 'sensitivity',
    'specificity'. Items are padded up to a multiple of s so pools are
    complete; padding items are simulated but excluded from all reported
    statistics.
    """
    if s < 1:
        raise ValueError("s must be >= 1")
    rng = np.random.default_rng(seed)

    if s == 1:
        truth = rng.random(n_items) < p
        u = rng.random(n_items)
        classified_pos = np.where(truth, u < se, u < (1.0 - sp))
        tests = np.ones(n_items)
    else:
        n_pools = -(-n_items // s)  # ceil division
        padded = n_pools * s
        truth = np.zeros(padded, dtype=bool)
        truth[:n_items] = rng.random(n_items) < p

        pools = truth.reshape(n_pools, s)
        pool_has_true = pools.any(axis=1)
        se_s = se * (s ** (-dilution))

        u_pool = rng.random(n_pools)
        pool_test_pos = np.where(pool_has_true, u_pool < se_s, u_pool < (1.0 - sp))
        pool_test_pos_expanded = np.repeat(pool_test_pos, s)

        u_indiv = rng.random(padded)
        indiv_pos = np.where(truth, u_indiv < se, u_indiv < (1.0 - sp))

        classified_pos_padded = np.where(pool_test_pos_expanded, indiv_pos, False)
        tests_padded = 1.0 / s + pool_test_pos_expanded.astype(float)

        truth = truth[:n_items]
        classified_pos = classified_pos_padded[:n_items]
        tests = tests_padded[:n_items]

    true_mask = truth
    false_mask = ~truth
    sensitivity = float(classified_pos[true_mask].mean()) if true_mask.any() else float("nan")
    specificity = (
        float((~classified_pos[false_mask]).mean()) if false_mask.any() else float("nan")
    )
    return {
        "tests_per_item": float(tests.mean()),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }
