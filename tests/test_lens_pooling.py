"""Tests for graph_engine.lens_pooling. Synthetic data only, fixed seeds."""

import numpy as np
import pytest

from graph_engine.lens_pooling import (
    StratifiedPlatt,
    balanced_lenses,
    effective_lenses,
    error_correlation,
    logit,
    participation_ratio,
    pool,
    sigmoid,
    tempered_weight,
)


# ---------------------------------------------------------------------------
# (a) StratifiedPlatt recovers a lens that is reliably wrong on one stratum
# ---------------------------------------------------------------------------

def _make_stratum_data(rng, n, sign0, sign_other, noise_scale):
    y = rng.integers(0, 2, size=n).astype(bool)
    y_signed = 2 * y.astype(float) - 1.0
    logit0 = sign0 * 3.0 * y_signed + rng.normal(scale=noise_scale, size=n)
    logit1 = sign_other * 1.0 * y_signed + rng.normal(scale=noise_scale, size=n)
    logit2 = sign_other * 1.0 * y_signed + rng.normal(scale=noise_scale, size=n)
    P = np.stack([sigmoid(logit0), sigmoid(logit1), sigmoid(logit2)], axis=1)
    return P, y


def test_stratified_platt_recovers_inverted_lens():
    rng = np.random.default_rng(0)
    n = 300

    # Stratum 0: all three lenses point the right way.
    P0_train, y0_train = _make_stratum_data(rng, n, sign0=1.0, sign_other=1.0, noise_scale=1.0)
    P0_test, y0_test = _make_stratum_data(rng, n, sign0=1.0, sign_other=1.0, noise_scale=1.0)

    # Stratum 1: lens 0 is reliably wrong (inverted, and dominant in magnitude).
    P1_train, y1_train = _make_stratum_data(rng, n, sign0=-1.0, sign_other=1.0, noise_scale=1.0)
    P1_test, y1_test = _make_stratum_data(rng, n, sign0=-1.0, sign_other=1.0, noise_scale=1.0)

    P_train = np.concatenate([P0_train, P1_train], axis=0)
    y_train = np.concatenate([y0_train, y1_train], axis=0)
    strata_train = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    P_test = np.concatenate([P0_test, P1_test], axis=0)
    y_test = np.concatenate([y0_test, y1_test], axis=0)
    strata_test = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    # Raw pooling: sum of raw logits, no calibration.
    raw_L_test = logit(P_test)
    raw_pred = pool(raw_L_test) > 0.5
    raw_acc = (raw_pred == y_test).mean()

    # Calibrated pooling.
    platt = StratifiedPlatt(min_items=8).fit(P_train, y_train, strata_train)
    cal_L_test = platt.transform(P_test, strata_test)
    cal_pred = pool(cal_L_test) > 0.5
    cal_acc = (cal_pred == y_test).mean()

    assert cal_acc > raw_acc + 0.1, (raw_acc, cal_acc)

    model = platt.models_[(1, 0)]
    assert model is not None
    assert model.coef_[0, 0] < 0.0


# ---------------------------------------------------------------------------
# (b) error_correlation handles a constant column
# ---------------------------------------------------------------------------

def test_error_correlation_constant_column_no_nan():
    y = np.array([True, False, True, False, True, False, True, False])
    # Lens 0/1 noisy-ish, lens 2 always correct -> constant (all-zero) error column.
    L0 = np.array([2, -2, 2, -2, -2, 2, 2, -2], dtype=float)
    L1 = np.array([1, -1, -1, 1, 1, -1, 1, -1], dtype=float)
    L2 = np.where(y, 5.0, -5.0)
    L = np.stack([L0, L1, L2], axis=1)

    C = error_correlation(L, y)

    assert not np.any(np.isnan(C))
    assert C.shape == (3, 3)
    assert C[2, 0] == 0.0
    assert C[2, 1] == 0.0
    assert C[0, 2] == 0.0
    np.testing.assert_allclose(np.diag(C), 1.0)


# ---------------------------------------------------------------------------
# (c) effective_lenses / participation_ratio closed forms for equicorrelation
# ---------------------------------------------------------------------------

def _equicorrelated(k, r):
    C = np.full((k, k), r, dtype=float)
    np.fill_diagonal(C, 1.0)
    return C


@pytest.mark.parametrize("r", [0.5, 0.0, -0.1])
def test_equicorrelated_closed_forms(r):
    k = 4
    C = _equicorrelated(k, r)

    expected_eff = k / (1.0 + (k - 1) * r)
    expected_pr = k / (1.0 + (k - 1) * r ** 2)

    assert effective_lenses(C) == pytest.approx(expected_eff, rel=1e-9)
    assert participation_ratio(C) == pytest.approx(expected_pr, rel=1e-9)


def test_anticorrelation_sign_sensitivity():
    k = 4
    r = -0.1
    C = _equicorrelated(k, r)
    assert effective_lenses(C) > k
    assert participation_ratio(C) < k


# ---------------------------------------------------------------------------
# (d) tempered pooling beats naive pooling under correlated errors
# ---------------------------------------------------------------------------

def _simulate_pool(rng, n, k, rho, indiv_noise, signal=1.0):
    y = rng.integers(0, 2, size=n).astype(bool)
    y_signed = 2 * y.astype(float) - 1.0
    shared = rng.normal(size=n)
    P = np.zeros((n, k))
    for lens in range(k):
        noise = rng.normal(scale=indiv_noise, size=n)
        raw_logit = signal * y_signed + rho * shared + noise
        P[:, lens] = sigmoid(raw_logit)
    return P, y


def test_tempered_pooling_beats_naive_under_correlated_errors():
    rng = np.random.default_rng(1)
    n, k = 2000, 8
    P, y = _simulate_pool(rng, n, k, rho=2.0, indiv_noise=0.5)

    L = logit(P)
    naive = pool(L, weight=1.0)
    C = error_correlation(L, y)
    w = tempered_weight(C)
    tempered = pool(L, weight=w)

    brier_naive = np.mean((naive - y.astype(float)) ** 2)
    brier_tempered = np.mean((tempered - y.astype(float)) ** 2)

    assert w < 1.0
    assert brier_tempered < brier_naive - 0.005, (brier_naive, brier_tempered, w)


def test_tempered_pooling_not_much_worse_under_independent_errors():
    rng = np.random.default_rng(2)
    n, k = 2000, 8
    P, y = _simulate_pool(rng, n, k, rho=0.0, indiv_noise=1.0)

    L = logit(P)
    naive = pool(L, weight=1.0)
    C = error_correlation(L, y)
    w = tempered_weight(C)
    tempered = pool(L, weight=w)

    brier_naive = np.mean((naive - y.astype(float)) ** 2)
    brier_tempered = np.mean((tempered - y.astype(float)) ** 2)

    assert brier_tempered < brier_naive + 0.01, (brier_naive, brier_tempered, w)


# ---------------------------------------------------------------------------
# (e) balanced_lenses: balance, orthogonality, exact bias cancellation
# ---------------------------------------------------------------------------

def test_balanced_lenses_balance_and_orthogonality():
    factors = {"option_order": 2, "polarity": 2, "position": 2}
    rows = balanced_lenses(factors)

    assert len(rows) == 4
    for name in factors:
        levels = [row[name] for row in rows]
        assert levels.count(0) == levels.count(1) == 2

    names = list(factors.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            fi, fj = names[i], names[j]
            combos = [(row[fi], row[fj]) for row in rows]
            for combo in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                assert combos.count(combo) == 1


def _bias_sum(rows, biases):
    total = 0.0
    for row in rows:
        for factor, level in row.items():
            s = 1.0 if level == 0 else -1.0
            total += biases[factor] * s
    return total


def test_balanced_lenses_cancel_additive_bias_exactly():
    factors = {"option_order": 2, "polarity": 2, "position": 2}
    rows = balanced_lenses(factors)
    biases = {"option_order": 1.7, "polarity": -0.9, "position": 3.3}

    total = _bias_sum(rows, biases)
    assert abs(total) < 1e-12

    unbalanced_rows = [rows[0], rows[0], rows[1], rows[2]]
    unbalanced_total = _bias_sum(unbalanced_rows, biases)
    assert abs(unbalanced_total) > 1e-6
