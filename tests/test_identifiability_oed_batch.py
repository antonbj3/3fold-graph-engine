"""greedy_oed(method="batch"): stacked LAPACK keys give the loop's picks and traces bit for bit."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_engine.identifiability_oed import greedy_oed  # noqa: E402


def random_pool(seed):
    rng = np.random.default_rng(seed)
    K = int(rng.choice([1, 2, 5, 8, 16]))
    P = int(rng.choice([1, 3, 12, 40]))
    pool = []
    for _ in range(P):
        kind = rng.integers(0, 4)
        if kind == 0:                                   # rank-1 on one parameter (the contact-solver shape)
            F = np.zeros((K, K)); c = rng.integers(0, K); F[c, c] = 10 ** rng.uniform(-3, 7)
        elif kind == 1:                                 # low-rank Jacobian Fisher
            J = rng.standard_normal((int(rng.integers(1, 3)), K)) * (rng.random(K) < 0.4); F = J.T @ J
        elif kind == 2 and pool:                        # an exact duplicate: ties the loop must break the same way
            F = pool[int(rng.integers(0, len(pool)))].copy()
        else:
            J = rng.standard_normal((K, K)); F = J @ J.T
        pool.append(F)
    if seed % 7 == 3:
        pool[0] = pool[0].copy(); pool[0][0, 0] = np.nan  # a degenerate candidate: dropped by both
    costs = None if seed % 2 else list(rng.choice([0.5, 1.0, 2.0], size=P))
    base = None if seed % 3 else [pool[-1]]
    return pool, costs, base, K


@pytest.mark.parametrize("seed", range(40))
def test_batch_equals_loop_bit_for_bit(seed):
    pool, costs, base, K = random_pool(seed)
    for k in (1, len(pool)):
        for prior in (1e-9, 1e-3):
            out = []
            for m in ("direct", "batch"):
                try:
                    out.append(repr(greedy_oed(pool, k, base=base, costs=costs, prior=prior, method=m)))
                except ValueError as e:                 # a degenerate base fails closed the same way on both
                    out.append("ValueError: " + str(e))
            assert out[0] == out[1], (seed, k, prior)


def test_the_measured_contact_solver_pool_is_reproduced():
    CRB = (2.34e-4, 2.00e-4, 2.21e-4, 2.65e-4, 3.31e-4, 4.42e-4, 6.62e-4, 1.26e-3)
    IMP = (0.12612, 0.18675, 0.08869, 0.07319, 0.06931, 0.04641, 0.04481, 0.01378)
    pool, costs = [], []
    for c in range(8):
        F = np.zeros((8, 8)); F[c, c] = 1.0 / CRB[c] ** 2
        for _ in range(4):
            pool.append(F.copy()); costs.append(IMP[c])
    a = greedy_oed(pool, 12, costs=costs)
    assert a == greedy_oed(pool, 12, costs=costs, method="batch")


def test_sigma_min_branch_and_bad_method():
    pool, costs, base, K = random_pool(5)
    assert greedy_oed(pool, 3, tie_break="sigma_min", method="batch") == greedy_oed(pool, 3, tie_break="sigma_min")
    with pytest.raises(ValueError):
        greedy_oed(pool, 3, method="gpu")
