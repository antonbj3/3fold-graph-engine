"""Exact ranks change real decisions; unlucky primes never certify singularity."""
from decimal import Decimal
from fractions import Fraction as F
import math

import numpy as np
import pytest

from graph_engine import _exact_rank as exact
from graph_engine.claim_federation import lineage_information
from graph_engine.margin_net import EdgeEstimate, MarginNet
from graph_engine.precision_form import PrecisionForm


def _s4_forbid_float_rank(monkeypatch):
    def _s4_forbidden(*args, **kwargs):
        raise AssertionError("a floating rank operation entered the exact path")
    for name in ("pinv", "svd", "matrix_rank"):
        monkeypatch.setattr(np.linalg, name, _s4_forbidden)


@pytest.mark.parametrize("matrix, rank", [
    ([[1, 2], [2, 4]], 1),
    ([[1, 2, 3], [2, 4, 6]], 1),
    ([[1, 2, 0], [0, 1, 1], [1, 3, 1]], 2),
    ([[1, 2, 0, 3], [0, 1, 1, 2], [1, 3, 1, 5]], 2),
    ([[1, 2], [3, 4], [5, 6]], 2),
    ([[1, 3, 5], [2, 4, 6]], 2),
    ([[1, F(1, 3)], [0, F(1, 10**12)]], 2),
    ([[0, 0], [0, 0]], 0),
    (np.empty((0, 2), dtype=object), 0),
    (np.empty((2, 0), dtype=object), 0),
])
def test_exact_pinv_satisfies_all_four_penrose_identities(matrix, rank, monkeypatch):
    _s4_forbid_float_rank(monkeypatch)
    a = exact._s4_rational_matrix(matrix)
    ap, certificate = exact._s4_pinv(matrix)
    assert certificate.rank == rank
    assert ap.shape == a.T.shape
    assert all(isinstance(x, F) for x in ap.flat)
    assert np.array_equal(a @ ap @ a, a)
    assert np.array_equal(ap @ a @ ap, ap)
    assert np.array_equal(a @ ap, (a @ ap).T)
    assert np.array_equal(ap @ a, (ap @ a).T)


@pytest.mark.parametrize("primes", [(2, 3), exact._S4_PRIMES])
def test_an_unlucky_prime_retries_without_claiming_singularity(primes, monkeypatch):
    monkeypatch.setattr(exact, "_S4_PRIMES", primes)
    _s4_forbid_float_rank(monkeypatch)
    p, q = primes
    ap, certificate = exact._s4_pinv([[1, 1], [1, 1 + p]])
    assert certificate.rank == 2 and certificate.prime_used == q
    assert certificate.rank_over_prime == 2
    assert np.array_equal(ap, [[F(1 + p, p), F(-1, p)], [F(-1, p), F(1, p)]])


@pytest.mark.parametrize("primes", [(2, 3), exact._S4_PRIMES])
def test_two_low_prime_ranks_require_elimination_over_q(primes, monkeypatch):
    monkeypatch.setattr(exact, "_S4_PRIMES", primes)
    _s4_forbid_float_rank(monkeypatch)
    # Both primes divide det, yet the matrix is nonsingular over Q.
    determinant = primes[0] * primes[1]
    ap, certificate = exact._s4_pinv([[1, 1], [1, 1 + determinant]])
    assert certificate.rank_over_prime == 1 and certificate.rank == 2
    assert certificate.rule == "elimination over Q"
    assert np.array_equal(ap, [[F(1 + determinant, determinant), F(-1, determinant)],
                               [F(-1, determinant), F(1, determinant)]])
    _, singular = exact._s4_pinv([[1, 1], [1, 1]])
    assert singular.rank == 1 and singular.rule == "elimination over Q"


@pytest.mark.parametrize("value, expected", [
    (0.1, F(*0.1.as_integer_ratio())),
    (np.float32(0.1), F(*np.float32(0.1).as_integer_ratio())),
    (np.int64(7), F(7)),
    (Decimal("0.1"), F(1, 10)),
    (F(1, 10**400), F(1, 10**400)),
])
def test_exact_conversion_does_not_guess_or_round_rationals(value, expected):
    assert exact._s4_fraction(value) == expected


class _S4Irrational:
    """An explicitly non-rational real, still usable by the existing float API."""
    def __float__(self):
        return math.sqrt(2)


@pytest.mark.parametrize("value", [_S4Irrational(), float("inf"), float("nan"), Decimal("Infinity")])
def test_unsupported_or_nonfinite_entries_do_not_acquire_a_rational_certificate(value):
    assert exact._s4_pinv([[value]]) is None


def _s4_thin_tree(scale=F(1)):
    n = MarginNet()
    theta = F(1, 10**12)
    n.add_sources([{"id": "root", "parent": None, "share": 1 - theta},
                   {"id": "a", "parent": "root", "share": theta},
                   {"id": "b", "parent": "root", "share": theta}])
    n.add_edge("e", ["u", "v"], [{"margin": 0, "sigma": scale, "sources": ["a"]},
                                  {"margin": scale / 100000, "sigma": scale, "sources": ["b"]}])
    return n


@pytest.mark.parametrize("scale", [F(1, 10**6), F(1), F(10**6)])
def test_exact_margin_rank_restores_a_disagreement_hidden_by_rcond(scale, monkeypatch):
    n = _s4_thin_tree(scale)
    old = n.estimate("e")
    assert old.dof == 0 and old.kind == "STRESSED"
    assert n.estimate("e", exact_rank=False) == old
    _s4_forbid_float_rank(monkeypatch)
    new = n.estimate("e", exact_rank=True)
    # For two equal variances, Q = (m1-m2)^2 / (2*(1-rho)) = 50.
    assert isinstance(new, EdgeEstimate)
    assert new.dof == 1 and new.q == 50.0
    assert new.kind == "CONTRADICTION" and new.p_agree < 0.01


def test_exact_margin_copies_retain_the_minimum_norm_model(monkeypatch):
    n = MarginNet()
    n.add_edge("e", ["u", "v"], [{"margin": F(2, 5), "sigma": F(1, 10), "sources": ["r"]},
                                  {"margin": F(3, 5), "sigma": F(1, 5), "sources": ["r"]}])
    _s4_forbid_float_rank(monkeypatch)
    e = n.estimate("e", exact_rank=True)
    assert e.dof == 0 and e.q == 0
    assert e.m == float(F(8, 15)) and e.s == pytest.approx(float(F(1, 6)))
    assert e.kind == "CONTRADICTION"  # incompatible copies are still checked separately


def test_exact_default_lineage_covariance_is_built_before_float_gram_rounding(monkeypatch):
    n = MarginNet()
    n.add_sources([{"id": "mixed", "derives_from": ["a", "b", "c"]}])
    n.add_edge("e", ["u", "v"], [{"margin": F(1, 3), "sigma": F(1, 7), "sources": [x]}
                                  for x in ["mixed", "mixed", "a"]])
    _s4_forbid_float_rank(monkeypatch)
    e = n.estimate("e", exact_rank=True)
    assert e.dof == 1 and e.q == 0.0
    assert e.m == float(F(1, 3)) and e.kind == "OK"
    assert e.n_eff == 3.0


def test_rational_shared_covariance_uses_exact_path_despite_irrational_factors(monkeypatch):
    n = MarginNet()
    n.add_sources([{"id": x, "shares": {"g": F(1, 2)}} for x in ["a", "b"]])
    n.add_edge("e", ["u", "v"], [{"margin": F(1, 5), "sigma": 1, "sources": [x]} for x in ["a", "b"]])
    _s4_forbid_float_rank(monkeypatch)
    e = n.estimate("e", exact_rank=True)
    assert e.dof == 1 and e.q == 0.0
    assert e.s == pytest.approx(math.sqrt(0.75))


def test_irrational_shared_covariance_retains_both_float_rank_operations(monkeypatch):
    n = MarginNet()
    n.add_sources([{"id": "a", "shares": {"g": F(1, 4)}}, {"id": "b", "shares": {"g": F(1, 2)}}])
    n.add_edge("e", ["u", "v"], [{"margin": 0.2, "sigma": 1, "sources": [x]} for x in ["a", "b"]])
    old = n.estimate("e")
    calls = []
    pinv, svd = np.linalg.pinv, np.linalg.svd
    def _s4_pinv_spy(*args, **kwargs):
        calls.append("pinv")
        assert kwargs["rcond"] == 1e-10
        return pinv(*args, **kwargs)
    def _s4_svd_spy(*args, **kwargs):
        calls.append("svd")
        return svd(*args, **kwargs)
    monkeypatch.setattr(np.linalg, "pinv", _s4_pinv_spy)
    monkeypatch.setattr(np.linalg, "svd", _s4_svd_spy)
    assert n.estimate("e", exact_rank=True) == old
    assert calls == ["pinv", "svd"]


@pytest.mark.parametrize("sets, information", [
    ([{"a", "b"}] * 4, 2.0),
    ([{"a", "b"}, {"b", "c"}], float(F(8, 3))),
    ([{"a", "b"}, {"b", "c"}, {"a", "b", "c"}], 3.0),
    ([], 0.0),
    ([{"a"}, {"a"}, {"b"}], 2.0),
])
def test_exact_lineage_returns_the_minimum_norm_information(sets, information, monkeypatch):
    assert lineage_information(sets) == pytest.approx(information)
    _s4_forbid_float_rank(monkeypatch)
    assert lineage_information(sets, exact_rank=True) == information


@pytest.mark.parametrize("as_float", [False, True])
def test_exact_measurement_preserves_a_small_rational_noise_direction(as_float, monkeypatch):
    sigma = [[1, 0], [0, F(1, 10**12)]]
    if as_float:
        sigma = np.asarray(sigma, float)
    old = PrecisionForm.zeros(2).add_measurement(np.eye(2), [1, 2], sigma)
    assert old.J[1, 1] == 0.0
    _s4_forbid_float_rank(monkeypatch)
    f = PrecisionForm.zeros(2)
    assert f.add_measurement(np.eye(2), [1, 2], sigma, exact_rank=True) is f
    assert np.array_equal(f.J, np.diag([1.0, 1e12]))
    assert np.array_equal(f.b, [1.0, 2e12])


def test_exact_measurement_singular_noise_uses_moore_penrose_not_an_arbitrary_solution(monkeypatch):
    _s4_forbid_float_rank(monkeypatch)
    f = PrecisionForm.zeros(2).add_measurement(np.eye(2), [3, 6], [[1, 2], [2, 4]], exact_rank=True)
    assert np.array_equal(f.J, np.array([[1, 2], [2, 4]]) / 25)
    assert np.array_equal(f.b, [0.6, 1.2])


def test_irrational_measurement_noise_retains_the_existing_rcond(monkeypatch):
    calls = []
    pinv = np.linalg.pinv
    def _s4_pinv_spy(*args, **kwargs):
        calls.append(kwargs["rcond"])
        return pinv(*args, **kwargs)
    monkeypatch.setattr(np.linalg, "pinv", _s4_pinv_spy)
    f = PrecisionForm.zeros(1, tol=1e-7).add_measurement([[1]], [2], [[_S4Irrational()]], exact_rank=True)
    assert f.J[0, 0] == pytest.approx(1 / math.sqrt(2))
    assert calls == [1e-7]


def test_exact_margin_keeps_representable_uncertainty_when_inverse_overflows_float():
    n = MarginNet()
    n.add_edge("e", ["u", "v"], [{"margin": F(1, 10**200), "sigma": F(1, 10**200), "sources": ["a"]}])
    e = n.estimate("e", exact_rank=True)
    assert e.m == 1e-200 and e.s == 1e-200 and e.z == 1.0
    assert e.n_eff == 1.0 and e.roots["a"] == 1e-200


def test_exact_tree_effective_count_survives_overflowing_intermediate_information():
    n = MarginNet()
    n.add_sources([{"id": "root", "parent": None, "share": F(1, 2)},
                   {"id": "a", "parent": "root", "share": F(1, 2)},
                   {"id": "b", "parent": "root", "share": F(1, 2)}])
    n.add_edge("e", ["u", "v"], [{"margin": F(1, 10**200), "sigma": F(1, 10**200), "sources": [x]}
                                  for x in ("a", "b")])
    e = n.estimate("e", exact_rank=True)
    assert e.s == pytest.approx(math.sqrt(0.75) * 1e-200, rel=1e-14, abs=0)
    assert e.n_eff == pytest.approx(4 / 3)


def test_exact_measurement_contracts_before_converting_large_inverse():
    f = PrecisionForm.zeros(1)
    f.add_measurement([[1e-200]], [1e-200], [[F(1, 10**400)]], exact_rank=True)
    assert f.J[0, 0] == pytest.approx(1.0)
    assert f.b[0] == pytest.approx(1.0)


@pytest.mark.parametrize("A, y, Sigma, message", [
    ([[1]], [1], [[F(1, 10**400)]], "precision update"),
    ([[1]], [1e200], [[F(1, 10**200)]], "information update"),
])
def test_unrepresentable_exact_measurement_rejects_without_mutation(A, y, Sigma, message):
    f = PrecisionForm.zeros(1)
    f.J[0, 0] = 7.0
    f.b[0] = 3.0
    with pytest.raises(ValueError, match=message):
        f.add_measurement(A, y, Sigma, exact_rank=True)
    assert f.J[0, 0] == 7.0 and f.b[0] == 3.0


@pytest.mark.parametrize('validity, expected', [
    ({}, 'UNDECLARED'), ({'temperature': [0, 1]}, 'IDENTICAL'),
])
def test_exact_estimate_retains_current_validity_metadata(validity, expected):
    n = MarginNet()
    n.add_edge('e', ['u', 'v'], [
        {'margin': F(1, 3), 'sigma': 1, 'sources': [source], 'validity': validity}
        for source in ['a', 'b']
    ])
    assert n.estimate('e', exact_rank=True).validity_relation == expected
