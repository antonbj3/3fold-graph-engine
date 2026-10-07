"""Opt-in rational rank and Moore–Penrose algebra; no numerical rank decisions.

Finite floats mean their exact binary values, not guessed nearby fractions.
Unsupported (including irrational symbolic) entries return None for the caller's
existing floating path. A deficient modular rank is only a lower bound: after
trying both primes, a negative/full-rank decision is made by elimination over Q.
"""
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from math import isfinite, isqrt, lcm
from numbers import Rational

import numpy as np

from ._rational_interface import rref as _s4_rref, solve_right as _s4_solve_right

_S4_PRIMES = (1_000_000_007, 1_000_000_009)


def _s4_fraction(value):
    if isinstance(value, Rational):
        return Fraction(int(value.numerator), int(value.denominator))
    if isinstance(value, np.integer):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating, Decimal)):
        finite = value.is_finite() if isinstance(value, Decimal) else np.isfinite(value)
        if not finite:
            return None
        return Fraction(*value.as_integer_ratio())
    return None


def _s4_rational_matrix(matrix):
    a = np.asarray(matrix, dtype=object)
    if a.ndim != 2:
        return None
    q = np.empty(a.shape, dtype=object)
    for ij in np.ndindex(a.shape):
        value = _s4_fraction(a[ij])
        if value is None:
            return None
        q[ij] = value
    return q


def _s4_sqrt_fraction(value):
    """Return the rational square root, or None; never approximate a radical."""
    n, d = isqrt(value.numerator), isqrt(value.denominator)
    return Fraction(n, d) if n * n == value.numerator and d * d == value.denominator else None


def _s4_float(value, label):
    """Convert an exact result only when its public float is finite."""
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"exact {label} is outside finite float range") from exc
    if not isfinite(result):
        raise ValueError(f"exact {label} is outside finite float range")
    return result


def _s4_sqrt_float(value, label):
    """Take the square root before float conversion, avoiding variance underflow."""
    if value < 0:
        raise ValueError(f"exact {label} has negative variance")
    with localcontext() as ctx:
        ctx.prec = 50
        result = Decimal(value.numerator).sqrt() / Decimal(value.denominator).sqrt()
    converted = _s4_float(result, label)
    if value > 0 and converted == 0:
        raise ValueError(f"exact {label} is below finite float resolution")
    return converted


def _s4_integer_rows(a):
    # Multiplication of each row by a nonzero denominator preserves rank over Q.
    out = []
    for row in a:
        scale = lcm(*(x.denominator for x in row))
        out.append([x.numerator * (scale // x.denominator) for x in row])
    return out


def _s4_rank_mod_prime(integer_rows, n_columns, prime):
    a = [[x % prime for x in row] for row in integer_rows]
    rank = 0
    for col in range(n_columns):
        pivot = next((i for i in range(rank, len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        inv = pow(a[rank][col], -1, prime)
        a[rank] = [(x * inv) % prime for x in a[rank]]
        for i in range(rank + 1, len(a)):
            factor = a[i][col]
            if factor:
                a[i] = [(x - factor * y) % prime for x, y in zip(a[i], a[rank])]
        rank += 1
        if rank == len(a):
            break
    return rank


@dataclass(frozen=True)
class _S4RankCertificate:
    rank: int
    prime_used: int | None
    rank_over_prime: int
    n_columns: int
    rule: str


def _s4_rank(a):
    _, n_columns = a.shape
    if min(a.shape) == 0:
        return _S4RankCertificate(0, None, 0, n_columns, "empty")
    h = _s4_integer_rows(a)
    for prime in _S4_PRIMES:
        rp = _s4_rank_mod_prime(h, n_columns, prime)
        if rp == min(a.shape):
            return _S4RankCertificate(rp, prime, rp, n_columns, "full rank over prime field")
    # Neither low modular result proves deficiency, even if they agree.
    _, pivots = _s4_rref(a.tolist())
    return _S4RankCertificate(len(pivots), prime, rp, n_columns, "elimination over Q")


def _s4_solve(a, b):
    return np.asarray(_s4_solve_right(a.tolist(), b.tolist()), dtype=object)


def _s4_pinv(matrix):
    """(Fraction-valued A+, rank certificate), or None for unsupported entries.

    A full-rank factorization A = C R gives
    A+ = R^T (R R^T)^-1 (C^T C)^-1 C^T. All inverses and zero decisions
    are exact. Conversion to the public float return types belongs to callers.
    """
    a = _s4_rational_matrix(matrix)
    if a is None:
        return None
    certificate = _s4_rank(a)
    n_rows, n_columns = a.shape
    rank = certificate.rank
    if rank == 0:
        return np.full((n_columns, n_rows), Fraction(0), dtype=object), certificate
    if rank == n_rows == n_columns:
        identity = np.array([[Fraction(int(i == j)) for j in range(n_rows)] for i in range(n_rows)], dtype=object)
        inverse = _s4_solve(a, identity)
    elif rank == n_columns:
        inverse = _s4_solve(a.T @ a, a.T)
    elif rank == n_rows:
        inverse = _s4_solve(a @ a.T, a).T
    else:
        reduced, pivots = _s4_rref(a.tolist())
        c = a[:, pivots]
        r = np.array(reduced[:rank], dtype=object)
        c_plus = _s4_solve(c.T @ c, c.T)
        inverse = r.T @ _s4_solve(r @ r.T, c_plus)
    return inverse, certificate
