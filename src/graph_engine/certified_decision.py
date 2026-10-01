"""Exact threshold decisions over T1's unchanged flow/potential witnesses.

Graph identity is the ordered oriented edge representation, including loops
and parallel rows. A hash is an integrity binding, not source authentication.
Total soft wall budget includes graph validation/hash and field generation;
the independent checker is charged separately by the consumer.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as F
import hashlib
import math
from numbers import Integral, Real
from time import perf_counter

from .certified_resistance import (
    ResistanceBudget, ResistanceWitness, _edges, _node,
    certified_cross_resistance, verify_resistance_witness)

__all__ = ["DecisionWitness", "certified_decision", "verify_decision_witness",
           "edge_hash", "decision_verdict", "DecisionBatchWitness",
           "certified_decisions", "verify_decision_batch"]


def _threshold(theta):
    if isinstance(theta, bool) or not isinstance(theta, (F, Integral, Real)):
        raise TypeError("theta must be an exact rational or finite binary real")
    if isinstance(theta, Real) and not isinstance(theta, (F, Integral)):
        if not math.isfinite(float(theta)) or theta != float(theta):
            raise ValueError("theta must be finite and exactly representable")
    return F(theta)


def _row_hash(rows):
    digest = hashlib.sha256(b"T7-ordered-oriented-positive-edges-v1\n")
    for u, v, c in rows:
        digest.update(f"{u},{v},{c.numerator},{c.denominator}\n".encode("ascii"))
    return digest.hexdigest()


def edge_hash(edges):
    rows, _ = _edges(edges, 0, 0)
    return _row_hash(rows)


def decision_verdict(lo, hi, theta):
    if lo > theta:
        return "GREATER"
    if hi < theta:
        return "LESS"
    return "UNDECIDED"


@dataclass(frozen=True)
class DecisionWitness:
    graph_sha256: str
    a: int
    b: int
    theta: F
    lo: object
    hi: object
    interval: ResistanceWitness
    elapsed_seconds: float
    hash_seconds: float


def certified_decision(edges, a, b, theta, budget):
    started = perf_counter()
    a, b, theta = _node(a), _node(b), _threshold(theta)
    if isinstance(budget, Integral) and not isinstance(budget, bool):
        budget = ResistanceBudget(max_updates=int(budget))
    if not isinstance(budget, ResistanceBudget):
        raise TypeError("budget must be an integer or ResistanceBudget")
    # A single pass consumes generators and freezes mutable edge rows.
    rows, _ = _edges(edges, a, b)
    digest = _row_hash(rows)
    hash_seconds = perf_counter() - started
    if budget.seconds is not None:
        budget = replace(budget, seconds=max(0.0, budget.seconds-hash_seconds))
    lo, hi, interval = certified_cross_resistance(
        rows, a, b, budget,
        stop=lambda lo, hi: decision_verdict(lo, hi, theta) != "UNDECIDED")
    witness = DecisionWitness(digest, a, b, theta, lo, hi, interval,
                              perf_counter()-started, hash_seconds)
    return {"verdict": decision_verdict(lo, hi, theta),
            "updates_used": interval.updates, "witness": witness}


def verify_decision_witness(edges, a, b, theta, result):
    a, b, theta = _node(a), _node(b), _threshold(theta)
    if not isinstance(result, dict) or not isinstance(result.get("witness"), DecisionWitness):
        raise ValueError("invalid decision certificate")
    witness = result["witness"]
    rows, _ = _edges(edges, a, b)
    if _row_hash(rows) != witness.graph_sha256:
        raise ValueError("graph edge hash mismatch")
    if (a, b, theta) != (witness.a, witness.b, witness.theta):
        raise ValueError("query binding mismatch")
    verify_resistance_witness(rows, a, b, witness.lo, witness.hi, witness.interval)
    expected = decision_verdict(witness.lo, witness.hi, theta)
    if result.get("verdict") != expected:
        raise ValueError("verdict does not follow strictly from the bounds")
    if result.get("updates_used") != witness.interval.updates:
        raise ValueError("update count mismatch")
    return True


@dataclass(frozen=True)
class DecisionBatchWitness:
    graph_sha256: str
    a: int
    b: int
    thetas: tuple
    lo: object
    hi: object
    interval: ResistanceWitness
    elapsed_seconds: float
    hash_seconds: float


def certified_decisions(edges, a, b, thetas, budget):
    """Multiple thresholds share one exact interval, hash and field witness.

    The entire batch pays startup. A per-query amortized cost is meaningful
    only for a consumer that actually needs all these thresholds together.
    """
    started = perf_counter()
    a, b = _node(a), _node(b)
    thetas = tuple(_threshold(theta) for theta in thetas)
    if not thetas:
        raise ValueError("at least one threshold required")
    if isinstance(budget, Integral) and not isinstance(budget, bool):
        budget = ResistanceBudget(max_updates=int(budget))
    if not isinstance(budget, ResistanceBudget):
        raise TypeError("budget must be an integer or ResistanceBudget")
    rows, _ = _edges(edges, a, b)
    digest = _row_hash(rows)
    hash_seconds = perf_counter()-started
    if budget.seconds is not None:
        budget = replace(budget, seconds=max(0.0, budget.seconds-hash_seconds))
    lo, hi, interval = certified_cross_resistance(
        rows, a, b, budget,
        stop=lambda lo, hi: all(decision_verdict(lo, hi, theta) != "UNDECIDED" for theta in thetas))
    witness = DecisionBatchWitness(digest, a, b, thetas, lo, hi, interval,
                                   perf_counter()-started, hash_seconds)
    return {"verdicts": tuple(decision_verdict(lo, hi, theta) for theta in thetas),
            "updates_used": interval.updates, "witness": witness}


def verify_decision_batch(edges, a, b, thetas, result):
    a, b = _node(a), _node(b)
    thetas = tuple(_threshold(theta) for theta in thetas)
    if not isinstance(result, dict) or not isinstance(result.get("witness"), DecisionBatchWitness):
        raise ValueError("invalid decision batch certificate")
    witness = result['witness']
    rows, _ = _edges(edges, a, b)
    if _row_hash(rows) != witness.graph_sha256:
        raise ValueError("graph edge hash mismatch")
    if (a, b, thetas) != (witness.a, witness.b, witness.thetas):
        raise ValueError("query binding mismatch")
    verify_resistance_witness(rows, a, b, witness.lo, witness.hi, witness.interval)
    expected = tuple(decision_verdict(witness.lo, witness.hi, theta) for theta in thetas)
    if result.get('verdicts') != expected:
        raise ValueError("verdicts do not follow strictly from the bounds")
    if result.get('updates_used') != witness.interval.updates:
        raise ValueError("update count mismatch")
    return True
