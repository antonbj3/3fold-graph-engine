"""Observable node/pair geometry for the shared-latent-space hypothesis.

Points are predictions [node, query, outcome], with aligned query/outcome meanings.
Metrics distinguish observable predictions, not all possible underlying worlds.
Selected pairs avoid constructing a quadratic all-node distance matrix.
"""
from __future__ import annotations

import numpy as np

from .predictive_state import _probabilities


def _inputs(predictions, query_weights, pairs):
    p = _probabilities(predictions, 3)
    w = _probabilities(query_weights, 1)
    if p.shape[1] != len(w):
        raise ValueError("one weight per aligned query is required")
    e = np.asarray(pairs)
    if e.ndim != 2 or e.shape[1] != 2 or len(e) == 0 or e.dtype.kind not in "iu":
        raise ValueError("pairs must be a nonempty integer [edge, 2] array")
    if np.any(e < 0) or np.any(e >= len(p)):
        raise ValueError("pair endpoint outside the node set")
    return p, w, e


def squared_pair_distances(predictions, query_weights, pairs, *, metric="hellinger"):
    """Weighted product-query distances, one per requested pair.

    ``hellinger``: sum_q w_q * (1 - sum_y sqrt(p_y r_y)), standard H^2.
    ``fisher``: sum_q w_q * [2 acos(sum_y sqrt(p_y r_y))]^2.
    The latter is INTRINSIC PRODUCT distance, not an ambient concatenated sphere.
    A stable half-angle formula avoids acos roundoff near identical distributions.
    Both are squared distances; their local ratio is8, not1.
    Cost O(E*Q*K), plus input validation O(N*Q*K).
    """
    p, w, e = _inputs(predictions, query_weights, pairs)
    left, right = np.sqrt(p[e[:, 0]]), np.sqrt(p[e[:, 1]])
    diff2 = np.sum((left-right)**2, axis=-1)
    if metric == "hellinger":
        return 0.5 * (diff2 @ w)
    if metric != "fisher":
        raise ValueError("metric must be hellinger or fisher")
    sum2 = np.sum((left+right)**2, axis=-1)
    angles = 4 * np.arctan2(np.sqrt(diff2), np.sqrt(sum2))
    return (angles**2) @ w


def pair_distortion(student, teacher, query_weights, pairs, pair_weights=None,
                    *, metric="hellinger"):
    """Mean squared mismatch of squared distances; returns loss and per-edge residual.

    This relational loss alone cannot fix semantic permutations/reflections.
    Supply independently anchored node targets in a learning objective as well.
    """
    ds = squared_pair_distances(student, query_weights, pairs, metric=metric)
    dt = squared_pair_distances(teacher, query_weights, pairs, metric=metric)
    if np.shape(student) != np.shape(teacher):
        raise ValueError("teacher and student must use the same nodes/queries/outcomes")
    weights = (np.full(len(ds), 1/len(ds)) if pair_weights is None
               else _probabilities(pair_weights, 1))
    if len(weights) != len(ds):
        raise ValueError("one normalized weight per selected pair is required")
    residual = ds-dt
    return float(weights @ residual**2), residual


def js_pair_divergence(predictions, query_weights, pairs):
    """Query-weighted Jensen-Shannon divergence in BITS, not squared FR distance.

    Equal mixture weights correspond to a balanced optimal GAN discriminator.
    Locally JS_bits = d_FR^2/(8*ln(2)) to second order. Marginal distribution
    matching does not determine node correspondence or externally correct labels.
    """
    p, w, e = _inputs(predictions, query_weights, pairs)
    left, right = p[e[:, 0]], p[e[:, 1]]
    mean = 0.5*(left+right)
    result = np.zeros(mean.shape[:2])
    for side in (left, right):
        ratio = np.ones_like(side)
        np.divide(side, mean, out=ratio, where=side > 0)
        logratio = np.log2(ratio)
        result += 0.5*np.sum(side*logratio, axis=-1)
    return np.maximum(result @ w, 0.0)


def fisher_pullback(probabilities, jacobian, query_weights):
    """Local G = sum_q w_q J_q^T diag(1/p_q) J_q.

    p: [query,outcome], J: [query,outcome,coordinate]. Interior probabilities
    only. Each derivative sums to0 over outcomes, since probability mass is1.
    G may be singular; this function neither inverts it nor claims global
    identifiability. At the boundary the caller must derive the appropriate limit.
    """
    p = _probabilities(probabilities, 2)
    w = _probabilities(query_weights, 1)
    j = np.asarray(jacobian, dtype=float)
    if len(w) != len(p) or j.ndim != 3 or j.shape[:2] != p.shape or j.shape[2] == 0:
        raise ValueError("incompatible query/outcome/coordinate dimensions")
    if np.any(p <= 0) or not np.all(np.isfinite(j)):
        raise ValueError("finite derivatives and interior probabilities are required")
    if not np.allclose(j.sum(axis=1), 0, rtol=0, atol=1e-10):
        raise ValueError("probability derivatives must be tangent to the simplex")
    return np.einsum("qky,qk,qkz->yz", j, w[:, None]/p, j)
