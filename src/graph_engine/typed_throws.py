"""Typed triple operations integrated from TRIPLE_THROWS_20260930.

Port closure is a directed structural model with unit elasticities. Its bits
are surrogate information, never calibrated physical or worker PASS odds.
"""
from itertools import combinations
from dataclasses import dataclass
import math
import numpy as np

from .typed_throw_ops.parts import (Port, Part, Claim, OPERATORS, AXES, load, dump,
                                   regime_intersection, box_intersection_many,
                                   assumption_conflicts)
from .typed_throw_ops.compose import Link, Chain, link, evaluate_chain, enumerate_chains
from .typed_throw_ops.value import (StructuralImage, closure, closure_value, triple_terms,
                                   mediator_scores, best_target_for)
from .typed_throw_ops.triples import (mediation, sign_cycles, frustrated_cycles, interval_triples,
                                     order_cycles, transition_triples, Relation, satisfiable, classify)
from .typed_throw_ops.signature import port_signature, distance, word_distance
from .typed_throw_ops.draw import draw_from_universe, all_inclusion_probabilities
from ._cover_design import triple_index, greedy_cover, schonheim


def set_value_Q_bits(form, H, sigma=1.0, Q=None):
    """Gaussian target information via the existing value_bits/observe chain rule.

    No mutation of form; no submodularity or greedy approximation guarantee.
    This uses the form's Gaussian convention, including its fixed gauge.
    """
    H = np.asarray(H, float)
    if H.size == 0:
        return 0.0
    H = np.atleast_2d(H)
    sig = np.broadcast_to(np.asarray(sigma, float), (len(H),))
    if H.shape[1] != form.d or not np.all(np.isfinite(H)):
        raise ValueError("observation rows must be finite and match form dimension")
    if not np.all(np.isfinite(sig)) or np.any(sig <= 0):
        raise ValueError("observation noise must be finite and positive")
    if Q is not None:
        Q = np.atleast_2d(np.asarray(Q, float))
        if Q.shape[1] != form.d or not np.all(np.isfinite(Q)):
            raise ValueError("target rows must be finite and match form dimension")
    form.cov()  # share the prepared prior across exact candidate-set evaluations
    f, total = form.copy(), 0.0
    for h, s in zip(H, sig):
        total += f.value_bits(h, float(s), Q=Q)
        f.observe(h, float(s))
    return float(total)


def draw_triples(scores, k, temperature=1.0, floor=0.1, seed=0):
    """Sample declared triple indices with exact marginal inclusion probabilities."""
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("triple scores must be a finite vector")
    if k < 0 or temperature <= 0 or not math.isfinite(temperature) or not 0 <= floor <= 1:
        raise ValueError("invalid draw size, temperature or floor")
    if not len(scores) or k == 0:
        return []
    return draw_from_universe(scores, k, temperature, floor, seed)


@dataclass(frozen=True)
class CoverDesign:
    blocks: tuple
    triple_pi: float
    covered: int
    total: int
    lower_bound: int


def cover_triples(n, k, seed=0, budget=None, n_cand=30):
    """Johnson T2 greedy covering, then uniformly random relabeling.

    Every triple has pi = covered / C(n,3) over the relabeling randomization.
    This is coverage probability, conditional on the constructed design, not
    worker detection probability. Prefix budgets may leave uncovered triples.
    """
    if not 3 <= k <= n or n_cand < 1 or (budget is not None and budget < 0):
        raise ValueError("require 3 <= k <= n, positive candidates, nonnegative budget")
    design_seed, relabel_seed = np.random.SeedSequence(seed).spawn(2)
    idx, total = triple_index(n)
    blocks = greedy_cover(n, k, idx, total, np.random.default_rng(design_seed), n_cand)
    if budget is not None:
        blocks = blocks[:budget]
    permutation = np.random.default_rng(relabel_seed).permutation(n)
    blocks = tuple(tuple(sorted(map(int, permutation[b]))) for b in blocks)
    covered = len({t for b in blocks for t in combinations(b, 3)})
    return CoverDesign(blocks, covered / total, covered, total, schonheim(n, k, 3))
