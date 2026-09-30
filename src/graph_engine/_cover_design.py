"""Johnson T2 covering construction, reused verbatim (2026-09-30)."""
import itertools
import math
import numpy as np

def triple_index(n):
    idx = -np.ones((n, n, n), np.int32); c = 0
    for t in itertools.combinations(range(n), 3):
        idx[t] = c; c += 1
    return idx, c

def block_triples(block, idx):
    b = sorted(block)
    return np.array([idx[t] for t in itertools.combinations(b, 3)], np.int32)

def schonheim(n, k, t):
    if t == 0:
        return 1
    return math.ceil(n / k * schonheim(n - 1, k - 1, t - 1))

def greedy_cover(n, k, idx, T, rng, n_cand=30):
    """Each step: candidates = random blocks + blocks grown greedily from a random uncovered triple; keep the one covering the
    most new triples."""
    cov = np.zeros(T, bool); blocks = []
    trip = np.array(list(itertools.combinations(range(n), 3)))
    while not cov.all():
        best, bestgain = None, -1
        unc = np.flatnonzero(~cov)
        for c in range(n_cand):
            if c % 3 == 0:
                b = list(rng.choice(n, k, replace=False))
            else:
                b = list(trip[unc[rng.integers(len(unc))]])
                while len(b) < k:
                    rest = [x for x in range(n) if x not in b]
                    gains = [np.sum(~cov[block_triples(b + [x], idx)]) for x in rest]
                    top = np.flatnonzero(np.asarray(gains) == max(gains))
                    b.append(rest[int(rng.choice(top))])
            g = int(np.sum(~cov[block_triples(b, idx)]))
            if g > bestgain:
                best, bestgain = b, g
        blocks.append(np.array(best)); cov[block_triples(best, idx)] = True
    return blocks
