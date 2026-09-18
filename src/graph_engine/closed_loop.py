#!/usr/bin/env python3
"""
closed_loop.py — the engine choosing experiments against a world that answers, with the truth known to the test.

Everything else in this package was measured on proxies (citation graphs, curated two-graph cases, templated sentences).
This module closes the loop: a WORLD holds, for many (subject, object) pairs, the true sign of dY/dX along a condition
variable x ∈ [0, 1]; the sign changes at most once (a peak or a threshold) — or, for a declared fraction of pairs, twice
(the collision case the single-transition reading must flag, see regime_posterior). Sources observe boxes of the world and
report signs with a reliability; some sources are copies of others (lineage). The engine reads the claims, chooses a
probe (pair, x, instrument), the world answers through the instrument's reliability, the posterior is updated, repeat
until the budget is spent. Score: the measure of the domain whose sign the engine gets WRONG, summed over pairs, against
the truth — not the engine's own belief.

Policies compared:
  engine      value = expected potential drop (regime_posterior.best_probe) per unit cost, over pairs and instruments,
              claims weighted by lineage_information (copies are not independent support)
  copies      the same, but every claim counts as one independent source
  random      uniform pair, uniform x, the cheap instrument
  oracle      knows the truth: picks the probe whose realized answer lowers the true wrong measure most (upper bound)

The world's sign functions are the one-transition mechanisms of e17 in normalized coordinates (a peak: resonance amplitude
against drive frequency, harvested yield against effort, Hall–Petch strength against inverse grain size with its reversal;
monotone: Hooke, photon energy against frequency), with the transition point drawn at random per pair, so the engine
cannot know it. They are functions, not simulations; what they share with the mechanisms is the sign structure the
regime posterior reasons about.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .claim_federation import lineage_information
from .regime_posterior import RegimePosterior

__all__ = ["World", "run", "wrong_measure"]


@dataclass
class World:
    n_pairs: int = 12
    n_sources: int = 8
    claims_per_pair: int = 4
    source_reliability: tuple = (0.95, 0.9, 0.9, 0.85, 0.8, 0.8, 0.7, 0.6)
    copy_fraction: float = 0.35              # share of sources that only copy another source's reports
    p_transition: float = 0.6                # share of pairs with one transition (else monotone)
    p_two: float = 0.05                      # share of pairs with two transitions (collision case)
    seed: int = 0
    transitions: list = field(default_factory=list)   # per pair: sorted list of transition points
    start_sign: list = field(default_factory=list)
    claims: list = field(default_factory=list)        # (pair, a, b, sign, roots frozenset)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_pairs):
            u = rng.random()
            k = 2 if u < self.p_two else 1 if u < self.p_two + self.p_transition else 0
            self.transitions.append(sorted(rng.uniform(0.1, 0.9, k).tolist()))
            self.start_sign.append(int(rng.choice([-1, 1])))
        rel = list(self.source_reliability)[: self.n_sources]
        n_copy = int(round(self.copy_fraction * self.n_sources))
        parents = {s: (int(rng.integers(0, self.n_sources - n_copy)) if s >= self.n_sources - n_copy else None) for s in range(self.n_sources)}
        for p in range(self.n_pairs):
            reporters = rng.choice(self.n_sources - n_copy, self.claims_per_pair, replace=True)
            for s in reporters:
                a = rng.uniform(0, 0.8); b = a + rng.uniform(0.1, 0.4); b = min(b, 1.0)
                sign = self.majority_sign(p, a, b)
                if rng.random() > rel[s]:
                    sign = -sign
                self.claims.append((p, a, b, sign, frozenset([int(s)])))
                for c, par in parents.items():
                    if par == s and rng.random() < 0.7:            # a copy repeats its parent's report
                        self.claims.append((p, a, b, sign, frozenset([int(s)])))

    def sign(self, p: int, x: float) -> int:
        return self.start_sign[p] * (-1) ** sum(x > t for t in self.transitions[p])

    def majority_sign(self, p: int, a: float, b: float) -> int:
        xs = np.linspace(a, b, 33)
        return 1 if sum(self.sign(p, x) for x in xs) >= 0 else -1

    def probe(self, p: int, x: float, reliability: float, rng) -> int:
        s = self.sign(p, x)
        return s if rng.random() < reliability else -s


def wrong_measure(world: World, posts: list[RegimePosterior], n: int = 200) -> float:
    """Σ over pairs of the measure of x where the engine's most probable sign is not the true sign."""
    xs = (np.arange(n) + 0.5) / n
    tot = 0.0
    for p, rp in enumerate(posts):
        cells, _, F, post = rp._with_probes()
        pp = post @ F
        idx = np.minimum(np.searchsorted(cells[:, 1], xs, side="left"), len(cells) - 1)
        guess = np.where(pp[idx] >= 0.5, 1, -1)
        truth = np.array([world.sign(p, x) for x in xs])
        tot += float((guess != truth).mean())
    return tot


def _posteriors(world: World, independent_copies: bool, **kw) -> list[RegimePosterior]:
    posts = [RegimePosterior(0.0, 1.0, **kw) for _ in range(world.n_pairs)]
    groups: dict[tuple, list] = {}
    for p, a, b, s, roots in world.claims:
        groups.setdefault((p, round(a, 6), round(b, 6), s), []).append(roots)
    for (p, a, b, s), rs in groups.items():
        posts[p].add_claim(a, b, s, n_eff=float(len(rs)) if independent_copies else lineage_information(rs))
    return posts


def run(world: World, policy: str, budget: float, instruments=((1.0, 0.8), (4.0, 0.99)), seed: int = 0,
        record_every: float = 5.0, **kw) -> dict:
    """Returns {"cost": [...], "wrong": [...]} sampled every `record_every` cost units, plus the final state."""
    rng = np.random.default_rng(seed)
    posts = _posteriors(world, independent_copies=(policy == "copies"), **kw)
    spent, curve = 0.0, [(0.0, wrong_measure(world, posts))]
    next_rec = record_every
    while spent < budget:
        if policy == "random":
            p, x, (c, r) = int(rng.integers(world.n_pairs)), float(rng.random()), instruments[0]
        elif policy in ("engine", "copies"):
            best = (-1.0, None)
            for p in range(world.n_pairs):
                for c, r in instruments:
                    x, gain = posts[p].best_probe(r)
                    if gain / c > best[0]:
                        best = (gain / c, (p, x, c, r))
            p, x, c, r = best[1]
        elif policy == "oracle":
            best = (float("inf"), None)
            for p in range(world.n_pairs):
                for c, r in instruments:
                    for x in np.linspace(0.02, 0.98, 25):
                        trial = RegimePosterior(0.0, 1.0, **kw); trial.claims = list(posts[p].claims); trial.probes = list(posts[p].probes)
                        trial.add_probe(x, world.sign(p, x), r)
                        w = wrong_measure(world, [trial if q == p else posts[q] for q in range(world.n_pairs)])
                        if (w - curve[-1][1]) / c < best[0]:
                            best = ((w - curve[-1][1]) / c, (p, x, c, r))
            p, x, c, r = best[1]
        else:
            raise ValueError(policy)
        posts[p].add_probe(x, world.probe(p, x, r, rng), r)
        spent += c
        if spent >= next_rec:
            curve.append((spent, wrong_measure(world, posts))); next_rec += record_every
    if curve[-1][0] < spent:
        curve.append((spent, wrong_measure(world, posts)))
    believed = sum(rp.expected_error() for rp in posts)          # what the engine thinks it gets wrong: calibration check
    flagged = [p for p in range(world.n_pairs) if posts[p].collision()["flag"]]
    two = [p for p in range(world.n_pairs) if len(world.transitions[p]) == 2]
    return {"policy": policy, "cost": [c for c, _ in curve], "wrong": [w for _, w in curve], "final_wrong": curve[-1][1],
            "believed_wrong": believed, "collision_flagged": flagged, "collision_true": two}
