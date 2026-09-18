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

Three additions on top of `engine`, each a flag in the policy name so all of them run on the same worlds with common
random numbers (`engine+guard`, `engine+reliability`, `engine+replay`, `engine+all`):

  +guard        a share `guard` of the budget is bought against the MODEL error instead of the sign error:
                regime_posterior.model_check_probe (expected drop of the entropy of the one-vs-two-transition family
                indicator) per unit cost, maximized over pairs and instruments. Motivated by the e21 negative: the value
                rule flagged 0 of 25 two-transition pairs, because a probe that would expose a second transition lowers
                no sign potential while the single-transition reading still fits. Guard spending is capped so that it
                stays within one cost unit of `guard` × budget.
  +guard2       the same share, spent on PAIRS: regime_posterior.model_check_pair per unit cost of both probes, which
                are then executed back to back on that pair. Motivated by what +guard leaves on the table: a second
                transition is only exposed by probes on both sides of both transitions, and model_check_probe buys one
                at a time — after one answer the single-transition family usually still explains everything, so the next
                guard probe goes to another pair (3-4 of 25 collisions flagged, 21 missed). Mutually exclusive with
                +guard, same cap on the guard spend. Measured: alone it is WORSE than +guard (1 of 25); it pays only
                together with +replay and +majority (6 and 7 of 25) — see the numbers below.
  +eopt         the guard share is spent by the E-OPTIMAL rule (regime_posterior.weakest_direction_probe) instead of the
                family-entropy rule: the probe that most lifts the SMALLEST pairwise discrimination between hypotheses
                that carry mass. Motivated by the diagnosis that the second transition is a direction the entropy
                (D-optimal) criterion can ignore because it optimizes an average. Mutually exclusive with
                +guard/+guard2/+burst. MEASURED (e21c, same 40 worlds, budget 40): NEGATIVE on both axes —
                +eopt+replay+majority 1.181 (+0.101 ± 0.046 paired against +guard2+replay+majority's 1.08, 11 of 40
                wins), 4 of 25 collisions against 7; the pure rule without replay/majority 1.331 (+0.251 ± 0.051),
                0 of 25.
  +eoptmix      every probe (no guard share) by the mixed rule, value = expected potential drop + λ·(min-discrimination
                increase) with λ fixed on the first step so the terms have equal scale. MEASURED: 1.248
                (+0.169 ± 0.068, 12 of 40), 0 of 25 collisions — the calibration gap is the flattest of the four
                (−0.002) and that is all it buys.
  +majority     the box claims are read as majority reports (RegimePosterior claim_model="majority") instead of
                pointwise r-accurate labels — the reading the +reliability result points at as the seat of the loop's
                over-confidence.
  +reliability  claims stop entering at the fixed 0.75. After each probe the loop asks, per (pair, bin) question, every
                claim's ROOT source to vote its sign (copies collapse onto their parent's root, which is what the World's
                frozenset carries), pins the truth of the questions a probe has landed in (majority of the probe signs in
                that bin, only with ≥ 2 probes or one probe of reliability ≥ 0.95) and runs source_reliability.estimate
                (Dawid–Skene with those anchors). The per-root r, clipped to [0.5, 0.98], is pushed into every posterior
                with set_claim_reliabilities. The target was the CALIBRATION gap (actual wrong measure − believed), not
                accuracy; what it measured is below.
  +replay       empirical Bayes on the loop's own history: after each world the realized probe record gives p_flip (share
                of pairs whose probes revealed a sign change), p_two (share that revealed two) and the mean estimated
                source reliability; the running means (started from the defaults 0.3 / 0.05 / 0.75 as one pseudo-world)
                become the next world's priors instead of the defaults. The caller carries the state between worlds
                (`replay_prior=` in, `world_stats` out); nothing inside one world is changed by it.

Measured (e21, 40 worlds, budget 40; wrong measure at 40, calibration gap = actual − believed, collision tp of 25):
  engine 1.24, gap 0.21 ± 0.06, 0 tp | +guard 0.2: 1.29 (+0.04 ± 0.04 paired), gap 0.18, 3 tp 1 fp — the model-check
  probe buys the first true collision flags this package has seen at a cost inside the noise; guard 0.3 the same (3 tp,
  0 fp), guard 0.1 weaker (2 tp) and dearer (+0.10 ± 0.05).
  +reliability: 1.33 (+0.085 ± 0.036 paired, worse), gap 0.21 → 0.56. It REFUTES the diagnosis it was built on: the
  estimated root reliabilities are 0.80–0.93, above the fixed 0.75, so honest per-source weights make the engine more
  confident (believed 1.03 → 0.77) while the truth does not move. The residual over-confidence is not in the value of r;
  it is in the box likelihood (a source reports the majority sign of a wide box, the posterior reads that as an r-accurate
  label at every point of the box) and it survives any r.
  +replay: 1.16 (−0.085 ± 0.037 paired, 22 of 40 worlds; first 10 worlds −0.08, last 10 −0.26 as the history grows) and
  the gap falls 0.28 (first 10) → 0.12 (last 10) against the engine's own 0.27 → 0.29. The learned priors after 40 worlds:
  p_two 0.052 (true 0.05), mean source reliability 0.82 (true 0.88 over the five reporting roots), p_flip 0.49 against a
  true 0.63 — p_flip keeps a downward bias, because a transition no probe brackets leaves its family at prior mass.
  +all: 1.27, gap 0.31, 4 tp of 25 with 0 false flags.
  +guard2 (re-run of e21, same 40 worlds): ALONE it is worse than +guard — 1.38 (+0.135 ± 0.040 paired), 1 tp of 25:
  the pair of probes is bought at twice the cost and the two answers mostly confirm one transition, so the guard share
  buys half as many chances. Combined it is the opposite: +guard2+replay 1.19 (−0.056 ± 0.048, 24 of 40 worlds),
  gap 0.03, 6 tp 1 fp; +guard2+replay+majority 1.08 (−0.164 ± 0.053, 27 of 40), gap −0.08, 7 tp 2 fp of 25 — the best
  configuration measured here on BOTH axes. The reason the combination works is the prior: with the default p_two = 0.05
  a pair needs a large likelihood ratio to pass flag_at 0.5, and replay's learned p_two (0.052–0.070) plus the majority
  reading of the box claims (which no longer force the single-transition family to fit) let the pair's evidence land.
  Collision detection is 7 of 25, not 25 of 25: budget 40 over 12 pairs is 2–3 guard probes per pair.

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
from .source_reliability import estimate as estimate_reliability

__all__ = ["World", "run", "wrong_measure", "DEFAULT_PRIOR"]

DEFAULT_PRIOR = {"p_flip": 0.3, "p_two": 0.05, "reliability": 0.75}   # RegimePosterior's own defaults


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


def _posteriors(world: World, independent_copies: bool, with_roots: bool = False, **kw):
    """The posteriors, one per pair. with_roots also returns, per pair, the root source of each claim in claim order."""
    posts = [RegimePosterior(0.0, 1.0, **kw) for _ in range(world.n_pairs)]
    roots_of: list[list] = [[] for _ in range(world.n_pairs)]
    groups: dict[tuple, list] = {}
    for p, a, b, s, roots in world.claims:
        groups.setdefault((p, round(a, 6), round(b, 6), s), []).append(roots)
    for (p, a, b, s), rs in groups.items():
        posts[p].add_claim(a, b, s, n_eff=float(len(rs)) if independent_copies else lineage_information(rs))
        roots_of[p].append(min(set().union(*rs)))               # copies carry their parent's root
    return (posts, roots_of) if with_roots else posts


def _vote_questions(world: World, posts, roots_of, bins: int) -> dict:
    """{(pair, bin): {root: ±1}} — each claim's root votes its sign on every bin its box overlaps."""
    raw: dict[tuple, dict] = {}
    for p in range(world.n_pairs):
        for (a, b, sg, _n, _r), s in zip(posts[p].claims, roots_of[p]):
            for k in range(bins):
                lo, hi = k / bins, (k + 1) / bins
                if min(b, hi) - max(a, lo) > 1e-9:
                    raw.setdefault((p, k), {}).setdefault(s, []).append(sg)
    return {q: {s: (1 if sum(v) > 0 else -1) for s, v in d.items() if sum(v) != 0} for q, d in raw.items()}


def _probe_anchors(world: World, probe_log, bins: int, votes: dict) -> dict:
    """{(pair, bin): ±1} for the questions a probe has landed in: the majority probe sign, with ≥ 2 probes
    in the bin or one probe of reliability ≥ 0.95."""
    known = {}
    for p in range(world.n_pairs):
        for k in range(bins):
            pr = [(sg, r) for x, sg, r in probe_log[p] if min(int(x * bins), bins - 1) == k]
            if pr and (len(pr) >= 2 or max(r for _, r in pr) >= 0.95) and (p, k) in votes:
                known[(p, k)] = 1 if sum(sg for sg, _ in pr) > 0 else -1
    return known


def _estimate_roots(world: World, posts, roots_of, probe_log, bins: int) -> dict:
    """Per-root reliability from agreement + the probe anchors (Dawid–Skene), clipped to [0.5, 0.98]."""
    votes = _vote_questions(world, posts, roots_of, bins)
    known = _probe_anchors(world, probe_log, bins, votes)
    est = estimate_reliability(votes, known=known or None)
    return {s: min(max(float(v["r"]), 0.5), 0.98) for s, v in est.items()}


def _push_reliabilities(posts, roots_of, rs: dict, default: float) -> None:
    for p, rt in enumerate(roots_of):
        if rt:
            posts[p].set_claim_reliabilities([rs.get(s, default) for s in rt])


def _world_stats(world: World, posts, probe_log, rs: dict) -> dict:
    """What this world realized, for the replay prior.

    p_flip / p_two are read off the posterior AFTER the probes (mass on the one- and two-transition families), not off the
    raw probe sign sequence: with a cost-1 probe at reliability 0.8 the raw count of sign changes is dominated by probe
    noise (measured on world 0: 0.50 of the pairs "showed two transitions" where 0.05 have them). The posterior weighs an
    answer by the instrument's reliability, so a single flipped cheap probe does not create a transition.
    """
    m1 = m2 = 0.0
    for p, rp in enumerate(posts):
        _, _, _, post = rp._with_probes()
        two = float(post[rp._n_one:].sum())
        one = float(post[2:rp._n_one].sum())
        m1 += one; m2 += two
    m1 /= world.n_pairs; m2 /= world.n_pairs
    raw = []
    for p in range(world.n_pairs):
        sgs = [s for _, s in sorted((x, s) for x, s, _ in probe_log[p])]
        raw.append(any(s != sgs[0] for s in sgs) if sgs else False)
    raw_flip = float(np.mean(raw)) if raw else 0.0
    return {"p_flip": min(max(m1 / max(1 - m2, 1e-6), 0.02), 0.95), "p_two": min(max(m2, 0.002), 0.4),
            "reliability": min(max(float(np.mean(list(rs.values()))) if rs else DEFAULT_PRIOR["reliability"], 0.5), 0.98),
            "raw_probe_flip": raw_flip, "n_roots": len(rs)}


def _flags(policy: str) -> set:
    parts = policy.split("+")
    if parts[0] not in ("engine", "copies", "random", "oracle"):
        raise ValueError(policy)
    f = set(parts[1:])
    if "all" in f:
        f = {"guard", "reliability", "replay"}
    if f - {"guard", "guard2", "burst", "eopt", "eoptmix", "reliability", "replay", "majority"}:
        raise ValueError(policy)
    if len(f & {"guard", "guard2", "burst", "eopt", "eoptmix"}) > 1:
        raise ValueError(policy)
    return f


def run(world: World, policy: str, budget: float, instruments=((1.0, 0.8), (4.0, 0.99)), seed: int = 0,
        record_every: float = 5.0, guard: float = 0.2, burst_n: int = 6, rel_bins: int = 3, rel_every: float = 1.0,
        replay_prior: dict | None = None, **kw) -> dict:
    """Returns {"cost": [...], "wrong": [...]} sampled every `record_every` cost units, plus the final state.

    policy: "random" | "oracle" | "copies" | "engine" with any of "+guard", "+reliability", "+replay" (or "+all").
    guard: share of the budget spent on model_check_probe (policies with +guard).
    rel_bins: questions for the reliability estimate are (pair, one of `rel_bins` equal bins of the domain); a claim votes
        on every bin its box overlaps. 3 is what e21 ran: with 1 (the whole pair) most questions get fewer than the three
        origins Dawid-Skene needs, with 4 the bins are narrower than the claims.
    replay_prior: {"p_flip", "p_two", "reliability"} from earlier worlds (policies with +replay); None = the defaults.
    """
    flags = _flags(policy)
    base = policy.split("+")[0]
    rng = np.random.default_rng(seed)
    if "majority" in flags:
        kw = dict(kw)
        kw.setdefault("claim_model", "majority")
    if "replay" in flags and replay_prior:
        kw = dict(kw)
        kw.setdefault("p_flip", float(replay_prior["p_flip"]))
        kw.setdefault("p_two", float(replay_prior["p_two"]))
        kw.setdefault("reliability", float(replay_prior["reliability"]))
    default_rel = kw.get("reliability", RegimePosterior(0.0, 1.0).reliability)
    posts, roots_of = _posteriors(world, independent_copies=(base == "copies"), with_roots=True, **kw)
    probe_log: list[list] = [[] for _ in range(world.n_pairs)]
    spent, spent_guard, curve = 0.0, 0.0, [(0.0, wrong_measure(world, posts))]
    next_rec, next_rel = record_every, rel_every
    while spent < budget:
        kind = "value"
        if (flags & {"guard", "guard2", "burst", "eopt"}) and spent_guard < guard * budget:
            k = 2 if "guard2" in flags else 1                  # +guard2 buys the PAIR of probes back to back
            fits = [(c, r) for c, r in instruments if spent_guard + k * c <= guard * budget + 1.0]
            if fits:
                kind = "model"
                best = (-1.0, None)
                for p in range(world.n_pairs):
                    for c, r in fits:
                        if "eopt" in flags:                    # E-optimal: the guard share attacks the WEAKEST
                            x, gain = posts[p].weakest_direction_probe(r)   # discriminated direction instead of the
                            xs_now = [x]                       # average (family-entropy) direction
                        elif k == 2:
                            (x1, x2), gain = posts[p].model_check_pair(r)
                            xs_now = [x1, x2]
                        else:
                            x, gain = posts[p].model_check_probe(r)
                            xs_now = [x]
                        if gain / (k * c) > best[0]:
                            best = (gain / (k * c), (p, xs_now, c, r))
                p, xs_now, c, r = best[1]
        if kind == "model":
            pass
        elif base == "random":
            p, x, (c, r) = int(rng.integers(world.n_pairs)), float(rng.random()), instruments[0]
        elif base in ("engine", "copies"):
            best = (-1.0, None)
            for p in range(world.n_pairs):
                for c, r in instruments:
                    x, gain = (posts[p].weakest_direction_probe(r, mix=True) if "eoptmix" in flags
                               else posts[p].best_probe(r))   # +eoptmix: EVERY probe by the mixed rule, no guard share
                    if gain / c > best[0]:
                        best = (gain / c, (p, x, c, r))
            p, x, c, r = best[1]
        elif base == "oracle":
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
        if kind != "model":
            xs_now = [x]
        if kind == "model" and "burst" in flags:              # +burst: commit a run of probes to the chosen pair, each placed by
            xs_now = []                                        # the family-entropy rule AFTER the previous answer (e28: 12 probes
            for _ in range(burst_n):                           # on one pair find the second transition; 2-3 do not)
                if spent_guard + c > guard * budget + 1.0:
                    break
                x, _g = posts[p].model_check_probe(r); xs_now.append(x)
                ans = world.probe(p, x, r, rng); posts[p].add_probe(x, ans, r); probe_log[p].append((float(x), int(ans), float(r)))
                spent += c; spent_guard += c
            xs_now = []
        for x in xs_now:
            ans = world.probe(p, x, r, rng)
            posts[p].add_probe(x, ans, r)
            probe_log[p].append((float(x), int(ans), float(r)))
            spent += c
            if kind == "model":
                spent_guard += c
        if "reliability" in flags and spent >= next_rel:
            _push_reliabilities(posts, roots_of, _estimate_roots(world, posts, roots_of, probe_log, rel_bins), default_rel)
            next_rel = spent + rel_every
        if spent >= next_rec:
            curve.append((spent, wrong_measure(world, posts))); next_rec += record_every
    if curve[-1][0] < spent:
        curve.append((spent, wrong_measure(world, posts)))
    believed = sum(rp.expected_error() for rp in posts)          # what the engine thinks it gets wrong: calibration check
    rs_final = _estimate_roots(world, posts, roots_of, probe_log, rel_bins)
    flagged = [p for p in range(world.n_pairs) if posts[p].collision()["flag"]]
    two = [p for p in range(world.n_pairs) if len(world.transitions[p]) == 2]
    return {"policy": policy, "cost": [c for c, _ in curve], "wrong": [w for _, w in curve], "final_wrong": curve[-1][1],
            "believed_wrong": believed, "gap": curve[-1][1] - believed, "collision_flagged": flagged,
            "collision_true": two, "spent": spent, "spent_guard": spent_guard,
            "reliabilities": rs_final, "world_stats": _world_stats(world, posts, probe_log, rs_final)}
