#!/usr/bin/env python3
"""E37: the interval between two claims, filled twice — by the claims (regime posterior) and by a first-principles model
(sign of dy/dx over the declared parameter box) — and probed where the two fillings disagree most.

World: `disagreement_field.LinkedWorld`, a `closed_loop.World` whose pairs are one-transition peaks (harvest /
resonance family) and where `n_links` disjoint COUPLES of pairs share a latent θ, so their transition points coincide.
Nothing in the claim graph encodes θ: boxes, sources and the copy structure are drawn before the planting.

(a) PROBE PLACEMENT, 40 worlds × 4 pairs × 20 probes at reliability 0.9. Rules compared on the same worlds and the same
    random stream: `stress` (argmax of the pointwise JS divergence between the model field and the claims field),
    `value` (regime_posterior.best_probe, the expected drop of the claims' own entropy), `random`. Score: the measure of
    the domain whose sign the posterior gets WRONG against the planted truth (closed_loop.wrong_measure, one pair).
    Also: the model entered as probes, with the lineage rule (K answers, weight 1/K, N_eff = 1) and — as the control
    that shows what the rule buys — the same K answers entered as K independent ones.
(b) THROWS THAT ARE NOT PROXIMITY, 40 worlds × 12 pairs, 66 candidate couples, 3 planted links. Rankings compared by
    precision@5 of the true hidden links: `gap-corr` (cohort-baselined correlation of the disagreement fields),
    `gap-corr@probes` (the same after 8 probes per pair by the value rule), `claims-corr` (the same on the claim
    fields alone), `resistance` (effective resistance between the two pair nodes in the pair–source claim graph,
    near = candidate), `random`.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.claim_federation import lineage_information
from graph_engine.closed_loop import wrong_measure
from graph_engine.disagreement_field import (
    LinkedWorld, claims_sign_field, disagreement_field, harvest, link_scores, model_probes, model_sign_field,
    stress_probe,
)
from graph_engine.regime_posterior import RegimePosterior
from graph_engine.resistance_sketch import ResistanceSketch

N_WORLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N_PROBES, R_PROBE = 20, 0.9
THETA_BOX = {"theta": (0.2, 0.8)}
N_FIELD = 41
PROBE_PAIRS = (0, 2, 4, 6)                    # one member of each planted couple, plus a free pair
t0 = time.time()

MODEL = model_sign_field(harvest, (0.0, 1.0), THETA_BOX, n=N_FIELD, n_p=41, name="harvest")
MODEL_COARSE = model_sign_field(harvest, (0.0, 1.0), THETA_BOX, n=21, n_p=41, name="harvest")
MODEL_101 = model_sign_field(harvest, (0.0, 1.0), THETA_BOX, n=101, n_p=41, name="harvest")


class _OnePair:
    """closed_loop.wrong_measure reads only world.sign(index, x); this exposes one pair of a world as index 0."""

    def __init__(self, world, p):
        self.w, self.p = world, p

    def sign(self, _i, x):
        return self.w.sign(self.p, x)


def posteriors(world, **kw):
    posts = [RegimePosterior(0.0, 1.0, n_grid=24, p_two=0.0, p_flip=0.7, **kw) for _ in range(world.n_pairs)]
    groups = {}
    for p, a, b, s, roots in world.claims:
        groups.setdefault((p, round(a, 6), round(b, 6), s), []).append(roots)
    for (p, a, b, s), rs in groups.items():
        posts[p].add_claim(a, b, s, n_eff=lineage_information(rs))
    return posts


def place(rule, rp, rng):
    if rule == "random":
        return float(rng.random())
    if rule == "value":
        return float(rp.best_probe(R_PROBE)[0])
    d = disagreement_field(MODEL, claims_sign_field(rp, n=N_FIELD))
    if rule == "stress":
        return float(stress_probe(d)[0])
    # "stress-open": the disagreement WEIGHTED by how open the claims still are at x, H₂(P₊^claims) in bits. The plain
    # stress rule has a failure mode this fixes: where the claims are confident and the model is certain the other way,
    # the JS stays near 1 bit no matter how often the world answers, so the rule re-probes the same x forever.
    open_ = -(np.clip(d.p_claims, 1e-12, 1 - 1e-12) * np.log2(np.clip(d.p_claims, 1e-12, 1 - 1e-12))
              + np.clip(1 - d.p_claims, 1e-12, 1) * np.log2(np.clip(1 - d.p_claims, 1e-12, 1)))
    return float(d.xs[int(np.argmax(d.js * open_))])


# ------------------------------------------------------------------------------------------------
# (a) probe placement
# ------------------------------------------------------------------------------------------------
RULES = ("stress", "stress-open", "value", "random")
curves = {r: [] for r in RULES}                 # per (world, pair): wrong measure at 0, 5, 10, 20 probes
model_arms = {"none": [], "neff1": [], "naive": []}
gaps = {"none": [], "neff1": [], "naive": []}
for w_i in range(N_WORLDS):
    world = LinkedWorld(n_pairs=8, n_links=3, seed=w_i)
    for rule in RULES:
        posts = posteriors(world)
        rng = np.random.default_rng(10_000 + w_i)
        for p in PROBE_PAIRS:
            rp, one = posts[p], _OnePair(world, p)
            row = [wrong_measure(one, [rp])]
            for k in range(N_PROBES):
                x = place(rule, rp, rng)
                rp.add_probe(x, world.probe(p, x, R_PROBE, rng), R_PROBE)
                if k + 1 in (5, 10, 20):
                    row.append(wrong_measure(one, [rp]))
            curves[rule].append(row)
    # the model entered as evidence, then 20 probes by the value rule
    for arm in model_arms:
        posts = posteriors(world)
        rng = np.random.default_rng(10_000 + w_i)
        for p in PROBE_PAIRS:
            rp, one = posts[p], _OnePair(world, p)
            if arm == "neff1":
                model_probes(rp, MODEL_COARSE, r_model=0.9, name="harvest")
            elif arm == "naive":
                for x, pm in zip(MODEL_COARSE.xs, MODEL_COARSE.p_plus):
                    if abs(pm - 0.5) > 1e-12:
                        rp.add_probe(float(x), 1 if pm >= 0.5 else -1, reliability=0.9, weight=1.0)
            for _ in range(N_PROBES):
                x = float(rp.best_probe(R_PROBE)[0])
                rp.add_probe(x, world.probe(p, x, R_PROBE, rng), R_PROBE)
            actual = wrong_measure(one, [rp])
            model_arms[arm].append(actual)
            gaps[arm].append(actual - rp.expected_error())

A = {r: np.array(curves[r], float) for r in RULES}
paired = A["stress"][:, -1] - A["value"][:, -1]
part_a = {
    "n_cases": int(len(A["value"])),
    "wrong_at": {r: [float(A[r][:, k].mean()) for k in range(4)] for r in RULES},
    "probe_counts": [0, 5, 10, 20],
    "stress_minus_value_at_20": {"mean": float(paired.mean()), "sem": float(paired.std(ddof=1) / np.sqrt(len(paired))),
                                 "stress_wins": int((paired < -1e-12).sum()), "ties": int((abs(paired) <= 1e-12).sum())},
    "model_as_evidence_wrong_at_20": {k: float(np.mean(v)) for k, v in model_arms.items()},
    "model_as_evidence_calibration_gap": {k: float(np.mean(v)) for k, v in gaps.items()},
}

# ------------------------------------------------------------------------------------------------
# (b) throws
# ------------------------------------------------------------------------------------------------
def resistance_scores(world):
    """Effective resistance between pair nodes in the pair–source claim graph (pairs 0..n−1, sources n..). Near = the
    two pairs are described by the same sources; that is the proximity rule the disagreement field is tested against."""
    n_p = world.n_pairs
    wgt = {}
    for p, _a, _b, _s, roots in world.claims:
        wgt[(p, n_p + min(roots))] = wgt.get((p, n_p + min(roots)), 0.0) + 1.0
    nodes = sorted({i for e in wgt for i in e})
    ix = {v: i for i, v in enumerate(nodes)}
    edges = np.array([[ix[a], ix[b]] for a, b in wgt], int)
    sk = ResistanceSketch.build(len(nodes), edges, np.array(list(wgt.values()), float), k=256, seed=0)
    out = np.zeros((n_p, n_p))
    for p in range(n_p):
        for q in range(p + 1, n_p):
            r = float(sk.resistance(np.array([ix[p]]), np.array([ix[q]]))[0])
            out[p, q] = out[q, p] = -r                      # near = high score
    return out


def prec_at_k(score, truth, n_p, k=5, rng=None):
    cand = [(p, q) for p in range(n_p) for q in range(p + 1, n_p)]
    if rng is not None:
        order = rng.permutation(len(cand))
        top = [cand[i] for i in order[:k]]
    else:
        top = [c for _, c in sorted(((-score[p][q], (p, q)) for p, q in cand))[:k]]
    return sum((p, q) in truth for p, q in top) / k


N_PROBES_B = 8
arms = {"gap-corr": [], "gap-corr@probes": [], "claims-corr": [], "resistance": [], "random": []}
identity_gap = 0.0
rng_b = np.random.default_rng(7)
for w_i in range(N_WORLDS):
    world = LinkedWorld(n_pairs=12, n_links=3, seed=1000 + w_i)
    posts = posteriors(world)
    fields = [claims_sign_field(rp, n=101) for rp in posts]
    ds = [disagreement_field(MODEL_101, f) for f in fields]
    C_gap = link_scores(ds)
    Cc = np.corrcoef(np.array([f.p_plus for f in fields]) - np.mean([f.p_plus for f in fields], axis=0))
    np.fill_diagonal(Cc, 0.0)
    identity_gap = max(identity_gap, float(np.abs(C_gap - Cc).max()))
    truth = world.true_links()
    # the same ranking after the loop has PROBED: 8 probes per pair by the value rule sharpen where the transition is,
    # which is the only thing θ controls — the throw is read off the posterior the engine would have anyway
    rngp = np.random.default_rng(50_000 + w_i)
    for p_i, rp in enumerate(posts):
        for _ in range(N_PROBES_B):
            x = float(rp.best_probe(R_PROBE)[0])
            rp.add_probe(x, world.probe(p_i, x, R_PROBE, rngp), R_PROBE)
    ds_p = [disagreement_field(MODEL_101, claims_sign_field(rp, n=101)) for rp in posts]
    arms["gap-corr@probes"].append(prec_at_k(link_scores(ds_p), truth, world.n_pairs))
    arms["gap-corr"].append(prec_at_k(C_gap, truth, world.n_pairs))
    arms["claims-corr"].append(prec_at_k(Cc, truth, world.n_pairs))
    arms["resistance"].append(prec_at_k(resistance_scores(world), truth, world.n_pairs))
    arms["random"].append(prec_at_k(None, truth, world.n_pairs, rng=rng_b))

part_b = {"n_worlds": N_WORLDS, "n_pairs": 12, "n_candidates": 66, "n_true_links": 3, "max_possible_p5": 0.6,
          "probes_per_pair_in_probed_arm": N_PROBES_B,
          "precision_at_5": {k: float(np.mean(v)) for k, v in arms.items()},
          "sem": {k: float(np.std(v, ddof=1) / np.sqrt(len(v))) for k, v in arms.items()},
          "gap_vs_claims_max_abs_score_difference": identity_gap}

out = {"experiment": "e37", "n_worlds": N_WORLDS, "probe_reliability": R_PROBE, "n_probes": N_PROBES,
       "part_a_probe_placement": part_a, "part_b_throws": part_b, "seconds": round(time.time() - t0, 1)}
Path(__file__).with_name("e37_results.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
