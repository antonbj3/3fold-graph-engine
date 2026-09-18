#!/usr/bin/env python3
"""E35: the EXACT value of a SET of yes/no probes — no Gaussian image.

e24's T4 left one inexact step in `precision_form`: the Bernoulli block is a LOCAL Gaussian image, so
it ranks belief probes as regime_posterior does (Spearman 0.9999) but its bits are off by up to 43 %
(median 15 %) against the exact two-outcome entropy drop. For a SET the approximation need not be paid
at all: the exact value is a finite sum over outcome patterns.

A (a) INDEPENDENT BELIEFS: the joint posterior factorizes, so the exact set value is the SUM of the
    single exact values. Checked against a brute joint enumeration over 2^m truths × 2^k outcomes and
    against Monte Carlo.
B (b) ONE BELIEF, several probes: exact by enumeration over 2^k outcome patterns, equal to the chain
    rule Σ_k E[value of probe k | answers 1..k−1] in every order.
C (c) MIXED: sum over beliefs of (b) — `exact_bernoulli_set_value`.
D     the SPHERE form: the same set in the flat Fisher coordinate θ = 2 arcsin√p, second-order term
    Var(Δθ)/(2 ln 2), and its relative error against the exact value over a (p, r) grid.
E (d) ROUTING: `set_value_bits` takes the exact route when every row is a probe on an independent
    registered belief. e24's own T6 instance re-read: ranking unchanged, bits exact; and the residual
    of the mixed Gaussian+Bernoulli set, which still takes the Gaussian image.
F (e) SUBMODULARITY of the exact set value on one belief, 500 random (p, r, A ⊆ B, e).
"""
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
from graph_engine.precision_form import (  # noqa: E402
    PrecisionForm, bernoulli_exact_bits, bernoulli_probe, exact_bernoulli_chain,
    exact_bernoulli_one_belief, exact_bernoulli_set_value, sphere_error_grid, sphere_set_value,
    theta_of_p,
)
import test_precision_form as T  # noqa: E402

out = {}
LOG2 = np.log(2.0)


def _ent(p):                                        # entropy of a probability vector, bits
    p = np.asarray(p, float); p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def brute_joint_value(beliefs, probes) -> float:
    """Independent reference: the full 2^m × 2^k table. Value = H(X) − Σ_y P(y) H(X|y), X the joint
    truth vector of all m beliefs, y the joint answer pattern of all k probes."""
    m, k = len(beliefs), len(probes)
    x = ((np.arange(1 << m)[:, None] >> np.arange(m)[None, :]) & 1).astype(float)
    p = np.asarray(beliefs, float)
    prior = np.prod(np.where(x > 0, p, 1 - p), axis=1)
    y = ((np.arange(1 << k)[:, None] >> np.arange(k)[None, :]) & 1).astype(float)
    lik = np.ones((1 << m, 1 << k))
    for j, (b, r) in enumerate(probes):
        agree = (x[:, None, int(b)] == y[None, :, j])
        lik *= np.where(agree, float(r), 1.0 - float(r))
    joint = prior[:, None] * lik
    Py = joint.sum(0)
    post = joint / np.where(Py > 0, Py, 1.0)
    return float(_ent(prior) - float(np.sum(Py * np.array([_ent(post[:, c]) for c in range(1 << k)]))))


def mc_value(beliefs, probes, n=400_000, seed=0) -> float:
    """Monte-Carlo estimate of the same quantity: sample the truth, sample the answers, average the
    entropy of the exact posterior given the sampled answer pattern."""
    rng = np.random.default_rng(seed)
    m = len(beliefs)
    p = np.asarray(beliefs, float)
    x = (rng.random((n, m)) < p).astype(int)
    ys = []
    for b, r in probes:
        truth = x[:, int(b)]
        flip = rng.random(n) > float(r)
        ys.append(np.where(flip, 1 - truth, truth))
    y = np.stack(ys, 1) if ys else np.zeros((n, 0), int)
    k = len(probes)
    code = (y * (1 << np.arange(k))).sum(1) if k else np.zeros(n, int)
    xs = ((np.arange(1 << m)[:, None] >> np.arange(m)[None, :]) & 1).astype(float)
    prior = np.prod(np.where(xs > 0, p, 1 - p), axis=1)
    yb = ((np.arange(1 << k)[:, None] >> np.arange(k)[None, :]) & 1).astype(float)
    lik = np.ones((1 << m, 1 << k))
    for j, (b, r) in enumerate(probes):
        agree = (xs[:, None, int(b)] == yb[None, :, j])
        lik *= np.where(agree, float(r), 1.0 - float(r))
    joint = prior[:, None] * lik
    Py = joint.sum(0)
    post = joint / np.where(Py > 0, Py, 1.0)
    Hpost = np.array([_ent(post[:, c]) for c in range(1 << k)])
    return float(_ent(prior) - float(np.mean(Hpost[code])))


# -- A: additivity over independent beliefs ----------------------------------------------------------
rng = np.random.default_rng(0)
worst_add, worst_brute, cases = 0.0, 0.0, 0
for _ in range(200):
    m = int(rng.integers(2, 5))
    beliefs = list(rng.uniform(0.05, 0.95, m))
    probes = [(int(b), float(rng.uniform(0.55, 0.99))) for b in range(m)]
    s = sum(bernoulli_exact_bits(beliefs[b], r) for b, r in probes)
    worst_add = max(worst_add, abs(exact_bernoulli_set_value(beliefs, probes) - s))
    worst_brute = max(worst_brute, abs(brute_joint_value(beliefs, probes) - s))
    cases += 1
mc_cases = []
for seed in range(3):
    rg = np.random.default_rng(100 + seed)
    beliefs = list(rg.uniform(0.15, 0.85, 3))
    probes = [(b, float(rg.uniform(0.6, 0.95))) for b in range(3)]
    ex = exact_bernoulli_set_value(beliefs, probes)
    mc_cases.append({"exact_bits": ex, "monte_carlo_bits": mc_value(beliefs, probes, 400_000, seed),
                     "n_samples": 400_000})
out["A_independent_beliefs_additivity"] = {
    "instances": cases, "beliefs_per_instance": "2..4, one probe each",
    "max_abs_error_sum_vs_exact_set_value_bits": worst_add,
    "max_abs_error_sum_vs_brute_joint_enumeration_bits": worst_brute,
    "monte_carlo": [{**c, "abs_error_bits": abs(c["exact_bits"] - c["monte_carlo_bits"])} for c in mc_cases],
    "claim": "independent beliefs ⇒ the joint posterior factorizes ⇒ the exact set value is the SUM "
             "of the single exact values, with no cross term"}

# -- B: one belief, several probes: enumeration = chain rule in every order ---------------------------
rng = np.random.default_rng(1)
worst_chain, worst_order, worst_brute_b, ks = 0.0, 0.0, 0.0, []
for _ in range(120):
    k = int(rng.integers(1, 7))
    p = float(rng.uniform(0.05, 0.95))
    rels = list(rng.uniform(0.55, 0.99, k))
    ex = exact_bernoulli_one_belief(p, rels)
    worst_chain = max(worst_chain, abs(exact_bernoulli_chain(p, rels)[0] - ex))
    worst_brute_b = max(worst_brute_b, abs(brute_joint_value([p], [(0, r) for r in rels]) - ex))
    for perm in list(permutations(range(k)))[:6]:
        worst_order = max(worst_order, abs(exact_bernoulli_chain(p, [rels[i] for i in perm])[0] - ex))
    ks.append(k)
k12 = list(np.random.default_rng(2).uniform(0.55, 0.95, 12))
big = exact_bernoulli_one_belief(0.37, k12)
out["B_one_belief_chain_rule"] = {
    "instances": 120, "k_range": [int(min(ks)), int(max(ks))],
    "max_abs_error_chain_rule_vs_enumeration_bits": worst_chain,
    "max_abs_error_over_orderings_bits": worst_order,
    "max_abs_error_vs_brute_joint_enumeration_bits": worst_brute_b,
    "k12_instance": {"p": 0.37, "k": 12, "exact_bits": big, "prior_entropy_bits": float(
        -(0.37 * np.log2(0.37) + 0.63 * np.log2(0.63))),
        "sum_of_single_values_bits": float(sum(bernoulli_exact_bits(0.37, r) for r in k12)),
        "note": "the set value is far BELOW the sum of the single values: probes on one belief overlap"},
    "claim": "value(S) = Σ_k E[value of probe k | answers 1..k−1] telescopes to h(p) − E h(p(y)), "
             "so enumeration and any sequential order agree"}

# -- C: the mixed case ------------------------------------------------------------------------------
beliefs_c = [0.2, 0.5, 0.8]
probes_c = [(0, 0.7), (0, 0.9), (1, 0.6), (1, 0.6), (1, 0.85), (2, 0.95)]
per_belief = {b: exact_bernoulli_one_belief(beliefs_c[b], [r for bb, r in probes_c if bb == b])
              for b in range(3)}
out["C_mixed_sets"] = {
    "beliefs": beliefs_c, "probes": [[b, r] for b, r in probes_c],
    "per_belief_bits": {str(b): v for b, v in per_belief.items()},
    "exact_set_value_bits": exact_bernoulli_set_value(beliefs_c, probes_c),
    "sum_of_per_belief_bits": float(sum(per_belief.values())),
    "brute_joint_enumeration_bits": brute_joint_value(beliefs_c, probes_c),
    "abs_error_vs_brute_bits": abs(exact_bernoulli_set_value(beliefs_c, probes_c)
                                   - brute_joint_value(beliefs_c, probes_c)),
    "sum_of_all_single_probe_values_bits": float(sum(bernoulli_exact_bits(beliefs_c[b], r) for b, r in probes_c))}

# -- D: the sphere (θ) second-order term and what it costs -------------------------------------------
ps = np.round(np.linspace(0.02, 0.98, 49), 6)
rels = [0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
grids = {}
for k in (1, 2, 3):
    g = sphere_error_grid(ps, rels, k=k)
    gr = sphere_error_grid(ps, rels, k=k, moment="raw")
    grids[str(k)] = {"var_form": g, "raw_second_moment_form": {"max_rel_error": gr["max_rel_error"],
                                                               "median_rel_error": gr["median_rel_error"]}}
g1 = grids["1"]["var_form"]
p_eq = 0.5
small = {"p": 0.5, "r": 0.505,
         "exact_bits": exact_bernoulli_set_value([p_eq], [(0, 0.505)]),
         "sphere_bits": sphere_set_value([p_eq], [(0, 0.505)]),
         "rel_error": abs(sphere_set_value([p_eq], [(0, 0.505)]) / exact_bernoulli_set_value([p_eq], [(0, 0.505)]) - 1)}
# the Gaussian image of e24 on the same grid, for comparison of the two approximations
gauss_err = []
for p in ps:
    for r in rels:
        h, s = bernoulli_probe(float(p), float(r))
        gb = PrecisionForm.zeros(1).add_bernoulli(0, float(p)).set_value_bits([[h]], s, exact_bernoulli=False)
        gauss_err.append(abs(gb / exact_bernoulli_set_value([float(p)], [(0, float(r))]) - 1))
out["D_sphere_second_order"] = {
    "coordinate": "theta = 2 arcsin sqrt(p); ds = dtheta (the Fisher metric is flat, a quarter circle)",
    "second_order_term": "Var(dtheta) / (2 ln 2) bits",
    "theta_range": [float(theta_of_p(0.0)), float(theta_of_p(1.0))],
    "grid": {"n_p": len(ps), "p_range": [float(ps[0]), float(ps[-1])], "reliabilities": rels},
    "per_k": grids,
    "max_rel_error_k1": g1["max_rel_error"], "median_rel_error_k1": g1["median_rel_error"],
    "argmax_k1": {"p": g1["argmax_p"], "r": g1["argmax_r"], "exact_bits": g1["exact_at_argmax_bits"],
                  "sphere_bits": g1["sphere_at_argmax_bits"]},
    "max_rel_error_k1_at_r_le_0.9": g1["max_rel_error_at_r_le_0.9"],
    "small_step_limit": small,
    "gaussian_image_same_grid": {"max_rel_error": float(np.max(gauss_err)),
                                 "median_rel_error": float(np.median(gauss_err))},
    "finding": "the second-order sphere term OVERSHOOTS (the Jensen gap truncated at second order) "
               "where the Gaussian image UNDERSHOOTS (its t/(1+t) saturation); both are exact only as "
               "the step shrinks, and the exact two-outcome form costs the same two evaluations"}

# -- E: routing, on e24's own T6 instance -------------------------------------------------------------
f6, cands6, meta6, _ = T.unified_instance()
old = {i: (v, vc) for i, v, vc, _ in f6.rank(cands6)}
new = {i: (v, vc) for i, v, vc, _ in f6.rank(cands6, exact_bernoulli=True)}
bern_ids = [c.id for c in cands6 if c.kind == "regime"]
order_old = [i for i, _, _, _ in f6.rank(cands6) if i in bern_ids]
order_new = [i for i, _, _, _ in f6.rank(cands6, exact_bernoulli=True) if i in bern_ids]
exact_ref = {c.id: meta6[c.id]["exact_bits"] for c in cands6 if c.kind == "regime"}
non_bern_unchanged = max(abs(old[i][0] - new[i][0]) for i in old if i not in bern_ids)
# the MIXED set: one structure row, one margin row, one belief probe — still the Gaussian image
mix = [c for c in cands6 if c.kind == "structure"][:1] + [c for c in cands6 if c.kind == "margin"][:1] \
      + [c for c in cands6 if c.id == "probe_r0.95"]
H_mix = np.array([c.h for c in mix]); s_mix = np.array([c.sigma for c in mix])
mix_gauss = f6.set_value_bits(H_mix, s_mix)                       # rows are not all Bernoulli: no route
gauss_only = f6.set_value_bits(H_mix[:2], s_mix[:2])
hybrid = gauss_only + exact_ref["probe_r0.95"]                    # exact, since the belief is independent in C
out["E_routing_on_T6"] = {
    "n_candidates": len(cands6),
    "bernoulli_order_gaussian_image": order_old, "bernoulli_order_exact": order_new,
    "bernoulli_order_unchanged": order_old == order_new,
    "bits": {i: {"gaussian_image": old[i][0], "exact_route": new[i][0], "exact_two_outcome_reference":
                 exact_ref[i], "abs_error_exact_route_vs_reference": abs(new[i][0] - exact_ref[i]),
                 "rel_gap_gaussian_vs_exact": abs(old[i][0] / exact_ref[i] - 1)} for i in bern_ids},
    "max_change_of_non_bernoulli_bits": float(non_bern_unchanged),
    "joint_top1_gaussian": f6.rank(cands6)[0][0], "joint_top1_exact": f6.rank(cands6, exact_bernoulli=True)[0][0],
    "rank_of_best_bernoulli_gaussian": 1 + [i for i, _, _, _ in f6.rank(cands6)].index(order_old[0]),
    "rank_of_best_bernoulli_exact": 1 + [i for i, _, _, _ in f6.rank(cands6, exact_bernoulli=True)].index(order_new[0]),
    "mixed_set": {"members": [c.id for c in mix], "gaussian_image_bits": mix_gauss,
                  "gaussian_rows_only_bits": gauss_only, "exact_hybrid_bits": hybrid,
                  "residual_abs_bits": abs(mix_gauss - hybrid), "residual_rel": abs(mix_gauss / hybrid - 1),
                  "belief_coupling_in_C": float(np.max(np.abs(np.delete(f6.cov()[14], 14)))),
                  "note": "a mixed Gaussian+Bernoulli set keeps the Gaussian image; the residual is the "
                          "belief row's own e24 gap, and it does not vanish with the set"}}

# -- F: submodularity of the exact set value on one belief -------------------------------------------
rng = np.random.default_rng(35)
viol, checks, gaps, strict = 0, 0, [], 0
for _ in range(500):
    p = float(rng.uniform(0.02, 0.98))
    n = int(rng.integers(2, 8))
    rel = list(rng.uniform(0.5, 0.999, n))
    idx = list(range(n))
    rng.shuffle(idx)
    e = idx[0]
    rest = idx[1:]
    nb = int(rng.integers(0, len(rest) + 1))
    B = rest[:nb]
    na = int(rng.integers(0, nb + 1))
    A = B[:na]
    strict += int(na < nb)
    gA = exact_bernoulli_one_belief(p, [rel[i] for i in A + [e]]) - exact_bernoulli_one_belief(p, [rel[i] for i in A])
    gB = exact_bernoulli_one_belief(p, [rel[i] for i in B + [e]]) - exact_bernoulli_one_belief(p, [rel[i] for i in B])
    d = gA - gB                                    # diminishing returns: A ⊆ B ⇒ gain(A) ≥ gain(B)
    gaps.append(float(d))
    viol += int(d < -1e-12)
    checks += 1
mono = all(exact_bernoulli_one_belief(0.4, [0.8] * k) <= exact_bernoulli_one_belief(0.4, [0.8] * (k + 1)) + 1e-15
           for k in range(1, 10))
out["F_submodularity"] = {
    "checks": checks, "violations": viol,
    "min_diminishing_returns_gap_bits": float(np.min(gaps)),
    "max_gap_bits": float(np.max(gaps)), "median_gap_bits": float(np.median(gaps)),
    "n_strict_gaps": int(np.sum(np.array(gaps) > 1e-12)), "n_A_strictly_inside_B": strict,
    "always_satisfied": viol == 0, "monotone_in_k_ok": bool(mono),
    "theorem": "for observations conditionally independent GIVEN the truth X, S -> I(X;Y_S) is monotone "
               "submodular: the gain I(X;Y_e|Y_S) = H(Y_e|Y_S) − H(Y_e|X) has a subtrahend free of S and "
               "a first term non-increasing in S (Krause & Guestrin 2005). Greedy is within 1 − 1/e.",
    "diminishing_returns_example": {
        "p": 0.4, "r": 0.8,
        "gain_of_the_k_th_identical_probe_bits": [
            float(exact_bernoulli_one_belief(0.4, [0.8] * k) - exact_bernoulli_one_belief(0.4, [0.8] * (k - 1)))
            for k in range(1, 9)]}}

Path(__file__).with_name("e35_results.json").write_text(json.dumps(out, indent=1, sort_keys=False))
print(json.dumps(out, indent=1))
