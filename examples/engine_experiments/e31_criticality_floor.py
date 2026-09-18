#!/usr/bin/env python3
"""E31: two read-outs of the same precision form that a per-part check and a sketch read cannot give.

(1) COUPLED CRITICALITY. Two graphs glued through shared concepts (graph_interface's Kron/Schur
federation). Each part is healthy — σ_min well above zero with its Laplacian gauge deflated — and the
joint form is near-singular. Planted: one shared concept both parts lean on with total conductance c
and neither measures. Reported per c: σ_min of each part, of the joint, their ratio, and where the weak
mode lives (amplitude vs energy share). Two repairs are priced: observing the LEVEL of the shared
variable (h = e_s) and observing the CONTRAST across the seam (h = e_a − e_b, one cross link).

(2) NOISE FLOOR. Every small-eigenvalue read-out from a k-column sketch carries Marchenko–Pastur
noise: the bulk of (1/k)ZZᵀ reaches σ²(1+√(n/k))². Reported: the floor's own calibration on a pure
noise Gram, and the share of a sketched DPP kernel (300 nodes, k = 16/64/256) that lies inside the
bulk — i.e. how much of that kernel is sketch noise rather than graph.

(3) The two meet: the criticality read-out of the joint form is itself taken through a sketch. The
weak mode is a spike in C = J⁺; at which k does that spike clear its own MP floor?
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.precision_form import PrecisionForm, eigen_readout, joint_criticality, mp_floor
from graph_engine.resistance_sketch import ResistanceSketch

out = {}


def part(n_interior, w_in, c_shared, n_shared=1, lean=None):
    """Interior 0..n−1 strongly connected (weight w_in), shared concepts n..n+n_shared−1 attached to
    every interior node with total conductance c_shared each (lean = per-shared override)."""
    d = n_interior + n_shared
    f = PrecisionForm.zeros(d)
    inside = [(i, j) for i in range(n_interior) for j in range(i + 1, n_interior)]
    f.add_laplacian(inside, [w_in] * len(inside))
    for s in range(n_shared):
        c = c_shared if lean is None else lean[s]
        f.add_laplacian([(i, n_interior + s) for i in range(n_interior)], [c / n_interior] * n_interior)
    return f


def random_connected_graph(n, extra, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    edges = [(int(perm[i]), int(perm[i + 1])) for i in range(n - 1)]
    seen = {frozenset(e) for e in edges}
    while len(edges) < n - 1 + extra:
        i, j = rng.integers(0, n, 2)
        if i != j and frozenset((int(i), int(j))) not in seen:
            seen.add(frozenset((int(i), int(j)))); edges.append((int(i), int(j)))
    return np.array(edges), rng.uniform(0.3, 3.0, len(edges))


def corr(M):
    d = np.sqrt(np.clip(np.diag(M), 1e-300, None))
    return M / np.outer(d, d)


# -- (1) planted shared-variable weakness ------------------------------------------------------------
nA = nB = 20
rows = {}
for c in (1.0, 0.3, 0.1, 0.03):
    A, B = part(nA, 5.0, c), part(nB, 5.0, c)
    r = joint_criticality(A, B, {nA: nB})
    j, s = r["joint"], r["shared"][0]
    lvl = {}
    for sigma in (1.0, 0.05):
        f = j.copy(); h = np.zeros(f.d); h[s["joint"]] = 1.0
        sm = f.observe(h, sigma)
        lvl[f"sigma={sigma}"] = {"sigma_min": round(f.criticality()["sigma_min"], 6), "sherman_morrison": sm}
    f = j.copy(); h = np.zeros(f.d); h[0] = 1.0; h[int(r["b_map"][0])] = -1.0
    sm_contrast = f.observe(h, 1.0)
    exact = float(np.abs(f.cov() - np.linalg.pinv(f.J, rcond=f.tol, hermitian=True)).max())
    rows[f"c={c}"] = {
        "sigma_min_a": round(r["sigma_min_a"], 6), "sigma_min_b": round(r["sigma_min_b"], 6),
        "sigma_min_joint": round(r["sigma_min_joint"], 6), "ratio_joint_over_part": round(r["ratio"], 5),
        "shared_amplitude": round(s["amplitude"], 9), "shared_energy_share": round(s["energy_share"], 6),
        "max_energy_share_elsewhere": round(float(np.delete(r["report"]["energy_share"], s["joint"]).max()), 6),
        "mass_in_a_interior": round(float((r["report"]["direction"][r["a_map"]] ** 2).sum()), 4),
        "repair_level_e_s": lvl,
        "repair_contrast_e_a_minus_e_b": {"sigma_min": round(f.criticality()["sigma_min"], 6),
                                          "sherman_morrison": sm_contrast, "sm_vs_pinv_max_abs": exact},
    }
out["planted_shared_weakness"] = rows

# asymmetric leans and two shared concepts: which one carries the weak mode?
A = part(nA, 5.0, 0.0, n_shared=2, lean=[1.0, 3.0])
B = part(nB, 5.0, 0.0, n_shared=2, lean=[0.1, 3.0])
r = joint_criticality(A, B, {nA: nB, nA + 1: nB + 1})
out["two_shared_concepts"] = {
    "leans_a": [1.0, 3.0], "leans_b": [0.1, 3.0],
    "sigma_min_a": round(r["sigma_min_a"], 6), "sigma_min_b": round(r["sigma_min_b"], 6),
    "sigma_min_joint": round(r["sigma_min_joint"], 6), "ratio": round(r["ratio"], 5),
    "shared_ranked": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items()} for s in r["shared"]],
}

# -- (2) the noise floor -----------------------------------------------------------------------------
rng = np.random.default_rng(0)
cal = {}
for n, k in ((200, 800), (100, 400), (300, 1200)):
    Z = rng.standard_normal((n, k)) * 1.3
    rr = eigen_readout(Z @ Z.T / k, k, sigma2=1.3 ** 2)
    cal[f"n={n},k={k}"] = {"floor": round(rr["floor"], 4), "top": round(rr["top"], 4),
                           "top_over_floor": round(rr["top"] / rr["floor"], 4),
                           "share_below": round(rr["share_below"], 4)}
u = rng.standard_normal(200); u /= np.linalg.norm(u)
theta = 10 * mp_floor(200, 800, 1.3 ** 2)
Z = rng.standard_normal((200, 800)) * 1.3 + np.sqrt(theta) * np.outer(u, rng.standard_normal(800))
rr = eigen_readout(Z @ Z.T / 800, 800, sigma2=1.3 ** 2)
cal["planted_spike_10x_edge"] = {"n_above_floor": rr["n_above"], "top": round(rr["top"], 3),
                                 "floor": round(rr["floor"], 3),
                                 "overlap_with_planted_direction": round(float(abs(
                                     np.linalg.eigh(Z @ Z.T / 800)[1][:, -1] @ u)), 4)}
out["mp_floor_calibration"] = cal

n = 300
edges, w = random_connected_graph(n, 600, 3)
form = PrecisionForm.zeros(n).add_laplacian(edges, w)
K_exact = form._dpp_kernel(np.arange(n))
dpp = {}
for k in (16, 64, 256):
    Z = ResistanceSketch.build(n, edges, w, k=k, seed=1).Z
    K = Z @ Z.T
    plug, norm = eigen_readout(K, k), eigen_readout(corr(K), k, sigma2=1.0)
    ex = eigen_readout(corr(K_exact), k, sigma2=1.0)
    dpp[f"k={k}"] = {
        "gamma_n_over_k": round(plug["gamma"], 4),
        "plugin_sigma2": {"floor": round(plug["floor"], 4), "n_below": plug["n_below"],
                          "share_below": round(plug["share_below"], 4), "top": round(plug["top"], 4)},
        "unit_diagonal": {"floor": round(norm["floor"], 4), "n_below": norm["n_below"],
                          "share_below": round(norm["share_below"], 4), "top": round(norm["top"], 4),
                          "top_over_floor": round(norm["top"] / norm["floor"], 4)},
        "exact_kernel_same_floor": {"n_below": ex["n_below"], "top": round(ex["top"], 4)},
        "mean_rel_err_hole_field": round(float(np.abs(np.diag(K) / np.diag(K_exact) - 1).mean()), 4),
    }
out["dpp_kernel_sketch_noise"] = {"n_nodes": n, "n_edges": len(edges), "per_k": dpp}

# -- (3) reading the criticality itself through a sketch ---------------------------------------------
A, B = part(nA, 5.0, 1.0), part(nB, 5.0, 1.0)
r = joint_criticality(A, B, {nA: nB})
J = r["joint"].J
dj = len(J)
off = -J.copy(); np.fill_diagonal(off, 0.0)
ii, jj = np.triu_indices(dj, 1)
keep = off[ii, jj] > 1e-12
ge, gw = np.stack([ii[keep], jj[keep]], 1), off[ii, jj][keep]
Cex = r["joint"].cov()
seam = {}
for k in (16, 64, 256, 1024):
    Zs = ResistanceSketch.build(dj, ge, gw, k=k, seed=2).Z
    Cs = Zs @ Zs.T
    rr = eigen_readout(corr(Cs), k, sigma2=1.0)
    lam_s = np.linalg.eigvalsh(Cs)[-1]
    seam[f"k={k}"] = {"floor": round(rr["floor"], 4), "top_corr": round(rr["top"], 4),
                      "top_over_floor": round(rr["top"] / rr["floor"], 4),
                      "spike_clears_floor": bool(rr["top"] > rr["floor"]),
                      "sigma_min_from_sketch": round(float(1.0 / lam_s), 6)}
out["criticality_through_a_sketch"] = {
    "n_joint": dj, "sigma_min_exact": round(r["sigma_min_joint"], 6),
    "lambda_max_C_exact": round(float(np.linalg.eigvalsh(Cex)[-1]), 4), "per_k": seam}

print(json.dumps(out, indent=1))
json.dump(out, open(Path(__file__).parent / "e31_results.json", "w"), indent=1)
