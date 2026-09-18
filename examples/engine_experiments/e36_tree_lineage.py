#!/usr/bin/env python3
"""e36 — the three lineage states are ONE object: a tree of error components.

e_k = sigma_k * SUM_{v in path(k)} sqrt(theta_v) eps_v, SUM theta_v = 1 along every path  ->  Sigma = D T Tt D.
independent = own leaf theta 1; copy = theta 0 under a shared node; partially shared = an ancestor with theta = rho.

What this script measures:
 (1) tree_gls (O(n*depth), nested random effects, no Sigma) == the pinv route of MarginNet.estimate, on random trees
     including exact copies (Sigma singular), and the wall-clock of both at n = 2 000 reports.
 (2) N_eff: copies only + equal sigma -> claim_federation.lineage_information = ||M+1||^2; one shared ancestor with rho
     over m reports -> Kish m/(1+(m-1)rho).
 (3) the limits: rho -> 0 the independent case, rho -> 1 the copy case, s monotone in between.
 (4) COVERAGE with a KNOWN truth (what e23/e26 could not measure: there the truth is unknown). Two-level
     (collaboration -> paper) and three-level (collaboration -> detector -> paper) synthetic data, 500 draws:
     recovery of the planted mean and the coverage of mhat +- 1.96 s under the true structure and under each
     mis-declaration (declaring copies when it is rho = 0.5; declaring independent when it is copies; ...).
Writes e36_results.json. Seeds fixed; numpy/scipy only.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.claim_federation import lineage_information          # noqa: E402
from graph_engine.margin_net import (MarginNet, tree_covariance, tree_from_shares,  # noqa: E402
                                     tree_gls, tree_matrix)

OUT = Path(__file__).with_name("e36_results.json")


# ---------------------------------------------------------------------------------------------------
def pinv_gls(nodes, sig, m, tree):
    S = tree_covariance(nodes, sig, tree); Sp = np.linalg.pinv(S, rcond=1e-10)
    one = np.ones(len(m)); info = float(one @ Sp @ one); mh = float(one @ Sp @ m) / info
    q = max(float(m @ Sp @ m) - mh * mh * info, 0.0)
    return mh, math.sqrt(1.0 / info), info, q


def random_tree(rng, n_reports, depth, copies):
    par = {"root": None}; layers = [["root"]]
    for d in range(depth):
        new = []
        for p in layers[-1]:
            for c in range(int(rng.integers(1, 3))):
                v = f"n{d}_{len(new)}"; par[v] = p; new.append(v)
        layers.append(new)
    sh = rng.dirichlet(np.ones(depth + 1))
    if copies:
        sh[-2] += sh[-1]; sh[-1] = 0.0
    tree = [{"id": v, "parent": par[v], "share": float(sh[d])} for d, layer in enumerate(layers) for v in layer]
    nodes = [layers[-1][int(rng.integers(len(layers[-1])))] for _ in range(n_reports)]
    return tree, nodes


def identities():
    rng = np.random.default_rng(0)
    err = {"m": 0.0, "s": 0.0, "q": 0.0}; n_sing = 0; n_tr = 200
    for t in range(n_tr):
        tree, nodes = random_tree(rng, int(rng.integers(3, 14)), int(rng.integers(1, 5)), copies=bool(t % 2))
        sig = rng.uniform(0.05, 0.5, len(nodes)); m = rng.normal(0.3, 0.2, len(nodes))
        r = tree_gls([{"margin": m[k], "sigma": sig[k], "node": nodes[k]} for k in range(len(nodes))], tree)
        a, b, info, q = pinv_gls(nodes, sig, m, tree)
        n_sing += int(np.linalg.matrix_rank(tree_covariance(nodes, sig, tree), tol=1e-10) < len(nodes))
        err["m"] = max(err["m"], abs(r["m"] - a)); err["s"] = max(err["s"], abs(r["s"] - b))
        err["q"] = max(err["q"], abs(r["q"] - q) / max(1.0, abs(q)))
    # MarginNet.add_sources on tree records == tree_gls
    vals = [0.30, 0.42, 0.20]
    tree = [{"id": "CDF", "parent": None, "share": 0.5}] + [{"id": f"p{k}", "parent": "CDF", "share": 0.5} for k in range(3)]
    n = MarginNet(default_sigma=0.1); n.add_sources(tree)
    n.add_edge("e", ["a", "b"], [{"margin": v, "sources": [f"p{k}"]} for k, v in enumerate(vals)])
    est = n.estimate("e"); g = tree_gls([{"margin": v, "sigma": 0.1, "node": f"p{k}"} for k, v in enumerate(vals)], tree)
    sh = MarginNet(default_sigma=0.1); sh.add_sources([{"id": f"p{k}", "shares": {"CDF": 0.5}} for k in range(3)])
    sh.add_edge("e", ["a", "b"], [{"margin": v, "sources": [f"p{k}"]} for k, v in enumerate(vals)])
    # two exact copies with different sigma: 1 is outside range(Sigma)
    ctree = [{"id": "o", "parent": None, "share": 1.0}, {"id": "a", "parent": "o", "share": 0.0},
             {"id": "b", "parent": "o", "share": 0.0}]
    cr = tree_gls([{"margin": 0.4, "sigma": 0.1, "node": "a"}, {"margin": 0.6, "sigma": 0.2, "node": "b"}], ctree)
    return {"n_trees": n_tr, "n_with_singular_sigma": n_sing,
            "max_abs_err_mhat": err["m"], "max_abs_err_s": err["s"], "max_rel_err_chi2": err["q"],
            "marginnet_tree_vs_tree_gls_abs_err": max(abs(est.m - g["m"]), abs(est.s - g["s"]), abs(est.n_eff - g["n_eff"])),
            "tree_rho_vs_old_shares_records_abs_err": max(abs(sh.estimate("e").m - est.m), abs(sh.estimate("e").s - est.s)),
            "two_copies_sigma_0.1_0.2": {"mhat": cr["m"], "closed_form_mhat": (0.1 * 0.4 + 0.2 * 0.6) / 0.3,
                                         "s": cr["s"], "closed_form_s": (0.01 + 0.04) / 0.3,
                                         "note": "sigma-weighted, not 1/sigma^2-weighted: that is what the pinv does when 1 is outside range(Sigma)"}}


def timing(n_reports=2000, n_groups=50, rho=0.5, repeat=3):
    rng = np.random.default_rng(1)
    groups = [f"g{k % n_groups}" for k in range(n_reports)]
    tree, nodes = tree_from_shares(groups, rho)
    sig = rng.uniform(0.05, 0.3, n_reports); m = rng.normal(0.3, 0.2, n_reports)
    reps = [{"margin": m[k], "sigma": sig[k], "node": nodes[k]} for k in range(n_reports)]
    t0 = time.perf_counter()
    for _ in range(repeat):
        r = tree_gls(reps, tree)
    t_tree = (time.perf_counter() - t0) / repeat
    t0 = time.perf_counter()
    a, b, info, q = pinv_gls(nodes, sig, m, tree)
    t_pinv = time.perf_counter() - t0
    # scaling of the recursion
    scal = {}
    for n in (250, 500, 1000, 2000, 4000, 8000):
        gg = [f"g{k % n_groups}" for k in range(n)]
        tr, nd = tree_from_shares(gg, rho)
        rr = [{"margin": float(m[k % n_reports]), "sigma": float(sig[k % n_reports]), "node": nd[k]} for k in range(n)]
        t0 = time.perf_counter(); tree_gls(rr, tr); scal[n] = time.perf_counter() - t0
    return {"n_reports": n_reports, "n_groups": n_groups, "rho": rho, "depth": 2,
            "t_tree_gls_s": t_tree, "t_pinv_s": t_pinv, "speedup": t_pinv / t_tree,
            "abs_err_mhat": abs(r["m"] - a), "abs_err_s": abs(r["s"] - b),
            "tree_gls_seconds_by_n": scal}


def neff_checks():
    rng = np.random.default_rng(2); copies = []
    for origins, per in ((3, 2), (4, 1), (2, 5), (5, 3), (10, 4)):
        tree = [{"id": f"o{j}", "parent": None, "share": 1.0} for j in range(origins)]
        nodes = []
        for j in range(origins):
            for c in range(per):
                tree.append({"id": f"c{j}_{c}", "parent": f"o{j}", "share": 0.0}); nodes.append(f"c{j}_{c}")
        m = rng.normal(0.2, 0.05, len(nodes))
        r = tree_gls([{"margin": m[k], "sigma": 0.1, "node": nodes[k]} for k in range(len(nodes))], tree)
        li = lineage_information([frozenset([n.split("_")[0].replace("c", "o")]) for n in nodes])
        copies.append({"origins": origins, "copies_each": per, "n_reports": len(nodes),
                       "n_eff_tree": r["n_eff"], "lineage_information": li, "abs_err": abs(r["n_eff"] - li)})
    kish = []
    for m_rep in (2, 3, 10, 50, 200):
        for rho in (0.05, 0.2, 0.5, 0.9):
            tree, nodes = tree_from_shares(["g"] * m_rep, rho)
            r = tree_gls([{"margin": 0.3, "sigma": 0.2, "node": n} for n in nodes], tree)
            k = m_rep / (1 + (m_rep - 1) * rho)
            kish.append({"m": m_rep, "rho": rho, "n_eff_tree": r["n_eff"], "kish": k, "abs_err": abs(r["n_eff"] - k)})
    return {"copies_only_equal_sigma": copies, "max_abs_err_vs_lineage_information": max(c["abs_err"] for c in copies),
            "one_shared_ancestor": kish, "max_abs_err_vs_kish": max(k["abs_err"] for k in kish)}


def rho_limits(m_rep=6, sig=0.15):
    out = []
    for rho in np.linspace(0.0, 1.0, 21):
        tree, nodes = tree_from_shares(["g"] * m_rep, float(rho))
        r = tree_gls([{"margin": 0.3, "sigma": sig, "node": n} for n in nodes], tree)
        out.append({"rho": float(rho), "s": r["s"], "n_eff": r["n_eff"]})
    s = [o["s"] for o in out]
    return {"m": m_rep, "sigma": sig, "curve": out, "s_at_rho0": s[0], "independent_s": sig / math.sqrt(m_rep),
            "s_at_rho1": s[-1], "copy_s": sig, "monotone_increasing": all(b > a for a, b in zip(s, s[1:]))}


# ---------------------------------------------------------------------------------------------------
# (4) coverage with a KNOWN truth
# ---------------------------------------------------------------------------------------------------
def build_tree(structure, n_coll, n_det, n_pap):
    """structure = tuple of shares (collaboration, detector, paper); a share of 0 at the paper level = copies."""
    th_c, th_d, th_p = structure
    tree, nodes = [], []
    for c in range(n_coll):
        tree.append({"id": f"C{c}", "parent": None, "share": th_c})
        for d in range(n_det):
            tree.append({"id": f"C{c}D{d}", "parent": f"C{c}", "share": th_d})
            for p in range(n_pap):
                tree.append({"id": f"C{c}D{d}P{p}", "parent": f"C{c}D{d}", "share": th_p})
                nodes.append(f"C{c}D{d}P{p}")
    return tree, nodes


def coverage(truth, declared, n_coll, n_det, n_pap, sigma=0.1, mu=0.3, draws=500, seed=7):
    rng = np.random.default_rng(seed)
    t_tree, nodes = build_tree(truth, n_coll, n_det, n_pap)
    T, _ = tree_matrix(nodes, t_tree)
    n = len(nodes); sig = np.full(n, sigma)
    d_tree, _ = build_tree(declared, n_coll, n_det, n_pap)
    hit = 0; errs = []; ss = []; neffs = []
    for _ in range(draws):
        y = mu + sig * (T @ rng.standard_normal(T.shape[1]))
        r = tree_gls([{"margin": y[k], "sigma": sigma, "node": nodes[k]} for k in range(n)], d_tree)
        errs.append(r["m"] - mu); ss.append(r["s"]); neffs.append(r["n_eff"])
        hit += int(abs(r["m"] - mu) <= 1.96 * r["s"])
    errs = np.array(errs); ss = np.array(ss)
    return {"truth": list(truth), "declared": list(declared), "n_reports": n,
            "coverage_95": hit / draws, "bias": float(errs.mean()), "rmse": float(np.sqrt((errs ** 2).mean())),
            "s_declared_mean": float(ss.mean()), "s_true_sd_of_mhat": float(errs.std(ddof=1)),
            "s_ratio_declared_over_true": float(ss.mean() / errs.std(ddof=1)), "n_eff": float(np.mean(neffs))}


def coverage_tables(draws=500):
    two = {"design": "8 collaborations x 5 papers = 40 reports, sigma 0.1, mu 0.3, %d draws" % draws, "rows": []}
    truths = {"rho_0.5": (0.5, 0.0, 0.5), "copies": (1.0, 0.0, 0.0), "independent": (0.0, 0.0, 1.0)}
    decls = {"rho_0.5 (true structure)": (0.5, 0.0, 0.5), "copies": (1.0, 0.0, 0.0), "independent": (0.0, 0.0, 1.0),
             "rho_0.2": (0.2, 0.0, 0.8), "rho_0.8": (0.8, 0.0, 0.2)}
    for tn, tv in truths.items():
        for dn, dv in decls.items():
            r = coverage(tv, dv, 8, 1, 5, draws=draws)
            r["truth_name"], r["declared_name"] = tn, dn
            two["rows"].append(r)
    three = {"design": "6 collaborations x 2 detectors x 4 papers = 48 reports, theta = (0.3 coll, 0.2 detector, 0.5 paper)",
             "rows": []}
    t3 = (0.3, 0.2, 0.5)
    d3 = {"three-level (true structure)": t3, "two-level rho=0.5 (detector folded into the collaboration)": (0.5, 0.0, 0.5),
          "two-level rho=0.3 (detector dropped)": (0.3, 0.0, 0.7), "independent": (0.0, 0.0, 1.0),
          "copies of the collaboration": (1.0, 0.0, 0.0), "copies of the detector": (0.0, 1.0, 0.0)}
    for dn, dv in d3.items():
        r = coverage(t3, dv, 6, 2, 4, draws=draws); r["truth_name"], r["declared_name"] = "three-level", dn
        three["rows"].append(r)
    note = ("the design is balanced with equal sigma, so every declared structure gives the SAME point estimate (the "
            "plain mean: GLS weights are uniform for an exchangeable design whatever rho) — the whole cost of a "
            "mis-declaration is in s and therefore in the coverage, not in the estimate. rmse is identical across the "
            "rows of one truth for that reason; s_ratio_declared_over_true is the mis-statement of the uncertainty.")
    two["note"] = three["note"] = note
    return {"two_level": two, "three_level": three}


def main():
    res = {"what": "the three lineage states as one tree of error components; GLS in O(n*depth); N_eff; "
                   "the cost of a mis-declared lineage in coverage, with a known truth",
           "identities": identities(), "timing": timing(), "n_eff": neff_checks(), "rho_limits": rho_limits(),
           "coverage": coverage_tables()}
    OUT.write_text(json.dumps(res, indent=1))
    i, t, k = res["identities"], res["timing"], res["n_eff"]
    print("identities: max |dmhat| %.2e, max |ds| %.2e, max rel dchi2 %.2e over %d random trees (%d singular Sigma)"
          % (i["max_abs_err_mhat"], i["max_abs_err_s"], i["max_rel_err_chi2"], i["n_trees"], i["n_with_singular_sigma"]))
    print("timing n=2000: tree_gls %.4f s vs pinv %.3f s (%.0fx), same answer to %.1e"
          % (t["t_tree_gls_s"], t["t_pinv_s"], t["speedup"], max(t["abs_err_mhat"], t["abs_err_s"])))
    print("N_eff: vs lineage_information %.1e, vs Kish %.1e" % (k["max_abs_err_vs_lineage_information"], k["max_abs_err_vs_kish"]))
    print("rho limits: s(0) %.5f = %.5f indep; s(1) %.5f = %.5f copy; monotone %s"
          % (res["rho_limits"]["s_at_rho0"], res["rho_limits"]["independent_s"], res["rho_limits"]["s_at_rho1"],
             res["rho_limits"]["copy_s"], res["rho_limits"]["monotone_increasing"]))
    for name, tab in res["coverage"].items():
        print("--", name)
        for r in tab["rows"]:
            print("   truth %-12s declared %-55s cover %.3f  s/true %.2f  N_eff %5.1f  rmse %.4f"
                  % (r["truth_name"], r["declared_name"], r["coverage_95"], r["s_ratio_declared_over_true"],
                     r["n_eff"], r["rmse"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
