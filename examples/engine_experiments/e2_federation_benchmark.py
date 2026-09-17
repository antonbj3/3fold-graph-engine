#!/usr/bin/env python3
"""E2: two TEST-GRAPHS with planted truth. Shared concepts, regime flips, copied sources, noisy signs.
Compares the federation against the same code with lineage and/or validity switched off, and the
probe ranking against random and against a reach/cost ranking (the shape of anchor_graph_tools.priority)."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.claim_federation import Federation

def world(rng, n_a=25, n_b=25, n_shared=8, p_edge=0.07, p_flip=0.3):
    A = [f"a{i}" for i in range(n_a)]; B = [f"b{i}" for i in range(n_b)]; S = [f"s{i}" for i in range(n_shared)]
    truth = {}
    for dom in (A + S, B + S):
        for s in dom:
            for o in dom:
                if s != o and rng.random() < p_edge and (s, o) not in truth:
                    sg = int(rng.choice([-1, 1]))
                    truth[(s, o)] = (sg, float(rng.uniform(0.25, 0.75)) if rng.random() < p_flip else None)
    return A, B, S, truth

def true_sign(truth, pair, T):
    sg, th = truth[pair]
    return sg if th is None or T < th else -sg

def observe(rng, truth, dom, gid, err, echo_rate, n_reports=(1, 4), shared_pool=None):
    """Each root source reports one pair in one regime box; with prob echo_rate it is copied 3-6 times."""
    sources, claims = [], []
    for pair, (sg, th) in truth.items():
        if pair[0] not in dom or pair[1] not in dom:
            continue
        for _ in range(int(rng.integers(*n_reports))):
            lo, hi = (0.0, 1.0) if th is None else ((0.0, th) if rng.random() < 0.5 else (th, 1.0))
            w = (hi - lo) * rng.uniform(0.15, 0.5); a = rng.uniform(lo, hi - w)
            s_true = true_sign(truth, pair, a + w / 2)
            sign = -s_true if rng.random() < err else s_true
            root = f"{gid}-r{len(sources)}"; sources.append({"id": root, "derives_from": []}); ev = [root]
            if rng.random() < echo_rate:
                for _ in range(int(rng.integers(3, 7))):
                    c = f"{gid}-e{len(sources)}"; sources.append({"id": c, "derives_from": [root]}); ev.append(c)
            claims.append({"id": f"c{len(claims)}", "subject": pair[0], "object": pair[1], "sign": sign,
                           "validity": {"T": [a, a + w]}, "evidence": ev})
    return {"graph_id": gid, "label": "TEST-GRAPH", "concepts": [{"id": c} for c in dom], "sources": sources, "claims": claims}

def fused_accuracy(fed, truth, rng, n_pts=19, count_abstain=False):
    """Share of (pair, T) grid points with the right decided sign. count_abstain=False scores only the
    points some claim covers; True scores every point, an undecided point being a miss."""
    ok = tot = 0
    pairs = {c["pair"] for c in fed.claims.values()}
    for pair in pairs:
        for T in np.linspace(0.025, 0.975, n_pts):
            p = fed.belief(pair, {"T": float(T)})[0]
            if abs(p - 0.5) < 1e-9:
                tot += count_abstain
                continue
            ok += (p > 0.5) == (true_sign(truth, pair, T) > 0); tot += 1
    return ok / max(tot, 1), tot

def link_good(fed, truth, links):
    """An inferred link is right iff its centre lies in both legs' boxes and its sign is the product of
    the true signs there."""
    good = 0
    for l in links:
        T = np.mean(l.validity.get("T", (0, 1)))
        ca, cb = fed.claims[l.legs[0]], fed.claims[l.legs[1]]
        inside = all(c["box"]["T"][0] <= T <= c["box"]["T"][1] for c in (ca, cb))
        good += inside and l.sign == true_sign(truth, ca["pair"], T) * true_sign(truth, cb["pair"], T)
    return good

def confident_link_precision(fed, truth, top=50):
    links, _ = fed.inferred_links()
    links = sorted(links, key=lambda l: -l.p)[:top]
    return link_good(fed, truth, links) / max(len(links), 1)

def run(seed, err, echo):
    rng = np.random.default_rng(seed)
    A, B, S, truth = world(rng)
    gA = observe(rng, truth, set(A + S), "A", err, echo); gB = observe(rng, truth, set(B + S), "B", err, echo)
    out = {}
    for name, kw in {"full": {}, "no_lineage": {"use_lineage": False}, "no_validity": {"use_validity": False},
                     "neither": {"use_lineage": False, "use_validity": False}}.items():
        fed = Federation(**kw); fed.add_graph(gA); fed.add_graph(gB)
        acc, _ = fused_accuracy(fed, truth, rng)
        # conf vs boundary: truth = the pair has a planted flip between the two boxes → BOUNDARY
        sp = fed.stress_points(); tp = fp = fn = 0
        for pr in sp:
            a, b = (fed.claims[c] for c in pr.claims)
            ta, tb = (np.mean(c["box"]["T"]) for c in (a, b))
            real_boundary = true_sign(truth, pr.pair, ta) != true_sign(truth, pr.pair, tb)
            tp += pr.kind == "BOUNDARY" and real_boundary; fp += pr.kind == "BOUNDARY" and not real_boundary
            fn += pr.kind != "BOUNDARY" and real_boundary
        links, rej = fed.inferred_links(); good = link_good(fed, truth, links)
        out[name] = dict(sign_acc=acc, n_stress=len(sp), boundary_prec=tp / max(tp + fp, 1), boundary_rec=tp / max(tp + fn, 1),
                         n_links=len(links), link_prec=good / max(len(links), 1), n_rejected=len(rej))
    # experiment selection: budget of probes, oracle answers with the true sign (error 5 %), accuracy after
    sel, linkp = {}, {}
    for policy in ["default(open_first,then_evpi_x_extent)", "random", "reach_over_cost", "entropy_only", "entropy_x_extent", "evpi_x_extent", "evpi_x_extent_x_dependents"]:
        fed = Federation(); fed.add_graph(gA); fed.add_graph(gB); prng = np.random.default_rng(seed + 1)
        done = set(); curve = [fused_accuracy(fed, truth, prng, count_abstain=True)[0]]
        for step in range(60):
            cand = [p for p in fed.next_experiments(include_inferred=False)]
            if policy == "reach_over_cost":                # a measured node is PROVEN and leaves the ranking
                cand = [p for p in cand if p.pair not in done]
            if not cand: break
            if policy == "random": pr = cand[int(prng.integers(len(cand)))]
            elif policy == "reach_over_cost": pr = max(cand, key=lambda p: p.dependents / p.cost)
            elif policy == "entropy_only": pr = max(cand, key=lambda p: p.entropy)
            elif policy == "evpi_x_extent": pr = max(cand, key=lambda p: p.evpi * p.extent / p.cost)
            elif policy == "evpi_x_extent_x_dependents": pr = max(cand, key=lambda p: p.evpi * p.extent * (1 + p.dependents) / p.cost)
            elif policy == "entropy_x_extent": pr = max(cand, key=lambda p: p.entropy * p.extent)
            else: pr = cand[0]
            done.add(pr.pair); T = pr.point.get("T", 0.5); s = true_sign(truth, pr.pair, T)
            for rep in range(2):                           # two independent measurements per probe
                fed.record(pr, -s if prng.random() < 0.05 else s, f"meas-{policy}-{step}-{rep}")
            if step % 10 == 9: curve.append(fused_accuracy(fed, truth, prng, count_abstain=True)[0])
        sel[policy] = curve; linkp[policy] = confident_link_precision(fed, truth)
    return out, sel, linkp

if __name__ == "__main__":
    res = {}
    for err, echo in [(0.15, 0.0), (0.15, 0.3), (0.15, 0.6), (0.3, 0.3)]:
        with Pool(8) as pool: runs = pool.starmap(run, [(s, err, echo) for s in range(8)])
        agg = {k: {m: round(float(np.mean([r[0][k][m] for r in runs])), 4) for m in runs[0][0][k]} for k in runs[0][0]}
        n = min(len(r[1][p]) for r in runs for p in r[1])
        sel = {p: [round(float(np.mean([r[1][p][i] for r in runs])), 4) for i in range(n)] for p in runs[0][1]}
        lp = {p: round(float(np.mean([r[2][p] for r in runs])), 4) for p in runs[0][2]}
        res[f"err={err},echo={echo}"] = {"fusion": agg, "decided_and_right_share_every_10_probes": sel, "top50_inferred_link_precision_after_60_probes": lp}
        print("  link precision after probes", lp)
        print(f"err={err} echo={echo}"); [print("  ", k, v) for k, v in agg.items()]; [print("  sel", k, v) for k, v in sel.items()]
    json.dump(res, open(Path(__file__).parent / "e2_results.json", "w"), indent=1)
