#!/usr/bin/env python3
"""E3c: recursion as growth of the feature span, not as choice of replay samples.
Round r: fit the logistic policy on span S_r; take its residual y − p on the TRAINING episodes; from a
pool of candidate directions (pairwise products and squares of the base features) add the one whose
component orthogonal to S_r correlates most with the residual; refit. Test episodes are never used
for selection. Compared with: the base policy, a random-direction control, gradient boosting."""
import json, sys
from itertools import combinations_with_replacement
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.replay_policy import FEATURES, build_episode, average_precision
from _data import snap_citations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

H, ROUNDS = 24, 8
PLANS = [dict(test=95, train=[35, 47, 59, 71]), dict(test=107, train=[47, 59, 71, 83])]
prep = lambda X: np.sign(X) * np.log1p(np.abs(X))
POOL = list(combinations_with_replacement(range(len(FEATURES)), 2))
name = lambda ab: FEATURES[ab[0]] + "*" + FEATURES[ab[1]]

def grow(Xtr, ytr, Xte, yte, mode, rng):
    sc = StandardScaler().fit(prep(Xtr)); A, B = sc.transform(prep(Xtr)), sc.transform(prep(Xte))
    cand_tr = np.stack([A[:, a] * A[:, b] for a, b in POOL], 1); cand_te = np.stack([B[:, a] * B[:, b] for a, b in POOL], 1)
    mu, sd = cand_tr.mean(0), cand_tr.std(0) + 1e-12
    cand_tr, cand_te = (cand_tr - mu) / sd, (cand_te - mu) / sd
    S_tr, S_te, used, curve, picked = A, B, set(), [], []
    for r in range(ROUNDS + 1):
        m = LogisticRegression(C=1.0, max_iter=400).fit(S_tr, ytr)
        curve.append(average_precision(yte, m.decision_function(S_te)))
        if r == ROUNDS: break
        res = ytr - m.predict_proba(S_tr)[:, 1]
        free = [j for j in range(len(POOL)) if j not in used]
        if mode == "random": j = int(rng.choice(free))
        else:
            Q, _ = np.linalg.qr(np.c_[np.ones(len(S_tr)), S_tr])          # orthogonal complement of the current span
            C = cand_tr[:, free]; C = C - Q @ (Q.T @ C)
            score = np.abs(C.T @ res) / (np.linalg.norm(C, axis=0) + 1e-12)
            j = free[int(np.argmax(score))]
        used.add(j); picked.append(name(POOL[j]))
        S_tr, S_te = np.c_[S_tr, cand_tr[:, j]], np.c_[S_te, cand_te[:, j]]
    return curve, picked

out = {}
for g in ["HepTh", "HepPh"]:
    e, mth, _ = snap_citations(g)
    for plan in PLANS:
        tr = [build_episode(e, mth, T, H) for T in plan["train"]]; te = build_episode(e, mth, plan["test"], H)
        Xtr, ytr = np.concatenate([x.X for x in tr]), np.concatenate([x.y for x in tr])
        sub = np.random.default_rng(0).choice(len(ytr), 400_000, replace=False); Xtr, ytr = Xtr[sub], ytr[sub]
        cur, picked = grow(Xtr, ytr, te.X, te.y, "residual", None)
        rnd = np.mean([grow(Xtr, ytr, te.X, te.y, "random", np.random.default_rng(s))[0] for s in range(3)], 0)
        out[f"{g}@{plan['test']}"] = dict(residual_growth_ap_by_round=[round(x, 4) for x in cur], picked=picked,
                                          random_growth_ap_by_round=[round(float(x), 4) for x in rnd])
        print(g, plan["test"], out[f"{g}@{plan['test']}"], flush=True)
        json.dump(out, open(Path(__file__).parent / "e3c_results.json", "w"), indent=1)
