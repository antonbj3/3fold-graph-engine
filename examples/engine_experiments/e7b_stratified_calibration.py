#!/usr/bin/env python3
"""E7b: the same measured answers (e7_probabilities.npy), recalibrated per stratum. A lens that is reliably
WRONG on a stratum carries as much information as one that is reliably right: what counts is |accuracy − 0.5|.
Platt scaling a·logit(p)+b per (template, lens), fitted on 10 quantity pairs, applied to the 10 others
(split by pair, 20 random splits). Pooling tempered by N_eff measured on the training half."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
P = np.load(Path(__file__).parent / "e7_probabilities.npy").astype(float); n, K = P.shape
pair = np.repeat(np.arange(20), 16); sign = np.tile(np.repeat([1, -1], 8), 20) > 0; tmpl = np.tile(np.arange(8), 40)
L = np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6)))
rows = []
for seed in range(20):
    rng = np.random.default_rng(seed); tr_pairs = rng.choice(20, 10, replace=False); tr = np.isin(pair, tr_pairs); te = ~tr
    Lc = np.zeros_like(L)
    for t in range(8):
        for k in range(K):
            m = tr & (tmpl == t); clf = LogisticRegression(C=1.0).fit(L[m, k:k + 1], sign[m])
            mt = tmpl == t; Lc[mt, k] = clf.decision_function(L[mt, k:k + 1])
    err = (Lc[tr] > 0) != sign[tr][:, None]; C = np.corrcoef(err.T.astype(float)); rho = max(float(np.nanmean(C[np.triu_indices(K, 1)])), 0)
    w = 1 / (1 + (K - 1) * rho)
    acc = lambda z: float(((z > 0) == sign[te]).mean())
    rows.append(dict(raw_single=float(np.mean([acc(L[te, k]) for k in range(K)])), raw_pooled=acc(L[te].sum(1)),
                     calibrated_single=float(np.mean([acc(Lc[te, k]) for k in range(K)])), calibrated_pooled=acc(Lc[te].sum(1)),
                     brier_cal_naive=float(np.mean((1 / (1 + np.exp(-Lc[te].sum(1))) - sign[te]) ** 2)),
                     brier_cal_tempered=float(np.mean((1 / (1 + np.exp(-w * Lc[te].sum(1))) - sign[te]) ** 2)), rho_after=rho))
res = {k: [round(float(np.mean([r[k] for r in rows])), 3), round(float(np.std([r[k] for r in rows])), 3)] for k in rows[0]}
json.dump(res, open(Path(__file__).parent / "e7b_results.json", "w"), indent=1); [print(k, v) for k, v in res.items()]
