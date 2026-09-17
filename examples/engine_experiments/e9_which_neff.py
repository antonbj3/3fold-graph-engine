#!/usr/bin/env python3
"""E9: three "effective number of lenses" on the MEASURED error matrix of e7 (8 lenses, 320 sentences).
  PR    = (Σλ)²/Σλ² = 1/Tr(ρ²), ρ = C/tr C      (inverse purity; for equicorrelation r it is K/(1+(K−1)r²): SIGN-BLIND)
  Kish  = K² / 1ᵀC1                               (equal weights;   K/(1+(K−1)r): sees the sign)
  BLUE  = 1ᵀC⁻¹1                                  (optimal weights; Cauchy–Schwarz: Kish ≤ BLUE, equal iff C1 ∝ 1)
Which one predicts how much pooling a SUBSET of lenses actually helps? For every subset of 2, 3, 4 lenses:
pooled accuracy (sum of log-odds) minus the mean single-lens accuracy of the subset, against each measure."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
P = np.load(Path(__file__).parent / "e7_probabilities.npy").astype(float); n, K = P.shape
truth = np.tile(np.repeat([True, False], 8), 20)
L = np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6))); err = ((L > 0) != truth[:, None]).astype(float)
C = np.corrcoef(err.T)
rows = []
for k in (2, 3, 4):
    for S in combinations(range(K), k):
        S = list(S); c = C[np.ix_(S, S)]; lam = np.linalg.eigvalsh(c)
        pr = lam.sum() ** 2 / (lam ** 2).sum(); kish = k * k / c.sum(); blue = float(np.ones(k) @ np.linalg.pinv(c) @ np.ones(k))
        gain = ((L[:, S].sum(1) > 0) == truth).mean() - (1 - err[:, S].mean())
        rows.append((k, pr, kish, blue, gain, (np.array(S) % 2).sum()))
R = np.array(rows); out = {}
for k in (2, 3, 4):
    m = R[:, 0] == k
    out[f"subsets_of_{k}"] = {"n": int(m.sum()), "spearman_gain_vs_PR": round(float(spearmanr(R[m, 1], R[m, 4])[0]), 3),
                              "spearman_gain_vs_Kish": round(float(spearmanr(R[m, 2], R[m, 4])[0]), 3),
                              "spearman_gain_vs_BLUE": round(float(spearmanr(R[m, 3], R[m, 4])[0]), 3)}
m2 = R[:, 0] == 2; mixed = R[m2, 5] == 1
out["pairs"] = {"mixed_option_order": {"n": int(mixed.sum()), "PR": round(float(R[m2][mixed, 1].mean()), 2), "Kish": round(float(R[m2][mixed, 2].mean()), 2), "gain": round(float(R[m2][mixed, 4].mean()), 3)},
                "same_option_order": {"n": int((~mixed).sum()), "PR": round(float(R[m2][~mixed, 1].mean()), 2), "Kish": round(float(R[m2][~mixed, 2].mean()), 2), "gain": round(float(R[m2][~mixed, 4].mean()), 3)}}
out["kish_le_blue_everywhere"] = bool((R[:, 2] <= R[:, 3] + 1e-9).all())
json.dump(out, open(Path(__file__).parent / "e9_results.json", "w"), indent=1); [print(k, v) for k, v in out.items()]
