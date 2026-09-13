"""overdet_neff — how many EFFECTIVE decorrelated legs does a cross-substrate over-determination actually have?
(agent-worktree deployable; validation-backbone / n_eff domain.)

The agent pool keeps stacking "over-determined across N substrates" claims -- but N is the RAW leg count, and legs that share
a MECHANISM (e.g. two 'sigma_min of an observation-Gram' legs on different substrates) are MECHANISM-IDENTITY over-det,
not decorrelated blind spots. Measured 3x in one session (B optics null-fence): raw leg count over-
states the independent evidence. This callable reports the EFFECTIVE decorrelated-leg count from the legs' per-item
SCORES on a shared test set -- so a cert says n_eff, not N.

  overdet_neff(scores, method='spearman') -> {n_legs, n_eff, overcount_factor, corr, mechanism_identity_pairs, verdict}

scores: (n_items, n_legs) -- each leg's per-item score/verdict on the SAME items (the shared space where mechanism-
identity shows up as high score-correlation). n_eff = participation ratio of the leg-correlation matrix eigenvalues
(= n_legs^2 / sum C_ij^2): n_eff==n_legs for orthogonal legs, ->1 for identical legs. A pair with |corr|>=corr_hi is
flagged MECHANISM-IDENTITY (they compute ~the same thing; their agreement is expected, not independent confirmation).

numpy-only. Selftest: run this file.
"""
import numpy as np


def _rankise(x):
    """TIE-AVERAGED column ranks (== scipy.stats.rankdata 'average'), numpy-only.
    ★ fix (agent pool argsort-tie-bug): the old np.argsort(np.argsort(x)) does NOT tie-average -- it breaks
    ties by original order, biasing the Spearman correlation (hence n_eff) on TIED/QUANTIZED/BINARY leg scores (which
    overdet_neff is routinely fed: discrete certs, pass/fail legs). Measured bias up to |Δn_eff|=0.226 on quantized
    scores (10 unique of 60) -- enough to flip a near-boundary over-det verdict. Tie-averaging removes it."""
    x = np.asarray(x, float)
    R = np.empty_like(x)
    n = x.shape[0]
    pos = np.arange(1.0, n + 1.0)                                   # 1-based ranks (scale-irrelevant for correlation)
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind="mergesort")              # stable sort
        s = x[order, j]
        newgrp = np.ones(n, bool); newgrp[1:] = s[1:] != s[:-1]    # tie-group boundaries
        grp = np.cumsum(newgrp) - 1
        gavg = np.bincount(grp, weights=pos) / np.bincount(grp)    # average 1-based rank within each tie group
        R[order, j] = gavg[grp]
    return R


def _corr_neff(X):
    """participation-ratio n_eff + rank + correlation matrix from a mean-centered (n_items,n_legs) score matrix."""
    sd = X.std(0)
    C = (X / (sd + 1e-30)).T @ (X / (sd + 1e-30)) / X.shape[0]
    C = np.clip(C, -1, 1)
    lam = np.linalg.eigvalsh(C); lam = lam[lam > 1e-12]
    n_eff = float((lam.sum() ** 2) / np.sum(lam ** 2)) if lam.size else float(X.shape[1])
    eff_rank = int(np.sum(lam > 0.05 * lam.max())) if lam.size else X.shape[1]
    return n_eff, eff_rank, C


def overdet_neff(scores, method="spearman", corr_hi=0.85, hard_mask=None, hard_quantile=0.2):
    """★hard_mask: report n_eff on the DECISIVE subset too — legs decorrelated on the easy BULK can share a
    blind spot on the HARD items (where the cert must actually cover), so full-set n_eff over-states the over-det. Pass
    hard_mask (bool, the decisive/coherent-fake items) if you have it; else the top `hard_quantile` most-extreme-
    aggregate items are used as a heuristic. The DECISIVE n_eff (hard subset) is the one that certifies the over-det."""
    scores = np.asarray(scores, float)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("scores must be (n_items, n_legs) with n_legs>=2")
    n_items, n_legs = scores.shape
    # ★NaN-IN-RANK FABRICATION GUARD (self-sweep, class on B's OWN decorrelation gate): np.argsort in
    # _rankise ASSIGNS a non-finite score a finite (max) rank -> a NaN/inf item FABRICATES a rank -> DEFLATES the leg
    # correlation -> INFLATES n_eff = a corrupt/missing measurement silently read as EVIDENCE OF DECORRELATION (the
    # dangerous direction; measured: 1 NaN shifted n_eff 1.010->1.070). Drop items with any non-finite leg score BEFORE
    # ranking (rank finite-only), and ABSTAIN if too few finite items remain (never fabricate, never compute on a stub).
    finite_rows = np.isfinite(scores).all(axis=1)
    if not finite_rows.all():
        if int(finite_rows.sum()) < 3:
            return dict(n_legs=n_legs, n_eff=float("nan"), overcount_factor=float("nan"), corr=None,
                        mechanism_identity_pairs=[], verdict="ABSTAIN: <3 finite items after dropping non-finite "
                        "(NaN-in-rank fabrication guarded; a non-finite score cannot certify decorrelation)")
        scores = scores[finite_rows]
        n_items = scores.shape[0]
    X = _rankise(scores) if method == "spearman" else scores.copy()
    X = X - X.mean(0)
    sd = X.std(0)
    if np.any(sd < 1e-12):
        keep = sd >= 1e-12
        if keep.sum() < 2:
            return dict(n_legs=n_legs, n_eff=float(keep.sum()), overcount_factor=float("nan"),
                        corr=None, mechanism_identity_pairs=[], verdict="degenerate: <2 discriminating legs")
        return _with_dropped(scores, keep, method, corr_hi, hard_mask, hard_quantile)
    n_eff, eff_rank, C = _corr_neff(X)
    mi = [(int(i), int(j), round(float(C[i, j]), 3))
          for i in range(n_legs) for j in range(i + 1, n_legs) if abs(C[i, j]) >= corr_hi]
    overcount = n_legs / n_eff

    # ---- DECISIVE (hard) subset n_eff ----
    if hard_mask is not None:
        hard = np.asarray(hard_mask, bool)
    else:  # heuristic: the most-extreme-aggregate items (highest-stakes; where a shared blind spot = a coherent fake)
        agg = X.mean(1); dev = np.abs(agg - np.median(agg))
        k = max(20, int(hard_quantile * n_items))
        hard = np.zeros(n_items, bool); hard[np.argsort(dev)[-k:]] = True
    hard_neff = None; shared_blindspot = False
    if hard.sum() >= 20:
        Xh = X[hard] - X[hard].mean(0)
        if np.all(Xh.std(0) > 1e-12):
            hard_neff, _, _ = _corr_neff(Xh)
            shared_blindspot = hard_neff < 0.6 * n_eff        # over-det EVAPORATES on the decisive items

    verdict = ("legs ~independent (n_eff≈n_legs): over-determination genuine" if overcount < 1.25 else
               "MECHANISM-IDENTITY over-count: raw n_legs=%d but n_eff=%.1f (%.1fx) -- report n_eff, treat high-corr "
               "pairs as ONE leg" % (n_legs, n_eff, overcount))
    if shared_blindspot:
        verdict = ("★SHARED-BLIND-SPOT (heed L88): full n_eff=%.1f but DECISIVE-subset n_eff=%.1f -- the legs are "
                   "decorrelated on the easy bulk yet ~identical on the HARD items where the cert must cover. The over-"
                   "determination EVAPORATES where it matters; report the DECISIVE n_eff (%.1f), not the full one." %
                   (n_eff, hard_neff, hard_neff))
    return dict(n_legs=n_legs, n_eff=round(n_eff, 2), effective_rank=eff_rank,
                decisive_n_eff=(round(hard_neff, 2) if hard_neff is not None else None),
                shared_blindspot=bool(shared_blindspot), overcount_factor=round(overcount, 2),
                corr=np.round(C, 3), mechanism_identity_pairs=mi, verdict=verdict)


def _with_dropped(scores, keep, method, corr_hi, hard_mask=None, hard_quantile=0.2):
    sub = overdet_neff(scores[:, keep], method=method, corr_hi=corr_hi, hard_mask=hard_mask, hard_quantile=hard_quantile)
    sub["n_legs"] = scores.shape[1]
    sub["verdict"] = "dropped %d constant leg(s); " % int((~keep).sum()) + sub["verdict"]
    return sub


def _selftest():
    rng = np.random.default_rng(0)
    n = 300
    checks = []
    # (1) 3 independent legs -> n_eff ~ 3
    S = rng.normal(size=(n, 3))
    r1 = overdet_neff(S)
    checks.append(("3 independent legs -> n_eff~3", 2.6 <= r1["n_eff"] <= 3.0, r1["n_eff"]))
    # (2) 3 legs, 2 are near-duplicates (mechanism-identity) -> n_eff ~ 2
    base = rng.normal(size=(n, 2))
    S2 = np.column_stack([base[:, 0], base[:, 0] + 0.05 * rng.normal(size=n), base[:, 1]])  # legs 0,1 identical
    r2 = overdet_neff(S2)
    # participation-ratio n_eff of {2 perfect-dup + 1 indep} = 9/5 = 1.8 (eigs {2,1,0}); it weights by dominance
    # (rank=2 counts distinct directions). Both reported; the 1.8 is the correct effective-EVIDENCE count.
    checks.append(("2 duplicate + 1 indep -> n_eff~1.8, rank=2", 1.65 <= r2["n_eff"] <= 2.0 and r2["effective_rank"] == 2, (r2["n_eff"], r2["effective_rank"])))
    checks.append(("duplicate pair flagged mechanism-identity", (0, 1) in [(a, b) for a, b, _ in r2["mechanism_identity_pairs"]], r2["mechanism_identity_pairs"]))
    # (3) 3 identical legs -> n_eff ~ 1
    b = rng.normal(size=(n, 1))
    S3 = np.column_stack([b[:, 0], b[:, 0] + 1e-3 * rng.normal(size=n), b[:, 0] + 1e-3 * rng.normal(size=n)])
    r3 = overdet_neff(S3)
    checks.append(("3 identical legs -> n_eff~1", r3["n_eff"] <= 1.3, r3["n_eff"]))
    # (4) the optics/geometry/LLM motivating case: geo1,geo2 correlated (same mechanism) + LLM decorrelated -> n_eff~2
    g = rng.normal(size=(n,))
    geo1 = g + 0.2 * rng.normal(size=n); geo2 = g + 0.2 * rng.normal(size=n)      # same mechanism
    llm = rng.normal(size=n)                                                       # different mechanism
    r4 = overdet_neff(np.column_stack([geo1, geo2, llm]))
    checks.append(("geo1~geo2 + LLM -> n_eff~2 (not 3)", 1.7 <= r4["n_eff"] <= 2.4, r4["n_eff"]))
    # ★(5) scope gap: 3 legs DECORRELATED on the easy bulk but ~IDENTICAL on a HARD subset -> full n_eff high,
    # DECISIVE n_eff ~1, shared_blindspot flagged. hard = last 25% of items (where the legs collapse to one).
    nb = 400; hard_n = 120
    bulk = rng.normal(size=(nb, 3))                                  # decorrelated on the bulk
    shared = rng.normal(size=(hard_n, 1))
    hardblock = np.repeat(shared, 3, axis=1) + 0.02 * rng.normal(size=(hard_n, 3))  # ~identical on hard items
    Sh = np.vstack([bulk, hardblock]); hm = np.zeros(nb + hard_n, bool); hm[nb:] = True
    r5 = overdet_neff(Sh, hard_mask=hm)
    checks.append(("shared-blindspot: full n_eff high, decisive n_eff~1, flagged",
                   r5["n_eff"] > 2.4 and r5["decisive_n_eff"] is not None and r5["decisive_n_eff"] < 1.4 and r5["shared_blindspot"],
                   (r5["n_eff"], r5["decisive_n_eff"], r5["shared_blindspot"])))
    # ★(6) TIE-AVERAGING REGRESSION (agent pool argsort-tie-bug): _rankise MUST tie-average, not ordinal-rank.
    # Hand-verifiable: x=[10,20,20,30] -> tie-avg 1-based ranks [1,2.5,2.5,4]; the old argsort.argsort gave [1,2,3,4].
    # This check FAILS if _rankise ever regresses to np.argsort(np.argsort(x)). No scipy dependency (self-contained).
    tr = _rankise(np.array([[10.0], [20.0], [20.0], [30.0]]))[:, 0]
    tie_ok = np.allclose(tr, [1.0, 2.5, 2.5, 4.0])
    allsame = np.allclose(_rankise(np.ones((6, 1)))[:, 0], 3.5)      # all-tied col -> all ranks = avg(1..6)=3.5
    checks.append(("_rankise tie-AVERAGED (not ordinal argsort) -- regression guard for the fleet tie-bug",
                   bool(tie_ok and allsame), (list(tr), "allsame=%s" % allsame)))
    print("overdet_neff selftest:")
    for name, ok, val in checks:
        print("  [%s] %s  (%s)" % ("PASS" if ok else "FAIL", name, val))
    npass = sum(o for _, o, _ in checks)
    print("  %d/%d PASS" % (npass, len(checks)))
    return npass == len(checks)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
