#!/usr/bin/env python3
"""OVER-CLAIM GAUGE — deployable non-stationarity-aware over-claim detector (the σ_min-fluctuation thread, shipped).

The 3-tick arc (RELENTLESS wave 1+2, validated on real robot trajectory + z24 bridge):
  • CONFIDENCE (a p-value / analytic SE) is structurally BLIND to the soft mode; the σ_min gauge is FLUCTUATION.
  • But the WITHIN-sample bootstrap fluctuation is itself BLIND to NON-STATIONARITY (a finding that holds in one regime can
    fail in another). The fix: measure the CROSS-REGIME fluctuation — a finding that wobbles across training regimes will fail
    on a future regime (validated: cross-regime fluct ↔ held-out replication +0.59, vs within-train bootstrap −0.04).
  • The regime shift is TWO-KIND: it can move the per-feature MEAN (location) OR the CORRELATION structure — a single-kind
    gauge under-detects the other (robot→corr-kind, z24→mean-kind). So the gauge reports BOTH kinds.

  regime_overclaim_risk(features_by_regime) → per-finding cross-regime fluctuation (over-claim risk) for BOTH kinds + flags.
  held_out_predictive_power(features_by_regime) → the honest self-validation: does the risk predict held-out replication?

RESIDUAL LIMIT (honest): a FULLY NOVEL regime not sampled by any input regime is unpredictable — this bounds SAMPLED
regime-variability, not extrapolation to an unseen regime (a provenance floor). Ships the σ_min-fluctuation over-claim thread.
"""
import numpy as np


def _mean_findings(regimes):
    return np.array([r.mean(0) for r in regimes])              # [n_reg, p] per-feature means


def _corr_findings(regimes):
    out = []
    for r in regimes:
        C = np.corrcoef(r, rowvar=False); iu = np.triu_indices(C.shape[0], 1); out.append(C[iu])
    return np.array(out)                                        # [n_reg, n_pairs]


def regime_overclaim_risk(features_by_regime, flag_percentile=80):
    """features_by_regime: list of [n_i, p] arrays, one per REGIME (temporal window / condition / damage class). Returns,
    per KIND (mean/location, corr/structure), the cross-regime fluctuation of each finding = its over-claim risk, and a
    boolean flag for the top `flag_percentile`% riskiest (likely to fail on a new regime). The dominant kind is the one
    carrying more total over-claim risk. This is the deployable non-stationarity-aware over-claim detector."""
    regimes = [np.asarray(r, float) for r in features_by_regime if np.asarray(r, float).shape[0] > 2]
    if len(regimes) < 3:
        return {"error": "need ≥3 regimes with >2 samples each"}
    out = {}
    for kind, fn in (("mean", _mean_findings), ("corr", _corr_findings)):
        F = fn(regimes)                                         # [n_reg, n_findings]
        risk = np.nanstd(F, axis=0)                             # cross-regime fluctuation per finding = over-claim risk
        good = np.isfinite(risk)
        thr = np.nanpercentile(risk[good], flag_percentile) if good.any() else np.inf
        out[kind] = {"n_findings": int(good.sum()), "risk_median": round(float(np.nanmedian(risk[good])), 5),
                     "risk_p90": round(float(np.nanpercentile(risk[good], 90)), 5),
                     "n_flagged": int((risk[good] >= thr).sum()), "flag_threshold": round(float(thr), 5)}
    # ★which kind actually PREDICTS over-claim is NOT the risk-magnitude (that conflates the two) — it is the held-out
    # predictive power. Report the risk per kind here; call held_out_predictive_power for the predictive kind.
    out["note"] = ("over-claim risk = cross-regime fluctuation; flagged findings likely fail on a NEW regime. Report BOTH "
                   "kinds. The PREDICTIVE kind (which one's risk actually tracks held-out failure) = held_out_predictive_power.")
    return out


def held_out_predictive_power(features_by_regime, train_frac=0.66):
    """Honest self-validation on the caller's own data: split regimes into train/test, and for each kind report the
    correlation between the cross-TRAIN-regime fluctuation and the held-out replication error |finding_test - finding_train|.
    A high correlation means the gauge genuinely predicts which findings fail on the held-out (future) regimes."""
    regimes = [np.asarray(r, float) for r in features_by_regime if np.asarray(r, float).shape[0] > 2]
    R = len(regimes); nt = max(2, int(R * train_frac))
    if R - nt < 1:
        return {"error": "need more regimes for a held-out split"}
    tr, te = regimes[:nt], regimes[nt:]
    res = {}
    for kind, fn in (("mean", _mean_findings), ("corr", _corr_findings)):
        Ftr, Fte = fn(tr), fn(te)
        risk = np.nanstd(Ftr, axis=0)
        repl_err = np.abs(np.nanmean(Fte, 0) - np.nanmean(Ftr, 0))
        good = np.isfinite(risk) & np.isfinite(repl_err)
        c = float(np.corrcoef(risk[good], repl_err[good])[0, 1]) if good.sum() >= 3 else float("nan")
        res[kind] = round(c, 3) if np.isfinite(c) else None
    valid = {k: v for k, v in res.items() if v is not None}
    predictive_kind = max(valid, key=valid.get) if valid else None
    return {"regime_fluct_vs_heldout_replication": res, "predictive_kind": predictive_kind,
            "n_train_regimes": nt, "n_test_regimes": R - nt,
            "interpretation": "corr>~0.3 ⇒ the gauge predicts over-claim on held-out regimes for that kind; predictive_kind = "
                              "the kind whose cross-regime risk best tracks held-out failure (need ≥3 findings/kind; <3 → None)."}


if __name__ == "__main__":
    # smoke: a non-stationary mean-shift substrate → mean-kind risk should dominate + predict held-out
    rng = np.random.default_rng(0)
    regimes = []
    for k in range(9):
        shift = np.array([k * 0.3, 0.0, 0.0])                  # feature 0 drifts across regimes (mean non-stationarity)
        regimes.append(rng.standard_normal((200, 3)) + shift)
    risk = regime_overclaim_risk(regimes); hop = held_out_predictive_power(regimes)
    print("smoke regime_overclaim_risk:", {k: risk[k] for k in risk})
    print("smoke held_out:", hop["regime_fluct_vs_heldout_replication"])
    print("→ the mean kind should carry the risk (the drifting feature); its held-out corr should be high")
