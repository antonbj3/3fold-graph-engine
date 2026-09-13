 #!/usr/bin/env python3
"""
reliability_gate_decorr_check — is an agreement gate's reliability lift real, or shared bias?

An agreement gate ("certify only when all K members agree") lifts reliability ONLY through the
members' ERROR decorrelation. As error correlation rho -> 1 (shared pretraining prior, tokenizer,
data) the gate FALSE-CERTIFIES: the members agree on the same wrong answer. This module makes
that precondition testable BEFORE anyone quotes a certified reliability number.

CONTRACT (data in / data out, implementation hot-swappable behind it):
  reliability_gate_check(model_correct, base_margin=0.10, min_coverage=0.05, rho_abstain=0.6) -> dict
    model_correct : (N_items, K) boolean/0-1 held-out correctness of the K ensemble members.
  Returns {verdict, certified_reliability, base_accuracy, coverage, error_correlation,
           false_certify_rate, note}:
    CERTIFY_OK                 — members decorrelated enough: P(correct | all agree) exceeds base
                                 accuracy by >= base_margin, error correlation < rho_abstain, and
                                 coverage >= min_coverage.
    ABSTAIN_MEMBERS_CORRELATED — the members share a common mode; the gate would false-certify.
                                 Acquire a genuinely decorrelated member (different prior/data),
                                 not more same-prior members.
    ABSTAIN_LOW_COVERAGE       — the gate almost never fires; nothing is certified.
Pure numpy. Run `python reliability_gate_decorr_check.py` for the self-test.
"""
from __future__ import annotations
import numpy as np

__all__ = ["reliability_gate_check", "error_correlation"]


def error_correlation(model_correct: np.ndarray) -> float:
    """
    Mean pairwise correlation of the ERROR indicators (1−correct) across the K members. This is the residual-ρ that
    matters for the gate: models correlated on their MISTAKES (shared blind spots) agree-on-wrong. Returns mean off-diagonal
    Pearson ρ of the error matrix (0 = independent errors, →1 = shared common-mode).
    """
    E = 1.0 - np.asarray(model_correct, float) # error indicators, (N,K)
    K = E.shape[1]
    if K < 2:
        return 0.0
    # center per member; if a member never errs (or always errs) its variance is 0 → treat as uncorrelated (no info)
    Ec = E - E.mean(axis=0, keepdims=True)
    std = Ec.std(axis=0)
    rhos = []
    for i in range(K):
        for j in range(i + 1, K):
            if std[i] < 1e-9 or std[j] < 1e-9:
                continue
            rhos.append(float((Ec[:, i] @ Ec[:, j]) / (len(Ec) * std[i] * std[j])))
    return float(np.mean(rhos)) if rhos else 0.0


def reliability_gate_check(model_correct, base_margin: float = 0.10, min_coverage: float = 0.05,
                           rho_abstain: float = 0.6) -> dict:
    mc = np.asarray(model_correct)
    if mc.ndim != 2 or mc.shape[1] < 2:
        return {"verdict": "ABSTAIN_LOW_COVERAGE", "note": "need (N_items, K>=2) held-out correctness"}
    correct = mc.astype(bool)
    N, K = correct.shape
    all_correct = correct.all(axis=1)
    all_wrong = (~correct).all(axis=1)
    agree = all_correct | all_wrong
    n_agree = int(agree.sum())
    coverage = n_agree / N
    base = float(correct.mean())
    rel = float(all_correct.sum() / n_agree) if n_agree else float("nan")
    fc = float(all_wrong.sum() / n_agree) if n_agree else float("nan")
    rho_err = error_correlation(correct)

    if coverage < min_coverage:
        verdict = "ABSTAIN_LOW_COVERAGE"
        note = f"gate fires on only {coverage:.1%} of items — members disagree too often; almost nothing certified."
    elif (rel >= base + base_margin) and (rho_err < rho_abstain):
        verdict = "CERTIFY_OK"
        note = (f"certified reliability {rel:.3f} exceeds base {base:.3f} by {rel-base:+.3f} and error-correlation "
                f"{rho_err:.2f} < {rho_abstain} — members decorrelated enough; the agreement gate genuinely lifts reliability.")
    else:
        verdict = "ABSTAIN_MEMBERS_CORRELATED"
        note = (f"error-correlation {rho_err:.2f} (lift only {rel-base:+.3f} over base {base:.3f}) — members share a "
                f"common-mode; the agreement gate would FALSE-CERTIFY (agree-on-wrong rate {fc:.2f}). Acquire a genuinely "
                f"decorrelated member (different prior/data), NOT more same-prior members.")
    return {"verdict": verdict, "certified_reliability": round(rel, 4) if n_agree else None,
            "base_accuracy": round(base, 4), "coverage": round(coverage, 4),
            "error_correlation": round(rho_err, 4), "false_certify_rate": round(fc, 4) if n_agree else None, "note": note}


def _make_federation(N, K, p, rho, seed):
    rng = np.random.RandomState(seed)
    from math import sqrt
    # threshold so P(score>tau)=p, score=√ρ Z_shared+√(1-ρ) Z_i
    from statistics import NormalDist
    tau = -NormalDist().inv_cdf(p)
    Zs = rng.randn(N, 1); Zi = rng.randn(N, K)
    return (sqrt(rho) * Zs + sqrt(1 - rho) * Zi) > tau


def _selftest():
    ok = {}
    # G1: DECORRELATED members (ρ=0) → CERTIFY_OK (gate genuinely lifts reliability)
    r0 = reliability_gate_check(_make_federation(4000, 4, 0.70, 0.0, 1))
    ok["G1_decorrelated_members_CERTIFY_OK"] = bool(r0["verdict"] == "CERTIFY_OK" and r0["certified_reliability"] > 0.9)
    # G2: CORRELATED members (ρ=0.9, shared prior) → ABSTAIN (would false-certify)
    r1 = reliability_gate_check(_make_federation(4000, 4, 0.70, 0.9, 2))
    ok["G2_correlated_members_ABSTAIN"] = bool(r1["verdict"] == "ABSTAIN_MEMBERS_CORRELATED")
    # G3: error-correlation monotone (ρ0 < ρ1)
    ok["G3_error_correlation_detects_shared_mode"] = bool(r0["error_correlation"] < r1["error_correlation"] - 0.1)
    # G4: the check catches the FALSE-CERTIFY danger (correlated case has high agree-on-wrong)
    ok["G4_correlated_case_flags_false_certify"] = bool(r1["false_certify_rate"] > r0["false_certify_rate"] + 0.1)
    print("=== reliability_gate_decorr_check selftest ===")
    print(f"  G1 decorrelated (ρ=0): {r0['verdict']}  rel={r0['certified_reliability']} base={r0['base_accuracy']} err-ρ={r0['error_correlation']}")
    print(f"  G2 correlated  (ρ=0.9): {r1['verdict']}  rel={r1['certified_reliability']} base={r1['base_accuracy']} err-ρ={r1['error_correlation']} false-certify={r1['false_certify_rate']}")
    for k, v in ok.items():
        print(f"  {k}: {v}")
    allok = all(ok.values())
    print(f"  ALL: {allok}")
    return allok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
