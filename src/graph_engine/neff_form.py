"""neff_form — B's §9(a) contribution to the agent pool n_eff-unification: KILL THE FORM.

The equicorrelation n_eff form n_eff = N/(1 + (N-1)*x) is used repository-wide, but there are TWO distinct n_eff it can
mean, and they DIVERGE mid-correlation -- the divergence is the whole point:

  x = rho   (raw mean correlation)      -> KISH n_eff = variance-reduction of a MEAN over the N legs (noise-averaging;
                                           the effective sample size for averaging; SNR gain ~ sqrt(n_eff)). = N/lambda_max.
  x = rho^2 (squared corr = R^2)        -> PARTICIPATION-RATIO n_eff = the DECORRELATION RANK (how many independent
                                           evidence DIRECTIONS; the number over-determination coverage actually wants).
                                           = (sum lambda)^2 / sum lambda^2, EXACT for the equicorrelation eigenspectrum.

DERIVATION (equicorrelation corr matrix, N legs, common-mode corr rho): eigenvalues = {1+(N-1)rho (once), 1-rho (N-1
times)}. trace=N. Kish = N/lambda_max = N/(1+(N-1)rho). PR = N^2 / [ (1+(N-1)rho)^2 + (N-1)(1-rho)^2 ] = N/(1+(N-1)rho^2)
(algebraically exact). They agree ONLY at rho in {0 -> N, 1 -> 1}; mid-range Kish < PR (Kish uses rho, PR uses rho^2<rho).

★THE AGENT POOL CONSEQUENCE: an OVER-DETERMINATION / decorrelation claim wants the DECORRELATION RANK -> use rho^2 (R^2), NOT
raw rho -- else it UNDERCOUNTS the effective rank by up to ~2x mid-range (e.g. N=10, rho=0.3: Kish 2.70 vs PR 5.52). This
is B's D-neff-canon 'R^2 not signed-r', derived cleanly for equicorrelation. A VARIANCE-REDUCTION claim (averaging noisy
legs to estimate a mean) wants Kish (raw rho). Specify which; do not quote one form for the other goal.

  neff_form(N, rho, goal="decorrelation_rank") -> dict {n_eff, form, x_used, kish, participation_ratio,...}
    goal: "decorrelation_rank" (over-det coverage; uses rho^2) | "variance_reduction" (mean estimate; uses rho).
numpy only.
"""
import numpy as np


def _eigs(N, rho):
    return np.array([1 + (N - 1) * rho] + [1 - rho] * (N - 1), float)


def neff_form(N, rho, goal="decorrelation_rank"):
    N = int(N); rho = float(rho)
    if N < 1 or not (-1.0 / max(N - 1, 1) - 1e-9 <= rho <= 1.0 + 1e-9):
        return dict(n_eff=None, error="need N>=1 and rho in [-1/(N-1), 1]")
    kish = N / (1 + (N - 1) * rho)                          # variance-reduction (x=rho)
    pr = N / (1 + (N - 1) * rho ** 2)                       # decorrelation rank (x=rho^2 = R^2), == PR of the eigenspectrum
    if goal == "variance_reduction":
        n_eff, x_used, form = kish, "rho", "N/(1+(N-1)*rho)  [Kish: effective sample size for averaging a MEAN]"
    else:
        n_eff, x_used, form = pr, "rho^2 (R^2)", "N/(1+(N-1)*rho^2)  [participation ratio: decorrelation RANK]"
    return dict(n_eff=round(float(n_eff), 4), form=form, x_used=x_used, goal=goal,
                kish=round(float(kish), 4), participation_ratio=round(float(pr), 4),
                note=("the two n_eff DIVERGE mid-rho (Kish<PR); an over-det/decorrelation claim MUST use the PR (rho^2) "
                      "form, a mean-averaging claim uses Kish (rho). N=%d rho=%.3f -> Kish=%.2f PR=%.2f." % (N, rho, kish, pr)))


def _selftest():
    ok = tot = 0
    # (1) x=rho^2 form EXACTLY equals the participation ratio of the equicorrelation eigenspectrum
    for N, rho in [(10, 0.1), (10, 0.3), (10, 0.5), (5, 0.4), (20, 0.2)]:
        w = _eigs(N, rho); pr_true = (w.sum() ** 2) / np.sum(w ** 2)
        r = neff_form(N, rho, goal="decorrelation_rank")
        tot += 1; ok += (abs(r["n_eff"] - pr_true) < 1e-3)
    # (2) Kish and PR agree at rho=0 (=N) and rho=1 (=1), diverge mid-range (Kish<PR)
    r0 = neff_form(10, 0.0); r1 = neff_form(10, 1.0); rm = neff_form(10, 0.5)
    tot += 1; ok += (abs(r0["kish"] - r0["participation_ratio"]) < 1e-6 and abs(r0["kish"] - 10) < 1e-6)
    tot += 1; ok += (abs(r1["kish"] - 1) < 1e-6 and abs(r1["participation_ratio"] - 1) < 1e-6)
    tot += 1; ok += (rm["kish"] < rm["participation_ratio"] - 0.5)
    # (3) goal switch selects the right form
    tot += 1; ok += (neff_form(10, 0.3, "variance_reduction")["n_eff"] == neff_form(10, 0.3)["kish"])
    print("neff_form selftest: %d/%d PASS (rho^2 form == participation ratio exact; Kish<PR mid-range; agree at rho in {0,1})" % (ok, tot))
    print("  N=10: rho=0.3 -> Kish(var-red)=%.2f  PR(decorr-rank)=%.2f (diverge ~2x)" % (neff_form(10, 0.3)["kish"], neff_form(10, 0.3)["participation_ratio"]))
    return ok == tot


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
