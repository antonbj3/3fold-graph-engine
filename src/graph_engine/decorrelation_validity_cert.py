"""decorrelation_validity_cert — the executable form of the AUTO_ADMIT requirement ('validated=True must mean
validated ON the confusion mode = the adversarial minimum, NOT easy data'): error-correlation is confusion-mode-
conditional, worst on the adversarial minimum, and effective n_eff comes from OBSERVATION-CHANNEL diversity, not leg count.

The AUTO_ADMIT decorrelation gate credits a union as effectively-guarded iff family_error_corr < FAMILY_DECORR_THRESH.
 showed that number is CONFUSION-MODE-CONDITIONAL: a union can be decorrelated on EASY fakes (big-gap / approach) yet
PERFECTLY correlated on the coherent-fake ADVERSARIAL MINIMUM (content-matched / smooth corner) -> a naive easy-calibrated
family_error_corr FALSE-PASSES AUTO_ADMIT.

This cert takes each member's per-instance CORRECTNESS on the ADVERSARIAL-MINIMUM fake set (the caller must supply fakes
at the confusion mode, not easy), measures the max pairwise error-correlation THERE, and returns validated iff it clears
the threshold ON THAT mode. It also reports the effective n_eff  so a union of K same-channel legs is not credited
as n_eff=K. numpy only. Selftest: `python -m graph_engine.decorrelation_validity_cert`."""
from __future__ import annotations
import numpy as np

FAMILY_DECORR_THRESH = 0.50   # max pairwise error-correlation (ON the adversarial minimum) to count as decorrelated
# ★ VOID-FLOOR (systemic sweep of the keystone defect): the NO-STRATA base correctness-corr is measured over
# ALL instances, so easy-instance agreement (both members trivially correct on easy) INFLATES it and understates the
# decorrelation that matters -- measured: two members decorrelated on the hard sub-mode (corr -0.03) read corr +0.23
# over all instances (70%% easy). Here it stays LATENT (loose thresh 0.50 masks +0.23), and the STRATA guard IS the
# fix: pass strata=<adversarial-minimum labels> to evaluate per-stratum on the hard mode. Under a STRICTER thresh, or
# with more easy mass, the no-strata base corr could false-over-reject (fail-CLOSED) -- so strata is not optional when
# the adversarial minimum is a small fraction. (The keystone's fake-detector version DID misgate; fixed .)


def _pair_error_corr(ci, cj):
    """phi-correlation of two members' per-instance CORRECTNESS (1=correct,0=wrong). NaN-safe -> 1.0 (assume correlated)."""
    ci = np.asarray(ci, float); cj = np.asarray(cj, float)
    if ci.std() < 1e-9 or cj.std() < 1e-9:
        return 1.0   # a constant/degenerate member -> cannot demonstrate decorrelation -> pessimistic
    r = float(np.corrcoef(ci, cj)[0, 1])
    return r if np.isfinite(r) else 1.0


COMPETENT, BLIND, ABSTAIN = "COMPETENT", "BLIND", "ABSTAIN"


def _wilson_ci(k, n, z=1.96):
    """Wilson score CI (lo, hi) for a binomial proportion k/n (95%)."""
    if n <= 0:
        return (0.0, 1.0)
    ph = k / n; den = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    m = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - m), min(1.0, c + m))


def _member_competence_verdict(c, strata, floor):
    """THREE-WAY competence verdict per member (L217d, ADOPTED from the b_wilson_competence_gate, own-run 4/4): compare
    the Wilson CI of the per-stratum competence to the floor -- CI-lo >= floor -> COMPETENT; CI-hi < floor -> BLIND;
    CI straddles floor -> ABSTAIN (thin n on this stratum, do not guess). Aggregate min-over-strata (L217b): BLIND if ANY
    stratum confidently blind; else ABSTAIN if ANY undecidable; else COMPETENT. This fixes the FP<->FN asymmetry flagged
    on my  one-sided Wilson-lower (which over-rejected genuine thin-n members as blind) -- it ABSTAINs instead."""
    def verdict(cc):
        n = int(len(cc))
        if n == 0:
            return ABSTAIN
        lo, hi = _wilson_ci(int(round(float(cc.sum()))), n)
        return COMPETENT if lo >= floor else (BLIND if hi < floor else ABSTAIN)
    if strata is None:
        return verdict(c)
    labs = np.unique(strata)
    vs = [verdict(c[strata == L]) for L in labs if int((strata == L).sum()) > 0]
    if any(v == BLIND for v in vs):
        return BLIND
    if any(v == ABSTAIN for v in vs):
        return ABSTAIN
    return COMPETENT


def decorrelation_validity_cert(member_correctness, mode="adversarial-minimum", thresh=FAMILY_DECORR_THRESH,
                                min_competence=None, strata=None):
    """member_correctness: list of (name, correctness_array) -- each member's per-instance CORRECTNESS on the SAME
    ADVERSARIAL-MINIMUM fake+clean set (1=correct, 0=wrong). Returns whether the union is validly decorrelated ON that mode.

    ★min_competence (the decorrelation-by-BLINDNESS mirage guard): low error-correlation is NECESSARY-NOT-SUFFICIENT
    -- a BLIND leg (insensitive to the QoI, correctness ~chance) has decorrelated errors but adds NO n_eff (mirage;
    validates good+blind AND even blind+blind at rho~0). Set min_competence to a sensitivity FLOOR strictly ABOVE the
    task's chance level (chance + margin, e.g. for a 5-class task chance=0.20 -> use ~0.35; a member exactly AT chance must
    fail); any member whose mean correctness < min_competence FAILS the sensitivity precondition -> NOT validated. Default
    None preserves the pure rho-gate (necessary condition only); for the SUFFICIENT gate that measures the actual
    certification outcome, use certifying_power_cert, which is robust to blindness by construction (no floor to pick).

    Returns dict(validated, max_pair_error_corr, effective_neff, n_members, mode, worst_pair, blind_members, reason)."""
    ms = [(n, np.asarray(c, float)) for n, c in member_correctness]
    K = len(ms)
    if K < 2:
        return dict(validated=False, max_pair_error_corr=None, effective_neff=float(K), n_members=K, mode=mode,
                    worst_pair=None, blind_members=[], abstain_members=[], reason="need >=2 members to certify decorrelation")
    def _fc(reason):   # fail-closed return
        return dict(validated=False, max_pair_error_corr=None, effective_neff=float(K), n_members=K, mode=mode,
                    worst_pair=None, blind_members=[], abstain_members=[], reason=reason)
    # ★ (inf-sibling) + (red-team HOLE4): non-finite thresholds fail OPEN under ±inf; an ARRAY threshold
    # broke scalar np.isfinite. Require finite SCALARS (np.isfinite closes {nan,+inf,-inf}; min_competence=None = off).
    def _bad(v, allow_none):
        if v is None:
            return not allow_none
        try:
            arr = np.asarray(v, float)
        except (ValueError, TypeError):
            return True
        return arr.ndim != 0 or not bool(np.isfinite(arr))
    if _bad(thresh, False) or _bad(min_competence, True):
        return _fc("bad threshold (thresh=%r, min_competence=%r) -- need finite scalar; NOT validated (fail-closed)"
                   % (thresh, min_competence))
    # ★ (red-team HOLE7): NaN/inf in the CORRECTNESS DATA crashed `int(round(cc.sum))` in the competence verdict
    # (data-level non-finite, not covered by the param guards -- this is a standalone deployable). Fail closed.
    if any(not bool(np.all(np.isfinite(c))) for _, c in ms):
        return _fc("non-finite value(s) in member_correctness data -- cannot certify; NOT validated (fail-closed)")
    # ★ (red-team HOLE1): a non-finite strata LABEL is silently dropped from the per-stratum loop (strata==nan False
    # everywhere) -> a member blind on that stratum is invisible -> false pass. Reject non-finite strata.
    if strata is not None:
        try:
            _s = np.asarray(strata, float)
        except (ValueError, TypeError):
            return _fc("strata must be a numeric array of finite labels; NOT validated (fail-closed)")
        if not bool(np.all(np.isfinite(_s))):
            return _fc("non-finite value(s) in strata -- the stratum would be silently dropped; NOT validated (fail-closed)")
    worst = -np.inf; worst_pair = None   # audit bug 4: -1.0 sentinel collided with a real corr of exactly -1.0 -> pair None
    for i in range(K):
        for j in range(i + 1, K):
            r = _pair_error_corr(ms[i][1], ms[j][1])
            if r > worst:
                worst = r; worst_pair = (ms[i][0], ms[j][0])
    # effective K-way n_eff (audit bug 3 fix: the old `2 / (1+corr)` HARD-CAPPED at 2.0 for any K -- a literal 2, not K --
    # so a 5- or 10-channel union looked identical to a 2-channel one). Conservative Kish-style using the WORST positive
    # pairwise correlation (consistent with the worst-pair validated decision): independent members -> neff~K, all-identical
    # -> neff~1, an anti-correlated pair -> credited as independent. Reduces to the old formula at K=2.
    neff = round(K / (1 + (K - 1) * max(worst, 0.0)), 2)
    # ★ blindness-mirage guard (L217b strata + L217d THREE-WAY fix): each member gets a COMPETENT/BLIND/ABSTAIN
    # verdict from the Wilson CI of its per-stratum competence vs the floor (min over sub-modes). BLIND = confidently at/
    # below chance (mirage -> VETO); ABSTAIN = thin n on a stratum, undecidable (honest, NOT a false blind-flag -- fixes
    # the FP<->FN asymmetry flagged on my one-sided Wilson-lower); COMPETENT = confidently sensitive.
    blind, abstain = [], []
    if min_competence is not None:
        for nm, c in ms:
            v = _member_competence_verdict(c, strata, min_competence)
            if v == BLIND:
                blind.append(nm)
            elif v == ABSTAIN:
                abstain.append(nm)
    validated = bool(worst < thresh and not blind and not abstain)
    if blind:
        return dict(validated=False, max_pair_error_corr=round(worst, 3), effective_neff=neff, n_members=K, mode=mode,
                    worst_pair=worst_pair, blind_members=blind, abstain_members=abstain,
                    reason=("BLINDNESS MIRAGE (L217): member(s) %s CONFIDENTLY at/below sensitivity floor %.2f (Wilson CI "
                            "hi < floor) -> their decorrelation is a mirage; NOT validated even though worst pair-corr "
                            "%.2f < %.2f. Use certifying_power_cert for the sufficient gate." % (blind, min_competence, worst, thresh)))
    if abstain:
        return dict(validated=False, max_pair_error_corr=round(worst, 3), effective_neff=neff, n_members=K, mode=mode,
                    worst_pair=worst_pair, blind_members=[], abstain_members=abstain,
                    reason=("COMPETENCE-ABSTAIN (L217d, three-way): member(s) %s have too few instances on a stratum to "
                            "decide competence (Wilson CI straddles floor %.2f) -> ABSTAIN, not a guess. Gather more data on "
                            "the thin (adversarial-minimum) stratum. NOT a blindness veto, NOT a silent admit." % (abstain, min_competence)))
    return dict(validated=validated, max_pair_error_corr=round(worst, 3), effective_neff=neff, n_members=K, mode=mode,
                worst_pair=worst_pair, blind_members=[], abstain_members=[],
                reason=("decorrelated on the %s (worst pair-corr %.2f < %.2f) -> validated" % (mode, worst, thresh)
                        if validated else
                        "CORRELATED on the %s (worst pair %s corr %.2f >= %.2f) -> NOT validated; a union decorrelated on "
                        "EASY fakes can be n_eff~1 here (L198). Add a leg on a DISTINCT observation channel (L199) or "
                        "calibrate on this mode." % (mode, worst_pair, worst, thresh)))


def _selftest():
    ok = tot = 0
    rng = np.random.default_rng(0); n = 400
    # (1) SAME-channel union: on the adversarial minimum both members share the blind spot -> correlated -> NOT validated.
    # Model: both correct on the same ~40% (easy part), both WRONG together on the ~60% adversarial corner.
    base = rng.random(n)
    A = (base > 0.6).astype(int); B = (base > 0.62).astype(int)          # highly correlated correctness
    r = decorrelation_validity_cert([("velocity", A), ("accel_residual", B)])
    tot += 1; ok += (not r["validated"] and r["max_pair_error_corr"] > 0.5 and r["effective_neff"] < 1.4)
    # (2) CROSS-channel union: each catches a DIFFERENT sub-mode -> misses on different instances -> decorrelated -> validated.
    half = n // 2
    V = np.r_[np.ones(half), rng.integers(0, 2, n - half)]               # correct on first half
    N = np.r_[rng.integers(0, 2, half), np.ones(n - half)]               # correct on second half (different instances)
    rng.shuffle(V); rng.shuffle(N)                                        # (shuffle preserves marginal, breaks alignment)
    V = np.r_[np.ones(half), np.zeros(n - half)]; N = np.r_[np.zeros(half), np.ones(n - half)]  # anti-correlated correctness
    r2 = decorrelation_validity_cert([("velocity", V), ("acquisition_noise", N)])
    tot += 1; ok += (r2["validated"] and r2["max_pair_error_corr"] < 0.5 and r2["effective_neff"] > 1.7)
    # (3) degenerate member (constant correctness) -> pessimistic (corr 1.0) -> NOT validated (safe)
    C = np.ones(n)
    r3 = decorrelation_validity_cert([("velocity", V), ("blind_const", C)])
    tot += 1; ok += (not r3["validated"])
    # (4) <2 members -> not validated
    r4 = decorrelation_validity_cert([("only", V)])
    tot += 1; ok += (not r4["validated"])
    # (5) K=5 mutually-independent channels -> effective_neff scales PAST 2 (audit bug 3: old formula hard-capped at 2.0)
    r5 = decorrelation_validity_cert([("c%d" % k, rng.integers(0, 2, n)) for k in range(5)])
    tot += 1; ok += (r5["validated"] and r5["effective_neff"] > 2.5)
    # (6) a pair with correctness EXACTLY anti-correlated (corr -1.0) still names its worst_pair (audit bug 4)
    A6 = np.r_[np.ones(50), np.zeros(50)]; r6 = decorrelation_validity_cert([("m1", A6), ("m2", 1 - A6)])
    tot += 1; ok += (r6["worst_pair"] is not None and r6["max_pair_error_corr"] == -1.0)
    # (7) ★ BLINDNESS-MIRAGE guard: a competent + BLIND (chance-level correctness) pair has LOW rho (decorrelated
    # errors) so the pure rho-gate validates (mirage), but with min_competence set it must FAIL (blind leg n_eff=1).
    good = (rng.random(n) < 0.75).astype(int); blind = (rng.random(n) < 0.20).astype(int)   # blind ~ chance
    r7u = decorrelation_validity_cert([("good", good), ("blind", blind)])                     # unguarded -> mirage
    r7g = decorrelation_validity_cert([("good", good), ("blind", blind)], min_competence=0.35)
    tot += 1; ok += (r7u["validated"] and not r7g["validated"] and r7g["blind_members"] == ["blind"])
    # (8) ★L217b (fix): a member competent GLOBALLY but BLIND on a sub-mode (the adversarial minimum) fools the GLOBAL
    # guard but is caught by the STRATA (min-over-sub-mode) guard. A_good everywhere; B easy(0.9)/hard-chance(0.2).
    hard = rng.random(n) < 0.30
    Ag = (rng.random(n) < 0.80).astype(int)
    Bcond = np.where(hard, rng.random(n) < 0.20, rng.random(n) < 0.90).astype(int)
    r8_global = decorrelation_validity_cert([("A", Ag), ("B", Bcond)], min_competence=0.35)             # global -> fooled
    r8_strata = decorrelation_validity_cert([("A", Ag), ("B", Bcond)], min_competence=0.35, strata=hard.astype(int))
    tot += 1; ok += (r8_global["blind_members"] == [] and r8_strata["blind_members"] == ["B"] and not r8_strata["validated"])
    # (9) ★ inf-sibling sentinel (completeness-audit): non-finite thresholds must fail CLOSED. thresh=+inf made
    # `worst < +inf` -> True -> validated; min_competence=-inf made every mean >= -inf -> COMPETENT. np.isfinite closes all.
    A9 = (rng.random(n) < 0.75).astype(int); B9 = (rng.random(n) < 0.75).astype(int)
    tot += 1; ok += all(not decorrelation_validity_cert([("A", A9), ("B", B9)], thresh=t, min_competence=mc)["validated"]
                        for (t, mc) in [(float("inf"), 0.3), (float("-inf"), 0.3), (float("nan"), 0.3),
                                        (0.5, float("-inf")), (0.5, float("inf")), (0.5, float("nan"))])
    print("decorrelation_validity_cert selftest: %d/%d (same-channel->NOT validated neff %.2f | cross-channel->validated "
          "neff %.2f | degenerate-safe | <2-members | blindness-mirage: unguarded=%s guarded=%s)"
          % (ok, tot, r["effective_neff"], r2["effective_neff"], r7u["validated"], r7g["validated"]))
    return ok == tot


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
