#!/usr/bin/env python3
"""agreement_admission_gate — one gate that composes the decorrelation-mirage guards for a K>=2 member ensemble
(the video-admission quality gate). It returns ADMIT / ABSTAIN /
VETO with power AND coverage, and stress-tests show EACH guard is load-bearing (removing any leaves a mirage hole).

Composed guards (see the mirage taxonomy):
  1. decorrelation_validity_cert(min_competence, strata)  -> GLOBAL blindness  + CONDITIONAL/sub-mode blindness
     + the rho decorrelation necessary-condition.
  2. certifying_power_cert (Wilson lower bound)            -> SHARED CONFUSION  + small-n over-claim.
  3. coverage under the agreement rule (all vs majority)   -> K>2 blind-member coverage VETO; report power AND coverage.

★HONEST RESIDUAL (the one guard this static gate CANNOT self-run): the CALIBRATION-COVERAGE meta-mirage  -- if the
supplied calibration data does not SPAN the adversarial minimum, certifying power is measured on easy data and over-certifies.
This gate cannot verify its own calibration coverage from static answers; that needs adversarial_min_registry_check
with a fake-generator. So the gate DECLARES this precondition (calibration must span the adversarial minimum) rather than
silently guaranteeing it. numpy only. Selftest: `python -m graph_engine.agreement_admission_gate`."""
from __future__ import annotations
import numpy as np

try:
    from .decorrelation_validity_cert import decorrelation_validity_cert
    from .certifying_power_cert import certifying_power_cert
except ImportError:
    from decorrelation_validity_cert import decorrelation_validity_cert
    from certifying_power_cert import certifying_power_cert

STRATA_PRECONDITION = ("sub-mode/conditional blindness (L219) is only checked when `strata` is passed; with strata=None "
                       "guard #1 tests GLOBAL competence only -- a member blind on an UN-passed adversarial-minimum "
                       "sub-mode is NOT caught and can be ADMITTED. Pass strata=<sub-mode labels> to close this. "
                       "★L264: even WITH strata, label QUALITY is only partially self-checkable: mis-labels hiding "
                       "SHARED-CONFUSION are caught (q witness, L263), but a hidden DISAGREEING-blind sub-mode is "
                       "structurally unwitnessable from calibration data (measured: composed gate ADMITs at true "
                       "hidden-stratum power 0.85 vs bar 0.90, aggregate 0.98) -- labels spanning the blindness "
                       "regimes is a DECLARED precondition, verify it upstream (e.g. adversarial_min_search).")
CALIBRATION_PRECONDITION = ("calibration data must SPAN the adversarial minimum (hardest confusion mode); this static gate "
                            "cannot self-verify that -- use adversarial_min_registry_check (L210) with a fake-generator.")


def _coverage(answers, rule, need):
    stk = np.stack(answers, 0); n = stk.shape[1]
    if rule == "all":
        return float(np.all(stk == stk[0], axis=0).mean())
    # majority: a value shared by >= need members exists
    cov = 0
    for i in range(n):
        _, counts = np.unique(stk[:, i], return_counts=True)
        cov += int(counts.max() >= need)
    return float(cov / n)


def agreement_admission_gate(member_answers, true_labels, strata=None, min_power=0.90, min_competence=None,
                             agreement_rule="majority", min_coverage=0.05):
    """member_answers: list of K arrays (predicted labels); true_labels: array. Returns dict(admit, verdict,
    decorrelation_validated, certifying_power, coverage, reason, calibration_precondition). verdict in
    {ADMIT, ABSTAIN, VETO}: VETO = a member is blind/correlated (decorrelation fails); ABSTAIN = power below the confident
    bar or coverage too thin; ADMIT = decorrelated + confidently-powered + covered."""
    A = [np.asarray(a) for a in member_answers]
    y = np.asarray(true_labels)
    K = len(A)
    # ★ null-safety: validate INPUT SHAPE before any computation. A deployment can pass an EMPTY candidate pool
    # (see the vacuous-pool failure class) or ragged member arrays; the prior code crashed with an uncontrolled
    # ZeroDivisionError (empty -> _coverage cov/n=0/0) / AttributeError (ragged -> scalar (A==y).astype). A safety
    # gate must ABSTAIN with a reason on malformed input, never crash uncontrolled (safety-tool-must-abstain).
    def _malformed(reason):
        return dict(admit=False, verdict="ABSTAIN", decorrelation_validated=False, blind_members=[], abstain_members=[],
                    certifying_power=None, certifying_power_lower_ci=None, coverage=0.0, n_members=K,
                    reason="malformed input (ABSTAIN, not a pass): " + reason, calibration_precondition=CALIBRATION_PRECONDITION)
    if K < 2:
        return _malformed("need >=2 members, got K=%d" % K)
    # ★ (red-team HOLE5/6): 0-d / non-1-D arrays crashed on `.shape[0]` BEFORE the length guards. Check ndim first.
    if y.ndim != 1 or any(a.ndim != 1 for a in A):
        return _malformed("member_answers/true_labels must be 1-D (got member ndims %s, label ndim %d)"
                          % ([a.ndim for a in A], y.ndim))
    n = y.shape[0]
    if n == 0:
        return _malformed("empty true_labels / candidate pool (n=0) -- nothing to certify")
    if any(a.shape[0] != n for a in A):
        return _malformed("ragged input: member lengths %s != n_labels %d" % ([int(a.shape[0]) for a in A], n))
    # ★ (§4c) + (red-team HOLE2/3/4): a non-finite THRESHOLD silently disables a reject-gate -> false ADMIT.
    # HOLE2/3: the old `is not None` guard SKIPPED None -> min_power/min_coverage=None crashed downstream; only
    # min_competence legitimately accepts None (=off). HOLE4: an ARRAY threshold broke scalar np.isfinite. Require a
    # finite SCALAR (or None only where allowed).
    def _bad_thresh(v, allow_none):
        if v is None:
            return not allow_none
        try:
            arr = np.asarray(v, float)
        except (ValueError, TypeError):
            return True
        return arr.ndim != 0 or not bool(np.isfinite(arr))
    for _tn, _tv, _allow_none in (("min_power", min_power, False), ("min_coverage", min_coverage, False),
                                  ("min_competence", min_competence, True)):
        if _bad_thresh(_tv, _allow_none):
            return _malformed("bad threshold %s=%r -- need a finite scalar%s; ABSTAIN"
                              % (_tn, _tv, " or None" if _allow_none else ""))
    # ★ (red-team HOLE1, headline false-ADMIT): a NON-FINITE strata LABEL is silently DROPPED from the per-stratum
    # loop (strata==nan is False everywhere), so a member BLIND on exactly the adversarial-minimum stratum becomes
    # invisible -> ADMIT instead of VETO (20/20 seeds). Reject non-finite / mis-shaped strata here (and in the sub-cert).
    if strata is not None:
        try:
            s_arr = np.asarray(strata, float)
        except (ValueError, TypeError):
            return _malformed("strata must be a numeric 1-D array of finite labels")
        if s_arr.ndim != 1 or s_arr.shape[0] != n or s_arr.size == 0 or not bool(np.all(np.isfinite(s_arr))):
            return _malformed("strata must be a finite 1-D array of length n=%d (shape=%s, all-finite=%s)"
                              % (n, s_arr.shape, bool(np.all(np.isfinite(s_arr))) if s_arr.size else False))
    correctness = [("m%d" % i, (A[i] == y).astype(int)) for i in range(K)]
    dec = decorrelation_validity_cert(correctness, min_competence=min_competence, strata=strata)
    # ★: route strata into the POWER leg too -- the aggregation mirage (a q~1 subgroup hidden under an aggregate
    # power_lo>=bar) is caught by the worst-stratum gate exactly like sub-mode blindness is in the decorrelation leg.
    powr = certifying_power_cert(A, y, min_power=min_power, strata=strata)
    # ★ (subagent verdict-correctness DEFECT2): majority must SCALE with K. The old `2` was correct only for
    # K in {2,3}; at K>=4 a correlated MINORITY pair (2 of 4) satisfied "majority" coverage while true 3-of-4 consensus
    # never occurred -> wrong ADMIT. Strict majority = K//2 + 1 (reduces to 2 at K=2,3 -> no regression).
    need = (K // 2 + 1) if agreement_rule == "majority" else K
    cov = _coverage(A, agreement_rule, need)

    # ★ (end-to-end re-grade): ROUTE the decorrelation THREE-WAY verdict -- do NOT collapse ABSTAIN into
    # VETO. A blind member is a PERMANENT VETO; a competence-ABSTAIN (thin n on a stratum) is a RECOVERABLE ABSTAIN
    # (gather more data), NOT a veto; a correlated pair is a permanent VETO. Collapsing to `not validated -> VETO` lost
    # the recoverable-vs-permanent distinction (VETO'd a genuine thin-n member 116/200).
    if dec.get("blind_members"):
        verdict, admit = "VETO", False
        reason = "BLIND member(s) %s (permanent): %s" % (dec["blind_members"], dec["reason"])
    elif dec.get("abstain_members"):
        verdict, admit = "ABSTAIN", False
        reason = ("competence UNDECIDABLE (thin n, RECOVERABLE) for %s -- gather more data on the adversarial-minimum "
                  "stratum; NOT a veto: %s" % (dec["abstain_members"], dec["reason"]))
    elif not dec["validated"]:
        verdict, admit = "VETO", False
        reason = "decorrelation FAILED (correlated members, permanent): %s" % dec["reason"]
    elif not powr["validated"]:
        verdict, admit = "ABSTAIN", False
        reason = "certifying power not confidently above bar: %s" % powr["reason"]
    elif not (cov >= min_coverage):   # ★L251 NaN-safe: `cov < min_coverage` fails-OPEN on cov=NaN; `not (>=)` fails-CLOSED (ABSTAIN)
        verdict, admit = "ABSTAIN", False
        reason = ("coverage %.3f < %.2f under '%s' rule -- a blind/broken member may be vetoing coverage (L218); "
                  "try majority or gate out blind legs" % (cov, min_coverage, agreement_rule))
    else:
        verdict, admit = "ADMIT", True
        reason = ("decorrelated + certifying power %.3f (Wilson lower) >= %.2f + coverage %.3f under '%s'"
                  % (powr.get("certifying_power_lower_ci", powr["certifying_power"]), min_power, cov, agreement_rule))
    return dict(admit=admit, verdict=verdict, decorrelation_validated=dec["validated"],
                blind_members=dec.get("blind_members", []), abstain_members=dec.get("abstain_members", []),
                certifying_power=powr["certifying_power"], certifying_power_lower_ci=powr.get("certifying_power_lower_ci"),
                coverage=round(cov, 3), n_members=K, reason=reason, calibration_precondition=CALIBRATION_PRECONDITION,
                # ★ DEFECT1 (documented-but-ungated): surface the strata precondition as a WARNING when strata is
                # omitted, so an ADMIT with strata=None is not mistaken for sub-mode-blindness-checked.
                strata_precondition=(None if strata is not None else STRATA_PRECONDITION))


def _selftest():
    ok = tot = 0
    C = 5; rng = np.random.default_rng(0); n = 8000
    y = rng.integers(0, C, n)
    hard = rng.random(n) < 0.30

    def good(p=0.78):
        a = y.copy(); w = rng.random(n) >= p; a[w] = (y[w] + rng.integers(1, C, int(w.sum()))) % C; return a
    g1, g2 = good(), good()
    blind = rng.integers(0, C, n)
    cond = np.where(hard, rng.integers(0, C, n), good(0.9))          # competent on easy, blind on hard sub-mode
    shared = y.copy(); shared2 = y.copy()                            # shared confusion on hard
    for M in (shared, shared2):
        e = (~hard) & (rng.random(n) < 0.1); M[e] = (y[e] + rng.integers(1, C, int(e.sum()))) % C
    att = (y[hard] + 1) % C; shared[hard] = att; shared2[hard] = att

    # (1) good+good decorrelated competent -> ADMIT (+★ field-completeness: assert admit + empty member fields)
    r1 = agreement_admission_gate([g1, g2], y, min_competence=0.35)
    tot += 1; ok += (r1["verdict"] == "ADMIT" and r1["admit"] is True and not r1["blind_members"] and not r1["abstain_members"])
    # (2) GLOBAL blindness -> VETO (min_competence guard); ★ also assert the SURFACED blind_members NAMES the
    # blind member + admit False (field-completeness -- a downstream router trusts these fields; break-test method:
    # assert every watched field, not just the verdict, else a silently-emptied field is a blind spot).
    r2 = agreement_admission_gate([g1, blind], y, min_competence=0.35)
    tot += 1; ok += (r2["verdict"] == "VETO" and r2["blind_members"] and r2["admit"] is False)
    # (3) CONDITIONAL blindness -> VETO with strata guard; assert blind_members surfaced + admit False
    r3s = agreement_admission_gate([g1, cond], y, min_competence=0.35, strata=hard.astype(int))
    tot += 1; ok += (r3s["verdict"] == "VETO" and r3s["blind_members"] and r3s["admit"] is False)
    # (4) SHARED CONFUSION on a FULL calibration (spans hard) -> NOT ADMITTED. Caught by the DECORRELATION guard (VETO):
    # on the hard mode both members are wrong TOGETHER (correlated errors, high rho), so decorrelation fails first --
    # the shared-confusion mirage is a mode-conditional CORRELATION on the hard mode, visible only when the
    # calibration spans it. (If errors were decorrelated-but-agreeing, the power guard would ABSTAIN instead.)
    r4 = agreement_admission_gate([shared, shared2], y, min_competence=0.35)
    tot += 1; ok += (r4["verdict"] in ("VETO", "ABSTAIN"))
    # (5) ★each guard LOAD-BEARING: shared-confusion on EASY-ONLY calibration SLIPS THROUGH (meta-mirage) -> the
    # honest residual the static gate cannot self-catch; verify it ADMITS (documenting the precondition, not a silent bug)
    easy = ~hard
    r5 = agreement_admission_gate([shared[easy], shared2[easy]], y[easy], min_competence=0.35)
    tot += 1; ok += (r5["verdict"] == "ADMIT" and "adversarial minimum" in r5["calibration_precondition"])
    # (6) K=3 with one blind under ALL-agree -> coverage collapses -> ABSTAIN; majority recovers
    r6 = agreement_admission_gate([g1, g2, blind], y, min_competence=None, agreement_rule="all", min_coverage=0.2)
    tot += 1; ok += (r6["verdict"] in ("ABSTAIN", "VETO"))
    # (7) ★ (end-to-end): a genuine thin-n MARGINAL member (0.50 on a small hard stratum) must NEVER be permanently
    # VETO'd (it was 116/200) -- it routes to the RECOVERABLE ABSTAIN (or ADMIT when it reads competent), and the
    # ABSTAIN state IS exercised. The three-way must not collapse into VETO.
    veto = abst = 0
    for s in range(40):
        rg = np.random.default_rng(s); m = 330; ys = rg.integers(0, C, m); hs = np.zeros(m, bool); hs[300:] = True
        gg = ys.copy(); wg = rg.random(m) >= 0.85; gg[wg] = (ys[wg] + rg.integers(1, C, int(wg.sum()))) % C
        mg = ys.copy(); wm = np.where(hs, rg.random(m) >= 0.50, rg.random(m) >= 0.90); mg[wm] = (ys[wm] + rg.integers(1, C, int(wm.sum()))) % C
        v = agreement_admission_gate([gg, mg], ys, min_competence=0.35, strata=hs.astype(int))["verdict"]
        veto += (v == "VETO"); abst += (v == "ABSTAIN")
    tot += 1; ok += (veto <= 2 and abst >= 10)   # recoverable member almost never VETO'd; ABSTAIN state exercised
    r7 = dict(verdict="veto=%d/40 abstain=%d/40" % (veto, abst))
    # (8) ★ null-safety REGRESSION sentinel: malformed input (empty pool / ragged members) must return a CLEAN
    # ABSTAIN (admit False, no crash), never a ZeroDivisionError/AttributeError. Guards the input-validation block.
    r8a = agreement_admission_gate([np.array([]), np.array([])], np.array([]))
    r8b = agreement_admission_gate([np.zeros(50, int), np.zeros(30, int)], np.zeros(50, int))
    tot += 1; ok += (r8a["verdict"] == "ABSTAIN" and r8a["admit"] is False and "malformed" in r8a["reason"]
                     and r8b["verdict"] == "ABSTAIN" and r8b["admit"] is False)
    # (9) ★ fail-OPEN sentinel (§4c class, decorrelated cross-check): a NaN THRESHOLD PARAMETER must NOT
    # silently disable a gate. min_coverage=NaN previously ADMITTED (cov < NaN is False -> ABSTAIN never fires).
    # Every non-finite threshold must fail CLOSED (ABSTAIN, admit False), never ADMIT.
    gg = y.copy(); wgg = rng.random(n) >= 0.8; gg[wgg] = (y[wgg] + rng.integers(1, C, int(wgg.sum()))) % C
    gg2 = y.copy(); wg2 = rng.random(n) >= 0.8; gg2[wg2] = (y[wg2] + rng.integers(1, C, int(wg2.sum()))) % C
    nan = float("nan")
    r9 = [agreement_admission_gate([gg, gg2], y, min_competence=0.35, min_coverage=nan),
          agreement_admission_gate([gg, gg2], y, min_competence=0.35, min_power=nan),
          agreement_admission_gate([gg, gg2], y, min_competence=nan)]
    tot += 1; ok += all(r["admit"] is False for r in r9)   # NO non-finite threshold may yield ADMIT
    # (10) ★ red-team sentinels: (a) headline false-ADMIT — a NaN-tagged strata label dropped the adversarial-min
    # stratum, hiding a blind member -> ADMIT; must now VETO/ABSTAIN (admit False). (b) None min_power/min_coverage
    # and 0-d/array inputs must ABSTAIN, not crash. Ground truth: a blind-on-hard member must NEVER be admitted.
    hard10 = rng.random(n) < 0.30
    cond10 = np.where(hard10, rng.integers(0, C, n), gg)          # competent easy, chance-blind on hard sub-mode
    s_nan = np.where(hard10, np.nan, 0.0)
    r10a = agreement_admission_gate([gg, cond10], y, min_competence=0.35, strata=s_nan)   # NaN strata attack
    r10b = agreement_admission_gate([gg, gg2], y, min_power=None)                          # None threshold
    r10c = agreement_admission_gate([np.asarray(5), gg2], y)                               # 0-d member
    tot += 1; ok += (r10a["admit"] is False and r10b["admit"] is False and r10c["admit"] is False)

    print("agreement_admission_gate selftest: %d/%d (good->%s | global-blind->%s | cond-blind(strata)->%s | "
          "shared-conf(full)->%s | shared-conf(easy=meta-mirage)->%s | K3-blind-allagree->%s | thin-n-abstain->%s)"
          % (ok, tot, r1["verdict"], r2["verdict"], r3s["verdict"], r4["verdict"], r5["verdict"], r6["verdict"], r7["verdict"]))
    return ok == tot


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
