#!/usr/bin/env python3
"""★ STATUS CORRECTED (my over-deprecated this; the KRONAN crown refuted that empirically). Relationship to
the `video_admission_cert.py`: they are DISTINCT composers for DIFFERENT leg-input patterns, NOT duplicates --
the §7 cert composes METHOD-SIGNAL sync (SfM/mono-depth/mesh/flow redundancy) + temporal vetoes; THIS §11 cert
composes N GENERIC decorrelated legs by (covers, errvec_set) with the coverage-span vs within-mode-n_eff split, the
single-source guard, and the CI-lo(AUC)/CI-upper(corr) blind gates. The first real fresh-video crown attempt
(KRONAN, fdd0ff069) ran through THIS cert and it ABSTAINed 25/25 correctly on the single-source precondition
(verified on real substrate). => it IS the working agreement composer; keep it registered.
They OVERLAP in the two-axis accounting concept but serve different callers. (My  corr-CI-upper-from-n remains a
genuine addition the scalar interface delegates to the caller.)

video_admission_agreement_cert -- compose the temporal-family + cross-modal legs into ONE
GT-free video-admission verdict (ADMIT / ABSTAIN / VETO), with the TWO decorrelation axes kept HONESTLY SEPARATE
per the co-observability clarification (d_coobservability_composition.md) which grounds in
:

  AXIS 1 -- within-mode n_eff (DECORRELATION): among legs that CO-OBSERVE the same fake set (same errvec keys),
            the worst-pair error-correlation -> Kish n_eff. This is the ONLY place a pairwise "decorrelation" number
            is defined. (geom x photo on  spatial-swap: corr 0.284 -> the locked co-observing pair.)
  AXIS 2 -- cross-mode COVERAGE (SPAN): mode-specific legs (each AUC~1 on its OWN fake mode, BLIND on others) each
            close a distinct FAKE_DIM. This is rank/span, NOT pairwise corr. Counting a coverage leg toward n_eff
            OVER-COUNTS independence (a shared-weight failure mode).

ADMIT iff: every admitted leg passes its BLIND-LEG-GUARD (own-AUC on its own mode > bar) AND the required FAKE_DIMS
are all COVERED by passing legs AND every CO-OBSERVED mode reaches n_eff >= the bar (a co-observed mode with only
one non-blind leg is single-source -> ABSTAIN on that mode, not ADMIT). A leg failing its blind-leg-guard is
measured-blind -> its credit (coverage AND n_eff) is REFUSED (cannot be fooled into counting a blind leg). numpy only.
Selftest: `python -m graph_engine.video_admission_agreement_cert`."""
from __future__ import annotations
import numpy as np

try:
    from .decorrelation_validity_cert import _pair_error_corr
except ImportError:
    try:
        from decorrelation_validity_cert import _pair_error_corr
    except ImportError:
        def _pair_error_corr(a, b):
            a = np.asarray(a, float); b = np.asarray(b, float)
            if a.std() < 1e-9 or b.std() < 1e-9:
                return 1.0
            r = float(np.corrcoef(a, b)[0, 1])
            return r if np.isfinite(r) else 1.0

BLIND_GUARD_BAR = 0.60     # own-AUC on the leg's OWN mode must exceed this (§11 board)
NEFF_BAR = 1.30            # a co-observed mode needs at least this Kish n_eff to certify (else single-source)


def _hanley_mcneil_se(auc, n_pos, n_neg):
    """Hanley-McNeil (1982) SE of an AUC given the class sizes it was measured on. Classical formula (NOT lane-A IP;
    surfaced the discipline -- gate the blind-leg-guard on the CI-lower-bound, not the point AUC)."""
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    Q1 = auc / (2.0 - auc)
    Q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (Q1 - auc * auc) + (n_neg - 1) * (Q2 - auc * auc)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def _blind_guard_value(lg, z=1.96):
    """The value the blind-leg-guard compares to the bar. If the leg supplies its AUC's class sizes, gate on the
    Hanley-McNeil CI-LOWER-BOUND (own_auc - z*SE) so a thin-n high-point-AUC leg cannot pass under-powered (
    gate-on-CI-lower-bound). Else gate the POINT auc and mark it ci-unverified (declared, not silently trusted)."""
    auc = lg.get("own_auc")
    if auc is None or not np.isfinite(auc):
        return None, "no own_auc", True
    npos, nneg = lg.get("own_auc_n_pos"), lg.get("own_auc_n_neg")
    if npos and nneg:
        se = _hanley_mcneil_se(float(auc), int(npos), int(nneg))
        return float(auc) - z * se, "CI-lo (HM n=%d/%d)" % (npos, nneg), False
    return float(auc), "point (ci-UNVERIFIED)", True


def _corr_ci_upper(r, n, z=1.96):
    """Fisher z-transform CI-UPPER bound of a correlation r measured on n samples (conservative for a decorrelation
    claim: the true corr could be this high). At large n -> r; at thin n -> substantially above r. ."""
    if n is None or n < 4 or not np.isfinite(r):
        return 1.0 if not np.isfinite(r) else min(1.0, r + 0.5)   # thin/undefined -> pessimistic
    r = float(np.clip(r, -0.999, 0.999))
    zr = np.arctanh(r) + z / np.sqrt(n - 3)
    return float(np.tanh(zr))


def _kish_neff(worst_corr, k):
    """K-way Kish n_eff from the worst positive pairwise error-corr (matches decorrelation_validity_cert)."""
    w = max(worst_corr, 0.0)
    return k / (1.0 + (k - 1) * w)


def _participation_neff(vecs, n_shared, neff_bar=NEFF_BAR):
    """/ -- the within-mode corroboration n_eff FORM. The worst-pair-Kish form _kish_neff(worst_pair_corr, k) plugs
    the WORST pairwise corr into the K-way formula -- for k>=3 it assumes ALL pairs are that bad and DISCARDS the
    decorrelation of legs OUTSIDE the worst pair (a redundant pair A,B makes it ABSTAIN even when a third leg C is fully
    decorrelated). Use the PARTICIPATION RATIO k^2/(1'R1) on the CI-upper-adjusted corr matrix -- it credits every leg's
    decorrelation. Guards:
      (1) each off-diagonal is the Fisher-z CI-UPPER of the pair corr (conservative / fail-closed, preserving),
      (2) negatives CLAMPED to 0, result CAPPED at k -- else participation-ratio over-credits on anti-corr (3.33>k=2).
      (3) ★ SHARED-BLIND-SPOT GUARD (found by an adversarial red-team: swapping the n_eff functional under the FIXED
          bar 1.30 is a functional-swap-under-fixed-bar fail-open): the participation ratio
          AVERAGES the pairwise corrs, so a config with NO genuinely-decorrelated pair (every pair moderately correlated,
          e.g..72/.55/.55 -- all a shared blind spot) can average PAST the bar and ADMIT on shared error, while worst-pair
          -Kish correctly abstains. FIX: credit the participation ratio ONLY IF a genuinely-corroborating pair exists -- the
          MOST-decorrelated pair (min off-diagonal) must alone clear the bar (2/(1+min_off) >= neff_bar). Else there is no
          decorrelated corroboration -> fall back to the conservative worst-pair-Kish (abstain). This closes the fail-open
          while preserving the k>=3 over-abstention fix (a redundant pair has a decorrelated pair elsewhere -> still ADMIT).
    For k=2 this is IDENTICAL to worst-pair-Kish (2/(1+w)) -- no regression."""
    k = len(vecs)
    R = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            r_hi = _corr_ci_upper(_pair_error_corr(vecs[i], vecs[j]), n_shared)
            R[i, j] = R[j, i] = min(max(r_hi, 0.0), 1.0)      # clamp neg->0 (cap at k), clamp to 1
    iu = np.triu_indices(k, 1)
    offs = R[iu]
    worst = float(max(offs.max(), 0.0)) if offs.size else 0.0
    min_off = float(max(offs.min(), 0.0)) if offs.size else 0.0
    # (3) shared-blind-spot guard: require a genuinely-decorrelated pair (the best pair alone clears the bar)
    if offs.size and 2.0 / (1.0 + min_off) < neff_bar:
        return _kish_neff(worst, k)                            # no decorrelated pair -> conservative (fail-closed)
    one = np.ones(k)
    pr = float((one @ one) ** 2 / (one @ R @ one))
    return min(pr, float(k))


def video_admission_agreement_cert(legs, required_fake_dims, blind_guard_bar=BLIND_GUARD_BAR, neff_bar=NEFF_BAR):
    """legs: list of dicts, each:
        name: str
        own_auc: float in [0,1]  -- the leg's discrimination on ITS OWN fake mode (blind-leg-guard input)
        covers: list[str]       -- the FAKE_DIMS this leg closes
        errvec: dict{sample_id: score} or None -- per-sample scores on a fake set (for n_eff among co-observers)
        errvec_set: str or None     -- the NAME of the fake set errvec is on (co-observability key)
        errvec_labels: dict{sample_id: 0/1} or None -- ground-truth clean/fake on that set (for error-correctness)
    required_fake_dims: iterable[str] -- the dims a full video-admission must cover.
    Returns dict(verdict, admit, covered_dims, uncovered_dims, blind_legs, coobserved_neff, reason)."""
    required = set(required_fake_dims)
    # (1) blind-leg-guard: refuse any leg not CONFIDENTLY competent on its OWN mode. ★ (CI discipline on
    # the sibling video_admission_cert): gate on the AUC CI-LOWER-BOUND when class sizes are supplied -- a thin-n
    # own_auc=0.65 (CI-lo 0.41 < chance) must NOT pass the 0.6 point bar.
    passing, blind, ci_unverified = [], [], []
    for lg in legs:
        val, basis, unver = _blind_guard_value(lg)
        if val is None or val <= blind_guard_bar:
            blind.append(lg["name"])
        else:
            passing.append(lg)
            if unver:
                ci_unverified.append(lg["name"])
    if not passing:
        return dict(verdict="VETO", admit=False, covered_dims=[], uncovered_dims=sorted(required),
                    blind_legs=blind, coobserved_neff={}, reason="no leg passed the blind-leg-guard (all measured-blind)")

    # (2) AXIS 2 -- coverage span from passing legs only
    covered = set()
    for lg in passing:
        covered |= set(lg.get("covers", []))
    uncovered = required - covered

    # (3) AXIS 1 -- within-mode n_eff among CO-OBSERVING passing legs (grouped by errvec_set). Pairwise corr is only
    # defined here. A mode with a single non-blind co-observer is single-source (n_eff undefined -> flagged).
    groups = {}
    for lg in passing:
        s = lg.get("errvec_set")
        if s and lg.get("errvec") is not None:
            groups.setdefault(s, []).append(lg)
    coobs_neff, single_source = {}, []
    for s, grp in groups.items():
        if len(grp) < 2:
            single_source.append(s)
            coobs_neff[s] = None
            continue
        # error vectors aligned on the shared sample ids; error = |score - label| when labels present else score
        ids = set(grp[0]["errvec"])
        for g in grp[1:]:
            ids &= set(g["errvec"])
        ids = sorted(ids)
        # ★ VOID-FLOOR FIX (confirmed defect, independently caught by the M3D×CoTracker subagent): the family
        # error-corr must be measured on the POSITIVES (fakes) ONLY when labels are present. Including negatives (reals)
        # injects a void-floor -- both legs correctly score reals ~0 (error ~0), a correlated-LOW block that inflates
        # the corr and UNDERSTATES n_eff -> genuinely-decorrelated legs that MISS DIFFERENT fakes (anti-correlated
        # fake-errors, the ideal case) are FALSELY ABSTAINED (measured: decorrelated pair all-clips corr 0.65 -> n_eff
        # 1.12 <bar, but fake-only corr -0.68 -> n_eff 1.47). Decorrelation for a fake-detector = do the legs miss the
        # SAME fakes (correlated blind-spots) or DIFFERENT fakes (decorrelated)? -- a POSITIVES-only question.
        labs = [g.get("errvec_labels") for g in grp]
        if all(l is not None for l in labs):
            pos_ids = [i for i in ids if all(l.get(i, 0) == 1 for l in labs)]
        else:
            pos_ids = ids                       # no labels -> fall back to all shared ids (raw errvec)
        n_shared = len(pos_ids)
        MIN_FAKES = 8
        if n_shared < MIN_FAKES:                # too few positives to estimate the decorrelation -> undecidable
            single_source.append(s); coobs_neff[s] = None
            continue
        vecs = []
        for g, lab in zip(grp, labs):
            vecs.append(np.array([(abs(g["errvec"][i] - lab[i]) if lab else g["errvec"][i]) for i in pos_ids], float))
        # ★ (fixed-magnitude-no-significance sweep, self-caught): n_eff gated on the POINT worst-pair corr
        # false-passes a TRULY-correlated pair at thin n (measured: 2.5% false-ADMIT at n=12) because sampling noise
        # reads it as decorrelated. Gate on the Fisher-z CI-UPPER bound of the correlation -> a thin-n pair whose corr
        # COULD be higher gets a conservatively-LOWER n_eff (fail-closed). Thick n -> CI tight -> ~unchanged.
        # ★ (n_eff-form kill): the worst-pair-Kish form over-abstains at k>=3 (discards decorrelation of legs
        # outside the worst pair); use the clamped-capped participation ratio (identical to worst-pair-Kish at k=2, no
        # regression) which credits every leg's decorrelation while capping the anti-corr over-credit. Each off-diagonal
        # keeps the Fisher-z CI-UPPER (conservative).
        coobs_neff[s] = round(_participation_neff(vecs, n_shared, neff_bar), 3)

    # (4) verdict
    weak_neff = [s for s, v in coobs_neff.items() if v is not None and v < neff_bar]
    if uncovered:
        verdict, admit = "ABSTAIN", False
        reason = ("uncovered FAKE_DIMS %s -- no passing leg closes them; the video is un-admissible until a leg "
                  "covering each is added (coverage axis)" % sorted(uncovered))
    elif weak_neff:
        verdict, admit = "ABSTAIN", False
        reason = ("co-observed mode(s) %s below n_eff bar %.2f (worst-pair too correlated) -- the within-mode "
                  "decorrelation is insufficient there; agreement could be shared error" % (weak_neff, neff_bar))
    elif single_source:
        verdict, admit = "ABSTAIN", False
        reason = ("mode(s) %s are certified by a SINGLE non-blind co-observer -- no pairwise decorrelation available; "
                  "agreement is unmeasurable (need a 2nd co-observing leg on that set)" % single_source)
    elif not coobs_neff:
        # ★fix (D-SWEEP-L-AGREEMENT-CERT-LATENT-VACUOUS-EMPTY-FAILOPEN): NO leg supplied an errvec on any shared
        # set -> the within-mode n_eff (AXIS-1, the DECORRELATION axis) was NEVER measured, so coobs_neff is {} and the
        # old ADMIT branch asserted "every co-observed mode n_eff>=bar ({})" -- a VACUOUS universal over an EMPTY set
        #: measured decorrelation ABSENT read as PASS. Correct: ABSTAIN -- coverage alone
        # (own-AUC + declared covers) is identity/nominal, NOT the measured decorrelation this cert certifies.
        verdict, admit = "ABSTAIN", False
        reason = ("no co-observed mode has a measured n_eff -- no passing leg supplied errvec on a shared set, so the "
                  "within-mode DECORRELATION axis is UNMEASURED; coverage alone does not certify agreement. Supply "
                  "errvec/errvec_set on >=2 co-observing legs per required mode.")
    else:
        verdict, admit = "ADMIT", True
        reason = ("ADMIT: all required FAKE_DIMS %s covered; every co-observed mode n_eff>=%.2f (%s); %d legs pass the "
                  "blind-leg-guard. Coverage(span)+within-mode-n_eff both satisfied." % (sorted(required), neff_bar,
                  {s: coobs_neff[s] for s in coobs_neff}, len(passing)))
    return dict(verdict=verdict, admit=admit, covered_dims=sorted(covered), uncovered_dims=sorted(uncovered),
                blind_legs=blind, coobserved_neff=coobs_neff, single_source_modes=single_source,
                ci_unverified_legs=ci_unverified,   # ★L273: legs admitted on a POINT own_auc (no class sizes) --
                # their competence is not CONFIDENCE-checked; supply own_auc_n_pos/n_neg for a Hanley-McNeil CI gate.
                reason=reason)


def _selftest():
    ok = tot = 0
    rng = np.random.default_rng(11)
    n = 400
    lab = {("s%d" % i): int(rng.random() < 0.5) for i in range(n)}
    # two CO-OBSERVING decorrelated legs on the "spatial_swap" set (geom x photo analog)
    def make_errvec(sep, corr_src=None, shared=0.0):
        ev = {}
        for i in range(n):
            base = (0.2 if lab["s%d" % i] == 0 else 0.8) + rng.normal(0, sep)
            if corr_src is not None:
                base = shared * corr_src["s%d" % i] + (1 - shared) * base
            ev["s%d" % i] = float(np.clip(base, 0, 1))
        return ev
    geom = make_errvec(0.12)
    photo = make_errvec(0.12)                 # independent -> low error-corr -> n_eff ~ 2
    temporal = dict(name="temporal", own_auc=1.0, covers=["temporal_splice"], errvec=None, errvec_set=None)
    sfm = dict(name="sfm_pose", own_auc=0.95, covers=["motion_splice"], errvec=None, errvec_set=None)
    L_geom = dict(name="geom", own_auc=0.85, covers=["spatial_swap"], errvec=geom, errvec_set="g1170",
                  errvec_labels=lab)
    L_photo = dict(name="photo", own_auc=0.84, covers=["spatial_swap"], errvec=photo, errvec_set="g1170",
                   errvec_labels=lab)
    REQ = ["spatial_swap", "temporal_splice", "motion_splice"]

    # (1) full set -> ADMIT (coverage complete + co-observing pair n_eff ok)
    r1 = video_admission_agreement_cert([temporal, sfm, L_geom, L_photo], REQ)
    tot += 1; ok += (r1["verdict"] == "ADMIT" and r1["coobserved_neff"]["g1170"] >= NEFF_BAR)
    # (2) drop the temporal leg -> temporal_splice uncovered -> ABSTAIN (coverage axis)
    r2 = video_admission_agreement_cert([sfm, L_geom, L_photo], REQ)
    tot += 1; ok += (r2["verdict"] == "ABSTAIN" and "temporal_splice" in r2["uncovered_dims"])
    # (3) a BLIND leg (own_auc below bar) claiming to cover a dim -> refused -> that dim uncovered -> ABSTAIN
    blind_temporal = dict(name="temporal", own_auc=0.55, covers=["temporal_splice"], errvec=None, errvec_set=None)
    r3 = video_admission_agreement_cert([blind_temporal, sfm, L_geom, L_photo], REQ)
    tot += 1; ok += (r3["verdict"] == "ABSTAIN" and "temporal" in r3["blind_legs"]
                     and "temporal_splice" in r3["uncovered_dims"])
    # (4) CORRELATED co-observing pair (photo:= mostly geom) -> n_eff collapses -> ABSTAIN (n_eff axis, NOT coverage)
    photo_corr = make_errvec(0.12, corr_src=geom, shared=0.92)
    L_photo_c = dict(name="photo", own_auc=0.84, covers=["spatial_swap"], errvec=photo_corr, errvec_set="g1170",
                     errvec_labels=lab)
    r4 = video_admission_agreement_cert([temporal, sfm, L_geom, L_photo_c], REQ)
    tot += 1; ok += (r4["verdict"] == "ABSTAIN" and r4["coobserved_neff"]["g1170"] < NEFF_BAR
                     and not r4["uncovered_dims"])          # coverage fine, n_eff is the blocker
    # (5) SINGLE co-observer on spatial_swap (drop photo) -> single-source -> ABSTAIN (agreement unmeasurable)
    r5 = video_admission_agreement_cert([temporal, sfm, L_geom], REQ)
    tot += 1; ok += (r5["verdict"] == "ABSTAIN" and "g1170" in r5["single_source_modes"])
    # (6) mode-specific coverage leg must NOT inflate n_eff: n_eff is keyed on only, temporal/sfm absent from it
    tot += 1; ok += (set(r1["coobserved_neff"]) == {"g1170"})
    # (7) ★ CI-lower-bound blind-leg-guard: a leg with a HIGH POINT own_auc=0.65 but THIN n (10/10, CI-lo 0.41 <
    # chance) must be REFUSED (measured-blind), while the SAME point auc at thick n (300/300, CI-lo>0.6) passes.
    thin = dict(name="temporal", own_auc=0.65, own_auc_n_pos=10, own_auc_n_neg=10, covers=["temporal_splice"],
                errvec=None, errvec_set=None)
    thick = dict(name="temporal", own_auc=0.65, own_auc_n_pos=300, own_auc_n_neg=300, covers=["temporal_splice"],
                 errvec=None, errvec_set=None)
    r7t = video_admission_agreement_cert([thin, sfm, L_geom, L_photo], REQ)
    r7k = video_admission_agreement_cert([thick, sfm, L_geom, L_photo], REQ)
    tot += 1; ok += ("temporal" in r7t["blind_legs"] and r7t["verdict"] == "ABSTAIN"
                     and "temporal" not in r7k["blind_legs"])
    # (8) ★ ci_unverified surfaced: a point-only own_auc (no class sizes) admits but is flagged ci-unverified
    tot += 1; ok += ("temporal" in r1["ci_unverified_legs"])   # r1's temporal leg has no class sizes
    # (9) ★ (fixed-magnitude-no-significance class, self-caught): a TRULY-correlated co-observing pair at
    # THIN n must NOT false-ADMIT (point corr reads decorrelated by noise -> gate on Fisher-z CI-upper). And a
    # DECORRELATED pair at THICK n must still ADMIT (no over-abstain).
    def _corr_pair(nn, sd, shared):
        rg = np.random.default_rng(sd)
        lb = {"s%d" % i: int(rg.random() < 0.5) for i in range(nn)}
        sh = rg.normal(0, 1, nn)
        ev1 = {"s%d" % i: float(lb["s%d" % i] + shared * sh[i] + (1 - shared) * rg.normal()) for i in range(nn)}
        ev2 = {"s%d" % i: float(lb["s%d" % i] + shared * sh[i] + (1 - shared) * rg.normal()) for i in range(nn)}
        a = dict(name="a", own_auc=0.9, covers=["x"], errvec=ev1, errvec_set="S", errvec_labels=lb)
        b = dict(name="b", own_auc=0.9, covers=["x"], errvec=ev2, errvec_set="S", errvec_labels=lb)
        return video_admission_agreement_cert([a, b], ["x"])["verdict"]
    corr_thin_admits = sum(1 for sd in range(60) if _corr_pair(12, sd, 0.8) == "ADMIT")
    deco_thick_admits = sum(1 for sd in range(60) if _corr_pair(600, sd, 0.0) == "ADMIT")
    tot += 1; ok += (corr_thin_admits == 0 and deco_thick_admits >= 55)
    # (10) ★ VOID-FLOOR sentinel: two legs that MISS DIFFERENT fakes (anti-correlated fake-errors = genuinely
    # decorrelated) must ADMIT -- the family-corr is measured POSITIVES-ONLY so real-clip agreement can't inflate
    # it into a false ABSTAIN; while two legs that miss the SAME fakes must still ABSTAIN (real correlation).
    rgv = np.random.default_rng(285); nv = 80; nfv = nv - nv // 2
    yv = np.r_[np.zeros(nv // 2), np.ones(nfv)]; kv = np.r_[np.full(nv // 2, -1), rgv.integers(0, 2, nfv)]
    dv = np.where(kv == 0, rgv.uniform(.3, .7, nv), rgv.uniform(0, .15, nv))
    av = np.where(kv == 1, rgv.uniform(.3, .7, nv), rgv.uniform(0, .15, nv))
    dx = np.where(yv == 1, rgv.uniform(.3, .7, nv), rgv.uniform(0, .15, nv))
    def _vleg(nm, sc):
        return dict(name=nm, own_auc=0.9, own_auc_n_pos=int((yv == 1).sum()), own_auc_n_neg=int((yv == 0).sum()),
                    covers=["geo"], errvec={"s%d" % i: float(sc[i]) for i in range(nv)}, errvec_set="G",
                    errvec_labels={"s%d" % i: int(yv[i]) for i in range(nv)})
    A_diff = np.clip(dv + rgv.normal(0, .1, nv), 0, 1); B_diff = np.clip(av + rgv.normal(0, .1, nv), 0, 1)
    A_same = np.clip(dx + rgv.normal(0, .1, nv), 0, 1); B_same = np.clip(dx + rgv.normal(0, .1, nv), 0, 1)
    r_diff = video_admission_agreement_cert([_vleg("a", A_diff), _vleg("b", B_diff)], ["geo"])
    r_same = video_admission_agreement_cert([_vleg("a", A_same), _vleg("b", B_same)], ["geo"])
    tot += 1; ok += (r_diff["verdict"] == "ADMIT" and r_diff["coobserved_neff"]["G"] >= 1.30
                     and r_same["verdict"] == "ABSTAIN" and r_same["coobserved_neff"]["G"] < 1.30)

    print("video_admission_agreement_cert selftest: %d/%d (full->ADMIT neff=%.2f | drop-temporal->ABSTAIN uncovered | "
          "blind-leg-refused->ABSTAIN | correlated-pair->ABSTAIN neff=%.2f | single-source->ABSTAIN | neff-not-inflated "
          "| thin-n-AUC-refused(CI-lo) | ci-unverified-flagged)"
          % (ok, tot, r1["coobserved_neff"]["g1170"], r4["coobserved_neff"]["g1170"]))
    return ok == tot


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
