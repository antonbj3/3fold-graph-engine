"""video_admission_cert — an agent worktree: the §7 (RACE video-at-scale) QUALITY GATE that decides whether a video
SEGMENT enters the corpus (ADMIT) or is rejected with a reason (ABSTAIN). No GT at scale → the quality signal is
CROSS-METHOD SYNC (SfM ⊥ mono-depth ⊥ mesh ⊥ flow agree) composed with the fresh Temporal certs.

Composition doctrine (D capstone, raw-max-or fusion collapses → safe_fuse; leg-admission by competence):
  - The multi-method GEOMETRY legs (SfM/mono-depth/mesh/flow) are REDUNDANT (all estimate the same geometry) →
    fuse their pairwise agreement into ONE sync score with a SAFE (conservative, weakest-link-aware) fuse, not a
    raw mean/max that a single confident-wrong method could dominate.
  - The TEMPORAL certs (motion-extrapolation reliability / motion-validity / deformation) are COMPLEMENTARY (they
    catch failure modes geometry-agreement is blind to: unpredictable maneuvers, invalid motion, non-rigidity) →
    each acts as an independent VETO with a named reason. Composition is NOT the AND of part-certs; it is
    safe_fuse(redundant) gated by the union of complementary vetoes.

★HONEST SCOPE (D coherent-fake thread — necessary-not-sufficient): cross-method agreement WITHOUT GT certifies
SYNC, not CORRECTNESS. A coherent fake — all methods sharing the SAME systematic bias (shared prior, common
mis-registration, identical depth-net backbone) — AGREES and would pass. ADMIT therefore means "cross-method
synced + temporally valid", NOT "geometrically correct". Coverage against coherent fakes requires ≥1 GENUINELY
DECORRELATED method in the mix (e.g. classical SfM ⊥ a learned mono-depth net); the cert REPORTS the method
diversity it saw and DOWN-WEIGHTS sync among near-duplicate methods, but cannot fully close the coherent-fake
floor without a decorrelated leg. Do not read ADMIT as a correctness guarantee.

null-safe: <2 methods, non-finite, too-short, or (near-)constant signals → cert_valid=False, ABSTAIN (never a
false ADMIT on structureless / degenerate input).

gate: python3 src/graph_engine/video_admission_cert.py
numpy only. Composes L motion_extrapolation_abstain_cert, C motion_validity_cert, J deformation_cert as the
temporal vetoes (caller passes their verdicts; this module does not import them so it stays functionally-live
standalone — see the temporal= contract).
"""
from __future__ import annotations
import numpy as np

SYNC_THRESH = 0.60          # median pairwise cross-method agreement to ADMIT
MIN_PAIR_THRESH = 0.40      # NO method-pair may fall below this (weakest-link floor)
DEGEN_STD = 1e-8            # a method signal with std below this is (near-)constant = degenerate
MIN_FRAMES = 8             # min per-signal length: below this a rank sync can't be certified (finite-permutation noise)
EXTRAP_RELIABLE_FRAC = 0.60 # motion-extrapolation reliable-fraction veto floor
FAMILY_DECORR_THRESH = 0.50 # max per-family-pair error-corr CI-UPPER-bound to count as decorrelated (fragile dir)
EFF_COVER_MAG = 0.60        # min CONTINUOUS-channel score_effective (injection-AUC magnitude) to count a dim EFFECTIVELY covered
MAG_MAX = 1.0               # max valid coverage magnitude = 2*(AUC-0.5) at AUC=1.0; a value >1 is an impossible/corrupt AUC and
# must fail-CLOSED (D fail-open fix, cert-layer twin of C's §14 helper AUC<=1.0 guard):
# a hand-built/corrupt leg with mag>1 bypasses C's helper and previously false-VALIDATED -> AUTO_ADMIT.
BLIND_POWER_THRESH = 0.10   # min per-leg CI-LOWER-BOUND magnitude 2*(AUC-0.5) on the corr set; below = BLIND (fragile dir)


def _worst_corr(v):
    """★N-LEG guard (§11: 2-ben -> >=3-ben). family_error_corr may be a SCALAR (a single pair) or a LIST of the
    PAIRWISE error-corrs among >=3 legs (a 3rd leg has 2 pairwise corrs, one vs each existing leg). Book the WORST
    (max) — a single CORRELATED pair collapses the family's n_eff, so the union is only as decorrelated as its worst
    pair. Returns a scalar; None on empty/None; preserves non-finite so the inf-sibling guards still fire downstream."""
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.asarray(v, float).ravel()
        return float(np.max(arr)) if arr.size else None
    return float(v)

# The video-fake feature space the veto union must SPAN to be coherent-fake-secure (veto_stack_fake_security /
# intersection-null theorem: a union catches a fake iff rank(veto detection dirs)=fake_dim; n_eff of the legs is
# IRRELEVANT). Measured the LIVE union spans only {geometry,motion,deformation,acquisition} ->
# a 2-dim coherent-fake NULL on {appearance,semantic} (D re-verified: joint_null_dim=2, worst-fake=appearance).
FAKE_DIMS = ("geometry", "motion", "deformation", "acquisition", "appearance", "semantic")


# ── SHARED VETO-LEG CONTRACT (serves the KORSREFERENS-KRAV: one API, no per-lane silo) ────────────
# Any lane's video cert (deformation, SfM-rigidity, clip-vs-motion-OED, stream, video-CRB-floor,...) plugs
# into the admission gate by emitting this standard dict — NOT by this module importing the lane's code
# (which would be a cross-worktree soft-import = silently-inert / mechanism-identity risk). The caller runs
# each lane's cert and passes the standard verdicts here:
# {"name": str, "cert_valid": bool, "veto": bool, "reason": str}
# cert_valid: did the leg have enough signal to decide? (False = leg abstained/couldn't run; NOT a veto,
# but reported — a segment whose legs mostly could-not-run is weakly evidenced)
# veto: True = this leg says ABSTAIN (a real failure mode it detected), with `reason`.
# A COMPLEMENTARY leg (different failure mode than geometry-sync) vetoes; the gate unions the vetoes.
def veto_leg(name, cert_valid=True, veto=False, reason="", covers=None, score=None, score_effective=None,
             score_effective_ci_lo=None, veto_threshold=None, coverage_strata=None):
    """Build one standard pluggable veto-leg verdict. Any video-cert lane emits these; the admission gate
    unions them. Keeps composition provenance-clean (no shared load-bearing import between lanes).
    covers: optional list of FAKE_DIMS names this leg's detection SPANS (for the coherent-fake coverage report).
    ★score / score_effective (4-lane §7-posture-v3 fix, consensus): the BINARY `veto` bool discards
      the leg's CONTINUOUS discrimination — measured, a leg whose continuous channel has injection-AUC 0.77-0.85
      collapses to an EFFECTIVE binary magnitude of 0.00-0.38 (semantic/deformation -> 0.0 on subtle fakes). So a
      leg SHOULD also emit `score` = its continuous detection statistic (higher = more fake-like) and
      `score_effective` in [0,1] = the CONTINUOUS-channel calibrated magnitude (e.g. injection-AUC mapped
      2*(AUC-0.5)). The gate then reports EFFECTIVE coverage on the CONTINUOUS channel (not the lossy binary one).
    ★veto_threshold (the rewire, 4-lane consensus): the per-leg CALIBRATED operating point on the
      continuous `score`. When BOTH `score` and `veto_threshold` are present the gate fires a CONTINUOUS veto iff
      score >= veto_threshold — the union DECISION then uses the continuous channel (recovering the 0.77-0.85 AUC),
      not only the lossy binary `veto` bool. ADDITIVE / non-breaking: a leg that emits no veto_threshold keeps the
      exact binary behaviour. The threshold MUST be calibrated at the leg's adversarial-minimum stratum (worst
      confusion mode), not a favourable mean — an easy-only operating point under-vetoes subtle fakes (mirrors the
      family_corr_strata booking); the CALLER owns that calibration and books the conservative point."""
    d = {"name": str(name), "cert_valid": bool(cert_valid), "veto": bool(veto), "reason": str(reason)}
    if covers is not None:
        d["covers"] = [c for c in covers if c in FAKE_DIMS]
    if score is not None:
        d["score"] = float(score)
    if score_effective is not None:
        d["score_effective"] = float(score_effective)
    if score_effective_ci_lo is not None:
        d["score_effective_ci_lo"] = float(score_effective_ci_lo)
    if veto_threshold is not None:
        d["veto_threshold"] = float(veto_threshold)
    if coverage_strata is not None:
        d["coverage_strata"] = {str(k): float(v) for k, v in coverage_strata.items()}
    return d


def _rank(x):
    # ★DEFENSE-IN-DEPTH (NaN-in-rank-fabrication class, D symmetric-QC): argsort ranks a NaN as an
    # ordinary/extreme value, so a rank statistic over non-finite data FABRICATES structure (two signals with NaN at the
    # same positions spuriously concord ~0.94). The cert's upstream '<2 finite non-degenerate methods' guard already makes
    # this UNREACHABLE here (verified: any NaN signal -> ABSTAIN), but fail-fast so a FUTURE caller that forgets the guard
    # gets a clear error, never a fabricated rank (the rule: isfinite-mask/reject before every rank/quantile/Spearman).
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)):
        raise ValueError("_rank received non-finite input; rank statistics fabricate structure from NaN/inf -- the caller "
                         "must finite-guard/abstain before ranking (video_admission_cert does this upstream)")
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float)
    return order


def _pairwise_agreement(a, b):
    """Spearman-style concordance in [-1,1] between two per-frame method signals (flattened per frame)."""
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else 0.0


def _flatten_frames(sig):
    sig = np.asarray(sig, dtype=float)
    return sig.reshape(sig.shape[0], -1) if sig.ndim > 1 else sig.reshape(-1, 1)


def _fake_coverage(veto_legs, temporal):
    """The coherent-fake COVERAGE of the veto stack — SIGNAL-INDEPENDENT (a pure function of veto_legs + the temporal
    convenience, NOT of method_signals). Extracted so the coverage CONTRACT stays LIVE even on the signal-invalid
    early-return path: the veto-leg coverage does not depend on having certifiable signals (C contract-liveness
    sentinel catch,  — my MIN_FRAMES=8 had made coverage go inert for short signals). Returns fake_coverage."""
    t = temporal or {}
    covered = {"geometry"}
    for leg in (veto_legs or []):
        if bool(leg.get("cert_valid", True)):                 # only a leg that actually RAN covers its dims
            covered.update(c for c in (leg.get("covers") or []) if c in FAKE_DIMS)
    if t.get("motion_extrap_reliable_frac") is not None:
        covered.add("motion")
    if "deformation_flagged" in t:
        covered.add("deformation")
    uncovered = [d for d in FAKE_DIMS if d not in covered]
    fc = dict(fake_dim=len(FAKE_DIMS), covered_rank=len(covered), covered_dims=sorted(covered),
              uncovered_dims=uncovered, nominal_secure=(len(uncovered) == 0), basis="declared-nominal",
              note="declared coverage; run effective_detection_probe (injection-AUC) for EFFECTIVE security")
    eff_covered, eff_point_only = set(), set()
    eff_covered_validated, eff_strata_unvalidated = set(), set()
    for leg in (veto_legs or []):
        se = leg.get("score_effective")
        # ★NULL-SAFETY (§4c): skip a NON-FINITE score_effective (NaN = the leg could not compute its magnitude) — else
        # a NaN passes `is not None` and a stale ci_lo would credit coverage the leg never actually established.
        if not bool(leg.get("cert_valid", True)) or se is None or not np.isfinite(se):
            continue
        dims = [c for c in (leg.get("covers") or []) if c in FAKE_DIMS]
        cilo = leg.get("score_effective_ci_lo")
        if cilo is not None and not np.isfinite(cilo):   # a non-finite ci_lo cannot certify coverage -> treat as absent
            cilo = None
        if cilo is not None and EFF_COVER_MAG <= float(cilo) <= MAG_MAX + 1e-9:
            eff_covered.update(dims)
        elif cilo is None and EFF_COVER_MAG <= float(leg["score_effective"]) <= MAG_MAX + 1e-9:
            eff_point_only.update(dims)
        # ★COVERAGE-STRATA guard: a scalar
        # score_effective_ci_lo credits effective_covered above, but 'book on the adversarial minimum' was UNENFORCED
        # — an EASY-ONLY-calibrated ci_lo false-passes the coverage axis exactly like an easy-only family_error_corr
        # false-passed the family axis. A dim is VALIDATED-covered only if the leg supplies a coverage_strata dict that
        # (a) CONTAINS the adversarial_minimum stratum and (b) whose WORST (min) ci_lo still clears EFF_COVER_MAG.
        # Scalar-ci_lo legs stay in effective_covered_dims (backward-compat) but are NOT validated; the band gates
        # AUTO_ADMIT on the validated set, so an unvalidated coverage credit is at most DEMAND, never AUTO_ADMIT.
        strata = leg.get("coverage_strata")
        if strata:
            vals = list(strata.values())
            # ★INF-SIBLING null-safety (agent pool sweep, inf-critic): a NON-FINITE stratum value (inf ci_lo) would pass
            # `>= EFF_COVER_MAG` and falsely VALIDATE (min ignores +inf or worst==+inf clears the bar). A corrupt /
            # degenerate calibration cannot validate coverage — require ALL strata values finite.
            worst = min(vals)
            # ★UPPER-BOUND null-safety (D fail-open fix): a magnitude >1.0 is an impossible AUC (>1.0) and
            # must NOT validate — a corrupt/hand-built supra-unit stratum previously cleared `worst >= EFF_COVER_MAG`
            # and false-VALIDATED -> AUTO_ADMIT (twin of the non-finite guard and of C's §14 helper AUC<=1.0 raise).
            in_range = all(np.isfinite(v) and v <= MAG_MAX + 1e-9 for v in vals)
            if "adversarial_minimum" in strata and in_range and worst >= EFF_COVER_MAG:
                eff_covered_validated.update(dims)
            else:
                eff_strata_unvalidated.update(dims)   # missing adv-min, non-finite/supra-unit value, or failing at worst stratum
    fc["effective_covered_dims"] = sorted(eff_covered)
    fc["effective_point_only_dims"] = sorted(eff_point_only - eff_covered)
    fc["effective_uncovered_dims"] = [d for d in FAKE_DIMS if d not in eff_covered]
    fc["effective_rank"] = len(eff_covered)
    fc["effective_secure"] = (len(eff_covered) == len(FAKE_DIMS))
    # VALIDATED (adversarial-min-booked) coverage — the band gates AUTO_ADMIT on THIS, not the scalar-ci_lo set.
    fc["effective_covered_dims_validated"] = sorted(eff_covered_validated)
    fc["effective_uncovered_dims_validated"] = [d for d in FAKE_DIMS if d not in eff_covered_validated]
    fc["effective_strata_unvalidated_dims"] = sorted(eff_strata_unvalidated - eff_covered_validated)
    return fc


def video_admission_cert(method_signals, temporal=None, veto_legs=None, method_families=None,
                         family_error_corr=None, family_corr_validated=False, family_corr_strata=None,
                         family_leg_min_power=None, family_leg_n=None,
                         sync_thresh=SYNC_THRESH, min_pair_thresh=MIN_PAIR_THRESH, seed=0):
    """Decide ADMIT / ABSTAIN for one video segment.

    method_signals: dict {method_name: array (T) or (T,D)} — a per-frame geometry proxy from each method
                     (e.g. per-frame camera-position, median depth, mesh-vertex centroid). Need >=2 methods.
                     ★COMPARABILITY PRECONDITION (caller's responsibility; the gate CANNOT infer it from the arrays).
                     The sync is a RANK (Spearman) correlation, so it is MONOTONE-INVARIANT: two signals sync iff
                     they are the same underlying quantity UP TO A MONOTONE TRANSFORM, at low estimator noise. NB
                     (D-measured, §9b): a depth STATE vs a looming RATE CAN still sync when the rate is a monotone
                     function of the state (e.g. a constant-velocity approach: rate ∝ state gave D-measured
                     rank-sync 0.97, run scripts/physics_exp/qc_d_redundant_vs_decorr_sync.py) — so "different
                     KIND" is NOT automatically "won't sync". Whether the REAL mono-depth + sfm-depth legs
                     sync on approach is OPEN as of this writing: the reproducer that suggested a decay-to-0
                      was found by to carry an off-by-one, which D's/others' retractions
                     inherited — under re-measurement, do not cite a specific decay number here. The PRINCIPLE
                     that holds regardless: certify comparability against each leg's REAL output — never a
                     comparable-BY-CONSTRUCTION stand-in (which syncs trivially = tautology, the slip D's own
                     §9b "CERT" fell into).
    temporal: optional dict of the complementary Temporal-cert verdicts for this segment:
                       {"motion_extrap_reliable_frac": float in [0,1],   # L motion_extrapolation_abstain_cert
                        "motion_valid": bool,                            # C motion_validity_cert
                        "deformation_flagged": bool}                     # J deformation_cert
                     Missing keys are treated as "leg not run" (no veto, and reported as such).
    veto_legs: optional list of SHARED-contract leg verdicts (see veto_leg) from ANY lane's video cert
                     (SfM-rigidity, clip-vs-motion-OED, stream-cert, video-CRB-floor,...). This is the
                     non-siloed composition API (KORSREFERENS-KRAV): a lane runs its cert and appends
                     veto_leg(name, cert_valid, veto, reason); the gate unions all vetoes. No cross-worktree import.
    method_families: optional dict {method_name: family_label} declaring each method's ALGORITHM family
                     (e.g. {"sfm":"classical-geo", "depth":"learned-mono", "flow":"classical-geo"}). Coherent-fake
                     coverage requires >=2 DISTINCT families — agreement WITHIN one family can be shared common-mode
                     bias, NOT independent confirmation. Decorrelation is a property of the METHODS, not of their
                     output agreement, so it CANNOT be inferred from the signals; it must be declared here. If
                     omitted, the cert cannot vouch for decorrelation and marks coherent_fake_guarded=False.
    Returns dict: verdict, reason, sync_score, min_pair, n_methods, n_families, coherent_fake_guarded, vetoes, cert_valid.
    """
    reasons = []
    # ---- null-safety: enough finite, non-degenerate methods? ----
    names = list(method_signals.keys())
    sigs, good = {}, []
    for n in names:
        # ★ROBUST parse (crash-class, completeness): a 0-frame / malformed / unshapeable signal must ABSTAIN this
        # segment, not CRASH the batch (a real extraction can yield 0 frames at scale). A signal that cannot be
        # flattened is degenerate -> skip it; if <2 usable remain the segment ABSTAINs (cert_invalid) below.
        try:
            s = _flatten_frames(method_signals[n])
        except Exception:
            continue
        if s.ndim != 2 or s.shape[0] == 0:
            continue
        # MIN_FRAMES=8: below this a Spearman rank sync cannot be certified — at T=4 there are only 24 permutations
        # so structureless noise clears ANY threshold ~4% of the time (D self-audit gap 5). Short segments ABSTAIN.
        if s.shape[0] >= MIN_FRAMES and np.all(np.isfinite(s)) and float(s.std()) > DEGEN_STD:
            sigs[n] = s; good.append(n)
    if len(good) < 2:
        # ★coverage stays LIVE even when signals are un-certifiable (C contract-liveness sentinel catch): the veto-leg
        # coverage is signal-INDEPENDENT, so report it (and legs_seen) rather than dropping it to None. cert_valid=False
        # still means don't-admit; a consumer probing the coverage contract can read it regardless of signal quality.
        _ls = [{"name": lg.get("name", "leg"), "cert_valid": bool(lg.get("cert_valid", True)),
                "veto": bool(lg.get("veto", False))} for lg in (veto_legs or [])]
        return dict(verdict="ABSTAIN", reason="cert_invalid: <2 finite non-degenerate methods (null-safe)",
                    sync_score=float("nan"), min_pair=float("nan"), n_methods=len(good),
                    n_families=0, coherent_fake_guarded=False, guard_basis="nominal-declared",
                    effective_guarded=None, vetoes=[], veto_legs_seen=_ls,
                    fake_coverage=_fake_coverage(veto_legs, temporal), cert_valid=False)
    T = min(sigs[n].shape[0] for n in good)
    # coherent-fake guard: distinct ALGORITHM families among the admitted methods (declared metadata, NOT inferred
    # from agreement — agreement cannot tell "correct" from "shared-bias"). >=2 families => decorrelated evidence.
    if method_families:
        n_families = len({method_families.get(n) for n in good if method_families.get(n) is not None})
    else:
        n_families = 0
    # coherent_fake_guarded: NOMINAL by default (>=2 DECLARED families) = necessary-not-sufficient. ★ (D-verified,
    # TIER-A): 2 DECLARED families can SHARE a coherent bias (measured error-corr ~1.0, cert-power ~0) -> the
    # nominal family count FALSE-guards, indistinguishable from genuine decorrelation. EFFECTIVE guard needs a MEASURED
    # per-family-pair error-correlation (a validation signal); mirrors the covers nominal->effective scoping.
    nominal_guarded = n_families >= 2
    # ★N-LEG normalization (§11: guard 2->3-ben): a >=3-leg agreement-cert reports the PAIRWISE error-corrs as a
    # LIST; book the WORST pair (one correlated pair collapses n_eff). Reduces to the scalar 2-leg case unchanged.
    # ★CI-UPPER-BOUND contract (significance_checked_correlation Fisher-z):
    # each family_error_corr value MUST be the CI-UPPER-bound of the pairwise corr, NOT the point. Crediting
    # decorrelation is the FRAGILE direction — at small n a point r=0.30 can have a Fisher-z ci_hi > 0.50 (r=0.45
    # -> ci_hi 0.877), so a point-corr attest would false-credit a not-reliably-decorrelated pair. This is the corr-
    # side mirror of family_leg_min_power (CI-lo); both effective-credit inputs are now CI-robust in the fragile dir.
    family_error_corr = _worst_corr(family_error_corr)
    if family_corr_strata is not None:
        # ★META-MIRAGE guard (D, closes / mirrors O2-V2 at the cert level): a family_error_corr
        # calibrated on an EASY-ONLY set is falsely LOW and false-passes a pair that shares bias only on the HARD
        # (adversarial-minimum) subset. So a STRATA dict {stratum: corr} must (a) CONTAIN the adversarial_minimum
        # stratum — else the calibration is incomplete and is booked NOMINAL (→ DEMAND), never trusted for AUTO_ADMIT —
        # and (b) be booked at the WORST stratum (corr higher = less decorrelated → book the max), not the easy/mean.
        # ★ROBUST cast (crash-class): a NON-NUMERIC stratum value (e.g. a stray string) must reject the calibration,
        # not CRASH admission. A corrupt strata dict is treated as missing adversarial_minimum -> measured-unvalidated.
        try:
            # each stratum value may itself be a LIST of pairwise corrs (>=3 legs) -> book the worst pair per stratum
            strata = {str(k): _worst_corr(v) for k, v in family_corr_strata.items()}
            strata = {k: v for k, v in strata.items() if v is not None}
        except (TypeError, ValueError):
            strata = {}
        # ★INF-SIBLING null-safety (agent pool sweep, inf-critic + adversarial agent): a NON-FINITE stratum falsely
        # books 'effective-measured' — a -inf value passes `< FAMILY_DECORR_THRESH` (spurious decorrelation) and a
        # NaN is SILENTLY DROPPED by max whenever it isn't the leading dict value (this codebase writes easy-first).
        # Require adversarial_minimum present AND all values finite; else the calibration is corrupt → NOMINAL (DEMAND).
        if "adversarial_minimum" not in strata or not all(np.isfinite(v) for v in strata.values()):
            effective_guarded = None
            coherent_fake_guarded = nominal_guarded
            guard_basis = "measured-unvalidated"       # missing adv-min OR non-finite stratum → not trustworthy
        else:
            family_error_corr = max(strata.values())   # worst-stratum booking (conservative)
            effective_guarded = bool(family_error_corr < FAMILY_DECORR_THRESH)
            coherent_fake_guarded = effective_guarded
            guard_basis = "effective-measured"
    elif family_error_corr is not None and family_corr_validated and np.isfinite(family_error_corr):
        # ★★NAMED CALLER-PRECONDITION (grade of D-sweep, D-verified): `family_corr_validated` is a DELEGATED
        # precondition the cert cannot self-verify -- it MUST be the `.validated` OUTPUT of a real family-corr
        # validation protocol (e.g. The family_corr_validation_check V1-V5, which rejects synthetic_only), NEVER a
        # caller-asserted literal True. A bare True on a SYNTHETIC/unvalidated corr unlocks this HARD effective-measured
        # path off an UNVERIFIABLE declaration = the identity!=decorrelation class on the VALIDATION axis (a 'validated'
        # assertion is necessary-not-sufficient). The cert trusts the flag; the CALLER owns discharging it (like the
        # type-3 total-shared-bias anchor delegation). hardens its own consumer to derive; other callers MUST too.
        # ★INF-SIBLING null-safety (completeness matrix): a NON-FINITE scalar corr must not book 'effective-measured'.
        # family_error_corr=-inf passes `< FAMILY_DECORR_THRESH` -> spurious decorrelation (fail-open); NaN/+inf are
        # safe-by-polarity but still not a real measurement -> require finite, else fall through to NOMINAL below.
        effective_guarded = bool(float(family_error_corr) < FAMILY_DECORR_THRESH)
        coherent_fake_guarded = effective_guarded
        guard_basis = "effective-measured"
    elif family_error_corr is not None:
        # ★MEASURED-but-UNVALIDATED corr (e.g. synthetic/offline-only, real-data gap undischarged): it must NOT drive
        # the HARD effective path (AUTO_ADMIT or QUARANTINE) on real data it was not validated on — BOTH directions
        # inherit the synthetic→real gap. Book as NOMINAL (→ DEMAND) until validated on the deployment domain.
        # (D catch of the synthetic-calibrated family_error_corr row: a synthetic 0.563 booked as-is would
        # else hard-QUARANTINE every real ×segment on a measurement whose real-data validity is undischarged.)
        effective_guarded = None
        coherent_fake_guarded = nominal_guarded
        guard_basis = "measured-unvalidated"
    else:
        effective_guarded = None
        coherent_fake_guarded = nominal_guarded            # DECLARED families only — necessary-not-sufficient (H79)
        guard_basis = "nominal-declared"

    # ★BLIND-LEG guard (decorrelation necessary-not-sufficient; appearance×geometry unlock): a LOW
    # family_error_corr only MEANS decorrelation if BOTH legs are NON-BLIND on the corr-calibration set. A leg reading
    # ~0 error-corr because it is BLIND to the fake (its own detection ~chance — e.g. a geometry leg on a pure-RGB
    # appearance fake, blind by construction) is the n_eff=1 trap, NOT decorrelation. If the caller attests the MINIMUM
    # per-leg detection power and it is blind, the effective-measured decorrelation credit is downgraded to NOMINAL.
    # ★CI-LOWER-BOUND contract (fresh-eyes finding):
    # family_leg_min_power MUST be the CI-LOWER-BOUND of the min per-leg magnitude 2*(AUC-0.5), NOT the point estimate.
    # Crediting decorrelation is the FRAGILE direction — at small n a point-AUC just above the bar can have a CI-lo
    # below it (winner's curse), so a point-magnitude attest would false-credit a not-reliably-non-blind leg. Gating on
    # the ci_lo mirrors the coverage axis (score_effective_ci_lo) and makes both effective-credit paths CI-robust.
    # ★n-AWARE ENFORCEMENT (D self-audit sweep, significance_checked_min_power flagged this by name):
    # the CI-lo contract above was DOCUMENTED but not ENFORCED — a caller passing the POINT magnitude sailed through
    # at a sample size where the leg is chance (confirmed: AUC 0.55 @ n=489 -> point mag 0.10 PASSES 0.10 bar, but
    # Hanley-McNeil CI-lo=0.4991<0.5). If family_leg_n=(n_pos,n_neg) is supplied, recompute the min_power as the
    # CI-LOWER-BOUND magnitude from the implied AUC + n, so a point attest can no longer false-credit. Absent n, behave
    # as before (contract stays a documented caller-obligation).
    if family_leg_min_power is not None and family_leg_n is not None and np.isfinite(family_leg_min_power):
        try:
            npos, nneg = (int(family_leg_n[0]), int(family_leg_n[1])) if hasattr(family_leg_n, "__len__") \
                else (int(family_leg_n), int(family_leg_n))
            # ★NEGATIVE-n fabrication (D behavioral fuzz): npos*nneg with BOTH negative silently
            # double-negates to a positive denominator (a physically impossible sample size sails through the
            # arithmetic unnoticed), computing a finite, PASSING se/family_leg_min_power instead of raising --
            # a nonsensical n would otherwise get FULL 'effective-measured' decorrelation credit. Reject non-
            # positive n explicitly so it falls into the same except-branch as other unusable-n cases below.
            if npos <= 0 or nneg <= 0:
                raise ValueError(f"family_leg_n must be positive (got npos={npos}, nneg={nneg})")
            auc = 0.5 + float(family_leg_min_power) / 2.0     # power magnitude -> implied point AUC
            q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
            se = np.sqrt(max((auc * (1 - auc) + (npos - 1) * (q1 - auc * auc) + (nneg - 1) * (q2 - auc * auc)) / (npos * nneg), 0.0))
            family_leg_min_power = max(0.0, 2.0 * ((auc - 1.96 * se) - 0.5))   # CI-lo magnitude (enforced)
        except (TypeError, ValueError, ZeroDivisionError):
            family_leg_min_power = float("nan")               # unusable n -> treat power as unverifiable (blind, below)
    if guard_basis == "effective-measured" and family_leg_min_power is not None \
            and (not np.isfinite(family_leg_min_power) or float(family_leg_min_power) < BLIND_POWER_THRESH):
        # ★NON-FINITE power (fail-open, completeness): a NaN/+inf attest bypasses `< BLIND_POWER_THRESH` (nan<x is
        # False) and would silently CREDIT decorrelation — a power that can't be verified is treated as BLIND, not seen.
        effective_guarded = None
        coherent_fake_guarded = nominal_guarded
        guard_basis = "measured-blind-leg"                 # blind OR unverifiable leg power -> not orthogonality

    # ---- cross-method agreement (REDUNDANT legs) -> safe fuse ----
    pairs = []
    for i in range(len(good)):
        for j in range(i + 1, len(good)):
            a, b = sigs[good[i]][:T], sigs[good[j]][:T]
            # per-column agreement averaged over the (few) geometry columns
            ags = [_pairwise_agreement(a[:, c], b[:, c]) for c in range(min(a.shape[1], b.shape[1]))]
            pairs.append(float(np.mean(ags)))
    pairs = np.array(pairs)
    sync_score = float(np.median(pairs))          # robust central agreement
    min_pair = float(pairs.min())                 # weakest link (safe-fuse: one bad method must drag it down)

    # ★n-AWARE thresholds (D self-audit gap 5): a rank correlation of sync_thresh is NOT statistically significant at
    # small T — pure independent noise clears 0.60 ~5-20% of the time at T=4 (finite-permutation variance of a Spearman
    # rho under the null is ~1/sqrt(T-1)). Raise the effective bars by a ~2.5-sigma small-n term so structureless input
    # ABSTAINs (honoring the "never a false ADMIT on structureless input" invariant); at large T the term is <= the
    # nominal thresholds (inert, no regression), and a genuinely-synced short segment (sync~0.9) still clears it.
    small_n = 2.5 / np.sqrt(max(T - 1, 1))
    _SYNC_CAP = 0.90            # cap the small-n bar BELOW 1.0 so a genuine near-perfect rank sync is still admissible
    eff_sync_thresh = min(_SYNC_CAP, max(sync_thresh, small_n))
    eff_min_pair_thresh = min(_SYNC_CAP, max(min_pair_thresh, small_n))
    if min_pair < eff_min_pair_thresh:
        reasons.append(f"method_disagreement: a method-pair agrees only {min_pair:.2f}<{eff_min_pair_thresh:.2f} (weakest link, n-aware)")
    elif sync_score < eff_sync_thresh:
        reasons.append(f"low_sync: median cross-method agreement {sync_score:.2f}<{eff_sync_thresh:.2f} (n-aware; small-n bar raised)")

    # ---- COMPLEMENTARY veto legs -> per-mode VETO with named reason ----
    vetoes = []
    # (a) `temporal=` convenience: the known L/C/J legs, mapped to the standard contract (backward-compat).
    t = temporal or {}
    # ★NaN/non-finite FAIL-CLOSED across ALL THREE temporal legs (D self-census): a FAILED/degenerate
    # upstream temporal cert returns a non-finite / non-bool verdict; the old `NaN < thresh` (False) and `is False`/
    # `is True` checks silently NO-veto'd it = ADMIT (unassessable motion treated as benign). An UNASSESSABLE leg must
    # VETO (fail-closed) -- matching the veto_legs NaN null-safety (sibling consistency; nan-fail-open class).
    if "motion_extrap_reliable_frac" in t and t["motion_extrap_reliable_frac"] is not None:
        _erf = float(t["motion_extrap_reliable_frac"])
        if not np.isfinite(_erf):
            vetoes.append("unpredictable_motion: extrap-reliable-frac non-finite (unassessable, fail-closed)")
        elif _erf < EXTRAP_RELIABLE_FRAC:
            vetoes.append(f"unpredictable_motion: extrap-reliable-frac {_erf:.2f}<{EXTRAP_RELIABLE_FRAC}")
    _mv = t.get("motion_valid")
    if _mv is not None:
        if isinstance(_mv, (bool, np.bool_)):
            if not bool(_mv):
                vetoes.append("invalid_motion: motion_validity_cert=False")
        else:
            vetoes.append("invalid_motion: motion_valid non-bool (unassessable, fail-closed)")
    _df = t.get("deformation_flagged")
    if _df is not None:
        if isinstance(_df, (bool, np.bool_)):
            if bool(_df):
                vetoes.append("non_rigid: deformation_cert flagged")
        else:
            vetoes.append("non_rigid: deformation_flagged non-bool (unassessable, fail-closed)")
    # (b) SHARED pluggable veto legs: any lane's video cert emitted via veto_leg (the non-siloed API).
    legs_seen = []
    for leg in (veto_legs or []):
        nm = leg.get("name", "leg"); cv = bool(leg.get("cert_valid", True)); vt = bool(leg.get("veto", False))
        # ★CONTINUOUS veto (-REWIRE): if the leg emits a continuous `score` AND a calibrated `veto_threshold`, the
        # union DECISION re-thresholds the continuous channel (score >= threshold => fake-like => veto), recovering
        # the discrimination the binary bool discards. ADDITIVE: legs without both fields fall through to binary only.
        sc = leg.get("score"); th = leg.get("veto_threshold")
        # ★NULL-SAFETY (§4c audit, D): a leg that ASKS for a continuous veto (both score AND threshold
        # present) but whose operating point is NON-FINITE (NaN score = detector failed to compute; NaN/inf threshold
        # = mis-calibration) must NOT silently no-veto — `nan >= thr` is False, which would fail OPEN (silent ADMIT).
        # Fail CLOSED instead: a degenerate continuous-veto leg vetoes.
        cont_degen = (sc is not None and th is not None and not (np.isfinite(sc) and np.isfinite(th)))
        cont_veto = cont_degen or (sc is not None and th is not None
                                   and np.isfinite(sc) and np.isfinite(th) and float(sc) >= float(th))
        fired = cv and (vt or cont_veto)
        legs_seen.append({"name": nm, "cert_valid": cv, "veto": vt,
                          "continuous_veto": bool(cv and cont_veto), "score": sc, "veto_threshold": th})
        if fired:
            if vt:
                vetoes.append(f"{nm}: {leg.get('reason', 'veto')}")
            elif cont_degen:
                vetoes.append(f"{nm}: degenerate continuous-veto operating point (non-finite score/threshold) -> fail-closed veto")
            else:   # continuous-only trigger names the operating point for auditability
                vetoes.append(f"{nm}: continuous score {float(sc):.3f}>=thr {float(th):.3f} (calibrated per-leg veto)")
    reasons += vetoes

    # coherent-fake COVERAGE (extracted to _fake_coverage — signal-independent; also computed on the early-return so
    # the coverage contract stays live for short/invalid signals). Does the RUNNING veto stack SPAN the fake-feature
    # space? Uncovered dims = the coherent-fake NULL (a fake living ONLY there passes the union; rank secures, not n_eff).
    fake_coverage = _fake_coverage(veto_legs, temporal)

    verdict = "ADMIT" if not reasons else "ABSTAIN"
    if verdict == "ADMIT":
        reason = "synced + temporally valid (SYNC not correctness — see scope)"
        if not coherent_fake_guarded:
            reason += (f"; ⚠coherent-fake UNGUARDED ({guard_basis}): n_families={n_families}<2 — agreement may be "
                       "shared-bias, declare method_families or add a decorrelated method")
        elif guard_basis == "nominal-declared":
            reason += ("; ⚠guard is NOMINAL (declared families only) — 2 declared families can SHARE a coherent bias "
                       "(H79); pass family_error_corr (measured) for an EFFECTIVE guard")
        elif guard_basis == "measured-unvalidated":
            reason += (f"; ⚠family_error_corr={family_error_corr} provided but UNVALIDATED on the deployment domain "
                       "(synthetic/offline-only) — booked as NOMINAL (→DEMAND); pass family_corr_validated=True only "
                       "after discharging the calibration→deployment gap")
        if fake_coverage["uncovered_dims"]:
            reason += (f"; ⚠veto-coverage(nominal) {fake_coverage['covered_rank']}/{len(FAKE_DIMS)}, UNCOVERED "
                       f"{fake_coverage['uncovered_dims']} — a coherent fake living only in these dims passes the "
                       "union; add a veto covering them (effective: probe)")
    else:
        reason = "; ".join(reasons)
    return dict(verdict=verdict, reason=reason, sync_score=sync_score, min_pair=min_pair,
                n_methods=len(good), n_families=n_families, coherent_fake_guarded=coherent_fake_guarded,
                guard_basis=guard_basis, effective_guarded=effective_guarded,
                family_error_corr=family_error_corr, family_corr_validated=bool(family_corr_validated),
                family_corr_strata=family_corr_strata,
                vetoes=vetoes, veto_legs_seen=legs_seen, fake_coverage=fake_coverage, cert_valid=True)


def coherent_fake_band(cert):
    """CANONICAL converged coherent-fake admission band from a video_admission_cert result.
    Maps to {AUTO_ADMIT, DEMAND, QUARANTINE}. This is the single source of truth for the v1-v4 arc (D found the
    band_admission coherent-fake leak; refined it): band_admission.AUTO_ADMIT should gate on this instead of
    on the raw 'cert ADMIT' verdict. AUTO_ADMIT only when BOTH axes are EFFECTIVE:
      family axis: guarded=False -> QUARANTINE; guard_basis nominal-declared -> DEMAND; effective-measured -> ok
                      (v4: coherent_fake_guarded/guard_basis, -hardened + family_error_corr effective path)
      coverage axis: any NON-geometry dim not in effective_covered_dims_VALIDATED -> DEMAND (geometry gated by SYNC,
                      excluded = v3; CI-reliable = v4; adversarial-min-VALIDATED = v5, -flagged meta-mirage: a
                      scalar-ci_lo coverage credit is NOT enough for AUTO_ADMIT, mirrors family_corr_strata). Missing
                      coverage -> conservative DEMAND.
    Returns the WORST (most conservative) of the two axes. cert_valid=False -> QUARANTINE."""
    if not cert.get("cert_valid", True):
        return "QUARANTINE"
    # ★SYNC axis (D self-audit / FP-hunter): the band composes THREE necessary conditions, not two — the
    # cert's own verdict (synced + temporally valid) is the first. A cert that did NOT ADMIT must NOT reach AUTO_ADMIT
    # however good the family/coverage axes look (callers are told to gate on this band, so it must carry sync). But
    # distinguish WHY it didn't admit: a fired VETO (detected non-rigidity / unpredictable motion / etc.) is a real
    # refusal -> QUARANTINE; a mere low-sync / can't-certify ABSTAIN (no veto — e.g. a genuine moderate sync at small T
    # that is statistically indistinguishable from noise) is NOT a detected fake -> DEMAND (review), not a hard reject.
    if cert.get("verdict") != "ADMIT":
        return "QUARANTINE" if cert.get("vetoes") else "DEMAND"
    order = {"QUARANTINE": 0, "DEMAND": 1, "AUTO_ADMIT": 2}
    if not cert.get("coherent_fake_guarded"):
        fam = "QUARANTINE"
    elif cert.get("guard_basis") == "effective-measured":
        fam = "AUTO_ADMIT"
    else:
        fam = "DEMAND"
    fc = cert.get("fake_coverage") or {}
    # ★v5 (meta-mirage): AUTO_ADMIT on the coverage axis requires ADVERSARIAL-MIN-VALIDATED coverage of every
    # non-geometry dim (effective_covered_dims_validated), not a scalar-ci_lo credit. A cert lacking the validated
    # field (older cert / no coverage_strata supplied) -> default ALL uncovered -> DEMAND (conservative, safe).
    unc = [d for d in fc.get("effective_uncovered_dims_validated", list(FAKE_DIMS)) if d != "geometry"]
    cov = "AUTO_ADMIT" if not unc else "DEMAND"
    return min([fam, cov], key=lambda b: order[b])


def _selftest():
    rng = np.random.default_rng(0); T = 60; ok = True
    # a shared latent "true geometry" trajectory
    truth = np.cumsum(rng.standard_normal((T, 3)), axis=0)

    def noisy(scale):  # a method that tracks truth with noise
        return truth + scale * rng.standard_normal((T, 3))

    cases = []
    # 1) clean synced rigid segment, temporal OK, 2 declared families -> ADMIT + coherent_fake_guarded
    r = video_admission_cert({"sfm": noisy(0.05), "depth": noisy(0.05), "flow": noisy(0.05)},
                             temporal={"motion_extrap_reliable_frac": 0.9, "motion_valid": True, "deformation_flagged": False},
                             method_families={"sfm": "classical-geo", "depth": "learned-mono", "flow": "classical-geo"})
    cases.append(("clean synced (2 families)", r["verdict"] == "ADMIT" and r["coherent_fake_guarded"] is True, r))
    # 2) one method is garbage (decorrelated) -> weakest-link -> ABSTAIN method_disagreement
    r = video_admission_cert({"sfm": noisy(0.05), "depth": noisy(0.05), "flow": rng.standard_normal((T, 3))},
                             temporal={"motion_valid": True})
    cases.append(("one garbage method", r["verdict"] == "ABSTAIN" and "disagreement" in r["reason"], r))
    # 3) deformation flagged -> ABSTAIN non_rigid (even though synced)
    r = video_admission_cert({"sfm": noisy(0.05), "depth": noisy(0.05)},
                             temporal={"deformation_flagged": True, "motion_valid": True})
    cases.append(("deformation veto", r["verdict"] == "ABSTAIN" and "non_rigid" in r["reason"], r))
    # 4) unpredictable motion (low extrap frac) -> ABSTAIN unpredictable_motion
    r = video_admission_cert({"sfm": noisy(0.05), "depth": noisy(0.05)},
                             temporal={"motion_extrap_reliable_frac": 0.2, "motion_valid": True})
    cases.append(("unpredictable motion veto", r["verdict"] == "ABSTAIN" and "unpredictable" in r["reason"], r))
    # 5) structureless: all methods independent noise -> ABSTAIN (won't sync) [null-control]
    r = video_admission_cert({"a": rng.standard_normal((T, 3)), "b": rng.standard_normal((T, 3)),
                              "c": rng.standard_normal((T, 3))})
    cases.append(("structureless null-control", r["verdict"] == "ABSTAIN", r))
    # 6) non-finite input -> ABSTAIN cert_invalid [null-safe]
    bad = noisy(0.05); bad[3] = np.nan
    r = video_admission_cert({"sfm": bad, "depth": noisy(0.05)})
    cases.append(("non-finite null-safe", r["verdict"] == "ABSTAIN" and r["cert_valid"] is False, r))
    # 7) <2 methods -> ABSTAIN cert_invalid
    r = video_admission_cert({"only": noisy(0.05)})
    cases.append(("single method", r["verdict"] == "ABSTAIN" and r["cert_valid"] is False, r))
    # 8) coherent-fake: near-DUPLICATE methods agree (shared bias), NO families declared -> ADMIT but
    # coherent_fake_guarded=False + caveat in reason (agreement != correctness; the honest floor)
    base = noisy(0.05)
    r = video_admission_cert({"m1": base, "m2": base + 0.001 * rng.standard_normal((T, 3)),
                              "m3": base + 0.001 * rng.standard_normal((T, 3))},
                             temporal={"motion_valid": True})
    cases.append(("coherent-fake unguarded flag", r["verdict"] == "ADMIT" and r["coherent_fake_guarded"] is False
                  and "UNGUARDED" in r["reason"], r))
    # 9) SHARED pluggable veto-leg API (KORSREFERENS-KRAV): a lane's cert (e.g. SfM-rigidity) vetoes via
    # veto_leg even though geometry syncs -> ABSTAIN with that leg's reason (non-siloed composition; a
    # non-vetoing leg is reported but does not block).
    r = video_admission_cert({"sfm": noisy(0.05), "depth": noisy(0.05)},
                             method_families={"sfm": "classical-geo", "depth": "learned-mono"},
                             veto_legs=[veto_leg("sfm_rigidity", cert_valid=True, veto=True, reason="scene non-rigid over window"),
                                        veto_leg("clip_vs_motion_oed", cert_valid=True, veto=False, reason="")])
    cases.append(("pluggable veto-leg API", r["verdict"] == "ABSTAIN" and "sfm_rigidity" in r["reason"]
                  and len(r["veto_legs_seen"]) == 2, r))
    # 10) coherent-fake COVERAGE (intersection-null): a synced segment whose running legs cover only
    # {geometry,motion,deformation,acquisition} -> ADMIT but reports the 2-dim {appearance,semantic} NULL.
    r = video_admission_cert({"sfm": noisy(0.05), "depth": noisy(0.05)},
                             method_families={"sfm": "classical-geo", "depth": "learned-mono"},
                             veto_legs=[veto_leg("deformation", veto=False, covers=["deformation"]),
                                        veto_leg("rolling_shutter", veto=False, covers=["acquisition"]),
                                        veto_leg("motion_extrap", veto=False, covers=["motion"])])
    cases.append(("coherent-fake coverage null", r["verdict"] == "ADMIT"
                  and r["fake_coverage"]["uncovered_dims"] == ["appearance", "semantic"]
                  and "UNCOVERED" in r["reason"], r))
    # 11) full coverage: legs span all 6 fake dims -> fake_secure=True, no coverage caveat
    r = video_admission_cert({"sfm": noisy(0.05), "depth": noisy(0.05)},
                             method_families={"sfm": "classical-geo", "depth": "learned-mono"},
                             veto_legs=[veto_leg("motion", veto=False, covers=["motion"]),
                                        veto_leg("deform", veto=False, covers=["deformation"]),
                                        veto_leg("acq", veto=False, covers=["acquisition"]),
                                        veto_leg("appear", veto=False, covers=["appearance"]),
                                        veto_leg("sem", veto=False, covers=["semantic"])])
    cases.append(("full DECLARED coverage -> nominal_secure", r["verdict"] == "ADMIT"
                  and r["fake_coverage"]["nominal_secure"] is True and r["fake_coverage"]["basis"] == "declared-nominal", r))
    # 12) fix: EFFECTIVE family-decorrelation guard. 2 declared families but a MEASURED shared bias (high error-corr)
    # -> coherent_fake_guarded=False (effective); a measured-decorrelated pair -> True. Nominal count alone can't tell.
    fams2 = {"sfm": "classical-geo", "depth": "learned-mono"}
    r_share = video_admission_cert({"sfm": noisy(0.05), "depth": noisy(0.05)}, method_families=fams2, family_error_corr=0.99, family_corr_validated=True)
    r_dec = video_admission_cert({"sfm": noisy(0.05), "depth": noisy(0.05)}, method_families=fams2, family_error_corr=0.05, family_corr_validated=True)
    r_nom = video_admission_cert({"sfm": noisy(0.05), "depth": noisy(0.05)}, method_families=fams2)
    pass12 = (r_share["coherent_fake_guarded"] is False and r_share["guard_basis"] == "effective-measured"
              and r_dec["coherent_fake_guarded"] is True and r_dec["guard_basis"] == "effective-measured"
              and r_nom["coherent_fake_guarded"] is True and r_nom["guard_basis"] == "nominal-declared")
    cases.append(("H79 effective family-guard", pass12, r_share))
    # 13) CI-ROBUST effective coverage (batch winner's-curse fix): CI lower bound >= bar -> RELIABLE covered;
    # point >= bar but NO CI -> point_only (flagged, not reliable); below bar -> uncovered.
    r13 = video_admission_cert({"sfm": noisy(0.05), "depth": noisy(0.05)}, method_families=fams2,
                               veto_legs=[veto_leg("semantic", covers=["semantic"], score_effective=0.80, score_effective_ci_lo=0.70),
                                          veto_leg("appearance", covers=["appearance"], score_effective=0.80),  # point-only, no CI
                                          veto_leg("deformation", covers=["deformation"], score_effective=0.30)])
    fc13 = r13["fake_coverage"]
    pass13 = ("semantic" in fc13["effective_covered_dims"]                        # CI lower bound clears -> reliable
              and "appearance" in fc13["effective_point_only_dims"]              # point clears but no CI -> flagged
              and "appearance" not in fc13["effective_covered_dims"]
              and "deformation" in fc13["effective_uncovered_dims"]             # below bar
              and fc13["effective_rank"] == 1)
    cases.append(("CI-robust effective coverage", pass13, r13))
    # 14) CANONICAL coherent_fake_band (v1-v4): single-family fake -> QUARANTINE; family-guarded but a non-geo dim
    # uncovered -> DEMAND; family-effective-guarded + all non-geo dims ADVERSARIAL-MIN-VALIDATED-covered -> AUTO_ADMIT.
    # (v5: covlegs now supply coverage_strata with an adversarial_minimum clearing the bar — a scalar ci_lo no
    # longer suffices for AUTO_ADMIT, per the meta-mirage; see case 17.)
    covlegs = [veto_leg(d, covers=[d], score_effective=0.9, score_effective_ci_lo=0.8,
                        coverage_strata={"easy": 0.9, "adversarial_minimum": 0.75})
               for d in ("motion", "deformation", "acquisition", "appearance", "semantic")]
    c_q = video_admission_cert({"m0": base + .001 * rng.standard_normal((T, 3)), "m1": base + .001 * rng.standard_normal((T, 3))},
                               method_families={"m0": "same", "m1": "same"})                       # single-family fake
    c_d = video_admission_cert({"m0": base + .05 * rng.standard_normal((T, 3)), "m1": base + .05 * rng.standard_normal((T, 3))},
                               method_families={"m0": "classical", "m1": "learned"}, family_error_corr=0.10, family_corr_validated=True)  # guarded, no coverage
    c_a = video_admission_cert({"m0": base + .05 * rng.standard_normal((T, 3)), "m1": base + .05 * rng.standard_normal((T, 3))},
                               method_families={"m0": "classical", "m1": "learned"}, family_error_corr=0.10, family_corr_validated=True, veto_legs=covlegs)
    pass14 = (coherent_fake_band(c_q) == "QUARANTINE" and coherent_fake_band(c_d) == "DEMAND"
              and coherent_fake_band(c_a) == "AUTO_ADMIT")
    cases.append(("canonical coherent_fake_band", pass14, c_q))
    # 15) VALIDATION-DOMAIN gate (D catch of the synthetic-calibrated family_error_corr): a MEASURED corr
    # that is UNVALIDATED on the deployment domain must NOT drive the hard effective path. The synthetic 0.563
    # booked as-is -> QUARANTINE (would hard-gate every real segment on undischarged synthetic→real); the SAME
    # value unvalidated must fall back to nominal -> DEMAND (honest: unknown at scale, review). Validated -> effective.
    ms15 = {"sfm": noisy(0.05), "depth": noisy(0.05)}
    c_unval = video_admission_cert(ms15, method_families=fams2, family_error_corr=0.563)                        # no validated flag
    c_val = video_admission_cert(ms15, method_families=fams2, family_error_corr=0.563, family_corr_validated=True)
    pass15 = (c_unval["guard_basis"] == "measured-unvalidated" and coherent_fake_band(c_unval) == "DEMAND"
              and c_val["guard_basis"] == "effective-measured" and coherent_fake_band(c_val) == "QUARANTINE")
    cases.append(("validation-domain gate (synthetic corr -> nominal)", pass15, c_unval))
    # 16) META-MIRAGE guard (D, closes): a family_error_corr STRATA dict must contain the
    # adversarial_minimum stratum and is booked at the WORST stratum. Easy-only calibration (0.10) that hides a
    # shared-bias-on-the-hard-subset (0.75) must NOT AUTO_ADMIT.
    ms16 = {"sfm": noisy(0.05), "depth": noisy(0.05)}
    # (a) easy-low but hard-high -> book worst 0.75 -> not decorrelated -> NOT auto-admit
    c_meta = video_admission_cert(ms16, method_families=fams2, family_corr_strata={"easy": 0.10, "adversarial_minimum": 0.75})
    # (b) strata missing the adversarial_minimum -> incomplete calibration -> nominal (measured-unvalidated)
    c_incomplete = video_admission_cert(ms16, method_families=fams2, family_corr_strata={"easy": 0.10})
    # (c) genuinely decorrelated on the hard subset too -> worst 0.20 < 0.50 -> effective-guarded
    c_ok = video_admission_cert(ms16, method_families=fams2, family_corr_strata={"easy": 0.10, "adversarial_minimum": 0.20})
    pass16 = (coherent_fake_band(c_meta) != "AUTO_ADMIT" and c_meta["family_error_corr"] == 0.75
              and c_incomplete["guard_basis"] == "measured-unvalidated"
              and c_ok["guard_basis"] == "effective-measured" and c_ok["coherent_fake_guarded"] is True)
    cases.append(("meta-mirage strata guard (worst-stratum + adv-min required)", pass16, c_meta))

    # 17) ★COVERAGE meta-mirage closure (-flagged, the twin of case 16 on the coverage axis): the band must NOT grant
    # AUTO_ADMIT on an UNVALIDATED coverage credit. Same synced base + effective family guard; vary only the
    # coverage EVIDENCE for the 5 non-geo dims. Scalar ci_lo stays credited in effective_covered_dims (backward-
    # compat) but only an adversarial-min-validated coverage_strata reaches AUTO_ADMIT.
    fam_ok = {"m0": "classical", "m1": "learned"}
    def _cov_cert(mk):
        return video_admission_cert({"m0": base + .05 * rng.standard_normal((T, 3)), "m1": base + .05 * rng.standard_normal((T, 3))},
                                    method_families=fam_ok, family_error_corr=0.10, family_corr_validated=True,
                                    veto_legs=[mk(d) for d in ("motion", "deformation", "acquisition", "appearance", "semantic")])
    c_scalar = _cov_cert(lambda d: veto_leg(d, covers=[d], score_effective=0.9, score_effective_ci_lo=0.8))                       # (a) scalar only -> DEMAND
    c_noadv = _cov_cert(lambda d: veto_leg(d, covers=[d], score_effective=0.9, score_effective_ci_lo=0.8, coverage_strata={"easy": 0.9, "medium": 0.85}))  # (b) no adv-min -> DEMAND
    c_hardlow = _cov_cert(lambda d: veto_leg(d, covers=[d], score_effective=0.9, score_effective_ci_lo=0.8, coverage_strata={"easy": 0.95, "adversarial_minimum": 0.40}))  # (c) adv-min FAILS bar -> DEMAND
    c_valid = _cov_cert(lambda d: veto_leg(d, covers=[d], score_effective=0.9, score_effective_ci_lo=0.8, coverage_strata={"easy": 0.95, "adversarial_minimum": 0.72}))    # (d) adv-min clears -> AUTO_ADMIT
    pass17c = (coherent_fake_band(c_scalar) == "DEMAND" and coherent_fake_band(c_noadv) == "DEMAND"
               and coherent_fake_band(c_hardlow) == "DEMAND" and coherent_fake_band(c_valid) == "AUTO_ADMIT"
               and "semantic" in c_scalar["fake_coverage"]["effective_covered_dims"]                     # scalar still credited (non-breaking)
               and "semantic" not in c_hardlow["fake_coverage"]["effective_covered_dims_validated"])      # but the mirage is NOT validated
    cases.append(("coverage meta-mirage closure (adv-min-validated for AUTO_ADMIT)", pass17c, c_hardlow))

    # 18) ★-REWIRE recovery MEASUREMENT: subtle fakes where the BINARY veto bool is silent (veto=False) but the leg's
    # CONTINUOUS score IS discriminative (real vs fake AUC ~0.83). The binary union admits both (blind, decision
    # AUC 0.5, 0 fakes caught); the continuous union re-thresholds score at the calibrated point and RECOVERS the
    # discrimination. Measures the gate DECISION quality, not a narrative — the whole point of the 4-lane rewire.
    def _auc(labels, preds):                       # tie-aware Mann-Whitney AUC (discrete preds -> ties matter)
        pos = [p for l, p in zip(labels, preds) if l]; neg = [p for l, p in zip(labels, preds) if not l]
        if not pos or not neg:
            return float("nan")
        s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return s / (len(pos) * len(neg))
    rng17 = np.random.default_rng(17); N = 200; THR = 0.5
    base = np.cumsum(rng17.standard_normal((T, 3)), axis=0)
    def _seg():                                    # a sync-passing 2-method segment (leg governs the verdict)
        return {"m1": base + 0.05 * rng17.standard_normal((T, 3)), "m2": base + 0.05 * rng17.standard_normal((T, 3))}
    labels, vet_bin, vet_cont, raw = [], [], [], []
    rep_cont = None
    for k in range(2 * N):
        fake = k >= N
        labels.append(fake)
        sc = float(np.clip(rng17.normal(0.65 if fake else 0.35, 0.22), 0, 1))   # continuous detector, binary bool SILENT
        raw.append(sc)
        leg_bin = veto_leg("subtle_det", veto=False, score=sc)                  # binary-only: no threshold
        leg_ct = veto_leg("subtle_det", veto=False, score=sc, veto_threshold=THR)
        rb = video_admission_cert(_seg(), temporal={"motion_valid": True}, veto_legs=[leg_bin])
        rc = video_admission_cert(_seg(), temporal={"motion_valid": True}, veto_legs=[leg_ct])
        vet_bin.append(rb["verdict"] == "ABSTAIN")
        vet_cont.append(rc["verdict"] == "ABSTAIN")
        if fake and rep_cont is None and rc["verdict"] == "ABSTAIN":
            rep_cont = rc
    bin_auc = _auc(labels, [1.0 if v else 0.0 for v in vet_bin])
    cont_auc = _auc(labels, [1.0 if v else 0.0 for v in vet_cont])
    raw_auc = _auc(labels, raw)
    fakes_caught_bin = sum(v for v, l in zip(vet_bin, labels) if l) / N
    fakes_caught_cont = sum(v for v, l in zip(vet_cont, labels) if l) / N
    reals_vetoed_cont = sum(v for v, l in zip(vet_cont, labels) if not l) / N
    pass17 = (bin_auc == 0.5 and fakes_caught_bin == 0.0        # binary union is BLIND to subtle fakes
              and raw_auc >= 0.78                               # the discrimination genuinely exists (~0.83)
              and cont_auc >= 0.72 and fakes_caught_cont >= 0.6)  # continuous union RECOVERS it at the calibrated point
    print(f" [-REWIRE] binary-union decision AUC={bin_auc:.3f} (caught {fakes_caught_bin:.0%} fakes) | "
          f"continuous-union AUC={cont_auc:.3f} (caught {fakes_caught_cont:.0%}, false-veto {reals_vetoed_cont:.0%}) | "
          f"raw-score AUC={raw_auc:.3f}")
    cases.append(("-REWIRE continuous-veto recovers discarded AUC", pass17,
                  rep_cont or dict(verdict="ABSTAIN", sync_score=float("nan"), reason="continuous veto", coherent_fake_guarded=None)))

    # 19) ★NULL-SAFETY (§4c audit): a degenerate continuous-veto operating point must FAIL CLOSED, and a NaN coverage
    # magnitude must NOT credit coverage. The seg is a synced REAL segment (sync passes) so any ABSTAIN is due to
    # the leg alone — a silent ADMIT here would be a fail-OPEN null hole. Controls confirm no over-veto of reals.
    nan = float("nan")
    def _nseg():
        tr = np.cumsum(rng.standard_normal((T, 3)), axis=0)
        return {"m1": tr + 0.03 * rng.standard_normal((T, 3)), "m2": tr + 0.03 * rng.standard_normal((T, 3))}
    r_nsc = video_admission_cert(_nseg(), temporal={"motion_valid": True},
                                 veto_legs=[veto_leg("nanscore", veto=False, score=nan, veto_threshold=1.0)])
    r_nth = video_admission_cert(_nseg(), temporal={"motion_valid": True},
                                 veto_legs=[veto_leg("nanthr", veto=False, score=5.0, veto_threshold=nan)])
    r_nse = video_admission_cert(_nseg(), temporal={"motion_valid": True},
                                 veto_legs=[veto_leg("nanse", covers=["semantic"], score_effective=nan, score_effective_ci_lo=0.8)])
    r_realok = video_admission_cert(_nseg(), temporal={"motion_valid": True},
                                    veto_legs=[veto_leg("realok", veto=False, score=0.1, veto_threshold=1.0)])
    pass19 = (r_nsc["verdict"] == "ABSTAIN" and "fail-closed" in r_nsc["reason"]        # NaN score -> veto, not silent admit
              and r_nth["verdict"] == "ABSTAIN"                                          # NaN threshold -> veto
              and "semantic" not in r_nse["fake_coverage"]["effective_covered_dims"]     # NaN magnitude -> no coverage credit
              and r_realok["verdict"] == "ADMIT")                                         # legit low score -> NOT over-vetoed
    cases.append(("null-safety: degenerate continuous-veto fails closed", pass19, r_nsc))

    # 20) ★INF-SIBLING null-safety (agent pool sweep + adversarial agent): a NON-FINITE value INSIDE a strata dict must not
    # book effective/validated. coverage_strata adv_min=+inf (min ignores it / worst==+inf clears the bar) and
    # family_corr_strata adv_min=-inf (< thresh = spurious decorrelation) / NaN (silently dropped by max, easy-first)
    # all previously reached AUTO_ADMIT. Require ALL strata values finite.
    inf = float("inf")
    fam_ok2 = {"m0": "classical", "m1": "learned"}
    def _iseg():
        tr = np.cumsum(rng.standard_normal((T, 3)), axis=0)
        return {"m0": tr + .05 * rng.standard_normal((T, 3)), "m1": tr + .05 * rng.standard_normal((T, 3))}
    c_cinf = video_admission_cert(_iseg(), method_families=fam_ok2, family_error_corr=0.1, family_corr_validated=True,
                                  veto_legs=[veto_leg(d, covers=[d], score_effective=0.9, coverage_strata={"easy": 0.9, "adversarial_minimum": inf})
                                             for d in ("motion", "deformation", "acquisition", "appearance", "semantic")])
    c_fninf = video_admission_cert(_iseg(), method_families=fam_ok2, family_corr_strata={"adversarial_minimum": -inf})
    c_fnan = video_admission_cert(_iseg(), method_families=fam_ok2, family_corr_strata={"easy": 0.05, "adversarial_minimum": nan})
    pass20 = (not c_cinf["fake_coverage"]["effective_covered_dims_validated"]      # +inf coverage strata -> NOT validated
              and c_fninf["guard_basis"] == "measured-unvalidated"                 # -inf family strata -> not effective
              and c_fnan["guard_basis"] == "measured-unvalidated"                  # NaN family strata -> not effective (not silently dropped)
              and coherent_fake_band(c_cinf) != "AUTO_ADMIT" and coherent_fake_band(c_fninf) != "AUTO_ADMIT")
    # SCALAR family_error_corr path (completeness matrix — the strata fix missed it): -inf < FAMILY_DECORR_THRESH
    # would spuriously book 'effective-measured'; a non-finite validated corr must fall to measured-unvalidated.
    c_sninf = video_admission_cert(_iseg(), method_families=fam_ok2, family_error_corr=-inf, family_corr_validated=True)
    c_sok = video_admission_cert(_iseg(), method_families=fam_ok2, family_error_corr=0.1, family_corr_validated=True)
    pass20 = pass20 and c_sninf["guard_basis"] == "measured-unvalidated" and c_sok["guard_basis"] == "effective-measured"
    cases.append(("inf-sibling: non-finite strata value never validates", pass20, c_fninf))

    # 21) ★BLIND-LEG guard (appearance×geometry unlock): a LOW family_error_corr with an attested BLIND leg
    # (min per-leg power below chance) is the n_eff=1 trap, NOT decorrelation -> must not credit effective-measured.
    c_blind = video_admission_cert(_iseg(), method_families={"m0": "geom", "m1": "appearance"},
                                   family_error_corr=0.05, family_corr_validated=True, family_leg_min_power=0.02)
    c_seen = video_admission_cert(_iseg(), method_families={"m0": "geom", "m1": "appearance"},
                                  family_error_corr=0.05, family_corr_validated=True, family_leg_min_power=0.6)
    c_noattest = video_admission_cert(_iseg(), method_families={"m0": "geom", "m1": "appearance"},
                                      family_error_corr=0.05, family_corr_validated=True)
    c_pnan = video_admission_cert(_iseg(), method_families={"m0": "geom", "m1": "appearance"},
                                  family_error_corr=0.05, family_corr_validated=True, family_leg_min_power=nan)
    pass21 = (c_blind["guard_basis"] == "measured-blind-leg" and c_blind["effective_guarded"] is None   # blind -> downgraded
              and c_seen["guard_basis"] == "effective-measured"                                          # both detect -> credited
              and c_noattest["guard_basis"] == "effective-measured"                                      # no attest -> backward-compat
              and c_pnan["guard_basis"] == "measured-blind-leg")                                          # NaN attest -> unverifiable = blind
    cases.append(("blind-leg: low corr from a blind leg is not decorrelation", pass21, c_blind))

    # 22) ★ROBUST parse (crash-class, completeness): degenerate/malformed input must ABSTAIN this segment, not CRASH
    # the batch — a 0-frame extraction and a non-numeric strata are both real at scale.
    try:
        c_zero = video_admission_cert({"m0": np.zeros((0, 3)), "m1": np.zeros((0, 3))}, temporal={"motion_valid": True})
        c_junk = video_admission_cert({"m0": rng.standard_normal((T, 3)), "m1": rng.standard_normal((T, 3))},
                                      method_families={"m0": "a", "m1": "b"}, family_corr_strata={"adversarial_minimum": "garbage"})
        pass22 = (c_zero["verdict"] == "ABSTAIN" and c_zero["cert_valid"] is False
                  and c_junk["verdict"] in ("ABSTAIN", "ADMIT") and c_junk["guard_basis"] == "measured-unvalidated")
    except Exception:
        pass22 = False        # any crash on degenerate input fails this case
    cases.append(("robust-parse: degenerate input abstains, never crashes", pass22,
                  c_zero if pass22 else dict(verdict="CRASH", sync_score=float("nan"), reason="crashed", coherent_fake_guarded=None)))

    # 23) ★N-LEG guard (§11: guard 2->3-ben): family_error_corr as a LIST of pairwise corrs books the WORST pair —
    # a >=3-leg family is only as decorrelated as its worst-correlated pair; a blind leg still downgrades.
    fam3 = {"m0": "geom", "m1": "photo"}
    c_3ok = video_admission_cert(_iseg(), method_families=fam3, family_error_corr=[0.284, 0.30], family_corr_validated=True, family_leg_min_power=0.234)
    c_3corr = video_admission_cert(_iseg(), method_families=fam3, family_error_corr=[0.284, 0.70], family_corr_validated=True, family_leg_min_power=0.234)
    c_2scalar = video_admission_cert(_iseg(), method_families=fam3, family_error_corr=0.284, family_corr_validated=True, family_leg_min_power=0.234)
    c_3blind = video_admission_cert(_iseg(), method_families=fam3, family_error_corr=[0.284, 0.30], family_corr_validated=True, family_leg_min_power=0.05)
    pass23 = (c_3ok["effective_guarded"] is True                    # all pairs decorrelated -> credited
              and c_3corr["effective_guarded"] is False            # one correlated pair (0.70) -> worst-book -> not credited
              and c_2scalar["effective_guarded"] is True           # scalar 2-leg unchanged (LOCKED case backward-compat)
              and c_3blind["guard_basis"] == "measured-blind-leg")  # a blind leg among the 3 still downgrades
    cases.append(("N-leg: worst-pair booking + blind-leg still fires", pass23, c_3corr))

    # 24) ★CI-LOWER-BOUND blind-leg contract (finding): the attest must be the CI-LO magnitude, not the point.
    # A winner's-curse leg (point AUC 0.62 -> mag 0.24 > 0.10 bar, but CI-lo AUC 0.53 -> mag 0.06 < 0.10) must be
    # DOWNGRADED when its ci_lo is passed (the fragile direction gates on the lower bound, like coverage's ci_lo).
    fam2 = {"m0": "geom", "m1": "photo"}
    c_point = video_admission_cert(_iseg(), method_families=fam2, family_error_corr=0.284, family_corr_validated=True, family_leg_min_power=0.24)   # point (over-credits)
    c_cilo = video_admission_cert(_iseg(), method_families=fam2, family_error_corr=0.284, family_corr_validated=True, family_leg_min_power=0.06)    # CI-lo (correct)
    pass24 = (c_point["guard_basis"] == "effective-measured"          # point-mag would credit (the failure mode flags)
              and c_cilo["guard_basis"] == "measured-blind-leg")      # ci_lo correctly downgrades the winner's-curse leg
    cases.append(("CI-lo blind-leg contract (fragile-dir): ci_lo downgrades winner's-curse", pass24, c_cilo))

    # 25) ★CI-UPPER-BOUND corr contract (significance_checked_correlation): family_error_corr must be the CI-HI,
    # not the point. A small-n pair with POINT corr 0.30 (< 0.50, would credit) but Fisher-z ci_hi 0.60 (>= 0.50)
    # is NOT reliably decorrelated — passing the ci_hi must refuse the effective-measured decorrelation credit.
    fam25 = {"m0": "geom", "m1": "photo"}
    c_corrpt = video_admission_cert(_iseg(), method_families=fam25, family_error_corr=0.30, family_corr_validated=True, family_leg_min_power=0.3)   # point (over-credits)
    c_corrhi = video_admission_cert(_iseg(), method_families=fam25, family_error_corr=0.60, family_corr_validated=True, family_leg_min_power=0.3)   # ci_hi (correct)
    pass25 = (c_corrpt["effective_guarded"] is True           # point corr 0.30 credits (the failure mode flags)
              and c_corrhi["effective_guarded"] is False)     # ci_hi 0.60 >= 0.50 -> not reliably decorrelated -> refused
    cases.append(("CI-hi corr contract (fragile-dir): ci_hi refuses small-n over-credit", pass25, c_corrhi))

    # 26) ★n-AWARE blind-power enforcement (D self-audit sweep; flagged BLIND_POWER_THRESH by name): a POINT
    # magnitude 0.10 (AUC 0.55) at n=489 clears the 0.10 bar but its Hanley-McNeil CI-lo=0.4991<0.5 = chance.
    # Without family_leg_n the documented contract stands (credits); WITH family_leg_n the gate recomputes the
    # CI-lo magnitude and correctly flags the leg BLIND -> no false-credit.
    fam26 = {"m0": "geom", "m1": "photo"}
    c_pt_no_n = video_admission_cert(_iseg(), method_families=fam26, family_error_corr=0.16, family_corr_validated=True, family_leg_min_power=0.10)
    c_pt_n489 = video_admission_cert(_iseg(), method_families=fam26, family_error_corr=0.16, family_corr_validated=True, family_leg_min_power=0.10, family_leg_n=(245, 244))
    pass26 = (c_pt_no_n["guard_basis"] != "measured-blind-leg"        # no n -> documented-contract path (credits)
              and c_pt_n489["guard_basis"] == "measured-blind-leg")   # n=489 -> CI-lo 0.499 -> flagged blind (fixed)
    cases.append(("n-aware blind-power: point 0.10 @ n=489 flagged blind (was false-credit)", pass26, c_pt_n489))

    # 27) ★TEMPORAL-LEG NaN FAIL-CLOSED (D self-census): a non-finite extrap-frac / non-bool motion_valid /
    # deformation_flagged is UNASSESSABLE -> must VETO (was silently no-veto = ADMIT, nan-fail-open; sibling of the
    # veto_legs null-safety cases 19-20). Controls: valid values still ADMIT (no over-veto).
    _tn = np.cumsum(rng.standard_normal((T, 3)), axis=0)
    _tseg = lambda: {"m1": _tn + 0.03 * rng.standard_normal((T, 3)), "m2": _tn + 0.03 * rng.standard_normal((T, 3))}
    _r_nanextrap = video_admission_cert(_tseg(), temporal={"motion_valid": True, "motion_extrap_reliable_frac": float("nan")})
    _r_nanmv = video_admission_cert(_tseg(), temporal={"motion_valid": float("nan")})
    _r_nandf = video_admission_cert(_tseg(), temporal={"motion_valid": True, "deformation_flagged": float("nan")})
    _r_tok = video_admission_cert(_tseg(), temporal={"motion_valid": True, "motion_extrap_reliable_frac": 0.9, "deformation_flagged": False})
    pass27 = (_r_nanextrap["verdict"] == "ABSTAIN" and _r_nanmv["verdict"] == "ABSTAIN"
              and _r_nandf["verdict"] == "ABSTAIN" and _r_tok["verdict"] == "ADMIT")
    cases.append(("temporal-leg NaN/non-bool -> VETO (fail-closed); valid values still ADMIT", pass27, _r_nanextrap))

    print("video_admission_cert selftest:")
    for name, passed, r in cases:
        ok = ok and passed
        g = r.get("coherent_fake_guarded")
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:32s} -> {r['verdict']:7s} "
              f"sync={r['sync_score'] if r['sync_score']==r['sync_score'] else float('nan'):.2f} "
              f"guard={g} :: {r['reason'][:58]}")
    print(f"\n  -> {f'PASS: all {len(cases)} cases' if ok else 'FAIL: see above'}")
    print(f"  Thresholds: SYNC={SYNC_THRESH} MIN_PAIR={MIN_PAIR_THRESH} EXTRAP={EXTRAP_RELIABLE_FRAC} (conventional; ADMIT=SYNC not correctness)")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
