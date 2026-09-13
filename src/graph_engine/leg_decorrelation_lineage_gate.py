#!/usr/bin/env python3
"""d_leg_decorrelation_lineage_gate -- the STRUCTURAL (layer-1) decorrelation precondition for booking two models as
INDEPENDENT admission legs, gating ANCHOR node MULTI-CHANNEL-ADMISSION. Implements the agent pool-order rule stated twice
: "modeller med DELAD backbone raknas som EN kanal -- mat family_error_corr innan nagot bokas som eget ben"
(shared-backbone models count as ONE channel; measure family_error_corr before booking any as its own leg).

TWO-LAYER decorrelation gate (nominal vs effective -- both required, weakest-link):
  * LAYER 1 (STRUCTURAL / NOMINAL, THIS module): backbone weights-lineage from data/model_registry. If two candidate
    legs share a weights lineage (or either lineage is UNVERIFIED), they are NOT independent -> FENCE. This is
    NECESSARY, not sufficient: a distinct-lineage pair still may have correlated errors (shared training data /
    shared failure modes on hard inputs).
  * LAYER 2 (EMPIRICAL / EFFECTIVE, reconstruction_trust_map.element_trust `family_error_corr` + video_admission_cert):
    the MEASURED error correlation of the two legs on real segments. This module NEVER substitutes for it -- it says
    which pairs are even ELIGIBLE to be measured (a shared/unverified pair should not be measured-then-booked; it is
    fenced up front), and flags which distinct pairs MUST have the empirical corr measured before booking.

Why layer 1 is a real gap: element_trust already gates on family_error_corr, but it TRUSTS the caller's declaration
that the legs are distinct families. Nothing checks the backbone lineage. A caller can pass two DINOv2-encoder depth
legs with a low measured corr on an easy set and book them as decorrelated -- a  slip
(declared-distinct is nominal; shared-weights is the effective correlation). This module supplies the missing check.

Verdicts (fail-closed):
  FENCED-UNVERIFIED       -- either lineage unknown/UNVERIFIED  -> cannot certify independence
  FENCED-SHARED-LINEAGE   -- identical canonical weights lineage -> one channel (N_eff~1)
  FLAGGED-SHARED-FAMILY   -- a shared weights-lineage FAMILY heuristic fired (HYPOTHESIS, confirm lineage) -> treat as
                             fenced until the shared-weights claim is confirmed/refuted
  PROVISIONAL-NEEDS-EMPIRICAL -- distinct verified lineage -> ELIGIBLE; now REQUIRES measured family_error_corr (layer 2)

Run: python3 scripts/d_leg_decorrelation_lineage_gate.py [--json OUT]   (selftest + real-registry scan, no GPU)
"""
import argparse
import glob
import json
import math
import os
import sys

_REGISTRY = os.environ.get("MODEL_REGISTRY", "data/model_registry")

# Shared WEIGHTS-lineage families (HEURISTIC -- a match is a HYPOTHESIS of shared weights to confirm, not a proof).
# Only families with a well-known shared-weights descent are listed; keep conservative (fail-closed handles the rest).
WEIGHTS_LINEAGE_FAMILIES = {
    # ★DINOv2-encoder family: Depth-Anything AND Metric3Dv2 AND VideoDepthAnything all run a DINOv2-ViT encoder
    # (registry-measured, agent pool order-korrigering; confirmed by the L-TWO-NEURAL-DEPTH retraction
    # that M3Dv2 & VDA are COMMON-MODE, not decorrelated). Registering the monocular-neural-depth models here makes
    # layer-1 FENCE a 2-neural-depth stacking (the retraction's error) instead of passing it to empirical as "provisional".
    "dinov2-encoder": ("dinov2", "depth-anything", "depth anything", "metric3d", "metric3dv2", "metric-3d",
                       "video-depth-anything", "videodepthanything", "video depth anything", "vda"),
    "dust3r-croco": ("dust3r", "mast3r", "monst3r", "cut3r", "croco"),  # DUSt3R/CroCo geometric weights descent
    "superpoint": ("superpoint", "lightglue"),                          # LightGlue matches on SuperPoint features
    "sam": ("segment anything", "sam 1", "sam 2", "sam 3", "sam1", "sam2", "sam3"),
    "encodec-codec": ("encodec", "soundstream", "neural audio codec", "neural codec"),
    "qwen-vl": ("qwen", "internvl"),
}
_UNVERIFIED = {"", "unknown", "unverified", "?", "none", "tbd", "n/a"}


def canon_lineage(bl):
    """Canonicalize a backbone_lineage string; return None if unknown/UNVERIFIED (fail-closed)."""
    s = str(bl or "").strip().lower()
    if s in _UNVERIFIED or "unverified" in s:
        return None
    return s


def lineage_family(bl_canon):
    """Return the shared-weights family name if the (canonical) lineage matches a known family, else None."""
    if not bl_canon:
        return None
    for fam, keys in WEIGHTS_LINEAGE_FAMILIES.items():
        if any(k in bl_canon for k in keys):
            return fam
    return None


def pair_decorrelation_verdict(lineage_a, lineage_b):
    """Structural (layer-1) verdict for booking two models as INDEPENDENT legs. Fail-closed."""
    ca, cb = canon_lineage(lineage_a), canon_lineage(lineage_b)
    if ca is None or cb is None:
        which = "both" if (ca is None and cb is None) else ("A" if ca is None else "B")
        return {"verdict": "FENCED-UNVERIFIED",
                "reason": f"lineage unverified ({which}) -> cannot certify independence", "independent": False}
    if ca == cb:
        return {"verdict": "FENCED-SHARED-LINEAGE",
                "reason": f"identical weights lineage '{ca}' -> one channel (N_eff~1)", "independent": False}
    fa, fb = lineage_family(ca), lineage_family(cb)
    if fa and fa == fb:
        return {"verdict": "FLAGGED-SHARED-FAMILY",
                "reason": f"shared weights-lineage family '{fa}' (HYPOTHESIS -- confirm) -> fenced until refuted",
                "independent": False}
    return {"verdict": "PROVISIONAL-NEEDS-EMPIRICAL",
            "reason": "distinct verified lineage -> ELIGIBLE; measure family_error_corr (layer 2) before booking",
            "independent": None}   # None = not yet established; layer-2 decides


NEFF_BAR = 1.3333 # 2-leg n_eff bar (worst-corr-Kish); n_eff < bar => legs correlated => FENCE.
                  # ★Reconciled 1.30->1.3333 (L-RECONCILE, D-owned gate): 1.30 <=> rho<=0.538 admitted a
                  # [0.50,0.538] error-corr band the video cert FAMILY_DECORR_THRESH=0.50 fences (same n_eff=2/(1+rho)
                  # curve, pure constant mismatch). 1.3333=2/(1+0.50) adopts rho<=0.50 UNIFORMLY so the lineage gate is
                  # never a SOLE admission path looser than the video cert (fail-closed; conservative-governs).
                  # ★★PURPOSE-SPECIFIC (D self-correction, supersedes the earlier "recalibrate to 1.600"
                  # note): this gate books DECORRELATED = "a fake is unlikely to fool BOTH legs" = a FALSE-ACCEPT FLOOR,
                  # whose n_eff is the joint-tail exponent = CANON 2/(1+rho) (Ledford-Tawn, C-verified, D re-verified
                  # via the bivariate orthant). So NEFF_BAR=1.3333 (canon at rho=0.50) is CORRECT, and n_eff>K on NEGATIVE
                  # rho is PHYSICAL here (anti-correlated legs almost never both fire). empirical_neff MUST be the canon
                  # floor form. Do NOT feed a participation_ratio n_eff: participation OVER-credits the floor (rho=0.6
                  # participation 1.471 > BAR passes a pair the canon 1.25 correctly FENCES = fail-open).
                  # "use participation_ratio" fix is purpose-specific to the VARIANCE/SELECTION object; it must NOT be
                  # propagated into false-accept / decorrelation thresholds. Repro: scripts/physics_exp/d_correct_neff_
                  # bar_purpose_confusion_lineage_gate_is_false_accept_floor_canon_correct_participation_overcredits.py


def compose_decorrelation_verdict(lineage_a, lineage_b, empirical_verdict=None, empirical_neff=None,
                                  empirical_min_stratum_n=None, min_stratum_floor=25, nonlinear_dependent=False):
    """FULL 2-layer composed decorrelation verdict for booking two legs as INDEPENDENT.

    ★Layer-1 (structural lineage, this module) is a CHEAP A-PRIORI PRE-FILTER, NOT a decorrelation DETERMINANT.
    proved (l_sub_m3d_cotracker_leg.py) that DISTINCT-lineage legs (Metric3D-dinov2 x CoTracker-kubric) are STILL
    error-correlated via a SHARED DISCONTINUITY-TRIGGER (empirical n_eff 1.134 < 1.3333) -- a mechanism layer-1 is blind
    to. The DETERMINANT is the measured error-correlation on adversarial-min -- ★MEASURED ON A PRE-REGISTERED SET,
    NOT AN AGREEMENT-GATED SUBSET (select-then-test): gating on the tracks/items where the legs AGREE forces
    e1~=e2 on that subset -> n_eff deflates 2.0->1.0 -> a genuinely-decorrelated pair reads as common-mode. D's
    n-floor PARTIALLY guards this (a small agreement-gated subset -> n-starved -> UNDECIDABLE, not FENCE) but a LARGE
    agreement-gated subset (e.g. 75% agree) defeats the floor and FALSE-FENCES the pair. So the caller MUST supply the
    empirical from a pre-registered/random set. ★Layer-2 = evidence-gate
    (verified by D, EXIT=0): FENCE on CI-LOWER(corr;alpha/S) > w_fence (EVIDENCE of common-mode), ADMIT on
    CI-UPPER < w_fence (EVIDENCE of decorrelation), else UNDECIDABLE (thin data -> honest abstain, NOT fence, NOT
    admit). Pass its 3-way `empirical_verdict` in. Layer-1's value is two things: (a) fence known shared-WEIGHTS
    pairs a-priori to SAVE the empirical budget; (b) narrow the family (fewer strata -> tighter alpha/S).
    ★Correction (adopting): thin data is UNDECIDABLE, NOT FENCE -- a fence mislabels a RECOVERABLE pair as
    correlated, and D's earlier 'fall back to pooled' suggestion REOPENS (proved); use the evidence-gate.
    Weakest-link:
      layer-1 FENCE (shared-weights/shared-family/unverified)  -> FENCE  (don't even measure -- save budget)
      layer-1 PROVISIONAL + empirical_verdict ADMIT            -> DECORRELATED (booked -- the ONLY admit path)
      layer-1 PROVISIONAL + empirical_verdict FENCE            -> FENCE  (measured common-mode; layer-1 was blind)
      layer-1 PROVISIONAL + empirical_verdict UNDECIDABLE      -> UNDECIDABLE (recoverable with more n; not bookable)
      layer-1 PROVISIONAL + no empirical                       -> NOT-DETERMINED (run  before booking)
    ★★COMPLETENESS BOUNDARY -- the THREE common-mode types this composition covers, and the ONE it delegates
    (KRAV2 grade of D-OWN-LINEAGE-GATE, verified by driving this gate): a booked DECORRELATED means decorrelated
    against types 1&2 ONLY, NOT the BIAS axis:
      type-1 RELABEL (shared weights, declared distinct) -> caught by layer-1 lineage (or model-hash upstream)
      type-2 VARYING common-mode (shared trigger, err-corr>0)-> caught by layer-2 err-corr/n_eff (n_eff<NEFF_BAR->FENCE)
      type-3 TOTAL SHARED BIAS (constant/first-moment DC)    -> ★INVISIBLE HERE. err-corr is a SECOND-MOMENT probe:
             two legs sharing a CONSTANT bias c have err-corr ~0 (a constant has zero variance) -> n_eff ~2 -> this gate
             books DECORRELATED though they share a total bias. No member-derived proxy can see a bias shared by ALL
             members (the no-member-proxy-sees-shared-bias THEOREM); type-3 is caught
             ONLY by an EXTERNAL verified-decorrelated ANCHOR (first-moment residual vs a measured truth). This gate
             DELEGATES type-3 to the caller -- a booked DECORRELATED is NECESSARY-NOT-SUFFICIENT on the bias axis; the
             caller MUST compose an external anchor before treating the pair as bias-decorrelated. Do NOT read
             'distinct lineage AND err-corr n_eff>=bar' as closing the shared-bias axis.
    """
    l1 = pair_decorrelation_verdict(lineage_a, lineage_b)
    if not l1["verdict"].startswith("PROVISIONAL"):
        return {"verdict": "FENCE", "basis": "layer-1-structural", "layer1": l1["verdict"],
                "reason": f"layer-1 (cheap a-priori) fenced: {l1['reason']}", "bookable": False}
    # ★NONLINEAR-DEPENDENCE guard: the empirical n_eff below is a rho/rank
    # (LINEAR/monotone) decorrelation measure -- BLIND to a NONLINEARLY-dependent common-mode pair (b=f(a), rho~0 ->
    # high n_eff -> would falsely book DECORRELATED). If the caller ran the nonlinear_dependence_audit (or a
    # distance-correlation / MI measure) and it flags method-common-mode, FENCE regardless of the linear n_eff.
    # (Nonlinear extension of the DC-offset[] / monotone[] blindspot hierarchy of rho-based measures.)
    if nonlinear_dependent:
        return {"verdict": "FENCE", "basis": "layer-2-nonlinear-dependence", "layer1": l1["verdict"],
                "reason": "nonlinear dependence flagged (nonlinear_dependence_audit): the pair is common-mode via a "
                          "nonlinear coupling the rho/rank n_eff is blind to -- FENCE (do not book the linear n_eff)",
                "bookable": False}
    # ★Layer-2 = evidence-gate 3-way (preferred). Consume its verdict directly.
    if empirical_verdict is not None:
        ev = str(empirical_verdict).upper()
        if ev == "ADMIT":
            return {"verdict": "DECORRELATED", "basis": "layer-2-L330-evidence", "layer1": l1["verdict"],
                    "reason": "distinct lineage (layer-1) AND L330 evidence-gate ADMIT (CI-upper < w_fence): "
                              "bookable as independent legs", "bookable": True}
        if ev == "FENCE":
            return {"verdict": "FENCE", "basis": "layer-2-L330-evidence", "layer1": l1["verdict"],
                    "reason": "L330 evidence-gate FENCE (CI-lower > w_fence = evidence of common-mode; layer-1 "
                              "blind to a shared trigger)", "bookable": False}
        return {"verdict": "UNDECIDABLE", "basis": "layer-2-L330-evidence", "layer1": l1["verdict"],
                "reason": "L330 evidence-gate UNDECIDABLE (thin data can't decide) -- recoverable with more n, "
                          "NOT a fence, NOT bookable", "bookable": False}
    if empirical_neff is None:
        return {"verdict": "NOT-DETERMINED", "basis": "layer-1-provisional-only", "layer1": l1["verdict"],
                "reason": "distinct lineage clears layer-1 but empirical is the DETERMINANT and was not "
                          "measured. Run L330 evidence-gate on adversarial-min before booking.", "bookable": False}
    # crude n_eff path (no verdict): thin data -> UNDECIDABLE, NOT fence (correction).
    if empirical_min_stratum_n is not None and empirical_min_stratum_n < min_stratum_floor:
        return {"verdict": "UNDECIDABLE", "basis": "empirical-n-starved", "layer1": l1["verdict"],
                "reason": f"empirical n-STARVED (min stratum {empirical_min_stratum_n} < {min_stratum_floor}); thin "
                          f"data can't decide -> UNDECIDABLE (recoverable), NOT fence (L330). Prefer the evidence-gate.",
                "bookable": False}
    # ★NON-FINITE GUARD (D self-sweep, NaN-evades-threshold class applied to D's own gate): a NaN/inf
    # empirical_neff (an EMPTY/degenerate empirical set -> np.std([])=NaN, or a corrupt participation ratio) previously
    # EVADED the `< NEFF_BAR` check (NaN < x is False) and fell through to DECORRELATED/bookable=True -- a FAIL-OPEN on
    # the FRAGILE credit direction (false-decorrelated inflates n_eff -> false ADMIT). A non-finite n_eff cannot decide
    # decorrelation -> UNDECIDABLE (recoverable, fail-closed), NOT bookable. Mirrors the n-starved branch above.
    if not math.isfinite(empirical_neff):
        return {"verdict": "UNDECIDABLE", "basis": "empirical-nonfinite", "layer1": l1["verdict"],
                "reason": f"empirical n_eff is non-finite ({empirical_neff}) -- an empty/degenerate empirical set cannot "
                          f"decide decorrelation; UNDECIDABLE (recoverable), NOT bookable (fail-closed, NaN-evades class)",
                "bookable": False}
    if empirical_neff < NEFF_BAR:
        return {"verdict": "FENCE", "basis": "layer-2-empirical", "layer1": l1["verdict"],
                "reason": f"measured n_eff {empirical_neff:.3f} < {NEFF_BAR}: common-mode (shared trigger -- layer-1 "
                          f"was blind)", "bookable": False}
    return {"verdict": "DECORRELATED", "basis": "layer-2-empirical", "layer1": l1["verdict"],
            "reason": f"distinct lineage AND measured n_eff {empirical_neff:.3f} >= {NEFF_BAR}: bookable", "bookable": True}


def verify_lineage_kind(lineage_id, declared_kind):
    """Verify a claim leg's SELF-DECLARED `lineage_kind` against the identity-keyed registry -- the antidote to the
    C4 self-declaration evasion (red-team, cert_grade_engine C4): a claim can declare lineage_kind=
    'classical-formula' to enter C4's LENIENT branch (a shared classical FORMULA on distinct data is not weights
    common-mode -> lenient), thereby DODGING the shared-WEIGHTS fence for a pair that actually shares trained
    weights (e.g. two dinov2-vit-s legs). Validation rule: a gate that branches safe-vs-unsafe on a leg FIELD must
    VERIFY that field from an identity-keyed registry, never the producer's word.

    This resolves lineage_id against the known TRAINED-WEIGHTS families (the registry identity map): if lineage_id
    denotes trained weights (dinov2 / dust3r / superpoint / sam / encodec / qwen), then a 'classical-formula'
    declaration is a MISLABEL and the weights-fence must apply. A genuinely classical lineage_id (not a weights
    family) is NOT auto-trusted either -- it cannot be registry-confirmed, so C4 should ESCALATE (the self-declared
    empirical ADMIT is the separate residual), never auto-PASS. Returns {verified_kind, mislabel, basis}."""
    canon = canon_lineage(lineage_id)
    fam = lineage_family(canon)
    if fam is not None:                                    # lineage_id resolves to a known trained-weights backbone
        mis = (declared_kind == "classical-formula")
        return {"verified_kind": "trained-weights", "mislabel": mis,
                "basis": (f"lineage_id '{lineage_id}' resolves to trained-weights family '{fam}' -- declared "
                          f"'classical-formula' is a MISLABEL; weights-fence applies" if mis
                          else f"lineage_id in trained-weights family '{fam}' (consistent with declaration)")}
    if canon is None:                                      # unknown/unverified lineage_id -> can't confirm anything
        return {"verified_kind": None, "mislabel": False,
                "basis": "lineage_id unverified/unknown -- cannot confirm classical vs weights (ESCALATE)"}
    return {"verified_kind": declared_kind, "mislabel": False,   # a plausible classical method, but not registry-proven
            "basis": ("lineage_id not in a known trained-weights family -- 'classical-formula' plausible but "
                      "UNVERIFIABLE against the registry (ESCALATE; do not auto-ADMIT on a self-declared verdict)")}


# reconstruction-relevant channels (category substring -> channel)
CHANNELS = {
    "depth/geometry": ("depth", "geometry", "3d recon", "4d recon", "stereo", "sfm", "metric"),
    "track/flow": ("track", "flow", "point", "optical"),
    "appearance/embed": ("embedding", "appearance", "re-id", "feature"),
    "pose/human": ("pose", "human", "hmr", "body", "smpl", "keypoint"),
    "audio": ("audio", "codec", "speech", "sound", "separation"),
}


def load_registry():
    regs = []
    for f in glob.glob(os.path.join(_REGISTRY, "*.json")):
        try:
            d = json.load(open(f))
            regs.extend(d) if isinstance(d, list) else regs.append(d)
        except Exception:
            continue
    return [r for r in regs if isinstance(r, dict) and r.get("name")]


def channel_of(cat):
    cat = str(cat or "").lower()
    for ch, keys in CHANNELS.items():
        if any(k in cat for k in keys):
            return ch
    return None


def scan(regs):
    by = {ch: [] for ch in CHANNELS}
    for r in regs:
        ch = channel_of(r.get("category"))
        if ch:
            by[ch].append({"name": r.get("name"), "lineage": r.get("backbone_lineage"),
                           "canon": canon_lineage(r.get("backbone_lineage"))})
    report = {}
    for ch, ms in by.items():
        verified = [m for m in ms if m["canon"]]
        # count decorrelated-eligible: >=2 whose PAIR verdict is PROVISIONAL (distinct verified, distinct family)
        eligible_pair = False
        flagged_family_pairs = []
        for i in range(len(verified)):
            for j in range(i + 1, len(verified)):
                v = pair_decorrelation_verdict(verified[i]["lineage"], verified[j]["lineage"])
                if v["verdict"] == "PROVISIONAL-NEEDS-EMPIRICAL":
                    eligible_pair = True
                elif v["verdict"] in ("FLAGGED-SHARED-FAMILY", "FENCED-SHARED-LINEAGE"):
                    flagged_family_pairs.append((verified[i]["name"], verified[j]["name"], v["verdict"]))
        report[ch] = {
            "n_models": len(ms), "n_lineage_verified": len(verified),
            "can_supply_decorrelated_pair_structural": eligible_pair,
            "shared_lineage_pairs_among_verified": flagged_family_pairs,
            "unverified": [m["name"] for m in ms if not m["canon"]],
        }
    return report


def _selftest():
    ok = True
    checks = []
    # synthetic pair verdicts
    checks.append(("unknown -> FENCED-UNVERIFIED",
                   pair_decorrelation_verdict("unknown", "ViT-pose")["verdict"] == "FENCED-UNVERIFIED"))
    checks.append(("same lineage -> FENCED-SHARED-LINEAGE",
                   pair_decorrelation_verdict("ViT-pose backbone", "ViT-pose backbone")["verdict"]
                   == "FENCED-SHARED-LINEAGE"))
    checks.append(("★DINOv2 depth vs DINOv2 embed -> FLAGGED-SHARED-FAMILY (string-distinct, weights-shared)",
                   pair_decorrelation_verdict("Depth-Anything (DINOv2 encoder)", "DINOv2 ViT-S/14")["verdict"]
                   == "FLAGGED-SHARED-FAMILY"))
    checks.append(("DUSt3R vs MASt3R -> FLAGGED-SHARED-FAMILY",
                   pair_decorrelation_verdict("DUSt3R stereo", "MASt3R-SfM")["verdict"] == "FLAGGED-SHARED-FAMILY"))
    checks.append(("distinct verified -> PROVISIONAL-NEEDS-EMPIRICAL (not auto-independent)",
                   pair_decorrelation_verdict("kubric-cnn-transformer", "SAM lineage")["verdict"]
                   == "PROVISIONAL-NEEDS-EMPIRICAL"))
    checks.append(("distinct verified is NOT booked independent without layer-2",
                   pair_decorrelation_verdict("kubric-cnn-transformer", "SAM lineage")["independent"] is None))

    # ---- composed 2-layer verdict on the two REAL pairs (layer-1 cheap pre-filter + layer-2 empirical determinant) ----
    # (A) dinov2 cluster: layer-1 fences a-priori (cheap) -- no empirical needed
    cA = compose_decorrelation_verdict("dinov2-vit-s (Metric3D)", "dinov2-vit-s (VDA)")
    checks.append(("compose: M3D x VDA (shared dinov2) -> FENCE by layer-1 a-priori (no empirical spent)",
                   cA["verdict"] == "FENCE" and cA["basis"] == "layer-1-structural"))
    # (B) the counterexample: distinct lineage clears layer-1 but empirical (n_eff 1.134) FENCES -- shared trigger
    cB0 = compose_decorrelation_verdict("dinov2-vit-s (Metric3D)", "kubric-cnn-transformer (CoTracker)")
    checks.append(("compose: M3D x CoTracker w/o empirical -> NOT-DETERMINED (layer-1 can't decide)",
                   cB0["verdict"] == "NOT-DETERMINED"))
    cB = compose_decorrelation_verdict("dinov2-vit-s (Metric3D)", "kubric-cnn-transformer (CoTracker)",
                                       empirical_neff=1.134, empirical_min_stratum_n=40)
    checks.append(("compose: M3D x CoTracker + empirical n_eff 1.134 -> FENCE by layer-2 (shared trigger, "
                   "layer-1 was BLIND) ", cB["verdict"] == "FENCE" and cB["basis"] == "layer-2-empirical"))
    # (C) a genuinely decorrelated distinct pair with healthy empirical -> the ONLY admit path
    cC = compose_decorrelation_verdict("kubric-cnn-transformer", "whisper conv+transformer audio",
                                       empirical_neff=1.85, empirical_min_stratum_n=60)
    checks.append(("compose: distinct + empirical n_eff 1.85 (n ok) -> DECORRELATED (bookable)",
                   cC["verdict"] == "DECORRELATED" and cC["bookable"] is True))
    # (D) distinct + empirical measured but n-STARVED -> UNDECIDABLE (recoverable, NOT fence) -- correction
    cD = compose_decorrelation_verdict("kubric-cnn-transformer", "whisper conv+transformer audio",
                                       empirical_neff=1.85, empirical_min_stratum_n=10)
    checks.append(("compose: distinct + empirical n-STARVED -> UNDECIDABLE not FENCE (recoverable, L330)",
                   cD["verdict"] == "UNDECIDABLE" and cD["bookable"] is False))
    # (E) distinct + evidence-gate 3-way verdict consumed directly (the preferred path)
    cE1 = compose_decorrelation_verdict("kubric", "whisper audio", empirical_verdict="ADMIT")
    cE2 = compose_decorrelation_verdict("kubric", "whisper audio", empirical_verdict="FENCE")
    cE3 = compose_decorrelation_verdict("kubric", "whisper audio", empirical_verdict="UNDECIDABLE")
    checks.append(("compose consumes L330 3-way: ADMIT->DECORRELATED(bookable), FENCE->FENCE, UNDECIDABLE->UNDECIDABLE",
                   cE1["verdict"] == "DECORRELATED" and cE1["bookable"] is True
                   and cE2["verdict"] == "FENCE" and cE2["bookable"] is False
                   and cE3["verdict"] == "UNDECIDABLE" and cE3["bookable"] is False))
    # (F) layer-1 FENCE dominates even if empirical says ADMIT (shared-weights a-priori wins -- don't spend empirical)
    cF = compose_decorrelation_verdict("dinov2-vit-s", "dinov2-vit-s", empirical_verdict="ADMIT")
    checks.append(("compose: layer-1 shared-weights FENCE dominates a spurious empirical ADMIT",
                   cF["verdict"] == "FENCE" and cF["basis"] == "layer-1-structural"))

    # ---- verify_lineage_kind: the C4 self-declaration antidote (red-team) ----
    # (G) ★EVASION CAUGHT: a dinov2 trained-weights leg SELF-DECLARED classical-formula -> MISLABEL (registry says
    # trained-weights). This is exactly how a shared-weights pair dodges C4's classical-formula lenient branch.
    g1 = verify_lineage_kind("dinov2-vit-s (Metric3D)", "classical-formula")
    g2 = verify_lineage_kind("dust3r stereo", "classical-formula")
    checks.append(("★verify_lineage_kind: dinov2/dust3r declared classical-formula -> MISLABEL caught (evasion shut)",
                   g1["mislabel"] and g1["verified_kind"] == "trained-weights" and g2["mislabel"]))
    # (H) honest declaration: a dinov2 leg declared trained-weights -> consistent, no mislabel
    g3 = verify_lineage_kind("dinov2-vit-s", "trained-weights")
    checks.append(("verify_lineage_kind: dinov2 declared trained-weights -> consistent (no false mislabel)",
                   not g3["mislabel"] and g3["verified_kind"] == "trained-weights"))
    # (I) genuine classical method (not a weights family): plausible but NOT registry-provable -> not auto-trusted
    g4 = verify_lineage_kind("fisher-z-confidence-interval", "classical-formula")
    checks.append(("verify_lineage_kind: genuine classical (fisher-z) -> not mislabel but UNVERIFIABLE (ESCALATE zone)",
                   not g4["mislabel"] and g4["verified_kind"] == "classical-formula" and "UNVERIFIABLE" in g4["basis"]))

    # (J) ★select-then-test: an AGREEMENT-GATED empirical deflates n_eff 2.0->1.0. The n-floor PARTIALLY guards
    # (small gated subset -> UNDECIDABLE), but a LARGE gated subset defeats it -> FALSE-FENCE => the empirical MUST
    # come from a PRE-REGISTERED (non-agreement-gated) set (contract requirement, documented in the docstring).
    _dist = ("kubric-cnn-transformer", "whisper conv+transformer audio")
    j_small = compose_decorrelation_verdict(*_dist, empirical_neff=1.0, empirical_min_stratum_n=10)
    j_large = compose_decorrelation_verdict(*_dist, empirical_neff=1.0, empirical_min_stratum_n=200)
    checks.append(("L364 select-then-test: n-floor guards a SMALL agreement-gated subset -> UNDECIDABLE (not false-fence)",
                   j_small["verdict"] == "UNDECIDABLE"))
    checks.append(("L364 select-then-test: a LARGE agreement-gated subset defeats the floor -> FENCE (empirical MUST be "
                   "pre-registered, not agreement-gated)", j_large["verdict"] == "FENCE"))

    # real-registry grounded assertions
    regs = load_registry()
    rep = scan(regs)
    ap = rep.get("appearance/embed", {})
    checks.append((f"appearance/embed 0 lineage-verified -> no structural decorrelated pair "
                   f"(measured {ap.get('n_lineage_verified')}/{ap.get('n_models')})",
                   ap.get("n_lineage_verified") == 0 and ap.get("can_supply_decorrelated_pair_structural") is False))
    checks.append((f"registry loaded ({len(regs)} models, non-trivial)", len(regs) >= 40))
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("\n  --- real per-channel structural decorrelation status ---")
    for ch, r in rep.items():
        print(f"  {ch:<18} models={r['n_models']:>2} verified={r['n_lineage_verified']:>2} "
              f"decorrelated-pair-eligible={r['can_supply_decorrelated_pair_structural']}")
    print(f"\nd_leg_decorrelation_lineage_gate selftest: {'ALL PASS' if ok else 'FAILURES'} "
          f"({sum(p for _, p in checks)}/{len(checks)})")
    return ok, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ok, rep = _selftest()
    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        json.dump({"per_channel": rep}, open(a.json, "w"), indent=1)
        print(f"  wrote {a.json}")
    sys.exit(0 if ok else 1)
