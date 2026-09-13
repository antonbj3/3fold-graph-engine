#!/usr/bin/env python3
"""search_cert_abstain.py — a CERTIFIED knowledge-graph paper-search.

DOGFOOD: point this package's own cert layer (σ_min / identifiability / over-determination /
decorrelation / ABSTAIN-when-insufficient) at the paper-search that ranks papers by how they
FILL HOLES / CONFIRM / REFUTE our knowledge-graph claims. The recursion closes on our own tool:
the search is held to the same discipline we hold our physics-certs to.

════════════════════════════════════════════════════════════════════════════════════════════════
THE CERT/ABSTAIN RULE (mapped from the physics-cert machinery). The search proposes matches over a
TARGET UNIVERSE = HOLES (unknown graph regions) ∪ CLAIMS (to confirm/refute); for a paper P it
DISCOVERS P's best target T and certifies THAT match, emitting ASSERT or ABSTAIN(rung, reason).
Three DECORRELATED signals are the "observation channels" (the measurement-Jacobian rows) — each
with a GENUINELY DIFFERENT blind spot:

    s_sem   semantic relevance   (embedding/token cosine)   blind: shared VOCABULARY w/o substance;
                                                                   echoes OUR own phrasing.
    s_cite  citation-bridge      (P in the citation nbhd of T's anchors)  blind: popularity/cartels;
                                                                   blind to brand-new & cross-field.
    s_topo  hole-topology match  (structured 4-axis claim↔need: QUANTITY×MECHANISM×REGIME×DYN-RANGE,
)   blind: encodes OUR guess of
                                                                   what fills the hole (diff-mechanism filler missed).

Decision is an ORDERED DISPATCH (like the over-det typer). EACH gate computes a REAL number:

  G0 EXTRACTION FENCE — extraction_confidence < τ ⇒ ABSTAIN (front-end; reused from b_literature_grader).

  G1 IDENTIFIABILITY (the σ_min analog).  Build P's relevance-PROFILE over the WHOLE target universe and ask:
     is the "which-target" assignment IDENTIFIABLE, or a GAUGE-ORBIT (P relevant to many targets equally =
     uninformative,  null-space)?  The discrete σ_min = argmax-margin
     margin_rel=(top−runnerup)/top  (→0 = degenerate tie = gauge).  The flatness = participation-ratio of the
     profile  flat=PR/M, PR=1/Σp²  (→1 = relevant-to-all = gauge orbit; the SAME PR machinery as N_eff below).
     margin_rel < τ_margin ⇒ NOT identifiable ⇒ ABSTAIN, sub-classified by the 3-RUNG hierarchy
     (3-rung hierarchy):
        (A) GAUGE   — flat AND a substantive broad match (a survey; relevant to ~all targets; intrinsic,
                      full-text won't pin ONE)  → EXTERNAL anchor (human scopes / accept multi-hole prior-only).
                      the cost→∞ limit of the resolution-cost axis.
        (C) REACH   — NOT flat / thin signal (a specific paper whose metadata just doesn't reach the
                      discriminating quantity)  → MORE same-kind measurement (FETCH FULL TEXT).  finite budget.

  G2 OVER-DETERMINATION (decorrelation).  For the identified top target, take the signals that FIRE and
     compute N_eff = participation-ratio of their evidence-vectors' correlation (N_eff = |S|²/Σρ_ij,
).  N_eff < 2 ⇒ WEAK-IDENTIFIABILITY rung (B) ⇒ ABSTAIN
     ("need a DECORRELATED 2nd signal").  ★COUNT ≠ over-det: two signals that share a blind spot
     (keyword-topo borrows sem's VOCABULARY axis → collinear evidence-vectors) give N_eff≈1 despite firing 2.

  G3 FALSE-ACCEPT FLOOR (pre-registered, tail-corrected).  floor = max(p̄^N_eff, β·p̄) — the β-clamp
     (variance-ESS floor clamped by the common-cause term; β = shared-source fraction, high if all
     signals read only the abstract).  floor > FA_target ⇒ ABSTAIN.

  G4 SIGN IDENTIFIABILITY (CLAIM targets only).  confirm/refute is a SIGN; identifiable ONLY if P is on the
     SAME QUANTITY as the claim (topo quantity-axis match).  Different quantity ⇒ can't tell confirm from
     refute ⇒ ABSTAIN "sign-gauge".  Same quantity ⇒ ASSERT confirm|refute per the result direction.

  ⇒ ASSERT only when identifiable (G1) ∧ over-determined N_eff≥2 (G2) ∧ floor≤FA_target (G3) ∧ (claims) sign
    identifiable (G4).  honest-negative = PASS: abstaining is the CORRECT output, not a failure.

EXTERNAL ANCHOR (non-tautology): every mock case carries a GROUND-TRUTH action set INDEPENDENTLY of the
gate; the gate is scored against it, and the pre-registered thresholds sit in WIDE gaps in the raw data
(margin_rel: identifiable∈[0.89,1.0] vs gauge/reach∈[0.12,0.47]; flat: gauge 0.79 vs reach 0.25).
FALSIFIER: an ASSERT on a spurious/gauge match (false-accept) OR an ABSTAIN on a genuinely over-determined
clean match breaks the cert.  A void-floor (always-ABSTAIN) baseline would trivially clear the abstain
cases but FAIL P1/P5 — so ASSERT-rate>0 with 0 false-accepts is required.
Figures forbidden — the verdict table + the gates ARE the verification.
"""
import numpy as np

# ─────────────────────────── PRE-REGISTERED thresholds (set BEFORE any verdict — non-tautological) ──────────────
TAU_EXTRACT = 0.60   # G0 front-end claim-extraction reliability floor
TAU_MARGIN  = 0.70   # G1 σ_min analog: argmax-margin (top−runnerup)/top must exceed this to be identifiable
TAU_FLAT    = 0.50   # G1 rung split: profile participation-ratio PR/M above this (among non-identifiable) ⇒ gauge (A)
TAU_LEVEL   = 0.15   # G1 rung split: a substantive top match must clear this to be a GAUGE (else no-signal ⇒ REACH)
N_EFF_MIN   = 2.00   # G2 need ≥2 DECORRELATED signals agreeing
FA_TARGET   = 0.05   # G3 pre-registered false-assert floor
SEM_FIRE, CITE_FIRE, TOPO_FIRE = 0.30, 0.15, 0.50   # per-signal firing thresholds
# per-signal individual false-match rate p_i (prob a RANDOM paper fires this signal on a RANDOM target — calibratable)
P_MISS = {"sem": 0.30, "cite": 0.12, "topo": 0.08}
# each signal's evidence-vector by its MECHANISM (a keyword-topo BORROWS the vocabulary axis → correlates w/ sem).
# axes = [vocabulary, citation-topology, structured-quantity]; DECORRELATED signals ⇒ orthogonal ⇒ N_eff = count.
EVID = {
    "sem":             np.array([1.0, 0.0, 0.0]),   # keys on vocabulary
    "cite":            np.array([0.0, 1.0, 0.0]),   # keys on citation-topology
    "topo_structured": np.array([0.0, 0.0, 1.0]),   # keys on the structured quantity (decorrelated)
    "topo_keyword":    np.array([0.9, 0.0, 0.436]), # keyword-topo shares the VOCABULARY axis ⇒ collinear-ish w/ sem
}


# ─────────────────────────── primitives ────────────────────────────────────────────────────────────────────────
def perp_fraction(J, b):
    """ / b_perp_fraction_grader.py — fraction of an error
    direction b NOT explained by range(J); perp≈0 ⇒ ALIASED (shared blind spot / agree-while-wrong)."""
    J = np.asarray(J, float); b = np.asarray(b, float)
    b_perp = b - J @ (np.linalg.pinv(J) @ b)
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(b_perp) / nb) if nb > 1e-30 else 0.0


def n_eff(evidence_vectors):
    """N_eff = participation-ratio of the correlation matrix of the FIRING signals' evidence-vectors
    (= N²/Σρ_ij).  Orthogonal ⇒ N_eff=count; collinear ⇒ →1.
    This IS the perp-fraction/blind-spot-independence discipline: collinear evidence = a SHARED blind spot."""
    if len(evidence_vectors) == 0:
        return 0.0
    E = np.array([v / (np.linalg.norm(v) + 1e-30) for v in evidence_vectors])   # unit rows
    R = E @ E.T                                                                 # Gram = correlation of unit vectors
    lam = np.linalg.eigvalsh(R); lam = lam[lam > 1e-12]
    return float((lam.sum() ** 2) / (lam ** 2).sum())                          # participation ratio


def fa_floor(pbar, neff, beta):
    """tail-corrected false-accept floor = max(p̄^N_eff, β·p̄)  (β-clamp; β = common-cause fraction)."""
    return float(max(pbar ** neff, beta * pbar))


def _cos(a_set, b_set):
    """set-cosine over a shared token vocabulary (a cheap, HONEST stand-in for an embedding cosine)."""
    return len(a_set & b_set) / (np.sqrt(len(a_set)) * np.sqrt(len(b_set)) + 1e-30) if a_set and b_set else 0.0


def _jaccard(a_set, b_set):
    return len(a_set & b_set) / (len(a_set | b_set) + 1e-30) if (a_set or b_set) else 0.0


def topo_match(paper_claim, target_need):
    """structured 3-axis claim↔need overlap (QUANTITY×MECHANISM×REGIME); returns (score, quantity_axis_hit).
     — a QUANTITY match alone is necessary-not-sufficient; need the axes."""
    axes = ["quantity", "mechanism", "regime"]
    hits = [1.0 if (paper_claim.get(a, set()) & target_need.get(a, set())) else 0.0 for a in axes]
    return float(np.mean(hits)), bool(paper_claim.get("quantity", set()) & target_need.get("quantity", set()))


def signals(paper, target):
    s_sem = _cos(paper["tokens"], target["tokens"])
    s_cite = _jaccard(paper["refs"], target["anchor_refs"])
    s_topo, q_hit = topo_match(paper.get("claim", {}), target.get("need", {}))
    return {"sem": s_sem, "cite": s_cite, "topo": s_topo, "quantity_hit": q_hit}


def firing_evidence(paper, sig):
    """which signals FIRE (exceed their threshold) and their evidence-vectors (mechanism-dependent)."""
    fired = {}
    if sig["sem"] >= SEM_FIRE:
        fired["sem"] = EVID["sem"]
    if sig["cite"] >= CITE_FIRE:
        fired["cite"] = EVID["cite"]
    if sig["topo"] >= TOPO_FIRE:
        # a STRUCTURED topo (decorrelated) vs a KEYWORD topo (borrows the vocabulary axis) — declared by the paper
        fired["topo"] = EVID["topo_keyword" if paper.get("topo_is_keyword") else "topo_structured"]
    return fired


# ─────────────────────────── the certifier (ordered dispatch over the target universe) ─────────────────────────
def certify_match(paper, universe):
    # ---- G0 EXTRACTION FENCE ----------------------------------------------------------------------------------
    ec = paper.get("extraction_confidence", 1.0)
    if ec < TAU_EXTRACT:
        return dict(verdict="ABSTAIN", rung="G0-extract",
                    reason=f"claim-extraction confidence {ec:.2f} < τ={TAU_EXTRACT:.2f} (front-end unreliable)")

    # ---- profile over the WHOLE universe (the σ_min / identifiability object) ----------------------------------
    M = len(universe)
    sig_by_t = [signals(paper, t) for t in universe]
    combined = np.array([s["sem"] + s["cite"] + s["topo"] for s in sig_by_t])     # per-target aggregate match
    order = np.argsort(combined)[::-1]
    top, runner = order[0], order[1]
    top_level = float(combined[top])
    margin = float((combined[top] - combined[runner]) / (top_level + 1e-9))        # discrete σ_min (argmax-margin)
    tot = combined.sum()
    p = combined / (tot + 1e-9)
    PR = 1.0 / ((p ** 2).sum() + 1e-30)                                            # profile participation ratio
    flat = float(PR / M)                                                           # →1 = gauge orbit
    top_t = universe[top]
    target_type = top_t.get("kind", "hole")
    diag = dict(top=top_t["id"], target_type=target_type, margin=round(margin, 2),
                flat=round(flat, 2), top_level=round(top_level, 2))

    # ---- G1 IDENTIFIABILITY (gauge-orbit abstain, 3-rung split) ------------------------------------------------
    if margin < TAU_MARGIN:
        if flat >= TAU_FLAT and top_level >= TAU_LEVEL:
            return dict(verdict="ABSTAIN", rung="A-gauge", **diag,
                        reason=f"GAUGE-ORBIT: relevant to many targets equally (flat={flat:.2f}≥{TAU_FLAT}, "
                               f"margin={margin:.2f}<{TAU_MARGIN}); which-target unidentifiable → EXTERNAL ANCHOR "
                               f"(human scopes / multi-hole prior-only)")
        return dict(verdict="ABSTAIN", rung="C-reach", **diag,
                    reason=f"REACH-LIMITED: margin={margin:.2f}<{TAU_MARGIN}, thin signal (flat={flat:.2f}, "
                           f"top_level={top_level:.2f}) — metadata doesn't reach the discriminating quantity "
                           f"→ MORE same-kind measurement (FETCH FULL TEXT)")

    # ---- G2 OVER-DETERMINATION (decorrelated legs) ------------------------------------------------------------
    sig_top = sig_by_t[top]
    fired = firing_evidence(paper, sig_top)
    neff = n_eff(list(fired.values()))
    diag.update(fired=sorted(fired), n_eff=round(neff, 2))
    if neff < N_EFF_MIN:
        return dict(verdict="ABSTAIN", rung="B-weak", **diag,
                    reason=f"WEAK-IDENTIFIABILITY: N_eff={neff:.2f}<{N_EFF_MIN:.0f} decorrelated legs "
                           f"(fired={sorted(fired)}; COUNT≠over-det when collinear) → need a DECORRELATED 2nd signal")

    # ---- G3 FALSE-ACCEPT FLOOR (tail-corrected) ---------------------------------------------------------------
    pbar = float(np.exp(np.mean([np.log(P_MISS[k]) for k in fired])))   # geo-mean individual false-match rate
    beta = paper.get("beta_common_cause", 0.02)                         # shared-source fraction (high if abstract-only)
    floor = fa_floor(pbar, neff, beta)
    diag.update(fa_floor=round(floor, 4))
    if floor > FA_TARGET:
        return dict(verdict="ABSTAIN", rung="G3-floor", **diag,
                    reason=f"FA-floor max(p̄^N_eff, β·p̄)={floor:.3f} > target {FA_TARGET} (β={beta} common-cause) "
                           f"→ evidence insufficient vs pre-registered false-accept target")

    # ---- G4 SIGN IDENTIFIABILITY (claims only) ----------------------------------------------------------------
    if target_type == "claim":
        if not sig_top["quantity_hit"]:
            return dict(verdict="ABSTAIN", rung="G4-sign-gauge", **diag,
                        reason="SIGN-GAUGE: P is on a DIFFERENT quantity than the claim → confirm/refute "
                               "unidentifiable (can't tell agreement from opposition)")
        direction = "REFUTE" if paper.get("result_opposes_claim") else "CONFIRM"
        return dict(verdict=f"ASSERT-{direction}", rung="assert", **diag,
                    reason=f"identifiable (margin {margin:.2f}), over-det (N_eff {neff:.2f}), floor {floor:.3f}≤{FA_TARGET}, "
                           f"SAME quantity → sign identifiable")

    return dict(verdict="ASSERT-FILL", rung="assert", **diag,
                reason=f"identifiable (margin {margin:.2f}≥{TAU_MARGIN}), over-det (N_eff {neff:.2f}≥{N_EFF_MIN:.0f}, "
                       f"fired={sorted(fired)}), floor {floor:.3f}≤{FA_TARGET}")


# ═════════════════════════════ MOCK GRAPH + CASES (external ground-truth) ═══════════════════════════════════════
def build_universe():
    holes = [
        dict(id="H_procarm", kind="hole", tokens={"process","operational","coordination","calibration","observability","incident","reliability","organization"},
             anchor_refs={"rushby_assurance","cmu_sei"},
             need=dict(quantity={"factor-model","surprise-residual"}, mechanism={"operational-anomaly"}, regime={"fleet-process"})),
        dict(id="H_v2sym", kind="hole", tokens={"symmetry","equivariance","gauge","group","representation","invariance","schur-weyl"},
             anchor_refs={"cohen_equivariant","kondor_reptheory"},
             need=dict(quantity={"equivariance-efficiency","sample-complexity"}, mechanism={"group-representation"}, regime={"permutation-symmetry"})),
        dict(id="H_neuralop", kind="hole", tokens={"neural","operator","fourier","surrogate","data-driven","learned","pde"},
             anchor_refs={"li_fno","kovachki_no"},
             need=dict(quantity={"operator-learning"}, mechanism={"fourier-layer"}, regime={"pde-surrogate"})),
        dict(id="H_worldmodel", kind="hole", tokens={"world","model","latent","dynamics","reinforcement","policy","planning"},
             anchor_refs={"ha_worldmodels","hafner_dreamer"},
             need=dict(quantity={"latent-dynamics"}, mechanism={"model-based-rl"}, regime={"control"})),
        dict(id="H_tailess", kind="hole", tokens={"tail","dependence","extremal","effective","sample","size","copula","coefficient","false","accept","common","cause"},
             anchor_refs={"assurance20","eckhardt_lee","kish_deff"},
             need=dict(quantity={"tail-effective-sample-size","extremal-coefficient"}, mechanism={"extreme-value-tail-dependence"}, regime={"rare-event-false-accept"})),
        dict(id="H_sigkteff", kind="hole", tokens={"fluctuating","lattice","boltzmann","landauer","thermal","entropy","fluctuation","dissipation","storage"},
             anchor_refs={"landauer","adhikari_flbm"},
             need=dict(quantity={"storage-floor","kt-eff"}, mechanism={"fluctuation-dissipation"}, regime={"near-degenerate-spectrum"})),
    ]
    claims = [
        dict(id="C_sigmin", kind="claim",
             tokens={"sigma","min","identifiability","observability","fisher","detectable","storable","floor"},
             anchor_refs={"cramer_rao","observability_gramian"},
             need=dict(quantity={"identifiability-floor","sigma-min-unification"}, mechanism={"fisher-observability"}, regime={"linear-gaussian-estimation"})),
        dict(id="C_reducedrep", kind="claim",
             tokens={"reduced","representation","generative","design","held-out","load","generalize","topology","full"},
             anchor_refs={"selto_dataset","topopt_generalization"},
             need=dict(quantity={"reduced-representation","generalization-gap","held-out-load"}, mechanism={"parametrized-manifold-optimization"}, regime={"topology-optimization"})),
    ]
    return holes + claims


def build_cases():
    """each paper carries a GROUND-TRUTH 'should' (external anchor, set independent of the gate logic)."""
    return [
        # P1 — genuine OVER-DETERMINED fill of H_tailess (3 decorrelated signals, structured topo, diverse sources)
        dict(name="P1 tail-ESS/EVT paper (extremal coeff for false-accept)",
             tokens={"tail","dependence","extremal","copula","coefficient","effective","sample","false","accept"},
             refs={"assurance20","eckhardt_lee","gumbel_evt"},
             claim=dict(quantity={"tail-effective-sample-size","extremal-coefficient"}, mechanism={"extreme-value-tail-dependence"}, regime={"rare-event-false-accept"}),
             beta_common_cause=0.02, should="ASSERT"),
        # P2 — broad SURVEY: high semantic to MANY targets, no specific quantity → GAUGE (rung A). The paper a naive
        # top-semantic search ranks #1 for many holes; the cert must REFUSE it.
        dict(name="P2 'Survey of UQ & reliability in scientific ML'",
             tokens={"uncertainty","reliability","identifiability","surrogate","model","error","observability","operator",
                     "dynamics","calibration","representation","thermal","tail","symmetry","process"},
             refs={"generic_review_a","generic_review_b"}, claim={}, beta_common_cause=0.30, should="ABSTAIN-gauge"),
        # P3 — WEAK: fires ONLY sem for H_v2sym (loose 'equivariance' vocab), no anchors, no structured quantity.
        dict(name="P3 loose-'equivariance' paper (sem only, new+cross-field)",
             tokens={"symmetry","equivariance","invariance","group"}, refs={"unrelated_x","unrelated_y"},
             claim={}, beta_common_cause=0.10, should="ABSTAIN-weak"),
        # P3b — CORRELATED-WEAK: fires sem AND a KEYWORD-topo for H_v2sym, but topo borrows the vocabulary axis →
        # N_eff≈1 despite 2 signals. Demonstrates COUNT ≠ over-determination.
        dict(name="P3b keyword-'equivariance-efficiency' (sem+kw-topo, collinear)",
             tokens={"symmetry","equivariance","invariance","group","representation","schur-weyl"}, refs={"unrelated_z"},
             claim=dict(quantity={"equivariance-efficiency"}, mechanism={"group-representation"}, regime=set()), topo_is_keyword=True,
             beta_common_cause=0.10, should="ABSTAIN-weak"),
        # P4 — REACH: terse abstract, weak sem straddling a couple targets, low absolute signal, NOT a survey.
        dict(name="P4 terse abstract (thermal model, ambiguous)",
             tokens={"thermal","model"}, refs=set(), claim={}, beta_common_cause=0.30, should="ABSTAIN-reach"),
        # P5 — REFUTE a CLAIM: same QUANTITY as C_reducedrep, over-determined, RESULT OPPOSES.
        dict(name="P5 'reduced-rep does NOT beat full-N on held-out' (same quantity, opposes)",
             tokens={"reduced","representation","held-out","load","generalize","topology","full","design"},
             refs={"selto_dataset","topopt_generalization"},
             claim=dict(quantity={"reduced-representation","generalization-gap","held-out-load"}, mechanism={"parametrized-manifold-optimization"}, regime={"topology-optimization"}),
             result_opposes_claim=True, beta_common_cause=0.02, should="ASSERT-REFUTE"),
        # P6 — SIGN-GAUGE: high sem+cite to C_sigmin ('σ_min','identifiability') but DIFFERENT quantity (solver
        # matrix-conditioning, not the storable/identifiable/detectable unification) → can't tell confirm/refute.
        dict(name="P6 'σ_min conditioning of a linear solver' (diff quantity)",
             tokens={"sigma","min","identifiability","observability","fisher","conditioning","solver","matrix"},
             refs={"cramer_rao","observability_gramian"},
             claim=dict(quantity={"matrix-conditioning"}, mechanism={"fisher-observability"}, regime={"linear-algebra"}),
             beta_common_cause=0.02, should="ABSTAIN-sign-gauge"),
        # P7 — EXTRACTION FENCE: hype abstract, extraction unreliable.
        dict(name="P7 hype abstract ('AI-powered physics')",
             tokens={"ai","physics","revolutionary"}, refs=set(), claim={}, extraction_confidence=0.30,
             beta_common_cause=0.30, should="ABSTAIN-extract"),
    ]


def _category(verdict, rung):
    """map a full verdict to the ground-truth category for scoring (external anchor)."""
    if verdict.startswith("ASSERT-REFUTE"):  return "ASSERT-REFUTE"
    if verdict.startswith("ASSERT"):          return "ASSERT"
    return {"A-gauge":"ABSTAIN-gauge","C-reach":"ABSTAIN-reach","B-weak":"ABSTAIN-weak",
            "G4-sign-gauge":"ABSTAIN-sign-gauge","G0-extract":"ABSTAIN-extract",
            "G3-floor":"ABSTAIN-floor"}.get(rung, "ABSTAIN-other")


def main():
    print("=" * 112)
    print("CERTIFIED PAPER-SEARCH — σ_min-identifiability + N_eff-decorrelation + tail-FA-floor abstain gate")
    print("=" * 112)
    universe = build_universe()
    cases = build_cases()
    print(f"\n  universe = {sum(t['kind']=='hole' for t in universe)} holes + {sum(t['kind']=='claim' for t in universe)} claims = {len(universe)} targets")
    print(f"  pre-registered: τ_margin={TAU_MARGIN} τ_flat={TAU_FLAT} τ_level={TAU_LEVEL} "
          f"N_eff_min={N_EFF_MIN} FA_target={FA_TARGET} τ_extract={TAU_EXTRACT}")
    print(f"\n  {'case':<52}{'verdict':<15}{'rung':<14}{'margin':>7}{'flat':>6}{'N_eff':>7}{'floor':>8}  gt✓")
    print("  " + "-" * 108)

    results, ok = [], True
    for c in cases:
        r = certify_match(c, universe)
        cat = _category(r["verdict"], r["rung"])
        gt_ok = (cat == c["should"])
        ok &= gt_ok
        results.append((c, r, cat, gt_ok))
        print(f"  {c['name'][:51]:<52}{r['verdict']:<15}{r['rung']:<14}"
              f"{r.get('margin','—'):>7}{r.get('flat','—'):>6}{r.get('n_eff','—'):>7}{r.get('fa_floor','—'):>8}"
              f"  {'✓' if gt_ok else '✗ want '+c['should']}")
    print()
    for c, r, cat, gt_ok in results:
        print(f"  • {c['name'][:66]}\n      → {r['verdict']} [{r['rung']}]: {r['reason']}")

    # ── FORCE THE POSITIVE (§'don't let the search assert what it can't certify') ─────────────────────────────
    print("\n  " + "─" * 108)
    print("  NAIVE vs CERT (force the positive): a naive top-semantic search asserts its argmax target for every paper.")
    naive_false, cert_false = 0, 0
    n_abstain_gt = sum(c["should"].startswith("ABSTAIN") for c, _, _, _ in results)
    for c, r, cat, gt_ok in results:
        if c.get("extraction_confidence", 1.0) < TAU_EXTRACT:
            continue
        sig = [signals(c, t) for t in universe]
        naive_top = universe[int(np.argmax([s["sem"] for s in sig]))]["id"]
        naive_asserts = max(s["sem"] for s in sig) >= SEM_FIRE
        cert_asserts = r["verdict"].startswith("ASSERT")
        if naive_asserts and c["should"].startswith("ABSTAIN"): naive_false += 1
        if cert_asserts and c["should"].startswith("ABSTAIN"):  cert_false += 1
        if naive_asserts and not cert_asserts and c["should"].startswith("ABSTAIN"):
            print(f"      naive→ASSERT '{naive_top}'  |  cert→{r['verdict']} ({r['rung']})   [{c['name'][:36]}]")
    print(f"  ⇒ naive false-asserts = {naive_false}/{n_abstain_gt} abstain-cases ;  "
          f"CERT false-asserts = {cert_false}  (the cert refuses what it can't certify)")

    # ── FLIP-TEST (teeth: the honest-negative is not permanent — add the decorrelated evidence ⇒ it ASSERTS) ──
    print("\n  FLIP-TEST (OODA force-the-fix): give P3 the missing DECORRELATED legs (anchor-ref + structured quantity).")
    p3_fixed = dict(build_cases()[2],
                    refs={"cohen_equivariant","kondor_reptheory"},
                    claim=dict(quantity={"equivariance-efficiency","sample-complexity"}, mechanism={"group-representation"}, regime={"permutation-symmetry"}))
    rf = certify_match(p3_fixed, universe)
    print(f"      P3 (sem only)  → ABSTAIN-weak   (N_eff≈1)")
    print(f"      P3-FIXED       → {rf['verdict']} [{rf['rung']}]  (N_eff={rf.get('n_eff')}, margin={rf.get('margin')})")
    flip_ok = rf["verdict"] == "ASSERT-FILL"

    # ── VOID-FLOOR guard (a degenerate always-ABSTAIN baseline is NOT a valid cert) ──────────────────────────
    assert_rate = float(np.mean([r["verdict"].startswith("ASSERT") for _, r, _, _ in results]))
    void_floor_fails = sum(c["should"].startswith("ASSERT") for c, _, _, _ in results)

    # ── GATES ────────────────────────────────────────────────────────────────────────────────────────────────
    print("\n  " + "=" * 108)
    g_gt = ok
    g_fa = (cert_false == 0)
    g_nondegen = (assert_rate > 0)
    g_beats_naive = (cert_false < naive_false)
    g_flip = flip_ok
    print(f"  [{'PASS' if g_gt else 'FAIL'}] G-GT            every verdict matches the EXTERNAL ground-truth action ({sum(x[3] for x in results)}/{len(results)})")
    print(f"  [{'PASS' if g_fa else 'FAIL'}] G-FALSE-ACCEPT  cert makes 0 false-asserts on ground-truth-ABSTAIN cases")
    print(f"  [{'PASS' if g_nondegen else 'FAIL'}] G-NONDEGEN      cert ASSERTs a non-empty set (assert-rate={assert_rate:.2f}) — NOT a void always-abstain floor")
    print(f"  [{'PASS' if g_beats_naive else 'FAIL'}] G-BEATS-NAIVE   cert false-asserts ({cert_false}) < naive top-semantic ({naive_false})")
    print(f"  [{'PASS' if g_flip else 'FAIL'}] G-FLIP          honest-negative NOT permanent: +decorrelated evidence ⇒ P3 flips to ASSERT")
    allok = g_gt and g_fa and g_nondegen and g_beats_naive and g_flip
    print("  " + "=" * 108)
    print(f"  void-floor check: an always-ABSTAIN baseline clears the abstain cases but FAILS {void_floor_fails} ASSERT cases (P1,P5).")
    print("\n  ALL PASS" if allok else "\n  SOME GATE FAILED — inspect above")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
