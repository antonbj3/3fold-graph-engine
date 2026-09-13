"""
LITERATURE-GRADER PROTOTYPE: repoint the patent-grader
machinery (grade_claim / dominates / claim_membership, grounded in code)
at PAPERS instead of patents. RELIABILITY-FIRST-THEN-SCALE: build the CORE grading as a sound machine-checkable primitive,
and put the reliability RISK where D named it — the CLAIM-EXTRACTION front-end — behind an ABSTAIN gate (never mis-grade a
paper whose claim can't be reliably structured). Worked example = the co-design paper arXiv 2604.25193 (D's grounding: it
scooped the co-design CONCEPT — photonics, NO convergence certification — while the platform CERTIFIES co-design vs
self-confirming collapse via the σ_min-governor; so the platform DOMINATES on the certification axis).

★SCHEMA (a claim = a structured, machine-checkable object; the front-end must produce this from prose):
  Claim = {domain, mechanism:set(capability-atoms), scope:dict(capability->bool/value), provenance}
★PRIMITIVES (patent-grader repointed):
  claim_membership(paper, platform) — does the platform's mechanism COVER the paper's? (platform can do what the paper does)
  dominates(platform, paper)        — platform covers the paper's mechanism AND has a STRICTLY broader scope (a capability
                                      the paper LACKS) ⇒ the platform dominates (forward claim-match, the patent-grader rule)
  novelty_gap(paper, platform)      — mechanism/scope the paper has that the platform LACKS (the paper's genuine residual
                                      novelty over the platform = a gap to fill, NOT a domination)
★RELIABILITY GATE (the front-end risk): extraction_confidence < τ ⇒ ABSTAIN (do NOT grade); a mis-extracted claim is worse
  than no grade. This is the  discipline at the claim-extraction layer.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

TAU_EXTRACT = 0.6   # abstain below this claim-extraction confidence

def claim_membership(paper, platform):
    """platform COVERS the paper's mechanism (capability atoms, excluding domain-instance-only atoms)."""
    core = paper["mechanism"] - paper.get("domain_instance_atoms", set())
    return core <= platform["mechanism"]

# scope LATTICE per axis (H contract: existence-grounding necessary-but-INSUFFICIENT — need established≥claimed)
SCOPE_LATTICE = {"hardware": ["none", "one-robot", "robot-class", "universal"],
                 "policy_class": ["none", "restricted", "broad", "any"],
                 "systems": ["none", "one-system", "few-systems", "cross-domain"]}
def _scope_ok(platform, k):
    """an atom is scope-matched iff its anchor ESTABLISHED a scope >= the CLAIMED scope (else it's a scope-launder)."""
    lvl = platform.get("scope_level", {}).get(k)
    if not lvl:
        return True   # no scope axis declared for this capability → scope-agnostic (treat as matched)
    axis, claimed, established = lvl
    L = SCOPE_LATTICE.get(axis, [])
    return (established in L) and (claimed in L) and L.index(established) >= L.index(claimed) >= 1

def dominating_atoms(platform, paper):
    """the GROUNDED + SCOPE-MATCHED platform capability-atoms the paper lacks. Each must carry a committed-cert provenance
    anchor (H b67eef24) AND its anchor's established scope must be >= the claimed scope (H 33a6adeb scope-matched). An
    un-grounded padding atom OR a scope-laundered atom (established<claimed) does NOT count = the machine-checkable don't-launder gate."""
    g = platform.get("grounding", {})
    return {k: g[k] for k in platform["scope"]
            if bool(platform["scope"].get(k)) and g.get(k) and _scope_ok(platform, k) and not bool(paper["scope"].get(k))}

def dominates(platform, paper):
    """forward claim-match: platform covers the paper's mechanism AND has a GROUNDED capability the paper's scope LACKS."""
    return bool(claim_membership(paper, platform) and dominating_atoms(platform, paper))

def novelty_gap(paper, platform):
    """what the paper has that the platform LACKS (residual novelty over the platform)."""
    mech = paper["mechanism"] - platform["mechanism"]
    scope = {k: v for k, v in paper["scope"].items() if v and not platform["scope"].get(k)}
    return mech, scope

def grade_paper(paper, platform):
    if paper.get("extraction_confidence", 0.0) < TAU_EXTRACT:
        return {"verdict": "ABSTAIN", "reason": "claim-extraction confidence %.2f < τ=%.2f (front-end unreliable)" %
                (paper.get("extraction_confidence", 0.0), TAU_EXTRACT)}
    mem = claim_membership(paper, platform); dom = dominates(platform, paper)
    gap_m, gap_s = novelty_gap(paper, platform)
    return {"verdict": "PLATFORM-DOMINATES" if dom else ("PLATFORM-COVERS" if mem else "PAPER-OUTSIDE-PLATFORM"),
            "membership": mem, "dominates": dom,
            "grounded_dominating_atoms": dominating_atoms(platform, paper),   # H contract: every DOMINATES self-documents its provenance anchors
            "paper_residual_novelty": {"mechanism": sorted(gap_m), "scope": gap_s}}


def main():
    print("=" * 100)
    print("B LITERATURE-GRADER PROTOTYPE — patent-grader (membership/dominates/novelty-gap) repointed at PAPERS")
    print("=" * 100)

    # ── Platform capabilities represented by the reference implementation ──
    platform = {
        "domain": "certified co-design / OED",
        "mechanism": {"adjoint", "topology-opt", "info-gain", "co-design", "sigma_min-governor", "off-policy-probe",
                      "conservation-cert", "model-class-ensemble"},
        "scope": {"convergence_certified": True, "self_confirming_guard": True, "policy_class_any": True,
                  "real_hardware_grounded": True, "unspanned_fraction_ceiling_honest": True},
        # ★grounding (H contract): each scope-capability must carry a committed-cert PROVENANCE anchor to count toward DOMINATES
        "grounding": {"convergence_certified": "three-leg convergence guard (edcc5186)", "self_confirming_guard": "A-V55 arc ⊗ J batch (9341092d)",
                      "policy_class_any": "A-V55-1 span-limit + C461 3-leg (c92980b6)", "real_hardware_grounded": "J batch real UR10e (9341092d)",
                      "unspanned_fraction_ceiling_honest": "B A-V55-1 span-limit → H unspanned-fraction law (c92980b6)"},
        # ★scope_level (H contract): (axis, CLAIMED, ESTABLISHED); an atom counts only if established >= claimed (honest self-scope)
        "scope_level": {"real_hardware_grounded": ("hardware", "one-robot", "one-robot"),   # honest: batch = ONE robot, not universal
                        "policy_class_any": ("policy_class", "any", "broad"),                # H's latent launder: claim 'any', established 'broad' → DROPPED
                        "convergence_certified": ("systems", "few-systems", "cross-domain")},# established broader (A-V55-3 turbulence) → OK
        "provenance": "three-leg convergence guard, UR10e batch evaluation, self-confirmation guard, and capability grades",
    }

    # ── WORKED EXAMPLE 1: the co-design paper arXiv 2604.25193 (D's grounding — scooped concept, no convergence cert) ──
    codesign_paper = {
        "domain": "co-design (photonics)",
        "mechanism": {"adjoint", "topology-opt", "co-design", "photonics"},
        "domain_instance_atoms": {"photonics"},          # photonics = a domain instance, not a distinct capability
        "scope": {"convergence_certified": False, "self_confirming_guard": False, "policy_class_any": False,
                  "restricted_policy_class": True},        # the paper DODGES self-confirming by RESTRICTION (D's read)
        "extraction_confidence": 0.85,                     # high — the paper's claim is clearly stated
        "provenance": "arXiv 2604.25193 (Apr 2026)",
    }
    r1 = grade_paper(codesign_paper, platform)
    print("\n  [1] co-design paper (arXiv 2604.25193):")
    print("      verdict=%s ; membership=%s dominates=%s" % (r1["verdict"], r1["membership"], r1["dominates"]))
    print(" grounded dominating-atoms (H contract — every DOMINATES carries provenance): %s" %
          {k: v for k, v in r1["grounded_dominating_atoms"].items()})
    print("      paper residual novelty over platform: %s" % r1["paper_residual_novelty"])
    print("      ⇒ platform DOMINATES on the certification axis (adds certified convergence + self-confirming guard +")
    print("        real-hardware grounding, which the paper LACKS). NOTE: 'any-policy-class' is DROPPED by the scope-matched")
    print("        gate (H 33a6adeb — established only 'broad', not 'any'); the prose is graded at the honest surviving scope.")
    print("        The paper's residual novelty = the PHOTONICS domain instance (a real gap = demonstrate the guard on a photonics twin).")

    # ── WORKED EXAMPLE 2: a NULL — a paper OUTSIDE the platform (a genuinely different mechanism the platform lacks) ──
    outside_paper = {
        "domain": "neural operator surrogate",
        "mechanism": {"neural-operator", "fourier-layer", "data-driven-surrogate"},
        "scope": {"amortized_inference": True, "convergence_certified": False},
        "extraction_confidence": 0.80,
        "provenance": "hypothetical NO paper",
    }
    r2 = grade_paper(outside_paper, platform)
    print("\n  [2] NULL — neural-operator surrogate paper (different mechanism):")
    print("      verdict=%s ; membership=%s dominates=%s ; paper novelty=%s" %
          (r2["verdict"], r2["membership"], r2["dominates"], r2["paper_residual_novelty"]["mechanism"]))
    print("      ⇒ NOT dominated (different mechanism the platform lacks) — the grader correctly does NOT over-claim domination.")

    # ── WORKED EXAMPLE 3: the RELIABILITY GATE — a paper whose claim can't be reliably extracted → ABSTAIN ──
    ambiguous_paper = {
        "domain": "vague/hype abstract", "mechanism": {"ai", "physics"}, "scope": {},
        "extraction_confidence": 0.35, "provenance": "low-signal abstract",
    }
    r3 = grade_paper(ambiguous_paper, platform)
    print("\n  [3] RELIABILITY GATE — low-extraction-confidence paper:")
    print("      verdict=%s (%s)" % (r3["verdict"], r3["reason"]))
    print("      ⇒ the front-end risk (D's 'reliability-risk = claim-extraction') is fenced by an ABSTAIN gate: never mis-grade.")

    # gates for the prototype's own soundness
    g1 = r1["verdict"] == "PLATFORM-DOMINATES" and r1["paper_residual_novelty"]["mechanism"] == ["photonics"]
    g2 = r2["verdict"] == "PAPER-OUTSIDE-PLATFORM" and not r2["dominates"]
    g3 = r3["verdict"] == "ABSTAIN"
    ok = g1 and g2 and g3
    print("\n  PROTOTYPE GATES: dominates-worked-example=%s ; null-not-overclaimed=%s ; abstain-on-low-extraction=%s" % (g1, g2, g3))
    print("  ★HONEST SCOPE: the CORE grading (membership/dominates/novelty-gap) is a sound machine-checkable primitive; the")
    print("  RELIABILITY RISK is the claim-EXTRACTION front-end (prose→structured Claim), fenced here by the abstain gate.")
    print("  NEXT (scale, per D 'reliability-first-then-scale'): a calibrated extractor with a held-out confidence + a human/")
    print("  cross-worktree spot-check on the ABSTAIN and DOMINATES verdicts (the two decision-relevant tails). NOT yet an NLP parser.")
    print("ALL_PASS =", ok)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
