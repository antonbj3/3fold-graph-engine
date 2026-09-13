"""
gauge_completeness_cert.py — DEPLOYABLE anti-hallucination COMPLETENESS cert for a vision-dynamics twin. Makes J's gauge arc
(an earlier batch-656) importable by the pool's full_sim_ready_gate + the authenticity_completeness. Two laws, both proven on real physics:

  (1) COHERENT-FAKES ARE THE GAUGE ORBITS (an earlier batch/an earlier batch). A generative completion can fill any σ_min-null (an unanchored gauge) with a
      wrong value and it is UNDETECTABLE from the base substrate alone (a coherent-fake). The orbits of a vision-dynamics twin are
      the dimensional gauges {LENGTH, MASS, TIME} (continuous, exact monocular nulls) + the discrete CHIRALITY/reflection orbit
      (a null only for a RIGIDITY-strength base cert, an earlier batch). Each orbit is fenced ONLY by its own DECORRELATED anchor.
  (2) DETECTION POWER follows the unified two-correlation law (J an earlier batch ⊕ H bf5feae48, cross-substrate over-determination):
          d'(k) = (1 − ρ_g) · δ / sqrt(ρ_a·σ² + (1 − ρ_a)·σ²/k)
      k anchors, δ hallucination magnitude, σ per-anchor noise, ρ_a anchor↔anchor corr (variance floor), ρ_g anchor↔generator corr
      (signal gate: ρ_g=1 ⇒ the anchor CONFIRMS the fake). Per-gauge correctness needs ≥2 anchors decorrelated from each other AND
      the generator (an earlier batch/an earlier batch) — a 1-anchor gauge (e.g. MASS from density alone) is RIGIDITY-lifted but its value is ASSUMED.

⟹ a twin cert is anti-hallucination-COMPLETE iff it fences EVERY gauge orbit (has ≥1 decorrelated anchor each); the orbits it leaves
unanchored are exactly the coherent-fakes it will pass. Which orbits leak depends on the base cert's strength (rigidity vs full-recon).


front-end dual: gate.fence_orbits ⊇ REQUIRED_ANCHORS.values. QC: run `python gauge_completeness_cert.py` (selftest reproduces
an earlier batch diagonal catch-matrix + an earlier batch law).
"""
from __future__ import annotations
import math, sys

__all__ = ["GAUGE_ORBITS", "REQUIRED_ANCHORS", "completeness_cert", "detection_power", "per_gauge_correctness_certifiable"]

# each gauge orbit → the decorrelated anchor that fences its coherent-fake, and whether it is a null for a rigidity-only base cert
GAUGE_ORBITS = {
    "LENGTH":    {"kind": "continuous", "anchor": "metric-length", "rigidity_null": True,  "note": "monocular scale null (an earlier batch/an earlier batch)"},
    "MASS":      {"kind": "continuous", "anchor": "material-density", "rigidity_null": True, "note": "force/mass-ratio null; 1 anchor ⇒ value ASSUMED (an earlier batch)"},
    "TIME":      {"kind": "continuous", "anchor": "fps-clock", "rigidity_null": True,  "note": "no absolute clock null (an earlier batch)"},
    "CHIRALITY": {"kind": "discrete",   "anchor": "second-viewpoint", "rigidity_null": True, "note": "reflection null for a RIGIDITY-only cert; full-perspective catches it (an earlier batch/an earlier batch)"},
}
REQUIRED_ANCHORS = {g: v["anchor"] for g, v in GAUGE_ORBITS.items()}


def detection_power(delta, sigma, k=1, rho_a=0.0, rho_g=0.0):
    """the unified two-correlation detection-power law d'(k) (J an earlier batch ⊕ H). k anchors, ρ_a anchor↔anchor, ρ_g anchor↔generator."""
    if not (0.0 <= rho_a < 1.0 + 1e-12) or not (0.0 <= rho_g <= 1.0 + 1e-12):
        raise ValueError("rho_a in [0,1), rho_g in [0,1]")
    var = rho_a * sigma**2 + (1.0 - rho_a) * sigma**2 / max(k, 1)
    return (1.0 - rho_g) * delta / math.sqrt(var) if var > 0 else float("inf")


def per_gauge_correctness_certifiable(anchor_counts, rho_a=0.0, rho_g=0.0):
    """a gauge's ABSOLUTE VALUE is correctness-certifiable iff it has >=2 anchors decorrelated from each other AND the generator (an earlier batch/an earlier batch)."""
    out = {}
    for g in GAUGE_ORBITS:
        k = anchor_counts.get(g, 0)
        out[g] = bool(k >= 2 and rho_a < 0.99 and rho_g < 0.99)
    return out


def completeness_cert(anchors_present, base_cert="rigidity"):
    """
    anchors_present: iterable of anchor channel names the pipeline HAS (e.g. {'metric-length','fps-clock'}).
    base_cert: 'rigidity' (self-consistency; leaks the chirality orbit) or 'full-recon' (fences chirality itself).
    returns a dict: per-orbit {fenced|BLIND} + the list of coherent-fakes that will PASS + overall complete? verdict.
    """
    present = set(anchors_present)
    per_orbit, leaks = {}, []
    for g, v in GAUGE_ORBITS.items():
        # a full-recon base cert fences the discrete chirality orbit by itself (an earlier batch (d))
        self_fenced = (g == "CHIRALITY" and base_cert == "full-recon")
        fenced = self_fenced or (v["anchor"] in present)
        per_orbit[g] = "fenced" + (" (by base cert)" if self_fenced else "") if fenced else "BLIND"
        if not fenced:
            leaks.append({"orbit": g, "coherent_fake": v["note"], "needs_anchor": v["anchor"]})
    complete = len(leaks) == 0
    return {"complete": complete, "per_orbit": per_orbit, "coherent_fakes_that_pass": leaks,
            "base_cert": base_cert, "n_orbits": len(GAUGE_ORBITS), "n_fenced": len(GAUGE_ORBITS) - len(leaks)}


# ------------------------------------------------------------------ selftest (QC by re-run) ------------------------------------------------------------------
def _selftest():
    ok = True
    # 1. an earlier batch catch-matrix is DIAGONAL: each orbit fenced ONLY by its own anchor
    for g, v in GAUGE_ORBITS.items():
        rep = completeness_cert({v["anchor"]}, base_cert="rigidity")   # only THIS orbit's anchor present
        this_fenced = rep["per_orbit"][g].startswith("fenced")
        others_blind = all(rep["per_orbit"][o] == "BLIND" for o in GAUGE_ORBITS if o != g)
        ok &= this_fenced and others_blind
    print("  [1] diagonal catch-matrix (each orbit fenced only by its own anchor): %s" % ("PASS" if ok else "FAIL"))

    # 2. missing an anchor => that orbit's coherent-fake PASSES
    rep = completeness_cert({"metric-length", "fps-clock", "material-density"}, base_cert="rigidity")  # no 2nd-viewpoint
    chir_leaks = any(l["orbit"] == "CHIRALITY" for l in rep["coherent_fakes_that_pass"]) and not rep["complete"]
    ok &= chir_leaks
    print("  [2] rigidity cert w/o 2nd-viewpoint leaks the CHIRALITY orbit (H's missed fake): %s" % ("PASS" if chir_leaks else "FAIL"))

    # 3. full-recon base cert self-fences chirality
    rep2 = completeness_cert({"metric-length", "fps-clock", "material-density"}, base_cert="full-recon")
    ok &= rep2["complete"]
    print("  [3] full-recon base cert self-fences CHIRALITY (complete w/o 2nd-viewpoint): %s" % ("PASS" if rep2["complete"] else "FAIL"))

    # 4. an earlier batch detection-power law: monotone in k, → 0 as ρ_g→1, saturates at ρ_a floor
    d_k1 = detection_power(2.0, 1.0, k=1, rho_a=0.0, rho_g=0.0)
    d_k16 = detection_power(2.0, 1.0, k=16, rho_a=0.0, rho_g=0.0)
    d_rg1 = detection_power(2.0, 1.0, k=8, rho_a=0.0, rho_g=1.0)
    floor = detection_power(2.0, 1.0, k=10**9, rho_a=0.4, rho_g=0.0)   # k→∞ saturates at the ρ_a variance floor
    law = abs(d_k1 - 2.0) < 1e-9 and abs(d_k16 - 8.0) < 1e-9 and d_rg1 < 1e-12 and abs(floor - 2.0/math.sqrt(0.4)) < 1e-4
    ok &= law
    print("  [4] detection-power law (d'(1)=2, d'(16)=8, d'(ρ_g=1)=0, ρ_a=0.4 floor=%.3f): %s" % (2.0/math.sqrt(0.4), "PASS" if law else "FAIL"))

    # 5. per-gauge correctness: MASS with 1 anchor is NOT correctness-certifiable; TIME/LENGTH with 2 are
    pgc = per_gauge_correctness_certifiable({"LENGTH": 2, "TIME": 2, "MASS": 1, "CHIRALITY": 2})
    cc = (pgc["LENGTH"] and pgc["TIME"] and not pgc["MASS"])
    ok &= cc
    print("  [5] per-gauge correctness (LENGTH✓ TIME✓ MASS✗ with 1 anchor, an earlier batch): %s" % ("PASS" if cc else "FAIL"))

    print("\n  gauge_completeness_cert selftest: %s" % ("ALL PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    print("=" * 100); print("gauge_completeness_cert — deployable anti-hallucination completeness (an earlier batch-656)"); print("=" * 100)
    # demo: a pipeline with only a metric anchor (monocular recon + 1 known length)
    demo = completeness_cert({"metric-length"}, base_cert="rigidity")
    print("\n  demo — pipeline with only {metric-length}, rigidity base cert:")
    print("    complete=%s (%d/%d orbits fenced)" % (demo["complete"], demo["n_fenced"], demo["n_orbits"]))
    for g, s in demo["per_orbit"].items(): print("      %-10s %s" % (g, s))
    print("    coherent-fakes that PASS: %s" % [l["orbit"] for l in demo["coherent_fakes_that_pass"]])
    print()
    sys.exit(0 if _selftest() else 1)
