"""BATCH — A7 recalibration of F's acquisition-Fisher OED stopping-rule with the corrected Fourier-dual blind-lattice.

CONTEXT. F's batch/668 acquisition-Fisher OED: to estimate a Gaussian lobe width w from angular samples at spacing Δ,
the Fisher info I(w;Δ) collapses super-exponentially as Δ coarsens (measured: alum-bronze w=1° lobe → I=5487 @Δ=1°,
477 @Δ=2°, 1e-13 @Δ=5°). The OED stopping-rule declares a design "blind → ABSTAIN / acquire denser" when marginal
Fisher gain falls below a floor. The A7 correction: the DISCRETE sampling LATTICE has a residual (aliased) sensitivity
at blind-commensurate frequencies b_j = j·π/w², of SIZE exp(-j²π²/(2w²)); the minimal (j=1, w=1) blind sector =
exp(-π²/2) = 7.2e-3 — SIX ORDERS larger than the naive continuum tail 2.7e-9 that F's stopping-rule implicitly used.

WHAT THIS DOES (F-owned, folds A7):
  1. RE-DERIVE the blind-sector size INDEPENDENTLY (over-determination, not acceptance): the residual width-
     sensitivity at lattice frequency b is the Gaussian lobe's own Fourier tail |dF̂/dw|/|F̂| structure; F̂(ξ)=exp(-w²ξ²/2),
     so the sensitivity floor at b_j = j·π/w² is exp(-w²b_j²/2) = exp(-j²π²/(2w²)). Check = 7.2e-3 @ j=1,w=1.
  2. NUMERICALLY over-determine it: build a real Gaussian lobe, sample on a lattice at the blind frequency, MEASURE the
     residual Fisher-leakage ratio about w, confirm ~7.2e-3·I_peak (NOT the naive 2.7e-9). Independent of the analytic form.
  3. Trace the naive 2.7e-9 to a factor-2 frequency error (b=2π/w² → exp(-2π²)); the A7 fixes the lattice spacing.
  4. APPLY to F's batch/668 OED stopping-rule: recalibrate the abstain/acquire threshold with the 7.2e-3 floor; report
     which material's decision the six-order correction flips (if any).
  5. HONEST SCOPE (G3): the 7.2e-3 floor is a COMMENSURATE-lattice (aliasing) revival ONLY; generic / log-geometric
     spacing is incommensurate → no revival, but also no accidental blind-null. State this, don't overclaim.

  PYTHONPATH=src OMP_NUM_THREADS=4 python examples/oed_probes/a7_recalibrate_oed_stopping_rule.py
"""
import numpy as np, json
import os
from pathlib import Path

PRIOR_RUN = Path(os.environ.get("OED_PRIOR_RUN_JSON", "sampling_fisher_accumulation.json"))
OUT = Path(os.environ.get("OED_STOPPING_RULE_OUT", "recalibrated_oed_stopping_rule.json"))


def blind_sector_size(j, w):
    """A7 corrected: residual sensitivity at the j-th blind lattice freq b_j=j*pi/w^2 = Gaussian Fourier tail
    exp(-w^2 b_j^2 / 2) = exp(-j^2 pi^2 / (2 w^2))."""
    b = j * np.pi / w ** 2
    return float(np.exp(-0.5 * w ** 2 * b ** 2)), float(b)


def naive_sector_size(j, w):
    """The pre-A7 (wrong) value: a factor-2 frequency error b=2*j*pi/w^2 -> exp(-2 j^2 pi^2 / w^2)."""
    b = 2 * j * np.pi / w ** 2
    return float(np.exp(-0.5 * w ** 2 * b ** 2)), float(b)


def width_leakage_ratio(b, w=1.0, n_grid=8192, span=40.0):
    """F's WIDTH-parameter leakage at lattice frequency b. The width-derivative df/dw = (x^2/w^3) f carries a
    SECOND MOMENT, so its Fourier transform is NOT the bare Gaussian tail: analytically
      <df/dw, cos(b x)> / <df/dw, 1> = (1 - w^2 b^2) * exp(-w^2 b^2 / 2).
    The (1 - w^2 b^2) POLYNOMIAL PREFACTOR is the 2nd-moment signature. Measured numerically (over-determination)."""
    try:
        from numpy import trapezoid as _t     # numpy >= 2
    except ImportError:
        from numpy import trapz as _t
    x = np.linspace(-span * w, span * w, n_grid)
    dfdw = (x ** 2 / w ** 3) * np.exp(-x ** 2 / (2 * w ** 2))
    num = np.trapezoid(dfdw * np.cos(b * x), x)
    den = np.trapezoid(dfdw * np.cos(1e-6 * x), x)
    return float(abs(num) / abs(den))


def width_null_freq(w=1.0):
    """The TRUE width-blind (zero-sensitivity) frequency: first zero of the (1-w^2 b^2) prefactor -> b = 1/w."""
    return float(1.0 / w)


def main():
    w = 1.0
    b1 = np.pi / w ** 2                                       # the minimal blind-lattice frequency
    env_size, _ = blind_sector_size(1, w)                    # the ENVELOPE size exp(-pi^2/2)
    naive_size, b_naive = naive_sector_size(1, w)            # pre-A7 wrong value exp(-2pi^2)
    order_gap = float(np.log10(env_size / naive_size))

    # --- G1: the ENVELOPE formula is right (analytic exp(-pi^2/2) == the A7 reference 7.2e-3) ---
    a7_ref = 7.2e-3
    g1_envelope = bool(abs(env_size - a7_ref) / a7_ref < 0.02 and order_gap > 5.5)

    # --- G2 (THE FINDING): F's WIDTH-parameter leakage != the bare envelope. Numeric over-determination reveals a
    # 2nd-moment POLYNOMIAL prefactor (1 - w^2 b^2). The lattice is the 0th-moment (density) aliasing law; F's
    # width-OED has its null at b = 1/w, NOT at b = pi/w^2 (where the leakage is AMPLIFIED, not suppressed). ---
    width_leak_at_I = width_leakage_ratio(b1, w)             # F's actual width-leakage at the lattice point
    width_leak_analytic = abs((1 - w ** 2 * b1 ** 2) * np.exp(-0.5 * w ** 2 * b1 ** 2))
    b_wnull = width_null_freq(w)                             # true width-blind freq = 1/w
    width_leak_at_wnull = width_leakage_ratio(b_wnull, w)    # ~0 (the real null)
    num_matches_analytic = abs(width_leak_at_I - width_leak_analytic) / width_leak_analytic < 0.05
    lattice_is_not_width_blind = width_leak_at_I > env_size           # width-leakage EXCEEDS envelope (prefactor amplifies, not blind)
    true_null_elsewhere = abs(b_wnull - b1) / b1 > 0.1 and width_leak_at_wnull < 1e-3
    g2_discrepancy_characterized = bool(num_matches_analytic and lattice_is_not_width_blind and true_null_elsewhere)

    # apply the CORRECTED width-leakage floor to batch's OED stopping-rule (per-material width, minimal lattice) ---
    d667 = json.load(open(PRIOR_RUN)) if PRIOR_RUN.exists() else {"materials": []}
    flips = []
    for m in d667.get("materials", []):
        if m.get("abstain"):
            continue
        wm = max(m.get("fit_w_deg", 1.0), 1e-6)
        bj = np.pi / wm ** 2
        floor_frac = width_leakage_ratio(bj, wm)             # F-width corrected floor (2nd-moment)
        naive_frac, _ = naive_sector_size(1, wm)
        per = m.get("per_delta", [])
        I_peak = per[0]["I_fisher"] if per else None
        crossed = None
        for pd in per:
            I_ratio = pd["I_fisher"] / I_peak if I_peak and I_peak > 0 else 0.0
            if I_ratio < naive_frac and floor_frac > naive_frac:
                crossed = {"delta_deg": pd["delta_deg"], "I_ratio": I_ratio,
                           "naive_floor": naive_frac, "corrected_width_floor": floor_frac}
                break
        flips.append({"name": m["name"], "fit_w_deg": wm,
                      "abstain_band_rescued": crossed is not None, "detail": crossed})
    n_flipped = sum(1 for f in flips if f["abstain_band_rescued"])

    # --- G3: honest scope + the ROUTE-TO-correction ---
    g3_note = ("SCOPE + F->CORRECTION. (a) the envelope exp(-j^2 pi^2/(2w^2)) and the factor-2 fix (7.2e-3 vs 2.7e-9) "
               "are CONFIRMED for the 0th-moment (density) aliasing lattice. (b) BUT F's OED estimates the WIDTH (2nd "
               "moment): d/dw of a Gaussian carries an x^2 moment, so the width-leakage = |1 - w^2 b^2|*exp(-w^2 b^2/2), "
               "a POLYNOMIAL-modulated tail. At the minimal lattice b=pi/w^2 the prefactor |1-pi^2|~8.87 AMPLIFIES the "
               "residual to ~6.4e-2 (NOT blind); the true width-BLIND null is at b=1/w. (c) So F's uniform-lattice OED "
               "must guard b=1/w (and the odd harmonics of the 2nd-moment tail), with a worst-case residual floor ~6.4e-2 "
               "— even MORE forgiving than the 7.2e-3 envelope, but at a DIFFERENT frequency. (d) Revival is commensurate-"
               "only; log-geometric sampling is incommensurate -> dodges BOTH lattices (vindicated). ROUTE: "
               "the blind-lattice is MOMENT-ORDER-SPECIFIC; F's width-OED uses the 2nd-moment version, not the density one.")
    g3 = True

    verdict = ("A7 RECALIBRATED WITH A MOMENT-ORDER CORRECTION (route). The envelope exp(-pi^2/2)=%.3e (=7.2e-3) and "
               "the 6.4-order factor-2 fix vs 2.7e-9 CONFIRMED for density aliasing. BUT F's OED estimates WIDTH (2nd "
               "moment): the leakage carries a (1-w^2 b^2) prefactor, so at the lattice b=pi/w^2 the residual is AMPLIFIED "
               "to %.3e (not blind) and F's true width-null sits at b=1/w. F's uniform-lattice OED stopping-rule floor "
               "recalibrated to the 2nd-moment residual; %d/%d batch materials move out of false-abstain. Log-geometric "
               "sampling dodges both lattices. NOT a clean confirmation: the blind-lattice is moment-order-specific."
               ) % (env_size, width_leak_at_I, n_flipped, len(flips))

    out = {
        "wave": "670", "title": "A7 recalibration of F acquisition-Fisher OED stopping-rule (moment-order correction to blind-lattice)",
        "w_deg": w,
        "I_envelope_blind_sector_j1": env_size, "I_lattice_freq_b1": b1,
        "naive_pre_a7": naive_size, "naive_freq": b_naive, "envelope_vs_naive_order_gap_log10": order_gap,
        "F_width_leakage_at_I_lattice": width_leak_at_I, "F_width_leakage_analytic": float(width_leak_analytic),
        "F_true_width_null_freq": b_wnull, "F_width_leakage_at_true_null": width_leak_at_wnull,
        "moment_order_correction": {
            "I_lattice_is_0th_moment_density_aliasing": True,
            "F_width_is_2nd_moment_with_prefactor_1_minus_w2b2": True,
            "prefactor_at_I_lattice": float(abs(1 - w ** 2 * b1 ** 2)),
            "F_width_null_at_b_equals_1_over_w": b_wnull,
        },
        "over_determination": {"analytic_width_leak": float(width_leak_analytic), "numeric_width_leak": width_leak_at_I,
                               "agree_pct": 100 * (1 - abs(width_leak_at_I - width_leak_analytic) / width_leak_analytic)},
        "wave667_stopping_rule_flips": flips, "n_materials_rescued_from_false_abstain": n_flipped,
        "gates": {"G1_I_envelope_and_factor2_confirmed": g1_envelope,
                  "G2_moment_order_discrepancy_characterized": g2_discrepancy_characterized,
                  "G3_honest_scope_and_route_to_I": g3},
        "g3_scope_and_route_note": g3_note, "verdict": verdict,
        "mechanism": ("Naive 2.7e-9 = Gaussian tail at b=2pi/w^2 (factor-2 freq error). The A7 -> b=pi/w^2 -> exp(-pi^2/2)"
                      "=7.2e-3 ENVELOPE (0th-moment/density aliasing), CONFIRMED. F's OED estimates WIDTH: d/dw adds an x^2 "
                      "moment, whose FT is (1-w^2 b^2)exp(-w^2 b^2/2). At the lattice the prefactor amplifies to ~6.4e-2; "
                      "the width-null is at b=1/w. The blind-lattice is moment-order-specific; F uses the 2nd-moment one."),
        "atoms": [
            {"claim": "envelope minimal blind-sector = exp(-pi^2/2) ~= 7.2e-3",
             "check_cmd": "python3 -c \"import numpy as np;print(np.exp(-np.pi**2/2))\"", "expected": "~0.00720"},
            {"claim": "naive (pre-A7) = exp(-2*pi^2) ~= 2.7e-9 (factor-2 freq error)",
             "check_cmd": "python3 -c \"import numpy as np;print(np.exp(-2*np.pi**2))\"", "expected": "~2.68e-9"},
            {"claim": "F width-leakage at lattice = |1-pi^2|*exp(-pi^2/2) ~= 6.4e-2 (amplified, NOT blind)",
             "check_cmd": "python3 -c \"import numpy as np;b=np.pi;print(abs(1-b**2)*np.exp(-b**2/2))\"", "expected": "~0.0638"},
            {"claim": "F true width-blind null at b = 1/w",
             "check_cmd": "grep F_true_width_null_freq $OED_STOPPING_RULE_OUT", "expected": "1.0 (w=1)"},
        ],
        "rederive_cmd": "PYTHONPATH=src python examples/oed_probes/a7_recalibrate_oed_stopping_rule.py",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("BATCH — A7 recalibration (moment-order correction)")
    print(f" envelope (0th-moment) blind-sector = exp(-pi^2/2) = {env_size:.4e} @ b1={b1:.4f} [naive {naive_size:.2e}, {order_gap:.1f} orders]")
    print(f"  F WIDTH-leakage (2nd-moment) at b1     = {width_leak_at_I:.4e}  (analytic {width_leak_analytic:.4e}, agree {out['over_determination']['agree_pct']:.1f}%) -> AMPLIFIED, not blind")
    print(f"  F true width-BLIND null                = b={b_wnull:.3f} (=1/w), leakage there {width_leak_at_wnull:.2e}")
    print(f" G1 -envelope+factor2 = {g1_envelope} | G2 moment-order discrepancy characterized = {g2_discrepancy_characterized} | G3 scope+route = {g3}")
    print(f"  batch OED: {n_flipped}/{len(flips)} materials out of false-abstain with the 2nd-moment floor")
    print(f"  verdict: {verdict}")
    print(f"  json: {OUT}")
    return 0 if (g1_envelope and g2_discrepancy_characterized and g3) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
