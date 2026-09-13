#!/usr/bin/env python3
"""OVER/UNDER-DETERMINED classification — REFUSE a false point, report a feasible region (D's linchpin) (an agent worktree).

Per overseer D: the ~85 newly-harvested inverse-target patents have UNAUDITED I-grades, and a VARIABLE-COUNT MISCOUNT —
quoting a recovered value for an UNDER-determined target — propagates a scope error into the whole vertical. The fix is a
service gate that counts INDEPENDENT constraints (the Jacobian RANK, not the raw claim count) vs unknowns, and REFUSES a
point estimate when under-determined, returning a FEASIBLE REGION (the free null-space directions) instead. The linchpin
case is DEPENDENT claims: n_obs ≥ n_params looks "determined" by a naive count, but if the claims are dependent the rank
is short and any quoted point is spurious.

GATES (null/control each):
 (G0) OVER-DETERMINED → TRUSTWORTHY POINT — 3 independent claims, 2 unknowns: status='over-determined', point_trustworthy=True,
      x̂ recovers the truth with a finite Σ.
 (G1) UNDER-DETERMINED → REFUSED + REGION — 1 claim, 2 unknowns: status='under-determined', point_trustworthy=False, NO Σ,
      deficiency=1, a null-space basis returned (the data pins only 1 combination).
 (G2) THE LINCHPIN — DEPENDENT CLAIMS — 2 claims, 2 unknowns but the 2nd claim is a multiple of the 1st (rank 1): a NAIVE
      count (n_obs ≥ n_params ⇒ 'determined') would quote a point; the rank-based gate flags UNDER-DETERMINED and REFUSES.
 (G3) THE REGION IS REAL — moving x̂ along the returned null direction leaves the observables UNCHANGED (the data genuinely
      cannot distinguish along it) — so it is a true feasible direction, not an artifact.

Run: python3 d.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from graph_engine.inverse_design.persona_design_gi_service import inverse_classified, identifiability

bnds = ([-5.0, -5.0], [5.0, 5.0]); X_TRUE = np.array([1.0, 0.8])


def lin(M):
    M = np.asarray(M, float)
    return lambda x: M @ np.asarray(x, float)


def main():
    print("=" * 98)
    print("OVER/UNDER-DETERMINED classification — refuse a false point, report a feasible region (D's linchpin)")
    print("=" * 98)

    # G0: 3 independent claims, 2 unknowns → over-determined → trustworthy point
    f3 = lin([[1.0, 0.6], [0.5, 1.0], [0.8, 0.3]])
    r0 = inverse_classified(f3, f3(X_TRUE), [0.0, 0.0], bnds, noise_rel=0.02)
    g0 = r0["status"] == "over-determined" and r0["point_trustworthy"] and np.linalg.norm(r0["x_hat"] - X_TRUE) < 0.05 and r0["cov"] is not None
    print(f"\n(G0) OVER-DETERMINED → TRUSTWORTHY — {r0['n_constraints']} constraints / {r0['n_unknowns']} unknowns → status='{r0['status']}', "
          f"point_trustworthy={r0['point_trustworthy']}, x̂={np.round(r0['x_hat'],3)} (err {np.linalg.norm(r0['x_hat']-X_TRUE):.3f}): {'PASS' if g0 else 'FAIL'}")

    # G1: 1 claim, 2 unknowns → under-determined → refuse + region
    f1 = lin([[1.0, 0.6]])
    r1 = inverse_classified(f1, f1(X_TRUE), [0.0, 0.0], bnds, noise_rel=0.02)
    g1 = r1["status"] == "under-determined" and not r1["point_trustworthy"] and r1["cov"] is None and r1["deficiency"] == 1 and r1["null_basis"] is not None
    print(f"\n(G1) UNDER-DETERMINED → REFUSED — {r1['n_constraints']}/{r1['n_unknowns']} → status='{r1['status']}', point_trustworthy="
          f"{r1['point_trustworthy']}, cov={r1['cov']}, deficiency={r1['deficiency']}, null-dir {np.round(r1['null_basis'][0],2)}: {'PASS' if g1 else 'FAIL'}")

    # G2: THE LINCHPIN — 2 claims but the 2nd is 2× the 1st (rank 1) → under-determined despite n_obs == n_params
    Md = np.array([[1.0, 0.6], [2.0, 1.2]])                          # row2 = 2·row1 → DEPENDENT claim
    fd = lin(Md)
    rd = inverse_classified(fd, fd(X_TRUE), [0.0, 0.0], bnds, noise_rel=0.02)
    naive_says = "determined" if Md.shape[0] >= 2 else "under"        # the naive claim-count verdict
    g2 = rd["status"] == "under-determined" and not rd["point_trustworthy"] and rd["n_constraints"] == 1
    print(f"\n(G2) LINCHPIN — DEPENDENT CLAIMS — 2 claims/2 unknowns but rank {rd['n_constraints']}; NAIVE count says '{naive_says}' (would quote a "
          f"point), the rank gate says '{rd['status']}' → REFUSED ({rd['point_trustworthy']}): {'PASS' if g2 else 'FAIL'}")

    # G3: the returned null direction is a REAL feasible direction — observables unchanged along it
    nd = r1["null_basis"][0]
    base = f1(r1["x_hat"]); moved = f1(r1["x_hat"] + 0.5 * nd)
    obs_change = float(np.max(np.abs(moved - base)))
    g3 = obs_change < 1e-6
    print(f"\n(G3) REGION IS REAL — moving x̂ by 0.5·null_dir changes the observable by {obs_change:.2e} (≈0 ⇒ the data cannot distinguish "
          f"along it, a true free direction): {'PASS' if g3 else 'FAIL'}")

    # G4: the demand-lane PRE-AUDIT — identifiability gives the SAME over/under verdict from the Jacobian ALONE (no
    # observed values), so the ~85 patents can be audited before their claim NUMBERS are in hand (only the forward is needed).
    idg = identifiability(fd, [1.0, 0.8], noise_rel=0.02)            # dependent-claim forward, Jacobian-only
    ido = identifiability(f3, [1.0, 0.8], noise_rel=0.02)            # over-determined forward
    g4 = idg["status"] == "under-determined" and idg["n_independent"] == 1 and ido["status"] == "over-determined"
    print(f"\n(G4) PRE-AUDIT (Jacobian-only, NO patent numbers) — identifiability() flags the dependent-claim forward status='{idg['status']}' "
          f"(n_indep {idg['n_independent']}/{idg['n_params']}) vs the over-determined one '{ido['status']}': the I-grade audit needs only the forward: {'PASS' if g4 else 'FAIL'}")

    # G5: the feasible REGION is CONCRETE + reportable — the in-bounds extent endpoints are at a bound and ALL match the data
    ext = r1["feasible_extent"][0]; nd = r1["null_basis"][0]
    x_lo = r1["x_hat"] + ext[0] * nd; x_hi = r1["x_hat"] + ext[1] * nd
    lob, hib = np.array(bnds[0]), np.array(bnds[1])
    in_bounds = bool(np.all(x_lo >= lob - 1e-6) and np.all(x_hi >= lob - 1e-6) and np.all(x_lo <= hib + 1e-6) and np.all(x_hi <= hib + 1e-6))
    at_bound = bool(np.min(np.abs(np.r_[x_lo - lob, x_lo - hib, x_hi - lob, x_hi - hib])) < 1e-4)
    data_ok = float(np.max(np.abs(f1(x_lo) - f1(r1["x_hat"])))) < 1e-6 and float(np.max(np.abs(f1(x_hi) - f1(r1["x_hat"])))) < 1e-6
    g5 = in_bounds and at_bound and data_ok
    print(f"\n(G5) FEASIBLE REGION IS CONCRETE — free over α∈[{ext[0]:.2f},{ext[1]:.2f}], endpoints x∈[{np.round(x_lo,2)} .. {np.round(x_hi,2)}] "
          f"in-bounds, at a bound, ALL matching the data: {'PASS' if g5 else 'FAIL'}")

    # G6: D's SHARPENING — a FULL-RANK but SLOPPY forward (the 2nd eigen-direction barely moves the data) is REFUSED by the
    # eigen-spectrum where rank/count alone say 'determined'. M≥N + full rank ≠ recoverable; only the σ-spectrum is the truth.
    Ms = np.array([[1.0, 0.5], [1.0, 0.5001]])                      # rank 2 but the 2nd eigen-direction is sloppy (near-flat)
    rs = inverse_classified(lin(Ms), lin(Ms)(X_TRUE), [0.9, 0.7], bnds, noise_rel=0.02)
    g6 = rs["status"] == "under-determined" and not rs["point_trustworthy"] and rs["deficiency"] == 1 and float(rs["sigma_per_direction"].max()) > 5.0
    print(f"\n(G6) SLOPPY DIRECTION (D's sharpening) — full rank {np.linalg.matrix_rank(Ms)} but σ-per-eigendirection "
          f"{np.round(rs['sigma_per_direction'],1)}: status='{rs['status']}' (deficiency {rs['deficiency']}) → REFUSED where rank/count say 'determined': {'PASS' if g6 else 'FAIL'}")

    allok = g0 and g1 and g2 and g3 and g4 and g5 and g6
    print("\n" + "=" * 98)
    if allok:
        print("VERDICT: the inverse now CLASSIFIES by the Fisher EIGEN-SPECTRUM (σ per eigen-direction) and REFUSES a point unless ALL")
        print("  directions are well-conditioned — catching a SLOPPY direction (σ=533, full rank) rank/count alone would miss. It quotes a")
        print(f"  feasible REGION (the null-space free directions) instead. The linchpin holds: 2 DEPENDENT claims that a naive count calls")
        print(f"  'determined' are correctly flagged under-determined (rank 1) and refused, where a quoted point would be a SPURIOUS recovered")
        print(f"  value. This is the variable-count rigor the ~85 unaudited patent I-grades need before any vertical trusts a recovery.")
    else:
        print(f"VERDICT: NOT all pass — G0 {g0} G1 {g1} G2 {g2} G3 {g3}. Fix at SOURCE.")
    print("=" * 98)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
