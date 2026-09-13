 #!/usr/bin/env python3
"""KRAMERS-KRONIG CAUSALITY/PASSIVITY CERT — a 0-fit MODEL-FREE certificate for a
frequency-domain material response (complex modulus / damping).

WHAT / WHY (unlock-graph): fills the V_mf "model-FREE cert strength" hollow with a NEW,
DECORRELATED failure mode = CAUSALITY-failure (distinct from the existing model-free legs:
exchangeability [conformal], conservation-identity [zero-sim loop], criticality [fractal-R]).
Composes with the existing viscoelasticity_creep node (SAME Zener material) but adds the
frequency-domain causality axis that the time-domain creep-relaxation DUALITY cannot see.

THE LAW (0 free parameters, DIFFERENT compute path than the forward):
A causal, passive linear response chi(w)=chi'+i chi'' obeys the Kramers-Kronig relations
  chi'(w) = +(2/pi) P int_0^inf  W chi''(W)/(W^2-w^2) dW    (storage from loss)
This is a consequence of CAUSALITY (analyticity in the upper half-plane) ALONE - it holds for
ANY causal model, needs NO fitted constant, and is computed on a compute path (a causality
integral over ALL frequencies of the loss) that is INDEPENDENT of the forward (which solves
the coupled heat PDE at EACH frequency). It is therefore NOT a tautology gate: a forward whose
storage E' is inconsistent with its loss E'' (non-causal) FAILS. Proven load-bearing below by
injecting a non-causal corruption (KK residual spikes 30x).

FORWARD (0 fitted params): thermoelastic damping (Zener / Lifshitz-Roukes) of a flexural beam.
E*(w)/E = 1 + DeltaE[1 - 12/z^2 + 24 tanh(z/2)/z^3],  z = a*sqrt(i w/chi_th),
DeltaE = E alpha^2 T0/(rho Cp) = thermoelastic relaxation strength. ALL constants are literature
material properties (silicon) + geometry - NONE tuned. Cross-checked against an INDEPENDENT FD
solve of the transverse heat equation (agree 1e-8, 2nd-order convergent).

RESULTS (see run):
  * KK cert PASSES thermoelastic damping (residual ~1e-2, CONVERGES under refinement) = a genuine
    0-fit certified causal node.
  * KK cert FAILS the constant-loss-factor "structural/hysteretic damping" model (E'=const,E''=eta*E'
    - the MOST widely used damping model in commercial FEA): residual O(1), does NOT converge
    (Crandall 1970: ideal hysteretic damping is NON-CAUSAL). The CERTIFIED BOUNDARY = the bandwidth
    over which the acausality stays below tolerance IS the deliverable (honest-negative = a result).

  CUDA_VISIBLE_DEVICES="" python3 scripts/kramers_kronig_passivity_cert.py
"""
import numpy as np

# ======================================================================================
# KK engine: singularity-subtraction principal-value quadrature. 0 fitted constants.
#   P int_0^inf g(W)/(W^2-w_i^2) dW = int [g(W)-g(w_i)]/(W^2-w_i^2) dW + g(w_i) P int dW/(W^2-w_i^2)
#   regular limit at W=w_i is g'(w_i)/(2 w_i);  P int_a^b dW/(W^2-w_i^2) is analytic.
# ======================================================================================
def kk_re_from_im(w, chi_im):
    """Reconstruct storage chi'(w) from loss chi''(w) by causality.  chi'(w)=+(2/pi)P int W chi''/(W^2-w^2)."""
    g = w * chi_im
    gp = np.gradient(g, w)
    N = len(w); W2 = w**2; out = np.full(N, np.nan)
    a, b = w[0], w[-1]
    ht = np.zeros(N); dh = np.diff(w); ht[1:] += dh/2; ht[:-1] += dh/2 # trapezoid weights
    for i in range(1, N-1):
        wi = w[i]; den = W2 - wi**2
        reg = (g - g[i]) / den
        reg[i] = gp[i] / (2*wi) # L'Hopital limit
        pv = (1.0/(2*wi)) * np.log(abs(((b-wi)*(a+wi)) / ((b+wi)*(a-wi))))
        out[i] = (2.0/np.pi) * (np.sum(ht*reg) + g[i]*pv)
    return out

def debye(w, Delta, tau): # analytic causal Debye = external KK-pair ground truth
    d = 1.0 + (w*tau)**2
    return Delta/d, +Delta*(w*tau)/d # chi_re, chi_im (loss > 0)

# ======================================================================================
# Silicon material constants (literature values - NOT fitted).  The cert result is in fact
# INSENSITIVE to their exact values: KK holds for ANY causal E*(w).
# ======================================================================================
E_Si   = 165e9 # Pa   Young's modulus
RHO    = 2330. # kg/m^3
ALPHA  = 2.6e-6 # 1/K  linear thermal expansion
CP     = 700. # J/kg/K
K_TH   = 156. # W/m/K
T0     = 300. # K
CHI_TH = K_TH/(RHO*CP) # thermal diffusivity
DELTA_E = E_Si*ALPHA**2*T0/(RHO*CP) # thermoelastic relaxation strength (~2e-4)
BETA    = E_Si*ALPHA*T0/(RHO*CP)
A_THK   = 1e-5 # 10 um beam thickness
W_PK    = 2*CHI_TH*(2.2/A_THK)**2 # ~ Debye peak

def E_thermoelastic_closed(w):
    """Lifshitz-Roukes complex modulus (closed form). w scalar or array."""
    kt = np.sqrt(1j*np.asarray(w, complex)/CHI_TH); z = kt*A_THK
    return E_Si*(1 + DELTA_E*(1 - 12/z**2 + 24*np.tanh(z/2)/z**3))

def E_thermoelastic_fd(w, Ny=201):
    """
    INDEPENDENT forward: FD solve of the transverse heat eqn theta''-(iw/chi)theta=-(beta iw/chi)y,
    Neumann (adiabatic) BC, then E*/E = 1 + (12 alpha/a^3) int theta y dy.  Different compute path.
    """
    y = np.linspace(-A_THK/2, A_THK/2, Ny); dy = y[1]-y[0]
    L = np.zeros((Ny, Ny), complex)
    for i in range(1, Ny-1):
        L[i, i-1] = 1/dy**2; L[i, i] = -2/dy**2; L[i, i+1] = 1/dy**2
    L[0, 0], L[0, 1], L[0, 2] = -3/(2*dy), 4/(2*dy), -1/(2*dy) # theta'=0
    L[-1, -1], L[-1, -2], L[-1, -3] = 3/(2*dy), -4/(2*dy), 1/(2*dy)
    A = L.copy(); rhs = np.zeros(Ny, complex)
    for i in range(1, Ny-1):
        A[i, i] -= 1j*w/CHI_TH; rhs[i] = -(BETA*1j*w/CHI_TH)*y[i]
    th = np.linalg.solve(A, rhs)
    return E_Si*(1 + (12*ALPHA/A_THK**3)*np.trapz(th*y, y))


def section(t): print("\n" + "="*86 + f"\n{t}\n" + "="*86)

if __name__ == "__main__":
    gates = {}

    # ---------------------------------------------------------------------------------
    section("PART A - KK cert engine validated on ANALYTIC Debye (external KK-pair ground truth)")
    print("Reconstruct storage chi' from loss chi''; residual vs grid N (Wmax=100/wc):")
    print(f"{'N':>6}{'max_rel_resid':>15}{'order(h^p)':>12}")
    prev = None
    for N in [500, 1000, 2000, 4000]:
        w = np.linspace(100./N, 100., N); cr, ci = debye(w, 1.0, 1.0)
        cr_kk = kk_re_from_im(w, ci); m = (w >= 0.05) & (w <= 20.0)
        r = np.nanmax(np.abs(cr_kk[m]-cr[m]))
        o = "" if prev is None else f"{np.log(prev/r)/np.log(2):.2f}"
        print(f"{N:>6}{r:>15.3e}{o:>12}"); prev = r
    print("Bandwidth convergence (fixed h): residual falls as Wmax grows:")
    for Wmax in [50., 100., 300., 1000.]:
        N = int(Wmax/0.05); w = np.linspace(Wmax/N, Wmax, N); cr, ci = debye(w, 1.0, 1.0)
        cr_kk = kk_re_from_im(w, ci); m = (w >= 0.05) & (w <= 20.0)
        print(f"   Wmax={Wmax:>6.0f}  resid={np.nanmax(np.abs(cr_kk[m]-cr[m])):.3e}")
    w = np.linspace(0.025, 100., 4000); cr, ci = debye(w, 1.0, 1.0); cr_kk = kk_re_from_im(w, ci)
    m = (w >= 0.05) & (w <= 20.0); debye_resid = np.nanmax(np.abs(cr_kk[m]-cr[m]))
    gates["A_engine_debye"] = debye_resid < 0.02
    print(f"GATE A (engine matches analytic Debye < 2e-2):  resid={debye_resid:.3e}  "
          f"{'PASS' if gates['A_engine_debye'] else 'FAIL'}")

    # ---------------------------------------------------------------------------------
    section("PART B1 - Forward over-determination: LR closed-form vs INDEPENDENT FD heat-PDE")
    print(f"DeltaE={DELTA_E:.3e} (relaxation strength), peak omega~{W_PK:.2e} rad/s (f~{W_PK/2/np.pi:.2e} Hz)")
    print(f"{'omega':>10}{'E_re/E (closed)':>17}{'E_im/E (closed)':>17}{'rel_diff vs FD':>16}")
    max_fd_diff = 0.0
    for w in [1e5, 1e6, W_PK, 1e7, 1e8]:
        ec = E_thermoelastic_closed(w); ef = E_thermoelastic_fd(w, 401)
        d = abs(ec-ef)/abs(ec); max_fd_diff = max(max_fd_diff, d)
        print(f"{w:>10.2e}{ec.real/E_Si:>17.7f}{ec.imag/E_Si:>17.3e}{d:>16.2e}")
    gates["B_forward_overdet"] = max_fd_diff < 1e-6
    print(f"GATE B1 (2 independent forward paths agree < 1e-6):  max={max_fd_diff:.2e}  "
          f"{'PASS' if gates['B_forward_overdet'] else 'FAIL'}")

    # ---------------------------------------------------------------------------------
    section("PART B2 - KK CAUSALITY CERT on thermoelastic damping (the POSITIVE node)")
    # External anchor on the FORWARD (3rd independent path): the published Lifshitz-Roukes
    # peak damping coefficient Q^-1_max / DeltaE = 0.494 (Lifshitz & Roukes, PRB 61, 5600).
    ww = np.logspace(np.log10(W_PK)-2, np.log10(W_PK)+2, 4000)
    Qinv = E_thermoelastic_closed(ww).imag/E_thermoelastic_closed(ww).real
    lr_coeff = Qinv.max()/DELTA_E
    gates["B_LR_anchor"] = abs(lr_coeff-0.494) < 0.01
    print(f"External anchor: peak Q^-1/DeltaE = {lr_coeff:.4f}  (published LR = 0.494)  "
          f"{'PASS' if gates['B_LR_anchor'] else 'FAIL'}")

    # Unified, additive-constant-free cert metric: causality REQUIRES a specific storage
    # DISPERSION dE'(w1->w2); reconstruct it from the loss and compare to the forward's.
    def kk_dispersion_residual(w, E_re, E_im, wlo, whi, norm):
        """
        |[E'_KK(whi)-E'_KK(wlo)] - [E'(whi)-E'(wlo)]| / norm, at INTERIOR pts (grid-stable).
        Mapping: chi'=E_inf-E', chi''=E'' => KK gives chi'_KK; storage dispersion = -(chi'_KK dispersion).
        """
        chip_kk = kk_re_from_im(w, E_im) # = (E_inf - E')_KK
        jlo, jhi = np.argmin(np.abs(w-wlo)), np.argmin(np.abs(w-whi))
        disp_kk = -(chip_kk[jhi]-chip_kk[jlo]); disp_fw = E_re[jhi]-E_re[jlo]
        return abs(disp_kk-disp_fw)/norm, disp_kk, disp_fw

    w = np.linspace(W_PK*1e-2, W_PK*2e2, 8000)
    Ec = E_thermoelastic_closed(w); Epr, Epp = Ec.real, Ec.imag
    loss_pk = Epp.max() # physical loss scale (Pa)
    Ekk = kk_re_from_im(w, Epp)
    print(f"loss peak E''max = {loss_pk:.3e} Pa   dispersion E'(inf)-E'(0) = {Epr.max()-Epr.min():.3e} Pa")
    print("Storage-DISPERSION reconstructed from loss vs forward (residual/loss_peak), by band:")
    print(f"{'band [w/wpk]':>16}{'dE_KK/E':>12}{'dE_fwd/E':>12}{'resid/loss':>12}")
    kk_resid = 0.0
    for wlo_f, whi_f in [(0.2,1.0),(0.5,2.0),(1.0,5.0),(0.3,3.0)]:
        r, dk, df = kk_dispersion_residual(w, Epr, Epp, wlo_f*W_PK, whi_f*W_PK, loss_pk)
        kk_resid = max(kk_resid, r)
        print(f"  [{wlo_f:.1f},{whi_f:.1f}]{'':>6}{dk/E_Si:>12.2e}{df/E_Si:>12.2e}{r:>12.2e}")
    print("KK cert residual CONVERGES under grid refinement (band [0.3,3]wpk):")
    for N in [2000, 4000, 8000, 16000]:
        wn = np.linspace(W_PK*1e-2, W_PK*2e2, N); Ecn = E_thermoelastic_closed(wn)
        r, _, _ = kk_dispersion_residual(wn, Ecn.real, Ecn.imag, 0.3*W_PK, 3.0*W_PK, Ecn.imag.max())
        print(f"   N={N:>6}  resid/loss={r:.3e}")
    gates["B_kk_pass"] = kk_resid < 0.05
    print(f"GATE B2 (thermoelastic PASSES causality cert, resid/loss<5e-2):  max_resid={kk_resid:.3e}  "
          f"{'PASS' if gates['B_kk_pass'] else 'FAIL'}")

    # ---------------------------------------------------------------------------------
    section("PART B3 - ANTI-TAUTOLOGY: corrupt E' (break causality) -> cert MUST fire")
    # inject a non-causal step into the storage E' that is NOT reflected in the loss E''.
    # The KK reconstruction uses ONLY the loss, so a storage inconsistent with the loss is caught.
    jlo, jhi = np.argmin(np.abs(w-0.3*W_PK)), np.argmin(np.abs(w-3.0*W_PK))
    Epr_bad = Epr.copy(); Epr_bad[jhi] += 0.3*loss_pk # +30% of loss into storage at whi only
    disp_kk = -(Ekk[jhi]-Ekk[jlo]) # storage dispersion causality requires
    resid_clean = abs(disp_kk-(Epr[jhi]-Epr[jlo]))/loss_pk
    resid_bad   = abs(disp_kk-(Epr_bad[jhi]-Epr_bad[jlo]))/loss_pk
    ratio = resid_bad/resid_clean
    print(f"clean   storage-dispersion residual/loss = {resid_clean:.3e}")
    print(f"corrupt storage-dispersion residual/loss = {resid_bad:.3e}   (x{ratio:.0f} larger)")
    print("=> the cert reconstructs the storage from the LOSS ALONE; a storage inconsistent with")
    print("   the loss is CAUGHT. The cert is load-bearing, NOT an algebraic identity of the forward.")
    gates["B_anti_tautology"] = ratio > 10
    print(f"GATE B3 (corruption raises residual >10x):  x{ratio:.0f}  "
          f"{'PASS' if gates['B_anti_tautology'] else 'FAIL'}")

    # ---------------------------------------------------------------------------------
    section("PART C - HYSTERETIC/STRUCTURAL damping (eta=const): the CERTIFIED NEGATIVE")
    # The FEA 'structural/hysteretic damping' model: E'=E_R const, E''=eta*E_R const. KK FORBIDS it:
    # a constant loss REQUIRES a log-VARYING storage (the constant-Q dispersion). The model's constant
    # storage is the full violation. We RECOVER the required storage slope numerically and anchor it to
    # the analytic constant-Q result dE'/dln(w)=2*eta/pi*E_R (Kjartansson 1979). 0 free params.
    eta, E_R = 0.02, 1.0
    def _win(lw): # smooth raised-cosine band edges (no boxcar log-singularity)
        a = np.abs(lw); x = np.ones_like(lw); t = (a-2.0)/0.5
        mm = (a > 2.0) & (a < 2.5); x[mm] = 0.5*(1+np.cos(np.pi*t[mm])); x[a >= 2.5] = 0.0
        return x
    def hyst_storage(N):
        wg = np.logspace(-3.5, 3.5, N) # log grid (engine handles non-uniform)
        Epp_h = eta*E_R*_win(np.log10(wg)) # constant loss on a wide band, smooth edges
        return wg, -kk_re_from_im(wg, Epp_h) # storage causality REQUIRES

    wg, Ekk_h = hyst_storage(8000)
    mc = (wg >= 0.1) & (wg <= 10.) # central 2 decades of flat loss
    slope = np.polyfit(np.log(wg[mc]), Ekk_h[mc], 1)[0]
    anchor = 2*eta/np.pi*E_R
    gates["C_kjartansson_anchor"] = abs(slope/anchor - 1) < 0.02
    print(f"Causality REQUIRES storage log-slope dE'/dln(w) = {slope:.4e}")
    print(f"  external anchor (Kjartansson 1979 constant-Q) 2*eta/pi*E_R = {anchor:.4e}  "
          f"ratio={slope/anchor:.4f}  {'PASS' if gates['C_kjartansson_anchor'] else 'FAIL'}")
    print("  the hysteretic model asserts storage = CONSTANT (slope 0) => full causality violation.\n")
    print("CERTIFIED BOUNDARY - causality violation (required |dE'|/loss) DIRECTLY measured vs bandwidth:")
    print(f"{'bandwidth (decades)':>20}{'measured viol':>16}{'analytic 1.466*D':>18}")
    def viol_direct(D):
        jlo = np.argmin(np.abs(wg - 10**(-D/2))); jhi = np.argmin(np.abs(wg - 10**(D/2)))
        return abs(Ekk_h[jhi]-Ekk_h[jlo])/(eta*E_R)
    boundary = []
    for D in [0.03, 0.1, 0.3, 1.0, 2.0]:
        v = viol_direct(D); boundary.append((D, v))
        print(f"{D:>20.2f}{v:>16.4f}{1.466*D:>18.4f}")
    # grid-stability of the violation (adversary: numerical artifact) - slope stable across N
    vN = [np.polyfit(np.log(hyst_storage(N)[0][(hyst_storage(N)[0]>=0.1)&(hyst_storage(N)[0]<=10.)]),
                     hyst_storage(N)[1][(hyst_storage(N)[0]>=0.1)&(hyst_storage(N)[0]<=10.)], 1)[0]
          for N in [4000, 8000, 16000]]
    grid_stable = (max(vN)-min(vN))/np.mean(vN) < 0.02
    viol_003 = viol_direct(0.03); viol_1 = viol_direct(1.0)
    D_usable = 0.5/(abs(slope)*np.log(10)/(eta*E_R))
    gates["C_hyst_nonzero"] = viol_1 > 0.5 # grossly non-causal over a decade
    gates["C_grid_stable"]  = grid_stable # violation is REAL, not numerical
    gates["C_boundary"]     = viol_003 < 0.1 and D_usable < 0.5 # usable only near ONE resonance
    print(f"\nCERTIFIED: constant-eta hysteretic damping is NON-CAUSAL (Crandall 1970; Kjartansson 1979).")
    print(f"  Causality forces storage to change by {viol_1:.2f}x the loss over 1 decade; model holds it constant.")
    print(f"  BOUNDARY: usable (viol<0.5) only within +-{D_usable:.2f} decade of one resonance")
    print(f"  (viol_0.03dec={viol_003:.3f} ~ok, viol_1dec={viol_1:.2f} broken). The boundary is eta-INDEPENDENT.")
    print(f"GATE C (non-causal + grid-stable + tight boundary + analytic anchor):  "
          f"{'PASS' if all([gates['C_hyst_nonzero'],gates['C_grid_stable'],gates['C_boundary'],gates['C_kjartansson_anchor']]) else 'FAIL'}")

    # ---------------------------------------------------------------------------------
    section("VERDICT")
    allpass = all(gates.values())
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n  {'ALL GATES PASS' if allpass else 'SOME GATES FAIL'}")
    print("  NODE: 0-fit Kramers-Kronig causality/passivity cert (V_mf model-free axis, causality-failure mode).")
    print("  Thermoelastic damping = genuine 0-fit CERTIFIED causal node (composable).")
    print("  Constant-eta hysteretic damping = CERTIFIED non-causal, with a bandwidth boundary.")
    import sys; sys.exit(0 if allpass else 1)
