 #!/usr/bin/env python3
"""kk_cert_lib — Kramers-Kronig CAUSALITY CERT for REAL optical (n,k) dispersion data.

Reuses (imports, does NOT reimplement) the validated KK engine `kk_re_from_im` from
`scripts/kramers_kronig_passivity_cert.py` (committed afb3eadd3, grid-convergent, externally
anchored on Debye/Lifshitz-Roukes/Kjartansson). See kk_cert_PREREG.md for the full derivation,
frozen procedure, controls and thresholds -- this module implements exactly that spec.

Optical KK relation used: n(E)-1 = (2/pi) P int_0^inf E' kappa(E') / (E'^2-E^2) dE'  (E = photon
energy in eV, NOT omega in rad/s -- the KK integral is invariant under any positive linear
rescaling of the frequency variable, so eV is a legitimate, better-conditioned unit choice,
standard in the optical-constants literature). This is algebraically IDENTICAL in form to the
existing engine's chi'(w)=(2/pi) P int W chi''(W)/(W^2-w^2) dW with chi_im -> kappa.
"""
import os
import sys
import numpy as np
import yaml

HC_EV_UM = 1.2398419843320025 # eV*um (CODATA hc = 1239.8419843320025 eV*nm), physical constant

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = _THIS_DIR
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from kramers_kronig_passivity_cert import kk_re_from_im # noqa: E402  (reused, not reimplemented)


# =====================================================================================
# DATA LOADING (real refractiveindex.info-database tables, cached in kk_cert_data/)
# =====================================================================================
def load_real_instance(path, lam_range_um=None):
    """
    Parse a refractiveindex.info 'tabulated nk' YAML -> (E_eV asc, n, k, meta). Real data only
    (asserts type=='tabulated nk', rejects formula-only entries like Malitson's Sellmeier n(lambda)
    which carries no k channel and is therefore untestable for KK).
    """
    with open(path) as f:
        d = yaml.safe_load(f)
    entry = d["DATA"][0]
    assert entry["type"] == "tabulated nk", f"{path}: not real tabulated (n,k) data (type={entry.get('type')})"
    rows = np.array([[float(x) for x in line.split()] for line in entry["data"].strip().split("\n")])
    lam, n, k = rows[:, 0], rows[:, 1], rows[:, 2]
    order = np.argsort(lam)
    lam, n, k = lam[order], n[order], k[order]
    if lam_range_um is not None:
        m = (lam >= lam_range_um[0]) & (lam <= lam_range_um[1])
        lam, n, k = lam[m], n[m], k[m]
    E = HC_EV_UM / lam
    order2 = np.argsort(E)
    E, n, k = E[order2], n[order2], k[order2]
    keep = np.concatenate([[True], np.diff(E) > 0]) # dedupe: kk_re_from_im/CubicSpline need strictly increasing E
    E, n, k = E[keep], n[keep], k[keep]
    meta = dict(path=path, reference=str(d.get("REFERENCES", ""))[:200].replace("\n", " "),
                n_pts=len(E), lam_range_um=(float(lam.min()), float(lam.max())),
                E_range_eV=(float(E.min()), float(E.max())))
    return E, n, k, meta


# =====================================================================================
# KK PREDICTIONS (naive = reused engine directly; subtractive = derived, zero new numerics
# -- see PREREG Sec.0: algebraically KK[k](E) - KK[k](E0), by partial-fraction linearity)
# =====================================================================================
def anchor_index_logmedian(E):
    """Frozen, non-cherry-picked anchor rule: sample nearest the log-median of the band."""
    target = np.exp(0.5 * (np.log(E[0]) + np.log(E[-1])))
    idx = int(np.argmin(np.abs(E - target)))
    return int(np.clip(idx, 1, len(E) - 2)) # avoid the NaN endpoints of kk_re_from_im


def naive_predict(E, k):
    """n_naive(E) = 1 + KK[kappa](E).  Direct reuse of the existing engine, unmodified."""
    return 1.0 + kk_re_from_im(E, k)


def subtractive_predict(E, k, n_meas, anchor_idx=None):
    """
    Singly-subtractive (anchored) KK prediction, derived as KK[k](E)-KK[k](E0)+n_meas(E0).
    Returns (n_sub, anchor_idx, nonsub_array) -- nonsub reused so callers don't recompute.
    """
    nonsub = kk_re_from_im(E, k)
    if anchor_idx is None:
        anchor_idx = anchor_index_logmedian(E)
    n_sub = n_meas[anchor_idx] + nonsub - nonsub[anchor_idx]
    return n_sub, anchor_idx, nonsub


def interior_mask(E, frac_lo=0.2, frac_hi=0.8):
    """
    Central 60% of the band in log-E space -- excludes known PV-quadrature edge artifacts
    (same convention this repo's own engine validation already uses, e.g. dropping outer bands
    in kramers_kronig_passivity_cert.py's Debye convergence check).
    """
    lo, hi = np.log(E[0]), np.log(E[-1])
    elo, ehi = np.exp(lo + frac_lo * (hi - lo)), np.exp(lo + frac_hi * (hi - lo))
    return (E >= elo) & (E <= ehi)


def relative_residual(n_pred, n_meas, mask):
    """
    median |n_pred-n_meas| over mask, normalized by n_meas's OWN full-band dispersion range
    (self-normalizing -- no external per-point uncertainty numbers required; PREREG Sec.8).
    """
    m = mask & np.isfinite(n_pred) & np.isfinite(n_meas)
    rng = np.max(n_meas) - np.min(n_meas)
    if rng <= 0 or not np.any(m):
        return float("nan")
    return float(np.median(np.abs(n_pred[m] - n_meas[m])) / rng)


# =====================================================================================
# SYNTHETIC CAUSAL GROUND TRUTH (generic 3-oscillator dielectric, PREREG Sec.4 C1/C2/C3 -- NOT
# fit to any specific real material; illustrative dielectric-scale parameters only)
# =====================================================================================
OSC_PARAMS_DEFAULT = [(0.12, 0.02, 0.02),   # (E_j eV, gamma_j eV, f_j) IR/phonon-like
                      (2.5, 0.4, 1.0), # visible/near-UV electronic-like
                      (11.0, 3.0, 6.0)] # deep-UV band-gap-like (~SiO2 gap scale, illustrative)


def lorentz_eps(E, params=OSC_PARAMS_DEFAULT, acausal=False):
    """
    Multi-Lorentz-oscillator dielectric function. acausal=True flips gamma_j -> -gamma_j for
    ALL oscillators (poles moved to the acausal half-plane, magnitude-preserving pole-sign-flip
    -- the same construction used by the reference dispersion model).
    """
    eps = np.ones_like(E, dtype=complex)
    for (Ej, gj, fj) in params:
        g = -gj if acausal else gj
        eps = eps + fj * Ej**2 / (Ej**2 - E**2 - 1j * g * E)
    return eps


def oscillator_nk(E, params=OSC_PARAMS_DEFAULT, acausal=False):
    nk = np.sqrt(lorentz_eps(E, params, acausal))
    return nk.real, nk.imag


def make_log_grid(emin, emax, n):
    return np.exp(np.linspace(np.log(emin), np.log(emax), n))


# --- metal-appropriate ground truth (OODA fix: the generic 3-oscillator model above has NO
# free-electron/Drude term, so it is the WRONG physical class for testing a metal's truncation
# floor -- real gold's k rises sharply below ~0.6eV, just below the measured band's own lower
# edge, contributing heavily to the KK integral via the missing IR/Drude tail. Literature-typical
# gold Drude parameters (NOT fit to this cert's residual): Ep~9.0eV plasma energy, gamma_D~0.07eV
# relaxation (Ordal et al. 1985 / Rakic et al. 1998 scale), plus one illustrative interband
# oscillator near gold's known ~2.4eV onset -- used ONLY as a diagnostic to test whether the
# metal's elevated real/floor ratio is explained by the model class, not to "fix" a number.)
DRUDE_AU = dict(Ep=9.03, gamma_D=0.071)
INTERBAND_AU = [(2.4, 1.0, 1.0)]


def metal_nk(E, drude=DRUDE_AU, interband=INTERBAND_AU, acausal=False):
    gD = -drude["gamma_D"] if acausal else drude["gamma_D"]
    eps = np.ones_like(E, dtype=complex) - drude["Ep"] ** 2 / (E ** 2 + 1j * gD * E)
    for (Ej, gj, fj) in interband:
        g = -gj if acausal else gj
        eps = eps + fj * Ej ** 2 / (Ej ** 2 - E ** 2 - 1j * g * E)
    nk = np.sqrt(eps)
    return nk.real, nk.imag


def bootstrap_baseline_residuals(E, k, n, mask, n_boot=25, drop_frac=0.15, seed=0):
    """
    WITHIN-INSTANCE null: how much does rho wobble under pure finite-sample resampling alone
    (no k-edit at all)? Fixes the cross-instance leave-one-out contamination (an instance with a
    large intrinsic baseline, e.g. gold's missing-Drude-tail residual, would trivially 'detect'
    any perturbation under an absolute cross-instance threshold -- see kk_cert_run.py Part E).
    """
    rng = np.random.default_rng(seed)
    N = len(E)
    vals = []
    for _ in range(n_boot):
        keep = np.ones(N, dtype=bool)
        n_drop = max(1, int(drop_frac * N))
        drop_idx = rng.choice(N, size=n_drop, replace=False)
        keep[drop_idx] = False
        keep[0] = True
        keep[-1] = True
        if keep.sum() < 10:
            continue
        Eb, kb, nb, mb = E[keep], k[keep], n[keep], mask[keep]
        try:
            n_sub_b, _, _ = subtractive_predict(Eb, kb, nb)
            r = relative_residual(n_sub_b, nb, mb)
            if np.isfinite(r):
                vals.append(r)
        except Exception:
            continue
    return np.array(vals)


# =====================================================================================
# NON-CAUSAL PERTURBATION INJECTION (PREREG Sec.5)
# =====================================================================================
def inject_bump(E, k, pctile=0.25, width_log_frac=0.05, amplitude=1.0):
    """
    kappa_pert(E) = kappa(E) + A * kappa(E_c) * exp(-(lnE-lnEc)^2 / (2 sigma_log^2)).
    E_c fixed at the given log-percentile of the band (default 25th, DIFFERENT from the anchor's
    50th so the perturbation test doesn't interact with the subtractive anchor).
    """
    logE = np.log(E)
    lo, hi = logE[0], logE[-1]
    logEc = lo + pctile * (hi - lo)
    Ec = float(np.exp(logEc))
    kc = float(np.interp(Ec, E, k))
    sigma = width_log_frac * (hi - lo)
    bump = amplitude * kc * np.exp(-(logE - logEc) ** 2 / (2 * sigma ** 2))
    return k + bump, Ec, kc


# =====================================================================================
# MISC
# =====================================================================================
def section(title):
    print("\n" + "=" * 92 + f"\n{title}\n" + "=" * 92)
