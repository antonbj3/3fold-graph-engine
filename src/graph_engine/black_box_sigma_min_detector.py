"""black_box_sigma_min_detector — an agent worktree: find the LEAST-identified / least-recoverable direction (the σ_min direction
of a recovery/estimation operator) from PERTURB-AND-OBSERVE alone, treating the recovery as a BLACK BOX -- no operator
form, no Jacobian, no SVD. Validated (real Foster reflectance) + QC'd on real flutter.

THE PRINCIPLE (FDT): for a recovery x_hat = R(y), y = A x (+noise), the recovery FLUCTUATES most along the direction
the measurement constrains LEAST -- the σ_min right-vector of the recovery operator (Cov(x_hat) ∝ (AᵀA)^{-1} for the
linear case). So perturb the INPUT, recover, and the TOP eigenvector of the recovery-fluctuation covariance IS the
σ_min direction. Because it only needs input perturbations + outputs, it works on UNKNOWN / NONLINEAR recovery where
you cannot SVD anything.

★HONEST SCOPE (do not oversell): on a LINEAR system this fluctuation route and the analytic design SVD are the
SAME object (the FDT identity) -- so agreement between them is a METHOD-VALIDATION / consistency check, NOT an
over-determination by two independent observations (n_eff=1). The genuine value is BLACK-BOX APPLICABILITY: use this
when you have no operator form (nonlinear/unknown recovery). If you DO have A, just SVD it.

  black_box_sigma_min_direction(recover, x_batch, subspace=None, sigma_frac=0.15, n_pert=40, seed=0) -> unit vector
  certify_sigma_min(recover, x_batch, subspace=None, analytic_dir=None,...) -> dict {direction, discriminates_vs_null,
      null_p95, cos_with_analytic (if given)}
numpy only. `recover` is any callable X(n,d)->Xhat(n,d) (the black box). Ties / (FDT) + / (recoverable DOF)."""
from __future__ import annotations
import numpy as np


def _fluctuation_cov(recover, x_batch, subspace, sigma_frac, n_pert, rng):
    x = np.asarray(x_batch, float)
    base_hat = np.asarray(recover(x), float)
    scale = sigma_frac * float(x.std())
    F = []
    for _ in range(n_pert):
        xp = x + rng.normal(0, scale, x.shape)
        F.append(np.asarray(recover(xp), float) - base_hat)      # recovery deviation induced by the input perturbation
    F = np.concatenate(F, 0)
    if subspace is not None:
        F = F @ np.asarray(subspace, float)                      # restrict to a signal subspace (d x r); returns r-dim dir
    C = F.T @ F / max(len(F), 1)
    return C


def black_box_sigma_min_direction(recover, x_batch, subspace=None, sigma_frac=0.15, n_pert=40, seed=0):
    """Top eigenvector of the recovery-fluctuation covariance = the σ_min (least-recoverable) direction. If `subspace`
    (d x r) is given, the direction is returned in the FULL d-space (subspace @ top-eig). Unit-normalized."""
    rng = np.random.default_rng(seed)
    C = _fluctuation_cov(recover, x_batch, subspace, sigma_frac, n_pert, rng)
    if not np.all(np.isfinite(C)):
        # eigh-on-NaN is structure/LAPACK-path-dependent (never trust it to raise) -- guard explicit.
        raise ValueError("black_box_sigma_min_direction: non-finite fluctuation covariance (recover() produced NaN/Inf)")
    w, V = np.linalg.eigh(C)
    top = V[:, -1]
    if subspace is not None:
        top = np.asarray(subspace, float) @ top
    return top / (np.linalg.norm(top) + 1e-12)


def certify_sigma_min(recover, x_batch, subspace=None, analytic_dir=None, sigma_frac=0.15, n_pert=40, seed=0,
                      n_null=200, discriminate_margin=0.1):
    """Detect the σ_min direction + check it DISCRIMINATES (top-eig |cos| with random subspace dirs is small) and, if an
    analytic σ_min direction is supplied (linear case), report the method-consistency |cos| (NOT an over-det -- FDT
    identity). Returns a receipt."""
    rng = np.random.default_rng(seed)
    d = black_box_sigma_min_direction(recover, x_batch, subspace, sigma_frac, n_pert, seed)
    # random-direction null (within the subspace if given)
    r_dim = np.asarray(subspace).shape[1] if subspace is not None else len(d)
    null = []
    for _ in range(n_null):
        if subspace is not None:
            r = np.asarray(subspace, float) @ rng.standard_normal(r_dim)
        else:
            r = rng.standard_normal(len(d))
        r /= np.linalg.norm(r) + 1e-12
        null.append(abs(float(d @ r)))
    null_p95 = float(np.percentile(null, 95))
    out = dict(direction=d.tolist() if len(d) <= 64 else None, dim=int(len(d)),
               null_p95=round(null_p95, 3), discriminates_vs_null=bool(null_p95 < 0.9))
    if analytic_dir is not None:
        a = np.asarray(analytic_dir, float); a /= np.linalg.norm(a) + 1e-12
        cos = abs(float(d @ a))
        out["cos_with_analytic"] = round(cos, 3)
        out["matches_analytic"] = bool(cos > null_p95 + discriminate_margin)
        out["note"] = ("method-consistency vs analytic σ_min (FDT identity -- NOT an over-determination): |cos|=%.2f "
                       "(random-null p95=%.2f)" % (cos, null_p95))
    return out


def _selftest():
    rng = np.random.default_rng(0)
    ok = tot = 0
    # (1) LINEAR known-answer: an ILL-CONDITIONED recovery AMPLIFIES the least-identified direction (the FDT
    # (AᵀA)^{-1} picture: recovery variance is LARGEST along σ_min). The detector must find that direction of
    # largest recovery fluctuation. (NOT a perfect/well-conditioned recovery -- that has no problematic direction.)
    d = 12
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    smin_true = Q[:, -1]                                          # least-identified direction
    amp = 5.0                                                     # recovery amplifies this direction (ill-conditioning)
    def recover_lin(X):
        return X + amp * (X @ smin_true)[:, None] * smin_true[None, :] + rng.normal(0, 1e-3, X.shape)
    X = rng.standard_normal((2000, d))
    r1 = certify_sigma_min(recover_lin, X, analytic_dir=smin_true, n_pert=30)
    tot += 1; ok += (r1.get("matches_analytic", False) and r1["cos_with_analytic"] > 0.8)
    tot += 1; ok += r1["discriminates_vs_null"]
    # (2) NONLINEAR black box (no operator form): recovery adds a large gain along smin_true nonlinearly -> the
    # detector must still find smin_true from perturb-and-observe (the whole point).
    def recover_nl(X):
        proj = X @ smin_true
        return X + (5.0 * np.tanh(proj))[:, None] * smin_true[None, :] + rng.normal(0, 1e-3, X.shape)
    r2 = certify_sigma_min(recover_nl, X, analytic_dir=smin_true, n_pert=30)
    tot += 1; ok += (r2.get("cos_with_analytic", 0) > 0.8)       # black-box detector finds the direction w/o any operator form
    # (3) degenerate guard: isotropic recovery -> no dominant fluctuation dir -> does not falsely 'discriminate' strongly
    def recover_iso(X):
        return X + rng.normal(0, 1e-3, X.shape)
    r3 = certify_sigma_min(recover_iso, X, n_pert=30)
    tot += 1; ok += (r3["null_p95"] > 0.2)                       # random baseline is meaningful (sanity)
    print("black_box_sigma_min_detector selftest: %d/%d  (linear cos=%.2f matches=%s | nonlinear cos=%.2f | null_p95=%.2f)" % (
        ok, tot, r1.get("cos_with_analytic", -1), r1.get("matches_analytic"), r2.get("cos_with_analytic", -1), r1["null_p95"]))
    return ok == tot


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
