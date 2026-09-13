#!/usr/bin/env python3
"""GENERATIVE+INVERSE SERVICE (the spine, T2/T3) — one forward-model-agnostic engine the verticals call.

The convergence-map spine as a REUSABLE service (NIGHT_BACKLOG T2/T3): a single engine that, given ANY validated
forward model f: x→observables, does both jobs by changing only the objective —
  • generative(f, objective, x0, bounds)              → extremize a performance objective (design)
  • inverse(f, observed, x0, bounds) → (x̂, Σ, resid)  → match an observed target, with a σ-BOUNDED region (Fisher
                                                          covariance) + over-determination identifiability.
The physics (f) is FIXED & validated; only the INPUTS x are optimized (the right side of the fit-trap). A vertical
imports these two functions and passes its own forward model + its patent's quantified claims — build once, all jobs.

★Forward-model-AGNOSTIC: f is an arbitrary callable; gradients are taken by finite difference on the composed function,
so no per-physics adjoint is required to USE the service (a validated analytic adjoint just makes it faster/exacter).

GATES (the self-test; each gate has a null/control):
 (G0) GENERATIVE SERVICE — generative extremizes a beam's specific resonance to the bound-extremal design.
 (G1) INVERSE SERVICE + σ-REGION — inverse recovers a beam's true (w,t) from observed claims to < 1% and returns a
      Fisher covariance whose σ matches Monte-Carlo.
 (G2) FORWARD-AGNOSTIC ("any validated physics") — the SAME inverse recovers (v₀,θ) from projectile observables
      (range, apex, flight-time) — a totally different forward model, no code change. NULL = a 1-observable inverse of
      an unidentifiable param fails.
 (G3) OVER-DETERMINATION = IDENTIFIABILITY — adding a 3rd independent claim shrinks the recovered-parameter σ vs 2
      claims (the patent-reverse-engineering identifiability the verticals need).
 (G4) ILL-POSED → REGULARIZED (T7) — when the claims under-determine the inputs (rank-deficient Fisher), a prior
      (Tikhonov/Bayesian) bounds the null direction → a finite σ-region + a recovery that still meets the claim.
 (G5) MULTI-OBJECTIVE (Pareto) — generative_pareto traces the stiffness-vs-lightness Pareto front; the verticals
      design with real tradeoffs (≥2 competing objectives), not a single scalar.
 (G6) IDENTIFIABILITY ('I' grade) — identifiability pre-scores a claim-set's reverse-engineerability from the
      Jacobian alone; flags under-determined (1 claim, 2 params) AND dependent claims (apex+time only pin v₀sinθ).

Run: python3 e.py
"""
import sys
import numpy as np
from scipy.optimize import least_squares, minimize


# ───────────────────────────── THE SERVICE (importable by the verticals) ─────────────────────────────
def _fd_jac(forward, x, h=1e-7, diff_step=None):
    x = np.asarray(x, float); y0 = np.atleast_1d(forward(x)); J = np.zeros((len(y0), len(x)))
    for i in range(len(x)):
        s = (diff_step * max(abs(x[i]), 1e-12)) if diff_step else h   # diff_step ⇒ RELATIVE step (~ε^⅓) for float32-noisy forwards
        dx = np.zeros(len(x)); dx[i] = s
        J[:, i] = (np.atleast_1d(forward(x + dx)) - np.atleast_1d(forward(x - dx))) / (2 * s)
    return J


def _fd_grad(scalar_f, x, diff_step):
    g = np.zeros(len(x))
    for i in range(len(x)):
        s = diff_step * max(abs(x[i]), 1e-12)                        # RELATIVE step (~ε⅓) for float32-noisy forwards
        dx = np.zeros(len(x)); dx[i] = s
        g[i] = (scalar_f(x + dx) - scalar_f(x - dx)) / (2 * s)
    return g


def generative(forward, objective, x0, bounds, iters=None, diff_step=None, n_starts=1, **kw):
    """Extremize objective(forward(x)) over the FIXED forward model (L-BFGS-B, bound-constrained, FD gradient).
    objective: observables→scalar (MAXIMIZED). Robust + fast for the verticals' repeated calls on slow forwards.
    diff_step: enlarge the FD gradient step (~ε⅓, RELATIVE) for float32-noisy forwards — the default L-BFGS-B step is
    too fine on float32 and the optimizer stalls at x0 (same failure class as the inverse-side diff_step).
    n_starts>1 ⇒ MULTI-START global opt: solve from x0 + (n_starts-1) quasi-random starts over the bounds and keep the
    BEST — guards a NON-CONVEX design landscape where a single local solve lands in a wrong local optimum."""
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    neg = lambda x: -float(objective(np.atleast_1d(forward(x))))
    jac = (lambda x: _fd_grad(neg, np.asarray(x, float), diff_step)) if diff_step else None
    opt = {"maxfun": int(iters)} if iters else {}
    solve = lambda xi: minimize(neg, np.asarray(xi, float), method="L-BFGS-B", bounds=list(zip(lo, hi)), jac=jac, options=opt).x
    starts = [np.asarray(x0, float)] + [lo + np.random.default_rng(s).random(len(lo)) * (hi - lo) for s in range(1, n_starts)]
    return min((solve(xi) for xi in starts), key=neg)


def generative_pareto(forward, obj_a, obj_b, x0, bounds, n=7, **kw):
    """Multi-objective generative: trace the Pareto front of two competing MAXIMIZED objectives by a weighted-sum sweep,
    auto-normalized by the single-objective extremes. Returns [(weight, x*, obj_a(x*), obj_b(x*)),...]."""
    xa = generative(forward, obj_a, x0, bounds, **kw); xb = generative(forward, obj_b, x0, bounds, **kw)
    a_hi, a_lo = obj_a(forward(xa)), obj_a(forward(xb)); b_hi, b_lo = obj_b(forward(xb)), obj_b(forward(xa))
    na = lambda y: (obj_a(y) - a_lo) / (a_hi - a_lo + 1e-30); nb = lambda y: (obj_b(y) - b_lo) / (b_hi - b_lo + 1e-30)
    front = []
    for lam in np.linspace(0.0, 1.0, n):
        x = generative(forward, lambda y: (1 - lam) * na(y) + lam * nb(y), x0, bounds, **kw)
        y = forward(x); front.append((float(lam), x, float(obj_a(y)), float(obj_b(y))))
    return front


def generative_robust(forward_xi, objective, x0, bounds, xi_samples, mode="worstcase", diff_step=None):
    """ROBUST design — the σ-aware GENERATIVE half (complements the σ-aware inverse). The nominal generative optimizes for
    ONE condition; real designs face UNCERTAIN loads/materials/tolerances and a nominal-optimum can be FRAGILE. Here
    forward_xi(x, ξ) takes the design x AND an uncertain condition ξ; this maximizes the ROBUST objective over the ξ
    samples — mode='worstcase' (max min_ξ J, conservative) or 'expected' (max mean_ξ J). Returns the robust design x*."""
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)

    def negJ(x):
        Js = [float(objective(np.atleast_1d(forward_xi(x, xi)))) for xi in xi_samples]
        return -(float(np.mean(Js)) if mode == "expected" else float(np.min(Js)))
    jac = (lambda x: _fd_grad(negJ, np.asarray(x, float), diff_step)) if diff_step else None
    return minimize(negJ, np.asarray(x0, float), method="L-BFGS-B", bounds=list(zip(lo, hi)), jac=jac).x


def generative_constrained(forward, objective, constraints, x0, bounds, iters=None, diff_step=None):
    """HARD-constrained generative — maximize objective(forward(x)) subject to nonlinear constraints g(x) ≤ 0, via SLSQP.
    Real design specs are hard limits (stress ≤ yield, mass ≤ budget), NOT penalties: a penalty only approximately
    satisfies the limit, which is unsafe for a safety constraint. `constraints` = a list of callables g(x) (feasible
    when g(x) ≤ 0; each may return a scalar or a vector). Returns the best FEASIBLE design x*. diff_step sets the SLSQP
    FD step (eps) for float32-noisy forwards. Complements generative (box bounds) / _pareto / _robust."""
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    x0 = np.asarray(x0, float)
    span = np.where(hi > lo, hi - lo, 1.0)                           # SCALE to an O(1) problem (else SLSQP stalls on
    to_phys = lambda u: lo + np.asarray(u, float) * span             #   physical scales like w~1e-2, stiffness~1e4, stress~1e9)
    f0 = abs(float(objective(np.atleast_1d(forward(x0))))) or 1.0
    neg = lambda u: -float(objective(np.atleast_1d(forward(to_phys(u))))) / f0

    def mk_con(g):                                                   # SLSQP 'ineq': fun ≥ 0 ⟺ g(x) ≤ 0, each scaled by |g(x0)|
        s = float(np.max(np.abs(np.atleast_1d(np.asarray(g(x0), float))))) or 1.0
        return {"type": "ineq", "fun": (lambda u, g=g, s=s: -np.atleast_1d(np.asarray(g(to_phys(u)), float)) / s)}
    cons = [mk_con(g) for g in constraints]
    opts = {} if iters is None else {"maxiter": int(iters)}
    if diff_step:
        opts["eps"] = diff_step
    res = minimize(neg, (x0 - lo) / span, method="SLSQP", bounds=[(0.0, 1.0)] * len(lo), constraints=cons, options=opts)
    return to_phys(res.x)


def inverse(forward, observed, x0, bounds, noise_rel=0.02, use=None, x_prior=None, prior_sigma=None, iters=None, diff_step=None, robust=False, n_starts=1, **kw):
    """Recover x s.t. forward(x) matches `observed`. Returns (x̂, Σ_x, residual). `use` selects observable indices
    (over-determination); Σ_x is the σ-bounded posterior covariance. For ILL-POSED (under-determined) inverses, pass a
    prior (x_prior, prior_sigma rel.) → Tikhonov/Bayesian regularization bounds the null direction (NIGHT_BACKLOG T7):
        Σ_x = (Ĵᵀ Ĵ / noise_rel²  +  diag(1/(prior_sigma·x_prior)²))⁻¹   (prior=None ⇒ plain Fisher noise_rel²(ĴᵀĴ)⁻¹).
    robust=True ⇒ soft-L1 (robust) loss so an OUTLIER observation (a mis-transcribed claim, a bad sensor) doesn't derail
    the L2 fit; the per-observation residuals then flag WHICH observation is the outlier. (Σ_x is the inlier-curvature
    Gauss-Newton estimate; honest under robust loss.)
    n_starts>1 ⇒ MULTI-START: solve from x0 + (n_starts-1) quasi-random starts, keep the lowest-residual fit — guards a
    MULTIMODAL/non-convex inverse (two distinct x give similar observations) where one start lands on the wrong branch.
    NOISE MODEL: noise_rel may be a SCALAR or a per-observable VECTOR (heteroscedastic — down-weights imprecise
    observables, correct σ); pass noise_cov=Σ (relative-error covariance) for CORRELATED noise → whitened by Σ^(-1/2),
    so the σ accounts for the correlation a naive independent-noise σ gets wrong."""
    observed = np.atleast_1d(np.asarray(observed, float)); idx = np.arange(len(observed)) if use is None else np.asarray(use)
    reg = (x_prior is not None and prior_sigma is not None); xp = np.asarray(x_prior, float) if reg else None
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    noise_cov = kw.get("noise_cov")                                  # FULL noise model: scalar/vector noise_rel (independent,
    if noise_cov is not None:                                        # per-observable) OR a relative-error COVARIANCE (correlated)
        noise_cov = np.asarray(noise_cov, float)
        # eigh+maximum 2-primitive fail-open (C/D agent pool class, this tick): eigh silently returns NaN eigenvalues on a
        # NaN-corrupted Σ, and np.maximum(ev, 1e-12*np.max(ev)) does NOT sanitize NaN (np.max(ev) is itself NaN,
        # poisoning the WHOLE floor, not just the corrupted entry) -- W would silently corrupt every downstream fit.
        if not np.all(np.isfinite(noise_cov)):
            raise ValueError("noise_cov contains non-finite values -- the whitener would be corrupt")
        ev, V = np.linalg.eigh(noise_cov)
        # /D agent pool class (this tick): eigh can return FINITE ev with NaN-corrupted V even on clean input
        # (LAPACK-path-dependent) -- W=V@diag(...)@V.T would silently poison the whitener via V alone.
        if not np.all(np.isfinite(ev)) or not np.all(np.isfinite(V)):
            raise ValueError("eigh returned non-finite eigenvalues/eigenvectors -- corrupt Sigma, cannot build the whitener")
        ev = np.maximum(ev, 1e-12 * float(np.max(ev)))               # PSD FLOOR: clip eigenvalues so a near-singular/non-PSD Σ
        W = V @ np.diag(1.0 / np.sqrt(ev)) @ V.T                      #   doesn't make the whitener Σ^(-1/2) NaN/blow up

    ref = float(np.sqrt(np.mean(np.asarray(observed[idx], float) ** 2))) or 1.0   # observable RMS scale
    denom = np.abs(np.asarray(observed[idx], float)).copy()
    denom[denom == 0.0] = kw.get("obs_floor", ref)                    # floor ONLY exactly-zero observables (no /0); every NON-zero
    def resfun(x):                                                    # observable is untouched at ANY magnitude → fully backward-compatible
        r_raw = (np.atleast_1d(forward(x))[idx] - observed[idx]) / denom          # = forward/observed−1 when |observed|≫floor (backward-compatible)
        r = (W @ r_raw) if noise_cov is not None else (r_raw / noise_rel)   # correlated ⇒ Σ^(-1/2)·r; else element-wise /noise_rel
        return np.concatenate([r, (np.asarray(x) - xp) / (prior_sigma * xp)]) if reg else r
    solve = lambda xi: least_squares(resfun, np.asarray(xi, float), bounds=(lo, hi), method="trf",
                        max_nfev=(int(iters) if iters else None), diff_step=diff_step,    # diff_step: enlarge FD for float32-noisy forwards
                        loss=("soft_l1" if robust else "linear"), f_scale=3.0)            # robust: soft-L1 down-weights outlier residuals (>~3σ)
    starts = [np.asarray(x0, float)] + [lo + np.random.default_rng(s).random(len(lo)) * (hi - lo) for s in range(1, n_starts)]
    sol = min((solve(xi) for xi in starts), key=lambda s: s.cost)     # n_starts>1 ⇒ multi-start: keep the lowest-residual fit (multimodal/non-convex inverse)
    xhat = sol.x; JtJ = sol.jac.T @ sol.jac
    cov = np.linalg.inv(JtJ + 1e-12 * max(float(np.trace(JtJ)), 1e-30) / len(xhat) * np.eye(len(xhat)))  # ridge → robust to degenerate J
    if kw.get("inflate_lof") and noise_cov is None:                  # OPT-IN LACK-OF-FIT: inflate Σ_x by the reduced-χ² so the σ
        rw = np.atleast_1d(resfun(xhat))[:len(idx)]                  #   stays honest when the MODEL misfits the data (residuals ≫ noise_rel)
        chi2_red = float(np.sum(rw ** 2)) / max(len(idx) - len(xhat), 1)   #   — the Fisher σ alone trusts noise_rel and is then overconfident
        cov = cov * max(chi2_red, 1.0)
    resid = float(np.sum(((np.atleast_1d(forward(xhat))[idx] - observed[idx]) / denom) ** 2))
    return xhat, cov, resid


def predict_holdout(forward, observed, x0, bounds, train, test, noise_rel=0.02, k=3.0, **kw):
    """PATENT-EXAM / hold-out cross-validation — the FALSIFICATION proof. Recover x from the TRAIN observables, then
    PREDICT the held-out TEST observables and check they match within k·σ. A recovery is validated NOT by fitting (any
    model fits its own training data) but by predicting WITHHELD data: if the FIXED forward + the train-recovered x
    predicts the test observables within the combined (measurement + recovery) σ, the model passes the exam; if not, the
    model/recovery is FALSIFIED. This is exactly the patent-as-falsifiable-exam discipline — never fit the held-out claim,
    predict it. Returns dict(x_hat, pred, obs_test, rel_err, sigma_pred, passed). Honest σ: σ_pred linearized from Σ_x."""
    observed = np.atleast_1d(np.asarray(observed, float)); train = np.asarray(train); test = np.asarray(test)
    xhat, cov, _ = inverse(forward, observed, x0, bounds, noise_rel=noise_rel, use=train, **kw)
    pred = np.atleast_1d(forward(xhat))[test]; obs_test = observed[test]
    Jt = _fd_jac(lambda z: np.atleast_1d(forward(z))[test], xhat, diff_step=kw.get("diff_step"))
    sigma_pred = np.sqrt(np.clip(np.diag(Jt @ cov @ Jt.T), 0.0, None))    # propagate Σ_x → prediction σ (linearized)
    rel_err = np.abs(pred / obs_test - 1.0)
    tol = k * np.sqrt(noise_rel ** 2 + (sigma_pred / np.abs(obs_test)) ** 2)   # combined measurement + recovery uncertainty
    return {"x_hat": xhat, "pred": pred, "obs_test": obs_test, "rel_err": rel_err,
            "sigma_pred": sigma_pred, "passed": bool(np.all(rel_err < tol))}


def propagate_claim_sigma(claim_fn, x_hat, cov, diff_step=None):
    """Propagate the recovered-parameter covariance Σ_x to a DERIVED design CLAIM g(x) (delta method / first-order). A
    vertical recovers x via inverse then states a claim g(x̂) (a predicted failure load, lifetime, performance metric) —
    its σ MUST carry the recovery uncertainty, else the quoted claim is OVERCONFIDENT. Returns (g(x̂), claim_cov) where
    claim_cov = Jg Σ_x Jgᵀ with Jg = ∂g/∂x at x̂; for a scalar claim, √claim_cov is the claim's σ."""
    x_hat = np.asarray(x_hat, float)
    g0 = np.atleast_1d(np.asarray(claim_fn(x_hat), float))
    Jg = _fd_jac(claim_fn, x_hat, diff_step=diff_step)               # ∂g/∂x at the recovered point (n_claim × n_param)
    claim_cov = Jg @ np.asarray(cov, float) @ Jg.T
    return g0, claim_cov


def identifiability(forward, x0, use=None, noise_rel=0.02, diff_step=None):
    """Pre-score a claim-set's reverse-engineerability (the patent-corpus 'I' grade) from the forward's Jacobian ALONE
    — no observed values needed, so the demand/vertical lanes can rank inverse targets before having the patent's
    numbers. More rigorous than counting claims: it catches DEPENDENT claims (a redundant claim adds no Fisher info).
    Returns {condition (Fisher λmax/λmin), rel_sigma (per-param 1σ/|x|; ∞ ⇒ unidentifiable), identifiable (mask),
    n_claims, n_params}."""
    x0 = np.asarray(x0, float); y0 = np.atleast_1d(forward(x0))
    idx = np.arange(len(y0)) if use is None else np.asarray(use)
    Jhat = _fd_jac(forward, x0, diff_step=diff_step)[idx] / y0[idx, None]    # relative Jacobian d(obs/obs₀)/dx
    F = Jhat.T @ Jhat / noise_rel ** 2                                # Fisher information of the claim-set
    ev = np.linalg.eigvalsh(F); cond = float(ev.max() / max(ev.min(), 1e-30))
    n = len(x0)
    Jr = Jhat * np.abs(x0)[None, :]                                  # relative-PARAMETER Jacobian → meaningful eigen-directions
    sig_dir = 1.0 / np.sqrt(np.maximum(np.linalg.eigvalsh(Jr.T @ Jr / noise_rel ** 2), 1e-300))  # σ per EIGEN-DIRECTION (ascending λ)
    well = int(np.sum(sig_dir < 0.5))                                # well-conditioned directions — catches a SLOPPY (small-λ) direction
    status = ("over-determined" if len(idx) > n else "determined") if well >= n else "under-determined"  # too, not just exactly-flat (rank)
    eps = 1e-10 * max(float(ev.max()), 1e-30)                        # ridge → DETERMINED params keep finite σ, only the
    rel_sigma = np.sqrt(np.maximum(np.diag(np.linalg.inv(F + eps * np.eye(n))), 0.0)) / np.abs(x0)  # null-space blows up
    return {"condition": cond, "rel_sigma": rel_sigma, "identifiable": rel_sigma < 0.5,
            "n_claims": int(len(idx)), "n_params": int(n), "n_independent": well, "deficiency": n - well,
            "sigma_per_direction": sig_dir, "status": status}       # status: Jacobian-ONLY over/under verdict (sloppy-aware, no patent numbers)


def inverse_classified(forward, observed, x0, bounds, noise_rel=0.02, use=None, diff_step=None, sigma_tol=0.5, **kw):
    """RIGOROUS over/under-determined classification by the Fisher EIGEN-SPECTRUM + REFUSE a point when under-determined
    — the linchpin against scope error (a miscount → a spurious recovered value). ★Rank alone LIES (per D): it catches a
    dependent claim (exactly-flat direction, λ=0) but NOT a SLOPPY direction (λ small-but-nonzero ⇒ the variable is
    unrecoverable DESPITE full rank and M≥N). The truth is the sensitivity eigen-spectrum of the Fisher F=JᵀJ in RELATIVE
    parameter space: each eigen-direction iᵢ carries σᵢ = 1/√λᵢ (its relative recovery uncertainty). A point is returned
    ONLY if ALL N directions are well-conditioned (σᵢ < sigma_tol); else a REGION along the FLAT/sloppy directions.
    Returns {status, point_trustworthy, x_hat, cov, n_unknowns, n_constraints (=well-conditioned dirs), deficiency
    (=flat dirs), sigma_per_direction (σ per eigen-direction, ascending λ), eigen_directions, flat_basis, feasible_extent}."""
    x0 = np.asarray(x0, float); n = len(x0)
    observed = np.atleast_1d(np.asarray(observed, float))
    idx = np.arange(len(observed)) if use is None else np.asarray(use)
    xhat, cov, _ = inverse(forward, observed, x0, bounds, noise_rel=noise_rel, use=use, diff_step=diff_step, **kw)
    y = np.atleast_1d(forward(xhat)); ref = float(np.sqrt(np.mean(np.asarray(y[idx], float) ** 2))) or 1.0
    yden = np.where(np.abs(y[idx]) > 0, np.abs(y[idx]), ref)          # relative-obs denom (floored at exactly-zero)
    xsc = np.where(np.abs(xhat) > 0, np.abs(xhat), 1.0)
    Jrr = (_fd_jac(forward, xhat, diff_step=diff_step)[idx] / yden[:, None]) * xsc[None, :]   # d(rel-obs)/d(rel-param)
    F = Jrr.T @ Jrr / noise_rel ** 2                                  # Fisher in RELATIVE parameter space
    lam, V = np.linalg.eigh(F)                                        # ascending eigenvalues λᵢ; columns V[:,i] = eigen-direction i
    sig_dir = 1.0 / np.sqrt(np.maximum(lam, 1e-300))                  # σ per EIGEN-DIRECTION (relative recovery uncertainty)
    well = sig_dir < sigma_tol                                       # well-conditioned ⇔ recoverable along that direction
    out = {"x_hat": xhat, "n_unknowns": n, "n_constraints": int(np.sum(well)), "deficiency": int(np.sum(~well)),
           "sigma_per_direction": sig_dir, "eigen_directions": V.T}
    if np.all(well):
        out.update(status=("over-determined" if len(idx) > n else "determined"), point_trustworthy=True, cov=cov,
                   flat_basis=None, null_basis=None, feasible_extent=None)
    else:
        flat = (V.T[~well] * xsc[None, :])                           # flat eigen-directions → ABSOLUTE x-space
        flat = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-30)
        lo_b, hi_b = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
        extent = []                                                  # per flat direction: in-bounds slide range of x̂ along it
        for d in flat:
            a_lo, a_hi = -np.inf, np.inf
            for j in range(n):
                if abs(d[j]) > 1e-12:
                    t0, t1 = (lo_b[j] - xhat[j]) / d[j], (hi_b[j] - xhat[j]) / d[j]
                    a_lo, a_hi = max(a_lo, min(t0, t1)), min(a_hi, max(t0, t1))
            extent.append((float(a_lo), float(a_hi)))
        out.update(status="under-determined", point_trustworthy=False, cov=None, flat_basis=flat,
                   null_basis=flat, feasible_extent=extent)          # REGION along the flat/sloppy directions (null_basis alias kept)
    return out


def serve_inverse(forward, observed, x0, bounds, claim_fn=None, x_prior=None, prior_sigma=None, noise_rel=0.02, **kw):
    """VERTICAL-FACING one-call inverse (C-T4) — recover a patent's withheld design from its claims, HONESTLY, in ONE call.
    F/G/H must not have to orchestrate (or forget) a step, so this runs the WHOLE lifecycle: eigen-spectrum classify → a
    POINT only if every direction is well-conditioned, else a REGION → optional Tikhonov resolution GUARDED by a prior-data
    conflict check → optional derived-claim σ. Returns a flat dict a vertical consumes directly:
      {status, trustworthy, x_hat, sigma, n_recoverable, deficiency, region (flat_basis+extent | None),
       claim ({value, sigma} | None), note}. The contract: it NEVER returns a false point — under-determined ⇒ region (or a
       prior-resolved point with honestly-inflated σ, only if the prior is data-consistent)."""
    rc = inverse_classified(forward, observed, x0, bounds, noise_rel=noise_rel, **kw)
    out = {"status": rc["status"], "trustworthy": rc["point_trustworthy"], "x_hat": rc["x_hat"],
           "n_recoverable": rc["n_constraints"], "deficiency": rc["deficiency"],
           "sigma_per_direction": rc["sigma_per_direction"], "sigma": None, "region": None, "claim": None}
    if rc["point_trustworthy"]:
        out["sigma"] = np.sqrt(np.clip(np.diag(rc["cov"]), 0.0, None))
        out["note"] = "trustworthy point — all directions well-conditioned"
        if claim_fn is not None:
            val, ccov = propagate_claim_sigma(claim_fn, rc["x_hat"], rc["cov"])
            out["claim"] = {"value": float(np.atleast_1d(val)[0]), "sigma": float(np.sqrt(max(float(ccov[0, 0]), 0.0)))}
    else:
        out["region"] = {"flat_basis": rc["flat_basis"], "feasible_extent": rc["feasible_extent"]}
        if x_prior is not None:
            obs = np.atleast_1d(np.asarray(observed, float)); pr = np.atleast_1d(forward(x_prior))
            den = np.where(np.abs(obs) > 1e-12, np.abs(obs), 1.0)
            res = float(np.max(np.abs(pr - obs) / den))             # prior-data conflict guard (never trust a wrong prior)
            if res > 3 * noise_rel:
                out["note"] = f"under-determined; prior CONFLICTS with data (residual {res:.2f} > 3·noise) — prior refused, region only"
            else:
                xh, cov, _ = inverse(forward, observed, x0, bounds, noise_rel=noise_rel, x_prior=x_prior, prior_sigma=prior_sigma, **kw)
                out["x_hat"], out["sigma"] = xh, np.sqrt(np.clip(np.diag(cov), 0.0, None))
                out["note"] = f"under-determined; resolved with a data-consistent prior — honest σ (large in {rc['deficiency']} flat directions)"
        else:
            out["note"] = f"under-determined ({rc['deficiency']} flat directions) — REGION only, no false point quoted"
    return out


def select_model(candidates, observed, noise_rel=0.02, **kw):
    """Model selection — the verticals' 'WHICH forward explains the patent's claims?' step (e.g. thermodynamic vs kinetic-
    quench). Fit each candidate forward by inverse and score by AIC = n·ln(RSS/n) + 2k: the parsimony penalty (2k)
    REJECTS an over-complex model that merely fits the noise, the fit term rejects an under-complex one. Returns
    {best, table (name→{aic, rss, k, x_hat}), delta_aic (top-two gap), ambiguous (ΔAIC<2 ⇒ the data does NOT distinguish
    the top two — report it, do not force a pick)}. candidates: list of {name, forward, x0, bounds}. Composes with
    serve_inverse: select the model, then recover the design on it — never invert on a forward the data doesn't support."""
    observed = np.atleast_1d(np.asarray(observed, float)); n = len(observed)
    table = {}
    den = np.where(np.abs(observed) > 1e-12, np.abs(observed), 1.0)
    for c in candidates:
        xh = inverse(c["forward"], observed, c["x0"], c["bounds"], noise_rel=noise_rel, **kw)[0]
        rss = float(np.sum(((np.atleast_1d(c["forward"](xh)) - observed) / den) ** 2))   # RELATIVE RSS (matches inverse + multiplicative noise)
        k = len(np.atleast_1d(c["x0"]))
        table[c["name"]] = {"aic": float(n * np.log(max(rss, 1e-300) / n) + 2 * k), "rss": rss, "k": k, "x_hat": xh}
    ranked = sorted(table.items(), key=lambda kv: kv[1]["aic"])
    d_aic = (ranked[1][1]["aic"] - ranked[0][1]["aic"]) if len(ranked) > 1 else None
    return {"best": ranked[0][0], "table": table, "delta_aic": d_aic,
            "ambiguous": bool(d_aic is not None and d_aic < 2.0)}


def tolerance_sensitivity(forward, x, input_sigma, perf_idx=0, diff_step=None):
    """Tolerance ALLOCATION for a manufactured design — the FORWARD-variance dual of claim-σ. Given a design x and the
    per-parameter MANUFACTURING tolerances input_sigma (1σ), propagate to the performance σ (delta method) AND decompose
    which parameter's tolerance dominates: σ_perf² = Σ (∂perf/∂xᵢ)² σ_xᵢ². Returns {perf_sigma, contributions (per-param
    fraction of the performance variance), dominant (index of the parameter to hold tightest)}. claim-σ gives the
    magnitude; this gives the actionable ATTRIBUTION — control the dominant dimensions tightly, loosen the rest. Composes
    with generative: design → its manufacturing tolerance spec."""
    x = np.asarray(x, float); s = np.asarray(input_sigma, float)
    g = _fd_jac(forward, x, diff_step=diff_step)[perf_idx]           # ∂(performance observable)/∂x
    var_contrib = (g * s) ** 2                                       # per-parameter variance contribution
    perf_var = float(np.sum(var_contrib))
    return {"perf_sigma": float(np.sqrt(perf_var)),
            "contributions": var_contrib / max(perf_var, 1e-300), "dominant": int(np.argmax(var_contrib))}


def goodness_of_fit(forward, observed, x_hat, noise_rel=0.02, n_params=None):
    """Lack-of-fit test — after a recovery, is the FORWARD actually RIGHT, or just the least-bad select_model could find?
    select_model ranks candidates; if ALL of them are wrong it still returns one. The honest check is the residual: under
    the right forward the standardized residuals are O(1), so reduced χ² = Σ((pred−obs)/(σ·obs))² / dof ≈ 1; a WRONG forward
    leaves systematic residuals the noise can't explain, so reduced χ² ≫ 1. Returns {chi2_reduced, dof, p_value,
    lack_of_fit}: lack_of_fit ⇒ do NOT trust the recovery (no parameter value reconciles this forward with the data)."""
    observed = np.atleast_1d(np.asarray(observed, float)); pred = np.atleast_1d(np.asarray(forward(x_hat), float))
    den = np.where(np.abs(observed) > 1e-12, np.abs(observed), 1.0)
    chi2 = float(np.sum(((pred - observed) / den / noise_rel) ** 2))
    n_params = n_params if n_params is not None else len(np.atleast_1d(x_hat))
    dof = max(len(observed) - n_params, 1)
    from scipy.stats import chi2 as _chi2
    p = float(_chi2.sf(chi2, dof))
    return {"chi2_reduced": chi2 / dof, "dof": dof, "p_value": p, "lack_of_fit": bool(p < 1e-3)}


def bootstrap_sigma(forward, observed, x0, bounds, n_boot=120, noise_rel=0.02, seed=0, **kw):
    """NON-PARAMETRIC σ — cross-check the Fisher/linearized σ by resampling. The Fisher σ assumes Gaussian noise; this
    resamples the observation set WITH REPLACEMENT, refits each resample, and takes the spread — making no noise-model
    assumption. When they agree, the reported σ is trustworthy; when the bootstrap σ is LARGER, the Gaussian assumption is
    optimistic (heavy tails / outliers the Fisher σ doesn't see). Returns {sigma (per-parameter bootstrap std), x_mean}."""
    observed = np.atleast_1d(np.asarray(observed, float)); M = len(observed); r = np.random.default_rng(seed)
    recs = []
    for _ in range(n_boot):
        idx = r.integers(0, M, M)                                # resample observation indices with replacement
        recs.append(inverse(forward, observed, x0, bounds, noise_rel=noise_rel, use=idx, **kw)[0])
    recs = np.asarray(recs)
    return {"sigma": np.std(recs, axis=0), "x_mean": np.mean(recs, axis=0)}


def next_measurement(forward, x0, candidates, current=None, noise_rel=0.02, diff_step=None):
    """Active sensing / optimal experimental design — the ACTIONABLE complement to identifiability: not 'are these
    measurements enough?' but 'what should I measure NEXT?'. Of the CANDIDATE observables, pick the one that most
    reduces the recovered-parameter uncertainty when ADDED to `current` (D-optimal: maximize log-det Fisher = minimize
    the posterior σ-ellipsoid volume). Returns {best, scores (candidate→log-det info), gain (best vs current)}.
    ONE engine for perception next-best-view, motion next-best-sensor/trial, patent next-most-informative-claim."""
    x0 = np.asarray(x0, float); y0 = np.atleast_1d(forward(x0)); J = _fd_jac(forward, x0, diff_step=diff_step) / y0[:, None]
    cur = list(current) if current is not None else []

    def logdet(idx):
        Ji = J[np.asarray(idx)]; F = Ji.T @ Ji / noise_rel ** 2
        ev = np.linalg.eigvalsh(F); eps = 1e-10 * max(float(ev.max()), 1e-30)
        return float(np.sum(np.log(ev + eps)))
    scores = {int(c): logdet(cur + [c]) for c in candidates}
    best = int(max(scores, key=scores.get))
    return {"best": best, "scores": scores, "gain": (scores[best] - logdet(cur)) if cur else None}


def oed_sequence(forward, x0, candidates, n_select, prior_precision=None, noise_rel=0.02, diff_step=None):
    """Sequential Bayesian optimal experiment design — greedily pick n_select of the CANDIDATE observables that drive the
    posterior parameter uncertainty down FASTEST. Each step adds the observable maximizing log-det of the posterior Fisher
    (D-optimal); the posterior precision F = prior_precision + Σ (selected) JᵀJ/σ² carries forward, so the design is
    ADAPTIVE (never re-buys a direction it already pinned) and PRIOR-AWARE (a tight prior on a param steers the budget
    toward the uncertain directions). Returns {order, sigma_trace (√tr posterior cov per step), logdet (per step)}.
    ONE engine for perception next-best-view, motion next-best-trial, patent next-most-informative-claim — sequenced."""
    x0 = np.asarray(x0, float); y0 = np.atleast_1d(forward(x0))
    J = _fd_jac(forward, x0, diff_step=diff_step) / y0[:, None]      # relative-sensitivity rows
    n = J.shape[1]
    F = np.array(prior_precision, float) if prior_precision is not None else 1e-6 * np.eye(n)   # weak prior ⇒ proper posterior
    remaining, order, sig_trace, logdets = list(candidates), [], [], []
    for _ in range(min(n_select, len(remaining))):
        best_c, best_ld = remaining[0], -np.inf
        for c in remaining:
            Fc = F + np.outer(J[c], J[c]) / noise_rel ** 2
            ev = np.linalg.eigvalsh(Fc)
            ld = float(np.sum(np.log(np.maximum(ev, 1e-30))))
            if ld > best_ld:
                best_c, best_ld = c, ld
        F = F + np.outer(J[best_c], J[best_c]) / noise_rel ** 2
        remaining.remove(best_c); order.append(int(best_c)); logdets.append(best_ld)
        sig_trace.append(float(np.sqrt(np.trace(np.linalg.inv(F)))))
    return {"order": order, "sigma_trace": sig_trace, "logdet": logdets}


def sigma_check(forward, x_true, x0, bounds, use=None, noise_rel=0.02, n_mc=60, seed=0, diff_step=None):
    """FALSIFY the Fisher σ-region against Monte-Carlo — the σ the verticals trust for decisions is local-linear-Gaussian
    and can LIE for strongly nonlinear / multi-modal inverses. Re-inverts n_mc noisy observations and compares the actual
    recovery spread (+ bias) to the reported Fisher σ. Returns {fisher_sigma, mc_sigma, ratio (mc/fisher), bias (mc mean −
    truth, in σ units), reliable}. reliable=False ⇒ use MC, not the Fisher σ, for that recovery."""
    x_true = np.asarray(x_true, float); obs = np.atleast_1d(forward(x_true))
    fisher = np.sqrt(np.maximum(np.diag(inverse(forward, obs, x0, bounds, noise_rel=noise_rel, use=use, diff_step=diff_step)[1]), 0.0))
    rng = np.random.default_rng(seed); recs = []
    for _ in range(int(n_mc)):
        noisy = obs * (1 + noise_rel * rng.standard_normal(len(obs)))
        recs.append(inverse(forward, noisy, x0, bounds, noise_rel=noise_rel, use=use, diff_step=diff_step)[0])
    recs = np.asarray(recs); mc = np.std(recs, axis=0); bias = (np.mean(recs, axis=0) - x_true) / (mc + 1e-30)
    ratio = mc / (fisher + 1e-30)
    reliable = bool(np.all((ratio > 0.6) & (ratio < 1.6)) and np.all(np.abs(bias) < 1.0))
    return {"fisher_sigma": fisher, "mc_sigma": mc, "ratio": ratio, "bias": bias, "reliable": reliable}


# ───────────────────────────── self-test (validate the service end-to-end) ─────────────────────────────
def beam_forward(x):                                                  # (w,t) → (compliance, resonance, mass)
    w, t = x; L, E, RHO = 0.20, 70e9, 2700.0
    return np.array([4 * L ** 3 / (E * w * t ** 3),
                     (1.875 ** 2 / (2 * np.pi)) * (t / L ** 2) * np.sqrt(E / (12 * RHO)),
                     RHO * w * t * L])


def projectile_forward(x):                                           # (v0, theta_deg) → (range, apex, flight_time)
    v0, th = x; g = 9.81; r = np.radians(th)
    return np.array([v0 ** 2 * np.sin(2 * r) / g, v0 ** 2 * np.sin(r) ** 2 / (2 * g), 2 * v0 * np.sin(r) / g])


def federate_inverse(renderers, observed, x0, bounds, sigmas=None, jacs=None):
    """FEDERATE N renderers (each a forward x→obs_i on a SHARED geometry x) into ONE inverse over x — the
    over-determined multi-observable inverse (D-S2 QuadFluid-FC: 4 fluid renderers on one θ). Solve x by minimizing the
    STACKED σ-normalized residual of all renderers jointly (one θ-gradient federating every renderer's adjoint). The joint
    reduced-χ² is the BUILT-IN falsification statistic: if no x makes all renderers agree within σ (χ²≫1), the renderers
    DISAGREE → the shared-geometry model is falsified. Renderers plug in by the contract x→obs (A's validated forwards or a
    surrogate). Returns {x_hat, chi2_reduced, dof, per_renderer_rms, sigma_per_direction, cov, n_obs, fisher_cond, rank_deficient}; rank_deficient gates identifiability BEFORE trusting the point (S12 Fisher-rank, diversity not count)."""
    observed = [np.atleast_1d(np.asarray(o, float)) for o in observed]
    if sigmas is None:
        sigmas = [0.02 * np.abs(o) + 1e-12 for o in observed]              # default 2% relative σ per observable
    sigmas = [np.atleast_1d(np.asarray(s, float)) for s in sigmas]
    resid = lambda x: np.concatenate([(np.atleast_1d(renderers[i](x)) - observed[i]) / sigmas[i] for i in range(len(renderers))])
    if jacs is not None:                                                  # federate the ADJOINTS: stack each renderer's σ-normalized Jacobian ∂obs/∂x into one joint θ-Jacobian
        jac = lambda x: np.vstack([np.atleast_2d(jacs[i](x)) / sigmas[i][:, None] for i in range(len(renderers))])
        res = least_squares(resid, np.asarray(x0, float), bounds=bounds, jac=jac)
    else:
        res = least_squares(resid, np.asarray(x0, float), bounds=bounds)   # no adjoints → finite-difference Jacobian
    r = res.fun; dof = max(len(r) - len(x0), 1)
    per = [float(np.sqrt(np.mean(((np.atleast_1d(renderers[i](res.x)) - observed[i]) / sigmas[i]) ** 2))) for i in range(len(renderers))]
    F = res.jac.T @ res.jac                                                # joint Fisher (σ-normalized residual → F is in σ-units)
    eig = np.linalg.eigvalsh(F)
    cond = float(eig.max() / max(eig.min(), 1e-30))                        # Fisher-RANK identifiability gate (S12: diversity, not count)
    return {"x_hat": res.x, "chi2_reduced": float(r @ r / dof), "dof": dof, "per_renderer_rms": per,
            "sigma_per_direction": 1.0 / np.sqrt(np.clip(eig, 1e-12, None)), "cov": np.linalg.pinv(F), "n_obs": len(r),
            "fisher_cond": cond, "rank_deficient": bool(eig.min() / eig.max() < 1e-8)}   # rank_deficient → a θ-direction is unobserved; refuse the point


def spectral_diagnostic(J):
    """GRAPH-LAYER spectral diagnostic (D meta-meta-primitive #4): ONE eigen+SVD of a coupled Jacobian J → all margins together.
    criticality_margin = max real part of the spectrum → 0 at a FOLD (real leading eig) or a HOPF (complex leading pair);
    identifiability_margin = σ_min(J)² = λ_min(JᵀJ) → 0 = Fisher/inverse COLLAPSE; adjoint_margin = 1/σ_min → ∞ = adjoint-BLOWUP.
    All lanes consume this: A forward criticality-margin, C inverse/identifiability, federation adjoint-margin — one object J."""
    J = np.atleast_2d(np.asarray(J, float)); eig = np.linalg.eigvals(J); lead = eig[int(np.argmax(eig.real))]
    smin = float(np.linalg.svd(J, compute_uv=False).min()); osc = bool(abs(lead.imag) > 1e-9); crit = float(lead.real)
    regime = ("CRITICAL-HOPF" if abs(crit) < 0.05 and osc else "CRITICAL-FOLD" if abs(crit) < 0.05
              else "ILL-IDENTIFIED" if smin < 0.05 else "WELL-POSED")
    return {"criticality_margin": crit, "oscillatory": osc, "hopf_freq": float(abs(lead.imag)),
            "identifiability_margin": smin ** 2, "sigma_min": smin, "adjoint_margin": 1.0 / max(smin, 1e-300), "regime": regime}


def main():
    print("=" * 98)
    print("GENERATIVE+INVERSE SERVICE (spine T2/T3) — one forward-agnostic engine the verticals call")
    print("=" * 98)
    bb = ([0.004, 0.002], [0.030, 0.020])

    # G0: generative service
    xg = generative(beam_forward, lambda y: y[1] / np.sqrt(y[2]), [0.015, 0.008], bb)
    g0 = abs(xg[1] - bb[1][1]) < 1e-3 and abs(xg[0] - bb[0][0]) < 1e-3
    print(f"\n(G0) GENERATIVE SERVICE — generative(beam, f₁/√m) → (w,t)=({1000*xg[0]:.1f},{1000*xg[1]:.1f}) mm (bound-extremal): {'PASS' if g0 else 'FAIL'}")

    # G1: inverse service + σ-region vs Monte-Carlo
    x_true = np.array([0.011, 0.0065]); obs = beam_forward(x_true)
    xi, cov, res = inverse(beam_forward, obs, [0.02, 0.004], bb, noise_rel=0.03)
    rec_err = float(np.max(np.abs(xi - x_true) / x_true)); sig_pred = np.sqrt(np.diag(cov)) / x_true
    rng = np.random.default_rng(0)
    mc = np.array([inverse(beam_forward, obs * (1 + 0.03 * rng.standard_normal(3)), [0.02, 0.004], bb)[0] for _ in range(40)])
    sig_mc = np.std(mc / x_true, axis=0)
    g1 = rec_err < 0.01 and np.max(np.abs(sig_pred - sig_mc) / (sig_mc + 1e-12)) < 0.3
    print(f"\n(G1) INVERSE SERVICE + σ-REGION — recovered (w,t) err {100*rec_err:.2f}%; Fisher σ {np.round(sig_pred,3)} vs MC {np.round(sig_mc,3)}: {'PASS' if g1 else 'FAIL'}")

    # G2: forward-agnostic — same inverse on a totally different physics
    p_true = np.array([30.0, 40.0]); pobs = projectile_forward(p_true)
    pb = ([5.0, 5.0], [60.0, 85.0])
    pi, pcov, pres = inverse(projectile_forward, pobs, [20.0, 60.0], pb, noise_rel=0.02)
    p_err = float(np.max(np.abs(pi - p_true) / p_true))
    g2 = p_err < 0.01
    print(f"\n(G2) FORWARD-AGNOSTIC — SAME inverse() on PROJECTILE (range,apex,time): recovered (v₀,θ)=({pi[0]:.2f},{pi[1]:.2f}) "
          f"vs true ({p_true[0]:.0f},{p_true[1]:.0f}) (err {100*p_err:.2f}%) — no code change: {'PASS' if g2 else 'FAIL'}")

    # G3: over-determination tightens σ ONLY for INDEPENDENT claims (the identifiability subtlety the verticals must know).
    sb2 = np.sqrt(np.diag(inverse(beam_forward, obs, [0.02, 0.004], bb, noise_rel=0.03, use=[1, 2])[1])) / x_true
    sb3 = np.sqrt(np.diag(cov)) / x_true                              # beam: C,f₁,m are independent
    beam_shrink = float(np.mean(sb2) / np.mean(sb3))
    sp2 = np.sqrt(np.diag(inverse(projectile_forward, pobs, [20.0, 60.0], pb, noise_rel=0.02, use=[0, 1])[1])) / p_true
    proj_shrink = float(np.mean(sp2) / np.mean(np.sqrt(np.diag(pcov)) / p_true))   # projectile: apex≡gT²/8 (dependent)
    g3 = beam_shrink > 1.05 and proj_shrink < 1.10
    print(f"\n(G3) OVER-DETERMINATION = IDENTIFIABILITY — tightens σ ONLY when the added claim is INDEPENDENT:")
    print(f"     beam (C,f₁,m independent): 2→3 claims = {beam_shrink:.2f}× tighter; projectile (apex≡gT²/8 DEPENDENT): {proj_shrink:.2f}× (≈1, no new")
    print(f"     info) → the service exposes identifiability; the verticals must pick INDEPENDENT claims: {'PASS' if g3 else 'FAIL'}")

    # G4: ILL-POSED inverse stabilized by regularization (NIGHT_BACKLOG T7) — recover (w,t) from ONLY mass (1 claim, 2 params)
    mass_obs = beam_forward(x_true)
    Jh = _fd_jac(beam_forward, x_true)[[2]] / mass_obs[2]             # 1×2 Jacobian of mass → rank-deficient (null direction)
    ev = np.linalg.eigvalsh(Jh.T @ Jh); cond = float(ev.min() / ev.max())
    xp = np.array([0.012, 0.006])
    xr, covr, resr = inverse(beam_forward, mass_obs, xp.copy(), bb, noise_rel=0.03, use=[2], x_prior=xp, prior_sigma=0.3)
    sig_reg = np.sqrt(np.diag(covr)) / x_true; mass_ok = abs(beam_forward(xr)[2] / mass_obs[2] - 1) < 0.02
    g4 = cond < 1e-6 and np.max(sig_reg) < 0.6 and mass_ok
    print(f"\n(G4) ILL-POSED INVERSE → REGULARIZED (T7) — recover (w,t) from ONLY mass (1 claim, 2 params): unregularized Fisher")
    print(f"     rank-deficient (cond {cond:.0e}, null direction unbounded); a prior bounds it → finite σ {np.round(sig_reg,2)}, "
          f"recovery still meets the claim (resid {resr:.0e}): {'PASS' if g4 else 'FAIL'}")

    # G5: MULTI-OBJECTIVE generative — Pareto front of stiffness (−compliance) vs lightness (−mass) on the beam.
    front = generative_pareto(beam_forward, lambda y: -y[0], lambda y: -y[2], [0.015, 0.008], bb, n=6, iters=1500)
    comps = np.array([-f[2] for f in front]); masses = np.array([-f[3] for f in front])   # compliance, mass along the front
    o = np.argsort(masses); mono = np.all(np.diff(comps[o]) <= 1e-9)                       # heavier ⇒ stiffer (lower compliance)
    span = (masses.max() / masses.min() > 1.5) and (comps.max() / comps.min() > 1.5)
    g5 = mono and span
    print(f"\n(G5) MULTI-OBJECTIVE (Pareto) — stiffness vs lightness front ({len(front)} designs): mass ×{masses.max()/masses.min():.0f} span, "
          f"compliance ×{comps.max()/comps.min():.0f} span; heavier⇒stiffer monotone tradeoff: {'PASS' if g5 else 'FAIL'}")

    # G6: IDENTIFIABILITY diagnostic — pre-score the 'I' grade from the Jacobian alone (no observed values needed).
    id3 = identifiability(beam_forward, x_true, use=[0, 1, 2], noise_rel=0.03)    # 3 independent claims, 2 params
    id1 = identifiability(beam_forward, x_true, use=[2], noise_rel=0.03)          # 1 claim (mass), 2 params → ill-posed
    idp = identifiability(projectile_forward, p_true, use=[1, 2], noise_rel=0.02) # apex+time DEPENDENT (H≡gT²/8)
    g6 = (id3["identifiable"].all() and not id1["identifiable"].all()
          and id1["condition"] > 1e4 * id3["condition"] and not idp["identifiable"].all())
    print(f"\n(G6) IDENTIFIABILITY ('I' grade, Jacobian-only) — beam 3-claim: identifiable {id3['identifiable']} cond {id3['condition']:.0e}; "
          f"beam 1-claim(mass): {id1['identifiable']} cond {id1['condition']:.0e} (ill-posed); projectile apex+time (DEPENDENT): "
          f"{idp['identifiable']} → the diagnostic catches under- AND dependent-determination: {'PASS' if g6 else 'FAIL'}")

    # G7: GENERATIVE on a FLOAT32 forward needs diff_step for INTERIOR optima (bound optima are robust; interior aren't)
    f32 = lambda z: np.array([np.float32(z[0])]); pk = lambda y: -(y[0] - 0.5) ** 2    # interior optimum x*=0.5
    g_ds = [generative(f32, pk, [x0], ([0.0], [1.0]), diff_step=1e-3)[0] for x0 in (0.1, 0.9)]
    g_no = [generative(f32, pk, [x0], ([0.0], [1.0]))[0] for x0 in (0.1, 0.9)]          # default FD underflows the float32 quantization
    g7 = max(abs(v - 0.5) for v in g_ds) < 1e-3 and max(abs(v - 0.5) for v in g_no) > 0.05
    print(f"\n(G7) GENERATIVE on FLOAT32 (interior optimum) — with diff_step reaches x*=0.5 from both starts ({np.round(g_ds,3)}); "
          f"without (default FD underflows float32) stalls short ({np.round(g_no,3)}): the generative side needs diff_step too: {'PASS' if g7 else 'FAIL'}")

    allok = g0 and g1 and g2 and g3 and g4 and g5 and g6 and g7
    print("\n" + "=" * 98)
    if allok:
        print("VERDICT: the GENERATIVE+INVERSE SERVICE is stood up — two importable functions, generative() & inverse(), that")
        print(f"  run over ANY validated forward model (beam AND projectile, no code change), returning the extremal design or")
        print(f"  the recovered parameters + a σ-bounded Fisher region (matching MC), with over-determination tightening the σ.")
        print(f"  This is the spine the verticals CALL (NIGHT_BACKLOG T2/T3): pass your forward + your patent's claims → inverse/")
        print(f"  generative fall out, fixed engine, only the objective changes. Fit-trap right side: physics fixed, inputs optimized.")
    else:
        print(f"VERDICT: NOT all pass — G0 {g0} G1 {g1} G2 {g2} G3 {g3} G4 {g4} G5 {g5} G6 {g6} G7 {g7}. Fix at SOURCE.")
    print("=" * 98)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
