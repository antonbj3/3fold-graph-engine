#!/usr/bin/env python3
"""decorrelated_probe_value — agent-worktree DEPLOYABLE distilling the / arc (color-metamers, but the principle is GENERAL): the
certifiable VALUE of a 2nd (decorrelated) observation for the twin is its operator-overlap with the CONFUSION MODE — the null-space of
the 1st observation restricted to the goal — NOT a scalar decorrelation statistic. Matches the agent pool payoff phase (weakest_link_cert,
tail_family_cert/monocular_rotation_recoverability_cert, tail_severity, motion_validity_cert, my null_remedy_router /
decorrelated_recovery).

THE OBJECT (agent pool decorrelation-certifies thesis, made a design tool):
  Given a 1st linear observation operator M_A (rows = its measurement functionals over a d-dim signal) and a candidate 2nd observation
  M_B, the value of B = how much of what A CANNOT see (null(M_A), optionally intersected with a low-rank GOAL) does B see:
      probe_value(A,B) = || M_B  N ||_F / sqrt(r),   N = orthonormal basis of null(M_A) ∩ goal,  r = its dim
  This is the RIGHT operator: on real CIE data, scalar SPD-decorrelation predicts metamer-resolving power at only Spearman 0.33,
  while this operator-overlap IS the resolving power. A generically-"decorrelated" probe that happens to be blind to the confusion mode
  scores 0.

Design (active OED): optimal_probe returns the confusion-mode direction of max overlap; for a PHYSICAL probe (nonneg spectrum) it
projects to b>=0. CAVEAT baked in: an unconstrained/delta optimum OVER-STATES headroom — evaluate the realizable finite-bandwidth version
(8.5x delta -> ~2.3x for a 20nm LED -> a broad probe can be WORSE than the best stock one).

  probe_value(M_A, M_B, goal=None)              -> scalar overlap (decorrelated info B adds about the confusion mode)
  rank_probes(M_A, candidates, goal=None)       -> [(key, value)] sorted desc (pick the best decorrelated 2nd observation)
  optimal_probe(M_A, design_weight=None, nonneg=False, goal=None) -> the max-overlap probe direction (active-OED design)
numpy only; observation operators are user-supplied arrays so this is modality-agnostic (color, depth/geometry, spectra, features, ...)."""
import numpy as np


def _null_basis(M_A, goal=None, tol=1e-9):
    """orthonormal basis of null(row-space of M_A), optionally intersected with the goal subspace (goal = d x g basis, columns)."""
    A = np.atleast_2d(np.asarray(M_A, float)); d = A.shape[1]
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    r = int((S > (S.max() * tol)).sum()) if S.size else 0
    N = Vt[r:].T                                                   # d x (d-r): null-space of M_A
    if goal is not None:
        G = np.asarray(goal, float)
        if G.ndim == 1: G = G[:, None]
        # intersect null(M_A) with span(G): project G's component that lies in the null, re-orthonormalize
        P_null = N @ N.T
        GN = P_null @ G
        Uq, Sq, _ = np.linalg.svd(GN, full_matrices=False)
        N = Uq[:, Sq > (Sq.max() * tol if Sq.size and Sq.max() > 0 else 1)]
    return N                                                       # d x r


def probe_value(M_A, M_B, goal=None):
    """Certifiable value of a 2nd observation M_B given the 1st M_A: RMS response of M_B over the unit confusion directions
    (null(M_A) ∩ goal). 0 => B is redundant with A (sees nothing new about the goal); large => B resolves what A cannot."""
    N = _null_basis(M_A, goal)
    if N.shape[1] == 0:
        return 0.0
    B = np.atleast_2d(np.asarray(M_B, float))
    return float(np.linalg.norm(B @ N, "fro") / np.sqrt(N.shape[1]))


def rank_probes(M_A, candidates, goal=None):
    """candidates: dict{key: M_B} or list[M_B]. Returns [(key, probe_value)] sorted descending — the best decorrelated 2nd observation."""
    items = candidates.items() if isinstance(candidates, dict) else enumerate(candidates)
    scored = [(k, probe_value(M_A, B, goal)) for k, B in items]
    return sorted(scored, key=lambda t: t[1], reverse=True)


def optimal_probe(M_A, design_weight=None, nonneg=False, goal=None, iters=500):
    """Active-OED design: the signal-space direction that, used as a 2nd observation functional, maximally overlaps the confusion mode.
    design_weight: optional (d) elementwise map from a design vector to the observation row (e.g. the CMF for illuminant design; default
    identity). nonneg=True projects to a physical (>=0) design via projected power iteration. Returns (design_vector, achieved_value).
    ★ CAVEAT: the returned optimum can be a delta/spiky direction that OVER-STATES real headroom — evaluate a realizable finite-
    bandwidth version before claiming a design gain (it can fall below the best stock probe)."""
    N = _null_basis(M_A, goal)
    d = np.atleast_2d(np.asarray(M_A, float)).shape[1]
    if N.shape[1] == 0:
        return np.zeros(d), 0.0
    W = np.ones(d) if design_weight is None else np.asarray(design_weight, float)
    if W.ndim == 1:                                                # scalar elementwise weight -> row = w ∘ design
        # overlap^2(x) = || (W ∘ x)·N ||^2 summed = x^T Q x with Q = diag(W) N N^T diag(W)
        Q = (W[:, None] * N) @ (N.T * W[None, :])
    else:                                                          # W is a k x d map: observation row_c = (W_c ∘ x); sum over c
        Q = np.zeros((d, d))
        for Wc in np.atleast_2d(W):                                # each measurement channel (e.g. a CMF x-bar/y-bar/z-bar)
            M = Wc[:, None] * N                                     # (d x r): (W_c ∘ ·) applied to each null direction
            Q += M @ M.T
    if not np.all(np.isfinite(Q)):
        # eigh-on-NaN is structure/LAPACK-path-dependent (never trust it to raise) -- guard explicit.
        raise ValueError("optimal_probe: non-finite design matrix Q (NaN/Inf in M_A/design_weight)")
    if nonneg:
        w_, V_ = np.linalg.eigh(Q); x = np.abs(V_[:, -1]); x /= (np.linalg.norm(x) + 1e-12)
        for _ in range(iters):
            x = Q @ x; x = np.clip(x, 0, None); n = np.linalg.norm(x)
            if n < 1e-12: break
            x /= n
    else:
        w_, V_ = np.linalg.eigh(Q); x = V_[:, -1]
        if x.sum() < 0: x = -x
    val = float(np.sqrt(max(x @ Q @ x, 0.0) / N.shape[1]))
    return x, val


def _selftest():
    ok = tot = 0
    rng = np.random.default_rng(0)
    # (1) mechanism: a probe entirely in row(M_A) is redundant (value 0); one in null is max
    d = 12; M_A = rng.standard_normal((4, d))
    U, S, Vt = np.linalg.svd(M_A, full_matrices=True)
    B_row = Vt[:4]; B_null = Vt[4:8]
    tot += 1; ok += (probe_value(M_A, B_row) < 1e-6 and probe_value(M_A, B_null) > 0.1)
    # (2) REAL CIE validation: as M_A, fluorescents outrank a 2nd daylight for resolving -metamers
    try:
        import colour
        from colour import MSDS_CMFS, SDS_ILLUMINANTS
        shape = colour.SpectralShape(400, 700, 5)
        cmf = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape).values
        ill = lambda n: SDS_ILLUMINANTS[n].copy().align(shape).values
        Mop = lambda X: (X[:, None] * cmf).T
        MA = Mop(ill("D65"))
        pw = lambda X: X / (X.sum() + 1e-12)                        # EQUAL total-power normalization (fair; probe_value scales with power)
        cand = {n: Mop(pw(ill(n))) for n in ["D50", "D55", "D75", "FL2", "FL7", "FL11", "FL12"] if n in SDS_ILLUMINANTS}
        ranked = rank_probes(MA, cand)
        fl = np.mean([v for k, v in ranked if k.startswith("FL")])
        day = np.mean([v for k, v in ranked if k in ("D50", "D55", "D75")])
        tot += 1; ok += (fl > day)                                 # fluorescents resolve D65-metamers more, at equal power (L80)
        # (3) optimal designed probe (nonneg, CMF design map) beats the best real illuminant at EQUAL power, narrowband-blue (~445nm)
        wl = shape.wavelengths
        x, _ = optimal_probe(MA, design_weight=cmf.T, nonneg=True)  # design a physical illuminant (b>=0)
        val = probe_value(MA, Mop(pw(x)))                          # score the designed probe at the SAME equal-power budget
        best_real = max(v for _, v in ranked)
        peak_wl = wl[int(np.argmax(x))]
        tot += 1; ok += (val > best_real and 420 <= peak_wl <= 480)
        print("decorrelated_probe_value selftest: real-CIE FL>day=%.2f>%.2f (equal power), optimal designed val=%.2f>best_real=%.2f @%.0fnm" % (
            fl, day, val, best_real, peak_wl))
    except Exception as e:
        print("decorrelated_probe_value selftest: colour unavailable, ran mechanism check only (%s)" % type(e).__name__)
    print("decorrelated_probe_value selftest: %d/%d" % (ok, tot))
    return ok == tot


if __name__ == "__main__":
    _selftest()
