"""neff_channel_diversity -- effective number of independent detection channels.

Computes the EFFECTIVE number of independent detection channels a set of veto-legs provides on an axis, not the
nominal leg count: 'K legs agree' is nominal n_eff = K; a shared computational substrate or shared pretraining
collapses it (measured: dense_flow x mono_depth = 0.59, two Farneback-flow functionals = one channel).

★TWO correctness rules this core enforces (the traps a naive counter falls into):
  (1) CLUSTER by CORRELATION via CONNECTED COMPONENTS (not a greedy independent set): legs whose pairwise family_error_corr
      >= decorr_thresh are the SAME channel; a channel = a connected component of that "correlated" graph. (The greedy
      maximal-independent-set my PoC used can be order-dependent; connected components is the order-invariant correct form
      -- the clustering @ flagged.)
  (2) INFORMATIVE filter: a channel counts toward n_eff ONLY if it contains >=1 leg that actually DETECTS the fake
      (AUC >= detect_thresh). This excludes the BLIND-DECORRELATION TRAP: a near-chance leg's ~random errors are spuriously
      uncorrelated with everything, so a naive corr-counter would reward it with a phantom channel. (measured: on the content-
      matched splice all pixel legs go ~chance yet pairwise corr ~0.53 -> a naive counter says decorrelated; informative n_eff=0.)

corr entries are the _pairwise_agreement (Spearman concordance in [-1,1]); |corr| is used (anti-correlation is still shared info).
Pure numpy; no substrate. an agent worktree = cert-tooling (allowed).
"""
import math

import numpy as np

__all__ = ["informative_neff", "hanley_mcneil_auc_se", "required_n_for_auc_ci_lo",
           "auc_ci_lo_plateau", "min_binding_n_for_auc_ci_lo", "required_other_n_given_fixed",
           "min_point_auc_for_feasible_coverage", "expected_selection_inflation", "effective_variant_count"]


MIN_N_FLOOR = 30   # ★sample-floor: below this an AUC/corr CI bound is not a trustworthy bound (normal-approx/CLT floor;
# target 100). Mirrors coverage_strata_guard.MIN_N_FLOOR -- both C guards that consume a
# PRE-COMPUTED CI SCALAR (no raw counts) need the floor made explicit; the sibling veto_threshold_
# calibration gets it for free via Wilson widening on raw counts.


def hanley_mcneil_auc_se(auc, n_pos, n_neg):
    """Standard error of an AUC by Hanley & McNeil (1982), the canonical AUC-SE (external anchor). Q1=A/(2-A),
    Q2=2A^2/(1+A); SE = sqrt[(A(1-A) + (n_pos-1)(Q1-A^2) + (n_neg-1)(Q2-A^2)) / (n_pos*n_neg)]. Returns NaN on
    non-finite / non-positive counts (fail-closed)."""
    A = float(auc)
    if not (np.isfinite(A) and n_pos > 0 and n_neg > 0) or not (0.0 <= A <= 1.0):
        return float("nan")
    Q1 = A / (2 - A) if A < 2 else float("nan")
    Q2 = 2 * A * A / (1 + A)
    var = (A * (1 - A) + (n_pos - 1) * (Q1 - A * A) + (n_neg - 1) * (Q2 - A * A)) / (n_pos * n_neg)
    return float(np.sqrt(var)) if var >= 0 else float("nan")


# E[max of K i.i.d. standard normals] -- the expected-max order statistic; drives best-of-K selection inflation.
_EMAX_K = {1: 0.0, 2: 0.5642, 3: 0.8463, 4: 1.0294, 5: 1.1630, 6: 1.2672, 8: 1.4236, 10: 1.5388, 16: 1.7660, 20: 1.8675}


def expected_selection_inflation(k_variants, n_pos, n_neg, auc_null=0.5, rho=0.0):
    """★BEST-OF-K SELECTION INFLATION: the expected upward bias of a pilot AUC obtained by taking the MAX over
    `k_variants` variants, each measured at (n_pos, n_neg) -- distinct from sequential optional-stopping. Under the null
    (variants truly ~auc_null), E[reported max AUC] - auc_null ~= sqrt(1-rho) * E[max of K standard normals] *
    SE_HanleyMcNeil(auc_null, n). This is why a promising best-of-K pilot COLLAPSES on a frozen re-measure (R3
    0.673@14->0.583@56; 0.675 'selection-inflation small-n+best-of-4'). Fix: NEVER believe a best-of-K pilot -- FREEZE an
    independent set, re-test the SELECTED variant ONCE, gate on that CI-lo (or discount the pilot by this inflation).

    rho: ★AVERAGE PAIRWISE CORRELATION of the K variants' AUC estimates (rider). Agent pool variants are usually
        computed on the SAME clips -> their AUC estimates are POSITIVELY correlated, which REDUCES the inflation:
        max_i(sqrt(rho)*Z + sqrt(1-rho)*W_i) has E = sqrt(1-rho)*E[max of K indep] (the shared component Z does not lift
        the max above the mean). So the independence law (rho=0) OVER-estimates for correlated variants -> using it to
        discount OVER-corrects and can FALSE-FENCE a real leg. Pass an estimated rho (from the variants' per-item score
        correlation) for the calibrated discount; rho=0 is the conservative-against-false-POSITIVES default, rho->1 -> 0
        inflation (K identical variants = 1 variant). Validated vs Monte-Carlo for rho in [0, 0.8] to <0.008.

    Returns the expected inflation (>=0), or NaN on bad input. k_variants not in the table -> log-interpolated E[max]."""
    k = int(k_variants)
    if k < 1 or n_pos <= 0 or n_neg <= 0 or not np.isfinite(rho) or rho < 0 or rho >= 1:
        return float("nan")
    se = hanley_mcneil_auc_se(auc_null, n_pos, n_neg)
    if not np.isfinite(se):
        return float("nan")
    if k in _EMAX_K:
        emax = _EMAX_K[k]
    else:  # interpolate E[max of K] ~ sqrt(2 ln K) growth; use table anchors in log-K
        ks = sorted(_EMAX_K); import bisect; j = bisect.bisect_left(ks, k)
        if j == 0:
            emax = _EMAX_K[ks[0]]
        elif j >= len(ks):
            emax = float(np.sqrt(2 * np.log(max(k, 2))))
        else:
            k0, k1 = ks[j - 1], ks[j]; t = (np.log(k) - np.log(k0)) / (np.log(k1) - np.log(k0))
            emax = _EMAX_K[k0] + t * (_EMAX_K[k1] - _EMAX_K[k0])
    return float(np.sqrt(1.0 - rho) * emax * se)


def effective_variant_count(corr_matrix):
    """★EFFECTIVE number of independent variants among K whose AUC estimates have correlation `corr_matrix` (the
    nominal-vs-effective count applied to best-of-K selection -- sibling of informative_neff's effective-channel count).
    K nominal variants computed on the SAME clips are near-duplicates (same-statistic pairs measure ~rho 0.60), so
    the EFFECTIVE number of independent selection draws is K_eff = (sum lambda)^2 / sum(lambda^2) = K^2 / sum(lambda_i^2)
    (participation ratio of the correlation-matrix eigenvalues; = K at rho=0, = 1 at rho=1 equicorrelation). Use K_eff as
    the intuitive summary of how many independent 'tries' the best-of-K really had. Returns float K_eff, or NaN on bad input.

    ★PURPOSE SCOPE (adjudication, C-verified): this is the VARIANCE/SELECTION object -- K_eff in [1, K]. It is
    NOT a FALSE-ACCEPT FLOOR: a false-accept floor n_eff is a JOINT-TAIL EXPONENT (P(all K legs fooled)=p^n_eff), for which
    the correct form at VERDICT-level rho is ~2/(1+rho) (Ledford-Tawn) -- see false_accept_floor_neff. There, n_eff>K IS
    PHYSICAL for anti-correlated verdicts (they almost never both fire). My earlier 'n_eff>K is impossible/artifact' claim
    (body-768) and the non-PSD fence are correct ONLY for THIS variance object; do NOT apply them to a floor exponent.

    NOTE: for the EXACT selection inflation of a HETEROGENEOUS (block) correlation, the single-rho equicorrelation form
    sqrt(1-rho_bar)*E[maxK]*SE (expected_selection_inflation with rho=mean pairwise corr) is a GOOD approximation (within
    ~5% for measured block structures) but not exact -- exactness needs a Gaussian-max Monte-Carlo over the matrix. K_eff is
    the interpretable number; rho_bar is the practical discount input."""
    C = np.asarray(corr_matrix, float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] < 1 or not np.all(np.isfinite(C)):
        return float("nan")
    K = C.shape[0]
    # ★CORRELATION-MATRIX VALIDATION / FAIL-CLOSED (failure-envelope re-sweep): the participation ratio
    # K^2/sum(lambda^2) is defined for a CORRELATION matrix (unit diagonal). A COVARIANCE matrix (diagonal != 1, e.g.
    # [[2,0],[0,2]]) is PSD so it passes the eigenvalue fence below, but the ratio then returns a value OUTSIDE the [1,K]
    # range (0.5 for that input) -- an UNDER-count that understates best-of-K selection inflation (favourable direction).
    # Require a unit diagonal (a correlation matrix); reject a non-unit-diagonal input as not-a-correlation-matrix.
    if not np.allclose(np.diag(C), 1.0, atol=1e-6):
        return float("nan")
    ev = np.linalg.eigvalsh(C)
    # ★NOT-PSD FAIL-CLOSED (cross-substrate over-determination directive, G2 negative-rho artifact): a valid
    # correlation matrix is positive-SEMIDEFINITE (all eigenvalues >= 0). A NEGATIVE eigenvalue means the input is NOT a
    # valid correlation matrix -- a rho-ESTIMATION ARTIFACT (finite-sample / pairwise-deletion / a sign bug), NOT genuine
    # anti-correlation. The participation ratio K^2/sum(lambda^2) still returns a plausible-looking number on such input
    # (fail-OPEN: it credits an effective count from an invalid rho). A small float-noise negative (>= -1e-8) is tolerated;
    # a materially-negative eigenvalue -> NaN (invalid rho, not assessable). Mirrors the row-sum n_eff which DIVERGES to
    # +inf at the PSD boundary rho_bar = -1/(K-1); the participation ratio is bounded but must still fence non-PSD input.
    if float(ev.min()) < -1e-8:
        return float("nan")
    denom = float(np.sum(ev ** 2))
    return float(K * K / denom) if denom > 0 else float("nan")


def false_accept_floor_neff(rho, p=None):
    """★FALSE-ACCEPT FLOOR n_eff = the JOINT-TAIL EXPONENT for K=2 thresholded VERDICTS (P(both fooled at per-leg rate p) =
    p^n_eff). DISTINCT from effective_variant_count (the variance/selection object). Per the adjudication (Ledford-Tawn
    residual tail dependence, C-verified via bivariate orthant MC): for correlated-Gaussian verdicts the p->0 exponent is
    n_eff = 2/(1+rho) -- the SIGNED LINEAR form. Key consequences (opposite to the variance object):
      - rho>0: floor exponent < participation ratio -> the participation ratio OVER-CREDITS a floor (books p^1.60 at rho=0.5
        while the true asymptotic exponent is 1.33), i.e. UNDER-states the false-accept rate. Use THIS for floors.
      - rho<0 (anti-correlated verdicts): 2/(1+rho) > K=2 and diverges as rho->-1 -- and this is PHYSICAL: anti-correlated
        verdicts almost never BOTH fire, so P(both fooled) ~ p^large. The participation ratio (~1 at rho=-0.9) would
        catastrophically under-credit the floor.
    ★HAZARD (JARVIS): feed VERDICT-level rho (correlation of the thresholded accept/reject decisions), NOT reaction/score-
    level rho -- the signed form on score-level rho is catastrophic. rho in (-1, 1]. p optional (documentation only; the
    asymptotic exponent is p-independent). Returns the float floor exponent, or inf as rho->-1, NaN on bad rho."""
    r = float(rho)
    if not np.isfinite(r) or r <= -1.0 or r > 1.0:
        return float("nan")
    denom = 1.0 + r
    return float("inf") if denom <= 0 else float(2.0 / denom)


def source_portability_check(per_source_rho, per_source_n):
    """Is a two-leg decorrelation SOURCE-PORTABLE, or is the common-mode SOURCE-SPECIFIC?
    (operationalizes CLAIM:C-QC-I-CROSSLEG-COMMONMODE-IS-SOURCE-SPECIFIC-NOT-PORTABLE-STRATIFY-DONT-POOL.)

    A decorrelation certified on ONE data source can OVER-credit the agreement-cert on ANOTHER source where the legs share a
    common-mode (measured: flow<->appearance legs decorrelated on kansas rho~0 but common-mode on camerabench rho 0.67). A
    POOLED cross-source correlation averages the two into a misleading middle. Given a leg-pair's VERDICT-level correlation
    measured PER DATA SOURCE, this: (a) computes the false-accept-floor n_eff per source (2/(1+rho)); (b) tests every source-PAIR
    with a Fisher-z difference-of-correlations test -- if ANY pair differs at p<0.05 the decorrelation is NOT source-portable;
    (c) returns the MOST common-mode-heavy source (highest |rho|) and ITS n_eff as the SAFETY bound (never trust the more-
    decorrelated source's higher n_eff for a cross-source cert). per_source_rho: {source: rho}; per_source_n: {source: n}.
    ★HAZARD (same as false_accept_floor_neff): feed VERDICT-level rho, NOT score-level. n<=3 source-pairs are skipped (Fisher-z
    undefined). Returns per_source {rho,n,neff}, source_portable (bool), fisher_z_max_abs/min_p, most_common_mode_source, worst_source_neff."""
    srcs = [s for s in per_source_rho if s in per_source_n and np.isfinite(per_source_rho[s])]
    if len(srcs) < 1:
        return {"source_portable": None, "reason": "no-finite-sources", "n_sources": 0}
    per = {s: {"rho": float(per_source_rho[s]), "n": int(per_source_n[s]),
               "neff": float(false_accept_floor_neff(per_source_rho[s]))} for s in srcs}
    # ★worst = the MINIMUM false-accept floor = the most-common-mode source (SIGN-CORRECT, C red-team). 2/(1+rho)
    # is DECREASING in rho, so min-neff <=> max SIGNED rho <=> most common-mode (dangerous). The old `max(key=abs(rho))` was
    # SIGN-BLIND: a strongly ANTI-correlated source (rho<0, SAFE, HIGH floor) could out-rank a mildly common-mode one (rho>0,
    # dangerous, LOW floor) and its GENEROUS floor got reported as 'the safety bound' -- a FALSE-ACCEPT overstatement (measured
    # 9.3x on rho={kansas:+0.4, cam:-0.85}: reported 13.33 vs the true dangerous 1.43). false_accept_floor_neff already encodes
    # the danger via the sign -> take its MIN (matches the sibling ecc_admission_precondition_by_source's min(neffs)).
    worst = min(srcs, key=lambda s: per[s]["neff"])
    not_portable = False
    max_abs_z = 0.0
    min_p = 1.0
    n_comparisons = 0
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            r1, n1 = per[srcs[i]]["rho"], per[srcs[i]]["n"]
            r2, n2 = per[srcs[j]]["rho"], per[srcs[j]]["n"]
            if n1 <= 3 or n2 <= 3:
                continue
            n_comparisons += 1
            z = (np.arctanh(np.clip(r1, -0.999, 0.999)) - np.arctanh(np.clip(r2, -0.999, 0.999))) / np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
            p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(float(z)) / math.sqrt(2.0))))
            max_abs_z = max(max_abs_z, abs(float(z)))
            min_p = min(min_p, float(p))
            if p < 0.05:
                not_portable = True
    # ★VACUOUS-TRUTH FENCE (C red-team): portability cannot be CERTIFIED if NO Fisher-z comparison could be made --
    # a single source (no cross-source question), or every pair n<=3-starved. The old code left not_portable=False ->
    # source_portable=True with min_p=1.0, numerically indistinguishable from a genuinely-run null-result = a VACUOUS pass
    # (claims 'safe to reuse another source's decorrelation' with zero statistical power). ABSTAIN (source_portable=None),
    # matching informative_neff.pair_starved's conservative discipline (a starved pair cannot certify).
    if n_comparisons == 0:
        return {"per_source": per, "source_portable": None,
                "reason": ("single-source-no-cross-source-test" if len(srcs) < 2 else "all-source-pairs-n<=3-starved-abstain"),
                "fisher_z_max_abs": 0.0, "fisher_z_min_p": None, "most_common_mode_source": worst,
                "worst_source_neff": per[worst]["neff"], "n_sources": len(srcs), "n_comparisons": 0}
    return {"per_source": per, "source_portable": bool(not not_portable),
            "fisher_z_max_abs": round(max_abs_z, 3), "fisher_z_min_p": round(min_p, 5),
            "most_common_mode_source": worst, "worst_source_neff": per[worst]["neff"], "n_sources": len(srcs),
            "n_comparisons": n_comparisons}


def _rankdata(a):
    """Average-rank of a 1D array (numpy-only Spearman helper). Ties get mean rank."""
    a = np.asarray(a, float)
    # ★ARGSORT-NAN-FABRICATION INNER-PRIMITIVE HARDENING (exhaustive re-sweep of the class): argsort sorts
    # a NaN to the END (max rank) -> a downstream rank correlation FABRICATES on a non-finite input. The only shipped caller
    # (nonlinear_dependence_audit) already raises on non-finite BEFORE this; fail LOUD as a second-order guard for any
    # future caller rather than silently return a fabricated ranking.
    if not np.all(np.isfinite(a)):
        raise ValueError("_rankdata: non-finite input -> ranking would FABRICATE (NaN sorts to max rank); "
                         "the caller must guard non-finite before ranking")
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    r = np.arange(1, len(a) + 1, dtype=float)
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        r[i:j + 1] = r[i:j + 1].mean()
        i = j + 1
    ranks[order] = r
    return ranks


def distance_correlation(a, b, max_n=800, seed=0):
    """Szekely distance correlation dCor(a,b) in [0,1]: 0 IFF a,b are INDEPENDENT (unlike Pearson/|dev|-rank, it detects
    ANY dependence -- including NON-MONOTONIC / folded coupling that |dev|-rank misses, e.g. y=2x^2-1 on bounded support).
    O(m^2) in the subsample size m -- subsampled to max_n rows (deterministic seed) for large n. numpy-only.
    Returns a float in [0,1] (NaN on degenerate/constant input)."""
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    if a.shape != b.shape or a.size < 4 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    m = a.size
    if m > max_n:
        idx = np.random.default_rng(seed).choice(m, max_n, replace=False)
        a = a[idx]; b = b[idx]
    A = np.abs(a[:, None] - a[None, :]); B = np.abs(b[:, None] - b[None, :])
    A = A - A.mean(axis=0, keepdims=True) - A.mean(axis=1, keepdims=True) + A.mean()
    B = B - B.mean(axis=0, keepdims=True) - B.mean(axis=1, keepdims=True) + B.mean()
    dcov2 = float((A * B).mean()); dvarA = float((A * A).mean()); dvarB = float((B * B).mean())
    if dvarA <= 0 or dvarB <= 0:
        return float("nan")
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(np.sqrt(dvarA * dvarB)))


def nonlinear_dependence_audit(leg_scores, rho_thresh=0.2, nl_thresh=0.2, dcorr_thresh=0.30):
    """★METHOD-DIVERSITY probe (operationalizes the cross-substrate METHOD-COMMON-MODE limit, capstone
    spine_c_cross_substrate_agreement_certifies_object_not_method, ...): the rho-based n_eff (effective_variant_count) is
    LINEAR-correlation based, so it is BLIND to a NONLINEAR dependence with rho~0 (a fully-dependent pair y=x^2-1 reports
    n_eff~2). Before crediting an over-determination / multi-leg decorrelation, check whether any pair is rho-DECORRELATED
    but NONLINEARLY DEPENDENT = method-common-mode (one leg is a nonlinear function of the other's signal -> not an
    independent method). The nonlinear-dependence proxy is Spearman(|a - median(a)|, |b - median(b)|) -- a rank correlation
    of the |deviation-from-median|, which catches EVEN-symmetric dependence (x^2, |x|) that Pearson/Spearman on the raw
    values miss (their rho ~ 0). This is C's shared-bias theorem at the leg level: rho-agreement does not certify method
    independence; a decorrelated METHOD (this nonlinear probe) is the escape.

    leg_scores: (K, N) array or dict {name: (N) scores}. Returns dict:
      rho_neff (participation-ratio n_eff on the Pearson matrix -- what a naive decorr check reports),
      flagged_pairs [(i_or_name, j_or_name, |rho|, dev_rank_dep, distance_corr), ...] where |rho|<rho_thresh AND
        (dev_rank_dep>=nl_thresh OR distance_corr>=dcorr_thresh) -- dCor also catches NON-MONOTONIC/folded coupling,
      method_common_mode (bool: any flagged pair), n_pairs, verdict.
    Fails closed (raises ValueError) on <2 legs, mismatched lengths, non-finite, or a constant (zero-variance) leg."""
    if isinstance(leg_scores, dict):
        names = list(leg_scores.keys()); X = [np.asarray(leg_scores[k], float) for k in names]
    else:
        X = [np.asarray(r, float) for r in np.atleast_2d(np.asarray(leg_scores, float))]
        names = list(range(len(X)))
    K = len(X)
    if K < 2:
        raise ValueError("need >=2 legs to audit pairwise dependence")
    n = X[0].shape[0]
    for x in X:
        if x.ndim != 1 or x.shape[0] != n:
            raise ValueError("all legs must be 1D and the same length")
        if not np.all(np.isfinite(x)):
            raise ValueError("non-finite leg scores -> uncomputable dependence; fail-closed")
        if float(np.std(x)) < 1e-12:
            raise ValueError("a constant (zero-variance) leg -> dependence undefined; fail-closed")
    rho = np.corrcoef(np.array(X))
    rho_neff = effective_variant_count(rho)
    dev_ranks = [_rankdata(np.abs(x - np.median(x))) for x in X]
    flagged = []
    for i in range(K):
        for j in range(i + 1, K):
            rij = abs(float(rho[i, j]))
            if rij >= rho_thresh:
                continue                                       # rho already sees it -> not a rho-decorrelated surprise
            # (1) |dev|-rank = Spearman of |deviations| -- catches MONOTONIC magnitude coupling (y=x^2, |x|)
            a, b = dev_ranks[i], dev_ranks[j]
            nl = abs(float(np.corrcoef(a, b)[0, 1])) if (np.std(a) > 0 and np.std(b) > 0) else 0.0
            # (2) distance correlation -- catches ANY dependence incl NON-MONOTONIC/folded coupling |dev|-rank MISSES
            # (blind-spot dogfood: y=2x^2-1 on bounded support has rho~0 AND |dev|-rank~0 but dCor~0.45). relies
            # on this audit for their gate, so the audit must not inherit the |dev|-rank null space.
            dc = distance_correlation(X[i], X[j])
            dc_flag = (np.isfinite(dc) and dc >= dcorr_thresh)
            if nl >= nl_thresh or dc_flag:
                flagged.append((names[i], names[j], round(rij, 3), round(nl, 3), round(float(dc), 3) if np.isfinite(dc) else None))
    mcm = len(flagged) > 0
    verdict = ("METHOD-COMMON-MODE-DETECTED-rho-decorrelated-but-nonlinearly-dependent" if mcm
               else "NO-NONLINEAR-DEPENDENCE-rho-decorrelation-corroborated")
    return {"rho_neff": rho_neff, "flagged_pairs": flagged, "method_common_mode": bool(mcm),
            "n_pairs": K * (K - 1) // 2, "verdict": verdict}


def required_n_for_auc_ci_lo(auc, bar=0.65, z=1.96, balanced=True, max_n=200000):
    """★The COMPANION of the sample-floor: given an observed leg AUC, the per-class n at which its AUC CI-LOWER bound
    (point - z*SE_HanleyMcNeil) first reaches `bar` -- i.e. the n needed for the leg to certify DETECTION on the CI-lo
    (the winner's-curse-safe gate C's neff/coverage enforce). Turns 'grow n and re-run' into a TARGETED acquisition count.
    Assumes the point AUC HOLDS as n grows (a caveat: if AUC drifts down, required n rises steeply -- caller should
    re-project). balanced -> n_pos=n_neg=n. Returns the int n, or None if AUC<=bar (can never clear) / not reached by max_n."""
    A = float(auc)
    if not np.isfinite(A) or A <= bar:
        return None            # a point AUC at/below the bar can never have its CI-LOWER bound clear it
    n = 5
    while n <= max_n:
        se = hanley_mcneil_auc_se(A, n, n if balanced else n)
        if np.isfinite(se) and (A - z * se) >= bar:
            return n
        n += 1
    return None


def min_point_auc_for_feasible_coverage(bar, n_budget, z=1.96, step=0.001):
    """★FEASIBILITY of a CI-lo bar under a fixed acquisition BUDGET. The smallest point AUC whose CI-LOWER bound reaches
    `bar` within n_budget samples/class -- i.e. the leg-quality floor you MUST have to certify at `bar` given the data you
    can realistically get. Because required-n explodes as the point AUC approaches the bar (a 0.80 CI-lo bar needs ~1200/
    class at point AUC 0.817 but ~76 at 0.86), a high bar under a tight budget demands a STRONG leg, not just 'more n'.
    Returns the minimum point AUC (float), or None if even AUC->1 cannot reach `bar` within n_budget (bar itself too high)."""
    A = bar + step
    while A < 1.0:
        rn = required_n_for_auc_ci_lo(A, bar=bar, z=z)
        if rn is not None and rn <= n_budget:
            return round(A, 4)
        A += step
    return None


def auc_ci_lo_plateau(auc, n_scarce, scarce_is_pos=True, z=1.96):
    """★UNBALANCED-corpus floor. The BEST achievable AUC CI-LOWER bound for a fixed SCARCE-class count as the ABUNDANT
    class -> infinity. As n_abundant->inf the Hanley-McNeil variance -> (Q-A^2)/n_scarce (Q2 if pos scarce, Q1 if neg
    scarce) -- a FLOOR set by the scarce class alone. So if this plateau is below the bar, NO amount of the abundant class
    can certify: acquire the SCARCE (binding) class. Returns the plateau ci_lo (NaN on bad input)."""
    A = float(auc)
    if not (np.isfinite(A) and 0.0 <= A <= 1.0 and n_scarce > 0):
        return float("nan")
    Q1 = A / (2 - A); Q2 = 2 * A * A / (1 + A)
    var_floor = ((Q2 if scarce_is_pos else Q1) - A * A) / n_scarce
    return float(A - z * np.sqrt(var_floor)) if var_floor >= 0 else float("nan")


def min_binding_n_for_auc_ci_lo(auc, bar=0.65, scarce_is_pos=True, z=1.96, max_n=200000):
    """★The BINDING-class minimum: the smallest SCARCE-class n whose PLATEAU ci_lo (abundant class -> inf) reaches `bar`.
    Below this n, the bar is UNREACHABLE no matter how much of the abundant class is acquired -> the acquisition strategy
    MUST target the scarce class. Returns the int n, or None if AUC<=bar (never clears) / not reached by max_n."""
    A = float(auc)
    if not np.isfinite(A) or A <= bar:
        return None
    n = 2
    while n <= max_n:
        if auc_ci_lo_plateau(A, n, scarce_is_pos, z) >= bar:
            return n
        n += 1
    return None


def required_other_n_given_fixed(auc, n_fixed, fixed_is_pos=True, bar=0.65, z=1.96, max_n=2000000):
    """★UNBALANCED acquisition target. Given a FIXED count for one class, the count of the OTHER class needed for the AUC
    CI-lower bound to reach `bar`. Returns the int other-n, or None if AUC<=bar, or the string 'UNREACHABLE' when the
    FIXED class is itself below its binding floor (its plateau ci_lo < bar) so no amount of the other class can clear ->
    the caller must instead grow the FIXED (scarce) class. This is the strategy correction for real unbalanced corpora."""
    A = float(auc)
    if not np.isfinite(A) or A <= bar:
        return None
    # if the fixed class is the SCARCE binding one and already below its plateau floor -> UNREACHABLE via the other class
    if auc_ci_lo_plateau(A, n_fixed, scarce_is_pos=fixed_is_pos, z=z) < bar:
        return "UNREACHABLE"
    n_other = 2
    while n_other <= max_n:
        se = hanley_mcneil_auc_se(A, n_fixed if fixed_is_pos else n_other,
                                  n_other if fixed_is_pos else n_fixed)
        if np.isfinite(se) and (A - z * se) >= bar:
            return n_other
        n_other += 1
    return None


def informative_neff(leg_aucs, corr, detect_thresh=0.65, decorr_thresh=0.50, leg_auc_ci_lo=None, corr_ci_hi=None,
                     leg_n=None, pair_n=None, min_n_floor=MIN_N_FLOOR):
    """EFFECTIVE independent-channel count for a set of legs on one axis.

    leg_aucs: dict {leg_name: injection_AUC (splice-vs-clean or fake-vs-clean, measured on the ADVERSARIAL MINIMUM)}.
    corr: dict {"legA|legB": family_error_corr in [-1,1]} for each unordered pair (order within the key irrelevant --
               both "A|B" and "B|A" are accepted).
    detect_thresh: AUC below which a leg does NOT detect (contributes no information) -> blind. Default 0.65.
    decorr_thresh: |family_error_corr| at/above which two legs are the SAME channel (correlated). Default 0.50 (FAMILY_DECORR_THRESH).
    leg_auc_ci_lo: optional dict {leg_name: AUC CI-LOWER bound}. ★SIGNIFICANCE-CHECK (agent pool CI-lo discipline,
        significance_checked_min_power /): when provided, a leg counts as DETECTING iff its AUC CI-LOWER
        bound >= detect_thresh (NOT the point AUC). This closes the fixed-magnitude-no-significance-test class: a
        winner's-curse leg (point-AUC 0.66 at small n, CI-lo 0.40) would otherwise inflate n_eff. WITHOUT it the point
        AUC is used (backward-compatible) -- callers SHOULD pass CI-robust AUCs to avoid winner's-curse over-counting.

    Returns dict: n_eff (int, number of channels that contain a detecting leg), channels (list of leg-name lists =
    connected components), detecting_channels (the counted components), blind_excluded (legs in no counted channel),
    nominal_leg_count (len(legs)) -- so n_eff < nominal exposes shared-substrate / blind over-counting.
    """
    legs = sorted(leg_aucs)
    idx = {L: i for i, L in enumerate(legs)}
    n = len(legs)
    if n == 0:
        return {"n_eff": 0, "channels": [], "detecting_channels": [], "blind_excluded": [], "nominal_leg_count": 0}

    def getcorr(a, b):
        return corr.get("%s|%s" % (a, b), corr.get("%s|%s" % (b, a)))

    # union-find over the CORRELATED graph (edge iff |corr| >= decorr_thresh) -> connected components = channels
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def getcorr_hi(a, b):
        if corr_ci_hi is not None:
            v = corr_ci_hi.get("%s|%s" % (a, b), corr_ci_hi.get("%s|%s" % (b, a)))
            if v is not None:
                return v
        # ★AUTO-DERIVE the |corr| CI-upper from the point corr + pair_n (Fisher-z) when a precomputed CI is absent but the
        # pair's n is known (follow-on to the tuning-knob threshold-provenance review): the decorr_thresh merge
        # is a FIXED threshold on a NOISY statistic (rank corr, sampling SE ~1/sqrt(n-3)) -> regime-blind. The significance-
        # check that closes it (merge unless decorrelation is PROVEN by the |corr| CI-upper < thresh) was OPT-IN (only when
        # the caller precomputed corr_ci_hi); a caller supplying pair_n but not corr_ci_hi silently reverted to the point-corr
        # comparison -> a winner's-curse LOW sample |corr| (true-correlated pair, small n) SPLIT into separate channels ->
        # n_eff INFLATED (fail-open). Deriving the CI-upper from n makes the check AUTOMATIC and n-anchored: conservative
        # (merge) at small n, decisive (split) once n proves decorrelation. Merging is the SAFE direction (lower n_eff).
        # Without pair_n (or n<=3) -> None -> unchanged point-corr fallback (backward-compatible).
        if pair_n is not None:
            pc = getcorr(a, b)
            npv = pair_n.get("%s|%s" % (a, b), pair_n.get("%s|%s" % (b, a)))
            if pc is not None and np.isfinite(float(pc)) and npv is not None and np.isfinite(float(npv)) and float(npv) > 3:
                se = 1.0 / np.sqrt(float(npv) - 3.0)
                return float(np.tanh(np.arctanh(np.clip(abs(float(pc)), 0.0, 0.999)) + 1.96 * se))
        return None

    def pair_starved(a, b):
        # ★SAMPLE-FLOOR (real-control-starved, mirrors coverage_strata_guard): decorrelation can be PROVEN (a pair
        # kept in SEPARATE channels -> higher n_eff) only from an adequately-powered pair. When pair_n is supplied, a pair
        # whose n is missing / non-finite / below the floor CANNOT prove decorrelation -> force SAME channel (conservative
        # merge = LOWER n_eff, the safe direction). Without pair_n the behaviour is unchanged (backward-compatible).
        if pair_n is None:
            return False
        npv = pair_n.get("%s|%s" % (a, b), pair_n.get("%s|%s" % (b, a)))
        return npv is None or not np.isfinite(float(npv)) or float(npv) < min_n_floor

    for a in range(n):
        for b in range(a + 1, n):
            c = getcorr(legs[a], legs[b])
            cu = getcorr_hi(legs[a], legs[b])
            # ★NULL-SAFETY / FAIL-CLOSED (§4c): a NaN pairwise correlation -> treated as DECORRELATED ->
            # SEPARATE channels -> n_eff INFLATED. Unmeasurable corr canNOT certify decorrelation -> SAME channel.
            # ★SIGNIFICANCE-CHECK: a POINT corr below decorr_thresh may be a
            # WINNER'S-CURSE low value of a truly-correlated pair (its |corr| CI-UPPER reaches the threshold). Merging
            # only on the point corr would count them as SEPARATE channels -> n_eff INFLATED. When corr_ci_hi is
            # provided, treat the pair as the SAME channel unless decorrelation is PROVEN (|corr| CI-upper < thresh):
            # merge if the point |corr| >= thresh OR its CI-upper >= thresh (correlation not excludable).
            # ★ABSENT-PAIR / FAIL-CLOSED (failure-envelope wave): an ABSENT pair key (getcorr -> None) is as
            # unmeasurable as a NaN corr -- it CANNOT certify decorrelation. The original test `c is not None and...`
            # let an absent pair fall through to SEPARATE channels -> n_eff INFLATED (the exact null-safety hole the NaN
            # branch documents, one step upstream). Treat an absent point corr as SAME channel unless its CI-upper proves
            # otherwise (a provided cu can still resolve it), mirroring the NaN merge.
            cu_excludes = (cu is not None and np.isfinite(float(cu)) and float(cu) < decorr_thresh)
            same_channel = (c is None and not cu_excludes) \
                or (c is not None and (not np.isfinite(float(c)) or abs(float(c)) >= decorr_thresh)) \
                or (cu is not None and np.isfinite(float(cu)) and float(cu) >= decorr_thresh)
            # ★SAMPLE-FLOOR: if the pair is NOT already same-channel, decorrelation is being asserted -> it must rest on an
            # adequately-powered pair, else force the conservative merge (a starved pair cannot certify decorrelation).
            if not same_channel and pair_starved(legs[a], legs[b]):
                same_channel = True
            if same_channel:
                parent[find(a)] = find(b)

    comps = {}
    for L in legs:
        comps.setdefault(find(idx[L]), []).append(L)
    channels = [sorted(v) for v in comps.values()]

    def detects(L):
        # ★SAMPLE-FLOOR: a leg counts as DETECTING only from an adequately-powered measurement -- a CI-lo (or AUC) from a
        # starved stratum is not a trustworthy bound (real-control-starved; mirrors coverage_strata_guard). When
        # leg_n is supplied, a leg with missing / non-finite / below-floor n does NOT detect (under-count = safe direction,
        # never inflate n_eff on thin data). Without leg_n the behaviour is unchanged (backward-compatible).
        if leg_n is not None:
            nv = leg_n.get(L)
            if nv is None or not np.isfinite(float(nv)) or float(nv) < min_n_floor:
                return False
        # ★RANGE-VALIDATION / FAIL-CLOSED (failure-envelope re-sweep): an AUC (and its CI-lower bound) is a
        # probability in [0,1]. The `>= detect_thresh` test is one-sided, so an IMPOSSIBLE supra-unit value (AUC=3.0, CI-lo
        # =1.7 from a mis-scaled input) sails past it and is credited as a DETECTING channel, inflating n_eff. The sibling
        # hanley_mcneil_auc_se fences 0<=A<=1; mirror it here -- a value outside [0,1] is corrupt -> does NOT detect.
        # ★significance-check: gate on the AUC CI-LOWER bound when provided (winner's-curse guard), else the point AUC.
        # ★PARTIAL-DICT FAIL-CLOSED (C red-team): once the caller OPTS INTO CI-lo significance (leg_auc_ci_lo is not
        # None), a leg MISSING from the dict must NOT silently fall through to its winner's-curse POINT AUC -- that reverts to
        # exactly the behaviour the CI-lo param exists to CLOSE and inflates n_eff on an unproven leg (measured: omitting 'weak'
        # -> counted on its 0.66 point AUC, n_eff 2, vs n_eff 1 when its true CI-lo 0.40 is supplied). An ABSENT CI-lo = no
        # significance evidence -> does NOT detect (conservative, mirrors the leg_n missing-key fail-closed above). The point-AUC
        # path is used ONLY when leg_auc_ci_lo is None entirely (the documented backward-compatible all-or-nothing case).
        if leg_auc_ci_lo is not None:
            v = leg_auc_ci_lo.get(L)
            return v is not None and np.isfinite(v) and (detect_thresh <= float(v) <= 1.0)
        a = leg_aucs.get(L)
        return a is not None and np.isfinite(a) and (detect_thresh <= float(a) <= 1.0)

    detecting_channels = [ch for ch in channels if any(detects(L) for L in ch)]
    counted = {L for ch in detecting_channels for L in ch}
    blind_excluded = [L for L in legs if L not in counted]
    return {"n_eff": len(detecting_channels), "channels": channels, "detecting_channels": detecting_channels,
            "blind_excluded": blind_excluded, "nominal_leg_count": n}


def _selftest():
    ok = tot = 0
    # (1) three MUTUALLY-CORRELATED detecting legs -> ONE channel -> n_eff=1 (the video-splice PoC shape: all Farneback-flow)
    r = informative_neff({"df": 0.84, "md": 0.72, "app": 0.88},
                         {"df|md": 0.63, "df|app": 0.80, "md|app": 0.55})
    tot += 1; ok += (r["n_eff"] == 1 and len(r["channels"]) == 1)
    # (2) two MUTUALLY-DECORRELATED detecting legs -> TWO channels -> n_eff=2 (@ motion-axis shape)
    r = informative_neff({"legA": 0.85, "legB": 0.80}, {"legA|legB": 0.20})
    tot += 1; ok += (r["n_eff"] == 2)
    # (3) BLIND-DECORRELATION TRAP: a near-chance leg decorrelated from a detector -> its channel has NO detector -> excluded
    r = informative_neff({"det": 0.85, "blind": 0.52}, {"det|blind": 0.10})
    tot += 1; ok += (r["n_eff"] == 1 and r["blind_excluded"] == ["blind"])
    # (4) all near-chance on the hard mode -> n_eff=0 even though pairwise corr looks decorrelated (content-matched splice)
    r = informative_neff({"df": 0.648, "md": 0.631, "app": 0.633},
                         {"df|md": 0.589, "df|app": 0.765, "md|app": 0.534})
    tot += 1; ok += (r["n_eff"] == 0)
    # (5) connected-components: A~B correlated, B~C correlated, A~C decorrelated -> transitively ONE channel (greedy would err)
    r = informative_neff({"A": 0.8, "B": 0.8, "C": 0.8}, {"A|B": 0.7, "B|C": 0.7, "A|C": 0.2})
    tot += 1; ok += (r["n_eff"] == 1 and len(r["channels"]) == 1)
    # (6) anti-correlation counts as shared info (|corr|): two legs at corr -0.8 -> same channel
    r = informative_neff({"A": 0.8, "B": 0.8}, {"A|B": -0.8})
    tot += 1; ok += (r["n_eff"] == 1)
    # (7) empty
    tot += 1; ok += (informative_neff({}, {})["n_eff"] == 0)
    # (8) SAMPLE-FLOOR detection: a leg whose CI-lo clears the bar but is STARVED (n=18) does NOT count -> n_eff drops.
    # Without leg_n both count (n_eff=2); with leg_n the starved leg is excluded (n_eff=1).
    r_nofloor = informative_neff({"real": 0.85, "starved": 0.80}, {"real|starved": 0.20},
                                 leg_auc_ci_lo={"real": 0.78, "starved": 0.70})
    r_floor = informative_neff({"real": 0.85, "starved": 0.80}, {"real|starved": 0.20},
                               leg_auc_ci_lo={"real": 0.78, "starved": 0.70}, leg_n={"real": 400, "starved": 18})
    tot += 1; ok += (r_nofloor["n_eff"] == 2 and r_floor["n_eff"] == 1 and r_floor["blind_excluded"] == ["starved"])
    # (9) SAMPLE-FLOOR decorrelation: two detecting legs measured decorrelated on a STARVED pair (n=18) cannot PROVE
    # decorrelation -> conservative merge -> n_eff=1 (vs 2 when the pair is adequately powered).
    r_pair_starved = informative_neff({"A": 0.85, "B": 0.80}, {"A|B": 0.20},
                                      leg_n={"A": 400, "B": 400}, pair_n={"A|B": 18})
    r_pair_ok = informative_neff({"A": 0.85, "B": 0.80}, {"A|B": 0.20},
                                 leg_n={"A": 400, "B": 400}, pair_n={"A|B": 200})
    tot += 1; ok += (r_pair_starved["n_eff"] == 1 and r_pair_ok["n_eff"] == 2)
    # (10) SAMPLE-FLOOR fail-closed: a detecting leg with MISSING n does not count (caller cannot hide a thin leg).
    r_missing = informative_neff({"real": 0.85, "x": 0.80}, {"real|x": 0.20},
                                 leg_auc_ci_lo={"real": 0.78, "x": 0.70}, leg_n={"real": 400})
    tot += 1; ok += (r_missing["n_eff"] == 1 and r_missing["blind_excluded"] == ["x"])
    # (11) Hanley-McNeil AUC-SE ANCHOR: reproduces the KRONAN measured ci_lo (NCC AUC 0.718, n=25/class -> ci_lo 0.576).
    se = hanley_mcneil_auc_se(0.718, 25, 25); cilo = 0.718 - 1.96 * se
    tot += 1; ok += (abs(cilo - 0.576) < 0.005)
    # (12) required_n_for_auc_ci_lo: NCC (0.718) reaches 0.60 CI-lo at 37/class; a higher AUC needs FEWER n; an AUC at/below
    # the bar can NEVER clear (None).
    tot += 1; ok += (required_n_for_auc_ci_lo(0.718, bar=0.60) == 37
                     and required_n_for_auc_ci_lo(0.80, bar=0.60) < 37
                     and required_n_for_auc_ci_lo(0.60, bar=0.60) is None)
    # (13) UNBALANCED plateau: flow AUC 0.701, scarce class n=25 -> plateau ci_lo ~0.586 < 0.60 (abundant class can't rescue);
    # the binding scarce-n where the plateau reaches 0.60 is ~33; below it required_other_n_given_fixed -> UNREACHABLE.
    plat25 = auc_ci_lo_plateau(0.701, 25); binding = min_binding_n_for_auc_ci_lo(0.701, bar=0.60)
    unreach = required_other_n_given_fixed(0.701, n_fixed=25, fixed_is_pos=True, bar=0.60)
    reachable = required_other_n_given_fixed(0.701, n_fixed=40, fixed_is_pos=True, bar=0.60)
    tot += 1; ok += (abs(plat25 - 0.586) < 0.005 and 30 <= binding <= 36 and unreach == "UNREACHABLE"
                     and isinstance(reachable, int) and reachable > 0)
    # (14) COVERAGE-BAR FEASIBILITY: the 0.80 CI-lo bar needs ~1200/class at point AUC 0.817 but a modest n at 0.86+;
    # under a KRONAN-scale budget (69/class) the min point AUC to COVER is well above the bar (a strong-leg floor).
    n1202 = required_n_for_auc_ci_lo(0.817, bar=0.80); n76 = required_n_for_auc_ci_lo(0.86, bar=0.80)
    min_auc69 = min_point_auc_for_feasible_coverage(0.80, 69)
    tot += 1; ok += (n1202 is not None and n1202 > 1000 and n76 is not None and n76 < 100
                     and min_auc69 is not None and 0.80 < min_auc69 < 0.90)
    # (15) NOT-PSD FAIL-CLOSED (cross-substrate over-determination directive): effective_variant_count on a materially
    # non-PSD "correlation matrix" (a rho-estimation artifact) -> NaN (was fail-open: a plausible count on invalid rho).
    # A valid PSD equicorrelation matrix still computes; a NaN-diagonal-noise negative (>=-1e-8) is tolerated.
    def _urho(K, r):
        M = np.full((K, K), r); np.fill_diagonal(M, 1.0); return M
    psd_ok = effective_variant_count(_urho(5, 0.5))          # PSD -> ~2.5
    notpsd = effective_variant_count(_urho(5, -0.30))        # min eig -0.20 -> NOT PSD -> NaN
    tot += 1; ok += (np.isfinite(psd_ok) and 2.4 < psd_ok < 2.6 and not np.isfinite(notpsd))
    # (16) NONLINEAR-DEPENDENCE audit (method-common-mode probe): a nonlinear-dependent pair (y=x^2-1, rho~0) is FLAGGED
    # as method-common-mode even though rho_neff reports ~independent; genuinely independent legs are NOT flagged;
    # a constant leg fails closed.
    _rng = np.random.default_rng(5); _x = _rng.normal(size=4000)
    _mcm = nonlinear_dependence_audit({"a": _x, "b": _x ** 2 - 1.0, "c": _rng.normal(size=4000)})
    _ind = nonlinear_dependence_audit({"a": _rng.normal(size=4000), "b": _rng.normal(size=4000)})
    _raised = False
    try:
        nonlinear_dependence_audit({"a": _x, "b": np.ones(4000)})
    except ValueError:
        _raised = True
    tot += 1; ok += (_mcm["method_common_mode"] is True and _mcm["rho_neff"] > 2.5
                     and _ind["method_common_mode"] is False and _raised)
    # (17) DISTANCE-CORRELATION closes the |dev|-rank NULL SPACE (dogfood: my own nonlinear audit had a blind spot). A
    # FOLDED coupling y=cos(2th) = 2cos^2(th)-1 = 2x^2-1 on bounded (arcsine) support has rho~0 AND |dev|-rank~0 (the
    # |deviation| relation is NON-MONOTONIC) but is DETERMINISTICALLY dependent -- |dev|-rank alone MISSED it; the
    # distance-correlation leg CATCHES it (dCor~0.45). Independent legs stay unflagged (dCor bias << threshold).
    _th = np.random.default_rng(2).uniform(0, 2 * np.pi, 4000)
    _fold = nonlinear_dependence_audit({"a": np.cos(_th), "b": np.cos(2 * _th)})
    _dc_ind = distance_correlation(np.random.default_rng(3).normal(size=4000), np.random.default_rng(4).normal(size=4000))
    tot += 1; ok += (_fold["method_common_mode"] is True and np.isfinite(_dc_ind) and _dc_ind < 0.2)
    # (18) FALSE-ACCEPT FLOOR n_eff (adjudication, C-verified): DISTINCT purpose from the participation ratio -- a floor
    # is a JOINT-TAIL EXPONENT 2/(1+rho). rho=0.5 -> 1.333 (< participation 1.600, which OVER-credits a floor); rho=-0.9
    # -> 20 >> K=2 which is PHYSICAL for anti-correlated verdicts (participation says ~1.1, under-crediting the floor);
    # rho=0 -> both agree at 2; rho<=-1 or >1 -> NaN (bad).
    _fl05 = false_accept_floor_neff(0.5); _flm9 = false_accept_floor_neff(-0.9); _fl0 = false_accept_floor_neff(0.0)
    tot += 1; ok += (abs(_fl05 - 4.0 / 3.0) < 1e-6 and _fl05 < effective_variant_count([[1, 0.5], [0.5, 1]])
                     and _flm9 > 2.0 and abs(_fl0 - 2.0) < 1e-9 and not np.isfinite(false_accept_floor_neff(-1.5)))
    # (19) FAILURE-ENVELOPE-WAVE regression: informative_neff must treat an ABSENT pair key as unmeasurable
    # (SAME channel, like NaN) -- previously an absent corr fell through to SEPARATE channels -> n_eff INFLATED.
    # Absent-with-no-CI -> merge (n_eff 1); absent but CI-upper PROVES decorr (cu=0.1<thresh) -> separate (n_eff 2);
    # a normal decorrelated point corr still separates (n_eff 2).
    _abs_merge = informative_neff({"A": 0.9, "B": 0.9}, {})["n_eff"]
    _abs_ci = informative_neff({"A": 0.9, "B": 0.9}, {}, corr_ci_hi={"A|B": 0.1})["n_eff"]
    _pt_decorr = informative_neff({"A": 0.9, "B": 0.9}, {"A|B": 0.1})["n_eff"]
    tot += 1; ok += (_abs_merge == 1 and _abs_ci == 2 and _pt_decorr == 2)
    print("neff_channel_diversity selftest: %d/%d PASS (3-corr->1, 2-decorr->2, blind-trap-excluded, all-blind->0, "
          "connected-components-transitive, anti-corr-shared, empty, sample-floor-detect, sample-floor-decorr, "
          "sample-floor-missing-n-fail-closed)" % (ok, tot))
    return ok == tot


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
