"""identifiability_oed -- one callable tool compressing H's observable-selection + OED cert findings (session ).

The recurring FORM across substrates (compress-not-accrete of: anchor-OED main-mission node 1; optics n,k single-
reflectance rank-1 d1bbd0214; JWST wavefront phase-diversity 8cc84a75a + optimal-defocus OED 7db3eb913; battery
sim-conditioning identifiability 3b766fcb2):

  A parameter/mode vector theta is observed through a CHOICE of observables. Each observable o contributes a Fisher
  F_o(theta). The IDENTIFIABILITY of theta from a chosen SET S is sigma_min(sum_{o in S} F_o) -- the worst-identified
  mode. A single/coarse observable is generically RANK-DEFICIENT (sigma_min ~ 0); observation DIVERSITY lifts it.
  This tool answers, for a candidate pool of observables:
    * identifiability(S) = sigma_min of the summed Fisher  (report-the-bound: also the worst-mode direction)
    * greedy_oed(pool, k) = the k observables that MAXIMIZE sigma_min (which measurements to add)
    * optimal_amount(param_fn, grid) = the diversity AMOUNT that peaks sigma_min (SELECT don't MAXIMIZE: an interior
      optimum when too-much diversity loses SNR) -- the JWST defocus d* finding, generalized.

Numpy-only, no project deps. Each F_o is a (K,K) symmetric PSD matrix (the caller builds it from its physics:
reproj Jacobian J^T J, ellipsometry d(psi,Delta)/d(n,k), PSF Poisson Fisher, relaxation-timescale Fisher,...).
"""
import numpy as np

def identifiability(fishers):
    """sigma_min + worst-mode direction of the summed Fisher over a set of observables.
    fishers: iterable of (K,K) arrays. Returns dict(sigmin, cond, worst_mode_vec, rank_deficient)."""
    F = np.sum(list(fishers), axis=0)
    # (self-audit, triggered by the mechanized degeneracy-fail-open lint P2, re-swept identifiability_oed.py
    # -- the module's own OED-selection argmax sites (greedy_oed, optimal_amount) are only as safe as this
    # shared primitive): np.linalg.eigh's behavior on a NaN-containing symmetric matrix is UNRELIABLE, not a clean
    # raise -- reproduced BOTH ways on real (K,K) Fishers: some NaN placements raise LinAlgError (safe, loud), but
    # others silently return finite-looking eigenvalues that are NOT even sorted ascending (eigh's documented
    # contract), e.g. eigh([[1,0,0],[0,nan,0],[0,0,1]]) -> w=[1, nan, 1] not [nan,1,1] -- so `sigmin = w[0]` can
    # silently pick a WRONG (but plausible, finite) eigenvalue instead of surfacing the degeneracy. A degenerate
    # Fisher (a failed/unmeasurable observable contributing NaN) must fail closed at the SOURCE, not depend on
    # which LAPACK code path an unspecified matrix happens to hit.
    if not np.all(np.isfinite(F)):
        raise ValueError("identifiability: summed Fisher carries non-finite (NaN/inf) entries -- "
                          "cannot safely eigendecompose (a degenerate/unmeasurable observable contributed NaN)")
    F = 0.5 * (F + F.T)
    w, V = np.linalg.eigh(F)
    w = np.clip(w, 0, None)
    sigmin = float(w[0]); sigmax = float(w[-1])
    return dict(sigmin=sigmin, cond=float(sigmax / (sigmin + 1e-300)),
                worst_mode_vec=V[:, 0].tolist(),
                rank_deficient=bool(sigmin < 1e-9 * (sigmax + 1e-300)))

def greedy_oed(pool, k, base=None):
    """Greedy E-optimal selection: pick k observables from `pool` (list of (K,K) Fishers) that maximize sigma_min of
    the accumulated Fisher (optionally starting from `base` Fishers already committed). Returns (indices, sigmin_trace).
    This is the anchor-OED (main-mission node 1) generalized to any observable pool -- pick the measurement that most
    lifts the WORST-identified mode, iterate. Diversity-not-count: correlated observables add little sigma_min."""
    pool = [np.asarray(F, float) for F in pool]
    K = pool[0].shape[0]
    acc = np.zeros((K, K)) if base is None else np.sum(list(base), axis=0).astype(float)
    chosen, trace, remaining = [], [], list(range(len(pool)))
    for _ in range(min(k, len(pool))):
        best_i, best_s = None, -np.inf
        for i in remaining:
            #: a candidate observable whose Fisher carries a genuine degeneracy (NaN/inf) cannot be safely
            # scored -- exclude it from selection (never picked) rather than letting it crash the whole greedy
            # search or (worse) silently win via eigh's unreliable NaN ordering (see identifiability docstring).
            try:
                s = identifiability([acc + pool[i]])["sigmin"]
            except ValueError:
                continue
            if s > best_s:
                best_s, best_i = s, i
        if best_i is None:
            break   # every remaining candidate was degenerate -- stop, don't fabricate a pick
        chosen.append(best_i); trace.append(float(best_s))
        acc = acc + pool[best_i]; remaining.remove(best_i)
    return chosen, trace

def compose_typed(stages):
    """TWO-ALGEBRA composition (H 05d92f338, validated on real temple): route each cert quantity to its OWN algebra.
    A single-algebra composition engine LEAKS -- (min,max) under-counts accumulating magnitude (a single view is
    rank-deficient sigma_min=0, but the summed Fisher is full-rank); (+) over-counts readiness (a redundant OK stage
    doesn't make a system 'more ready').
    stages: list of dicts, each {'kind': 'readiness'|'magnitude', 'value': x}. For 'magnitude' the value is a (K,K)
    Fisher (composed by Fisher-ADD -> sigma_min) OR a scalar info (composed by SUM). For 'readiness' the value is a
    scalar status/margin (composed TROPICALLY = min: the worst stage gates).
    Returns dict(readiness, identifiability, binding_stage) -- readiness is the tropical min over status stages;
    identifiability is sigma_min of the summed Fisher (or the summed scalar info) over magnitude stages."""
    import numpy as _np
    status = [(i, s['value']) for i, s in enumerate(stages) if s.get('kind') == 'readiness']
    mags = [s['value'] for s in stages if s.get('kind') == 'magnitude']
    readiness = None; binding = None
    if status:
        # (routed, P5-flagged: builtin min(key=) over NaN -- D's mechanized lint hit, "not forced-traced,
        # your call"; independently reproduced before acting, per session discipline): plain min(status,
        # key=lambda iv: iv[1]) silently OMITS a non-finite readiness stage from the tropical gate whenever it
        # is not literally the first stage in the list (a `new < running_best` comparison against NaN is always
        # False, so NaN never displaces the running candidate once one is set) -- reproduced live: stages =
        # [0.8, NaN, 0.6] reports readiness=0.6, binding_stage=2, COMPLETELY dropping the unmeasurable stage 1
        # from consideration, as if it didn't exist. An unmeasurable/degenerate stage should be AT LEAST as
        # gating as the worst measured stage (pessimistic-on-unknown, same convention as coverage_collapse_
        # alarm's NaN-forces-fire fix and thermal_runaway_severity_cert), not silently absent from the
        # tropical min. Refines the hypothesis (which framed it as "reports better than true") with the exact
        # mechanism: position-dependent silent omission, not a falsely-optimistic real number.
        nonfinite = [(i, v) for i, v in status if not _np.isfinite(v)]
        if nonfinite:
            binding, readiness = nonfinite[0][0], float("nan")  # first unmeasurable stage gates; readiness=nan
        else:
            binding, readiness = min(status, key=lambda iv: iv[1])  # tropical: worst stage gates
            readiness = float(readiness)
    ident = None
    if mags:
        if _np.ndim(mags[0]) == 2:                              # Fisher matrices -> ADD then sigma_min
            ident = identifiability(mags)["sigmin"]
        else:                                                   # scalar infos -> SUM (additive accumulation)
            ident = float(_np.sum(mags))
    return dict(readiness=readiness, identifiability=ident, binding_stage=binding)

def optimal_amount(fisher_of_amount, grid):
    """Find the diversity AMOUNT that PEAKS sigma_min (SELECT don't MAXIMIZE). fisher_of_amount(a) -> a list of (K,K)
    Fishers (e.g. {in-focus, +a defocus, -a defocus}); grid = candidate amounts. Returns dict with the interior
    optimum a*, its sigmin, the sweep, and whether it is a genuine interior peak (too-much-diversity hurts = SNR loss).
    Generalizes the JWST optimal-defocus d* finding (7db3eb913)."""
    grid = list(grid)
    #: a grid point whose Fisher is degenerate (NaN/inf, e.g. a genuinely unmeasurable diversity amount) must
    # never win the argmax below -- exclude it (-inf, never selected as a*) rather than crash the whole sweep or
    # let it silently win via eigh's unreliable NaN behavior (see identifiability docstring).
    sweep = []
    for a in grid:
        try:
            sweep.append(identifiability(fisher_of_amount(a))["sigmin"])
        except ValueError:
            sweep.append(-np.inf)
    if all(s == -np.inf for s in sweep):
        raise ValueError("optimal_amount: every grid point is degenerate -- no amount yields a valid Fisher")
    i = int(np.argmax(sweep))
    interior = 0 < i < len(grid) - 1
    return dict(a_star=float(grid[i]), sigmin_peak=float(sweep[i]),
                sweep=list(zip([float(g) for g in grid], [float(s) for s in sweep])),
                interior_optimum=bool(interior),
                gain_vs_min_amount=float(sweep[i] / (sweep[0] + 1e-300)) if np.isfinite(sweep[0]) else float("nan"),
                gain_vs_max_amount=float(sweep[i] / (sweep[-1] + 1e-300)) if np.isfinite(sweep[-1]) else float("nan"))

# ------------------------------------------------------------------ selftest
def _selftest():
    rng = np.random.default_rng(0); ok = 0; tot = 0
    def check(name, cond):
        nonlocal ok, tot; tot += 1; ok += bool(cond)
        print(("  PASS " if cond else "  FAIL ") + name)

    # (1) single rank-1 observable is rank-deficient for a 2-param model; a second independent one lifts it
    J1 = np.array([[1.0, 0.0]]); F1 = J1.T @ J1               # rank-1 (only sees param 0)
    check("single rank-1 observable is rank-deficient", identifiability([F1])["rank_deficient"])
    J2 = np.array([[0.0, 1.0]]); F2 = J2.T @ J2               # sees param 1
    check("adding an orthogonal observable lifts sigma_min>0", identifiability([F1, F2])["sigmin"] > 0.5)

    # (2) greedy OED prefers the DIVERSE (spanning) pair over two redundant copies of the same observable
    pool = [F1, F1.copy(), F2]                                # two copies of F1 + one F2
    idx, trace = greedy_oed(pool, 2)
    check("greedy OED selects a spanning (diverse) pair", 2 in idx and trace[-1] > 0.5)
    check("greedy OED sigma_min is monotone non-decreasing", all(trace[i+1] >= trace[i]-1e-12 for i in range(len(trace)-1)))

    # (3) optimal_amount finds an INTERIOR peak when too-much diversity loses "SNR" (modeled: info ~ a^2 * exp(-a))
    def fisher_of_a(a):
        # a toy where the 2nd param's info rises then falls with the diversity amount a (breaks ambiguity then loses SNR)
        g = a * np.exp(-a / 1.0)                              # inverted-U in a
        return [np.array([[1.0, 0.0], [0.0, g**2]])]
    r = optimal_amount(fisher_of_a, np.linspace(0.05, 6.0, 40))
    check("optimal_amount finds an interior optimum (select-not-maximize)", r["interior_optimum"])
    check("optimal_amount a* near the analytic peak a=1", abs(r["a_star"] - 1.0) < 0.4)
    check("optimal_amount: too-much diversity is worse (gain_vs_max>1)", r["gain_vs_max_amount"] > 1.2)

    # (4) compose_typed routes by algebra: readiness = tropical min (worst gates); magnitude = additive (Fisher-add)
    stages = [
        {'kind': 'readiness', 'value': 0.9},
        {'kind': 'readiness', 'value': 0.3},                   # the worst -> gates
        {'kind': 'magnitude', 'value': F1},                    # rank-1 (sigma_min=0 alone)
        {'kind': 'magnitude', 'value': F2},                    # rank-1 -> summed with F1 = rank-2 sigma_min>0
    ]
    r = compose_typed(stages)
    check("compose_typed readiness = tropical min (worst stage gates)", abs(r["readiness"] - 0.3) < 1e-9 and r["binding_stage"] == 1)
    check("compose_typed magnitude = additive (rank-deficient parts sum to identifiable)", r["identifiability"] > 0.5)
    r2 = compose_typed([{'kind': 'magnitude', 'value': 2.0}, {'kind': 'magnitude', 'value': 3.0}])
    check("compose_typed scalar infos ADD (2+3=5)", abs(r2["identifiability"] - 5.0) < 1e-9)
    #: a non-finite readiness stage must gate (pessimistic-on-unknown), never be silently omitted from the
    # tropical min just because it isn't the first stage in the list (plain min(key=) skips a non-leading NaN).
    r3 = compose_typed([{'kind': 'readiness', 'value': 0.8}, {'kind': 'readiness', 'value': float('nan')},
                        {'kind': 'readiness', 'value': 0.6}])
    check("compose_typed: non-leading NaN readiness stage STILL gates (binding_stage=1, not silently dropped)",
          r3["binding_stage"] == 1 and np.isnan(r3["readiness"]))
    r4 = compose_typed([{'kind': 'readiness', 'value': float('nan')}, {'kind': 'readiness', 'value': 0.8}])
    check("compose_typed: leading NaN readiness stage also gates (binding_stage=0)",
          r4["binding_stage"] == 0 and np.isnan(r4["readiness"]))

    # (5) (self-audit, triggered by the mechanized degeneracy-fail-open lint): np.linalg.eigh's NaN behavior
    # is UNRELIABLE (proven: some NaN placements raise, others silently return unsorted/wrong-order eigenvalues) --
    # identifiability must fail closed EXPLICITLY, and its two selection-loop callers must exclude (not crash on
    # or silently prefer) a degenerate candidate/grid-point.
    F_nan = F1.copy(); F_nan[0, 0] = np.nan
    try:
        identifiability([F_nan]); raised = False
    except ValueError:
        raised = True
    check("identifiability() fails closed (raises) on a non-finite Fisher", raised)

    pool_deg = [F1, F2.copy(), F2]
    pool_deg[1] = pool_deg[1].copy(); pool_deg[1][0, 0] = np.nan   # candidate 1 is degenerate
    idx_deg, trace_deg = greedy_oed(pool_deg, 2)
    check("greedy_oed excludes a degenerate candidate (never selected)", 1 not in idx_deg and len(idx_deg) == 2)

    def fisher_of_a_deg(a):
        if abs(a - 3.0) < 1e-9:
            return [np.array([[1.0, 0.0], [0.0, np.nan]])]        # one degenerate grid point
        g = a * np.exp(-a / 1.0)
        return [np.array([[1.0, 0.0], [0.0, g**2]])]
    grid_deg = [0.5, 1.0, 3.0, 5.0]                                # 3.0 is degenerate, would otherwise be near-peak
    r_deg = optimal_amount(fisher_of_a_deg, grid_deg)
    check("optimal_amount never selects a degenerate grid point as a*", r_deg["a_star"] != 3.0)

    print(f"identifiability_oed selftest: {ok}/{tot}")
    return ok == tot

if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
