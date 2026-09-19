"""identifiability_oed -- one callable tool compressing H's observable-selection + OED cert findings (session ).

The recurring FORM across substrates (compress-not-accrete of: anchor-OED main-mission node 1; optics n,k single-
reflectance rank-1 d1bbd0214; JWST wavefront phase-diversity 8cc84a75a + optimal-defocus OED 7db3eb913; battery
sim-conditioning identifiability 3b766fcb2):

  A parameter/mode vector theta is observed through a CHOICE of observables. Each observable o contributes a Fisher
  F_o(theta). The IDENTIFIABILITY of theta from a chosen SET S is sigma_min(sum_{o in S} F_o) -- the worst-identified
  mode. A single/coarse observable is generically RANK-DEFICIENT (sigma_min ~ 0); observation DIVERSITY lifts it.
  This tool answers, for a candidate pool of observables:
    * identifiability(S) = sigma_min of the summed Fisher  (report-the-bound: also the worst-mode direction)
    * greedy_oed(pool, k) = the k observables to add, RANK FIRST and then value per cost (see below)
    * optimal_amount(param_fn, grid) = the diversity AMOUNT that peaks sigma_min (SELECT don't MAXIMIZE: an interior
      optimum when too-much diversity loses SNR) -- the JWST defocus d* finding, generalized.

THE PLATEAU, AND WHY THE E-OPTIMAL ARGMAX ALONE IS NOT A SELECTION RULE (measured on contact-solver
friction identification, a pool of 32 micro-slip probes -- 8 contacts x 4 directions -- for 8 friction
parameters). While the accumulated Fisher is rank-deficient, sigma_min is 0 for EVERY candidate: it is
exactly 0 for any parameter no selected observable has excited, so the argmax has no gradient over the
whole regime k < K and degenerates to input order. Measured there: the sigma_min-only rule bought four
redundant copies of contact 0, then four of contact 1, then four of contact 2 -- rank(F) 3/8 after
twelve probes, sigma_min = 0 at every step, 1.2515 N s spent. The fix in `greedy_oed` is lexicographic:

    (d rank / cost, d value_bits / cost)

RANK FIRST -- on the plateau the only term that moves is the rank, and dividing by cost picks one probe
per uncovered parameter in ASCENDING COST ORDER -- and the tie is broken by the engine's own PRICED
objective, the Gaussian information gain in bits, value_bits = 1/2 log2 det(I + (F_acc + prior)^-1 F_i)
(precision_form's value(h) = 1/2 log2(1 + h^T C h / sigma^2) for a matrix observable). Same pool, same
costs: rank 8/8 and sigma_min 0 -> 6.33e5 at k = 8 for 0.64906 N s, the trace continuing 1.27e6, 1.90e6,
2.28e6 at k = 9, 10, 11.

WHY THE TIE-BREAK IS THE PRICED OBJECTIVE AND NOT sigma_min/cost. A second measurement (the same probe
pool read as a sequential-selection problem against the adaptive optimum by full decision-tree
enumeration) priced the two candidate rules: a sigma_min-gain-per-cost objective reaches only 0.29-0.61
of the adaptive optimum of the engine's own entropy potential and falls BELOW 1 - 1/e = 0.6321 in 30 of
50 random Dirichlet priors (worst 0.2031), while the same cost-aware greedy run on the engine's own
priced objective stays at 0.77-0.92 and never falls below 1 - 1/e in those 50 priors. The bound is about
greedy on the objective being maximized: price the engine's objective by cost, do not swap the objective.
`tie_break="sigma_min"` reproduces the ungradiented E-optimal argmax (the shipped defect) for comparison.

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

def _spectrum(F, tol=1e-10):
    """(sigma_min, rank, smallest POSITIVE eigenvalue) of one symmetric matrix from a single eigendecomposition.
    The rank tolerance is RELATIVE to the largest eigenvalue, so the three readings are unit-invariant (a Fisher
    in different units is the same Fisher). Fails closed on a non-finite matrix, exactly as `identifiability`."""
    F = np.asarray(F, float)
    if not np.all(np.isfinite(F)):
        raise ValueError("spectrum: matrix carries non-finite (NaN/inf) entries -- cannot safely eigendecompose "
                         "(a degenerate/unmeasurable observable contributed NaN)")
    w = np.linalg.eigvalsh(0.5 * (F + F.T))
    top = max(float(w[-1]), 1e-300)
    pos = w[w > tol * top]
    return float(max(float(w[0]), 0.0)), int(pos.size), (float(pos[0]) if pos.size else 0.0)


def value_bits(acc, F, prior):
    """The engine's PRICED objective for a matrix observable: the Gaussian information gain in bits of adding the
    Fisher F to the accumulated Fisher `acc`, under a proper prior precision `prior` (a positive scalar ridge or a
    (K,K) matrix):  1/2 log2 det(I + (acc + prior)^-1 F).  This is precision_form's value(h) = 1/2 log2(1 +
    h^T C h / sigma^2) with C = (acc + prior)^-1 and a matrix-valued observable; log-det gain is submodular, which
    is why greedy on it keeps the 1 - 1/e property that a sigma_min-per-cost rule does not."""
    acc = np.asarray(acc, float); F = np.asarray(F, float)
    P = np.eye(acc.shape[0]) * float(prior) if np.ndim(prior) == 0 else np.asarray(prior, float)
    if not (np.all(np.isfinite(acc)) and np.all(np.isfinite(F)) and np.all(np.isfinite(P))):
        raise ValueError("value_bits: non-finite Fisher or prior -- a degenerate observable cannot be priced")
    A = 0.5 * ((acc + P) + (acc + P).T)
    sign, logdet_a = np.linalg.slogdet(A)
    sign_b, logdet_b = np.linalg.slogdet(0.5 * ((A + F) + (A + F).T))
    if sign <= 0 or sign_b <= 0:
        raise ValueError("value_bits: prior + accumulated Fisher is not positive definite -- raise `prior`")
    return float((logdet_b - logdet_a) / (2.0 * np.log(2.0)))


def select_with_rank_tiebreak(value, rank_gain, cost, *, exclude=frozenset(), start=0,
                              rel_tol=1e-9):
    """ONE greedy pick shared by the engine's selection loops.

    Rank increase first, then value, cost last; on a full plateau (equal rank
    gain, equal value) the pick rotates from `start`, i.e. round-robin, instead
    of sticking to one candidate. Three independent contact-solver measurements
    forced the same correction. (i) The 8-interface stack of `greedy_oed`'s own
    fixture: from a rank-deficient accumulation a pure value argmax buys four
    copies of one probe, rank-first reaches rank 8/8 and sigma_min 0 -> 6.3e5.
    (ii) A four-contact probing study: after every contact has been covered the
    belief saturates at 1.0 and every ΔV is 2e-6, so value is blind, and a rule
    that cycles beats the one that sticks by 1.73x on sigma_min at 1.5 % less
    actuation. (iii) The set-value coverage rule on the same stack: same shape,
    same correction, three times in a row -- hence one function.
    `value`, `rank_gain`, `cost`: one finite number per candidate; `cost` must
    be > 0. `exclude`: already-taken indices. Returns the winning index, or
    None when every candidate is excluded. Raises ValueError on a non-finite
    entry or a non-positive cost (fail closed: a degenerate candidate must not
    win by default). Ties in value/cost are within `rel_tol` (relative);
    rank_gain compares exactly (callers pass integer gains)."""
    v = np.asarray(value, float).ravel()
    g = np.asarray(rank_gain, float).ravel()
    c = np.asarray(cost, float).ravel()
    n = len(v)
    if not (len(g) == n and len(c) == n):
        raise ValueError("value, rank_gain and cost must have the same length")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(g)) and np.all(np.isfinite(c))):
        raise ValueError("select_with_rank_tiebreak: non-finite entry -- "
                         "a degenerate candidate cannot be scored")
    if not np.all(c > 0):
        raise ValueError("select_with_rank_tiebreak: costs must be positive")
    skip = set(int(i) for i in exclude)
    order = sorted((i for i in range(n) if i not in skip), key=lambda i: (i - start) % n)
    best = None
    for i in order:
        if best is None:
            best = i
            continue
        j = best
        if g[i] != g[j]:
            if g[i] > g[j]:
                best = i
            continue
        vi, vj = v[i], v[j]
        if not (abs(vi - vj) <= rel_tol * max(1.0, abs(vi), abs(vj))):
            if vi > vj:
                best = i
            continue
        ci, cj = c[i], c[j]
        if not (abs(ci - cj) <= rel_tol * max(1.0, abs(ci), abs(cj))):
            if ci < cj:
                best = i
            continue
        # full tie: keep the earlier one in rotation order (round-robin)
    return best


def greedy_oed(pool, k, base=None, costs=None, tie_break="rank", prior=1e-9):
    """Greedy selection: pick k observables from `pool` (list of (K,K) Fishers) to add to `base`, and return
    (indices, sigmin_trace) -- sigma_min of the accumulated Fisher after each pick.

    tie_break="rank" (default, the measured fix): maximize the lexicographic key

        (d rank / cost, d value_bits / cost)

    routed through select_with_rank_tiebreak -- the same rank-first rule two
    further measurements forced independently (see that docstring). The priced
    keys are passed as the helper's rank_gain/value legs with rel_tol=0, so the
    comparison is the exact tuple compare above; the helper's third leg adds one
    thing the tuple did not have: on an EXACT tie in both priced keys the cheaper
    candidate wins instead of the one that happened to come first.

    RANK FIRST, then the engine's own priced objective. On a rank-deficient accumulation sigma_min is exactly 0
    for every candidate (any parameter no selected observable excites contributes a zero column), so an E-optimal
    argmax has no gradient there and reduces to input order -- it buys redundant copies of one observable and
    never leaves sigma_min = 0. See the module docstring for the measured numbers, and for why the tie-break is
    the priced objective rather than sigma_min per cost.

    tie_break="sigma_min": the ungradiented E-optimal argmax as originally shipped (kept so the defect stays
    reproducible). `costs` is one positive number per pool entry (default 1.0 = count the measurements).
    `prior` is the prior precision for value_bits, relative to the mean eigenvalue scale of the whole pool, so
    the ranking is invariant to the units of the Fishers; pass a (K,K) matrix for a real prior.
    Diversity-not-count: correlated observables add little rank and few bits."""
    pool = [np.asarray(F, float) for F in pool]
    K = pool[0].shape[0]
    if tie_break not in ("rank", "sigma_min"):
        raise ValueError(f"tie_break must be 'rank' or 'sigma_min': {tie_break!r}")
    if costs is None:
        costs = np.ones(len(pool))
    else:
        costs = np.asarray(costs, float).ravel()
        if costs.shape != (len(pool),) or not np.all(np.isfinite(costs)) or not np.all(costs > 0):
            raise ValueError("costs must be one positive finite number per pool entry")
    acc = np.zeros((K, K)) if base is None else np.sum(list(base), axis=0).astype(float)
    if np.ndim(prior) == 0:                                   # scale-free ridge: relative to the pool's own scale
        finite = [float(np.trace(F)) for F in pool if np.all(np.isfinite(F))]   # a degenerate candidate must
        scale = (float(np.mean(finite)) if finite else 1.0) / max(K, 1)          # not poison the prior's scale
        P = float(prior) * max(scale, 1e-300) * np.eye(K)
    else:
        P = np.asarray(prior, float)
    chosen, trace, remaining = [], [], list(range(len(pool)))
    s_acc, r_acc, _ = _spectrum(acc)
    for _ in range(min(k, len(pool))):
        if tie_break == "sigma_min":
            best_i, best_key = None, None
            for i in remaining:
                #: a candidate observable whose Fisher carries a genuine degeneracy (NaN/inf) cannot be safely
                # scored -- exclude it from selection (never picked) rather than letting it crash the whole greedy
                # search or (worse) silently win via eigh's unreliable NaN ordering (see identifiability docstring).
                try:
                    s, r, _sp = _spectrum(acc + pool[i])
                    key = (s,)
                except ValueError:
                    continue
                if best_key is None or key > best_key:
                    best_key, best_i = key, i
        else:
            gain, val, ok = [], [], []
            for i in remaining:
                #: same exclusion as the sigma_min branch, and one step further: a candidate whose
                # PRICED keys come out non-finite is dropped here rather than handed to
                # select_with_rank_tiebreak, which fails closed on a non-finite entry -- one
                # degenerate candidate must not raise the whole greedy search.
                try:
                    s, r, _sp = _spectrum(acc + pool[i])
                    g_i = (r - r_acc) / costs[i]
                    v_i = value_bits(acc, pool[i], P) / costs[i]
                except ValueError:
                    continue
                if not (np.isfinite(g_i) and np.isfinite(v_i)):
                    continue
                gain.append(g_i); val.append(v_i); ok.append(i)
            if not ok:
                best_i = None
            else:
                # the lexicographic key above, with cost as the last leg: exact compare on the
                # priced legs (rel_tol=0), so only an exact tie in rank gain AND value reaches
                # the cost leg, where the cheaper candidate wins.
                sub = select_with_rank_tiebreak(val, gain, [costs[i] for i in ok],
                                                rel_tol=0.0)
                best_i = ok[sub] if sub is not None else None
        if best_i is None:
            break   # every remaining candidate was degenerate -- stop, don't fabricate a pick
        acc = acc + pool[best_i]
        s_acc, r_acc, _ = _spectrum(acc)
        chosen.append(best_i); trace.append(float(s_acc)); remaining.remove(best_i)
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
