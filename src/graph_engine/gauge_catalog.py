"""
gauge_catalog.py — the gauge catalog for reconstruction-to-simulation models. Collects cross-worktree, cross-substrate gauge findings —
over-determined SUBSTRATE-INDEPENDENT (J⊕C kinematics↔vision, J⊕L SR-photometric, an earlier batch/reconstruction-cert-keystone/agent pool-gauge-ceiling) — into ONE deployable reference: every
gauge a vision/motion→sim twin can carry, its σ_min-NULL dimension, and the MATCHING external data channel that lifts it. Two laws hold for EVERY entry (the agent pool rediscovered them
on independent substrates): (1) a gauge is a σ_min-null of the OBSERVATION Fisher — the estimator cannot see through it no matter how much of the SAME observation it gets; (2) it is
DATA-liftable, NEVER dither/diversity-liftable — only a DECORRELATED external channel lifts it (agent pool-gauge-ceiling), and each gauge is lifted ONLY by its MATCHING channel (cross-
specific, an earlier batch 3×3). ⟹ a recon→sim twin is certifiable UP TO its present gauges; report each gauge + its minimal anchor, ABSTAIN on the QoIs that live in an unlifted gauge.

Consumes C's gauge_detector (detect the σ_min-null subspace of a Fisher) + J's fleet_oed/an earlier batch (Δσ_min acquisition = the matching channel). This catalog is the LOOKUP that makes
`next_channel_to_lift` a provable minimal acquisition (each gauge-kind → exactly one lifting channel).
L in-span-gauge).
"""
from __future__ import annotations
from .linalg_guards import guarded_eigh, guarded_eigvalsh, guarded_svd  # ★fix: used (gdim/svd/eigh) but never imported = live NameError (pyflakes)

__all__ = ["CATALOG", "lifting_channel", "min_anchors", "gauge_report", "decorrelation_aware_min_anchors", "acquisition_plan",
           "null_subspace", "restricted_span_redundancy_certified"]

# each gauge family: null-DIM (or 'k' if data-dependent), the MATCHING lifting channel, and the source over-det. ALL are σ_min-nulls, ALL data-liftable-not-dither.
CATALOG = {
    "similarity_translation": {"null_dim": 3, "lifts_with": "known absolute POSITION (a datum / 1 world point)",   "substrate": "recon from relative meas", "src": "J an earlier batch · over-det C gauge_detector (an earlier batch)"},
    "similarity_rotation":    {"null_dim": 3, "lifts_with": "known absolute DIRECTION (gravity / 1 world edge)",     "substrate": "recon from relative meas", "src": "J an earlier batch · C gauge_detector (an earlier batch)"},
    "similarity_scale":       {"null_dim": 1, "lifts_with": "known absolute DISTANCE (1 known length / stereo baseline)", "substrate": "recon / monocular vision", "src": "J an earlier batch · C body-1000/1025 (an earlier batch over-det)"},
    "geometry_depth_scale":   {"null_dim": 1, "lifts_with": "known length / stereo baseline",                        "substrate": "monocular geometry", "src": "C body-1000/1025 · J vision-twin-scale-cert"},
    "photometric_matlight":   {"null_dim": 1, "lifts_with": "a white / albedo REFERENCE",                            "substrate": "photometric material×light", "src": "C body-1027 · H"},
    "photometric_metamer":    {"null_dim": "k", "lifts_with": "spectrally-DIVERSE illuminants (diversity not count)", "substrate": "spectral / color", "src": "H metamer-null · J metamer-diverse-illuminants"},
    "motion_depth_scale":     {"null_dim": 1, "lifts_with": "a 2nd view / metric anchor",                            "substrate": "monocular motion", "src": "C body-1031 · J structure_from_motion"},
    "dynamics_mass_scale":    {"null_dim": 1, "lifts_with": "1 known FORCE (+ 1 known length for full metric)",       "substrate": "dynamics from vision", "src": "J dynamics-from-vision-mass-scale-gauge"},
    "fleet_shared_prior":     {"null_dim": "k", "lifts_with": "an EXTERNAL DATASET (not a lane-dither)",              "substrate": "fleet evidence Gramian", "src": "J/D fleet-gauge-ceiling"},
    "sr_inspan_photometric":  {"null_dim": "k", "lifts_with": "an external anchor (NOT model diversity)",             "substrate": "super-res hallucination", "src": "L in-span-gauge · over-det J (null_fence)"},
    "arrhenius_compensation": {"null_dim": 1, "lifts_with": "a WIDER absolute-temperature RANGE (extends the 1/T lever arm) or a known Ea/A anchor", "substrate": "thermal-kinetics twin (battery ARC self-heating)", "src": "H bus 15:04 · J⊕H an earlier batch over-det (C gauge_detector, |cos|>0.99 isokinetic line)"},
}


def lifting_channel(gauge_kind):
    """the MATCHING external channel that lifts a gauge (cross-specific: a wrong-kind channel does NOT lift it, an earlier batch 3×3). Returns None if unknown."""
    e = CATALOG.get(gauge_kind); return e["lifts_with"] if e else None


def min_anchors(gauges_present):
    """the minimal EXTERNAL anchors to certify a twin carrying `gauges_present` (a set of gauge kinds) — one matching channel per gauge (deduped). This is the provably-minimal
    acquisition because each gauge is lifted ONLY by its matching channel (cross-specificity). Returns {gauge: lifting_channel} for the present gauges + the deduped anchor set."""
    per = {g: lifting_channel(g) for g in gauges_present if g in CATALOG}
    return dict(per_gauge=per, anchor_set=sorted(set(per.values())), n_anchors=len(set(per.values())),
                unknown=[g for g in gauges_present if g not in CATALOG])


def decorrelation_aware_min_anchors(F, channel_jacobians, tol_ratio=1e-6, min_redundancy=1, decorrel_tol=1e-3):
    """The DECORRELATION-AWARE minimal anchor COUNT for a specific deployment Fisher F (coach  stress: my `min_anchors`
    counts one channel per gauge KIND — naive when gauges are CORRELATED). Greedy matroid COVERAGE of the gauge (σ_min-null)
    subspace: a single channel that breaks MULTIPLE correlated gauge DOFs (e.g. a full-pose marker pins translation AND
    rotation at once) counts ONCE. `channel_jacobians` = {name: Jacobian (rows = observation gradients w.r.t. the params)}.
    Returns {chosen, n_anchors, initial_gauge_dim, final_gauge_dim, certified}. This is the true minimum; `min_anchors`
    (one-per-gauge-KIND) is its INDEPENDENT-GAUGE UPPER BOUND (tight only when the gauges are decorrelated). Mirrors C's
    gauge_detector.min_channels_to_certify (matroid) + H's decorrelation-AWARE σ_min water-filling — same principle, three lanes.

    ★ADAPTIVE HARDENING (batch, drop-in, DEFAULT-PRESERVING — min_redundancy=1 is a no-op, returns exactly the legacy
    dict): this matroid-COVERAGE objective stops the INSTANT gauge_dim hits 0 — for an N-dim gauge it structurally NEVER
    recommends more than N anchors, no matter how many candidate channels are offered (verified: a 1-D gauge with 1 or
    50 colinear candidates always returns n_anchors=1). That is correct for IDENTIFIABILITY (one honest channel is
    enough to make F full-rank) but wrong for ADVERSARIAL ROBUSTNESS: batch (O6 metric-anchor gauge-lift) measured
    that a 1-anchor plan is DEFEATED by an adaptive fake that fabricates that one channel's reading (AUC 1.0->~0.47,
    collapses to the blind/un-anchored level) — a σ_min-rank argument says nothing about whether the channel's VALUE
    is trustworthy, only whether its DIRECTION would resolve the null if trusted. Passing min_redundancy>=2 requires,
    for each chosen anchor, >=(min_redundancy-1) ADDITIONAL gauge-relevant companions whose Jacobian rows are NOT
    colinear with the already-accepted set for that direction (checked via the SMALLEST singular value of the stacked
    rows, relative to the largest, > decorrel_tol) — i.e. a genuinely independent measurement pathway, not a duplicate
    of the same one (guard-flag-count-is-nominal: a redundancy COUNT is nominal; this measures the effective SPAN).
    FORCED ADVERSARY (batch): 2 Jacobian-COLINEAR anchors (same direction, e.g. two redundant gauges reading the
    identical physical quantity) give NO protection — a single adjustment to that one shared quantity spoofs both
    simultaneously, so they fail the decorrelation check and `insufficient_redundancy` is reported; only a companion
    with a genuinely non-colinear row (a DIFFERENT measurement pathway, decorrelated from the first) satisfies it.
    CAVEAT (honest, not fixed here): Jacobian-row decorrelation is a NECESSARY structural proxy, not a full adversarial
    model — it says two channels measure geometrically different combinations, not that their REPORTING pathways are
    independently un-compromisable (an omniscient adversary with free per-channel fabrication defeats any finite anchor
    set; the module has no notion of channel TRUST/compromise-probability, only of what a channel geometrically
    measures). True spoof-robustness additionally needs the anchors to be ACCESS-independent (disjoint compromise
    surfaces), which is metadata this function does not have. Compose with an integrity/cross-modal check for that.
    ★SCOPE-CORRECTION (batch, owner-QC): this min_redundancy check is VALID FOR A 1-D GAUGE (k=1) but does NOT
    generalize to MULTI-DIMENSIONAL gauges (rotation/translation, null_dim=3). It checks Jacobian-row decorrelation in the
    FULL parameter space, which at k>1 both FALSELY CERTIFIES (a rank-k full-pose channel + a companion whose only
    non-redundant signal lies OUTSIDE the gauge → certified=True but the adversary collapses an uncovered gauge axis to
    AUC~0.46) and FALSELY REJECTS (two genuinely independent rank-k pathways sharing one Jacobian direction → certified=
    False though the plan is robust). For k>1 use `restricted_span_redundancy_certified` (this module) instead: it checks
    leave-one-PATHWAY-out rank+margin RESTRICTED to the null (gauge) subspace, per-direction ≥2. Derived law: the robust
    anchor count depends on channel RICHNESS, not k — 2 anchors suffice IF a rank-k (full-pose) pathway exists
    (k-independent), else up to 2·null_dim if only rank-1 channels exist. Pathway identity is caller-supplied metadata
    (load-bearing at k>1). (Not-forced flag, batch: a similar false-pos/neg asymmetry MAY be latent already at k=1 —
    unverified, out of scope; treat the k=1 min_redundancy result as sound only for the batch/batch tested cases.)
    Adds (only when min_redundancy>1) keys: adaptive_chosen, adaptive_certified, insufficient_redundancy."""
    import numpy as np
    F = np.asarray(F, float)
    # ★null-safety (agent pool fail-open sweep): non-finite/empty F, or non-finite tol_ratio/decorrel_tol -> CONSERVATIVE (certified=False),
    # NEVER an eigvalsh CRASH (LinAlgError on NaN) or a silent-certify (NaN tol_ratio -> gdim=0 -> certified w/ 0 anchors; -inf
    # decorrel_tol -> colinear-anchor anti-spoof defeated). certified=False on garbage input, not a false pass.
    def _bad(reason):
        r = {"chosen": [], "n_anchors": 0, "initial_gauge_dim": None, "final_gauge_dim": None, "certified": False, "reason": reason}
        if min_redundancy > 1:
            r.update(adaptive_chosen=[], adaptive_certified=False, insufficient_redundancy=[{"anchor": None, "reason": reason}])
        return r
    if F.ndim != 2 or F.shape[0] == 0 or F.shape[0] != F.shape[1] or not np.all(np.isfinite(F)):
        return _bad("non-finite or non-square/empty F")
    if not (np.isfinite(tol_ratio) and np.isfinite(decorrel_tol)):
        return _bad("non-finite tol_ratio/decorrel_tol")

    def gdim(M):
        w = guarded_eigvalsh((M + M.T) / 2.0)
        smax = max(float(w.max()), 1e-300)
        return int((w < tol_ratio * smax).sum()), np.sort(w)

    cur = F.copy(); chosen = []; remaining = dict(channel_jacobians)
    k0, _ = gdim(cur); steps = 0
    while gdim(cur)[0] > 0 and remaining and steps <= len(channel_jacobians):
        base_k, _ = gdim(cur); best_name = None; best_F = None; best_key = None
        for name, Jc in remaining.items():
            Jc = np.atleast_2d(np.asarray(Jc, float)); F2 = cur + Jc.T @ Jc
            k2, w2 = gdim(F2)
            key = (base_k - k2, float(w2[:max(base_k, 1)].sum()))    # 1st reduce gauge_dim, 2nd raise the smallest eigs
            if best_key is None or key > best_key:
                best_key = key; best_name = name; best_F = F2
        if best_key is None or best_key[0] <= 0 and best_key[1] <= gdim(cur)[1][:max(base_k, 1)].sum() + 1e-12:
            break                                                    # no channel makes progress → externally-irreducible
        chosen.append(best_name); cur = best_F; del remaining[best_name]; steps += 1
    kf, _ = gdim(cur)
    out = dict(chosen=chosen, n_anchors=len(chosen), initial_gauge_dim=k0, final_gauge_dim=kf, certified=bool(kf == 0))
    if min_redundancy <= 1 or not chosen:
        return out                                                   # LEGACY PATH — byte-identical to pre-batch behavior

    def gauge_relevant(name):
        Jc = np.atleast_2d(np.asarray(channel_jacobians[name], float))
        k2, _ = gdim(F + Jc.T @ Jc)
        return k2 < k0                                                # alone (from the ORIGINAL F) it would shrink the gauge

    adaptive_chosen = list(chosen); insufficient = []
    for c in chosen:
        group = [c]
        pool = [n for n in channel_jacobians if n not in adaptive_chosen and gauge_relevant(n)]
        for n in pool:
            if len(group) >= min_redundancy:
                break
            stacked = np.vstack([np.atleast_2d(np.asarray(channel_jacobians[g], float)) for g in group]
                                 + [np.atleast_2d(np.asarray(channel_jacobians[n], float))])
            s = guarded_svd(stacked, compute_uv=False)
            smax = max(float(s.max()), 1e-300)
            ratio = float(s.min()) / smax                            # ~0 iff the new row is colinear with the existing group's span
            if ratio > decorrel_tol:
                group.append(n); adaptive_chosen.append(n)
        if len(group) < min_redundancy:
            insufficient.append({"anchor": c, "have": len(group), "need": min_redundancy})
    out["adaptive_chosen"] = adaptive_chosen
    out["adaptive_certified"] = bool(kf == 0 and not insufficient)
    out["insufficient_redundancy"] = insufficient
    return out


def null_subspace(F, tol_ratio=1e-6):
    """Orthonormal basis (D x null_dim) for the σ_min-null (gauge) subspace of Fisher F — eigenvectors whose
    eigenvalue < tol_ratio*max(eigenvalue). Used by `restricted_span_redundancy_certified` to RESTRICT redundancy
    checks to the gauge subspace only (batch), instead of the full parameter space (batch's proxy)."""
    import numpy as np
    F = np.asarray(F, float)
    if F.ndim != 2 or F.shape[0] == 0 or F.shape[0] != F.shape[1] or not np.all(np.isfinite(F)) or not np.isfinite(tol_ratio):
        n = F.shape[0] if (F.ndim == 2 and F.shape[0] > 0) else 0
        return np.zeros((n, 0))                                        # ★null-safety: no gauge directions on non-finite/empty F (conservative, no eigh crash)
    w, V = guarded_eigh((F + F.T) / 2.0)
    smax = max(float(w.max()), 1e-300)
    return V[:, w < tol_ratio * smax]


def restricted_span_redundancy_certified(F, channel_jacobians, chosen, pathway_of=None, tol_ratio=1e-6, decorrel_tol=1e-3):
    """batch — MULTI-DIMENSIONAL (k=null_dim>1) extension of batch's min_redundancy hardening, ADDITIVE ONLY
    (does not alter decorrelation_aware_min_anchors; both stay exported and independently callable).

    batch validated `min_redundancy`/`decorrel_tol` on the 1-D similarity_scale gauge (null_dim=1), where checking
    the SMALLEST singular value of the STACKED anchor Jacobians in the FULL parameter space happens to coincide with
    a valid redundancy proxy. The multidimensional gauge experiment measured that for a k>1
    gauge (rotation/translation, null_dim=3) that proxy is INSUFFICIENT in BOTH directions, because it conflates
    "what a channel measures" (Jacobian direction) with "how independently it can be compromised" (reporting
    pathway) — two orthogonal axes batch's own CAVEAT already named but did not yet need to resolve:
      (i) FALSE POSITIVE (dangerous): once one anchor already spans the WHOLE k-D null subspace (the matroid-optimal
          single "full pose"-style anchor the base algorithm naturally picks), a companion whose only non-redundant
          contribution is an IRRELEVANT off-null (informative-axis) component satisfies the FULL-space stacked-SVD
          check while giving ZERO real backup for the null directions it does not touch (measured: shipped
          decorrelation_aware_min_anchors reports adaptive_certified=True; 2 of 3 gauge directions collapse to
          near-chance AUC under a single-pathway compromise of the primary anchor).
      (ii) FALSE NEGATIVE (wasteful, not dangerous): two genuinely INDEPENDENT reporting pathways that happen to
          measure the identical physical quantity (e.g. two motion-capture rigs each giving x,y,z — same Jacobian
          direction) are Jacobian-colinear in the full space and get rejected by the stacked-SVD check even though
          they form a fully robust, compromise-disjoint plan.
    FIX: (a) restrict the check to the NULL SUBSPACE ONLY (project each candidate's Jacobian onto `null_subspace(F)`
    BEFORE stacking — removes the informative-axis loophole causing (i)); (b) treat redundancy per REPORTING PATHWAY,
    not per raw channel-count: for every pathway present in `chosen` (default: each channel name is its own pathway,
    matching batch's implicit assumption — pass `pathway_of` to group differently-named channels that share one
    physical/reporting mechanism), verify LEAVE-ONE-PATHWAY-OUT still leaves the null-subspace-restricted Jacobian of
    the REMAINING pathways at rank==null_dim with smin/smax > decorrel_tol. This is the basis-independent
    generalization of batch's "≥2 decorrelated anchors" to k dimensions: no SINGLE pathway's compromise can ever
    collapse the certified span below null_dim, for ANY of the k directions, without cherry-picking a basis.
    Returns dict(null_dim, n_pathways, fully_spans, per_pathway=[{remove, remaining_rank, margin, survives_removal}],
    certified). `certified` requires fully_spans (catches the batch case-(d) non-spanning gap: a decorrelated-but-
    non-spanning anchor set never reaches rank==null_dim even with nothing removed — the EXISTING base algorithm's
    own kf==0 check already catches this too; this is a confirmatory, not a new, guard) AND every pathway removable
    without dropping below full rank+margin."""
    import numpy as np
    F = np.asarray(F, float)
    if F.ndim != 2 or F.shape[0] == 0 or F.shape[0] != F.shape[1] or not np.all(np.isfinite(F)) or not (np.isfinite(tol_ratio) and np.isfinite(decorrel_tol)):
        return {"null_dim": None, "n_pathways": 0, "fully_spans": False, "per_pathway": [], "certified": False,
                "reason": "non-finite/empty F or non-finite tol_ratio/decorrel_tol"}   # ★null-safety: conservative, no eigh crash / no silent-certify
    N = null_subspace(F, tol_ratio)
    k = N.shape[1]
    if k == 0:                                                         # no gauge (full-rank F) -> nothing to redundantly cover -> trivially certified (no anchors needed)
        return {"null_dim": 0, "n_pathways": 0, "fully_spans": True, "per_pathway": [], "certified": True,
                "reason": "no gauge subspace (F full-rank) -> no redundancy needed"}
    pathway_of = pathway_of or {}
    pathways = {}
    for c in chosen:
        pid = pathway_of.get(c, c)
        Jc = np.atleast_2d(np.asarray(channel_jacobians[c], float))
        pathways.setdefault(pid, []).append(Jc @ N)

    def stacked_rank_margin(pids):
        rows = [r for pid in pids for r in pathways[pid]]
        if not rows:
            return 0, 0.0
        M = np.vstack(rows)
        s = guarded_svd(M, compute_uv=False)
        smax = max(float(s.max()), 1e-300)
        rank = int((s > tol_ratio * smax).sum())
        margin = float(s.min()) / smax if s.size >= k else 0.0
        return rank, margin

    all_pids = list(pathways.keys())
    per_pathway = []
    for pid in all_pids:
        remaining = [p for p in all_pids if p != pid]
        rank, margin = stacked_rank_margin(remaining)
        per_pathway.append({"remove": pid, "remaining_rank": rank, "margin": margin,
                             "survives_removal": bool(rank == k and margin > decorrel_tol)})
    full_rank, _ = stacked_rank_margin(all_pids)
    fully_spans = bool(full_rank == k)
    certified = bool(fully_spans and k > 0 and all(p["survives_removal"] for p in per_pathway))
    return dict(null_dim=k, n_pathways=len(all_pids), fully_spans=fully_spans, per_pathway=per_pathway, certified=certified)


def acquisition_plan(kind_cert, F=None, channel_jacobians=None):
    """DEPLOYABLE acquisition plan — the single actionable answer to 'what do I acquire to CERTIFY this twin?'. Composes the
    KIND-gate (WHICH action per unresolved null) with matroid anchor coverage (the MINIMAL channel set for the RANK nulls):
      - RANK nulls (structural: a group-invariance / rank-deficient gauge) → ACQUIRE a NEW decorrelated channel; data/lever-arm
        will NOT lift. If F + channel_jacobians are given, `decorrelation_aware_min_anchors` returns the MINIMAL such set.
      - CONDITIONING nulls (soft: ill-conditioned/extrapolation) → POUR ~n_req more samples (or widen the lever-arm) — SAME channel.
    `kind_cert` = H's `gauge_kind_cert.null_kind_cert(...)` output (or any {'directions':[{'kind','n_req_to_resolve'}],...}); this
    stays self-contained (composes the classifier's OUTPUT, no cross-repo import). Unifies gauge_catalog (which channel) ×
    null_kind_cert (which kind) × water-filling (allocation) into ONE plan. Returns dict(ready, plan, summary, counts)."""
    dirs = kind_cert["directions"]
    rank = [d for d in dirs if d.get("kind") == "RANK_NULL"]
    cond = [d for d in dirs if d.get("kind") == "CONDITIONING_NULL"]
    plan = []
    if cond:
        nreqs = [int(d["n_req_to_resolve"]) for d in cond if d.get("n_req_to_resolve")]
        n_pour = max(nreqs) if nreqs else None
        plan.append({"null_kind": "CONDITIONING", "n_directions": len(cond), "n_req": n_pour,
                     "action": f"POUR ~{n_pour} samples (or widen the lever-arm) — SAME channel" if n_pour else "POUR more samples / widen lever-arm"})
    if rank:
        if F is not None and channel_jacobians:
            m = decorrelation_aware_min_anchors(F, channel_jacobians)
            plan.append({"null_kind": "RANK", "n_directions": len(rank), "channels": m["chosen"], "certified": m["certified"],
                         "action": f"ACQUIRE {m['n_anchors']} NEW decorrelated channel(s): {m['chosen']}"})
        else:
            plan.append({"null_kind": "RANK", "n_directions": len(rank), "channels": None,
                         "action": f"ACQUIRE a NEW decorrelated channel per rank null ({len(rank)}); data/lever-arm will NOT lift"})
    ready = not rank and not cond
    return {"ready": ready, "plan": plan, "n_rank_null": len(rank), "n_conditioning_null": len(cond),
            "summary": "twin certifiable — no acquisition needed" if ready
                       else f"{len(rank)} RANK→acquire-new · {len(cond)} CONDITIONING→pour/widen"}


def gauge_report(gauges_present):
    """deployable readout for a twin: its gauges, the minimal anchors, and the QoIs to ABSTAIN on until anchored. Every lane reads this."""
    m = min_anchors(gauges_present)
    return dict(n_gauges=len([g for g in gauges_present if g in CATALOG]),
                minimal_anchors=m["anchor_set"], n_anchors=m["n_anchors"], per_gauge=m["per_gauge"],
                unknown_gauges=m["unknown"],
                rule="certifiable UP TO these gauges; ABSTAIN on QoIs in an unlifted gauge; each gauge lifted ONLY by its matching channel (cross-specific), NEVER by dither/diversity")


def _selftest():
    ok = True
    # (1) every entry has a matching lifting channel + is data-liftable (the two universal laws)
    c1 = all("lifts_with" in e and "null_dim" in e for e in CATALOG.values())
    ok &= c1; print(f"  catalog: {len(CATALOG)} gauge families, all have (null_dim, matching lifting channel) {'OK' if c1 else 'FAIL'}")
    # (2) a monocular recon carrying {similarity} needs 3 distinct anchors (position, direction, distance) — cross-specific, one per kind
    r = gauge_report({"similarity_translation", "similarity_rotation", "similarity_scale"})
    c2 = r["n_gauges"] == 3 and r["n_anchors"] == 3
    ok &= c2; print(f"  similarity twin: {r['n_gauges']} gauges → {r['n_anchors']} distinct anchors {r['minimal_anchors']} {'OK' if c2 else 'FAIL'}")
    # (3) a motion→dynamics twin (monocular): motion-scale + mass-scale → 2 anchors (metric anchor + known force) — the twin certifiable up to (geom-scale)×(mass-scale)
    r2 = gauge_report({"motion_depth_scale", "dynamics_mass_scale"})
    c3 = r2["n_anchors"] == 2 and any("FORCE" in a for a in r2["minimal_anchors"]) and any("view" in a or "anchor" in a for a in r2["minimal_anchors"])
    ok &= c3; print(f"  motion→dynamics twin: anchors={r2['minimal_anchors']} (metric + force) {'OK' if c3 else 'FAIL'}")
    # (4) cross-specificity: the scale channel lifts scale, NOT rotation (lookup is per-gauge, matching only)
    c4 = lifting_channel("similarity_scale") != lifting_channel("similarity_rotation") and "DISTANCE" in lifting_channel("similarity_scale")
    ok &= c4; print(f"  cross-specific: scale→'{lifting_channel('similarity_scale')[:22]}...' ≠ rotation channel {'OK' if c4 else 'FAIL'}")
    # (5) unknown gauge handled
    r3 = gauge_report({"similarity_scale", "made_up_gauge"})
    c5 = r3["unknown_gauges"] == ["made_up_gauge"] and r3["n_gauges"] == 1
    ok &= c5; print(f"  unknown gauge flagged: {r3['unknown_gauges']} {'OK' if c5 else 'FAIL'}")
    # (6) decorrelation-aware min anchors: a single full-pose channel breaks 2 correlated gauges (2 nulls) → matroid min 1 < one-per-gauge 2
    import numpy as np
    n1 = np.array([1.0, 0, 0, 0]); n2 = np.array([0, 1.0, 0, 0])         # two gauge (null) directions
    F = np.outer([0, 0, 1.0, 0], [0, 0, 1, 0]) + np.outer([0, 0, 0, 1.0], [0, 0, 0, 1])  # rank-2, nulls along n1,n2
    single = {"anchor_n1": n1, "anchor_n2": n2}                          # one-per-gauge channels
    both = {"full_pose": np.stack([n1, n2]), "anchor_n2": n2}            # a channel spanning BOTH nulls
    r_single = decorrelation_aware_min_anchors(F, single); r_both = decorrelation_aware_min_anchors(F, both)
    c6 = r_single["n_anchors"] == 2 and r_both["n_anchors"] == 1 and r_both["certified"]
    ok &= c6; print(f"  decorrelation-aware anchors: one-per-gauge={r_single['n_anchors']} vs full-pose-matroid={r_both['n_anchors']} (correlated→1) {'OK' if c6 else 'FAIL'}")
    # (7) acquisition_plan composes a kind-cert (1 RANK + 1 CONDITIONING + identifiable) → plan has acquire-new AND pour actions
    import numpy as np
    kc = {"directions": [{"kind": "RANK_NULL", "n_req_to_resolve": None}, {"kind": "CONDITIONING_NULL", "n_req_to_resolve": 5},
                         {"kind": "IDENTIFIABLE", "n_req_to_resolve": None}]}
    Fk = np.diag([0.0, 0.01, 1.0]); chans = {"new_ch": np.array([1.0, 0, 0])}
    plan = acquisition_plan(kc, F=Fk, channel_jacobians=chans)
    acts = " ".join(p["action"] for p in plan["plan"])
    c7 = (not plan["ready"]) and plan["n_rank_null"] == 1 and plan["n_conditioning_null"] == 1 and "ACQUIRE 1 NEW" in acts and "POUR ~5" in acts
    ok &= c7; print(f"  acquisition_plan (1 RANK→acquire + 1 CONDITIONING→pour): ready={plan['ready']} summary='{plan['summary']}' {'OK' if c7 else 'FAIL'}")
    print(f"\n  gauge_catalog self-test: {'ALL PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _selftest()
