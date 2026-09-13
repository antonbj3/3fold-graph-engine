"""
cert_decorrelation.py — is a suite of checks INDEPENDENT evidence, or is it one check wearing
several hats?

Each check ("cert") is a predicate over system states. Run every cert against a battery of
injected failure modes and read the boolean CATCH MATRIX. Its columns are the certs'
catch-signatures:

  * distinct non-empty signatures = DECORRELATED coverage rank (how many independent things the
    suite actually sees),
  * identical signatures        = REDUNDANT certs (they add no rank; drop one to save compute),
  * a failure mode caught by no cert = a BLIND kind (an exact null direction of the coverage
    Gramian) = the kind of evidence to acquire next.

The catch matrix is a binary observability matrix over failure modes, so the same linear algebra
that answers "which parameter is unobservable?" answers "what is this test suite blind to?".

API: `analyze(certs, failure_states, baseline_state, names)` -> structured report;
`coverage_record(...)` -> a flat record for a coverage registry; `abstain_propagation` /
`compose_dag` / `blast_radius` / `acquisition_priority` / `cert_dag_report` extend this from one
stage to a DAG of composed checks.

Run `python cert_decorrelation.py` for the self-test (9 machine-checked assertions).
"""
from __future__ import annotations
import numpy as np

__all__ = ["catch_matrix", "analyze", "coverage_record", "eff_rank_of_margins", "abstain_propagation", "compose_dag", "blast_radius", "acquisition_priority", "cert_dag_report"]


def catch_matrix(certs, failure_states, baseline_state=None):
    """
    Boolean matrix M[failure, cert] = does cert flag on that failure's state. `certs` = list of callables state→bool (True = flags a problem). `failure_states` = list of
    states each triggering one intended failure mode. If baseline_state given, certs flagging on it are BASELINE-DIRTY (false alarms) — reported, not silently trusted.
    """
    C = list(certs); F = list(failure_states)
    M = np.zeros((len(F), len(C)), dtype=bool)
    for i, st in enumerate(F):
        for j, c in enumerate(C):
            M[i, j] = bool(c(st))
    dirty = np.array([bool(c(baseline_state)) for c in C]) if baseline_state is not None else np.zeros(len(C), bool)
    return M, dirty


def analyze(certs, failure_states, baseline_state=None, cert_names=None, failure_names=None):
    """
    Decompose a cert-suite's decorrelated coverage. Returns:
    coverage_rank      — # failure-modes caught by ≥1 cert (the decorrelated coverage = G_fleet knowledge-rank contribution)
    decorrelated_certs — # DISTINCT non-empty catch-signatures (redundant duplicates collapse; this is what adds σ_min(G_fleet))
    blind              — failure-modes caught by NO cert (coverage-Gramian null = the KIND to acquire)
    redundant_groups   — sets of certs with identical catch-signatures (keep one; the rest add volume not rank)
    dead               — certs that catch nothing (no coverage value)
    per_cert           — {name: {catches, unique (failures it alone catches), verdict}}
    baseline_dirty     — certs that false-alarm on the healthy baseline
    """
    M, dirty = catch_matrix(certs, failure_states, baseline_state)
    nF, nC = M.shape
    cn = cert_names or [f"cert{j}" for j in range(nC)]
    fn = failure_names or [f"fail{i}" for i in range(nF)]

    caught_by_any = M.any(axis=1)
    blind = [fn[i] for i in range(nF) if not caught_by_any[i]]
    coverage_rank = int(caught_by_any.sum())

    # signatures = each cert's column (which failures it catches); group identical, non-empty
    sigs = {}
    for j in range(nC):
        key = tuple(M[:, j].tolist()); sigs.setdefault(key, []).append(j)
    redundant_groups = [[cn[j] for j in idxs] for key, idxs in sigs.items() if any(key) and len(idxs) > 1]
    dead = [cn[j] for j in range(nC) if not M[:, j].any()]
    decorrelated_certs = sum(1 for key in sigs if any(key)) # distinct informative signatures

    # unique catches: failures caught by exactly one cert, credited to that cert
    per_cert = {}
    col_counts = M.sum(axis=1)
    for j in range(nC):
        catches = [fn[i] for i in range(nF) if M[i, j]]
        unique = [fn[i] for i in range(nF) if M[i, j] and col_counts[i] == 1]
        if not catches:
            verdict = "DEAD"
        elif any(cn[j] in g for g in redundant_groups):
            verdict = "REDUNDANT"
        else:
            verdict = "UNIQUE"
        per_cert[cn[j]] = dict(catches=catches, unique=unique, verdict=verdict, baseline_dirty=bool(dirty[j]))

    return dict(coverage_rank=coverage_rank, n_failures=nF, decorrelated_certs=decorrelated_certs,
                blind=blind, redundant_groups=redundant_groups, dead=dead,
                baseline_dirty=[cn[j] for j in range(nC) if dirty[j]], per_cert=per_cert,
                matrix=M.astype(int).tolist(), cert_names=list(cn), failure_names=list(fn))


def eff_rank_of_margins(margin_matrix):
    """Numeric companion (when certs return a MARGIN not a bool): eff_rank of the [failure × cert] margin-signature matrix = decorrelated cert count on a continuous scale"""
    X = np.asarray(margin_matrix, float)
    if X.shape[0] < 2 or X.shape[1] < 1:
        return 0.0
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    # svd is SILENT on NaN/inf input (it returns NaN singular values): guard the matrix before
    # the primitive instead of letting NaN poison everything downstream.
    if not np.isfinite(Z).all():
        raise ValueError("eff_rank_of_margins: matrix must be finite (svd on NaN/inf -> silent NaN singular values)")
    s = np.linalg.svd(Z, compute_uv=False); s = s[s > 1e-9]
    return float(s.sum() ** 2 / (s ** 2).sum()) if s.size else 0.0


def coverage_record(lane, suite_name, certs, failure_states, baseline_state=None, cert_names=None, failure_names=None):
    """A flat agent pool record to emit alongside the anchor_catalog: the suite's decorrelated-coverage contribution to G_fleet + its redundancy/blind spots."""
    a = analyze(certs, failure_states, baseline_state, cert_names, failure_names)
    return {"lane": lane, "suite": suite_name, "n_certs": len(list(certs)),
            "coverage_rank": a["coverage_rank"], "n_failures": a["n_failures"],
            "decorrelated_certs": a["decorrelated_certs"], "blind_kinds": a["blind"],
            "redundant_groups": a["redundant_groups"], "dead_certs": a["dead"],
            "baseline_dirty": a["baseline_dirty"], "_schema": "cert_decorrelation_v1"}


def abstain_propagation(items, upstream_grounded, downstream_fn, naive_downstream_fn=None):
    """
    In a PIPELINE cert (stage A feeds stage B), a passing DOWNSTREAM rung does NOT rescue an ABSTAINED UPSTREAM item — the
    composed verdict must INHERIT the abstain, never certify through it (ties downstream-cert-inherits-upstream-null,
    
    items: the per-item units (joints / points / parameters / sources). upstream_grounded: parallel bool list
    subset. naive_downstream_fn defaults to downstream_fn; it models the WRONG composition that ignores the upstream abstain
    (runs stage B on ALL items incl. abstained). Returns the HONEST verdict (stage B on GROUNDED items only), the NAIVE verdict,
    a FALSE_CERTIFY flag (naive passes while the honest composition must abstain), and model_abstains.
    """
    items = list(items); flags = list(upstream_grounded)
    # zip-truncation class: a short `flags` list silently drops the trailing items from BOTH
    # grounded and abstained (neither certified nor flagged = fail-open). Require parallel lengths.
    if len(items) != len(flags):
        raise ValueError(f"compose: upstream_grounded ({len(flags)}) must be parallel with items ({len(items)}) "
                         f"- zip would silently drop the trailing items from the cert")
    grounded = [it for it, ok in zip(items, flags) if ok]
    abstained = [it for it, ok in zip(items, flags) if not ok]
    honest = downstream_fn(grounded)
    naive = (naive_downstream_fn or downstream_fn)(items)
    model_abstains = bool(len(abstained) > 0 or not honest.get("pass", True))
    false_certify = bool(naive.get("pass", False) and model_abstains)
    return {"grounded": grounded, "abstained": abstained, "downstream_honest": honest, "downstream_naive": naive,
            "false_certify": false_certify, "model_abstains": model_abstains,
            "rule": "a passing downstream cert does NOT rescue an abstained upstream item; inherit the abstain"}


def _toposort(deps):
    """topological order of a DAG {node: [upstream deps]} (deps must exist BEFORE the node). Raises on a cycle."""
    nodes = set(deps) | {u for us in deps.values() for u in us}
    indeg = {n: 0 for n in nodes}
    for n, us in deps.items():
        for u in us:
            indeg[n] += 1
    ready = [n for n in nodes if indeg[n] == 0]; order = []
    children = {n: [] for n in nodes}
    for n, us in deps.items():
        for u in us: children[u].append(n)
    while ready:
        u = ready.pop(); order.append(u)
        for c in children[u]:
            indeg[c] -= 1
            if indeg[c] == 0: ready.append(c)
    if len(order) != len(nodes): raise ValueError("cert DAG has a cycle")
    return order


def compose_dag(deps, local_pass):
    """
    ABSTAIN-propagation through a cert DAG (agent pool-general — generalizes abstain_propagation from ONE stage to the whole
    federation graph of composed agent worktree certs). deps: {node: [upstream nodes it CONSUMES]} (a DAG). local_pass: {node: bool} =
    each node's OWN cert verdict IGNORING upstream. A node is GROUNDED iff its own cert passes AND every upstream dep is GROUNDED
    → abstain propagates transitively along the graph. false_certify = nodes whose OWN cert passes (a naive composition would
    certify) but which are NOT grounded because an upstream contributor abstained → they must INHERIT the abstain, never certify
    through it. Returns per-node grounded + reason (topo order), the false_certify list, and n_grounded. Raises on a cycle.
    """
    order = _toposort(deps)
    grounded, reason = {}, {}
    for n in order:
        ups = deps.get(n, [])
        bad = [u for u in ups if not grounded.get(u, False)]
        g = bool(local_pass.get(n, False)) and not bad
        grounded[n] = g
        reason[n] = "grounded" if g else ("local cert FAILED" if not local_pass.get(n, False) else f"upstream ABSTAINED: {bad}")
    false_certify = [n for n in order if local_pass.get(n, False) and not grounded[n]]
    return {"grounded": grounded, "reason": reason, "false_certify": false_certify,
            "n_grounded": int(sum(grounded.values())), "order": order,
            "rule": "a node is GROUNDED iff its own cert passes AND all upstream deps are grounded; abstain propagates transitively"}


def blast_radius(deps, root):
    """
    the set of nodes whose grounding TRANSITIVELY depends on `root` (root + all its descendants in the cert DAG) — i.e., the
    downstream certs an abstain at `root` would compromise. Used to prioritize which abstain to fix first.
    """
    order = _toposort(deps)
    hit = {root}
    for n in order:
        if any(u in hit for u in deps.get(n, [])):
            hit.add(n)
    return hit


def acquisition_priority(deps, abstained_roots):
    """
    DAG-impact-prioritized acquisition (composes compose_dag's blast-radius with the acquisition plan): rank the ABSTAINED
    upstream nodes by how many DOWNSTREAM certs each un-blocks = the size of its blast-radius among the currently-compromised set.
    Fixing the highest-impact root first maximally shrinks the federation's false-certify set. Returns a ranked list of
    {root, unblocks (blast-radius incl. root), n_unblocks}. VoI of an acquisition = # downstream certs it un-blocks.
    """
    ranked = []
    for r in abstained_roots:
        br = blast_radius(deps, r)
        ranked.append({"root": r, "unblocks": sorted(br), "n_unblocks": len(br)})
    ranked.sort(key=lambda d: -d["n_unblocks"])
    return ranked


_KIND_ACTION = {"RANK_NULL": "ACQUIRE a NEW decorrelated channel (data/lever-arm will NOT lift)",
                "CONDITIONING_NULL": "POUR more samples / widen the lever-arm (same channel)"}


def cert_dag_report(deps, local_pass, node_kind=None, node_nreq=None):
    """
    The COMPLETE actionable cert-DAG verdict for a composed (agent pool) cert graph — CONSOLIDATES this module's three cert-DAG
    node_kind: optional {node: 'RANK_NULL'|'CONDITIONING_NULL'} for the abstained-at-source nodes (their null KIND) → the fix.
    node_nreq: optional {node: int} QUANTITATIVE acquisition cost (from null_kind_cert n_req_to_resolve for a CONDITIONING null =
    ~N more samples/views; for a RANK null the count = # new decorrelated channels, defaults to 1) → the fix carries HOW MUCH, not
    just which. Returns per-node grounded map, false_certify set, and a RANKED fix-list (largest blast-radius first) with kind,
    KIND-correct action, and quantitative cost. 'For the whole pool's composed cert, per node: grounded? and if not, the highest-
    impact KIND-correct fix AND how much acquisition it costs.'.
    """
    dag = compose_dag(deps, local_pass)
    roots = [n for n in dag["order"] if not local_pass.get(n, False)] # abstained SOURCES = own cert failed
    pri = acquisition_priority(deps, roots)
    node_kind = node_kind or {}; node_nreq = node_nreq or {}
    fixes = []
    for i, p in enumerate(pri):
        r = p["root"]; k = node_kind.get(r); nreq = node_nreq.get(r)
        if k == "RANK_NULL":
            n = nreq if nreq else 1; cost = f"{n} new decorrelated channel(s)/view(s)"
        elif k == "CONDITIONING_NULL":
            cost = f"~{nreq} more samples/views (or widen the lever-arm)" if nreq else "more samples / wider lever-arm"
        else:
            cost = "classify the null KIND (null_kind_cert) for the cost"
        fixes.append({"root": r, "priority": i + 1, "blast_radius": p["unblocks"], "n_unblocks": p["n_unblocks"],
                      "kind": k, "action": _KIND_ACTION.get(k, "classify the null KIND (null_kind_cert) to pick the fix"),
                      "cost": cost})
    return {"grounded": dag["grounded"], "false_certify": dag["false_certify"], "n_grounded": dag["n_grounded"],
            "ready": not dag["false_certify"] and all(dag["grounded"].values()), "ranked_fixes": fixes}


def _selftest():
    ok = True
    # model a agent worktree cert-suite as certs(state)->bool over failure states. state = which failure is active (or 'healthy').
    def cert_for(mode):
        return lambda st: (st == mode)
    fails = ["singularity", "identifiability", "topology"]

    certs = [cert_for("singularity"), cert_for("identifiability"), cert_for("topology")]
    r = analyze(certs, fails, baseline_state="healthy", cert_names=["mobility?no", "singularity", "identifiability"][1:] + ["topology"],
                failure_names=fails)
    r = analyze(certs, fails, baseline_state="healthy", cert_names=["singularity", "identifiability", "topology"], failure_names=fails)
    c1 = r["coverage_rank"] == 3 and r["decorrelated_certs"] == 3 and not r["blind"] and not r["redundant_groups"] and not r["dead"]
    ok &= c1; print(f"  (1) sim-readiness diagonal: coverage_rank={r['coverage_rank']} decorrelated={r['decorrelated_certs']} blind={r['blind']} {'OK' if c1 else 'FAIL'}")

    # (2) REDUNDANT: a 4th cert duplicates 'singularity' → flagged redundant, decorrelated stays 3
    certs2 = certs + [cert_for("singularity")]
    r2 = analyze(certs2, fails, "healthy", cert_names=["singularity", "identifiability", "topology", "sing_dup"], failure_names=fails)
    c2 = r2["decorrelated_certs"] == 3 and any("sing_dup" in g for g in r2["redundant_groups"]) and r2["per_cert"]["sing_dup"]["verdict"] == "REDUNDANT"
    ok &= c2; print(f"  (2) redundant cert (volume not rank): decorrelated still {r2['decorrelated_certs']}, redundant_groups={r2['redundant_groups']} {'OK' if c2 else 'FAIL'}")

    # (3) BLIND: a 4th failure mode caught by none → uncovered = the kind to acquire; coverage_rank 3 of 4
    fails3 = fails + ["thermal_drift"]
    r3 = analyze(certs, fails3, "healthy", cert_names=["singularity", "identifiability", "topology"], failure_names=fails3)
    c3 = r3["coverage_rank"] == 3 and r3["blind"] == ["thermal_drift"]
    ok &= c3; print(f"  (3) blind failure-mode (acquire this kind): coverage {r3['coverage_rank']}/{r3['n_failures']}, blind={r3['blind']} {'OK' if c3 else 'FAIL'}")

    # (4) DEAD cert (catches nothing) + numeric eff_rank of a redundant margin matrix
    certs4 = certs + [lambda st: False]
    r4 = analyze(certs4, fails, "healthy", cert_names=["singularity", "identifiability", "topology", "broken"], failure_names=fails)
    er_indep = eff_rank_of_margins(np.random.default_rng(0).standard_normal((8, 3))) # 3 independent columns → eff_rank ~3
    base = np.random.default_rng(1).standard_normal((8, 1))
    er_redun = eff_rank_of_margins(np.hstack([base, base, base]) + 1e-6 * np.random.default_rng(2).standard_normal((8, 3))) # 3 ~identical cols → ~1
    c4 = r4["dead"] == ["broken"] and er_indep > 2.5 and er_redun < 1.5
    ok &= c4; print(f"  (4) dead cert={r4['dead']}; margin eff_rank indep={er_indep:.2f} redundant={er_redun:.2f} {'OK' if c4 else 'FAIL'}")

    # (5) agent pool record
    rec = coverage_record("J", "sim_readiness_vector", certs, fails, "healthy", ["singularity", "identifiability", "topology"], fails)
    c5 = rec["coverage_rank"] == 3 and rec["_schema"] == "cert_decorrelation_v1"
    ok &= c5; print(f"  (5) coverage_record: {{coverage_rank:{rec['coverage_rank']}, decorrelated:{rec['decorrelated_certs']}, blind:{rec['blind_kinds']}}} {'OK' if c5 else 'FAIL'}")

    def downstream(sub): return {"pass": len(sub) >= 4, "n": len(sub)}
    prop = abstain_propagation(["a", "b", "c", "d"], [True, True, True, False], downstream)
    c6 = prop["false_certify"] and prop["model_abstains"] and prop["downstream_naive"]["pass"] and not prop["downstream_honest"]["pass"]
    ok &= c6; print(f"  (6) sequential abstain-propagation: naive_pass={prop['downstream_naive']['pass']} honest_pass={prop['downstream_honest']['pass']} FALSE_CERTIFY={prop['false_certify']} {'OK' if c6 else 'FAIL'}")

    # (7) compose_dag: chain A->B->C + side D->C; A abstains (local fail) → B,C false_certify (own pass, upstream abstained); D grounded, E depends on D ok
    deps = {"B": ["A"], "C": ["B", "D"], "E": ["D"]}
    local = {"A": False, "B": True, "C": True, "D": True, "E": True}
    dag = compose_dag(deps, local)
    c7 = (dag["grounded"] == {"A": False, "D": True, "B": False, "C": False, "E": True}
          and set(dag["false_certify"]) == {"B", "C"} and dag["n_grounded"] == 2)
    ok &= c7; print(f"  (7) compose_dag (A abstains → B,C inherit; D,E grounded): false_certify={dag['false_certify']} n_grounded={dag['n_grounded']} {'OK' if c7 else 'FAIL'}")

    # (8) acquisition_priority: J_construct un-blocks {J_simready, fleet_twin, J_construct}=3 > H_optics {fleet_twin, H_optics}=2 → fix J_construct first
    deps8 = {"J_simready": ["J_construct"], "C_gauge": ["C_recon"], "fleet_twin": ["J_simready", "C_gauge", "H_optics"]}
    pri = acquisition_priority(deps8, ["H_optics", "J_construct"])
    c8 = pri[0]["root"] == "J_construct" and pri[0]["n_unblocks"] == 3 and pri[1]["root"] == "H_optics" and pri[1]["n_unblocks"] == 2
    ok &= c8; print(f"  (8) acquisition_priority (fix highest-impact abstain first): {[(p['root'], p['n_unblocks']) for p in pri]} {'OK' if c8 else 'FAIL'}")

    # (9) cert_dag_report: 2 abstained sources (J_construct RANK, H_optics CONDITIONING) → ranked fixes by blast-radius + KIND action
    deps9 = {"J_simready": ["J_construct"], "sim_ready": ["J_simready", "H_optics"]}
    local9 = {"J_construct": False, "H_optics": False, "J_simready": True, "sim_ready": True}
    kinds9 = {"J_construct": "RANK_NULL", "H_optics": "CONDITIONING_NULL"}
    rep = cert_dag_report(deps9, local9, kinds9, node_nreq={"H_optics": 7})
    f0 = rep["ranked_fixes"][0]; fH = next(f for f in rep["ranked_fixes"] if f["root"] == "H_optics")
    c9 = (not rep["ready"]) and f0["root"] == "J_construct" and f0["priority"] == 1 and "ACQUIRE a NEW" in f0["action"] \
        and "POUR" in fH["action"] and "~7" in fH["cost"] and "channel" in f0["cost"]
    ok &= c9; print(f"  (9) cert_dag_report (ranked fixes + quantitative cost): top={f0['root']}(n={f0['n_unblocks']}) cost='{f0['cost']}' | H_optics cost='{fH['cost']}' {'OK' if c9 else 'FAIL'}")

    print(f"\n  cert_decorrelation self-test: {'ALL PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _selftest()
