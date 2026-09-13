"""gauge_aware_admit — INNOV-2: the legitimate NON-EMPTY AUTO_ADMIT yield-path. The video/twin AUTO_ADMIT
gate is empirically EMPTY only because it demands FULL certification — and a σ_min-null GAUGE (an unobservable direction the data cannot see;
e.g. monocular scale, an absolute pose, a compositional constant-sum) blocks SOME quantities-of-interest. But it does NOT block the QoIs that
live OUTSIDE the gauge. This module admits those (certifiable UP TO the gauge = a real yield TODAY) and names the minimal metric ANCHOR to lift
the gauge-bound ones.

  gauge_aware_admit(F, qoi_gradients, channel_jacobians=None) -> per-QoI ADMIT / GAUGE-BOUND + acquisition plan + verdict.

Given a deployment observation Fisher F (D×D), the QoIs' gradients ∂QoI/∂params, and the available acquisition channels:
  (1) gauge = σ_min-NULL subspace of F (eigen-directions with eigenvalue < tol_ratio·λ_max) — the directions data cannot resolve;
  (2) a QoI is GAUGE-BOUND iff its gradient has a significant component IN the gauge (|P_gauge g|/|g| > in_gauge_frac) → its VALUE is
      unrecoverable → ABSTAIN/acquire; else it is IDENTIFIABLE up-to-gauge → ADMIT;
  (3) for the gauge-bound QoIs, gauge_catalog.decorrelation_aware_min_anchors names the minimal DECORRELATED channel set that lifts the gauge.
Composes J gauge_catalog (the two laws: a gauge is a σ_min-null; DATA-liftable ONLY by its matching decorrelated channel). Complements
O7 difficulty-stratified sync-gating (that admits the EASY-difficulty strata; this admits the GAUGE-FREE QoIs — both are partial-yield, not
empty). ★SCOPE: this is an IDENTIFIABILITY leg — it answers 'is the QoI RECOVERABLE from the data?', NOT 'is the data
AUTHENTIC?'. It is INTEGRITY-BLIND: a coherent-fake with a full-rank Fisher (no gauge) ADMITs every QoI. So gauge_aware_admit is NOT a
standalone AUTO_ADMIT — it must COMPOSE with an INTEGRITY check (an orthogonal cross-modal edge). `cross_modal_admit` is that composition:
axis certified by an orthogonal decorrelated edge (integrity, on the adversarial minimum) AND gauge-free (this leg). NULL-SAFE: degenerate/
non-finite F or no QoIs → ABSTAIN, never a false admit. numpy only. Selftest: run this file.
"""
from __future__ import annotations
import numpy as np, sys  # ★fix: sys used but not imported = live NameError (pyflakes)
from .linalg_guards import guarded_eigh

try:
    from graph_engine.gauge_catalog import decorrelation_aware_min_anchors
except Exception:
    decorrelation_aware_min_anchors = None

__all__ = ["gauge_aware_admit"]

def gauge_aware_admit(F, qoi_gradients, channel_jacobians=None, tol_ratio=1e-6, in_gauge_frac=0.10):
    """F: (D,D) observation Fisher (symmetric PSD). qoi_gradients: {name: (D,) gradient of the QoI w.r.t. the params}. channel_jacobians:
    optional {name: Jacobian (rows = observation gradients)} candidate acquisition channels. Returns dict(gauge_dim, admitted [(name,
    gauge_overlap)], gauge_bound [(name, gauge_overlap)], acquisition_plan, verdict, cert_valid)."""
    def _abstain(reason):
        return {"gauge_dim": None, "admitted": [], "gauge_bound": [], "acquisition_plan": None, "cert_valid": False,
                "verdict": "ABSTAIN: " + reason}
    try:
        F = np.asarray(F, float)
    except Exception as e:
        return _abstain("non-numeric F (%s)" % e)
    if F.ndim != 2 or F.shape[0] != F.shape[1] or not np.all(np.isfinite(F)):
        return _abstain("F must be a finite square matrix; got %s" % (F.shape,))
    if not qoi_gradients:
        return _abstain("no QoI gradients supplied (nothing to admit)")
    D = F.shape[0]
    w, V = guarded_eigh((F + F.T) / 2.0)
    smax = max(float(w.max()), 1e-300)
    gauge_cols = V[:, w < tol_ratio * smax]                              # σ_min-null subspace = the gauge
    g_dim = int(gauge_cols.shape[1])
    admitted, gauge_bound = [], []
    for name, grad in qoi_gradients.items():
        try:
            g = np.asarray(grad, float).reshape(-1)
        except Exception:
            gauge_bound.append((str(name), None)); continue            # unparseable QoI -> conservative gauge-bound (never silently admit)
        if g.size != D or not np.all(np.isfinite(g)):
            gauge_bound.append((str(name), None)); continue
        ng = float(np.linalg.norm(g)) + 1e-12
        in_gauge = float(np.linalg.norm(gauge_cols.T @ g) / ng) if g_dim else 0.0
        (gauge_bound if in_gauge > in_gauge_frac else admitted).append((str(name), round(in_gauge, 3)))
    plan = None
    if gauge_bound and channel_jacobians and decorrelation_aware_min_anchors is not None:
        try:
            m = decorrelation_aware_min_anchors(F, channel_jacobians); plan = {"channels": m["chosen"], "certified": m["certified"]}
        except Exception:
            plan = None
    verdict = ("NON-EMPTY: ADMIT %d QoI(s) up-to-gauge; %d gauge-bound need an anchor" % (len(admitted), len(gauge_bound))
               if admitted else "EMPTY: all %d QoIs are gauge-bound -> ACQUIRE before any admit" % len(gauge_bound))
    return {"gauge_dim": g_dim, "admitted": admitted, "gauge_bound": gauge_bound,
            "acquisition_plan": plan, "cert_valid": True, "integrity_verified": False,  # gate: identifiability only; a consumer MUST NOT treat ADMIT as final without an integrity edge (cross_modal_admit)
            "verdict": verdict}

def _selftest():
    ok = True
    # (1) monocular scale gauge: scale-free QoIs ADMIT, absolute-scale QoI GAUGE-BOUND + anchor named
    n = 3; J = np.eye(n) - np.ones((n, n)) / n; F = J.T @ J
    qois = {"ratio": np.array([1.0, -1.0, 0.0]), "shape": np.array([1.0, 0.0, -1.0]), "absolute": np.ones(n)/np.sqrt(n)}
    r1 = gauge_aware_admit(F, qois, {"stereo": np.ones((1, n))/np.sqrt(n)})
    c1 = ([a[0] for a in r1["admitted"]] == ["ratio", "shape"] and [g[0] for g in r1["gauge_bound"]] == ["absolute"]
          and r1["acquisition_plan"] and r1["acquisition_plan"]["certified"]
          and r1["integrity_verified"] is False)   # gate MUST be present + False on a valid admit (reflexive-sentinel fix)
    print("  (1) monocular scale gauge: admit=%s gauge_bound=%s plan_certified=%s  %s"
          % ([a[0] for a in r1["admitted"]], [g[0] for g in r1["gauge_bound"]], r1["acquisition_plan"]["certified"] if r1["acquisition_plan"] else None, "OK" if c1 else "FAIL"))
    # (2) FULL-RANK Fisher (no gauge): everything ADMITs
    r2 = gauge_aware_admit(np.eye(3), {"a": np.array([1., 0, 0]), "b": np.array([0, 1., 0])})
    c2 = r2["gauge_dim"] == 0 and len(r2["admitted"]) == 2 and not r2["gauge_bound"]
    print("  (2) full-rank (no gauge): admit=%d gauge_bound=%d  %s" % (len(r2["admitted"]), len(r2["gauge_bound"]), "OK" if c2 else "FAIL"))
    # (3) ALL-gauge QoI (only a gauge-aligned QoI): EMPTY verdict
    r3 = gauge_aware_admit(F, {"absolute_only": np.ones(n)/np.sqrt(n)})
    c3 = not r3["admitted"] and r3["verdict"].startswith("EMPTY")
    print("  (3) only-gauge-bound QoI: verdict=%s  %s" % (r3["verdict"][:28], "OK" if c3 else "FAIL"))
    # (4) NULL-SAFE: degenerate/non-finite F, empty QoIs, bad gradient -> ABSTAIN / conservative, no crash, no false admit
    c4 = (gauge_aware_admit(np.full((3, 3), np.nan), qois)["cert_valid"] is False
          and gauge_aware_admit(F, {})["cert_valid"] is False
          and [g[0] for g in gauge_aware_admit(F, {"bad": np.array([1.0, 2.0])})["gauge_bound"]] == ["bad"])   # wrong-dim grad -> gauge-bound (never admitted)
    print("  (4) NULL-SAFE (nan F / empty QoIs / bad-dim grad -> abstain/conservative): %s" % ("OK" if c4 else "FAIL"))
    ok = c1 and c2 and c3 and c4
    print("\n  gauge_aware_admit selftest: %s" % ("ALL PASS" if ok else "FAIL")); return ok

if __name__ == "__main__":
    import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # merge-robust: src dir relative to THIS file, not a hardcoded worktree (post-merge stale-import guard, fleet fix)
    from graph_engine.gauge_catalog import decorrelation_aware_min_anchors   # ensure available for __main__ selftest
    print("=" * 96); print("gauge_aware_admit — INNOV-2: non-empty AUTO_ADMIT via admit-up-to-gauge + name-the-anchor"); print("=" * 96)
    sys.exit(0 if _selftest() else 1)
