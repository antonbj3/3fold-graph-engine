"""
graph_hole_engine.__main__ — the demo.  `python3 -m graph_hole_engine`
======================================================================
Half 1 (SHIP): one GraphHoleEngine over the physics graph AND the paper graph.
Half 2 (SIBLING): the OED optimizer, run separately and labelled as the optimizer-half.
"""
from __future__ import annotations

import numpy as np

from .engine import GraphHoleEngine
from .adapters import PhysicsGraphAdapter, PapersAdapter
from .oed_sibling import OEDDesign


def _print_report(rep) -> None:
    top = rep.verdicts[0]
    print(f"\n[{rep.domain}]")
    print(f"  STRUCTURE leg — hole-ranking (resolvent-leverage L⁺_ii, most-hollow first): {rep.hole_ranking[:3]} ...")
    print(f"  OBSERVATION leg — residual-ranking (|C_obs⊖C_pred|, biggest anomaly first): {rep.residual_ranking[:3]} ...")
    print(f"  two decorrelated legs: ρ(residual, structure)={rep.rho_residual_structure:.2f} "
          f"({'✓ <0.8 = separate legs' if rep.rho_residual_structure < 0.8 else 'HIGH — smells like an OED problem'})")
    print(f"  top certified hole: {top.node!r}  score={top.score:.3g}  "
          f"(residual={top.residual:.3g}, leverage={top.leverage:.3g}, N_eff={top.n_eff:.2f}, "
          f"σ_min={top.sigma_min:.3g}, credited={top.credited})")
    print(f"  abstained (unspanned, identifiability<χ): {rep.abstained}")
    for lk in rep.leaks:
        print(f"  LEAK  {lk}")


def main() -> None:
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 100)
    print("GRAPH-HOLE ENGINE — the invariant structural-hole detector (physics ↔ papers) + the OED sibling")
    print("=" * 100)

    print("\n" + "-" * 100)
    print("HALF 1 (SHIP) — ONE GraphHoleEngine, TWO graph domains (physics + papers), shared kernel:")
    for adapter in (PhysicsGraphAdapter(), PapersAdapter()):
        _print_report(GraphHoleEngine(adapter).detect())

    print("\n" + "-" * 100)
    print("HALF 2 (SIBLING) — OED optimizer-half: SAME kernel functions (resolvent_leverage as d(x),")
    print("waterfill_reallocate), DIFFERENT question (drive a config-dependent design to the KW floor):")
    oed = OEDDesign(deg=3, n_grid=201)
    d0 = oed.kw_max_variance()
    oed.optimize(4000)
    print(f"  start (uniform design):  max_x d(x) = {d0:.3f}")
    print(f"  after OC ≡ Fedorov-Wynn: max_x d(x) = {oed.kw_max_variance():.3f} → KW D-optimal floor p = {oed.p}")
    print(f"  interior support centroid = {oed.interior_support_centroid():.4f} → analytic 1/√5 = {1/np.sqrt(5):.4f}")
    print("  NOTE: this reallocates a BUDGET (waterfill) — an operation the hole-engine never performs.")

    print("\n" + "-" * 100)
    print("Machine-checks (byte-equality vs C's & I's committed code, transfer, leaks, abstain, sibling):")
    print("  run:  python3 -m graph_hole_engine.tests")
    print("=" * 100)


if __name__ == "__main__":
    main()
