"""graph_hole_engine — a certified INVARIANT structural-hole detector + its OED sibling.

The kernel FUNCTIONS generalize as code, but they serve TWO SEMANTIC HALVES that ask them
DIFFERENT questions.

  HALF 1 — THE SHIPPABLE THING: the GraphHoleEngine.
    Finds INVARIANT structural holes in a domain's coupling graph:
        resolvent-leverage L⁺_ii  ⊕  N_eff over-determination  ⊕  abstain-when-unspanned  +  an
        external residual channel (C_obs ⊖ C_pred) that is a 2nd, decorrelated leg.
    Transfers physics ↔ PAPER on the SAME engine + SAME kernel. A new domain author subclasses
    DomainAdapter and gets detection + over-det + abstain for free.

  HALF 2 — THE DOCUMENTED SIBLING: OEDDesign  (graph_hole_engine.oed_sibling).
    CONFIG-DEPENDENT optimal experimental design (topopt / Fedorov-Wynn). Reuses the SAME kernel
    functions (resolvent_leverage read as d(x); waterfill_reallocate) but asks argmax-info of a
    design you CHANGE, not argmin-identifiability of a graph you DETECT. Kept, clearly labelled,
    NOT folded into the hole-engine — it is a sibling, not the same engine.

  THE SHARED KERNEL: graph_hole_engine.kernel — four byte-proven pure-numpy functions both halves
    draw on. resolvent_leverage is the bridge (drawn by both, read differently by each).

Public API for a NEW HOLE-DETECTION DOMAIN author (the common case):
    from graph_hole_engine import DomainAdapter, GraphHoleEngine, HoleReport
    class MyDomain(DomainAdapter):...          # implement nodes/governing_operator/c_obs/c_pred/confirmer_matrix
    report = GraphHoleEngine(MyDomain).detect

Public API for a DESIGN author (the sibling):
    from graph_hole_engine import OEDDesign
    design = OEDDesign(deg=3); design.optimize

Runnable:  python3 -m graph_hole_engine        (the demo — one engine over physics+papers, sibling OED)
Tested:    python3 -m graph_hole_engine.tests  (byte-equality machine-checks + transfer + leaks + abstain)
Pure numpy; no project dependencies.
"""
from .kernel import (
    participation_ratio,
    resolvent_leverage,
    spectral_sigma_min,
    waterfill_reallocate,
)
from .engine import (
    DomainAdapter,
    NodeVerdict,
    HoleReport,
    GraphHoleEngine,
)
from .adapters import (
    PhysicsGraphAdapter,
    PapersAdapter,
)
from .oed_sibling import OEDDesign

__all__ = [
    # --- the shared kernel (both halves draw on these) ---
    "participation_ratio", "resolvent_leverage", "spectral_sigma_min", "waterfill_reallocate",
    # --- HALF 1: the shippable hole-detector + its public adapter surface ---
    "DomainAdapter", "GraphHoleEngine", "HoleReport", "NodeVerdict",
    "PhysicsGraphAdapter", "PapersAdapter",
    # --- HALF 2: the documented OED sibling ---
    "OEDDesign",
]
