"""
unified_graph_engine — MOVED. This file is now a thin entrypoint into the graph_hole_engine package.
====================================================================================================
The original "one engine, three domains" monolith was refactored into a package after three
independent reviews agreed on the same split: the kernel FUNCTIONS generalize as code, but there
are TWO SEMANTIC HALVES —

  (1) the INVARIANT hole-detector (resolvent-leverage L⁺_ii ⊕ N_eff over-det ⊕ abstain-when-unspanned)
      transfers physics ↔ PAPER   →  papers/graph_hole_engine/  (GraphHoleEngine + DomainAdapter)
  (2) OED/topopt/AMR is the CONFIG-DEPENDENT optimizer-half — same kernel code, different question
      →  papers/graph_hole_engine/oed_sibling.py  (OEDDesign), kept as a documented SIBLING, NOT
         folded into the hole-engine.

RENAME MAP (the old monolith → the package):
    RecursiveGraphEngine        -> GraphHoleEngine        (.search  ->.detect)
    SearchResult                -> HoleReport             (dropped `endogenous`; always external channel)
    OEDAdapter (a DomainAdapter) -> OEDDesign             (a SIBLING; NO LONGER a DomainAdapter — that
                                                            was the over-unification H caught)
    participation_ratio / resolvent_leverage / spectral_sigma_min / waterfill_reallocate  -> unchanged (kernel)
    PhysicsGraphAdapter / PapersAdapter / DomainAdapter / NodeVerdict                       -> unchanged

New public API (target this):
    from graph_hole_engine import DomainAdapter, GraphHoleEngine, HoleReport   # a new hole-detection domain
    from graph_hole_engine import OEDDesign                                    # the optimizer sibling

Run:    python3 papers/unified_graph_engine.py    (runs the package demo; == python3 -m graph_hole_engine)
Tests:  python3 -m graph_hole_engine.tests
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # put papers/ on the path

# re-export the whole new public API so `from unified_graph_engine import *` still resolves ----------
from graph_hole_engine import (                                        # noqa: E402,F401
    participation_ratio, resolvent_leverage, spectral_sigma_min, waterfill_reallocate,
    DomainAdapter, GraphHoleEngine, HoleReport, NodeVerdict,
    PhysicsGraphAdapter, PapersAdapter,
    OEDDesign,
)

# one safe back-compat alias (pure rename, identical shape minus the dropped `endogenous` field) -----
SearchResult = HoleReport

# the re-exports above are deliberate (this module is a compat surface): declare them so `import *`
# and linters both see intent.
__all__ = [
    "participation_ratio", "resolvent_leverage", "spectral_sigma_min", "waterfill_reallocate",
    "DomainAdapter", "GraphHoleEngine", "HoleReport", "NodeVerdict",
    "PhysicsGraphAdapter", "PapersAdapter", "OEDDesign",
    "SearchResult",
]

if __name__ == "__main__":
    from graph_hole_engine.__main__ import main
    main()
