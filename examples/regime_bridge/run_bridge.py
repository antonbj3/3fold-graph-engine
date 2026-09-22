#!/usr/bin/env python3
"""Example: existing Graph worktree + Kernel worktree -> native compiled regime bridge.

Paths come from the ENVIRONMENT (no absolute paths live in the library):

  GRAPH_ENGINE_SRC   directory containing `graph_engine/` (an existing Graph worktree)
  KERNEL_ENGINE_SRC  directory containing `kernel_engine/` (an existing Kernel worktree)
  BRIDGE_BUILD_DIR   writable directory for the compiled native library

Run:
  GRAPH_ENGINE_SRC=... KERNEL_ENGINE_SRC=... BRIDGE_BUILD_DIR=/tmp/nib_build \
      python3 examples/regime_bridge/run_bridge.py
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    graph_src = os.environ.get("GRAPH_ENGINE_SRC")
    kernel_src = os.environ.get("KERNEL_ENGINE_SRC")
    build_dir = os.environ.get("BRIDGE_BUILD_DIR")
    if not (graph_src and kernel_src and build_dir):
        print("set GRAPH_ENGINE_SRC, KERNEL_ENGINE_SRC and BRIDGE_BUILD_DIR", file=sys.stderr)
        return 2
    for p in (graph_src, kernel_src):
        if p not in sys.path:
            sys.path.insert(0, p)

    from graph_engine.regime_posterior import RegimePosterior
    from graph_engine.compiled_regime import CompiledRegime
    from kernel_engine.inference.finite_regime import FiniteRegimeKernel

    kernel = FiniteRegimeKernel(build_dir)          # explicit caller injection
    regime = RegimePosterior(0.0, 1.0, n_grid=16, p_two=0.05)
    regime.add_claim(0.20, 0.40, +1, n_eff=3.0, reliability=0.8)
    regime.add_claim(0.70, 0.85, -1, n_eff=2.0, reliability=0.9)
    regime.add_probe(0.55, +1, reliability=0.9)

    compiled = CompiledRegime.from_regime(regime, kernel)

    print("cells:", compiled.n_cells, "| domain integral potential:",
          round(compiled.potential_value(), 6), "bits")
    print("normalized task risk (log):", round(compiled.task_risk(loss="log"), 6))
    print("p_plus(0.30):", round(compiled.p_plus(0.30), 6),
          "| graph p_plus(0.30):", round(regime.p_plus(0.30), 6))
    print("family mass 0/1/2:", [round(float(x), 6) for x in compiled.family_mass()])
    print("log evidence:", round(compiled.log_evidence(), 6))

    x, gain = compiled.best_probe(reliability=0.95)
    print(f"next probe: x={x:.4f} expected potential drop={gain:.6f} bits")
    compiled.add_probe(x, +1, reliability=0.95, evidence_id="instrument/run1/reading1")
    print("after probe p_plus(0.30):", round(compiled.p_plus(0.30), 6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
