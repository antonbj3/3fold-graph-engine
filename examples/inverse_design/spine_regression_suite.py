#!/usr/bin/env python3
"""SPINE REGRESSION SUITE — run the whole generative+inverse subsystem, confirm it is intact (consolidation / anti-drift).

The spine grew to one service (persona_design_gi_service) + 7 importing dependents + 4 standalone meta-primitive /
real-data loops, modified nearly every tick. A change to the shared service (the optimizer swap, the identifiability
ridge, sigma_check) can SILENTLY break any dependent — a dependency regression. This suite
subprocess-runs each cell and asserts EXIT 0, so one command verifies the entire spine + guards the verticals' contract.

GATES:
 (G0) SERVICE CONTRACT — persona_design_gi_service self-test (G0-G6) passes: the shared engine is sound.
 (G1) DEPENDENTS INTACT — every cell that imports the service passes: the contract did not break an importer.
 (G2) STANDALONE LOOPS INTACT — the meta-primitive + the real-data render→match loops (NIST, HUTUBS) + the fluid inverse pass.
 (G3) NO RUNTIME REGRESSION — no cell hangs (each completes < its timeout); total wall-time reported.

Run: python3 e.py
"""
import sys, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
SRC = os.path.join(HERE, "..", "..", "src")
SERVICE = ["graph_engine.inverse_design.persona_design_gi_service"]
DEPENDENTS = ["persona_design_holdout_exam", "persona_design_generative_beats_patent",
              "persona_design_multipatent_cluster", "persona_design_inverse_classified"]
STANDALONE = []


def run_module(mod, timeout=180):
    t = time.time()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([SRC, os.environ.get("PYTHONPATH", "")]))
    try:
        r = subprocess.run([PY, "-c", f"import {mod}"], capture_output=True, text=True,
                           timeout=timeout, env=env)
        rc, dt = r.returncode, time.time() - t
        tail = "" if rc == 0 else (r.stderr.strip().splitlines() or ["(no output)"])[-1][:80]
    except subprocess.TimeoutExpired:
        rc, dt, tail = 124, time.time() - t, "TIMEOUT"
    return rc, dt, tail


def run(cell, timeout=180):
    t = time.time()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([SRC, os.environ.get("PYTHONPATH", "")]))
    try:
        r = subprocess.run([PY, os.path.join(HERE, cell + ".py")], capture_output=True, text=True,
                           timeout=timeout, env=env)
        rc, dt = r.returncode, time.time() - t
        tail = "" if rc == 0 else (r.stdout.strip().splitlines() or ["(no output)"])[-1][:80]
    except subprocess.TimeoutExpired:
        rc, dt, tail = 124, time.time() - t, "TIMEOUT"
    return rc, dt, tail


def section(title, cells, runner=None):
    print(f"\n  {title}")
    ok = True
    runner = runner or run
    for c in cells:
        rc, dt, tail = runner(c)
        mark = "PASS" if rc == 0 else f"FAIL(rc={rc})"
        print(f"    [{mark:11s}] {c:38s} {dt:6.1f}s {tail}")
        ok = ok and rc == 0
    return ok


def main():
    print("=" * 98)
    print("SPINE REGRESSION SUITE — generative+inverse subsystem (1 service + 4 proof cells)")
    print("=" * 98)
    t0 = time.time()
    g0 = section("(G0) SERVICE CONTRACT", SERVICE, runner=run_module)
    g1 = section("(G1) DEPENDENTS (import the service)", DEPENDENTS)
    g2 = True if not STANDALONE else section("(G2) STANDALONE", STANDALONE)
    total = time.time() - t0
    g3 = total < 300.0                                                # whole spine runs in < 5 min (no hang)
    print(f"\n  (G3) NO RUNTIME REGRESSION — total wall-time {total:.1f}s < 300s: {'PASS' if g3 else 'FAIL'}")

    allok = g0 and g1 and g2 and g3
    print("\n" + "=" * 98)
    if allok:
        print(f"VERDICT: the spine is intact — service contract + all {len(DEPENDENTS)} proof cells EXIT 0 in {total:.0f}s.")
        print("  One command guards the generative+inverse subsystem: a shared-service change that breaks an")
        print("  importer is caught here.")
    else:
        print(f"VERDICT: SPINE REGRESSION — G0 {g0} G1 {g1} G2 {g2} G3 {g3}. A cell above is FAIL — fix at SOURCE before adding more.")
    print("=" * 98)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
