#!/usr/bin/env python3
"""system_map.py — an always-current map of what exists in the tree: compose previous builds instead of
rebuilding them.

Auto-generated instant overview of EVERY build (tool + one-line purpose + status + composition) so no session
ever re-builds what exists. Run at wake + on the pulse = the high-frequency loop-back. Perfect representation
= extracted live from the code's own docstrings, never hand-maintained (can't go stale).
"""
import os, glob, re, subprocess, time

# Resolve the repo root from THIS file's location, never a hardcoded absolute path
# (a hardcoded root silently maps another checkout's tree instead of the current one).
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# functional grouping — where each tool sits in the system
GROUPS = {
    "RECALL / MEMORY (loop-back)": ["op_loopback", "context_loader", "ground", "radar", "reground", "system_map", "context_loss_detector"],
    "PRESSURE / COACH (generate stress-points)": ["pressure_engine", "coach_forcing_questions", "coach_constant", "coach_selfscan"],
    "PULSE / SCHEDULING": ["metrics_snapshot", "wake_causality", "lane_context_model"],
    "DECISION / VoI": ["stop_gate", "voi_calibration", "voi_gate", "mstar_meter"],
    "SELF-MODEL / CONTEXT-TWIN": ["context_twin", "scratchpad_registry"],
    "GRAPH / CERT / FLEET": ["fleet_state", "fresh_landings_digest", "graph_store", "graph_resolution_potential", "g_fleet_gramian"],
}

def purpose(path):
    """first meaningful docstring line = the tool's purpose."""
    try:
        s = open(path, errors="ignore").read(4000)
    except Exception:
        return "?"
    m = re.search(r'"""(.+?)(?:\n|""")', s, re.S)
    if m:
        line = m.group(1).strip().split("\n")[0]
        return re.sub(r"^\w+\.py\s*[—-]\s*", "", line)[:110]
    return "(no docstring)"

def age(path):
    h = (time.time() - os.path.getmtime(path)) / 3600
    return f"{h:.0f}h" if h >= 1 else f"{int(h*60)}m"

def main():
    scripts = {os.path.splitext(os.path.basename(p))[0]: p
               for p in glob.glob(f"{BASE}/scripts/*.py") + glob.glob(f"{BASE}/*.py")}
    print("=" * 96 + "\n SYSTEM MAP — exactly where we are (compose these; DO NOT re-build)  ·  auto-live from docstrings\n" + "=" * 96)
    listed = set()
    for grp, names in GROUPS.items():
        rows = [(n, scripts[n]) for n in names if n in scripts]
        if not rows:
            continue
        print(f"\n▸ {grp}")
        for n, p in rows:
            listed.add(n)
            print(f"   {n:<26} [{age(p):>4}]  {purpose(p)}")
    # anything not grouped (new/unmapped builds)
    other = [n for n in scripts if n not in listed and not n.startswith("test") and n not in ("__init__",)]
    if other:
        print(f"\n▸ UNMAPPED (add to a group or consolidate — possible re-build/dup):")
        for n in sorted(other)[:20]:
            print(f"   {n:<26} [{age(scripts[n]):>4}]  {purpose(scripts[n])}")
    # durable artifacts (the state, not the tools)
    print("\n▸ DURABLE STATE (where the record lives)")
    for label, rel in [("graph", os.environ.get("GRAPH_DOC", "docs/GRAPH.md")),
                        ("coordination ledger", os.environ.get("LEDGER_DOC", "docs/LEDGER.md")),
                        ("notes index", os.environ.get("NOTES_INDEX", "docs/INDEX.md"))]:
        p = rel if rel.startswith("/") else f"{BASE}/{rel}"
        ex = "✓" if os.path.exists(p) else "✗MISSING"
        print(f"   {ex} {label:<32} {rel}")
    print("\n → composing beats re-building: read this map before adding a new build.")

if __name__ == "__main__":
    main()
