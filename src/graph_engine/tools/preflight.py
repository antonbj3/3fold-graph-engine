#!/usr/bin/env python3
"""preflight.py -- the session-start read path, as a skeleton.

An agent (or a person) starting work in a repository needs four things surfaced before the first
edit, every time, without remembering to ask:

  1. RECENCY      -- how old is the newest evidence-ledger entry? A stale ledger means the graph
                     you are about to trust has not been written to for a while.
  2. GRAPH ENTRY  -- the ranked next actions from the dependency graph, so work starts from the
                     graph rather than from whatever is on screen.
  3. SEARCH FIRST -- the prior-art search command, printed as an instruction: search before you
                     build.
  4. INTEGRITY    -- whether the graph currently validates against the ledger.

This is a skeleton: it prints the four sections and exits non-zero if the graph does not validate.
Wire it into whatever start-of-session hook your environment has.

    python3 preflight.py [--graph ANCHOR_GRAPH.json] [--ledger FOLD_LEDGER.jsonl] [--top 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anchor_graph_tools  # noqa: E402
import fold_ledger_tools  # noqa: E402


def ledger_recency(ledger_path: Path) -> str:
    """Age of the newest ledger row, from the file mtime. Returns a human-readable line."""
    if not ledger_path.exists():
        return f"no ledger at {ledger_path} -- nothing recorded yet"
    rows = fold_ledger_tools.load_ledger(ledger_path)
    age_h = (time.time() - ledger_path.stat().st_mtime) / 3600.0
    return f"{len(rows)} entries, newest write {age_h:.1f} h ago"


def main() -> int:
    ap = argparse.ArgumentParser(description="session-start read path")
    ap.add_argument("--graph", default=os.environ.get("ANCHOR_GRAPH", "ANCHOR_GRAPH.json"))
    ap.add_argument("--ledger", default=os.environ.get("FOLD_LEDGER", "FOLD_LEDGER.jsonl"))
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    graph_path, ledger_path = Path(args.graph), Path(args.ledger)
    print("=" * 78)
    print("PREFLIGHT")
    print("=" * 78)

    print("\n[1] EVIDENCE RECENCY")
    print("   ", ledger_recency(ledger_path))

    if not graph_path.exists():
        print(f"\n[2] GRAPH ENTRY\n    no graph at {graph_path}")
        return 1
    graph = anchor_graph_tools.load_graph(graph_path)

    print("\n[2] GRAPH ENTRY -- ranked next actions")
    for row in anchor_graph_tools.next_actions(graph)[: args.top]:
        print(f"    {row['id']:<16} load={row.get('load')} risk={row.get('risk')}  {row.get('action', '')}")

    print("\n[3] SEARCH BEFORE YOU BUILD")
    print("    python3 tool_find.py <keyword> [keyword...]   # exit 0 = prior art exists, read it")

    print("\n[4] GRAPH INTEGRITY")
    res = anchor_graph_tools.validate(graph, ledger_path=ledger_path)
    print(f"    ok={res['ok']} errors={res['error_count']} warnings={res['warning_count']}")
    if not res["ok"]:
        print(json.dumps(res["errors"], indent=2)[:2000])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
