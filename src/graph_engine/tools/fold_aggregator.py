#!/usr/bin/env python3
"""fold_aggregator.py — deterministic contradiction/weakness scanner over FOLD_LEDGER.jsonl.

Aggregating agent results is itself an estimation step with the same failure axes as any other
measurement. This scanner reads structured fields deterministically (code extracts, a human only
judges the flags) instead of trusting a narrative summary.

Checks:
  C1  CONFLICT: same node, incompatible verdicts (MEASURED/CONFIRMED vs REFUTED*) with no
      supersedes chain.
  C2  STALE-CONSUMPTION: a superseded entry whose consumers do not also consume the replacement.
  C3  UNVERIFIABLE: entry without gate_cmd (cannot be re-run -- flag, do not accept).
  C4  SINGLE-SOURCE: node whose entries all share one source (not decorrelated -- needs a second
      stream).
  C5  ORPHAN NODE: node that does not match the map structure (population.*|estimation.*|map.*)
      -- a candidate for a NEW map area, not an error.

Usage:  python3 fold_aggregator.py [--ledger data/FOLD_LEDGER.jsonl]
"""
import json, sys, argparse, collections, re

KNOWN_PREFIXES = ("population.", "estimation.", "map.")
NEG = ("REFUTED", "SUPERSEDED")


def load(path):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [PARSE] rad {i}: {e}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="data/FOLD_LEDGER.jsonl")
    a = ap.parse_args()
    rows = load(a.ledger)
    by_node = collections.defaultdict(list)
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        by_node[r.get("node", "?")].append(r)
    superseded = {r["supersedes"]: r for r in rows if r.get("supersedes")}
    flags = []

    for node, rs in sorted(by_node.items()):
        # C1 conflicting verdicts within node, no supersedes link between them
        pos = [r for r in rs if not any(n in r.get("verdict", "") for n in NEG)]
        neg = [r for r in rs if any(n in r.get("verdict", "") for n in NEG)
               and r.get("verdict") != "SUPERSEDED"]
        for p in pos:
            for n in neg:
                linked = (n.get("supersedes") == p["id"]) or (p.get("supersedes") == n["id"])
                # REFUTED-1D + MEASURED-ladder on same node is a scoped split, only flag if
                # decisive numbers reference the same claimed quantity (crude: same id-prefix)
                if not linked and p["id"].split("-")[0] == n["id"].split("-")[0]:
                    flags.append(("C1-CONFLICT", node, f"{p['id']} vs {n['id']}"))
        # C4 single-source node
        # count DISTINCT STREAMS: split source strings on '+'/',' (first-word parsing
        # collapsed "batch X + coordinator re-run" into a single stream)
        srcs = set()
        for r in rs:
            for part in re.split(r"[+,]", r.get("source", "?")):
                part = part.strip().split()[0] if part.strip() else "?"
                srcs.add(part)
        if len(srcs) <= 1 and len(rs) >= 1:
            flags.append(("C4-SINGLE-SOURCE", node, f"source={next(iter(srcs))} ids={[r['id'] for r in rs]}"))
        # C5 orphan node
        if not node.startswith(KNOWN_PREFIXES):
            flags.append(("C5-NEW-AREA", node, f"ids={[r['id'] for r in rs]} (candidate new map area)"))

    for r in rows:
        # C3 unverifiable
        if not r.get("gate_cmd"):
            flags.append(("C3-UNVERIFIABLE", r.get("node"), r["id"]))
        # C2 stale consumption
        if r["id"] in superseded:
            succ = superseded[r["id"]]
            stale = set(r.get("consumers", [])) - set(succ.get("consumers", []))
            if stale:
                flags.append(("C2-STALE", r.get("node"),
                              f"{r['id']} consumers {sorted(stale)} not on the replacement {succ['id']}"))

    print(f"FOLD_LEDGER: {len(rows)} entries, {len(by_node)} nodes")
    if not flags:
        print("No flags.")
    for kind, node, detail in flags:
        print(f"  [{kind}] {node}: {detail}")
    # exit non-zero on hard flags (C1/C2) so it can gate commits
    hard = [f for f in flags if f[0].startswith(("C1", "C2"))]
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
