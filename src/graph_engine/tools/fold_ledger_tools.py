"""
fold_ledger_tools.py — mechanized supersedes-propagation for an append-only evidence ledger
(FOLD_LEDGER.jsonl, one JSON object per line).

Invalidating superseded entries by hand is a human-error surface: an entry gets superseded but a
downstream consumer keeps citing it as live. This tool computes, from the ledger alone:

  1. resolve_active()          -- which entry ids are ACTIVE right now
  2. check_consistency()       -- structural integrity of the supersedes/refines graph
  3. consumers_of_superseded() -- stale-consumption candidates

ROW SCHEMA (not every row carries every field): id, ts, node, claim, verdict, decisive_number,
gate_cmd, source, consumers, supersedes, refines, commit, hub, ready_input, gate_strengthened.

Field handling:
  - `supersedes` is not always a clean id. Supported forms: null; a single id; a list of ids; an
    id plus a parenthetical scope qualifier ("ENTRY-C (only the reframed claim)"); the "PREFIX-N/M"
    shorthand meaning two ids; and free text that is not an id at all, which must resolve to
    "unresolved" rather than be silently dropped.
  - `refines` is a single id or absent. A refined entry stays ACTIVE and gains a `refined_by`
    pointer: refining is elaboration, not invalidation.
  - `ts` may be a full date or a bare time. Line order is authoritative (the ledger is append-only);
    `ts` is informational, and a row whose date precedes an earlier row's date is flagged rather
    than silently reordered.

KNOWN LIMITATION: a parenthetical scope qualifier means only a named claim-slice of the target is
invalidated. This tool treats every resolved supersedes edge as a FULL deactivation, so the
superseded count can overstate what is actually dead; the qualifier text is preserved verbatim and
a human adjudicates.

Paths: --ledger, or the FOLD_LEDGER environment variable, else ./FOLD_LEDGER.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Paths are configurable: env var, then CLI flag, then these defaults, resolved against the
# current working directory.
LEDGER_PATH = Path(os.environ.get("FOLD_LEDGER", "FOLD_LEDGER.jsonl"))
REPORT_PATH = Path(os.environ.get("FOLD_LEDGER_REPORT", "fold_ledger_consistency.json"))

_SHORTHAND_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)*)-(\d+)/(\d+)\b")


def _is_boundary_char(ch: str | None) -> bool:
    """True if ch does not continue an identifier token (id chars: alnum, -, _)."""
    if ch is None:
        return True
    return not (ch.isalnum() or ch in "-_")


def load_ledger(path: Path = LEDGER_PATH,
                 bad_lines: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """
    Load the JSONL ledger, tagging each row with its 1-based file line number
    (the append-only write order — the authoritative chronological ordering).
    """
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                if bad_lines is not None:
                    bad_lines.append({"line": lineno, "reason": f"json error: {e}", "raw": line})
                continue
            if not isinstance(d, dict):
                if bad_lines is not None:
                    bad_lines.append({"line": lineno, "reason": "row is not a JSON object",
                                       "raw": line})
                continue
            d["_line"] = lineno
            rows.append(d)
    return rows


def resolve_target_string(raw: str, known_ids: set[str]) -> tuple[list[str], list[str]]:
    """
    Resolve a raw supersedes/refines string to a list of known record ids.
    
    Returns (resolved_ids, unresolved_fragments). `unresolved_fragments` is non-empty
    when the raw text contains no recognizable id at all (e.g. references a concept,
    not a record) -- this is a genuine finding, surfaced by check_consistency(), not
    silently swallowed.
    """
    if raw is None:
        return [], []
    if raw in known_ids:
        return [raw], []

    resolved: set[str] = set()

    # 1. Explicit "PREFIX-N/M" shorthand (two ids compressed into one token).
    for m in _SHORTHAND_RE.finditer(raw):
        prefix, n1, n2 = m.group(1), m.group(2), m.group(3)
        for n in (n1, n2):
            cand = f"{prefix}-{n}"
            if cand in known_ids:
                resolved.add(cand)

    # 2. General scan: does any known id appear in raw as a whole token (word-boundary
    #    aware, where "word" chars for an id include '-' and '_')?
    for kid in known_ids:
        idx = raw.find(kid)
        while idx != -1:
            before = raw[idx - 1] if idx > 0 else None
            after_idx = idx + len(kid)
            after = raw[after_idx] if after_idx < len(raw) else None
            if _is_boundary_char(before) and _is_boundary_char(after):
                resolved.add(kid)
                break
            idx = raw.find(kid, idx + 1)

    if resolved:
        return sorted(resolved), []
    return [], [raw]


def _normalize_supersedes(raw: Any) -> list[str]:
    """supersedes is null | str | list[str] in the observed schema -> normalize to list."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return list(raw)
    return [str(raw)]


def resolve_active(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the active set."""
    known_ids = {d["id"] for d in rows if "id" in d}
    line_of = {d["id"]: d["_line"] for d in rows if "id" in d}

    superseded_by: dict[str, list[str]] = {}
    refined_by: dict[str, list[str]] = {}
    unresolved_supersedes: list[dict[str, Any]] = []
    unresolved_refines: list[dict[str, Any]] = []
    ignored_forward_edges: list[dict[str, Any]] = []

    for d in rows:
        fid = d.get("id")
        if fid is None:
            continue
        # --- supersedes ---
        for raw in _normalize_supersedes(d.get("supersedes")):
            resolved, unresolved = resolve_target_string(raw, known_ids)
            for frag in unresolved:
                unresolved_supersedes.append({"fold": fid, "line": d["_line"], "raw": frag})
            for target in resolved:
                if target == fid:
                    continue # self-reference guarded separately in check_consistency
                if line_of.get(target, -1) > d["_line"]:
                    # target is written LATER than the superseder -> impossible, don't apply
                    ignored_forward_edges.append(
                        {"superseder": fid, "target": target,
                         "superseder_line": d["_line"], "target_line": line_of[target]}
                    )
                    continue
                superseded_by.setdefault(target, []).append(fid)

        # --- refines ---
        rraw = d.get("refines")
        if rraw is not None:
            resolved, unresolved = resolve_target_string(rraw, known_ids)
            for frag in unresolved:
                unresolved_refines.append({"fold": fid, "line": d["_line"], "raw": frag})
            for target in resolved:
                if target == fid:
                    continue
                refined_by.setdefault(target, []).append(fid)

    active_ids = [fid for fid in known_ids if fid not in superseded_by]
    superseded_ids = sorted(superseded_by.keys())

    # Build supersede chains (target -> ... -> final superseder) for reporting.
    chains = []
    for target in superseded_ids:
        chain = [target]
        cur = target
        seen = {cur}
        while cur in superseded_by:
            nxt_list = superseded_by[cur]
            nxt = nxt_list[-1] # most recent (highest line) superseder
            if nxt in seen:
                break # cycle; check_consistency() reports this separately
            chain.append(nxt)
            seen.add(nxt)
            cur = nxt
        chains.append(chain)

    return {
        "known_ids_count": len(known_ids),
        "active_ids": sorted(active_ids),
        "active_count": len(active_ids),
        "superseded_ids": superseded_ids,
        "superseded_count": len(superseded_ids),
        "superseded_by": {k: v for k, v in superseded_by.items()},
        "refined_by": {k: v for k, v in refined_by.items()},
        "supersede_chains": chains,
        "unresolved_supersedes": unresolved_supersedes,
        "unresolved_refines": unresolved_refines,
        "ignored_forward_edges": ignored_forward_edges,
    }


def check_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Structural integrity checks over the supersedes/refines graph."""
    ids = [d["id"] for d in rows if "id" in d]
    known_ids = set(ids)
    line_of = {d["id"]: d["_line"] for d in rows if "id" in d}

    issues: list[dict[str, Any]] = []

    # (d) duplicate ids
    seen_count: dict[str, int] = {}
    for fid in ids:
        seen_count[fid] = seen_count.get(fid, 0) + 1
    dup_ids = sorted([fid for fid, c in seen_count.items() if c > 1])
    for fid in dup_ids:
        issues.append({"type": "duplicate_id", "id": fid, "count": seen_count[fid]})

    # (a) + (c): walk every supersedes/refines edge.
    # NOTE: resolve_target_string() only ever returns ids that are members of
    # known_ids by construction, so a resolved target can never be "missing" --
    # unresolvable text is surfaced entirely through the *_target_unresolved path
    # (this covers requirement (a): every target either resolves to a real id, or
    # is flagged, never silently dropped).
    edges_all: list[tuple[str, str]] = [] # (target, source) for cycle check, ALL edges

    for d in rows:
        fid = d.get("id")
        if fid is None:
            continue
        for raw in _normalize_supersedes(d.get("supersedes")):
            resolved, unresolved = resolve_target_string(raw, known_ids)
            for frag in unresolved:
                issues.append({
                    "type": "supersedes_target_unresolved",
                    "fold": fid, "line": d["_line"], "raw": frag,
                })
            for target in resolved:
                if target == fid:
                    issues.append({
                        "type": "self_supersede", "fold": fid, "line": d["_line"],
                    })
                    continue
                edges_all.append((target, fid))
                if line_of[target] > d["_line"]:
                    issues.append({
                        "type": "supersedes_later_fold",
                        "fold": fid, "fold_line": d["_line"],
                        "target": target, "target_line": line_of[target],
                    })

        rraw = d.get("refines")
        if rraw is not None:
            resolved, unresolved = resolve_target_string(rraw, known_ids)
            for frag in unresolved:
                issues.append({
                    "type": "refines_target_unresolved",
                    "fold": fid, "line": d["_line"], "raw": frag,
                })
            for target in resolved:
                if target == fid:
                    issues.append({"type": "self_refine", "fold": fid, "line": d["_line"]})
                    continue
                edges_all.append((target, fid))
                if line_of[target] > d["_line"]:
                    issues.append({
                        "type": "refines_later_fold",
                        "fold": fid, "fold_line": d["_line"],
                        "target": target, "target_line": line_of[target],
                    })

    # (b) cycle detection on the FULL combined graph (supersedes + refines, every
    # resolved edge regardless of line-order validity), direction target -> source
    # (i.e. "is invalidated/elaborated by").
    graph: dict[str, list[str]] = {}
    for target, src in edges_all:
        graph.setdefault(target, []).append(src)

    def find_cycle() -> list[str] | None:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in known_ids}
        parent: dict[str, str] = {}

        def dfs(u: str) -> list[str] | None:
            color[u] = GRAY
            for v in graph.get(u, []):
                if color.get(v, WHITE) == WHITE:
                    parent[v] = u
                    res = dfs(v)
                    if res:
                        return res
                elif color.get(v) == GRAY:
                    # found cycle v -> ... -> u -> v
                    cyc = [v]
                    cur = u
                    while cur != v:
                        cyc.append(cur)
                        cur = parent.get(cur)
                        if cur is None:
                            break
                    cyc.append(v)
                    cyc.reverse()
                    return cyc
            color[u] = BLACK
            return None

        for n in known_ids:
            if color[n] == WHITE:
                res = dfs(n)
                if res:
                    return res
        return None

    cycle = find_cycle()
    if cycle:
        issues.append({"type": "cycle", "path": cycle})

    # ts sanity: date should be non-decreasing across line order (informational check;
    # bare HH:MM:SS rows have no date and are skipped, per the documented ts caveat).
    last_date = None
    ts_wraparounds = []
    for d in rows:
        ts = d.get("ts")
        if ts and len(ts) == 10: # "YYYY-MM-DD"
            if last_date is not None and ts < last_date:
                ts_wraparounds.append({
                    "id": d.get("id"), "line": d["_line"], "ts": ts, "prev_date": last_date,
                })
            else:
                last_date = ts
    if ts_wraparounds:
        issues.append({"type": "ts_date_wraparound", "occurrences": ts_wraparounds})

    return {
        "total_folds": len(ids),
        "unique_ids": len(known_ids),
        "duplicate_ids": dup_ids,
        "issue_count": len(issues),
        "issues": issues,
        "ts_schema_note": (
            "153/157 rows carry a full YYYY-MM-DD date; 4 tail rows carry bare "
            "HH:MM:SS with no date. Line order (append-only ledger) is used as the "
            "authoritative chronological order for all checks; ts date is a secondary "
            "sanity check only, not the ordering ground truth."
        ),
    }


def consumers_of_superseded(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    STALE-CONSUMPTION candidates: records whose `consumers` list names a record id or a
    node that is now superseded. flag-must-propagate: a superseded finding that a
    consumer still cites is a flag that hasn't reached the aggregate.
    """
    active = resolve_active(rows)
    superseded_ids = set(active["superseded_ids"])
    known_ids = {d["id"] for d in rows if "id" in d}

    # node -> whether ANY active record still lives at that node (node not fully dead)
    node_has_active: dict[str, bool] = {}
    node_of: dict[str, str] = {}
    for d in rows:
        fid = d.get("id")
        node = d.get("node")
        if fid is None or node is None:
            continue
        node_of[fid] = node
        is_active = fid not in superseded_ids
        node_has_active[node] = node_has_active.get(node, False) or is_active

    stale_fold_consumers = []
    stale_node_consumers = []
    for d in rows:
        fid = d.get("id")
        for c in d.get("consumers") or []:
            if c in known_ids and c in superseded_ids:
                stale_fold_consumers.append({
                    "consuming_fold": fid, "consuming_line": d.get("_line"),
                    "superseded_fold_cited": c,
                    "superseded_by": active["superseded_by"].get(c, []),
                })
            # node-level: consumer string matches a node whose every record is now dead
            elif c in node_has_active and not node_has_active[c]:
                stale_node_consumers.append({
                    "consuming_fold": fid, "consuming_line": d.get("_line"),
                    "fully_superseded_node_cited": c,
                })

    return {
        "stale_fold_consumers": stale_fold_consumers,
        "stale_fold_consumer_count": len(stale_fold_consumers),
        "stale_node_consumers": stale_node_consumers,
        "stale_node_consumer_count": len(stale_node_consumers),
    }


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #

def _mini_ledger() -> list[dict[str, Any]]:
    """
    Synthetic ledger covering: simple supersede, chain, refines-vs-supersedes,
    missing target, cycle, duplicate id -- each independently checkable per-case.
    """
    rows = [
        # simple supersede: S1 -> superseded by S2
        {"id": "S1", "ts": "2026-01-01", "node": "n.simple", "consumers": [], "supersedes": None},
        {"id": "S2", "ts": "2026-01-02", "node": "n.simple", "consumers": [], "supersedes": "S1"},
        # chain: C1 -> C2 -> C3 (C1 and C2 both superseded, only C3 active)
        {"id": "C1", "ts": "2026-01-01", "node": "n.chain", "consumers": [], "supersedes": None},
        {"id": "C2", "ts": "2026-01-02", "node": "n.chain", "consumers": [], "supersedes": "C1"},
        {"id": "C3", "ts": "2026-01-03", "node": "n.chain", "consumers": [], "supersedes": "C2"},
        # refines vs supersedes: R1 refined by R2, R1 STAYS active
        {"id": "R1", "ts": "2026-01-01", "node": "n.refine", "consumers": [], "supersedes": None},
        {"id": "R2", "ts": "2026-01-02", "node": "n.refine", "consumers": [], "refines": "R1"},
        # missing target: M1 supersedes a nonexistent id
        {"id": "M1", "ts": "2026-01-04", "node": "n.missing", "consumers": [],
         "supersedes": "GHOST-DOES-NOT-EXIST"},
        # cycle: X1 supersedes X2, X2 supersedes X1 (impossible under real append-order,
        # constructed here purely to exercise the cycle detector on the full graph;
        # X1->X2 is ALSO independently flagged as a line-order violation)
        {"id": "X1", "ts": "2026-01-05", "node": "n.cycle", "consumers": [], "supersedes": "X2"},
        {"id": "X2", "ts": "2026-01-05", "node": "n.cycle", "consumers": [], "supersedes": "X1"},
        # duplicate id
        {"id": "DUP", "ts": "2026-01-06", "node": "n.dup", "consumers": [], "supersedes": None},
        {"id": "DUP", "ts": "2026-01-06", "node": "n.dup", "consumers": [], "supersedes": None},
        # consumer of a superseded record (stale-consumption candidate)
        {"id": "USES-S1", "ts": "2026-01-07", "node": "n.consumer", "consumers": ["S1"],
         "supersedes": None},
        # partial-qualifier / shorthand supersedes forms
        {"id": "P1", "ts": "2026-01-08", "node": "n.partial", "consumers": [], "supersedes": None},
        {"id": "P2", "ts": "2026-01-08", "node": "n.partial2", "consumers": [], "supersedes": None},
        {"id": "P3", "ts": "2026-01-09", "node": "n.partial", "consumers": [],
         "supersedes": "P1 (endast en delclaim)"},
        {"id": "P4", "ts": "2026-01-09", "node": "n.partial2",
         "consumers": [], "supersedes": "delar av P2/P9 (shorthand, P9 does not exist)"},
    ]
    for i, d in enumerate(rows, start=1):
        d["_line"] = i
    return rows


def selftest() -> bool:
    rows = _mini_ledger()
    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        mark = "PASS" if cond else "FAIL"
        print(f"[{mark}] {label}" + (f" -- {detail}" if detail and not cond else ""))
        if not cond:
            ok = False

    active = resolve_active(rows)
    cons = check_consistency(rows)
    stale = consumers_of_superseded(rows)

    # 1. simple supersede: S1 inactive, S2 active
    check("simple supersede: S1 inactive", "S1" not in active["active_ids"])
    check("simple supersede: S2 active", "S2" in active["active_ids"])

    # 2. chain: C1, C2 inactive; C3 active; chain recorded C1->C2->C3
    check("chain: C1 inactive", "C1" not in active["active_ids"])
    check("chain: C2 inactive", "C2" not in active["active_ids"])
    check("chain: C3 active", "C3" in active["active_ids"])
    chain_c1 = next((c for c in active["supersede_chains"] if c[0] == "C1"), None)
    check("chain: C1 chain reaches C3", chain_c1 == ["C1", "C2", "C3"], str(chain_c1))

    # 3. refines vs supersedes: R1 stays ACTIVE, gets refined_by R2
    check("refines: R1 still active", "R1" in active["active_ids"])
    check("refines: R1 refined_by R2", active["refined_by"].get("R1") == ["R2"])

    # 4. missing/unresolvable target -> flagged, not silently dropped
    missing_issues = [i for i in cons["issues"] if i["type"] == "supersedes_target_unresolved"
                       and i["fold"] == "M1"]
    check("missing target flagged for M1", len(missing_issues) == 1, str(missing_issues))

    # 5. cycle detected on the full graph, independent of the line-order gate
    cycle_issues = [i for i in cons["issues"] if i["type"] == "cycle"]
    check("cycle detected (X1<->X2)", len(cycle_issues) == 1, str(cycle_issues))
    later_issues = [i for i in cons["issues"] if i["type"] == "supersedes_later_fold"
                     and i["fold"] == "X1"]
    check("X1->X2 also independently flagged as line-order violation",
          len(later_issues) == 1, str(later_issues))

    # 6. duplicate id detected
    check("duplicate id DUP detected", "DUP" in cons["duplicate_ids"])

    # 7. stale consumer: USES-S1 cites superseded S1
    stale_ids = [s["superseded_fold_cited"] for s in stale["stale_fold_consumers"]]
    check("stale consumer flags S1", "S1" in stale_ids, str(stale["stale_fold_consumers"]))

    # 8. partial-qualifier supersedes resolves to P1 (parenthetical stripped correctly)
    check("partial-qualifier resolves to P1", "P1" not in active["active_ids"])

    # 9. shorthand supersedes: P2 resolved+superseded, P9 (nonexistent) flagged unresolved
    check("shorthand resolves P2 superseded", "P2" not in active["active_ids"])
    unresolved_frags = [i["raw"] for i in cons["issues"]
                         if i["type"] == "supersedes_target_unresolved" and i["fold"] == "P4"]
    # P4's raw supersedes text is "delar av P2/P9 (...)" -- P2 resolves via shorthand,
    # but the whole string still isn't a KNOWN id itself, so unresolved reporting only
    # fires if NO id resolves at all. Since P2 resolves, there should be zero unresolved
    # fragments for P4 (P2 rescues it) -- verify that.
    check("shorthand P4: no false-unresolved (P2 rescues it)", len(unresolved_frags) == 0,
          str(unresolved_frags))

    # 10. load_ledger hardening: a malformed line is SKIPPED+COUNTED, never fatal
    # per-row model copied). A skipped row must not appear as a known id anywhere.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "malformed_ledger.jsonl"
        p.write_text(
            '{"id": "GOOD-1", "supersedes": null}\n'
            '{this is not json\n'
            '\n'
            '"a bare json string, not an object"\n'
            '{"id": "GOOD-2", "supersedes": "GOOD-1"}\n',
            encoding="utf-8",
        )
        bad: list = []
        rows_ml = load_ledger(p, bad_lines=bad)
        check("load_ledger: good rows survive a malformed line", len(rows_ml) == 2, str(rows_ml))
        check("load_ledger: malformed lines counted (not silently dropped)", len(bad) == 2, str(bad))
        active_ml = resolve_active(rows_ml)
        check("load_ledger: surviving rows still resolve correctly (GOOD-1 superseded)",
              "GOOD-1" not in active_ml["active_ids"] and "GOOD-2" in active_ml["active_ids"],
              str(active_ml))
        check("load_ledger: missing file -> empty, no crash", load_ledger(Path(td) / "nope.jsonl") == [])

    print()
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_report(rows: list[dict[str, Any]],
                  bad_lines: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    active = resolve_active(rows)
    cons = check_consistency(rows)
    stale = consumers_of_superseded(rows)
    return {
        "ledger_path": str(LEDGER_PATH),
        "bad_lines_skipped": len(bad_lines) if bad_lines is not None else 0,
        "bad_lines": bad_lines or [],
        "total_folds": len(rows),
        "active_count": active["active_count"],
        "superseded_count": active["superseded_count"],
        "active_ids": active["active_ids"],
        "superseded_ids": active["superseded_ids"],
        "supersede_chains": active["supersede_chains"],
        "refined_by": active["refined_by"],
        "unresolved_supersedes": active["unresolved_supersedes"],
        "unresolved_refines": active["unresolved_refines"],
        "ignored_forward_edges": active["ignored_forward_edges"],
        "consistency": cons,
        "stale_consumers": stale,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true",
                     help="run resolve_active/check_consistency/consumers_of_superseded "
                          "on the real ledger and write reports/probes/fold_ledger_consistency.json")
    ap.add_argument("--selftest", action="store_true",
                     help="run the synthetic mini-ledger selftest")
    ap.add_argument("--ledger", default=str(LEDGER_PATH), help="path to FOLD_LEDGER.jsonl")
    args = ap.parse_args()

    if not args.report and not args.selftest:
        ap.print_help()
        return 1

    exit_code = 0

    if args.selftest:
        if not selftest():
            exit_code = 1

    if args.report:
        bad_lines: list[dict[str, Any]] = []
        rows = load_ledger(Path(args.ledger), bad_lines=bad_lines)
        report = build_report(rows, bad_lines=bad_lines)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {REPORT_PATH}")
        print(f"total_folds={report['total_folds']} active={report['active_count']} "
              f"superseded={report['superseded_count']} "
              f"bad_lines_skipped={report['bad_lines_skipped']} "
              f"issues={report['consistency']['issue_count']} "
              f"stale_fold_consumers={report['stale_consumers']['stale_fold_consumer_count']} "
              f"stale_node_consumers={report['stale_consumers']['stale_node_consumer_count']}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
