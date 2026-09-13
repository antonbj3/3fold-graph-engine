#!/usr/bin/env python3
"""
anchor_graph_tools.py — query/maintenance tools for data/ANCHOR_GRAPH.json,
the MAINTAINED dependency graph of the locked main line (replaces the throwaway
view in docs/RECON_ANCHOR_BUILD_ORDER.md).

⚠ NODE IDs ARE NOT LANE LETTERS: ids like "F"/"H" name BUILDS (read the node's
`claim` field), not agent pool lanes — a agent got confused here.
Non-coordinator lanes: this graph ranks work, it does NOT assign you coordinator duties
(role resolution = OPTIMAL_START.md §0).

Commands:
  validate        schema check + every fold-id in `evidence` exists in
                  data/FOLD_LEDGER.jsonl + every reports/probes/*.json evidence
                  path exists on disk + no dangling depends_on + DAG check +
                  stored `load` matches recomputed load. A docs/*.md evidence
                  entry is accepted as the weaker "doc-pointer" class (file-
                  existence checked, but reported separately in
                  `doc_pointer_evidence`, not promoted to fold/probe trust).
                  WARNINGS (not errors): PROVEN node depending on a non-PROVEN
                  node (that is how the graph names a hole, e.g. B -> GT-GEOM-
                  ANCHOR); PROVEN/REFUTED node with empty evidence IS an error.
  next-actions    rank OPEN/ASSUMED nodes by load x risk (LOW=1 MED=2 HIGH=3).
                  A node with a closed regime_gate is still ranked (its load x
                  risk is real) but its ACTION is rewritten to "open the regime
                  gate: <blocked_by>" — never "measure the node now" (the
                  GOAL-COND-DENSIFY wrong-regime lesson, mechanized).
  refresh-status  re-derive per-node evidence health from the live ledger via
                  fold_ledger_tools.resolve_active: a node citing a SUPERSEDED
                  fold is flagged STALE (with the superseding fold named); a
                  node citing an unknown fold id is an error. Report-only —
                  the coordinator adjudicates and edits the graph file.
  --selftest      tiny synthetic graph + synthetic ledger covering each check.

Design: status is stored (coordinator-adjudicated), never auto-rewritten; load is
stored for human reading but the validator recomputes it and errors on drift.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = REPO_ROOT / "data" / "ANCHOR_GRAPH.json"
LEDGER_PATH = REPO_ROOT / "data" / "FOLD_LEDGER.jsonl"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold_ledger_tools  # noqa: E402  (resolve_active / load_ledger)

VALID_TYPES = {"THEORY", "EMPIRICAL", "ENGINEERING", "DATA-ANCHOR", "GOAL"}
# GOAL: a top-level stretch objective (large explicit `cost`, HIGH risk, no upward dependents by
# construction) -- distinct from older "*-GOAL"-id nodes that predate this convention and stay
# type=EMPIRICAL (the id suffix is NOT the type signal, see the module docstring warning about ids).
# Multi-use was confirmed before extending the enum; a single-use deviant type was normalized to an
# existing type instead of extended here.
VALID_STATUS = {"PROVEN", "REFUTED", "DEFERRED", "OPEN", "ASSUMED"}
# SOURCE-COND: a node whose value/risk is a FUNCTION of the ingested source
# distribution, not a fixed scalar (e.g. audio adds geometry signal for reverb-rich indoor
# sources, ~zero for silent aerial). Cannot be ranked globally — its action is a cheap
# de-risk probe on a representative source first. Rank 0 so it never masquerades as a
# globally-ordered pick.
RISK_RANK = {"LOW": 1, "MED": 2, "HIGH": 3, "SOURCE-COND": 0}
REQUIRED_FIELDS = ["id", "claim", "type", "status", "evidence", "depends_on",
                   "load", "risk", "regime_note"]


def load_graph(path: Path = GRAPH_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ledger_index(ledger_path: Path) -> tuple[set[str], set[str], dict[str, list[str]]]:
    """(known fold ids, superseded fold ids, superseded_by map) from the live ledger."""
    rows = fold_ledger_tools.load_ledger(ledger_path)
    active = fold_ledger_tools.resolve_active(rows)
    known = {d["id"] for d in rows if "id" in d}
    return known, set(active["superseded_ids"]), active["superseded_by"]


def _computed_loads(nodes: list[dict[str, Any]]) -> dict[str, int]:
    loads = {n["id"]: 0 for n in nodes}
    for n in nodes:
        for dep in n.get("depends_on", []):
            if dep in loads:
                loads[dep] += 1
    return loads


def _find_cycle(nodes: list[dict[str, Any]]) -> list[str] | None:
    graph = {n["id"]: [d for d in n.get("depends_on", [])] for n in nodes}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {nid: WHITE for nid in graph}
    parent: dict[str, str] = {}

    def dfs(u: str) -> list[str] | None:
        color[u] = GRAY
        for v in graph.get(u, []):
            if v not in color:
                continue  # dangling dep -> separate error, not a cycle
            if color[v] == WHITE:
                parent[v] = u
                res = dfs(v)
                if res:
                    return res
            elif color[v] == GRAY:
                cyc, cur = [v], u
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

    for nid in graph:
        if color[nid] == WHITE:
            res = dfs(nid)
            if res:
                return res
    return None


def validate(graph: dict[str, Any],
             ledger_path: Path = LEDGER_PATH,
             repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    ids = [n.get("id") for n in nodes]
    id_set = set(ids)
    if len(ids) != len(id_set):
        dups = sorted({i for i in ids if ids.count(i) > 1})
        errors.append({"type": "duplicate_node_id", "ids": dups})

    known_folds, superseded, superseded_by = _ledger_index(ledger_path)

    status_of = {n.get("id"): n.get("status") for n in nodes}
    evidence_ids_checked = 0
    doc_pointer_evidence: list[dict[str, Any]] = []

    for n in nodes:
        nid = n.get("id", "<missing-id>")
        # schema
        for field in REQUIRED_FIELDS:
            if field not in n:
                errors.append({"type": "missing_field", "node": nid, "field": field})
        if n.get("type") not in VALID_TYPES:
            errors.append({"type": "bad_type", "node": nid, "value": n.get("type")})
        if n.get("status") not in VALID_STATUS:
            errors.append({"type": "bad_status", "node": nid, "value": n.get("status")})
        if n.get("risk") not in RISK_RANK:
            errors.append({"type": "bad_risk", "node": nid, "value": n.get("risk")})
        rg = n.get("regime_gate")
        if rg is not None and (not isinstance(rg, dict) or "open" not in rg
                               or "blocked_by" not in rg):
            errors.append({"type": "bad_regime_gate", "node": nid, "value": rg})

        # evidence wiring: each entry is a fold-id in the ledger OR an existing
        # reports/probes/*.json path (relative to repo root) OR a docs/*.md path — the latter is
        # the weaker "doc-pointer" class (added, graph_cleanup_v0): a narrative verdict
        # doc, not a machine-checkable probe artifact. Accepted (file-existence checked) but
        # tracked separately in `doc_pointer_evidence` below rather than silently promoted to the
        # same trust tier as a fold-id or a reports/probes/ JSON — NOT emitted as a warning
        # (warning_count stays a hard gate on real problems).
        for ev in n.get("evidence", []):
            if ev in known_folds:
                evidence_ids_checked += 1
                if ev in superseded:
                    warnings.append({
                        "type": "evidence_fold_superseded", "node": nid, "fold": ev,
                        "superseded_by": superseded_by.get(ev, []),
                        "note": "node status is STALE-candidate; run refresh-status",
                    })
            elif ev.startswith("reports/probes/") and ev.endswith(".json"):
                if not (repo_root / ev).exists():
                    errors.append({"type": "evidence_path_missing", "node": nid, "path": ev})
                else:
                    evidence_ids_checked += 1
            elif ev.startswith("docs/") and ev.endswith(".md"):
                if not (repo_root / ev).exists():
                    errors.append({"type": "evidence_path_missing", "node": nid, "path": ev})
                else:
                    evidence_ids_checked += 1
                    doc_pointer_evidence.append({"node": nid, "path": ev, "class": "doc-pointer"})
            else:
                errors.append({"type": "evidence_id_unknown", "node": nid, "evidence": ev})

        # PROVEN/REFUTED must be wired to something
        if n.get("status") in ("PROVEN", "REFUTED") and not n.get("evidence"):
            errors.append({"type": "adjudicated_status_without_evidence", "node": nid,
                           "status": n.get("status")})

        # depends_on
        for dep in n.get("depends_on", []):
            if dep == nid:
                errors.append({"type": "self_dependency", "node": nid})
            elif dep not in id_set:
                errors.append({"type": "dangling_depends_on", "node": nid, "dep": dep})
            elif n.get("status") == "PROVEN" and status_of.get(dep) != "PROVEN":
                warnings.append({
                    "type": "proven_node_on_unproven_dep", "node": nid, "dep": dep,
                    "dep_status": status_of.get(dep),
                    "note": "named hole: the PROVEN claim holds only in the regime "
                            "stated in its regime_note",
                })

    # DAG
    cycle = _find_cycle(nodes)
    if cycle:
        errors.append({"type": "cycle", "path": cycle})

    # Routing guards: priority is only
    # trustworthy if (a) the epoch weighting exists and points at real GOAL nodes and
    # (b) every weighted goal is wired into the dependency web so frontier work can
    # actually reach it. These fire in the SAME validate every fold already runs —
    # a weaker/amnesic driver inherits the guard mechanically.
    goal_ids = {n["id"] for n in nodes if n.get("type") == "GOAL"}
    stored_w = (graph.get("_meta") or {}).get("goal_weights") or {}
    if not stored_w:
        warnings.append({"type": "goal_weights_missing",
                         "note": "_meta.goal_weights unset — priority() runs unweighted; "
                                 "coordinator: set epoch weights per design goal"})
    for k in stored_w:
        if k not in goal_ids:
            errors.append({"type": "goal_weight_unknown_id", "id": k,
                           "note": "weight points at a non-GOAL/absent node — routing lies"})
    for n in nodes:
        if n.get("type") == "GOAL" and n.get("status") in ("OPEN", "ASSUMED"):
            w = float(stored_w.get(n["id"], 1.0))
            if w >= 1.0 and not n.get("depends_on"):
                warnings.append({"type": "goal_unwired", "node": n["id"], "weight": w,
                                 "note": "active goal with weight>=1 has no depends_on edges — "
                                         "no frontier work can route toward it; densify or down-weight"})

    # load drift
    computed = _computed_loads(nodes)
    for n in nodes:
        nid = n.get("id")
        if nid in computed and n.get("load") != computed[nid]:
            errors.append({"type": "load_mismatch", "node": nid,
                           "stored": n.get("load"), "computed": computed[nid]})

    return {
        "ok": not errors,
        "node_count": len(nodes),
        "evidence_refs_verified": evidence_ids_checked,
        "doc_pointer_evidence_count": len(doc_pointer_evidence),
        "doc_pointer_evidence": doc_pointer_evidence,
        "computed_loads": computed,
        "error_count": len(errors),
        "errors": errors,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def next_actions(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Rank OPEN/ASSUMED nodes. TRUTHFULNESS FIXES:
    the old `score = load*risk` lied on three counts — (1) it ignored READINESS, so it told you to
    "measure now" a node whose deps aren't built (it can't be measured); (2) risk-as-multiplier
    rewarded speculation, burying cheap ready wins under unbuilt HIGH-risk nodes; (3) no cost term,
    so it was never σ_min-OED (value/cost). Fix: a node is ACTIONABLE only if not regime-gated AND
    all deps PROVEN; non-ready nodes are surfaced as BLOCKED (build deps first), never "measure now".
    Actionable nodes sort first, then by load (dependents unblocked) then risk. `score` retained for
    continuity but is advisory, not authoritative — readiness is the real gate."""
    nodes = graph.get("nodes", [])
    computed = _computed_loads(nodes)
    status_of = {n["id"]: n.get("status") for n in nodes}
    out = []
    for n in nodes:
        if n.get("status") not in ("OPEN", "ASSUMED"):
            continue
        load = computed.get(n["id"], 0)
        risk = RISK_RANK.get(n.get("risk"), 0)
        rg = n.get("regime_gate")
        gated = bool(rg) and not rg.get("open", False)
        unmet = [d for d in n.get("depends_on", []) if status_of.get(d) != "PROVEN"]
        actionable = (not gated) and (not unmet)
        if gated:
            action = f"OPEN THE REGIME GATE first: {rg.get('blocked_by')}"
        elif unmet:
            action = f"BLOCKED — build deps first: {', '.join(unmet)}"
        elif n.get("risk") == "SOURCE-COND":
            action = ("value is SOURCE-CONDITIONAL — not globally rankable; run a cheap "
                      "de-risk probe on a representative source before committing")
        elif n.get("type") == "THEORY":
            action = "derive on paper (cheap, fail-fast) — then adjudicate status"
        elif n.get("type") == "DATA-ANCHOR":
            action = "acquire/build the anchor data, then wire it as evidence"
        else:
            action = "measure in the regime stated in regime_note; fold the result"
        out.append({
            "id": n["id"], "status": n["status"], "type": n["type"],
            "load": load, "risk": n.get("risk"), "score": load * risk,
            "actionable": actionable, "blocked_by": unmet or None,
            "regime_gated": gated, "action": action,
            "claim": n.get("claim", ""),
        })
    # ACTIONABLE first (readiness is the real gate); then load, then risk, then id.
    out.sort(key=lambda r: (not r["actionable"], -r["load"],
                            -RISK_RANK.get(r["risk"], 0), r["id"]))
    return out


# Coarse effort/cost proxy — HONEST: this is NOT real ktok, it's a type/risk heuristic so the
# ranking has a denominator at all (σ_min-OED is value/cost; a value-only rank is not OED). Override
# per-node with an explicit integer `cost` field when a real estimate exists.
COST_BY_TYPE = {"THEORY": 1, "ENGINEERING": 2, "DATA-ANCHOR": 3, "EMPIRICAL": 3}


def _node_cost(n: dict[str, Any]) -> float:
    c = n.get("cost")
    if isinstance(c, (int, float)) and c > 0:
        return float(c)
    base = COST_BY_TYPE.get(n.get("type"), 2)
    if n.get("risk") == "SOURCE-COND":
        base += 1  # you must pay a de-risk probe before the value is even known
    return float(base)


def _upward_reach(nid: str, rev: dict[str, set]) -> set:
    """Transitive set of nodes that (directly or indirectly) depend on nid."""
    seen, stack = set(), list(rev.get(nid, ()))
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack.extend(rev.get(x, ()))
    return seen


def priority(graph: dict[str, Any], goal_weights: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Compute prioritisation = value / cost among ACTIONABLE nodes (σ_min-OED shape). This is the
    thing next-actions could NOT do (operator-flagged: 'shouldn't the tool compute
    prioritisation too?'). value = weighted count of GOAL nodes this node transitively unblocks
    (upward reach) + a small credit for intermediate unblocks; cost = _node_cost. The ONE genuinely
    external input is goal_weights (which north star matters more) — everything else is computed from
    graph structure. Default = equal weight; pass weights to see the ranking shift."""
    nodes = graph.get("nodes", [])
    rev: dict[str, set] = {}
    for n in nodes:
        for d in n.get("depends_on", []):
            rev.setdefault(d, set()).add(n["id"])
    # Goals by TYPE, not id suffix: suffix matching silently dropped goal nodes whose id did not
    # end in "-GOAL", so priority routed toward the only goals it could see.
    goals = [n["id"] for n in nodes if n.get("type") == "GOAL"]
    # Epoch goal-weights live IN the graph (_meta.goal_weights) so every caller —
    # cron, coordinator, cold session — gets the same routing without remembering a flag.
    # An explicit goal_weights argument still overrides for what-if runs.
    stored = (graph.get("_meta") or {}).get("goal_weights") or {}
    gw = goal_weights or {**{g: 1.0 for g in goals}, **{k: float(v) for k, v in stored.items() if k in set(goals)}}
    act = {r["id"]: r for r in next_actions(graph)}
    out = []
    for n in nodes:
        if n.get("status") not in ("OPEN", "ASSUMED"):
            continue
        reach = _upward_reach(n["id"], rev)
        goals_reached = [g for g in goals if g in reach]
        value = sum(gw.get(g, 1.0) for g in goals_reached) + 0.1 * len(reach)
        cost = _node_cost(n)
        actionable = act.get(n["id"], {}).get("actionable", False)
        out.append({
            "id": n["id"], "actionable": actionable,
            "goals_reached": goals_reached, "reach": len(reach),
            "value": round(value, 2), "cost": cost,
            "priority": round(value / cost, 3) if actionable else 0.0,
            "action": act.get(n["id"], {}).get("action", ""),
        })
    out.sort(key=lambda r: (not r["actionable"], -r["priority"], r["id"]))
    return out


def refresh_status(graph: dict[str, Any],
                   ledger_path: Path = LEDGER_PATH) -> dict[str, Any]:
    """Re-derive evidence health per node from the live ledger. Report-only."""
    known_folds, superseded, superseded_by = _ledger_index(ledger_path)
    flags = []
    for n in graph.get("nodes", []):
        stale_folds = []
        unknown = []
        for ev in n.get("evidence", []):
            if ev.startswith("reports/probes/"):
                continue
            if ev not in known_folds:
                unknown.append(ev)
            elif ev in superseded:
                stale_folds.append({"fold": ev, "superseded_by": superseded_by.get(ev, [])})
        if stale_folds or unknown:
            flags.append({
                "node": n["id"], "stored_status": n.get("status"),
                "derived_flag": "STALE" if stale_folds else "EVIDENCE_UNKNOWN",
                "stale_evidence": stale_folds, "unknown_evidence": unknown,
                "action": "coordinator: re-adjudicate node against the superseding fold(s), "
                          "then update evidence + status in ANCHOR_GRAPH.json",
            })
    return {"nodes_checked": len(graph.get("nodes", [])),
            "flagged_count": len(flags), "flags": flags}


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #

def selftest() -> bool:
    import tempfile
    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not cond else ""))
        if not cond:
            ok = False

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # synthetic ledger: F-OLD superseded by F-NEW; F-LIVE active
        ledger = tdp / "ledger.jsonl"
        ledger.write_text(
            '{"id": "F-OLD", "supersedes": null}\n'
            '{"id": "F-NEW", "supersedes": "F-OLD"}\n'
            '{"id": "F-LIVE", "supersedes": null}\n', encoding="utf-8")
        probe = tdp / "reports" / "probes" / "p.json"
        probe.parent.mkdir(parents=True)
        probe.write_text("{}\n", encoding="utf-8")

        def node(nid, status="OPEN", ev=None, deps=None, load=0, risk="MED", **kw):
            d = {"id": nid, "claim": "c", "type": "EMPIRICAL", "status": status,
                 "evidence": ev or [], "depends_on": deps or [], "load": load,
                 "risk": risk, "regime_note": "n/a"}
            d.update(kw)
            return d

        # 1. clean graph validates, loads computed
        g = {"nodes": [
            node("N1", status="PROVEN", ev=["F-NEW"], risk="LOW"),
            node("N2", deps=["N1"], load=0, risk="HIGH"),
            node("N3", status="ASSUMED", deps=["N1"], load=0, risk="LOW",
                 type="THEORY"),
        ]}
        g["nodes"][0]["load"] = 2
        v = validate(g, ledger_path=ledger, repo_root=tdp)
        check("clean graph: ok", v["ok"], json.dumps(v["errors"]))
        check("clean graph: load computed N1=2", v["computed_loads"]["N1"] == 2)

        # 2. dangling depends_on caught
        g2 = {"nodes": [node("N1", deps=["GHOST"])]}
        v2 = validate(g2, ledger_path=ledger, repo_root=tdp)
        check("dangling dep caught", any(e["type"] == "dangling_depends_on" for e in v2["errors"]))

        # 3. cycle caught
        g3 = {"nodes": [node("N1", deps=["N2"], load=1), node("N2", deps=["N1"], load=1)]}
        v3 = validate(g3, ledger_path=ledger, repo_root=tdp)
        check("cycle caught", any(e["type"] == "cycle" for e in v3["errors"]))

        # 4. unknown evidence id caught; valid probe path accepted
        g4 = {"nodes": [node("N1", ev=["NO-SUCH-FOLD", "reports/probes/p.json"])]}
        v4 = validate(g4, ledger_path=ledger, repo_root=tdp)
        check("unknown evidence id caught",
              any(e["type"] == "evidence_id_unknown" and e["evidence"] == "NO-SUCH-FOLD"
                  for e in v4["errors"]))
        check("probe path accepted", not any(e.get("path") == "reports/probes/p.json"
                                             for e in v4["errors"]))

        # 5. superseded evidence -> validate warning + refresh-status STALE flag
        g5 = {"nodes": [node("N1", status="PROVEN", ev=["F-OLD"])]}
        v5 = validate(g5, ledger_path=ledger, repo_root=tdp)
        check("superseded evidence warned",
              any(w["type"] == "evidence_fold_superseded" for w in v5["warnings"]))
        r5 = refresh_status(g5, ledger_path=ledger)
        check("refresh-status flags STALE",
              r5["flagged_count"] == 1 and r5["flags"][0]["derived_flag"] == "STALE"
              and r5["flags"][0]["stale_evidence"][0]["superseded_by"] == ["F-NEW"],
              json.dumps(r5))

        # 6. PROVEN without evidence = error; PROVEN on unproven dep = warning only
        g6 = {"nodes": [node("N1", status="PROVEN"),
                        node("N2", status="PROVEN", ev=["F-LIVE"], deps=["N3"]),
                        node("N3", status="OPEN", load=1)]}
        v6 = validate(g6, ledger_path=ledger, repo_root=tdp)
        check("PROVEN without evidence is error",
              any(e["type"] == "adjudicated_status_without_evidence" for e in v6["errors"]))
        check("PROVEN-on-unproven-dep is warning not error",
              any(w["type"] == "proven_node_on_unproven_dep" for w in v6["warnings"])
              and not any(e.get("type") == "proven_node_on_unproven_dep" for e in v6["errors"]))

        # 7. next_actions: ranking by load x risk; closed gate rewrites action;
        # PROVEN node excluded
        g7 = {"nodes": [
            node("BASE", status="PROVEN", ev=["F-LIVE"], load=3, risk="LOW"),
            node("HI", status="OPEN", deps=["BASE"], risk="HIGH"),          # score 0*3=0
            node("MID", status="OPEN", deps=["BASE"], risk="MED"),          # 0
            node("LOADED", status="ASSUMED", risk="HIGH"),                  # depended on below
            node("USER1", status="OPEN", deps=["BASE", "LOADED"], risk="LOW"),
            node("GATED", status="OPEN", deps=["LOADED"], risk="HIGH",
                 regime_gate={"open": False, "blocked_by": "regime X not reached"}),
        ]}
        # loads: BASE=3, LOADED=2 -> LOADED score 2*3=6 top; GATED 0*3=0 but gated
        na = next_actions(g7)
        check("next_actions excludes PROVEN", all(r["id"] != "BASE" for r in na))
        check("next_actions top = LOADED (load x risk)", na[0]["id"] == "LOADED",
              json.dumps([r["id"] for r in na]))
        gated_row = next(r for r in na if r["id"] == "GATED")
        check("closed gate rewrites action",
              gated_row["regime_gated"] and gated_row["action"].startswith("OPEN THE REGIME GATE"))

        # 8. load drift caught
        g8 = {"nodes": [node("N1", load=5), node("N2", deps=["N1"])]}
        v8 = validate(g8, ledger_path=ledger, repo_root=tdp)
        check("load drift caught", any(e["type"] == "load_mismatch" and e["stored"] == 5
                                       and e["computed"] == 1 for e in v8["errors"]))

    print()
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?",
                    choices=["validate", "next-actions", "refresh-status", "priority"])
    ap.add_argument("--graph", default=str(GRAPH_PATH))
    ap.add_argument("--ledger", default=str(LEDGER_PATH))
    ap.add_argument("--weights", default=None,
                    help="goal weights as JSON, e.g. '{\"ANCHOR-GOAL\":2,\"CORPUS-GOAL\":1}'")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return 0 if selftest() else 1
    if not args.command:
        ap.print_help()
        return 1

    graph = load_graph(Path(args.graph))
    if args.command == "validate":
        res = validate(graph, ledger_path=Path(args.ledger))
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if res["ok"] else 1
    if args.command == "next-actions":
        print(json.dumps(next_actions(graph), indent=2, ensure_ascii=False))
        return 0
    if args.command == "refresh-status":
        res = refresh_status(graph, ledger_path=Path(args.ledger))
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if res["flagged_count"] == 0 else 2
    if args.command == "priority":
        gw = json.loads(args.weights) if args.weights else None
        print(json.dumps(priority(graph, goal_weights=gw), indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
