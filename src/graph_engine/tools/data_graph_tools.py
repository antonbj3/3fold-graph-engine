#!/usr/bin/env python3
"""Data-side graph tools — the SOURCE-SIDE value-router (dual of anchor_graph_tools).

Fiber-link framing (operator): max throughput needs a ROUTER at BOTH ends +
a closed feedback loop, not just a gate in the middle:
  SOURCE:  pre-detect (features) -> value-router (demand-ordered selection)
  RECEIVER: cert-gate (admission) -> channel-router (demux to corpus channels)
  LOOP:    receiver outcomes (corpus_coverage, admit/abstain) feed BACK to source demand.

This module is the source-side value-router. It reads DATA_GRAPH.json (sources + their
channel-fill features from cheap pre-detection) and computes INGEST ORDER:

  value(S) = sum_ch [ demand(ch) * fill(S,ch) ] * novelty(S) / cost(S)

where demand(ch) = under-coverage of that channel in the corpus (the SYNC to the objective:
an under-covered channel is in demand), novelty penalises sources whose dominant channels are
already saturated (diminishing returns), cost = ingest size. This is the OED dual of
anchor_graph priority: priority ranks ACTIONS, this ranks DATA.

The value function is EXPLICIT and honest — features are declared estimates until pre-detection
measures them (predetect field). No hidden magic.
"""
from __future__ import annotations
import json, sys, argparse
import os
from pathlib import Path

DATA_GRAPH = Path(os.environ.get("DATA_GRAPH", "DATA_GRAPH.json"))


def _demand(coverage: dict[str, float], channels: list[str]) -> dict[str, float]:
    """demand(ch) = how under-covered ch is. coverage is a count/score per channel;
    normalise by the max so the most-covered channel -> ~0 demand, an empty channel -> 1."""
    mx = max([coverage.get(c, 0) for c in channels] + [1e-9])
    return {c: 1.0 - (coverage.get(c, 0) / mx) for c in channels}


def _novelty(src: dict, ingested_classes: dict[str, int]) -> float:
    """Diminishing returns: a source whose CLASS is already ingested adds less.
    novelty = 1/(1+n_already_ingested_of_this_class)."""
    n = ingested_classes.get(src.get("class"), 0)
    return 1.0 / (1.0 + n)


def ingest_order(graph: dict) -> list[dict]:
    channels = graph["channels"]
    coverage = graph["corpus_coverage"]
    demand = _demand(coverage, channels)
    ingested_classes: dict[str, int] = {}
    for s in graph["sources"]:
        if s.get("ingested"):
            ingested_classes[s.get("class")] = ingested_classes.get(s.get("class"), 0) + 1
    out = []
    for s in graph["sources"]:
        if s.get("ingested"):
            continue  # only rank what's NOT yet in the corpus
        fill = s.get("fill", {})
        raw = sum(demand[c] * fill.get(c, 0.0) for c in channels)
        nov = _novelty(s, ingested_classes)
        cost = max(s.get("size_gb", 1.0), 0.01)
        value = raw * nov / cost
        # which under-covered channels this source most fills (the "why")
        fills = sorted(((demand[c] * fill.get(c, 0.0), c) for c in channels), reverse=True)
        top = [c for v, c in fills if v > 0.05][:3]
        out.append({
            "id": s["id"], "class": s.get("class"),
            "value_density": round(value, 3), "raw_demand_fill": round(raw, 3),
            "novelty": round(nov, 2), "cost_gb": cost,
            "fills_demand": top, "predetect": s.get("predetect"),
        })
    out.sort(key=lambda r: -r["value_density"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["ingest-order", "demand"], default="ingest-order")
    ap.add_argument("--graph", default=str(DATA_GRAPH))
    args = ap.parse_args()
    g = json.loads(Path(args.graph).read_text())
    if args.command == "demand":
        print(json.dumps(_demand(g["corpus_coverage"], g["channels"]), indent=2, ensure_ascii=False))
        return 0
    print(json.dumps(ingest_order(g), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
