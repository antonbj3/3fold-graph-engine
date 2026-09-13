#!/usr/bin/env python3
"""End-to-end paper-graph example on a fixed public arXiv set.

Input : a search-substrate graph written by build_paper_graph.py (--search-graph, the
        `<prefix>.search.pkl` export) or a node-link JSON of the same graph.
Steps : overlay knowledge nodes -> run paper_graph_search.search -> evaluate the search
        ranking against the module's own two baselines (citation count, semantic cosine).
Output: <out-prefix>_overlay.json   node-link JSON of the overlaid search substrate
        <out-prefix>_search_eval.json   ranking + baseline comparison + gate verdict

KNOWLEDGE OVERLAY (the consumer side of the drop-in contract in paper_graph_search.py).
The paper side of the substrate carries no knowledge nodes, so this runner derives them
from the corpus itself instead of hand-planting them: one knowledge node per arXiv
category held by at least --min-cat-papers papers (2: the minimum for a two-paper concept signature); its concept signature is the set of
concepts that at least two of that category's papers carry; a knowledge-knowledge edge
joins two categories co-listed on at least two papers. Category pairs with no such edge
are the missing pairs the coherent-triangle signal tries to complete.

GATE (quoted from paper_graph_search.py's own falsifier): "On the REAL paper-graph, if
rho(struct, citation) >= 0.7 OR rho(struct, semantic) >= 0.7, the method REDUCES to that
baseline there." The gate passes when both Spearman correlations stay below 0.7 and the
search still produces a non-degenerate ranking.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from graph_engine.paper_graph import paper_graph_search as pgs  # noqa: E402

RHO_CEILING = 0.7      # the module's own falsifier threshold; not a tolerance of this runner


def load_search_graph(path: str) -> nx.DiGraph:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as fh:
            return json_graph.node_link_graph(json.load(fh), directed=True, edges="links")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def overlay_knowledge(S: nx.DiGraph, min_cat_papers: int = 2, min_concept_papers: int = 2,
                      min_pair_papers: int = 2) -> nx.DiGraph:
    """Add kind='knowledge' nodes (one per sufficiently populated arXiv category) and the
    knowledge-knowledge edges of the current category adjacency. Returns a new graph."""
    G = S.copy()
    by_cat = defaultdict(list)
    for nid, attrs in S.nodes(data=True):
        for cat in attrs.get("categories", []):
            by_cat[cat].append(nid)
    cats = sorted(c for c, papers in by_cat.items() if len(papers) >= min_cat_papers)
    for cat in cats:
        counts = Counter()
        for nid in by_cat[cat]:
            counts.update(set(S.nodes[nid].get("concepts", [])))
        signature = sorted(c for c, n in counts.items() if n >= min_concept_papers)
        G.add_node("K_" + cat, kind="knowledge", concepts=signature, categories=[cat])
    pair_counts = Counter()
    for nid, attrs in S.nodes(data=True):
        present = sorted(set(attrs.get("categories", [])) & set(cats))
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                pair_counts[(present[i], present[j])] += 1
    for (a, b), n in pair_counts.items():
        if n >= min_pair_papers:
            G.add_edge("K_" + a, "K_" + b, kind="knowledge")
            G.add_edge("K_" + b, "K_" + a, kind="knowledge")
    return G


def _rho(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    return float(spearmanr(a[ok], b[ok]).statistic)


def evaluate(G: nx.DiGraph, top_k: int = 10) -> dict:
    rows, sig = pgs.search(G)
    papers = [r["paper"] for r in rows]
    struct = [r["struct"] for r in rows]
    cite = [r["cite"] for r in rows]
    sem = [r["semantic"] for r in rows]
    hole = [r["hole"] for r in rows]

    def top(key):
        return [r["paper"] for r in sorted(rows, key=lambda r: r[key], reverse=True)[:top_k]]

    top_struct, top_cite, top_sem = top("struct"), top("cite"), top("semantic")
    rho_cite = _rho(struct, cite)
    rho_sem = _rho(struct, sem)
    knowledge = [n for n, d in G.nodes(data=True) if d.get("kind") == "knowledge"]
    missing, _ = pgs.knowledge_missing_pairs(G)
    gate_pass = (rho_cite is not None and abs(rho_cite) < RHO_CEILING
                 and rho_sem is not None and abs(rho_sem) < RHO_CEILING
                 and len(set(struct)) > 1)
    return {
        "n_papers": len(papers),
        "n_knowledge_nodes": len(knowledge),
        "knowledge_nodes": sorted(knowledge),
        "n_missing_knowledge_pairs": len(missing),
        "n_edges": G.number_of_edges(),
        "classes": dict(Counter(r["cls"] for r in rows)),
        "rho_struct_citation": rho_cite,
        "abs_rho_struct_citation": None if rho_cite is None else abs(rho_cite),
        "abs_rho_struct_semantic": None if rho_sem is None else abs(rho_sem),
        "rho_struct_semantic": rho_sem,
        "rho_struct_hole_residual": _rho(struct, hole),
        "rho_citation_semantic": _rho(cite, sem),
        "top_k": top_k,
        "top_struct": top_struct,
        "top_citation": top_cite,
        "top_semantic": top_sem,
        "overlap_struct_citation": len(set(top_struct) & set(top_cite)),
        "overlap_struct_semantic": len(set(top_struct) & set(top_sem)),
        "n_papers_with_triangle_witness": sum(1 for r in rows if r["triangle"] > 0),
        "n_isolated": sum(1 for r in rows if r["cls"].startswith("isolated")),
        "gate": {
            "rule": "abs(rho(struct, citation)) < 0.7 and abs(rho(struct, semantic)) < 0.7 "
                    "and the struct ranking is non-degenerate",
            "rho_ceiling": RHO_CEILING,
            "pass": bool(gate_pass),
        },
        "ranking": [
            {k: (None if isinstance(r[k], float) and not np.isfinite(r[k]) else r[k])
             for k in ("paper", "struct", "hole", "triangle", "resolvent", "cutvertex",
                       "cite", "semantic", "cls")}
            for r in rows
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--search-graph", default=os.path.join(HERE, "paper_graph.search.pkl"))
    ap.add_argument("--out-prefix", default=os.path.join(HERE, "paper_graph"))
    ap.add_argument("--min-cat-papers", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args(argv)

    S = load_search_graph(args.search_graph)
    G = overlay_knowledge(S, min_cat_papers=args.min_cat_papers)
    data = json_graph.node_link_data(G, edges="links")
    overlay_path = args.out_prefix + "_overlay.json"
    with open(overlay_path, "w") as fh:
        json.dump(data, fh, indent=1, default=list)
    result = evaluate(G, top_k=args.top_k)
    eval_path = args.out_prefix + "_search_eval.json"
    with open(eval_path, "w") as fh:
        json.dump(result, fh, indent=1)

    print("papers=%d knowledge=%d edges=%d missing-pairs=%d"
          % (result["n_papers"], result["n_knowledge_nodes"], result["n_edges"],
             result["n_missing_knowledge_pairs"]))
    print("classes:", result["classes"])
    print("rho(struct, citation) = %s   rho(struct, semantic) = %s   (ceiling %.1f)"
          % (result["rho_struct_citation"], result["rho_struct_semantic"], RHO_CEILING))
    print("top-%d overlap: struct/citation=%d struct/semantic=%d"
          % (result["top_k"], result["overlap_struct_citation"], result["overlap_struct_semantic"]))
    print("GATE:", "PASS" if result["gate"]["pass"] else "FAIL")
    print("wrote", overlay_path, "and", eval_path)
    return 0 if result["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
