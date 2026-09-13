#!/usr/bin/env python3
"""
build_paper_graph.py
====================

Build a PAPER-GRAPH from a paper corpus for the paper-search engine.

This graph is the *substrate* that the discovery-graph search algorithm will later be
repointed onto, so the node / edge API is kept deliberately clean and self-documenting.

Corpus in  ->  paper_graph.{pkl,graphml,_stats.json}  out.

--------------------------------------------------------------------------------
GRAPH REPRESENTATION
--------------------------------------------------------------------------------
The primary deliverable is a single **undirected `networkx.Graph`** (the "search
substrate").  Two edge families are *merged* onto it with a `kind` tag:

  (a) CITATION edges   -- from reference IDs (undirected here, direction kept in
                          the `cites` edge attr + a companion DiGraph).
  (b) SIMILARITY edges -- TF-IDF cosine over (title + abstract), thresholded so the
                          graph stays sparse (default cosine > 0.15).

Why undirected + merged, with direction preserved separately:
  * Reachability, connected-components, weighted betweenness, clustering / triangle
    counts and community structure -- everything the discovery-graph search needs --
    are naturally defined on ONE undirected weighted graph.
  * Citation direction still matters for provenance, so it is NOT thrown away: it is
    stored per-edge (`cites`) and as a standalone `citation_digraph` in the bundle.
  * A single graph with a `kind`/`weight`/`cosine` per edge lets the downstream
    consumer re-weight the two edge families however it likes (or filter to one).

--------------------------------------------------------------------------------
NODE / EDGE / BUNDLE API  (contract for the downstream search consumer)
--------------------------------------------------------------------------------
NODE key      = paper id (str).  G.nodes[id] attributes:
    kind: 'paper'        (search-consumer contract; consumer adds 'knowledge')
    title: str
    abstract: str
    year: int | None
    authors: list[str]
    categories: list[str]
    concepts: list[str]      top TF-IDF terms U categories -- the concept signature
                                      the discovery-graph search overlaps on (required by it)
    citation_count: int
    tfidf_row: int            row index into bundle['tfidf_matrix']
    tfidf: scipy.sparse (1 x V) csr row  -- the node's TF-IDF feature vec
    n_refs_internal: int            # references that resolved to a node in-graph
    n_refs_dangling: int            # references pointing outside the corpus

EDGE (u, v) attributes  (undirected G):
    kind: 'citation' | 'similarity' | 'both'
    weight: float in (0, 1]   relatedness (HIGHER = more related)
    cosine: float             TF-IDF cosine (0.0 if pure citation)
    distance: float >= eps      = 1 - weight   (use this for shortest-path / betweenness)
    cites: list[(citing_id, cited_id)]   directed citation(s) on this edge ([] if none)

BUNDLE  (pickle, papers/paper_graph.pkl)  -- dict:
    'graph': nx.Graph        the undirected search substrate (primary object)
    'citation_digraph': nx.DiGraph      directed citation edges only
    'tfidf_matrix': scipy.sparse    (n_nodes x V), row i <-> node_order[i]
    'vectorizer': fitted TfidfVectorizer  (embed NEW query text with this)
    'node_order': list[str]       row-index -> node id
    'feature_names': list[str]       tf-idf vocabulary
    'meta': dict            params + build stats + timestamps

Convenience consumer entry points (import from this module):
    bundle = load_paper_graph('papers/paper_graph.pkl')
    hits   = query_similar(bundle, "certified mesh error bounds", top_k=5)
    G      = bundle['graph']

SEARCH-CONSUMER EXPORT  (papers/paper_graph.search.pkl)
    A *bare* nx.DiGraph (NOT the bundle) that satisfies the drop-in contract declared by
    paper_graph_search.py: nodes carry kind='paper' + concepts (required) + citation_count;
    edges are paper->paper citations tagged kind='cite'.  The discovery-graph search loads
    it directly with its own load_graph and overlays its 'knowledge' nodes on top.
    (The rich bundle above is for text search / analysis; this lean graph is for that search.)

--------------------------------------------------------------------------------
DROP-IN FOR THE REAL CORPUS
--------------------------------------------------------------------------------
`--corpus papers/corpus.jsonl` is the default.  If that file does not exist yet the
script falls back to `papers/corpus.mock.jsonl`, generating it from the embedded
MOCK_CORPUS if needed.  The loader normalises many field-name variants
(references/refs/citations, citation_count/n_citations/citationCount,...), so when
the real corpus lands it is picked up automatically with no code change.

Run:
    python build_paper_graph.py                      # auto: real corpus else mock
    python build_paper_graph.py --corpus X.jsonl --threshold 0.15 --out papers/paper_graph
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import sys
import time
from collections import Counter

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel  # == cosine for L2-normalised tf-idf

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = os.path.join(HERE, "corpus.jsonl")
MOCK_CORPUS_PATH = os.path.join(HERE, "corpus.mock.jsonl")
DEFAULT_OUT = os.path.join(HERE, "paper_graph")

SIM_THRESHOLD = 0.15   # cosine above which a similarity edge is added (keeps graph sparse)
CITATION_WEIGHT = 0.90  # relatedness weight assigned to an explicit citation edge
EPS = 1e-6             # floor on edge distance so path algos never see a 0/neg length
BETWEENNESS_SAMPLE = 100  # if n_nodes > this, betweenness is estimated on a k-sample


# --------------------------------------------------------------------------- #
# Embedded mock corpus (16 realistic domain papers, 4 clusters)
# --------------------------------------------------------------------------- #
MOCK_CORPUS = [
    {"id": "P001", "year": 2018, "citation_count": 142,
     "title": "Automatic Hex-Dominant Mesh Generation from CAD B-Rep Models",
     "abstract": "We present an automatic method for generating hex-dominant meshes directly from CAD boundary representation (B-Rep) models. The approach extracts the underlying solid topology and produces high-quality hexahedral and prismatic elements suitable for finite element analysis. Experiments on industrial CAD parts demonstrate robust meshing with minimal user intervention.",
     "authors": ["A. Lindqvist", "M. Rossi"], "references": [],
     "categories": ["mesh-generation", "CAD"]},
    {"id": "P002", "year": 2020, "citation_count": 78,
     "title": "Feature-Preserving Simplification of CAD Assemblies for Simulation",
     "abstract": "Large CAD assemblies contain fine geometric detail that is unnecessary for downstream simulation. We propose a feature-preserving simplification algorithm that removes small fillets, holes, and chamfers from B-Rep models while preserving simulation-relevant geometry. The simplified models reduce mesh element count and finite element solve time without significant loss of accuracy.",
     "authors": ["J. Chen", "S. Patel"], "references": ["P001"],
     "categories": ["CAD", "geometry-processing"]},
    {"id": "P003", "year": 2019, "citation_count": 96,
     "title": "Robust Tetrahedral Meshing of Dirty CAD Geometry",
     "abstract": "CAD models exported for simulation frequently contain gaps, overlaps, and non-manifold edges that defeat conventional meshers. We describe a robust tetrahedral meshing pipeline that tolerates such defects by combining constrained Delaunay refinement with local geometry healing. The method produces watertight volume meshes from dirty CAD input across a benchmark of mechanical parts.",
     "authors": ["R. Kumar", "L. Berg"], "references": ["P001"],
     "categories": ["mesh-generation", "CAD"]},
    {"id": "P004", "year": 2021, "citation_count": 54,
     "title": "Signed Distance Fields for Watertight Repair of CAD Meshes",
     "abstract": "We use signed distance fields (SDF) to repair non-watertight CAD surface meshes prior to simulation. By sampling an implicit signed distance representation and re-extracting the zero level set, the method closes gaps and removes self-intersections, yielding watertight geometry. We quantify the reconstruction error against the original B-Rep and show it stays within mesh resolution.",
     "authors": ["T. Nakamura", "E. Olsson"], "references": ["P003", "P001"],
     "categories": ["geometry-processing", "SDF"]},
    {"id": "P005", "year": 2019, "citation_count": 210,
     "title": "Digital Twins for Predictive Maintenance of Industrial Assets",
     "abstract": "Digital twins couple physics-based simulation models with live sensor data to predict the remaining useful life of industrial equipment. This paper reviews digital twin architectures for predictive maintenance and presents a case study on rotating machinery. We discuss data assimilation, model updating, and the role of high-fidelity simulation in twin accuracy.",
     "authors": ["M. Rossi", "K. Andersson"], "references": [],
     "categories": ["digital-twin", "simulation"]},
    {"id": "P006", "year": 2021, "citation_count": 63,
     "title": "A CAD-to-Simulation Pipeline for Automated Finite Element Analysis",
     "abstract": "We describe an end-to-end pipeline that converts CAD models into simulation-ready finite element analyses with minimal manual setup. The pipeline performs geometry simplification, automatic meshing, boundary condition assignment, and solver configuration. Integrated with a digital twin, it enables rapid re-analysis as the design or operating conditions change.",
     "authors": ["S. Patel", "A. Lindqvist", "K. Andersson"], "references": ["P002", "P005", "P003"],
     "categories": ["digital-twin", "CAD", "FEM"]},
    {"id": "P007", "year": 2022, "citation_count": 41,
     "title": "Real-Time Digital Twin Synchronization with Sensor Data Assimilation",
     "abstract": "Keeping a digital twin synchronized with its physical counterpart requires continuous assimilation of sensor measurements. We present a Kalman-filter-based data assimilation scheme that updates twin state and model parameters in real time. The approach improves prediction accuracy on a thermal monitoring benchmark compared to open-loop simulation.",
     "authors": ["K. Andersson", "H. Wang"], "references": ["P005"],
     "categories": ["digital-twin", "data-assimilation"]},
    {"id": "P008", "year": 2023, "citation_count": 19,
     "title": "USD-Based Scene Description for Simulation-Ready Digital Twins",
     "abstract": "Universal Scene Description (USD) provides a composable format for representing complex assemblies. We propose a USD-based schema for simulation-ready digital twins that captures geometry, materials, sensors, and boundary conditions in a single interoperable scene graph. The schema streamlines exchange between CAD, meshing, and physics-simulation tools.",
     "authors": ["E. Olsson", "H. Wang"], "references": ["P006", "P005"],
     "categories": ["digital-twin", "simulation", "USD"]},
    {"id": "P009", "year": 2017, "citation_count": 187,
     "title": "Uncertainty Quantification in Finite Element Simulation",
     "abstract": "Finite element predictions are affected by uncertainty in material properties, boundary conditions, and geometry. This paper surveys uncertainty quantification methods for finite element simulation, including Monte Carlo, stochastic Galerkin, and polynomial chaos expansions. We compare their cost and accuracy on structural mechanics benchmarks.",
     "authors": ["D. Fischer", "N. Gupta"], "references": [],
     "categories": ["UQ", "FEM"]},
    {"id": "P010", "year": 2019, "citation_count": 88,
     "title": "A Posteriori Error Estimation for Certified Finite Element Solutions",
     "abstract": "We derive a posteriori error estimators that bound the discretization error of finite element solutions, enabling certified predictions with guaranteed accuracy. The estimators drive adaptive mesh refinement to meet a user-specified error tolerance. Numerical experiments demonstrate reliable and efficient error control on elliptic problems.",
     "authors": ["N. Gupta", "P. Moreau"], "references": ["P009"],
     "categories": ["UQ", "verification", "FEM"]},
    {"id": "P011", "year": 2018, "citation_count": 74,
     "title": "Grid Convergence Index for Solution Verification in CFD",
     "abstract": "The grid convergence index (GCI) provides a standardized measure of discretization uncertainty for computational fluid dynamics. We review the GCI methodology, discuss its assumptions on asymptotic convergence, and apply it to verify solutions on a series of systematically refined meshes. The study highlights common pitfalls in solution verification.",
     "authors": ["P. Moreau", "D. Fischer"], "references": ["P009"],
     "categories": ["verification", "CFD"]},
    {"id": "P012", "year": 2021, "citation_count": 57,
     "title": "Certified Reduced-Order Models with Guaranteed Error Bounds",
     "abstract": "Reduced-order models accelerate simulation but risk uncontrolled error. We construct certified reduced-order models equipped with rigorous a posteriori error bounds, so that every reduced prediction carries a guaranteed accuracy certificate. The reduced basis is built greedily to minimize the worst-case error over a parameter range.",
     "authors": ["P. Moreau", "N. Gupta"], "references": ["P010", "P009"],
     "categories": ["UQ", "reduced-order", "verification"]},
    {"id": "P013", "year": 2020, "citation_count": 305,
     "title": "Graph Neural Networks for Mesh-Based Physics Simulation",
     "abstract": "We learn mesh-based physics simulators using graph neural networks that operate directly on the simulation mesh. Message passing over mesh edges predicts the dynamics of deformable solids and fluids, generalizing across mesh resolutions and geometries. The learned simulator runs orders of magnitude faster than a classical finite element solver.",
     "authors": ["H. Wang", "Y. Zhao"], "references": ["P009"],
     "categories": ["machine-learning", "GNN", "simulation"]},
    {"id": "P014", "year": 2019, "citation_count": 61,
     "title": "TF-IDF and Citation Graphs for Scientific Literature Retrieval",
     "abstract": "We combine TF-IDF text similarity with citation-graph structure to improve retrieval of scientific literature. Documents are embedded as sparse term-frequency vectors and linked by both cosine similarity and citation edges, forming a hybrid graph. Ranking over this graph outperforms text-only baselines on a literature-search benchmark.",
     "authors": ["Y. Zhao", "J. Chen"], "references": [],
     "categories": ["information-retrieval", "graph"]},
    {"id": "P015", "year": 2022, "citation_count": 44,
     "title": "Neural Surrogate Models for Accelerated Finite Element Analysis",
     "abstract": "Neural surrogate models approximate expensive finite element analyses at a fraction of the cost. We train surrogates on a dataset of parameterized CAD-to-simulation runs and show accurate prediction of stress and displacement fields. Coupled with a CAD-to-simulation pipeline, the surrogate enables interactive design exploration.",
     "authors": ["Y. Zhao", "S. Patel"], "references": ["P013", "P009", "P006"],
     "categories": ["machine-learning", "surrogate", "FEM"]},
    {"id": "P016", "year": 2023, "citation_count": 22,
     "title": "Physics-Informed Neural Networks for Certified Simulation Surrogates",
     "abstract": "Physics-informed neural networks (PINNs) embed governing equations into the training loss, but their predictions typically lack error guarantees. We augment PINN surrogates with a posteriori error estimation to produce certified simulation surrogates whose outputs carry validated error bounds. The approach bridges learned surrogates and rigorous solution verification.",
     "authors": ["N. Gupta", "H. Wang"], "references": ["P015", "P013", "P012"],
     "categories": ["machine-learning", "PINN", "verification"]},
]


# --------------------------------------------------------------------------- #
# Corpus loading + field normalisation (robust to real-corpus field names)
# --------------------------------------------------------------------------- #
def _first(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return default


def _as_id(x) -> str | None:
    """Canonicalise a paper id. Folds the scheme prefix to lowercase so that a node id
    'arxiv:1603.06046' and a reference id 'ARXIV:1603.06046' resolve to the SAME node --
    without this, every in-corpus citation in the real arXiv corpus silently dangles
    (measured: recovers 230 internal citation links / 131 papers on the live corpus)."""
    if x is None:
        return None
    if isinstance(x, dict):
        return _as_id(_first(x, ("id", "paper_id", "paperId", "corpusId", "doi")))
    s = str(x).strip()
    if not s:
        return None
    if ":" in s:
        scheme, rest = s.split(":", 1)
        if scheme.isalpha():        # fold real schemes (arxiv/doi/...), not numeric DOIs
            s = scheme.lower() + ":" + rest.strip()
    return s


def _as_year(x) -> int | None:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    s = str(x)
    for i in range(len(s) - 3):
        chunk = s[i:i + 4]
        if chunk.isdigit() and 1500 <= int(chunk) <= 2100:
            return int(chunk)
    return None


def _as_str_list(x) -> list[str]:
    if not x:
        return []
    if isinstance(x, str):
        # allow ';' or ',' delimited category strings
        return [t.strip() for t in x.replace(";", ",").split(",") if t.strip()]
    out = []
    for item in x:
        if isinstance(item, dict):
            v = _first(item, ("name", "id", "label"))
            if v:
                out.append(str(v))
        elif item is not None:
            out.append(str(item))
    return out


def normalize_record(raw: dict, index: int) -> dict:
    """Canonicalise one raw corpus record into the internal schema."""
    pid = _as_id(_first(raw, ("id", "paper_id", "paperId", "corpusId", "doi")))
    if not pid:
        pid = f"n{index}"
    refs_raw = _first(raw, ("references", "refs", "citations", "referenced_ids",
                            "outbound_citations", "reference_ids"), []) or []
    refs = [r for r in (_as_id(r) for r in refs_raw) if r]
    return {
        "id": pid,
        "title": str(_first(raw, ("title", "name"), "") or ""),
        "abstract": str(_first(raw, ("abstract", "summary", "text"), "") or ""),
        "authors": _as_str_list(_first(raw, ("authors", "author", "author_names"), [])),
        "year": _as_year(_first(raw, ("year", "published_year", "pub_year", "date", "publication_date"))),
        "references": refs,
        "citation_count": int(_first(raw, ("citation_count", "n_citations", "cited_by_count",
                                           "citationCount", "num_citations", "citations_count"), 0) or 0),
        "categories": _as_str_list(_first(raw, ("categories", "category", "fields_of_study",
                                                "fieldsOfStudy", "topics", "tags", "subjects"), [])),
    }


def write_mock_corpus(path: str = MOCK_CORPUS_PATH) -> str:
    with open(path, "w") as f:
        for rec in MOCK_CORPUS:
            f.write(json.dumps(rec) + "\n")
    return path


def resolve_corpus(path: str) -> str:
    """Pick the corpus to use: real if present, else mock (generating it if needed)."""
    if os.path.exists(path):
        print(f"[corpus] using real corpus: {path}")
        return path
    print(f"[corpus] real corpus not found at {path}")
    if not os.path.exists(MOCK_CORPUS_PATH):
        write_mock_corpus()
        print(f"[corpus] generated mock corpus: {MOCK_CORPUS_PATH}")
    print(f"[corpus] falling back to mock corpus: {MOCK_CORPUS_PATH}")
    return MOCK_CORPUS_PATH


def load_corpus(path: str) -> list[dict]:
    papers, seen = [], set()
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = normalize_record(json.loads(line), i)
            if rec["id"] in seen:            # de-dup on id (keep first)
                continue
            seen.add(rec["id"])
            papers.append(rec)
    return papers


# --------------------------------------------------------------------------- #
# TF-IDF
# --------------------------------------------------------------------------- #
def build_tfidf(papers: list[dict], min_df=None, ngram_max=2, max_df=0.9):
    """TF-IDF over (title + abstract).

    min_df=2 is the important knob: dropping hapax terms (esp. one-off bigrams)
    keeps the vocabulary on genuinely SHARED vocabulary, which is what makes cosine
    discriminating. On the mock corpus it collapses vocab 950 -> 105 and separates
    intra-cluster cosine (median ~0.24) from inter-cluster (~0.07); with min_df=1 the
    hapax bigrams inflate dimensionality and push every off-diagonal cosine < 0.18.
    Default adapts to corpus size and falls back to min_df=1 if the vocab collapses.
    """
    docs = [f"{p['title']}. {p['abstract']}".strip() for p in papers]
    if min_df is None:
        min_df = 2 if len(papers) >= 5 else 1

    def _fit(mdf):
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, ngram_max),   # bigrams: "finite element", "digital twin"
            min_df=mdf,
            max_df=max_df,
            sublinear_tf=True,            # log-scaled term frequency
            norm="l2",                    # so linear_kernel == cosine
        )
        return vec, vec.fit_transform(docs)

    try:
        vectorizer, matrix = _fit(min_df)
        if matrix.shape[1] == 0:
            raise ValueError("empty vocabulary")
    except ValueError:
        vectorizer, matrix = _fit(1)      # degenerate/tiny corpus fallback
    return matrix, vectorizer


def top_terms(tfidf_row, feature_names, k=12):
    """The k highest-weight TF-IDF terms for one paper -> its 'concept' signature.
    These are the discriminating, mostly-shared tokens the search consumer overlaps on."""
    coo = tfidf_row.tocoo()
    top = sorted(zip(coo.col, coo.data), key=lambda t: -t[1])[:k]
    return [feature_names[c] for c, _ in top]


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def build_graph(papers, tfidf_matrix, vectorizer,
                sim_threshold=SIM_THRESHOLD, citation_weight=CITATION_WEIGHT):
    G = nx.Graph()
    Gcit = nx.DiGraph()
    node_order = [p["id"] for p in papers]
    id_set = set(node_order)
    feature_names = list(vectorizer.get_feature_names_out())

    # -- nodes + features -----------------------------------------------------
    for row, p in enumerate(papers):
        tfidf_row = tfidf_matrix.getrow(row)
        # concept signature = top TF-IDF terms U categories (for the search consumer)
        concepts = list(dict.fromkeys(
            top_terms(tfidf_row, feature_names) + list(p["categories"])))
        attrs = {
            "kind": "paper",                     # search-consumer contract (paper|knowledge)
            "title": p["title"],
            "abstract": p["abstract"],
            "year": p["year"],
            "authors": p["authors"],
            "categories": p["categories"],
            "concepts": concepts,                # search-consumer contract (required)
            "citation_count": p["citation_count"],
            "tfidf_row": row,
            "tfidf": tfidf_row,                  # 1 x V sparse feature vector
            "n_refs_internal": 0,
            "n_refs_dangling": 0,
        }
        G.add_node(p["id"], **attrs)
        Gcit.add_node(p["id"])

    # -- citation edges -------------------------------------------------------
    n_cit_links = n_dangling = 0
    for p in papers:
        src = p["id"]
        for dst in p["references"]:
            if dst in id_set and dst != src:
                n_cit_links += 1
                G.nodes[src]["n_refs_internal"] += 1
                Gcit.add_edge(src, dst)          # directed: src cites dst
                if G.has_edge(src, dst):
                    e = G[src][dst]
                    if e["kind"] == "similarity":
                        e["kind"] = "both"
                    e["weight"] = max(e["weight"], citation_weight)
                    e.setdefault("cites", []).append((src, dst))
                else:
                    G.add_edge(src, dst, kind="citation", cosine=0.0,
                               weight=citation_weight, cites=[(src, dst)])
            else:
                n_dangling += 1
                G.nodes[src]["n_refs_dangling"] += 1

    # -- similarity edges (TF-IDF cosine > threshold) -------------------------
    # linear_kernel == cosine because rows are L2-normalised. Dense n x n is fine
    # for a few hundred nodes; see module docstring for the >~2k scaling path.
    sims = linear_kernel(tfidf_matrix, tfidf_matrix)
    n = len(node_order)
    n_sim_edges = 0
    for i in range(n):
        u = node_order[i]
        row = sims[i]
        for j in range(i + 1, n):
            c = float(row[j])
            if c > sim_threshold:
                v = node_order[j]
                if G.has_edge(u, v):             # citation edge already there -> "both"
                    e = G[u][v]
                    e["cosine"] = c
                    e["kind"] = "both"
                    e["weight"] = max(e["weight"], c)
                else:
                    G.add_edge(u, v, kind="similarity", cosine=c,
                               weight=c, cites=[])
                n_sim_edges += 1

    # -- derived edge fields (distance for path/centrality algorithms) --------
    for _u, _v, e in G.edges(data=True):
        e.setdefault("cites", [])
        e["distance"] = max(EPS, 1.0 - float(e["weight"]))

    meta = {
        "sim_threshold": sim_threshold,
        "citation_weight": citation_weight,
        "n_citation_links": n_cit_links,
        "n_dangling_refs": n_dangling,
        "n_similarity_edges": n_sim_edges,
    }
    return G, Gcit, node_order, meta


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def compute_stats(G, Gcit, meta):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [d for _, d in G.degree()]

    kinds = Counter(e["kind"] for _, _, e in G.edges(data=True))
    comps = list(nx.connected_components(G))
    comp_sizes = sorted((len(c) for c in comps), reverse=True)
    singletons = sum(1 for s in comp_sizes if s == 1)

    deg_cent = nx.degree_centrality(G) if n > 1 else {}

    # betweenness: exact for small graphs, k-sampled estimate for large ones
    if n > BETWEENNESS_SAMPLE:
        k = BETWEENNESS_SAMPLE
        btw = nx.betweenness_centrality(G, k=k, weight="distance", seed=42, normalized=True)
        btw_mode = f"k-sample (k={k} of {n})"
    else:
        k = None
        btw = nx.betweenness_centrality(G, weight="distance", normalized=True) if n > 2 else {}
        btw_mode = "exact (all nodes)"

    def _top(cent, topn=5):
        return [(nid, round(float(v), 4), G.nodes[nid].get("title", "")[:60])
                for nid, v in sorted(cent.items(), key=lambda kv: kv[1], reverse=True)[:topn]]

    tri_total = sum(nx.triangles(G).values()) // 3 if n else 0

    stats = {
        "n_nodes": n,
        "n_edges": m,
        "edges_by_kind": dict(kinds),
        "n_citation_links_directed": meta["n_citation_links"],
        "n_dangling_refs": meta["n_dangling_refs"],
        "density": round(nx.density(G), 4) if n > 1 else 0.0,
        "degree": {
            "min": min(degrees) if degrees else 0,
            "max": max(degrees) if degrees else 0,
            "mean": round(statistics.mean(degrees), 3) if degrees else 0.0,
            "median": statistics.median(degrees) if degrees else 0,
            "histogram": dict(sorted(Counter(degrees).items())),
        },
        "connected_components": {
            "count": len(comps),
            "largest_size": comp_sizes[0] if comp_sizes else 0,
            "sizes": comp_sizes,
            "singletons": singletons,
        },
        "avg_clustering": round(nx.average_clustering(G), 4) if n > 2 else 0.0,
        "triangles": tri_total,
        "betweenness_mode": btw_mode,
        "top_degree_centrality": _top(deg_cent),
        "top_betweenness": _top(btw),
        "citation_digraph": {
            "n_edges": Gcit.number_of_edges(),
            "n_weakly_connected": nx.number_weakly_connected_components(Gcit),
            "is_dag": nx.is_directed_acyclic_graph(Gcit),
            "top_cited_in_corpus": sorted(
                ((nid, Gcit.in_degree(nid), G.nodes[nid].get("title", "")[:60])
                 for nid in Gcit.nodes()),
                key=lambda t: t[1], reverse=True)[:5],
        },
    }
    return stats


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_bundle(G, Gcit, tfidf_matrix, vectorizer, node_order, meta, out_prefix):
    bundle = {
        "graph": G,
        "citation_digraph": Gcit,
        "tfidf_matrix": tfidf_matrix,
        "vectorizer": vectorizer,
        "node_order": node_order,
        "feature_names": list(vectorizer.get_feature_names_out()),
        "meta": {**meta, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()},
    }
    pkl_path = out_prefix + ".pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_path


def save_graphml(G, out_prefix):
    """Portable structural export (Gephi/inspection). GraphML only holds scalars,
    so vector/list attrs are stringified or dropped."""
    H = nx.Graph()
    for nid, a in G.nodes(data=True):
        H.add_node(
            nid,
            kind=a.get("kind", "paper"),
            title=a.get("title", ""),
            year=int(a["year"]) if a.get("year") is not None else -1,
            citation_count=int(a.get("citation_count", 0)),
            categories=";".join(a.get("categories", [])),
            concepts=";".join(a.get("concepts", [])),
            authors=";".join(a.get("authors", [])),
            n_refs_internal=int(a.get("n_refs_internal", 0)),
        )
    for u, v, e in G.edges(data=True):
        H.add_edge(u, v, kind=e["kind"], weight=float(e["weight"]),
                   cosine=float(e["cosine"]), distance=float(e["distance"]),
                   n_cites=len(e.get("cites", [])))
    path = out_prefix + ".graphml"
    nx.write_graphml(H, path)
    return path


def to_search_graph(G, Gcit):
    """Emit a BARE networkx.DiGraph that satisfies the discovery-graph search consumer's
    drop-in contract (see paper_graph_search.py): nodes carry kind='paper' + concepts
    (required) + citation_count; edges are the paper->paper citations tagged kind='cite'.
    The consumer overlays its own 'knowledge' nodes and recomputes concept-coherence, so
    this exports ONLY the paper side. Returned object is what `load_graph` expects back
    from pickle.load (a graph, not the analysis bundle)."""
    S = nx.DiGraph()
    for nid, a in G.nodes(data=True):
        S.add_node(nid, kind="paper", concepts=list(a.get("concepts", [])),
                   citation_count=int(a.get("citation_count", 0)),
                   title=a.get("title", ""), year=a.get("year"),
                   categories=list(a.get("categories", [])))
    for u, v in Gcit.edges():
        S.add_edge(u, v, kind="cite")          # u cites v
    return S


def save_search_graph(S, out_prefix):
    path = out_prefix + ".search.pkl"
    with open(path, "wb") as f:
        pickle.dump(S, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


# --------------------------------------------------------------------------- #
# Downstream consumer helpers
# --------------------------------------------------------------------------- #
def load_paper_graph(path=DEFAULT_OUT + ".pkl") -> dict:
    """Load the pickle bundle. Returns the dict described in the module docstring."""
    with open(path, "rb") as f:
        return pickle.load(f)


def query_similar(bundle: dict, text: str, top_k: int = 10):
    """Embed free query `text` with the fitted vectorizer and rank corpus papers by
    TF-IDF cosine. This is the primary text-search entry point for the search engine.
    Returns list[(node_id, cosine_score)] sorted high->low."""
    q = bundle["vectorizer"].transform([text])           # 1 x V, L2-normalised
    scores = linear_kernel(q, bundle["tfidf_matrix"]).ravel()
    order = np.argsort(-scores)[:top_k]
    node_order = bundle["node_order"]
    return [(node_order[i], float(scores[i])) for i in order if scores[i] > 0]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def print_report(stats, paths):
    p = print
    p("\n" + "=" * 70)
    p("PAPER GRAPH -- BUILD REPORT")
    p("=" * 70)
    p(f"nodes                : {stats['n_nodes']}")
    p(f"edges                : {stats['n_edges']}  {stats['edges_by_kind']}")
    p(f"  citation links (dir): {stats['n_citation_links_directed']}"
      f"   dangling refs: {stats['n_dangling_refs']}")
    p(f"density              : {stats['density']}")
    d = stats["degree"]
    p(f"degree  min/med/mean/max : {d['min']} / {d['median']} / {d['mean']} / {d['max']}")
    p(f"degree histogram     : {d['histogram']}")
    cc = stats["connected_components"]
    p(f"connected components : {cc['count']}  (largest={cc['largest_size']}, "
      f"singletons={cc['singletons']}, sizes={cc['sizes']})")
    p(f"avg clustering       : {stats['avg_clustering']}   triangles: {stats['triangles']}")
    p(f"\nbetweenness ({stats['betweenness_mode']}) -- top 5:")
    for nid, v, title in stats["top_betweenness"]:
        p(f"   {nid}  {v:.4f}  {title}")
    p("top degree-centrality -- top 5:")
    for nid, v, title in stats["top_degree_centrality"]:
        p(f"   {nid}  {v:.4f}  {title}")
    cg = stats["citation_digraph"]
    p(f"\ncitation DiGraph     : {cg['n_edges']} edges, "
      f"{cg['n_weakly_connected']} weak comp, DAG={cg['is_dag']}")
    p("most-cited in corpus -- top 5:")
    for nid, deg, title in cg["top_cited_in_corpus"]:
        p(f"   {nid}  in-deg={deg}  {title}")
    p("\noutputs:")
    for k, v in paths.items():
        p(f"   {k:8s} -> {v}")
    p("=" * 70 + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the paper graph.")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                    help="corpus .jsonl (falls back to mock if absent)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output path prefix")
    ap.add_argument("--threshold", type=float, default=SIM_THRESHOLD,
                    help="cosine similarity edge threshold")
    ap.add_argument("--citation-weight", type=float, default=CITATION_WEIGHT)
    ap.add_argument("--min-df", type=int, default=None,
                    help="TF-IDF min document frequency (default: 2 if >=5 papers else 1)")
    ap.add_argument("--no-graphml", action="store_true", help="skip graphml export")
    args = ap.parse_args(argv)

    corpus_path = resolve_corpus(args.corpus)
    papers = load_corpus(corpus_path)
    if not papers:
        print("ERROR: corpus is empty.", file=sys.stderr)
        return 1
    print(f"[load] {len(papers)} papers")

    tfidf_matrix, vectorizer = build_tfidf(papers, min_df=args.min_df)
    print(f"[tfidf] matrix {tfidf_matrix.shape} "
          f"(vocab={len(vectorizer.get_feature_names_out())}, nnz={tfidf_matrix.nnz})")

    G, Gcit, node_order, meta = build_graph(
        papers, tfidf_matrix, vectorizer,
        sim_threshold=args.threshold, citation_weight=args.citation_weight)
    print(f"[graph] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    stats = compute_stats(G, Gcit, meta)

    paths = {"pickle": save_bundle(G, Gcit, tfidf_matrix, vectorizer,
                                   node_order, meta, args.out)}
    paths["search"] = save_search_graph(to_search_graph(G, Gcit), args.out)
    if not args.no_graphml:
        paths["graphml"] = save_graphml(G, args.out)
    stats_path = args.out + "_stats.json"
    with open(stats_path, "w") as f:
        json.dump(_json_safe(stats), f, indent=2)
    paths["stats"] = stats_path
    paths["corpus"] = corpus_path

    print_report(stats, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
