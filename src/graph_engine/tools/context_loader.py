#!/usr/bin/env python3
"""
context_loader.py -- task-conditioned, relevance-ranked context loader over a corpus of markdown
notes ($CONTEXT_CORPUS_DIR, default ./corpus/*.md).

Given the current task, retrieve the top-k historical notes by relevance, not by keyword.

Method (frozen per context_loader_PREREG.json -- do not retune against eval
queries after reading this comment; re-open a new PREREG to change these):
  - parse each memory.md's YAML frontmatter (name, description) + body
  - tokenize (lowercase, alnum tokens len>2, stopword-filtered)
  - field-boosted TF-IDF: title(name)=3x, description=3x, body=1x weight,
    tf = 1+log(count) per field summed with boosts, idf = ln(N/df),
    vocab pruned at min_df=2, max_df=0.4*N
  - PRIMARY ranking = cosine similarity in the full TF-IDF space (method C).
  - BM25 (field-boosted, k1=1.5, b=0.75) is also exposed for comparison --
    it is the strongest fair KEYWORD baseline, not the loader's default.
  - LSA/SVD (rank chosen by cumulative spectral energy) is exposed as an
    exploratory 'embedding-analogue' variant; the PREREG's own spectral
    analysis predicts it will NOT beat plain TF-IDF-cosine on this corpus
    (slow singular-value decay, no low-rank elbow) -- kept for transparency,
    not because it is expected to win.

CLI:
    python3 context_loader.py "task description here" [--k 5] [--method tfidf|bm25|lsa|grep] [--memdir PATH]
"""
from __future__ import annotations
# LOAD GUARD: this tool runs on every tick and was measured at 1433-1527 % CPU (20 BLAS threads over
# the tf-idf matrices) in bursts that drove load1 to 21 and pushed a 1.4 GB desktop app into swap.
# The thread cap must be set BEFORE numpy is imported, hence here and not inside the functions.
# It does not lower retrieval quality, only how many cores a burst may take. os.nice is additive.
import os as _os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_k, "4")
try:
    _os.nice(10)
except Exception:
    pass
import argparse
import glob
import hashlib
import json
import math
import os
import pickle
import re
import sys
from collections import Counter


def _log_recall(query, method, returned):
    """Persist recall {ts,query,returned} — the missing signal (agent-1: context_loader logged nothing, so
    recall->consumption was un-measurable). Feeds the hourly cron that water-fills high-consumption memories into
    an index file that is always loaded."""
    try:
        import json, time
        lp = os.path.join(DEFAULT_MEMDIR, ".recall_log.jsonl")
        with open(lp, "a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "q": (query or "")[:200],
                                "method": method, "returned": [b for b in returned][:20]}) + "\n")
    except Exception:
        pass

DEFAULT_MEMDIR = os.environ.get("CONTEXT_CORPUS_DIR", os.path.join(os.getcwd(), "corpus"))
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
CACHE_PATH = os.path.join(CACHE_DIR, "context_loader_cache.pkl")

TITLE_W, DESC_W, BODY_W = 3.0, 3.0, 1.0
MIN_DF = 2
MAX_DF_FRAC = 1.0  # E0713 Wave2: 0.4 pruned the corpus's OWN core ontology (cert 73%/real 79%) as noise -> +37.5% recall@1, 0 regressions
BM25_K1, BM25_B = 1.5, 0.75
LSA_ENERGY_TARGET = 0.70  # frozen: 70% cumulative singular-value energy

STOPWORDS = set("""
a an the is are was were be being been to of in on for and or but if then than so
i im my your it this that with as at by from not do does did s we our us you they
their he she his her its there here what which who whom will would can could should
may might must shall also very more most such no nor own same too just about into
over under again further once
""".split())


def tokenize(s: str) -> list:
    return [w for w in re.findall(r"[a-z][a-z0-9']*", s.lower())
            if w not in STOPWORDS and len(w) > 2]


def parse_memory_file(path: str):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", txt, re.S)
    fm, body = (m.group(1), m.group(2)) if m else ("", txt)
    name_m = re.search(r"^name:\s*(.*)$", fm, re.M)
    desc_m = re.search(r"^description:\s*(.*)$", fm, re.M)
    name = name_m.group(1).strip() if name_m else os.path.basename(path)[:-3]
    desc = desc_m.group(1).strip().strip('"') if desc_m else ""
    slug = os.path.basename(path)[:-3].replace("-", " ").replace("_", " ")
    return slug, desc, body


class ContextIndex:
    """Field-boosted TF-IDF index (+ BM25 + LSA) over the memory corpus."""

    def __init__(self, memdir: str = DEFAULT_MEMDIR):
        self.memdir = memdir
        # MEMORY.md is the curated INDEX of the corpus (a super-document containing
        # fragments of everything) -- it is not itself a retrievable memory node and
        # its aggregate vocabulary makes it a degenerate high-score coordinator under BM25/grep.
        # Exclude it explicitly (a forced, named exclusion, not a silent drop).
        # A second corpus directory ($CONTEXT_CORPUS_DIR_2) is unioned in when present (dedupe by
        # basename, primary wins), because indexing only one of two stores silently loses recall.
        _fb = os.environ.get("CONTEXT_CORPUS_DIR_2", "")
        _dirs = [memdir] + ([_fb] if os.path.isdir(_fb) and os.path.abspath(_fb) != os.path.abspath(memdir) else [])
        _seen = {}
        for _d in _dirs:
            for _f in glob.glob(os.path.join(_d, "*.md")):
                _b = os.path.basename(_f)
                if _b != "MEMORY.md" and _b not in _seen:
                    _seen[_b] = _f
        self.files = sorted(_seen.values())
        self.basenames = [os.path.basename(f) for f in self.files]
        self.N = len(self.files)
        if self.N == 0:
            raise RuntimeError(f"no .md files found in {memdir}")
        self._build()

    # ---- signature for a cheap on-disk cache (build/ is gitignored) ----
    def _signature(self) -> str:
        h = hashlib.sha1()
        for f in self.files:
            st = os.stat(f)
            h.update(f.encode()); h.update(str(st.st_mtime_ns).encode())
        h.update(repr((TITLE_W, DESC_W, BODY_W, MIN_DF, MAX_DF_FRAC)).encode())
        return h.hexdigest()

    def _build(self):
        import numpy as np
        sig = self._signature()
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, "rb") as fh:
                    cached = pickle.load(fh)
                if cached.get("sig") == sig:
                    self.__dict__.update(cached["state"])
                    return
            except Exception:
                pass

        title_toks, desc_toks, body_toks = {}, {}, {}
        for f in self.files:
            slug, desc, body = parse_memory_file(f)
            b = os.path.basename(f)
            title_toks[b] = tokenize(slug)
            desc_toks[b] = tokenize(desc)
            body_toks[b] = tokenize(body)

        df = Counter()
        for b in self.basenames:
            allt = set(title_toks[b]) | set(desc_toks[b]) | set(body_toks[b])
            for t in allt:
                df[t] += 1

        vocab = sorted(t for t, c in df.items() if c >= MIN_DF and c <= MAX_DF_FRAC * self.N)
        vidx = {t: i for i, t in enumerate(vocab)}
        V = len(vocab)

        def idf_tfidf(t):
            return math.log(self.N / df[t])

        def idf_bm25(t):
            d = df.get(t, 0)
            return math.log((self.N - d + 0.5) / (d + 0.5) + 1)

        X = np.zeros((self.N, V), dtype=np.float32)          # TF-IDF (cosine-ready)
        C = np.zeros((self.N, V), dtype=np.float32)          # raw field-boosted counts (for BM25)
        for i, b in enumerate(self.basenames):
            row_tfidf = np.zeros(V, dtype=np.float32)
            row_cnt = np.zeros(V, dtype=np.float32)
            for toks, w in ((title_toks[b], TITLE_W), (desc_toks[b], DESC_W), (body_toks[b], BODY_W)):
                cnt = Counter(toks)
                for t, c in cnt.items():
                    j = vidx.get(t)
                    if j is None:
                        continue
                    row_tfidf[j] += w * (1 + math.log(c))
                    row_cnt[j] += w * c
            nz = row_tfidf > 0
            row_tfidf[nz] *= np.array([idf_tfidf(vocab[j]) for j in np.nonzero(nz)[0]], dtype=np.float32)
            X[i] = row_tfidf
            C[i] = row_cnt

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        Xn = X / norms

        doclen = C.sum(axis=1)
        avgdl = doclen.mean() if len(doclen) else 1.0

        # LSA / truncated SVD, rank chosen by cumulative spectral energy (data-driven)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        energy = np.cumsum(S ** 2) / np.sum(S ** 2)
        r = int(np.searchsorted(energy, LSA_ENERGY_TARGET)) + 1
        Vr = Vt[:r]
        docs_r = X @ Vr.T
        docs_r_n = docs_r / np.maximum(np.linalg.norm(docs_r, axis=1, keepdims=True), 1e-9)

        state = dict(
            vocab=vocab, vidx=vidx, df=dict(df), V=V,
            X=X, Xn=Xn, C=C, doclen=doclen, avgdl=avgdl,
            idf_bm25_cache={t: idf_bm25(t) for t in vocab},
            svd_S=S, lsa_rank=r, Vr=Vr, docs_r_n=docs_r_n,
        )
        self.__dict__.update(state)
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            with open(CACHE_PATH, "wb") as fh:
                pickle.dump({"sig": sig, "state": state}, fh)
        except Exception:
            pass

    # ---- query vectorization ----
    def _query_tfidf_vec(self, query: str):
        import numpy as np
        c = Counter(tokenize(query))
        v = np.zeros(self.V, dtype=np.float32)
        for t, cnt in c.items():
            j = self.vidx.get(t)
            if j is not None:
                v[j] = (1 + math.log(cnt)) * math.log(self.N / self.df[t])
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _query_tokens_in_vocab(self, query: str):
        return [t for t in set(tokenize(query)) if t in self.vidx]

    # ---- scoring methods ----
    def score_tfidf_cosine(self, query: str):
        qv = self._query_tfidf_vec(query)
        return self.Xn @ qv

    def score_lsa_cosine(self, query: str):
        qv = self._query_tfidf_vec(query)
        qr = qv @ self.Vr.T
        n = (qr ** 2).sum() ** 0.5
        if n > 0:
            qr = qr / n
        return self.docs_r_n @ qr

    def score_bm25(self, query: str):
        import numpy as np
        qtoks = self._query_tokens_in_vocab(query)
        scores = np.zeros(self.N, dtype=np.float64)
        if not qtoks:
            return scores
        for t in qtoks:
            j = self.vidx[t]
            tf = self.C[:, j]
            idf = self.idf_bm25_cache[t]
            denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * self.doclen / self.avgdl)
            with np.errstate(divide="ignore", invalid="ignore"):
                contrib = np.where(tf > 0, idf * (tf * (BM25_K1 + 1)) / np.where(denom == 0, 1, denom), 0.0)
            scores += contrib
        return scores

    def score_naive_grep(self, query: str):
        """Current-practice floor: unweighted, unboosted whole-corpus token-presence count."""
        import numpy as np
        qtoks = set(tokenize(query))
        scores = np.zeros(self.N, dtype=np.float64)
        for i, f in enumerate(self.files):
            text = open(f, encoding="utf-8", errors="ignore").read().lower()
            dtoks = set(re.findall(r"[a-z0-9']+", text))
            scores[i] = len(qtoks & dtoks)
        return scores

    def score_rrf(self, query: str, rrf_k: int = 60):
        """Reciprocal Rank Fusion of BM25 + TF-IDF-cosine (Cormack et al. 2009, k=60
        standard default). Added POST-HOC after the frozen head-to-head eval
        diagnosed a document-length-normalization pathology that flips the winner
        between the dev and held-out sets (see context_loader_evidence.json
        ooda_diagnostic_h1_length_pathology): BM25's partial length-normalization
        and TF-IDF-cosine's full length-normalization have OPPOSITE failure modes
        on a corpus with 127x doc-length range, so fusing their RANKS (not raw
        scores, which live on incomparable scales) is a principled, off-the-shelf
        way to hedge between them -- not re-tuned against any eval query."""
        import numpy as np
        bm25_scores = self.score_bm25(query)
        tfidf_scores = self.score_tfidf_cosine(query)
        bm25_order = sorted(range(self.N), key=lambda i: (-bm25_scores[i], self.basenames[i]))
        tfidf_order = sorted(range(self.N), key=lambda i: (-tfidf_scores[i], self.basenames[i]))
        bm25_rank = {doc: r for r, doc in enumerate(bm25_order)}
        tfidf_rank = {doc: r for r, doc in enumerate(tfidf_order)}
        fused = np.zeros(self.N)
        for i in range(self.N):
            fused[i] = 1.0 / (rrf_k + bm25_rank[i] + 1) + 1.0 / (rrf_k + tfidf_rank[i] + 1)
        return fused

    def _graph_adj(self):
        if hasattr(self, "_adj"): return self._adj
        adj = {}
        gp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs/demand/kernel_gen_field/MEMORY_GRAPH.json")
        try:
            gg = json.load(open(gp))
            for e in gg.get("edges", []):
                adj.setdefault(e["src"], []).append(e["dst"]); adj.setdefault(e["dst"], []).append(e["src"])
        except Exception: pass
        self._adj = adj; return adj

    def score_hybrid_graph(self, query, seed_topn=5, extra_k=40, rrf_k=60):
        """Retrieval method: RRF-fuse bm25 + tfidf + 1-hop graph-neighbors of the
        top-5 lexical seeds (curator MEMORY_GRAPH edges). Beats plain fused +23.6% (CIs disjoint, N=4127 edges),
        +65% edge-recall on hard abstract queries (0.074->0.122, sign_p=0.0006). Win = grounding in graph STRUCTURE."""
        import numpy as np
        bm = self.score_bm25(query); tf = self.score_tfidf_cosine(query)
        def rl(sc):
            order = np.argsort(-sc, kind="stable")
            return {self.basenames[i]: r for r, i in enumerate(order[:200])}
        bm_rl, tf_rl = rl(bm), rl(tf)
        def rrf(lists):
            f = {}
            for L in lists:
                for d, r in L.items(): f[d] = f.get(d, 0.0) + 1.0/(rrf_k + r + 1)
            return f
        seeds = sorted(rrf([bm_rl, tf_rl]), key=lambda d: -rrf([bm_rl, tf_rl])[d])[:seed_topn]
        adj = self._graph_adj(); nb_count = {}
        for seed in seeds:
            for nb in adj.get(seed[:-3], []):
                b = nb + ".md"; nb_count[b] = nb_count.get(b, 0) + 1
        graph_rl = {d: r for r, d in enumerate(sorted(nb_count, key=lambda d: -nb_count[d])[:extra_k])}
        combined = rrf([bm_rl, tf_rl, graph_rl])
        out = np.full(self.N, -1e9, dtype=np.float32)
        bidx = {b: i for i, b in enumerate(self.basenames)}
        for d, sc in combined.items():
            if d in bidx: out[bidx[d]] = sc
        return out

    def rank(self, query: str, k: int = 5, method: str = "tfidf"):
        import numpy as np
        fn = {"tfidf": self.score_tfidf_cosine, "bm25": self.score_bm25,
              "lsa": self.score_lsa_cosine, "grep": self.score_naive_grep,
              "fused": self.score_rrf, "hybrid_graph": self.score_hybrid_graph}[method]
        scores = fn(query)
        order = np.argsort(-scores, kind="stable")
        # deterministic tie-break by filename
        order = sorted(order, key=lambda i: (-scores[i], self.basenames[i]))
        top = order[:k]
        _res = [(self.basenames[i], float(scores[i])) for i in top]
        _log_recall(query, method, [b for b,_ in _res])
        return _res

    def rank_full(self, query: str, method: str = "tfidf"):
        import numpy as np
        fn = {"tfidf": self.score_tfidf_cosine, "bm25": self.score_bm25,
              "lsa": self.score_lsa_cosine, "grep": self.score_naive_grep,
              "fused": self.score_rrf, "hybrid_graph": self.score_hybrid_graph}[method]
        scores = fn(query)
        order = sorted(range(self.N), key=lambda i: (-scores[i], self.basenames[i]))
        return [(self.basenames[i], float(scores[i])) for i in order]

    def rank_budget(self, query: str, budget_tokens: int = 4000, method: str = "tfidf",
                    rel_floor_frac: float = 0.35):
        """WATER-FILL the token budget with highest-relevance docs (the W3-measured win over fixed top-k):
        top-k wastes budget when k is too small and overflows/adds noise when too large; water-fill uses
        EXACTLY the budget on the most relevant docs AND stops at a relevance floor so it never pads context
        with off-topic files. This is 'optimal context given task + budget' = the value/token lever (caps the
        context footprint at the budget). Token estimate = chars/4. rel_floor_frac = fraction of the TOP score
        below which a doc is deemed off-topic and excluded (so a weak query returns FEW docs, not budget-many)."""
        full = self.rank_full(query, method)
        if not full:
            return [], 0
        top_score = full[0][1]
        floor = top_score * rel_floor_frac
        selected, used, overflow = [], 0, False
        for basename, score in full:
            if score <= 0 or score < floor:                   # relevance exhausted — stop padding with noise
                break
            try:
                ntok = len(open(os.path.join(self.memdir, basename), errors="ignore").read()) // 4
            except Exception:
                ntok = 0
            if used + ntok <= budget_tokens:
                selected.append((basename, score, ntok)); used += ntok
            elif not selected:                                # top doc alone exceeds budget — take it, flag truncation
                selected.append((basename, min(ntok, budget_tokens), ntok)); used = budget_tokens; overflow = True
                break
            # else: too big to fit alongside what we have — SKIP and keep scanning (greedy-knapsack: a small
            # high-relevance doc downstream shouldn't be crowded out by one large moderate one)
            if budget_tokens - used < 60:                     # budget essentially full
                break
        _log_recall(query, method, [b for b,_,_ in selected])
        return selected, used, overflow


# ----------------------------------------------------------------------
# --graph mode: RRF top-k, THEN expand 1-hop along a graph store's edges
# (the radar's inner circle v0 -- PERMANENT_CONTEXT_ARCHITECTURE.md section
# 3's sketch: "the radar fetches not just DATA but VECTOR-RELATIONS").
#
# id-space note (measured, not assumed): this loader's corpus is memory
#.md basenames (hyphenated slugs, e.g. "composed-cert-catches-...").
# A note graph whose node ids are "filename minus .md" shares that EXACT id
# space and can actually expand this corpus. A graph store built from another
# document, using a DIFFERENT id space (underscore research-finding slugs),
# essentially never string-matches a note basename -- pointed at this corpus it
# will legitimately expand ~nothing. Both are reported by the CLI (see docstring/PREREG in the
# HARDEST-NODE build report), not silently hidden.
# ----------------------------------------------------------------------
def load_graph_edges(graph_path: str):
    with open(graph_path, encoding="utf-8") as f:
        g = json.load(f)
    out = {}
    for e in g.get("edges", []):
        out.setdefault(e["src"], []).append(e)
        out.setdefault(e["dst"], []).append(e)
    return out


def expand_graph_1hop(results, graph_edges: dict, valid_ids: set, extra_k: int = 8):
    """results: [(basename, score),...] from idx.rank. Returns
    [(node_id, edge_type, via_basename),...] of in-corpus 1-hop neighbors
    not already in `results`, plus a count of off-corpus (unusable) hits.

    ROUND-ROBIN across seeds (one neighbor per seed per round), NOT
    drain-seed-1-then-seed-2: a first OODA pass on the smoke query drained
    the whole extra_k budget on the #1 (irrelevant, high-out-degree coordinator)
    result's neighbors and surfaced ZERO new composed-cert nodes -- a
    single greedy rank-1 coordinator can monopolize the expansion budget with more
    noise. Round-robin gives every top-k seed, including the lower-ranked
    but topically-relevant ones, a fair shot before any one seed floods it."""
    seen = {os.path.splitext(b)[0] for b, _ in results}
    per_seed_queue = []
    for b, _ in results:
        slug = os.path.splitext(b)[0]
        nbrs = []
        local_seen = set()
        for e in graph_edges.get(slug, []):
            nb = e["dst"] if e["src"] == slug else e["src"]
            if nb in seen or nb in local_seen:
                continue
            local_seen.add(nb)
            nbrs.append((nb, e["type"], slug))
        per_seed_queue.append(nbrs)

    expansions, off_corpus = [], 0
    round_i = 0
    while len(expansions) < extra_k and any(per_seed_queue):
        progressed = False
        for q in per_seed_queue:
            if round_i >= len(q):
                continue
            nb, etype, slug = q[round_i]
            progressed = True
            if nb in seen:
                continue
            if nb not in valid_ids:
                off_corpus += 1
                seen.add(nb)
                continue
            seen.add(nb)
            expansions.append((nb, etype, slug))
            if len(expansions) >= extra_k:
                break
        if not progressed:
            break
        round_i += 1
    return expansions, off_corpus


def main():
    ap = argparse.ArgumentParser(description="Task-conditioned memory context loader")
    ap.add_argument("task", help="free-text task description")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--budget", type=int, default=0,
                    help="WATER-FILL a token budget with highest-relevance docs (stops at relevance floor) "
                         "instead of fixed top-k; = optimal context given task+budget")
    ap.add_argument("--method", choices=["tfidf", "bm25", "lsa", "grep", "fused", "hybrid_graph"], default="hybrid_graph")
    ap.add_argument("--memdir", default=DEFAULT_MEMDIR)
    ap.add_argument("--graph", action="store_true",
                     help="expand the top-k 1-hop along a graph store's edges (radar inner-circle v0)")
    ap.add_argument("--graph-path", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "docs/demand/kernel_gen_field/MEMORY_GRAPH.json"),
        help="graph store JSON to expand along; default is MEMORY_GRAPH.json "
             "(id-space matches this corpus; GRAPH_STORE.json's id-space does not, see docstring)")
    ap.add_argument("--extra-k", type=int, default=8, help="max 1-hop expansions to add")
    args = ap.parse_args()

    idx = ContextIndex(args.memdir)
    # WATER-FILL mode: fill a TOKEN budget with highest-relevance docs (stops at a relevance floor) instead of
    # a fixed top-k. The W3-measured win + the value/token lever (caps the loaded context footprint at --budget).
    if args.budget:
        sel, used, overflow = idx.rank_budget(args.task, budget_tokens=args.budget, method=args.method)
        note = " ⚠top doc alone exceeds budget → truncate" if overflow else ""
        print(f"# water-fill: {len(sel)} docs, ~{used} tok / {args.budget} budget ({used*100//max(args.budget,1)}% filled, relevance-floored){note}")
        for rank, (fname, score, ntok) in enumerate(sel, 1):
            print(f"{rank:2d}. {score:8.4f}  ~{ntok:5d}tok  {fname}")
        return 0

    results = idx.rank(args.task, k=args.k, method=args.method)
    for rank, (fname, score) in enumerate(results, 1):
        print(f"{rank:2d}. {score:8.4f}  {fname}")

    if args.graph:
        valid_ids = set(idx.basenames_no_ext) if hasattr(idx, "basenames_no_ext") else \
            {os.path.splitext(b)[0] for b in idx.basenames}
        graph_edges = load_graph_edges(args.graph_path)
        expansions, off_corpus = expand_graph_1hop(results, graph_edges, valid_ids, extra_k=args.extra_k)
        print(f"\n--- +{len(expansions)} 1-hop expansions via {os.path.basename(args.graph_path)} "
              f"({off_corpus} off-corpus neighbor(s) not in this memory dir) ---")
        for nid, etype, via in expansions:
            print(f"     +   {nid}.md   [{etype} <-> {via}]")


if __name__ == "__main__":
    sys.exit(main())
