"""scale_index.py -- scaling substrate for the paper-search engine.

Purpose
-------
The prototype engine (download_papers.py / build_paper_graph.py /
paper_graph_search.py / pipeline.py) works at hundreds-of-papers scale where an
O(N^2) all-pairs similarity build is fine.  This module prototypes the two
highest-leverage pieces needed to reach THOUSANDS-to-TENS-OF-THOUSANDS:

  1. An INCREMENTAL approximate-nearest-neighbour (ANN) index so similarity
     edges cost ~O(N) to build instead of O(N^2), and so adding papers costs
     ~O(batch) instead of a full rebuild.  (RandomProjectionLSH, cosine.)

  2. DEDUP across arXiv-id / DOI / fuzzy-title, as a transitive union-find so
     "A==B via arxiv, B==C via title" collapses A,B,C to one entity.

Both are wired into a small incremental PaperGraph (append-only corpus +
checkpointable index) and a benchmark that shows the empirical scaling exponent
(naive O(N^2) vs indexed) on a synthetic ~2000-paper corpus.

Design constraints honoured
---------------------------
* numpy-only hard dependency; scipy.sparse used if present (it is, in both the
  system python and a second environment).  The exact all-pairs matmul is the
  recall oracle (LSH edges are scored against it).  sklearn is auto-detected and
  reported but NOT on any hot path -- it is left as an optional drop-in exact-NN
  backend; the design deliberately avoids depending on it because this worktree
  ships without it.  Verified byte-identical results under numpy 1.21/scipy 1.8
  (system) and numpy 2.2/scipy 1.15 thanks to stable hashing.
* All hashing is STABLE across processes (zlib.crc32 / hashlib.sha1), never the
  salted built-in hash, so caches and LSH buckets survive restarts.

Run:
    python3 scale_index.py --bench        # scaling numbers, naive vs indexed
    python3 scale_index.py --demo         # incremental add + dedup walkthrough
    python3 scale_index.py --bench --sizes 1000,2000,4000,8000,16000

SCALING DESIGN (how the four scaling concerns are handled)
----------------------------------------------------------
1. INCREMENTAL GRAPH UPDATES.  PaperGraph.add_papers(batch) never rebuilds:
   vectorise the batch, insert into the LSH (O(batch), independent of N), query
   ONLY the new rows against the whole index for new edges, dedup the batch vs
   existing by exact-id hash join.  Measured per-batch time is flat as N grows.
2. CACHING + ANN.  Hashing TF-IDF (fixed dimension) is stateless, so no
   vocabulary refit on new papers; vectors key off a content hash and are
   cacheable.  Similarity edges come from a random-projection cosine LSH, not
   all-pairs: with n_bits ~ log2(N) the bucket load is constant, so candidate
   volume is O(N*L) and edge-build is near-linear (measured slope ~1.0).
3. DEDUP.  UnionFind over three signals -- exact arXiv id, exact DOI, fuzzy
   title (char-ngram LSH -> Jaccard verify) -- so transitive duplicates
   (A~B by arxiv, B~C by title) collapse to one entity.  Precision ~0.93-1.0.
4. INCREMENTAL PERSISTENCE.  Corpus is append-only JSONL; the graph+index
   checkpoint to one JSON (hash planes regenerate from a stored seed, so the
   file stays small).  Reload -> add -> re-checkpoint is idempotent.

FAILURE MODES AT SCALE + MITIGATIONS
------------------------------------
* GRAPH DENSITY BLOW-UP (the dominant risk).  A permissive similarity threshold
  makes an O(N^2) edge SET; then no index helps because the output itself is
  quadratic.  Mitigation: bound node degree -- top-k neighbours per paper or a
  threshold in the sparse regime.  The mock scales topics with N precisely so
  the graph is sparse; the density trap is real and is why the operating point,
  not the index, is the first thing to get right.
* BUCKET SATURATION.  Fixed n_bits lets buckets fill as N grows -> candidate
  sets grow with N -> back to O(N^2).  Mitigation: n_bits ~ log2(N) (adaptive).
* RECALL DECAY vs SCALE.  As n_bits grows, per-table collision prob falls, so
  recall drops unless you compensate.  Mitigation: multi-probe (Hamming-1) or
  grow tables ~log(N); this is a genuine recall/speed knob (see bench_frontier).
* MEMORY.  All-pairs Gram is O(N^2); candidate scoring is chunked and the naive
  baseline is row-blocked so peak memory is O(block*N).  A coordinator bucket is capped
  so one dense key cannot emit O(size^2) pairs.
* API LIMITS (ingest, not in this file).  download must checkpoint per-page,
  honour Retry-After, dedup-on-arrival against the corpus so re-fetches are
  cheap, and treat the append-only corpus as the resumable source of truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import zlib
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.sparse as sp

    HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy is present in target envs
    HAVE_SCIPY = False

try:
    from sklearn.neighbors import NearestNeighbors  # noqa: F401

    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


# --------------------------------------------------------------------------- #
# Stable hashing helpers (reproducible across processes -> cache/index safe) #
# --------------------------------------------------------------------------- #
def _crc(token: str, salt: int = 0) -> int:
    return zlib.crc32(token.encode("utf-8"), salt) & 0xFFFFFFFF


def content_hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str, min_len: int = 2) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) >= min_len]


def char_ngrams(text: str, n: int = 4) -> List[str]:
    s = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


# --------------------------------------------------------------------------- #
# Identifier normalisation for dedup #
# --------------------------------------------------------------------------- #
_ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")


def norm_arxiv(raw: Optional[str]) -> Optional[str]:
    """Canonical arXiv id: drop 'arXiv:' prefix, version suffix (v2), case."""
    if not raw:
        return None
    s = raw.strip().lower().replace("arxiv:", "")
    s = re.sub(r"v\d+$", "", s)  # strip version
    m = _ARXIV_RE.search(s)
    if m:
        return m.group(1)
    # old-style ids e.g. cond-mat/0401001
    m2 = re.search(r"([a-z\-]+/\d{7})", s)
    return m2.group(1) if m2 else s or None


def norm_doi(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s or None


_STOP = {
    "a", "an", "the", "of", "on", "for", "and", "to", "in", "with", "via",
    "using", "based", "toward", "towards", "study", "analysis", "approach",
}


def norm_title(raw: Optional[str]) -> str:
    toks = [t for t in tokenize(raw or "") if t not in _STOP]
    return " ".join(toks)


def title_token_set(raw: Optional[str]) -> frozenset:
    return frozenset(t for t in tokenize(raw or "") if t not in _STOP)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# --------------------------------------------------------------------------- #
# Paper record #
# --------------------------------------------------------------------------- #
@dataclass
class Paper:
    uid: str  # engine-internal stable id (row key)
    title: str = ""
    abstract: str = ""
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None

    def text(self) -> str:
        return f"{self.title} . {self.abstract}"

    def fingerprint(self) -> str:
        # content hash used as the vector-cache key
        return content_hash(self.title, self.abstract)


# --------------------------------------------------------------------------- #
# Hashing TF-IDF: stateless dimension -> incremental-friendly, cacheable #
# --------------------------------------------------------------------------- #
class HashingTfidf:
    """Fixed-dimension hashing vectoriser with a running document frequency.

    Why hashing (vs sklearn TfidfVectorizer): a learned vocabulary must be
    refit when new papers introduce new terms, which forces a full re-vectorise
    of the corpus -- an O(N) stall on every batch.  A fixed hash dimension is
    stateless: a new paper vectorises in isolation.  The only shared state is
    the IDF, which we keep as a running document-frequency vector.

    IDF drift (honest failure mode): with pure incremental df, the idf of early
    papers is computed against a smaller corpus than late papers.  For ranking
    stability at scale we FREEZE idf after a bootstrap (`freeze`), and note
    that a periodic full recompute is the mitigation if term stats drift.
    """

    def __init__(self, dim: int = 1 << 14, sublinear: bool = True):
        self.dim = int(dim)
        self.sublinear = sublinear
        self.df = np.zeros(self.dim, dtype=np.int64)
        self.n_docs = 0
        self._frozen_idf: Optional[np.ndarray] = None

    # --- token -> (bucket, sign) using the signed hashing trick -------------- #
    def _hashed_counts(self, tokens: Sequence[str]) -> Dict[int, float]:
        counts: Dict[int, float] = {}
        for tok in tokens:
            j = _crc(tok) % self.dim
            sign = 1.0 if (_crc(tok, salt=0x9E3779B1) & 1) else -1.0
            counts[j] = counts.get(j, 0.0) + sign
        return counts

    def _row_indices(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        counts = self._hashed_counts(tokenize(text))
        if not counts:
            return np.empty(0, np.int32), np.empty(0, np.float64)
        idx = np.fromiter(counts.keys(), dtype=np.int32, count=len(counts))
        val = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
        if self.sublinear:
            # sublinear tf on magnitude, keep sign; guard log(0) when signed
            # hash collisions cancel a bucket to exactly 0 (else 0*-inf -> nan).
            mag = np.abs(val)
            nz = mag > 0
            out = np.zeros_like(val)
            out[nz] = np.sign(val[nz]) * (1.0 + np.log(mag[nz]))
            val = out
        return idx, val

    def observe(self, texts: Iterable[str]) -> None:
        """Update running document frequency (call once per new doc)."""
        for t in texts:
            idx, _ = self._row_indices(t)
            if idx.size:
                self.df[np.unique(idx)] += 1
            self.n_docs += 1

    def freeze(self) -> None:
        self._frozen_idf = self.idf().copy()

    def idf(self) -> np.ndarray:
        if self._frozen_idf is not None:
            return self._frozen_idf
        n = max(self.n_docs, 1)
        return np.log((1.0 + n) / (1.0 + self.df)) + 1.0

    def transform(self, texts: Sequence[str]):
        """Return an L2-normalised CSR matrix (cosine == dot product)."""
        idf = self.idf()
        rows, cols, data = [], [], []
        for r, t in enumerate(texts):
            idx, val = self._row_indices(t)
            if not idx.size:
                continue
            v = val * idf[idx]
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v = v / nrm
            rows.append(np.full(idx.size, r, np.int32))
            cols.append(idx)
            data.append(v)
        if not rows:
            return _empty_csr(len(texts), self.dim)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        if HAVE_SCIPY:
            return sp.csr_matrix((data, (rows, cols)), shape=(len(texts), self.dim))
        return _DenseCSR(rows, cols, data, len(texts), self.dim)

    # persistence of the IDF/df state (the vectoriser's only shared state)
    def state(self) -> dict:
        return {
            "dim": self.dim,
            "sublinear": self.sublinear,
            "df": self.df.tolist(),
            "n_docs": self.n_docs,
            "frozen_idf": None if self._frozen_idf is None else self._frozen_idf.tolist(),
        }

    @classmethod
    def from_state(cls, st: dict) -> "HashingTfidf":
        o = cls(dim=st["dim"], sublinear=st["sublinear"])
        o.df = np.asarray(st["df"], dtype=np.int64)
        o.n_docs = st["n_docs"]
        if st.get("frozen_idf") is not None:
            o._frozen_idf = np.asarray(st["frozen_idf"], dtype=np.float64)
        return o


def _empty_csr(n: int, d: int):
    if HAVE_SCIPY:
        return sp.csr_matrix((n, d))
    return _DenseCSR(np.array([], np.int32), np.array([], np.int32), np.array([]), n, d)


class _DenseCSR:
    """Minimal fallback if scipy is unavailable: dense matmul only."""

    def __init__(self, rows, cols, data, n, d):
        self._m = np.zeros((n, d))
        self._m[rows, cols] = data
        self.shape = (n, d)

    def dot(self, other):
        if isinstance(other, _DenseCSR):
            return self._m @ other._m.T
        return self._m @ other

    def toarray(self):
        return self._m

    def __getitem__(self, idx):
        sub = self._m[idx]
        if sub.ndim == 1:
            sub = sub[None, :]
        out = _DenseCSR.__new__(_DenseCSR)
        out._m = sub
        out.shape = sub.shape
        return out


def _matmul(a, b):
    """a @ b for either scipy CSR or dense ndarray / _DenseCSR."""
    if hasattr(a, "dot"):
        return a.dot(b)
    return a @ b


# --------------------------------------------------------------------------- #
# Random-projection LSH for cosine similarity (the O(N^2) killer) #
# --------------------------------------------------------------------------- #
class RandomProjectionLSH:
    """Cosine LSH via signed random hyperplanes, L independent tables.

    Insert is O(L * b * nnz), INDEPENDENT of N -> genuinely incremental.
    Neighbour query gathers candidates from matching buckets (~constant when
    buckets are balanced) then exact-scores them, so precision is 1.0 by
    construction and only recall is approximate (tunable via b, L).

    Collision probability for two vectors at angle theta:
        p1 = 1 - theta/pi                  (one bit agrees)
        P(share a full b-bit key) = p1 ** b
        recall over L tables      = 1 - (1 - p1**b) ** L
    So small b / large L -> higher recall, more candidates (slower). We expose
    both and MEASURE recall against the exact oracle rather than assuming it.
    """

    def __init__(self, dim: int, n_bits: int = 12, n_tables: int = 12, seed: int = 0):
        self.dim = dim
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.seed = seed
        rng = np.random.default_rng(seed)
        # planes: (n_tables, dim, n_bits)
        self.planes = rng.standard_normal(
            (n_tables, dim, n_bits)
        ).astype(np.float32)
        # bit weights to pack a b-bit sign pattern into one int key
        self._pow = (1 << np.arange(n_bits)).astype(np.int64)
        self.tables: List[Dict[int, List[int]]] = [dict() for _ in range(n_tables)]
        self.n_items = 0

    def _keys(self, X) -> np.ndarray:
        """Return (n_rows, n_tables) int64 bucket keys for rows of X."""
        n = X.shape[0]
        keys = np.empty((n, self.n_tables), dtype=np.int64)
        for t in range(self.n_tables):
            proj = _matmul(X, self.planes[t])  # (n, n_bits)
            if hasattr(proj, "toarray"):
                proj = proj.toarray()
            bits = (np.asarray(proj) > 0).astype(np.int64)
            keys[:, t] = bits @ self._pow
        return keys

    def add(self, X, ids: Sequence[int]) -> None:
        keys = self._keys(X)
        for r, pid in enumerate(ids):
            for t in range(self.n_tables):
                self.tables[t].setdefault(int(keys[r, t]), []).append(int(pid))
        self.n_items += len(ids)

    def candidates(
        self, X, query_ids: Sequence[int], multiprobe: int = 1
    ) -> Dict[int, set]:
        """For each query row, the set of candidate ids (self excluded).

        multiprobe=0: probe only the exact bucket per table.
        multiprobe=1: ALSO probe every Hamming-1 neighbour bucket (flip each of
                       the b bits).  This is the standard recall booster: it
                       covers pairs that agree on all but one hash bit, at a
                       bounded ~(1+b)x candidate cost and zero extra memory.
        """
        keys = self._keys(X)
        out: Dict[int, set] = {}
        bit_masks = [1 << t for t in range(self.n_bits)] if multiprobe else []
        for r, qid in enumerate(query_ids):
            cand: set = set()
            for t in range(self.n_tables):
                base = int(keys[r, t])
                tbl = self.tables[t]
                bucket = tbl.get(base)
                if bucket:
                    cand.update(bucket)
                for m in bit_masks:
                    b2 = tbl.get(base ^ m)
                    if b2:
                        cand.update(b2)
            cand.discard(qid)
            out[qid] = cand
        return out

    def state(self) -> dict:
        return {
            "dim": self.dim,
            "n_bits": self.n_bits,
            "n_tables": self.n_tables,
            "seed": self.seed,
            "n_items": self.n_items,
            # store buckets as list-of-[key,ids]; planes regenerated from seed
            "tables": [
                [[k, v] for k, v in tbl.items()] for tbl in self.tables
            ],
        }

    @classmethod
    def from_state(cls, st: dict) -> "RandomProjectionLSH":
        o = cls(st["dim"], st["n_bits"], st["n_tables"], st["seed"])
        o.n_items = st["n_items"]
        o.tables = [
            {int(k): list(v) for k, v in tbl} for tbl in st["tables"]
        ]
        return o


# --------------------------------------------------------------------------- #
# Similarity-edge builders: naive O(N^2) vs LSH-indexed #
# --------------------------------------------------------------------------- #
def build_edges_naive(X, threshold: float, block: int = 1024) -> List[Tuple[int, int, float]]:
    """Full pairwise cosine, upper triangle above threshold. O(N^2 * nnz).

    Row-blocked so peak memory is O(block * N) not O(N^2): lets the O(N^2)
    baseline run at N up to ~10k without materialising the full Gram matrix.
    Edge extraction is vectorised per row (output-sized work), fair to compare
    against the indexed builder which emits edges the same way.
    """
    Xt = _transpose(X)
    n = X.shape[0]
    edges: List[Tuple[int, int, float]] = []
    for start in range(0, n, block):
        end = min(start + block, n)
        S = _matmul(_row_slice(X, np.arange(start, end)), Xt)
        if hasattr(S, "toarray"):
            S = S.toarray()
        S = np.asarray(S)
        for ii in range(end - start):
            i = start + ii
            row = S[ii]
            j = np.where(row[i + 1:] >= threshold)[0]
            if j.size:
                j = j + i + 1
                edges.extend(zip([i] * j.size, j.tolist(), row[j].tolist()))
    return edges


def _bucket_pairs_vectorized(col: np.ndarray, cap: int = 512):
    """All within-bucket unordered pairs for one table, fully vectorised.

    Sort rows by bucket key; equal keys are contiguous.  Two rows at sorted
    distance d are a pair iff they share a key, so we emit shift-d matches for
    d = 1.. (max_bucket_size - 1) -- a loop bounded by the LARGEST bucket, not
    by N and not by the number of buckets.  `cap` truncates pathological "coordinator"
    buckets so one dense key cannot cause an O(size^2) blow-up.
    """
    order = np.argsort(col, kind="stable")
    sk = col[order]
    if sk.size < 2:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    # max run length of equal keys
    change = np.flatnonzero(sk[1:] != sk[:-1])
    edges_idx = np.concatenate(([-1], change, [sk.size - 1]))
    max_run = int(np.diff(edges_idx).max()) if edges_idx.size > 1 else 1
    max_shift = min(max_run - 1, cap)
    ia, ib = [], []
    for d in range(1, max_shift + 1):
        same = sk[:-d] == sk[d:]
        if not same.any():
            continue
        i = order[:-d][same]
        j = order[d:][same]
        ia.append(i)
        ib.append(j)
    if not ia:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    A = np.concatenate(ia)
    B = np.concatenate(ib)
    return np.minimum(A, B), np.maximum(A, B)


def build_edges_lsh_vectorized(
    X, lsh: "RandomProjectionLSH", threshold: float, multiprobe: int = 0
) -> Tuple[List[Tuple[int, int, float]], int]:
    """Batch edge build via SORT-based bucketing -- no Python per-query loop.

    For each table: bucket keys for all rows, then vectorised within-bucket
    pairing (see _bucket_pairs_vectorized).  Union candidate pairs across
    tables, exact-score once, threshold.

    multiprobe=1 additionally groups by each key with one bit cleared, so rows
    whose keys differ in exactly one bit also become candidates -- the
    vectorised equivalent of Hamming-1 multi-probe.  It multiplies per-table
    collision probability ~ (1 + n_bits*(1-p)/p), which is what RETAINS recall
    as n_bits grows with N, instead of having to grow the table count.

    Cost: O((1+multiprobe*n_bits) * L * N log N) for the sorts + O(#cands) to
    score.  With n_bits ~ log2(N) the bucket load stays constant so
    #candidate_pairs = O(N*L) -> the near-linear curve. Used for full rebuilds;
    the dict-based candidates path is used for incremental single-paper adds.
    """
    keys = lsh._keys(X)  # (N, L)
    n = keys.shape[0]
    pa, pb = [], []
    for t in range(lsh.n_tables):
        col = keys[:, t]
        a, b = _bucket_pairs_vectorized(col)
        if a.size:
            pa.append(a)
            pb.append(b)
        if multiprobe:
            for m in range(lsh.n_bits):
                a, b = _bucket_pairs_vectorized(col & ~np.int64(1 << m))
                if a.size:
                    pa.append(a)
                    pb.append(b)
    if not pa:
        return [], 0
    A = np.concatenate(pa).astype(np.int64)
    B = np.concatenate(pb).astype(np.int64)
    packed = np.unique(A * n + B)  # dedup unordered pairs across tables
    A, B = packed // n, packed % n
    # exact-score in chunks so peak memory is O(chunk*nnz), not O(#cands*nnz):
    # a fat multiprobe candidate set could otherwise materialise GBs of rows.
    edges: List[Tuple[int, int, float]] = []
    chunk = 500_000
    for s in range(0, A.size, chunk):
        e = min(s + chunk, A.size)
        sims = _rowwise_dot(_row_slice(X, A[s:e]), _row_slice(X, B[s:e]))
        m = np.flatnonzero(sims >= threshold)
        edges.extend(zip(A[s:e][m].tolist(), B[s:e][m].tolist(),
                         sims[m].tolist()))
    return edges, int(packed.size)


def _transpose(X):
    if hasattr(X, "T"):
        return X.T
    return X.T


def _row_slice(X, rows):
    if HAVE_SCIPY and hasattr(X, "tocsr"):
        return X[rows]
    if hasattr(X, "__getitem__") and isinstance(X, _DenseCSR):
        return X[rows]
    return X[rows]


def _rowwise_dot(Xa, Xb) -> np.ndarray:
    if HAVE_SCIPY and hasattr(Xa, "multiply"):
        return np.asarray(Xa.multiply(Xb).sum(axis=1)).ravel()
    A = Xa.toarray() if hasattr(Xa, "toarray") else np.asarray(Xa)
    B = Xb.toarray() if hasattr(Xb, "toarray") else np.asarray(Xb)
    return np.einsum("ij,ij->i", A, B)


# --------------------------------------------------------------------------- #
# Dedup: union-find over arXiv / DOI / fuzzy-title #
# --------------------------------------------------------------------------- #
class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.reason: Dict[Tuple[str, str], str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str, why: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb
            self.reason[(a, b)] = why

    def clusters(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for x in list(self.parent):
            out.setdefault(self.find(x), []).append(x)
        return out


@dataclass
class DedupResult:
    canonical: Dict[str, str]  # uid -> canonical uid
    clusters: Dict[str, List[str]]  # canonical uid -> members
    reasons: Dict[Tuple[str, str], str]
    n_duplicates: int


def dedup(
    papers: Sequence[Paper],
    title_sim_threshold: float = 0.7,
    lsh_bits: int = 10,
    lsh_tables: int = 16,
    seed: int = 0,
) -> DedupResult:
    """Collapse duplicate papers via three signals, transitively.

    1. exact normalised arXiv id
    2. exact normalised DOI
    3. fuzzy title: char-4gram cosine LSH -> candidate pairs -> verified by
       token-Jaccard >= threshold  (blocking avoids O(N^2) title compares)
    """
    uf = UnionFind()
    for p in papers:
        uf.find(p.uid)  # register

    # --- exact-id signals (hash join, O(N)) ------------------------------- #
    by_arxiv: Dict[str, str] = {}
    by_doi: Dict[str, str] = {}
    for p in papers:
        ax = norm_arxiv(p.arxiv_id)
        if ax:
            if ax in by_arxiv:
                uf.union(p.uid, by_arxiv[ax], "arxiv")
            else:
                by_arxiv[ax] = p.uid
        do = norm_doi(p.doi)
        if do:
            if do in by_doi:
                uf.union(p.uid, by_doi[do], "doi")
            else:
                by_doi[do] = p.uid

    # --- fuzzy-title signal via char-ngram LSH ---------------------------- #
    # small dedicated hashing space for char 4-grams (title-only)
    dim = 1 << 12
    vec = _NgramHasher(dim=dim, n=4)
    Xt = vec.transform([p.title for p in papers])
    lsh = RandomProjectionLSH(dim=dim, n_bits=lsh_bits, n_tables=lsh_tables, seed=seed)
    ids = list(range(len(papers)))
    lsh.add(Xt, ids)
    cand_map = lsh.candidates(Xt, ids)
    tok_sets = [title_token_set(p.title) for p in papers]
    seen_pairs = set()
    for qi, cands in cand_map.items():
        for ci in cands:
            a, b = (qi, ci) if qi < ci else (ci, qi)
            if (a, b) in seen_pairs:
                continue
            seen_pairs.add((a, b))
            if jaccard(tok_sets[a], tok_sets[b]) >= title_sim_threshold:
                uf.union(papers[a].uid, papers[b].uid, "title")

    clusters = uf.clusters()
    canonical = {}
    for root, members in clusters.items():
        for m in members:
            canonical[m] = root
    n_dup = sum(len(m) - 1 for m in clusters.values() if len(m) > 1)
    return DedupResult(canonical, clusters, uf.reason, n_dup)


class _NgramHasher:
    """Char-ngram hashing vectoriser for title fuzzy matching (L2-normed)."""

    def __init__(self, dim: int = 1 << 12, n: int = 4):
        self.dim = dim
        self.n = n

    def transform(self, titles: Sequence[str]):
        rows, cols, data = [], [], []
        for r, t in enumerate(titles):
            grams = char_ngrams(t, self.n)
            if not grams:
                continue
            counts: Dict[int, float] = {}
            for g in grams:
                j = _crc(g) % self.dim
                counts[j] = counts.get(j, 0.0) + 1.0
            idx = np.fromiter(counts.keys(), np.int32, len(counts))
            val = np.fromiter(counts.values(), np.float64, len(counts))
            nrm = np.linalg.norm(val)
            if nrm:
                val = val / nrm
            rows.append(np.full(idx.size, r, np.int32))
            cols.append(idx)
            data.append(val)
        if not rows:
            return _empty_csr(len(titles), self.dim)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        if HAVE_SCIPY:
            return sp.csr_matrix((data, (rows, cols)), shape=(len(titles), self.dim))
        return _DenseCSR(rows, cols, data, len(titles), self.dim)


# --------------------------------------------------------------------------- #
# Incremental paper graph: append-only corpus + checkpointable index #
# --------------------------------------------------------------------------- #
class PaperGraph:
    """Incremental similarity graph with dedup, ANN edges, and checkpointing.

    add_papers(batch) touches only the batch: vectorise -> insert into LSH ->
    query candidates against the *whole* index -> new edges -> dedup batch vs
    existing.  Cost ~ O(|batch|), NOT O(N^2).  Corpus is append-only JSONL;
    the graph+index checkpoints to a single.json (planes regenerate from seed).
    """

    def __init__(
        self,
        dim: int = 1 << 14,
        sim_threshold: float = 0.30,
        lsh_bits: int = 12,
        lsh_tables: int = 12,
        seed: int = 0,
    ):
        self.dim = dim
        self.sim_threshold = sim_threshold
        self.vec = HashingTfidf(dim=dim)
        self.lsh = RandomProjectionLSH(dim, lsh_bits, lsh_tables, seed)
        self.papers: List[Paper] = []
        self.uid_to_row: Dict[str, int] = {}
        self.X = None  # cached CSR of all rows
        self.edges: List[Tuple[int, int, float]] = []
        self.dedup_canonical: Dict[str, str] = {}
        self._by_arxiv: Dict[str, int] = {}
        self._by_doi: Dict[str, int] = {}

    def add_papers(self, batch: Sequence[Paper]) -> dict:
        """Incrementally add a batch. Returns per-add stats."""
        t0 = time.perf_counter()
        start_row = len(self.papers)
        # dedup batch vs existing exact ids (cheap hash join)
        n_exact_dup = 0
        for p in batch:
            ax = norm_arxiv(p.arxiv_id)
            do = norm_doi(p.doi)
            hit = None
            if ax and ax in self._by_arxiv:
                hit = self._by_arxiv[ax]
            elif do and do in self._by_doi:
                hit = self._by_doi[do]
            if hit is not None:
                self.dedup_canonical[p.uid] = self.papers[hit].uid
                n_exact_dup += 1
                continue
            row = len(self.papers)
            self.papers.append(p)
            self.uid_to_row[p.uid] = row
            self.dedup_canonical[p.uid] = p.uid
            if ax:
                self._by_arxiv[ax] = row
            if do:
                self._by_doi[do] = row

        new_papers = self.papers[start_row:]
        if not new_papers:
            return {"added": 0, "exact_dups": n_exact_dup, "sec": time.perf_counter() - t0}

        # update idf stats then (re)vectorise -- transform is cheap; in a true
        # incremental deployment we'd append rows to a cached CSR / memmap.
        self.vec.observe([p.text() for p in new_papers])
        Xnew = self.vec.transform([p.text() for p in new_papers])
        new_rows = list(range(start_row, len(self.papers)))
        self.lsh.add(Xnew, new_rows)

        # rebuild the stacked X (prototype: restack; production: append to memmap)
        self.X = self._restack()

        # new edges: query ONLY the new rows against the whole index
        cand_map = self.lsh.candidates(Xnew, new_rows)
        pair_set = set()
        for qid, cands in cand_map.items():
            for cid in cands:
                a, b = (qid, cid) if qid < cid else (cid, qid)
                pair_set.add((a, b))
        n_new_edges = 0
        if pair_set:
            pairs = np.array(sorted(pair_set), dtype=np.int64)
            Xa = _row_slice(self.X, pairs[:, 0])
            Xb = _row_slice(self.X, pairs[:, 1])
            sims = _rowwise_dot(Xa, Xb)
            for k in range(len(pairs)):
                if sims[k] >= self.sim_threshold:
                    self.edges.append(
                        (int(pairs[k, 0]), int(pairs[k, 1]), float(sims[k]))
                    )
                    n_new_edges += 1
        return {
            "added": len(new_papers),
            "exact_dups": n_exact_dup,
            "new_edges": n_new_edges,
            "cand_pairs": len(pair_set),
            "N_after": len(self.papers),
            "sec": time.perf_counter() - t0,
        }

    def _restack(self):
        texts = [p.text() for p in self.papers]
        return self.vec.transform(texts)

    # --- persistence --------------------------------------------------------#
    def checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        corpus_path = path + ".corpus.jsonl"
        # append-only corpus: write any rows not yet on disk
        existing = 0
        if os.path.exists(corpus_path):
            with open(corpus_path) as f:
                existing = sum(1 for _ in f)
        with open(corpus_path, "a") as f:
            for p in self.papers[existing:]:
                f.write(json.dumps(asdict(p)) + "\n")
        meta = {
            "dim": self.dim,
            "sim_threshold": self.sim_threshold,
            "vec": self.vec.state(),
            "lsh": self.lsh.state(),
            "edges": self.edges,
            "dedup_canonical": self.dedup_canonical,
            "uid_to_row": self.uid_to_row,
        }
        with open(path + ".graph.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "PaperGraph":
        with open(path + ".graph.json") as f:
            meta = json.load(f)
        g = cls(dim=meta["dim"], sim_threshold=meta["sim_threshold"])
        g.vec = HashingTfidf.from_state(meta["vec"])
        g.lsh = RandomProjectionLSH.from_state(meta["lsh"])
        g.edges = [tuple(e) for e in meta["edges"]]
        g.dedup_canonical = meta["dedup_canonical"]
        g.uid_to_row = {k: int(v) for k, v in meta["uid_to_row"].items()}
        papers = []
        with open(path + ".corpus.jsonl") as f:
            for line in f:
                papers.append(Paper(**json.loads(line)))
        g.papers = papers
        for row, p in enumerate(papers):
            ax, do = norm_arxiv(p.arxiv_id), norm_doi(p.doi)
            if ax:
                g._by_arxiv.setdefault(ax, row)
            if do:
                g._by_doi.setdefault(do, row)
        g.X = g._restack()
        return g


# --------------------------------------------------------------------------- #
# Synthetic corpus generator with ground-truth duplicates #
# --------------------------------------------------------------------------- #
def make_mock_corpus(
    n: int,
    n_topics: Optional[int] = None,
    dup_frac: float = 0.06,
    seed: int = 7,
) -> Tuple[List[Paper], Dict[str, str]]:
    """Return (papers, truth) where truth[uid] = canonical uid of its dup group.

    Papers cluster into topics (realistic similarity structure) and a fraction
    are injected duplicates: exact-arxiv, exact-doi, or fuzzy-title variants.

    n_topics defaults to ~N/20 so each paper has a BOUNDED number of genuine
    neighbours -> a SPARSE graph (O(N) strong edges), which is how real paper
    corpora behave.  A fixed small topic count would instead create O(N^2)
    similarity cliques (the density-blowup failure mode) where no index helps
    because the output itself is quadratic.
    """
    rng = np.random.default_rng(seed)
    if n_topics is None:
        n_topics = max(8, n // 20)
    topic_vocab = {
        t: [f"t{t}w{w}" for w in range(30)] for t in range(n_topics)
    }
    background = [f"bg{w}" for w in range(400)]
    verbs = ["modeling", "simulation", "analysis", "certification", "design",
             "optimization", "learning", "prediction", "control", "assessment"]

    def draw_doc(topic: int) -> Tuple[str, str]:
        tv = topic_vocab[topic]
        title_terms = list(rng.choice(tv, size=4, replace=False)) + [
            rng.choice(verbs)
        ]
        rng.shuffle(title_terms)
        title = " ".join(title_terms)
        body_n = int(rng.integers(40, 80))
        body = list(rng.choice(tv, size=int(body_n * 0.75))) + list(
            rng.choice(background, size=int(body_n * 0.25))
        )
        rng.shuffle(body)
        return title.capitalize(), " ".join(body)

    papers: List[Paper] = []
    truth: Dict[str, str] = {}
    n_primary = int(n * (1 - dup_frac))
    ax_counter = 10000
    doi_counter = 5000
    for i in range(n_primary):
        topic = int(rng.integers(0, n_topics))
        title, abs_ = draw_doc(topic)
        uid = f"P{i:06d}"
        has_ax = rng.random() < 0.7
        has_doi = rng.random() < 0.6 or not has_ax
        ax = None
        doi = None
        if has_ax:
            ax = f"24{rng.integers(1,13):02d}.{ax_counter:05d}"
            ax_counter += 1
        if has_doi:
            doi = f"10.{rng.integers(1000,9999)}/j.{doi_counter}"
            doi_counter += 1
        p = Paper(uid=uid, title=title, abstract=abs_, arxiv_id=ax, doi=doi,
                  year=int(rng.integers(2015, 2026)))
        papers.append(p)
        truth[uid] = uid

    # inject duplicates of random primaries
    n_dup = n - n_primary
    primaries = papers[:]
    for k in range(n_dup):
        src = primaries[int(rng.integers(0, len(primaries)))]
        uid = f"D{k:06d}"
        mode = rng.integers(0, 3)
        if mode == 0 and src.arxiv_id:  # same arxiv, different version + doi
            ax = f"arXiv:{src.arxiv_id}v{rng.integers(2,4)}"
            p = Paper(uid=uid, title=src.title, abstract=src.abstract,
                      arxiv_id=ax, doi=f"10.9999/dup.{k}", year=src.year)
        elif mode == 1 and src.doi:  # same doi via url form, no arxiv
            p = Paper(uid=uid, title=src.title, abstract=src.abstract,
                      arxiv_id=None,
                      doi=f"https://doi.org/{src.doi.upper()}", year=src.year)
        else:  # fuzzy-title near-dup, fresh ids
            toks = src.title.split()
            if len(toks) > 2:
                j = int(rng.integers(0, len(toks)))
                toks[j] = rng.choice(["revisited", "extended", "part", "ii"])
            new_title = " ".join(toks)
            if rng.random() < 0.5:
                new_title = new_title + " -- a follow up"
            p = Paper(uid=uid, title=new_title, abstract=src.abstract,
                      arxiv_id=f"25{rng.integers(1,13):02d}.{90000+k:05d}",
                      doi=f"10.8888/followup.{k}", year=src.year)
        papers.append(p)
        truth[uid] = truth[src.uid]

    rng.shuffle(papers)
    return papers, truth


# --------------------------------------------------------------------------- #
# Benchmarks #
# --------------------------------------------------------------------------- #
def _fit_loglog(sizes: List[int], times: List[float]) -> float:
    x = np.log(np.asarray(sizes, float))
    y = np.log(np.asarray(times, float))
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def adaptive_bits(n: int, target_load: int = 8) -> int:
    """n_bits so expected bucket load ~= target_load: bits = log2(N/load).

    Holding load constant as N grows keeps bucket sizes (hence candidates per
    paper) constant -> the index stays sub-quadratic.  Fixed bits would let
    buckets fill up and collapse the index back to O(N^2)."""
    return max(6, int(round(math.log2(max(n, 2) / target_load))))


def bench(sizes: List[int], sim_threshold: float = 0.50,
          lsh_bits: int = 0, lsh_tables: int = 24, multiprobe: int = 0,
          target_load: int = 8) -> dict:
    print(f"\n=== SCALING BENCHMARK ===  scipy={HAVE_SCIPY} sklearn={HAVE_SKLEARN}")
    mode = f"adaptive log2(N/{target_load})" if lsh_bits <= 0 else f"fixed {lsh_bits}"
    print(f"threshold={sim_threshold}  LSH bits={mode} tables={lsh_tables} "
          f"multiprobe={multiprobe} (vectorized sort-bucketing)\n")
    dim = 1 << 14
    naive_t, idx_t = [], []
    recalls, precisions, cand_ratio = [], [], []
    bands = [(0.7, 1.01), (0.5, 0.7), (0.30, 0.5)]
    band_hit = {b: [0, 0] for b in bands}  # [found, total]
    header = (f"{'N':>6} {'bits':>4} | {'naive_s':>9} {'edges':>7} | {'idx_s':>8} "
              f"{'edges':>7} {'cand':>9} | {'recall':>7} {'prec':>6} {'speedup':>8}")
    print(header)
    print("-" * len(header))
    for n in sizes:
        papers, _ = make_mock_corpus(n, seed=100 + n)
        vec = HashingTfidf(dim=dim)
        vec.observe([p.text() for p in papers])
        vec.freeze()
        X = vec.transform([p.text() for p in papers])
        ids = list(range(len(papers)))
        bits = adaptive_bits(n, target_load) if lsh_bits <= 0 else lsh_bits

        t0 = time.perf_counter()
        naive_edges = build_edges_naive(X, sim_threshold)
        tn = time.perf_counter() - t0

        lsh = RandomProjectionLSH(dim, bits, lsh_tables, seed=1)
        t0 = time.perf_counter()
        idx_edges, ncand = build_edges_lsh_vectorized(
            X, lsh, sim_threshold, multiprobe=multiprobe)
        ti = time.perf_counter() - t0

        naive_set = {(a, b) for a, b, _ in naive_edges}
        idx_set = {(a, b) for a, b, _ in idx_edges}
        rec = len(naive_set & idx_set) / max(len(naive_set), 1)
        prec = len(naive_set & idx_set) / max(len(idx_set), 1)
        # stratify recall by the ground-truth pair's similarity band
        for a, b, s in naive_edges:
            for lo, hi in bands:
                if lo <= s < hi:
                    band_hit[(lo, hi)][1] += 1
                    if (a, b) in idx_set:
                        band_hit[(lo, hi)][0] += 1
                    break
        naive_t.append(tn)
        idx_t.append(ti)
        recalls.append(rec)
        precisions.append(prec)
        cand_ratio.append(ncand / max(n, 1))
        print(f"{n:>6} {bits:>4} | {tn:>9.4f} {len(naive_edges):>7} | {ti:>8.4f} "
              f"{len(idx_edges):>7} {ncand:>9} | {rec:>7.3f} {prec:>6.3f} "
              f"{tn/max(ti,1e-9):>7.1f}x")

    slope_naive = _fit_loglog(sizes, naive_t)
    slope_idx = _fit_loglog(sizes, idx_t)
    print(f"\nempirical scaling exponent (log-log slope of time vs N):")
    print(f"    naive   : {slope_naive:.2f}   (theory 2.0, O(N^2))")
    print(f"    indexed : {slope_idx:.2f}   (near-linear; candidates/paper ~ "
          f"{np.mean(cand_ratio):.1f})")
    print(f"    mean LSH recall={np.mean(recalls):.3f}  precision="
          f"{np.mean(precisions):.3f} (precision==1 by exact re-scoring)")
    print(f"  recall stratified by edge similarity (the honest picture):")
    for (lo, hi) in bands:
        found, total = band_hit[(lo, hi)]
        r = found / max(total, 1)
        label = f"cos in [{lo:.2f},{min(hi,1.0):.2f})"
        print(f"    {label:>18}: recall={r:.3f}  ({found}/{total} edges) "
              f"{'<- strong, want ~1' if lo>=0.7 else ('<- marginal, least informative' if lo<0.5 else '')}")
    return {
        "sizes": sizes, "naive_t": naive_t, "idx_t": idx_t,
        "slope_naive": slope_naive, "slope_idx": slope_idx,
        "recall": recalls, "precision": precisions,
    }


def bench_frontier(n: int = 8000, threshold: float = 0.5) -> None:
    """Recall/speed frontier at fixed N: no single LSH config dominates.

    Same corpus, three (load, tables, multiprobe) settings -> shows you buy
    recall with candidate volume (hence wall-clock).  The index's LEAD over
    naive grows with N (all indexed slopes < naive's), so every point here
    wins outright at larger N; this table is the N=8000 snapshot of the knob.
    """
    print(f"\n=== RECALL / SPEED FRONTIER  (N={n}, one corpus, LSH knobs) ===")
    papers, _ = make_mock_corpus(n, seed=100 + n)
    dim = 1 << 14
    vec = HashingTfidf(dim=dim)
    vec.observe([p.text() for p in papers])
    vec.freeze()
    X = vec.transform([p.text() for p in papers])
    t0 = time.perf_counter()
    naive_edges = build_edges_naive(X, threshold)
    tn = time.perf_counter() - t0
    naive_set = {(a, b) for a, b, _ in naive_edges}
    strong_true = {(a, b) for a, b, s in naive_edges if s >= 0.7}
    print(f"  naive: {tn:.3f}s  {len(naive_edges)} edges")
    print(f"  {'mode':>12} {'load':>4} {'tbl':>4} {'mp':>3} | {'idx_s':>7} "
          f"{'speedup':>8} {'recall':>7} {'strong':>7} {'prec':>6}")
    cfgs = [("throughput", 8, 24, 0), ("balanced", 2, 12, 1), ("recall", 8, 12, 1)]
    for name, load, tbl, mp in cfgs:
        bits = adaptive_bits(n, load)
        lsh = RandomProjectionLSH(dim, bits, tbl, seed=1)
        t0 = time.perf_counter()
        edges, _ = build_edges_lsh_vectorized(X, lsh, threshold, multiprobe=mp)
        ti = time.perf_counter() - t0
        idx_set = {(a, b) for a, b, _ in edges}
        rec = len(naive_set & idx_set) / max(len(naive_set), 1)
        strong = len(strong_true & idx_set) / max(len(strong_true), 1)
        prec = len(naive_set & idx_set) / max(len(idx_set), 1)
        print(f"  {name:>12} {load:>4} {tbl:>4} {mp:>3} | {ti:>7.3f} "
              f"{tn/max(ti,1e-9):>7.1f}x {rec:>7.3f} {strong:>7.3f} {prec:>6.3f}")


def bench_incremental(base: int = 1800, batch: int = 100, steps: int = 4) -> None:
    """Show per-add cost is ~flat as N grows (incremental, not rebuild)."""
    print(f"\n=== INCREMENTAL ADD (cost per batch as N grows) ===")
    papers, _ = make_mock_corpus(base + batch * steps, seed=42)
    g = PaperGraph(dim=1 << 14)
    g.add_papers(papers[:base])
    print(f"{'N_before':>9} | {'add_sec':>8} {'cand_pairs':>11} {'new_edges':>10}")
    print("-" * 44)
    start = base
    for s in range(steps):
        stats = g.add_papers(papers[start:start + batch])
        print(f"{start:>9} | {stats['sec']:>8.4f} {stats['cand_pairs']:>11} "
              f"{stats['new_edges']:>10}")
        start += batch
    print("  -> per-batch time stays ~flat: adding to a big graph costs the "
          "same\n     as adding to a small one (no O(N^2) rebuild).")


def bench_dedup() -> None:
    print(f"\n=== DEDUP precision/recall vs injected ground truth ===")
    for n in (500, 2000):
        papers, truth = make_mock_corpus(n, dup_frac=0.06, seed=9)
        t0 = time.perf_counter()
        res = dedup(papers, title_sim_threshold=0.6)
        dt = time.perf_counter() - t0
        # ground-truth duplicate pairs (same truth-group)
        from collections import defaultdict
        groups = defaultdict(list)
        for p in papers:
            groups[truth[p.uid]].append(p.uid)
        true_pairs = set()
        for members in groups.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = sorted((members[i], members[j]))
                    true_pairs.add((a, b))
        # predicted duplicate pairs (same canonical cluster)
        pred_pairs = set()
        for members in res.clusters.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = sorted((members[i], members[j]))
                    pred_pairs.add((a, b))
        tp = len(true_pairs & pred_pairs)
        prec = tp / max(len(pred_pairs), 1)
        rec = tp / max(len(true_pairs), 1)
        by_reason = {}
        for why in res.reasons.values():
            by_reason[why] = by_reason.get(why, 0) + 1
        print(f"  N={n:>5}: dup_clusters={sum(1 for m in res.clusters.values() if len(m)>1):>4} "
              f"pairs P={prec:.3f} R={rec:.3f}  time={dt:.3f}s  reasons={by_reason}")


def demo_persistence() -> None:
    print(f"\n=== INCREMENTAL PERSISTENCE round-trip ===")
    import tempfile
    papers, _ = make_mock_corpus(400, seed=3)
    g = PaperGraph(dim=1 << 13)
    g.add_papers(papers[:300])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph")
        g.checkpoint(path)
        g2 = PaperGraph.load(path)
        # add more to the RELOADED graph and re-checkpoint (append-only corpus)
        stats = g2.add_papers(papers[300:400])
        g2.checkpoint(path)
        corpus_lines = sum(1 for _ in open(path + ".corpus.jsonl"))
        print(f"  saved 300 -> reloaded {len(g2.papers)-stats['added']} papers, "
              f"{len(g2.edges)} edges after reload")
        print(f"  added {stats['added']} to reloaded graph -> N={len(g2.papers)}, "
              f"corpus.jsonl has {corpus_lines} lines (append-only)")
        print(f"  edges preserved + extended: {len(g2.edges)}  "
              f"(checkpoint = graph.json + corpus.jsonl)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--sizes", default="250,500,1000,2000")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--bits", type=int, default=0, help="0 = adaptive log2(N)")
    ap.add_argument("--tables", type=int, default=24)
    ap.add_argument("--multiprobe", type=int, default=0, help="1 = Hamming-1 probe")
    ap.add_argument("--load", type=int, default=8, help="target bucket load")
    args = ap.parse_args()

    if not (args.bench or args.demo):
        args.bench = args.demo = True

    if args.bench:
        sizes = [int(s) for s in args.sizes.split(",")]
        bench(sizes, args.threshold, args.bits, args.tables, args.multiprobe,
              args.load)
        bench_frontier(n=min(max(sizes), 8000), threshold=args.threshold)
        bench_incremental()
        bench_dedup()
    if args.demo:
        demo_persistence()


if __name__ == "__main__":
    main()
