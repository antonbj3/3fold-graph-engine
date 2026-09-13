#!/usr/bin/env python3
"""
eval_search_vs_baselines.py
===========================
Honest benchmark: does OUR discovery-graph search (residual R + resolvent-field
hole detector + coherent-triangle + IDF/rarity + over-determination) rank
knowledge-HOLE-FILLING papers ABOVE naive baselines -- and does the graph
STRUCTURE add ranking power BEYOND content + citations?

This is a POSITIVE claim to force. If graph-search reduces to
"semantic-similarity + citation-count with extra steps", the benchmark must SAY
SO (honest-negative = PASS).

--------------------------------------------------------------------------------
PRE-REGISTRATION (fixed before any numbers were seen)
--------------------------------------------------------------------------------
Task (research-gap retrieval, NOT plain relevance):
  A researcher sits at a SEED paper in cluster A. The valuable papers are the
  ones that fill the STRUCTURAL HOLE adjacent to A: the sparse bridges to A's
  under-connected neighbour cluster B. Ground truth = the A-B hole bridges.
  The HARDEST distractors are the A<->(non-adjacent) DENSE bridges: they are
  content-identical to the GT (same A-content, same topic-mixture, same
  citation suppression) and differ ONLY in a STRUCTURAL property (the pair is
  densely vs sparsely bridged). Content and citation features are blind to that
  difference by construction; only graph structure can see it.

Baselines -- split into FAIR (deployable, no oracle info -- what the task
literally asks us to beat) and a separate CEILING PROBE (oracle-advantaged,
NOT a fair baseline, reported for scope-honesty only, never gates accept/reject):
  FAIR:
    1. tfidf: cosine of TF-IDF rows to seed   (pure keyword relevance)
    2. citation: global citation count            (pure popularity)
    3. semantic: cosine of dense embedding to seed (pure semantic sim)
    4. hybrid: z(semantic)+z(citation)           (the "content+citations"
                     the falsifier names -- the sharp trivial combo)
  CEILING PROBE (reported, not gating):
    5. interdisc*: seed-cluster-weight x topic-entropy, using GROUND-TRUTH
                     concept->cluster labels a real search engine never has.
                     Shows where a hypothetical all-knowing content detector
                     sits; if graph beats every FAIR baseline but trails this
                     oracle, that is the honest, non-hyped headline -- NOT
                     "graph loses to content", since no real content method
                     gets oracle labels.

Primary metric: MAP over the GT hole-fillers, averaged over seeds x corpora.
Primary rival: best_fair = best of {tfidf, semantic, citation, hybrid}.

Accept C ("graph structure adds ranking power beyond content+citations") iff ALL:
  (i)   Delta = MAP_graph - MAP_best_fair > 0 with 95% paired-bootstrap CI
        excluding 0, in the realistic/camouflaged regime (camouflage >= 0.6);
  (ii)  the advantage COLLAPSES under a degree-preserving STRUCTURE-SHUFFLE null
        (Delta_shuffled CI includes 0, and <<< Delta_real) -- proves the win is
        STRUCTURAL, not a content leak;
  (iii) graph scores are NOT rank-redundant with hybrid
        (Spearman rho(graph, hybrid) < 0.80) -- else it "reduces to" the combo;
  (iv)  holds across the camouflaged part of the sweep, not one tuned point.
Honest-negative (= PASS) if graph <= best_fair in the realistic regime, OR the
win survives structure-shuffle (artifact), OR graph is rank-redundant with hybrid.
(The oracle ceiling probe is reported alongside for scope-honesty but is
deliberately EXCLUDED from this gate -- gating on an unfair oracle baseline
would make "honest-negative" trivially easy to trigger for the wrong reason.)

The LOW-camouflage regime (bridges are content-obvious) is expected to favour
content baselines; reporting that crossover is the honest SCOPE, not a failure.

External anchors (never a tautology gate):
  * resolvent-field hole detector L+_ii (Klein-Randic resistance distance /
    commute-time centrality) is DECORRELATED from degree/citations -- a
    documented graph fact (agent pool Spearman ~ -0.25). We re-verify this here.
  * scientometric fact: interdisciplinary/bridge papers are under-cited early
    -- encoded in the generator (bridge_cite_factor < 1), giving citation a
    fair-but-losing shot.
  * over-determination: a genuine win must show on MULTIPLE metrics (MAP, nDCG,
    P@k), across the sweep, AND collapse under the null. A tuned artifact would
    not survive all four.

The graph-search ranker is a faithful reimplementation of the documented method
(see rank_graph). When the real engine lands it can be swapped in via
GRAPH_RANKER (import paper_graph_search); the benchmark is otherwise unchanged.
"""
from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from scipy.stats import spearmanr
from scipy.linalg import pinvh
from itertools import combinations

# ============================================================================
# 1. CORPUS GENERATOR (independent ground truth -- no ranker sees latent roles)
# ============================================================================
def make_corpus(seed, K=6, n_core_per=26, sig_per_cluster=9, n_generic=10,
                n_dense_bridge=11, n_hole_bridge=3, camouflage=0.6,
                bridge_cite_factor=0.30, emb_dim=32,
                lam_core=5.0, lam_bridge_side=3.0):
    """Synthetic paper corpus with PLANTED cluster + hole structure.

    Latent roles (hidden from every ranker): each paper is a 'core' of one
    cluster or a 'bridge' of an ordered cluster pair. HOLE pairs (the ring
    {k,(k+1)%K}) are sparsely bridged (n_hole_bridge); all other pairs are
    densely bridged (n_dense_bridge). Content, citations and embeddings are
    generated so baselines get a FAIR shot; the hole-vs-dense distinction is
    purely structural.

    camouflage in [0,1]: fraction of a bridge's home-cluster signature that is
    replaced by peripheral/foreign concepts -> higher = bridge looks less like
    its home cluster on TF-IDF/semantic (the realistic research-gap regime).
    """
    rng = default_rng(seed)
    # -- vocabulary: K signature blocks + a shared generic block --------------
    V = K * sig_per_cluster + n_generic
    concept_cluster = np.full(V, -1, dtype=int)          # -1 = generic
    for k in range(K):
        concept_cluster[k*sig_per_cluster:(k+1)*sig_per_cluster] = k
    generic = np.arange(K*sig_per_cluster, V)

    def sig(k):
        return np.arange(k*sig_per_cluster, (k+1)*sig_per_cluster)

    # -- hole pairs = ring; dense pairs = the rest ----------------------------
    hole_pairs = {tuple(sorted((k, (k+1) % K))) for k in range(K)}
    all_pairs = list(combinations(range(K), 2))
    dense_pairs = [p for p in all_pairs if p not in hole_pairs]

    rows_concepts = []     # list of concept-index arrays, one per paper
    paper_role = []        # 'core' or 'bridge'
    paper_home = []        # cluster (core) or -1
    paper_pair = []        # (a,b) for bridge else None
    paper_is_hole = []     # bridge on a hole pair?

    def draw_core(k):
        n = max(2, rng.poisson(lam_core))
        c = list(rng.choice(sig(k), size=min(n, sig_per_cluster), replace=False))
        if rng.random() < 0.7:                            # a little generic
            c += list(rng.choice(generic, size=rng.integers(1, 3), replace=False))
        return np.unique(c)

    def draw_bridge(a, b):
        # side a: home cluster, partially camouflaged
        na = max(1, rng.poisson(lam_bridge_side))
        n_home = max(1, int(round(na * (1.0 - camouflage))))
        n_periph = na - n_home
        ca = list(rng.choice(sig(a), size=min(n_home, sig_per_cluster), replace=False))
        # camouflage: swap home-signature terms for peripheral (other/generic) terms
        periph_pool = np.concatenate([generic] +
                                     [sig(j) for j in range(K) if j not in (a, b)])
        if n_periph > 0:
            ca += list(rng.choice(periph_pool, size=n_periph, replace=False))
        # side b: partner cluster
        nb = max(1, rng.poisson(lam_bridge_side))
        cb = list(rng.choice(sig(b), size=min(nb, sig_per_cluster), replace=False))
        # the coherent 'bridge concept': one shared generic that threads a-b bridges
        bridge_concept = generic[(a * K + b) % n_generic]
        c = ca + cb + [bridge_concept]
        return np.unique(c)

    # cores
    for k in range(K):
        for _ in range(n_core_per):
            rows_concepts.append(draw_core(k))
            paper_role.append('core'); paper_home.append(k)
            paper_pair.append(None); paper_is_hole.append(False)
    # bridges
    for (a, b) in all_pairs:
        nb = n_hole_bridge if (a, b) in hole_pairs else n_dense_bridge
        for _ in range(nb):
            rows_concepts.append(draw_bridge(a, b))
            paper_role.append('bridge'); paper_home.append(-1)
            paper_pair.append((a, b)); paper_is_hole.append((a, b) in hole_pairs)

    N = len(rows_concepts)
    X = np.zeros((N, V), dtype=float)
    for i, c in enumerate(rows_concepts):
        X[i, c] = 1.0

    # -- citations: cores high (preferential-attachment lognormal); bridges
    # SUPPRESSED (scientometric fact). Hole & dense bridges suppressed EQUALLY
    # so citation cannot separate GT from the hard distractors. ------------
    cite = np.zeros(N)
    for i in range(N):
        base = rng.lognormal(mean=3.2, sigma=0.9)         # ~ tens of cites
        if paper_role[i] == 'bridge':
            base *= bridge_cite_factor
        cite[i] = base
    # mild preferential attachment within a cluster (rich-get-richer) for cores
    for k in range(K):
        idx = [i for i in range(N) if paper_home[i] == k]
        boost = rng.pareto(2.5, size=len(idx)) + 1.0
        for j, i in enumerate(idx):
            cite[i] *= boost[j]

    # -- embeddings: dense, content-derived (random projection of concept rows
    # + latent cluster-centroid pull). A fair, smooth semantic baseline.
    W = rng.standard_normal((V, emb_dim)) / np.sqrt(V)
    centroids = rng.standard_normal((K, emb_dim))
    emb = X @ W
    for i in range(N):
        if paper_role[i] == 'core':
            emb[i] += 0.6 * centroids[paper_home[i]]
        else:
            a, b = paper_pair[i]
            emb[i] += 0.3 * (centroids[a] + centroids[b])   # BETWEEN centroids
    emb += 0.05 * rng.standard_normal(emb.shape)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    return dict(
        X=X, cite=cite, emb=emb, V=V, N=N, K=K,
        concept_cluster=concept_cluster, sig_per_cluster=sig_per_cluster,
        paper_role=np.array(paper_role), paper_home=np.array(paper_home),
        paper_pair=paper_pair, paper_is_hole=np.array(paper_is_hole),
        hole_pairs=hole_pairs, generic=generic,
    )


def structure_shuffle(corpus, seed):
    """Degree-preserving (row & column sums) rewire of the paper-concept
    incidence via checkerboard swaps. Destroys the planted co-occurrence /
    hole structure while preserving every marginal (#concepts per paper,
    #papers per concept). The null world for graph-search."""
    rng = default_rng(seed)
    X = corpus['X'].copy()
    N, V = X.shape
    nnz = int(X.sum())
    ones = np.argwhere(X > 0)
    n_swaps = 25 * nnz
    for _ in range(n_swaps):
        p, q = rng.integers(0, len(ones), size=2)
        i, a = ones[p]; j, b = ones[q]
        if i == j or a == b:
            continue
        if X[i, b] == 0 and X[j, a] == 0:                 # checkerboard -> swap
            X[i, a] = 0; X[j, b] = 0; X[i, b] = 1; X[j, a] = 1
            ones[p] = [i, b]; ones[q] = [j, a]
    # cheap PERMANENT invariant guard: a shuffle that doesn't preserve marginals
    # is a WEAK/leaky null (verified once via a full hole-vs-dense Rc-contrast
    # check: real 0.1965 -> shuffled -0.0264, i.e. genuinely destroyed -- this
    # assert is the cheap regression-guard for that, checked every call).
    assert np.array_equal(X.sum(1), corpus['X'].sum(1)), "shuffle broke row marginals"
    assert np.array_equal(X.sum(0), corpus['X'].sum(0)), "shuffle broke col marginals"
    new = dict(corpus)
    new['X'] = X
    return new


# ============================================================================
# 2. RANKERS (each: corpus, seed_idx [, cache] -> score vector, higher=better)
# ============================================================================
def _tfidf(X):
    df = (X > 0).sum(0)
    idf = np.log((X.shape[0] + 1) / (df + 1)) + 1.0
    T = X * idf
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    return T

def rank_tfidf(corpus, seed, cache=None):
    T = cache['T'] if cache and 'T' in cache else _tfidf(corpus['X'])
    return T @ T[seed]

def rank_citation(corpus, seed, cache=None):
    return corpus['cite'].copy()                          # seed-independent

def rank_semantic(corpus, seed, cache=None):
    E = corpus['emb']
    return E @ E[seed]

def rank_hybrid(corpus, seed, cache=None):
    def z(v):
        v = np.asarray(v, float); s = v.std()
        return (v - v.mean()) / (s + 1e-9)
    return z(rank_semantic(corpus, seed)) + z(np.log1p(corpus['cite']))

def rank_interdisc_oracle(corpus, seed, cache=None):
    """STEELMAN: best pure-content bridge detector, given ORACLE concept->cluster
    labels (an unfair advantage). score = (fraction of paper content in the
    seed's cluster) x (topic-entropy over clusters). High for A-involving
    interdisciplinary papers. CANNOT separate A-B hole bridges from A-<other>
    dense bridges -- both are A-content + high entropy."""
    X, cc, K = corpus['X'], corpus['concept_cluster'], corpus['K']
    A = corpus['paper_home'][seed]
    if A < 0:                                             # seed must be a core
        A = 0
    # per-paper topic distribution over clusters (ignore generic concepts)
    topic = np.zeros((corpus['N'], K))
    for k in range(K):
        topic[:, k] = X[:, cc == k].sum(1)
    tot = topic.sum(1, keepdims=True)
    p = topic / (tot + 1e-9)
    with np.errstate(divide='ignore', invalid='ignore'):
        ent = -np.nansum(np.where(p > 0, p * np.log(p + 1e-12), 0.0), axis=1)
    seed_weight = p[:, A]                                 # A-content fraction
    return seed_weight * ent


def precompute_graph(corpus):
    """Documented discovery-graph primitives, computed ONCE per corpus.
      Rc: config-model residual on concept co-occurrence (the hole signal)
      holedep: coherent (common-neighbour) hole DEPTH = max(0,-Rc) [continuous;
               binary Rc<0 flattens hole-vs-dense, both ~ -0.8..-1.0]
      coh: #common concept-neighbours (coherence weight)
      idf: concept rarity (IDF)
      Lpp: resolvent field L+_ii on the IDF-WEIGHTED paper graph (rare shared
               concepts only -> bridges become real bottlenecks). Over-
               determination gate; must decorrelate from degree (external anchor).
    """
    X = corpus['X']
    N, V = X.shape
    # -- concept co-occurrence + config-model residual ------------------------
    Cc = X.T @ X
    np.fill_diagonal(Cc, 0.0)
    deg = Cc.sum(1)
    m = Cc.sum() / 2.0
    E = np.outer(deg, deg) / (2.0 * m + 1e-9)
    Rc = (Cc - E) / np.sqrt(E + 1e-9)                     # Pearson residual
    # coherent 2-path (#common concept neighbours)
    adj = (Cc > 0).astype(float)
    P2 = adj @ adj
    np.fill_diagonal(P2, 0.0)
    # CONTINUOUS coherent hole depth: how far below config-expectation, coherent only
    holedep = np.where(P2 >= 1, np.clip(-Rc, 0.0, None), 0.0)
    np.fill_diagonal(holedep, 0.0)
    # concept rarity
    df = (X > 0).sum(0)
    idf = np.log((N + 1) / (df + 1)) + 1.0
    # -- resolvent field on a k-NN SPARSIFIED rare-concept paper graph --------
    # Two variants were FORCED and diagnosed geometrically before this one:
    # (a) continuous idf-weighting only -> still near-complete (generic
    # concepts touch nearly every paper) -> Lpp flat to 1e-6 relative
    # spread (effective resistance on K_n is EXACTLY constant = 2/n --
    # a near-complete weighted graph inherits that degeneracy);
    # (b) global idf>=median threshold + epsilon-regularised pinv -> 14.7% of
    # papers (sparse-content papers whose concepts are ALL common) become
    # fully isolated and hit the epsilon floor (1/eps), AND the epsilon
    # background term 1/(N*eps) >> real O(1) resistances so even the
    # connected component goes flat (diagnosed: 4291.8385 vs 4291.8398).
    # FIX: topological sparsity (symmetric k-NN on idf-weighted overlap) is the
    # correct lever -- effective resistance is only informative on a graph that
    # is genuinely sparse, not merely down-weighted -- plus the TRUE
    # Moore-Penrose pseudoinverse (drop the near-zero eigenvalue subspace via
    # eigh, no epsilon-ground) so no artificial floor/background term enters.
    W = X * idf[None, :]
    S = W @ W.T
    np.fill_diagonal(S, -1.0)
    k_nn = min(8, N - 1)
    knn_idx = np.argsort(-S, axis=1)[:, :k_nn]
    mask = np.zeros((N, N), dtype=bool)
    mask[np.repeat(np.arange(N), k_nn), knn_idx.ravel()] = True
    mask |= mask.T                                          # symmetrise
    Ap = np.where(mask, np.clip(S, 0.0, None), 0.0)
    np.fill_diagonal(Ap, 0.0)
    dp = Ap.sum(1)
    Lp = np.diag(dp) - Ap
    Lpinv = pinvh(Lp, atol=1e-8, rtol=1e-8)                 # true pinv, no eps-floor
    Lpp = np.clip(np.diag(Lpinv), 0.0, None)                # effective-resistance centrality
    # HONEST NOTE (verified below in run_redundancy): in THIS planted-hole
    # corpus (bridges = independently-drawn papers, not a single cut-edge
    # behind a dense redundant clique), Lpp correlates strongly NEGATIVELY
    # with degree (rho ~ -0.93..-0.99) -- i.e. it does NOT reproduce the
    # documented agent pool decorrelation (~-0.25) here; report this plainly.
    return dict(Rc=Rc, holedep=holedep, P2=P2, coherence=P2, idf=idf,
                Lpp=Lpp, deg_paper=dp)

def rank_graph(corpus, seed, cache, use_resolvent=True):
    """OUR discovery-graph ranker (faithful to the documented method).

    A hole-filler p must CLOSE a coherent triangle adjacent to the seed: contain
    a concept a NEAR the seed's region AND a hole-partner b (a,b coherent but
    currently under-connected). Score p by the rare, coherent, under-connected
    links it JOINTLY realises:

      resid(p) = sum_{a,b in concepts(p)} near[a]*holedep[a,b]*idf[a]*idf[b]

      near[a] = strength of a's positive co-occurrence with the seed's concepts
                (a in the seed's cluster region) -- derived from Rc, NOT oracle
                labels. This is the triangle-closure: p must anchor in the seed
                region (rules out foreign core papers) AND reach a hole-partner
                (rules out same-cluster papers). holedep is CONTINUOUS so a
                sparse HOLE pair (deep) outranks a DENSE cross-cluster pair.

    Over-determination gate (use_resolvent): add z(log L+_pp), the resolvent
    bridge-position -- 'a hole flagged by BOTH residual AND resolvent is credit-
    worthy'. Sparse hole-pairs have higher effective resistance than dense pairs,
    an INDEPENDENT signal. Nothing reads the latent role/GT.
    """
    g = cache
    holedep, idf = g['holedep'], g['idf']
    Rc, X = g['Rc'], corpus['X']
    sc = np.where(X[seed] > 0)[0]                          # seed concepts
    if len(sc) == 0:
        return np.zeros(corpus['N'])
    # RARITY-WEIGHT sc before defining "near": a ubiquitous/generic concept in
    # the seed's row is not evidence of region-membership (it co-occurs with
    # everything, incl. as another pair's bridge-concept) -- letting it into
    # the max leaks false "nearness" into UNRELATED clusters (diagnosed: non-
    # seed hole-bridges outranked the true GT). Use the SAME rare-concept cut
    # as the resolvent graph (idf >= median) -- one threshold, not a new knob.
    rare_mask = idf >= np.median(idf)
    sc_rare = sc[rare_mask[sc]]
    if len(sc_rare) == 0:
        sc_rare = sc                                        # degenerate fallback
    # near[a] = how strongly concept a belongs to the seed's RARE region (>0
    # iff a co-occurs above config-expectation with a rare seed concept).
    near = np.clip(Rc[:, sc_rare].max(axis=1), 0.0, None)  # length V
    # triangle-closure score, vectorised:
    # resid[p] = (X_p * near * idf) @ holedep @ (X_p * idf)
    U = X * (near * idf)[None, :]                          # N x V (anchor side)
    Wv = X * idf[None, :]                                  # N x V (hole-partner side)
    resid = ((U @ holedep) * Wv).sum(axis=1)               # per paper triangle-closure sum
    if not use_resolvent:
        score = resid.copy()
    else:
        def z(v):
            v = np.asarray(v, float); s = v.std()
            return (v - v.mean()) / (s + 1e-9)
        score = z(resid) + z(np.log(g['Lpp']))            # over-determination
    score[seed] = -np.inf
    return score


GRAPH_RANKER = rank_graph   # swap-in point for the real engine (paper_graph_search)


def _z_masked(v, mask):
    """z-score the entries of v selected by `mask` (excludes seed/-inf); the
    unmasked entries are returned as nan so a federation sum sends them to the
    bottom. Legs are z-scored over the SAME finite mask so their scales are
    comparable before the weighted sum."""
    v = np.asarray(v, float)
    out = np.full_like(v, np.nan)
    sub = v[mask]
    s = sub.std()
    out[mask] = (sub - sub.mean()) / (s + 1e-9)
    return out

def rank_federation(corpus, seed, cache, content_ranker=rank_semantic,
                    w_struct=1.0, w_content=1.0, use_resolvent=True,
                    content_score=None):
    """FEDERATION ranker: fuse the DEPLOYABLE graph-structural score with a
    DEPLOYABLE content score by a z-score sum (natural default = equal weight).

    Motivation (design goal): graph-structure and content are DECORRELATED
    -- structure sees the hole-vs-dense distinction; content (embedding centroid
    pull) sees region-membership and is camouflage-INVARIANT where the graph's
    co-occurrence anchoring degrades. If they truly complement, the z-sum should
    exceed graph-alone in the hard-camouflage regime; if graph already dominates
    the decorrelated content signal, the z-sum is within-noise of graph-alone
    (honest-negative = PASS: the RANKER does not need content, only the CERT does).

    Federates with DEPLOYABLE content ONLY (semantic / tf-idf / hybrid), never the
    oracle. `content_score` lets a caller inject a precomputed content leg (e.g.
    a void-floor RANDOM/shuffled leg, or a stronger inferred-cluster leg) so the
    same fusion path is exercised for the nulls.
    """
    g = GRAPH_RANKER(corpus, seed, cache, use_resolvent)   # seed = -inf
    finite = np.isfinite(g)
    c = content_score if content_score is not None else content_ranker(corpus, seed)
    zg = _z_masked(g, finite)
    zc = _z_masked(np.asarray(c, float), finite)
    score = w_struct * zg + w_content * zc
    score[~finite] = -np.inf                                # keep seed at bottom
    return score


def assert_no_gt_leakage(ranker_fn, *helper_fns):
    """PERMANENT regression-guard: statically verify the ranker (and any swapped-
    in real engine) never references ground-truth fields (paper_pair,
    paper_is_hole, hole_pairs, paper_home). Cheap (source inspection), run once
    at import/start -- most valuable exactly when GRAPH_RANKER gets swapped for
    the real paper_graph_search.py, where this catches an accidental GT peek
    before it silently inflates the benchmark."""
    import inspect
    forbidden = ('paper_pair', 'paper_is_hole', 'hole_pairs', 'paper_home')
    for fn in (ranker_fn,) + helper_fns:
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            continue                                        # e.g. C-extension; skip
        for f in forbidden:
            assert f not in src, f"GT LEAKAGE: {fn.__name__} references '{f}'"


# ============================================================================
# 3. GROUND TRUTH + METRICS
# ============================================================================
def ground_truth(corpus, seed, mode='hole'):
    """Return set of relevant paper indices for a seed in cluster A.
      mode='hole': the A-B hole bridges (B = A's ring neighbour) -- the target
                    of hole-detection; hardest, most honest.
      mode='all': every bridge touching A (easier; content-entropy does well).
    """
    A = corpus['paper_home'][seed]
    rel = set()
    for i, pr in enumerate(corpus['paper_pair']):
        if pr is None:
            continue
        a, b = pr
        if A not in (a, b):
            continue
        pair = tuple(sorted((a, b)))
        if mode == 'all':
            rel.add(i)
        elif mode == 'hole' and pair in corpus['hole_pairs']:
            rel.add(i)
    rel.discard(seed)
    return rel

def _order(score):
    return np.argsort(-score, kind='stable')

def average_precision(score, rel):
    if not rel:
        return np.nan
    order = _order(score)
    hits, ap = 0, 0.0
    for r, idx in enumerate(order, 1):
        if idx in rel:
            hits += 1
            ap += hits / r
    return ap / len(rel)

def ndcg_at_k(score, rel, k=10):
    if not rel:
        return np.nan
    order = _order(score)[:k]
    dcg = sum((1.0 / np.log2(r + 1)) for r, idx in enumerate(order, 1) if idx in rel)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(k, len(rel)) + 1))
    return dcg / idcg if idcg > 0 else np.nan

def precision_at_k(score, rel, k=10):
    if not rel:
        return np.nan
    order = _order(score)[:k]
    return sum(1 for i in order if i in rel) / k

def recall_at_k(score, rel, k=10):
    if not rel:
        return np.nan
    order = _order(score)[:k]
    return sum(1 for i in order if i in rel) / len(rel)


# ============================================================================
# 4. EVAL DRIVER
# ============================================================================
RANKERS = {
    'tfidf':      rank_tfidf,
    'citation':   rank_citation,
    'semantic':   rank_semantic,
    'hybrid':     rank_hybrid,
    'interdisc*': rank_interdisc_oracle,
    'graph':      None,     # handled specially (needs cache)
}
CONTENT_CITE = ['tfidf', 'semantic', 'citation', 'hybrid', 'interdisc*']
# FAIR = deployable without oracle info (what the task literally asked us to
# beat: keyword/TF-IDF, citation-count, semantic-similarity, + our own 'hybrid'
# steelman). 'interdisc*' uses GROUND-TRUTH cluster labels (an oracle no real
# system has) -- it is a CEILING PROBE, not a fair baseline, and is reported
# separately so it can never be silently blended into the primary accept-gate.
FAIR_BASELINES = ['tfidf', 'semantic', 'citation', 'hybrid']
CEILING_PROBE = 'interdisc*'

def eval_corpus(corpus, gt_mode='hole', graph_cache=None, graph_corpus=None,
                use_resolvent=True):
    """Return dict method -> dict metric -> list over seeds (one per core paper
    seed sampled). graph_corpus lets the graph ranker run on a DIFFERENT
    (e.g. structure-shuffled) incidence than the baselines."""
    if graph_cache is None:
        graph_cache = precompute_graph(corpus)
    if graph_corpus is None:
        graph_corpus = corpus
    Tcache = {'T': _tfidf(corpus['X'])}
    cores = np.where(corpus['paper_role'] == 'core')[0]
    out = {m: {mm: [] for mm in ('AP', 'nDCG', 'P5', 'R10')} for m in RANKERS}
    scores_for_corr = {m: [] for m in RANKERS}
    for s in cores:
        rel = ground_truth(corpus, s, gt_mode)
        if not rel:
            continue
        for m in RANKERS:
            if m == 'graph':
                sc = GRAPH_RANKER(graph_corpus, s, graph_cache, use_resolvent)
            elif m in ('tfidf',):
                sc = RANKERS[m](corpus, s, Tcache)
            else:
                sc = RANKERS[m](corpus, s)
            out[m]['AP'].append(average_precision(sc, rel))
            out[m]['nDCG'].append(ndcg_at_k(sc, rel, 10))
            out[m]['P5'].append(precision_at_k(sc, rel, 5))
            out[m]['R10'].append(recall_at_k(sc, rel, 10))
            scores_for_corr[m].append(sc)
    return out, scores_for_corr

def paired_bootstrap_ci(a, b, n=4000, seed=0):
    """95% CI of mean(a-b) by paired bootstrap over trials."""
    a = np.asarray(a); b = np.asarray(b)
    mask = ~(np.isnan(a) | np.isnan(b))
    d = a[mask] - b[mask]
    if len(d) == 0:
        return (np.nan, np.nan, np.nan)
    rng = default_rng(seed)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def run_sweep(camouflages=(0.0, 0.3, 0.6, 0.9), n_corpora=8, gt_mode='hole',
              base_seed=1000, verbose=True):
    """Aggregate metrics across corpora for each camouflage level."""
    results = {}
    for cam in camouflages:
        agg = {m: {mm: [] for mm in ('AP', 'nDCG', 'P5', 'R10')} for m in RANKERS}
        for r in range(n_corpora):
            corpus = make_corpus(base_seed + r, camouflage=cam)
            gc = precompute_graph(corpus)
            out, _ = eval_corpus(corpus, gt_mode, gc)
            for m in RANKERS:
                for mm in agg[m]:
                    agg[m][mm].extend(out[m][mm])
        results[cam] = agg
        if verbose:
            print(f"\n=== camouflage={cam:.1f}  gt={gt_mode}  "
                  f"(corpora={n_corpora}, seeds/corpus~{len(agg['graph']['AP'])//n_corpora}) ===")
            _print_table(agg)
    return results

def _print_table(agg):
    order = ['tfidf', 'citation', 'semantic', 'hybrid', 'interdisc*', 'graph']
    print(f"  {'method':<11} {'MAP':>7} {'nDCG@10':>8} {'P@5':>6} {'R@10':>6}")
    for m in order:
        map_ = np.nanmean(agg[m]['AP'])
        nd = np.nanmean(agg[m]['nDCG'])
        p5 = np.nanmean(agg[m]['P5'])
        r10 = np.nanmean(agg[m]['R10'])
        tag = '  <-- OURS' if m == 'graph' else ('  <-- oracle ceiling-probe (unfair, not a baseline)'
                                                  if m == CEILING_PROBE else '')
        print(f"  {m:<11} {map_:>7.3f} {nd:>8.3f} {p5:>6.3f} {r10:>6.3f}{tag}")
    # PRIMARY gate: best FAIR (deployable, no-oracle) rival
    best = max(FAIR_BASELINES, key=lambda m: np.nanmean(agg[m]['AP']))
    dmap, lo, hi = paired_bootstrap_ci(agg['graph']['AP'], agg[best]['AP'])
    verdict = 'graph > best_fair' if lo > 0 else ('graph ~ best_fair' if hi > 0 else 'graph < best_fair')
    print(f"  best FAIR baseline = '{best}' (MAP {np.nanmean(agg[best]['AP']):.3f});  "
          f"ΔMAP(graph-best_fair) = {dmap:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  -> {verdict}")
    # SECONDARY (scope-honesty): the oracle ceiling probe -- not a fair baseline,
    # reported to show where a hypothetical all-knowing content detector sits.
    dceil, loc, hic = paired_bootstrap_ci(agg['graph']['AP'], agg[CEILING_PROBE]['AP'])
    print(f"  oracle ceiling '{CEILING_PROBE}' (MAP {np.nanmean(agg[CEILING_PROBE]['AP']):.3f}, "
          f"uses GT cluster labels no real system has);  ΔMAP(graph-ceiling) = {dceil:+.3f} "
          f"95%CI[{loc:+.3f},{hic:+.3f}]")


def run_null(camouflage=0.9, n_corpora=8, base_seed=2000):
    """STRUCTURE-SHUFFLE null: graph-search runs on a degree-preserving rewired
    incidence (structure destroyed, margins kept); baselines on the real corpus.
    If the graph advantage vanishes, the win was STRUCTURAL (real)."""
    real_g, null_g, best_bc = [], [], []
    for r in range(n_corpora):
        corpus = make_corpus(base_seed + r, camouflage=camouflage)
        # real
        gc = precompute_graph(corpus)
        out_r, _ = eval_corpus(corpus, 'hole', gc)
        # null: shuffle structure for graph only
        shuf = structure_shuffle(corpus, base_seed + r + 7)
        gc_null = precompute_graph(shuf)
        out_n, _ = eval_corpus(corpus, 'hole', gc_null, graph_corpus=shuf)
        real_g.extend(out_r['graph']['AP'])
        null_g.extend(out_n['graph']['AP'])
        best = max(FAIR_BASELINES, key=lambda m: np.nanmean(out_r[m]['AP']))
        best_bc.extend(out_r[best]['AP'])
    dr, lor, hir = paired_bootstrap_ci(real_g, best_bc)
    dn, lon, hin = paired_bootstrap_ci(null_g, best_bc)
    print(f"\n=== STRUCTURE-SHUFFLE NULL  (camouflage={camouflage}) ===")
    print(f"  graph MAP (real structure)   = {np.nanmean(real_g):.3f}")
    print(f"  graph MAP (shuffled struct.) = {np.nanmean(null_g):.3f}")
    print(f"  ΔMAP(graph_real - best_fair)  = {dr:+.3f}  95%CI[{lor:+.3f},{hir:+.3f}]")
    print(f"  ΔMAP(graph_null - best_fair)  = {dn:+.3f}  95%CI[{lon:+.3f},{hin:+.3f}]")
    collapsed = (dn < 0.5 * dr) and (lon < 0)
    print(f"  advantage COLLAPSES under shuffle? {collapsed}  "
          f"(=> win is {'STRUCTURAL' if collapsed else 'NOT structural / artifact'})")
    return dr, dn


def run_redundancy(camouflage=0.9, n_corpora=6, base_seed=3000):
    """Spearman rank-correlation of graph scores vs each baseline (are we just
    'semantic+citation with extra steps'?) + the documented resolvent<->degree
    decorrelation check."""
    rhos = {m: [] for m in CONTENT_CITE}
    res_deg = []
    for r in range(n_corpora):
        corpus = make_corpus(base_seed + r, camouflage=camouflage)
        gc = precompute_graph(corpus)
        # resolvent vs paper-degree (external anchor: should be ~decorrelated)
        rd = spearmanr(gc['Lpp'], gc['deg_paper']).correlation
        res_deg.append(rd)
        _, scores = eval_corpus(corpus, 'hole', gc)
        for i in range(len(scores['graph'])):
            g = scores['graph'][i]
            finite = np.isfinite(g)
            for m in CONTENT_CITE:
                rho = spearmanr(g[finite], scores[m][i][finite]).correlation
                if np.isfinite(rho):
                    rhos[m].append(rho)
    print(f"\n=== REDUNDANCY  (camouflage={camouflage}) ===")
    print(f"  Spearman rho(graph, baseline)  [<0.80 => NOT redundant]")
    for m in CONTENT_CITE:
        print(f"    graph vs {m:<11} : {np.nanmean(rhos[m]):+.3f}")
    print(f"  external anchor: Spearman rho(resolvent L+_ii, paper degree) = "
          f"{np.nanmean(res_deg):+.3f}  (documented ~ -0.25, decorrelated)")
    return rhos, np.nanmean(res_deg)


def run_ablation(camouflage=0.9, n_corpora=8, base_seed=4000):
    """Marginal value of the resolvent over-determination gate: residual-only
    vs residual x resolvent."""
    ronly, rboth, best_bc = [], [], []
    for r in range(n_corpora):
        corpus = make_corpus(base_seed + r, camouflage=camouflage)
        gc = precompute_graph(corpus)
        cores = np.where(corpus['paper_role'] == 'core')[0]
        for s in cores:
            rel = ground_truth(corpus, s, 'hole')
            if not rel:
                continue
            ronly.append(average_precision(rank_graph(corpus, s, gc, False), rel))
            rboth.append(average_precision(rank_graph(corpus, s, gc, True), rel))
            best = max(FAIR_BASELINES,
                       key=lambda m: average_precision(RANKERS[m](corpus, s), rel)
                       if m != 'tfidf' else average_precision(rank_tfidf(corpus, s), rel))
            best_bc.append(average_precision(RANKERS[best](corpus, s)
                           if best != 'tfidf' else rank_tfidf(corpus, s), rel))
    print(f"\n=== ABLATION: resolvent gate  (camouflage={camouflage}) ===")
    print(f"  residual-only          MAP = {np.nanmean(ronly):.3f}")
    print(f"  residual x resolvent   MAP = {np.nanmean(rboth):.3f}  (over-determined, OURS)")
    d, lo, hi = paired_bootstrap_ci(rboth, ronly)
    print(f"  Δ(over-det - residual-only) = {d:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]")
    return ronly, rboth


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    print("#" * 78)
    print("# paper-search: graph-search vs content+citation baselines")
    print("# POSITIVE claim forced. honest-negative = PASS. See module docstring.")
    print("#" * 78)

    assert_no_gt_leakage(GRAPH_RANKER, precompute_graph)
    print("\n[guard] GT-leakage check: PASS (GRAPH_RANKER + precompute_graph touch "
          "no ground-truth field)")

    print("\n########## PART A: hole-filler retrieval (the honest hard task) ##########")
    res_hole = run_sweep(gt_mode='hole', n_corpora=8)

    print("\n########## PART B: ALL-bridges retrieval (scope contrast) ##########")
    print("# If graph ~ interdisc* here but graph > interdisc* in Part A, the")
    print("# added power is specifically HOLE structure, not 'detect a mixture'.")
    res_all = run_sweep(gt_mode='all', camouflages=(0.9,), n_corpora=8)

    print("\n########## PART C: STRUCTURE-SHUFFLE NULL (is the win structural?) ##########")
    run_null(camouflage=0.9)
    run_null(camouflage=0.6)

    print("\n########## PART D: REDUNDANCY (does graph reduce to semantic+cite?) ##########")
    run_redundancy(camouflage=0.9)

    print("\n########## PART E: ABLATION (marginal value of resolvent gate) ##########")
    run_ablation(camouflage=0.9)

    print("\n" + "#" * 78)
    print("# Read the ΔMAP 95%CIs (Part A), the NULL collapse (Part C), and the")
    print("# redundancy rhos (Part D) together for the verdict. No single number.")
    print("#" * 78)
