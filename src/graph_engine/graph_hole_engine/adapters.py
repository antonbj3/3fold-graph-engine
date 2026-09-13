"""
graph_hole_engine.adapters — the two graph-domains that share the invariant hole-detector.
==========================================================================================
H's verdict: the hole-detector (invariant graph-residual: resolvent-leverage on a graph-Laplacian
→ structural holes) TRANSFERS physics ↔ PAPER. These are the two adapters that demonstrate it on
the SAME GraphHoleEngine with the SAME kernel functions.

  PhysicsGraphAdapter — Op = coupling graph Laplacian; the shared path is WATERTIGHT (no leaks).
  PapersAdapter       — Op = citation Laplacian; the leverage + N_eff paths transfer, but three
                        operators LEAK (declared): the prose→claim observation map is non-linear,
                        so σ_min, the residual algebra, and the ranking each need a domain override.
                        Leaks are boundaries, not failures of the spectral certificate for nonsmooth inputs.
"""
from __future__ import annotations

from typing import Hashable

import numpy as np

from .engine import DomainAdapter

__all__ = ["PhysicsGraphAdapter", "PapersAdapter"]


# ==================================================================================================
# ADAPTER 1 — PHYSICS KNOWLEDGE-GRAPH (the discovery-graph; Op = graph Laplacian). WATERTIGHT.
# mirrors C's spine_c_resolvent_field hole-detector + D's residual-field (C_obs=agent pool-work, C_pred=degree)
# ==================================================================================================


class PhysicsGraphAdapter(DomainAdapter):
    """Nodes = capability/theory factors (V-nodes). Edges = shared sub-variable couplings.
    C_obs = agent pool WORK on a factor (grep mentions); C_pred = graph DEGREE (what the coupling model
    predicts). Residual = high-work × low-degree = an under-encoded factor. Hole = L⁺_ii bottleneck
    (a factor weakly embedded in the coupling graph). σ_min = spectral (SHARED default path).

    The planted graph decorrelates the two legs on PURPOSE: the STRUCTURAL hole (pendant V6, high
    leverage, ordinary work) is a DIFFERENT node from the OBSERVATION anomaly (well-embedded V2, huge
    work, low leverage). residual ⊥ structure = 2 genuinely decorrelated leads (the honest picture).
    """

    def __init__(self) -> None:
        super().__init__()
        # dense core K6 + a pendant HOLE (V6) + a high-degree DECOY clique joined by ONE bottleneck
        # edge (degree misses the hole, the resolvent catches it).
        n = 13
        A = np.zeros((n, n))
        for a in range(6):
            for b in range(a + 1, 6):
                A[a, b] = A[b, a] = 1.0                        # core K6
        A[6, 0] = A[0, 6] = 1.0                                # node 6 = pendant HOLE (low degree)
        for a in range(7, 13):
            for b in range(a + 1, 13):
                A[a, b] = A[b, a] = 1.0                        # decoy K6 (high internal degree)
        A[7, 1] = A[1, 7] = 1.0                                # decoy joined by ONE bottleneck edge
        self.A = A
        self._nodes = [f"V{i}" for i in range(n)]
        self.degree = A.sum(1)
        rng = np.random.default_rng(0)
        self.work = 200 + 60 * rng.standard_normal(n)
        self.work[2] += 900                                    # V2: under-encoded factor — huge work, well-embedded

    def nodes(self) -> list[Hashable]:
        return self._nodes

    def governing_operator(self) -> np.ndarray:
        return np.diag(self.A.sum(1)) - self.A                 # graph Laplacian L = Deg − A  (C's operator)

    def c_obs(self, node: Hashable) -> float:
        return float(self.work[self._nodes.index(node)])       # fleet work (mentions)

    def c_pred(self, node: Hashable) -> float:
        # the coupling model predicts work ∝ degree; scaled to work units (rank, don't threshold)
        i = self._nodes.index(node)
        return float(self.degree[i] / self.degree.mean() * self.work.mean())

    def confirmer_matrix(self, node: Hashable) -> np.ndarray:
        # 3 decorrelated confirmers of a factor's importance: work-intensity, coupling, mixed.
        i = self._nodes.index(node)
        rng = np.random.default_rng(i + 1)
        base = self.work[i]
        c1 = base + 20 * rng.standard_normal(24)                                # work-intensity
        c2 = self.degree[i] * 30 + 20 * rng.standard_normal(24)                 # coupling (decorrelated)
        c3 = base * 0.5 + self.degree[i] * 15 + 20 * rng.standard_normal(24)    # mixed
        return np.column_stack([c1, c2, c3])

    # identifiability: inherits the SHARED spectral_sigma_min default — WATERTIGHT, no override, no leak.


# ==================================================================================================
# ADAPTER 2 — LITERATURE GRAPH (the patent/paper grader repointed; Op = citation Laplacian)
# mirrors B's b_literature_grader_prototype (claim_membership / dominates / novelty_gap + ABSTAIN gate)
# ==================================================================================================


class PapersAdapter(DomainAdapter):
    """Nodes = papers. Edges = citation/topic links (Op = citation Laplacian — so hole_rank & N_eff
    still run on the SHARED kernel). C_obs = the paper's claimed capability ATOMS; C_pred = the
    platform's coverage atoms. Residual = novelty_gap (atoms the paper has that we lack).

    ★DECLARED LEAKS (the honest boundary of the abstraction — the prose→claim map is non-linear):
        L1  residual is a LATTICE ⊖ (set-difference), not metric subtraction — capability atoms have no metric.
        L2  σ_min:= extraction_confidence (an NLP reliability proxy), NOT a spectral eigenvalue — there is no
            linear observation operator on prose to take σ_min of. Same abstain ROLE, different math.
        L3  the certified ranking is discrete TRIAGE (dominate/build/abstain), not a continuous score.
    The leverage (citation-Laplacian L⁺_ii) and the N_eff over-det gate do NOT leak — they transfer intact.
    """

    PLATFORM = {"adjoint", "topology-opt", "info-gain", "co-design", "sigma_min-governor",
                "off-policy-probe", "conservation-cert", "model-class-ensemble"}
    noise_floor = 0.6      # χ := τ_extract (the grader's ABSTAIN threshold): extraction<0.6 ⇒ un-extractable ⇒ abstain

    def __init__(self) -> None:
        super().__init__()
        # a tiny corpus: name -> (claimed atoms, extraction_confidence)
        self.corpus = {
            "codesign_2604.25193": ({"adjoint", "topology-opt", "co-design"}, 0.85),                     # DOMINATED
            "neural_operator_surrogate": ({"neural-operator", "fourier-layer", "data-driven"}, 0.80),    # BUILD-GAP
            "world_model_rl": ({"world-model", "latent-dynamics", "rl"}, 0.78),                           # BUILD-GAP
            "vague_hype_abstract": ({"ai", "physics"}, 0.35),                                             # ABSTAIN
        }
        self._nodes = list(self.corpus)
        # citation graph among the papers (Op = its Laplacian): the two build-gap papers cite each
        # other (a weakly-embedded 'new-capability' cluster = a structural hole in OUR coverage graph).
        n = len(self._nodes)
        A = np.zeros((n, n))
        idx = {k: i for i, k in enumerate(self._nodes)}
        for u, v in [("codesign_2604.25193", "vague_hype_abstract"),
                     ("neural_operator_surrogate", "world_model_rl")]:
            A[idx[u], idx[v]] = A[idx[v], idx[u]] = 1.0
        A[idx["codesign_2604.25193"], idx["neural_operator_surrogate"]] = 1.0     # one bridge edge
        A[idx["neural_operator_surrogate"], idx["codesign_2604.25193"]] = 1.0
        self.A = A
        self.leaks = [
            "L1 residual = LATTICE set-difference (novelty_gap), not metric subtraction (capability atoms lack a metric)",
            "L2 sigma_min := extraction_confidence (NLP reliability proxy), not a spectral eigenvalue (no linear operator on prose)",
            "L3 certified ranking = discrete triage {dominate|build|abstain}, not a continuous score",
        ]

    def nodes(self) -> list[Hashable]:
        return self._nodes

    def governing_operator(self) -> np.ndarray:
        return np.diag(self.A.sum(1)) - self.A                 # citation Laplacian — hole_rank & N_eff run SHARED

    def c_obs(self, node: Hashable) -> set:
        return self.corpus[node][0]                            # the paper's claimed capability atoms

    def c_pred(self, node: Hashable) -> set:
        return self.PLATFORM                                   # what the platform's mechanism covers

    def residual(self, node: Hashable) -> float:               # LEAK L1: lattice ⊖, scored by |gap|
        gap = self.c_obs(node) - self.c_pred(node)             # atoms the paper has that we LACK (novelty_gap)
        return float(len(gap))

    def confirmer_matrix(self, node: Hashable) -> np.ndarray:
        # decorrelated GRADER AXES (launder-fence): existence, scope-match, relevance, independence.
        # Same over-det gate (participation_ratio) as physics — this part does NOT leak.
        atoms, conf = self.corpus[node]
        rng = np.random.default_rng(abs(hash(node)) % 2**31)
        existence = conf + 0.05 * rng.standard_normal(16)
        scope = (len(atoms - self.PLATFORM) > 0) * 1.0 + 0.1 * rng.standard_normal(16)
        relevance = (len(atoms & self.PLATFORM) > 0) * 1.0 + 0.1 * rng.standard_normal(16)
        independence = 0.1 * rng.standard_normal(16)           # noise axis (decorrelated)
        return np.column_stack([existence, scope, relevance, independence])

    def identifiability(self, node: Hashable) -> float:        # LEAK L2: extraction_confidence, not σ_min(Op)
        return float(self.corpus[node][1])

    def verdict_bucket(self, node: Hashable) -> str:           # LEAK L3: discrete triage, not a continuous score
        atoms, conf = self.corpus[node]
        if conf < self.noise_floor:
            return "ABSTAIN"
        gap = atoms - self.PLATFORM
        return "PAPER-OUTSIDE(build-gap)" if gap else "PLATFORM-DOMINATES"
