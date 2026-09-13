"""
graph_hole_engine.engine — the shippable INVARIANT-hole detector (half 1 of 2).
==============================================================================
The GraphHoleEngine finds STRUCTURAL HOLES in a domain's coupling graph and certifies each with
an over-determination gate and an abstain floor. It is the *invariant-residual* half of H's
two-half verdict: the hole is a fixed property of the graph topology, independent of any design
choice. (The config-DEPENDENT optimizer-half — OED/topopt/AMR — is the SIBLING in oed_sibling.py.
It shares kernel code but asks a different question; it is deliberately NOT reachable from here.)

The engine is three composed kernel operations plus an external residual channel:

    1. resolvent-leverage L⁺_ii   (kernel.resolvent_leverage on the graph-Laplacian) -> the hole field
    2. N_eff over-determination   (kernel.participation_ratio of a node's confirmer spectrum) -> credit
    3. abstain-when-unspanned      (kernel.spectral_sigma_min below a noise floor) -> abstain
       + an EXTERNAL residual channel R = C_obs ⊖ C_pred, which is a SECOND, decorrelated leg.

★ Why there is no `endogenous` flag any more: hole-detection ALWAYS has an external observation
  channel, so residual ⊥ structure are two genuinely decorrelated legs (F's value⊥structure). If in
  your domain the residual *equals* the leverage (design IS the information), you are doing OED, not
  hole-detection — reach for the sibling. The engine MEASURES ρ(residual, structure) and reports it
  as evidence the two legs are real; a high ρ is a smell that you brought an optimizer problem to a
  detector.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Hashable

import numpy as np

from .kernel import participation_ratio, resolvent_leverage, spectral_sigma_min

__all__ = ["DomainAdapter", "NodeVerdict", "HoleReport", "GraphHoleEngine"]


# ==================================================================================================
# THE DOMAIN ADAPTER — the clean public surface a new hole-detection domain targets.
# ==================================================================================================


class DomainAdapter(abc.ABC):
    """Plug a graph domain in here to get hole-detection + over-det + abstain for free.

    The engine consumes ONLY this interface — never your domain internals. Map your domain onto the
    resolvent geometry:

        REQUIRED (abstract):
          nodes               -> ordered node set   (index i ↔ row/col i of Op and V)
          governing_operator  -> Op   the coupling operator whose resolvent holds the holes
                                        (symmetric PSD n×n; physics=graph Laplacian, papers=citation Laplacian)
          c_obs(node), c_pred(node)     the EXTERNAL observation and the model prediction; their ⊖ is the
                                        residual channel (the decorrelated 2nd leg)
          confirmer_matrix(node) -> (m×k)  k DECORRELATED confirmer columns; the over-det gate is
                                        participation_ratio(eigs(corr(·)))

        OPTIONAL (sensible defaults; override to declare a LEAK):
          point_embeddings    -> V   default = identity (node indicators) ⇒ ℓ_i = Op⁺_ii. Physics & papers
                                        keep this. (OED would override with sensitivity rows — but OED is a
                                        sibling, not a DomainAdapter.)
          residual(node)        -> R   default = metric subtraction float(c_obs) − float(c_pred).
          identifiability(node) -> σ   default = spectral_sigma_min(governing_operator).

    Anything you cannot express in this geometry: append a one-line note to self.leaks. A leak is the
    honest boundary of the abstraction, not a bug — the engine surfaces it in every report.
    """

    # --- knobs (override per domain) --------------------------------------------------------------
    noise_floor: float = 1e-3      # χ: identifiability below this ⇒ direction unspanned ⇒ abstain.
    overdet_frac: float = 0.6      # credit a node iff N_eff ≥ overdet_frac · (#confirmers).

    def __init__(self) -> None:
        self.leaks: list[str] = []

    # --- the graph (REQUIRED) ---------------------------------------------------------------------
    @abc.abstractmethod
    def nodes(self) -> list[Hashable]:
        """the ordered node set (index i ↔ row/col i of Op and V)."""

    @abc.abstractmethod
    def governing_operator(self) -> np.ndarray:
        """Op (n×n symmetric PSD): the operator whose resolvent L⁺ holds the structural holes."""

    # --- the leverage embedding (OPTIONAL) --------------------------------------------------------
    def point_embeddings(self) -> np.ndarray:
        """V (n×d): row i = embedding of node i. Default = identity ⇒ ℓ_i = Op⁺_ii (the L⁺_ii hole)."""
        n = len(self.nodes())
        return np.eye(n)

    # --- residual channel: observation vs prediction (REQUIRED c_obs/c_pred) ----------------------
    @abc.abstractmethod
    def c_obs(self, node: Hashable) -> Any:
        """the OBSERVED capability/coupling at a node (measured reality; the external channel)."""

    @abc.abstractmethod
    def c_pred(self, node: Hashable) -> Any:
        """the PREDICTED capability/coupling (what the current coupling model expects)."""

    def residual(self, node: Hashable) -> float:
        """R = C_obs ⊖ C_pred. Default = metric subtraction. Override + declare a leak if your
        C-space is not metric (papers uses a LATTICE set-difference)."""
        return float(self.c_obs(node)) - float(self.c_pred(node))

    # --- over-determination (REQUIRED) ------------------------------------------------------------
    @abc.abstractmethod
    def confirmer_matrix(self, node: Hashable) -> np.ndarray:
        """(m × k): k DECORRELATED confirmer signals (columns) for this node's evidence. The engine's
        over-det gate is participation_ratio(eigs(corr(this)))."""

    # --- identifiability floor (OPTIONAL) ---------------------------------------------------------
    def identifiability(self, node: Hashable) -> float:
        """σ_min for this node's direction. Default = spectral (from Op). Override + declare a leak if
        there is no linear operator to take σ_min of (papers uses extraction_confidence)."""
        return spectral_sigma_min(self.governing_operator())

    # --- self-report ------------------------------------------------------------------------------
    def name(self) -> str:
        return type(self).__name__


# ==================================================================================================
# REPORT TYPES
# ==================================================================================================


@dataclass
class NodeVerdict:
    node: Hashable
    residual: float          # |C_obs ⊖ C_pred|, the external search signal (ranked, never thresholded)
    leverage: float          # resolvent-leverage ℓ = L⁺_ii (invariant structural-hole strength)
    n_eff: float             # over-determination of the node's evidence (participation_ratio)
    sigma_min: float         # identifiability floor for this node's direction
    credited: bool           # over-det gate passed (N_eff ≥ frac·k)
    abstain: bool            # identifiability below the noise floor ⇒ unspanned ⇒ abstain
    score: float             # certified hole score (0 if abstained or un-credited)


@dataclass
class HoleReport:
    domain: str
    verdicts: list[NodeVerdict]           # sorted by certified hole score, descending
    hole_ranking: list[Hashable]          # nodes by resolvent-leverage (structural holes first)
    residual_ranking: list[Hashable]      # nodes by |residual| (observation anomalies first)
    abstained: list[Hashable]             # nodes below the identifiability floor (unspanned)
    leaks: list[str]                      # the adapter's declared boundaries of the abstraction
    rho_residual_structure: float         # MEASURED |corr(residual, leverage)|; low ⇒ 2 decorrelated legs


# ==================================================================================================
# THE ENGINE — domain-independent. Calls ONLY the kernel + the adapter API.
# ==================================================================================================


class GraphHoleEngine:
    """Invariant structural-hole detector for ANY DomainAdapter.

        residual  →  hole-field (L⁺_ii)  →  over-det gate (N_eff)  →  abstain (σ_min)  →  certified ranking

    Every heavy operation is a shared kernel function (resolvent_leverage, participation_ratio,
    spectral_sigma_min). The engine adds only the composition + the two-decorrelated-legs measurement.
    It NEVER calls waterfill_reallocate — reallocating a budget is the sibling optimizer's job.
    """

    def __init__(self, adapter: DomainAdapter) -> None:
        self.a = adapter

    # --- the resolvent-leverage hole field (SHARED kernel) ----------------------------------------
    def hole_field(self) -> np.ndarray:
        """ℓ = L⁺_ii over the nodes — the invariant structural-hole field."""
        return resolvent_leverage(self.a.governing_operator(), self.a.point_embeddings())

    def hole_ranking(self) -> list[Hashable]:
        nodes = self.a.nodes()
        return [nodes[i] for i in np.argsort(-self.hole_field())]

    # --- residual field (adapter arithmetic; ranked, never thresholded — mismatched-units lesson) --
    def residual_field(self) -> np.ndarray:
        return np.array([abs(self.a.residual(n)) for n in self.a.nodes()], float)

    # --- over-determination gate (SHARED kernel) --------------------------------------------------
    def overdet(self, node: Hashable) -> tuple[float, bool]:
        C = np.atleast_2d(self.a.confirmer_matrix(node))
        if C.shape[1] < 2:
            return 1.0, False                                  # a single confirmer can never over-determine
        X = (C - C.mean(0)) / (C.std(0) + 1e-12)
        eig = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))
        neff = participation_ratio(eig)                        # <-- the SAME participation_ratio everywhere
        return neff, bool(neff >= self.a.overdet_frac * C.shape[1])

    # --- abstain-when-unspanned gate --------------------------------------------------------------
    def abstains(self, node: Hashable) -> tuple[float, bool]:
        s = self.a.identifiability(node)
        return s, bool(s < self.a.noise_floor)

    # --- the composed detection -------------------------------------------------------------------
    def detect(self) -> HoleReport:
        nodes = self.a.nodes()
        lev = self.hole_field()
        res = self.residual_field()
        verdicts: list[NodeVerdict] = []
        for i, node in enumerate(nodes):
            neff, credited = self.overdet(node)
            smin, abstain = self.abstains(node)
            # certified hole score = observation-anomaly × structural-leverage / identifiability-cost,
            # gated to 0 unless over-determined AND above the abstain floor.
            score = 0.0
            if credited and not abstain:
                score = float(res[i] * lev[i] / (smin + self.a.noise_floor))
            verdicts.append(NodeVerdict(node, float(res[i]), float(lev[i]), neff, smin,
                                        credited, abstain, score))
        # MEASURE that the external residual channel is decorrelated from pure structure (2 legs, not 1).
        if res.std() > 1e-12 and lev.std() > 1e-12:
            rho = abs(float(np.corrcoef(res, lev)[0, 1]))
        else:
            rho = 0.0
        return HoleReport(
            domain=self.a.name(),
            verdicts=sorted(verdicts, key=lambda v: -v.score),
            hole_ranking=[nodes[i] for i in np.argsort(-lev)],
            residual_ranking=[nodes[i] for i in np.argsort(-res)],
            abstained=[v.node for v in verdicts if v.abstain],
            leaks=list(self.a.leaks),
            rho_residual_structure=rho,
        )
