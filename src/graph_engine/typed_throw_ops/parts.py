"""parts.py — a PART is one building block read out of a report: what it takes in, what it hands on, under which
assumptions and in which regime, and by which operator.

A part is not a report and not a sentence. It is the smallest unit a kast can combine:

    inputs   typed ports (canonical quantity id, unit string, kind); `required=True` means AND: the part produces
             its outputs only when every required input is available (U586's Φ needs BOTH e and r)
    outputs  typed ports
    operator operator-family tokens from OPERATORS below, plus the literal equation text from the source
    assumptions  tokens; declared incompatibilities in CONFLICTS
    regime   categorical axes (AXES) -> allowed values;  box: continuous axis -> (lo, hi)
    reach    HOW the part was obtained, and therefore what a distance computed from it may claim:
               'operator'  equation/code read at the source line; ports and operator are the source's own
               'ports'     units and in/out quantities read from the source text, operator label assigned by the reader
               'words'     only prose was available (auto-extracted, or a claim without equation); every distance that
                           touches a 'words' part is marked and never enters an operator-level ranking unmarked
    status   proved | measured | reported | gap | consumer   (what the SOURCE says, not a verdict here; 'gap' = a middle the
             reports name as missing, 'consumer' = a lane's need: only ever the TAIL of a triple or chain)

Claims (for conf detection) are separate records attached to parts: sign claims, bounds, orderings, boolean
properties and implication rules. See triples.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------------------------------------------------
# vocabularies. Small and declared; a token outside them is an error, not a silent new category.
# ---------------------------------------------------------------------------------------------------------------
OPERATORS = {
    "schur_reduction": "eliminate a block: S = A_bb − A_bi A_ii⁻¹ A_ib (Kron/Schur; covariance or stiffness or Laplacian)",
    "schur_conditioning": "Schur complement of a COVARIANCE on observed coordinates (statistical conditioning)",
    "delassus_contraction": "G = J M⁻¹ Jᵀ, contact-space response of generalized compliance",
    "cone_projection_ncp": "de Saxcé / Coulomb natural-map NCP, λ ∈ K_μ, complementarity with corrected velocity",
    "active_set_newton": "semismooth Newton on a fixed branch labelling with outer active-set updates",
    "low_rank_update": "Woodbury / Sherman–Morrison / bordering / downdate of a factorization",
    "fixed_branch_linearization": "implicit-function derivative D δλ = −∂θF on a fixed branch",
    "event_saltation": "hybrid sensitivity through a reset: ds+ = dR/ds ds− + dR/dp + f− dT*/dp",
    "root_selection": "choice among multiple verified roots (continuity/nearest/min-norm gauge)",
    "min_norm_gauge": "argmin ½‖λ‖² over a convex fibre with fixed generalized impulse",
    "transfer_matrix_decay": "Bloch/transfer-matrix evanescent decay exp(−αN) through N periodic cells",
    "green_walk_bound": "walk-series / Green-row bound on |D⁻¹| entries under r(|T|)<1",
    "pseudospectral_margin": "σ_min(D) > η  ⇔ 0 ∉ Λ_η(D)",
    "coherence_certificate": "ρ ≤ Φ(e, r) = tanh(2 atanh e + atanh r) from relative mechanical/source errors",
    "vertex_maximization": "max over an affine PSD family attained at a polytope vertex (LMI convexity)",
    "fisher_information": "information Gram of an observation Jacobian; CRB",
    "logdet_set_value": "½ log det(I + H C Hᵀ/σ²) = I(x; y_S); monotone submodular",
    "target_restricted_value": "entropy drop of Q x only (value_Q); NOT submodular in general",
    "greedy_selection": "greedy set construction by marginal value",
    "moment_clipping": "exact zeroth/first moments of a material inside a cell (moment-of-fluid)",
    "local_topology_guard": "local digital-topology accept/reject of a single-cell edit",
    "galerkin_enrichment": "local enrichment of a Galerkin space (bubbles, trace modes, patch radius)",
    "gram_pencil": "spec(J M Jᵀ) = spec(M^½ Jᵀ J M^½)",
    "resistance_geometry": "effective resistance / hole field of a graph Laplacian",
    "sign_composition": "sign(s→o) = sign(s→x)·sign(x→o) over an intersected validity box",
    "history_sketch": "bounded summary of internal history state that a later query must reopen",
    "quantized_cache": "full-history cache quantized to a byte budget",
    "ode_state_transport": "native ODE/PDE state evolution of a coupled compartment model",
}

AXES = {
    "branch": {"open", "stick", "slip", "switching"},
    "contact_model": {"rigid", "compliant", "none"},
    "linearity": {"linear", "nonlinear"},
    "band": {"gap", "passband", "static"},
    "boundary": {"periodic", "free", "clamped", "any"},
    "dim": {"1D", "2D", "3D", "abstract"},
    "noise": {"independent", "correlated", "none"},
    "objective": {"info_logdet", "target_variance", "coherence_decoupling", "residual", "cost"},
    "graph_class": {"chain", "parallel_routes", "branched_dag", "directed", "general"},
    "uncertainty_class": {"independent_box", "tied_gain", "unrestricted_ball", "conditioned_schur", "none"},
    "time_basis": {"fixed_frequency", "time_domain", "static", "event"},
    "history": {"memoryless", "path_dependent"},
    "operator_model": {"scalar_surrogate", "block_delassus", "compliant_contact", "continuum_fem"},
}

CONFLICTS = [
    ("rigid_contact", "compliant_contact"),
    ("fixed_active_set", "active_set_switching"),
    ("gaussian_noise", "non_gaussian_noise"),
    ("unit_synthetic_noise", "calibrated_noise"),
    ("SPD_operator", "directed_nonnormal_operator"),
    ("affine_source_family", "conditioned_nonaffine_family"),
    ("block_diagonal_response", "dense_response"),
]
_CONFLICT_SET = {frozenset(p) for p in CONFLICTS}

REACH_LEVELS = ("operator", "ports", "words")


@dataclass
class Port:
    q: str                          # canonical quantity id
    unit: str | None = None         # unit string for claim_types.dimension; None for operator/label/field objects
    kind: str = "scalar"            # scalar | operator | label | field
    required: bool = True           # inputs: AND semantics
    given: bool = False             # inputs: externally available (measured / supplied) in the source's own setting


@dataclass
class Part:
    id: str
    report: str                     # report id in the corpus (for graph distance), e.g. "U487", "ENGINE:precision_form"
    family: str                     # contact | spectral | material | graph | bodytwin | dental | combinatorics
    does: str
    inputs: list[Port]
    outputs: list[Port]
    operator: list[str]
    equation: str = ""
    assumptions: list[str] = field(default_factory=list)
    regime: dict[str, list[str]] = field(default_factory=dict)
    box: dict[str, tuple[float, float]] = field(default_factory=dict)
    reach: str = "operator"
    status: str = "reported"
    source: str = ""                # path[:anchor]
    consumer: str = ""

    def __post_init__(self):
        self.inputs = [p if isinstance(p, Port) else Port(**p) for p in self.inputs]
        self.outputs = [p if isinstance(p, Port) else Port(**p) for p in self.outputs]
        bad = [o for o in self.operator if o not in OPERATORS]
        if bad:
            raise ValueError(f"{self.id}: unknown operator tokens {bad}")
        for ax, vals in self.regime.items():
            if ax not in AXES:
                raise ValueError(f"{self.id}: unknown regime axis {ax}")
            badv = set(vals) - AXES[ax]
            if badv:
                raise ValueError(f"{self.id}: unknown values {badv} on axis {ax}")
        if self.reach not in REACH_LEVELS:
            raise ValueError(f"{self.id}: reach must be one of {REACH_LEVELS}")
        self.box = {k: (float(v[0]), float(v[1])) for k, v in self.box.items()}


@dataclass
class Claim:
    id: str
    part: str
    kind: str                       # sign | bound | order | prop | rule | sign_at
    subject: str = ""
    object: str = ""
    sign: int = 0
    lo: float = float("-inf")
    hi: float = float("inf")
    value: bool | None = None       # prop claims
    premises: list[str] = field(default_factory=list)   # rule claims: premises ⇒ object (all props)
    axis: str = ""                  # sign_at: position axis; `at` is the position
    at: float = 0.0
    max_transitions: int = -1       # rule on an axis (sign_at family): at most this many sign changes
    regime: dict[str, list[str]] = field(default_factory=dict)
    box: dict[str, tuple[float, float]] = field(default_factory=dict)
    source: str = ""
    instance: str = ""              # the physical/mathematical OBJECT the claim is about ("" = any). Numbers about
                                    # different objects never combine into a conf (first corpus run: a cross-void
                                    # mount's e and a gait frame's r made three spurious 'contradictions')

    def __post_init__(self):
        self.box = {k: (float(v[0]), float(v[1])) for k, v in self.box.items()}


def assumption_conflicts(a: list[str], b: list[str]) -> list[tuple[str, str]]:
    return [(x, y) for x in a for y in b if frozenset((x, y)) in _CONFLICT_SET]


def regime_intersection(regimes: list[dict[str, list[str]]]) -> dict[str, set[str]] | None:
    """Categorical intersection per axis; an axis absent from a regime is unrestricted there. None = empty."""
    out: dict[str, set[str]] = {}
    for r in regimes:
        for ax, vals in r.items():
            cur = out.get(ax, set(AXES[ax]))
            cur = cur & set(vals)
            if not cur:
                return None
            out[ax] = cur
    return out


def box_intersection_many(boxes: list[dict[str, tuple[float, float]]]) -> dict[str, tuple[float, float]] | None:
    out: dict[str, tuple[float, float]] = {}
    for b in boxes:
        for v, (lo, hi) in b.items():
            clo, chi = out.get(v, (float("-inf"), float("inf")))
            lo2, hi2 = max(clo, lo), min(chi, hi)
            if lo2 > hi2:
                return None
            out[v] = (lo2, hi2)
    return out


def load(path: str) -> tuple[list[Part], list[Claim]]:
    d = json.load(open(path))
    parts = [Part(**p) for p in d.get("parts", [])]
    claims = [Claim(**c) for c in d.get("claims", [])]
    return parts, claims


def dump(parts: list[Part], claims: list[Claim], path: str, meta: dict[str, Any] | None = None) -> None:
    def enc(o):
        d = asdict(o)
        d["box"] = {k: list(v) for k, v in d.get("box", {}).items()}
        return d
    json.dump({"meta": meta or {}, "parts": [enc(p) for p in parts], "claims": [enc(c) for c in claims]},
              open(path, "w"), ensure_ascii=False, indent=1)
