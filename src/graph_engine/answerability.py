"""Exact pre-compression certificates for explicitly declared linear states.

Observation records are deletable data, not physical laws. State x=Tz is
unrestricted over Q in the declared coordinates. Scientific prose, nonlinear
flow, noise and finite-precision reconstruction are outside this contract.
Floats, when supplied, mean their exact binary values, never an SVD tolerance.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations
from numbers import Integral
from collections.abc import Mapping

from ._rational_interface import rref, nullspace


class MissingLinearContract(ValueError):
    """The input has no declared exact linear semantics."""


class SearchBudgetExceeded(RuntimeError):
    """No global optimum is certified when exact subset search is interrupted."""


def _row(row, n):
    if row is None:
        raise MissingLinearContract("query/observation has no exact linear operator")
    items = row.items() if isinstance(row, Mapping) else enumerate(row)
    if not isinstance(row, Mapping) and len(row) != n:
        raise ValueError("row dimension differs from declared state")
    out = {}
    for i, value in items:
        if not isinstance(i, Integral) or not 0 <= i < n:
            raise ValueError("invalid state coordinate")
        v = F(value)
        if v:
            out[int(i)] = v
    return out


def _dot(a, b):
    return sum((v * b.get(i, F(0)) for i, v in a.items()), F(0))


def _rank(rows):
    """Sparse coordinate shortcut; otherwise NT2's exact RREF."""
    rows = [r for r in rows if r]
    if all(len(r) == 1 for r in rows):
        return len({next(iter(r)) for r in rows})
    # Decompose disconnected supports before invoking the unchanged NT2 kernel.
    # A partition-mean projection has many small blocks, not one dense matrix.
    parent = {}
    def root(i):
        parent.setdefault(i, i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for row in rows:
        first = next(iter(row))
        for i in row:
            parent[root(i)] = root(first)
    blocks = {}
    for row in rows:
        blocks.setdefault(root(next(iter(row))), []).append(row)
    total = 0
    for block in blocks.values():
        cols = sorted(set().union(*(set(r) for r in block)))
        total += len(rref([[r.get(i, F(0)) for i in cols] for r in block])[1])
    return total


@dataclass(frozen=True)
class LinearStateGraph:
    """Record -> exact observation row, with optional J1-style state map T.

    basis is a sequence of columns in physical state coordinates. None means
    independent physical coordinates. The basis can be nonorthogonal; no
    Euclidean metric or prior is assumed. Dependent columns are also valid.
    """
    state_dim: int
    observations: Mapping
    basis: object = None
    provenance: str = "caller-declared exact linear state"

    def __post_init__(self):
        if not isinstance(self.state_dim, Integral) or self.state_dim < 0:
            raise ValueError("nonnegative integer state_dim required")
        object.__setattr__(self, "observations",
                           {k: _row(r, self.state_dim) for k, r in self.observations.items()})
        if self.basis is not None:
            object.__setattr__(self, "basis", tuple(_row(r, self.state_dim) for r in self.basis))

    @property
    def coordinate_dim(self):
        return self.state_dim if self.basis is None else len(self.basis)

    def pullback(self, row):
        row = _row(row, self.state_dim)
        if self.basis is None:
            return row
        return {j: v for j, col in enumerate(self.basis) if (v := _dot(row, col))}

    def lift(self, coordinates):
        if self.basis is None:
            return coordinates.copy()
        out = {}
        for j, value in coordinates.items():
            for i, coefficient in self.basis[j].items():
                out[i] = out.get(i, F(0)) + value * coefficient
        return {i: v for i, v in out.items() if v}


@dataclass(frozen=True)
class Projection:
    """A declared exact linear compression operator in physical coordinates."""
    rows: object


def _inputs(graph, Q, keep):
    if not isinstance(graph, LinearStateGraph):
        raise MissingLinearContract("use LinearStateGraph; topology alone is insufficient")
    if not isinstance(Q, Mapping):
        Q = {i: q for i, q in enumerate(Q)}
    queries = {k: graph.pullback(q) for k, q in Q.items()}
    if isinstance(keep, Projection):
        kept = list(range(len(keep.rows)))
        rows = [graph.pullback(r) for r in keep.rows]
    else:
        kept = list(keep)
        if len(set(kept)) != len(kept) or any(k not in graph.observations for k in kept):
            raise ValueError("keep must contain distinct declared record identifiers")
        rows = [graph.pullback(graph.observations[k]) for k in kept]
    return queries, kept, rows


def answerable_before_compression(graph, Q, keep):
    """Return (intersection_rank, lost_query_ids, exact_witness).

    Recovery is a row of weights indexed by kept IDs. A lost question has an
    admissible delta with A_K delta=0 and q delta!=0; states 0 and delta collide.
    All statements quantify over every declared state, before any state values
    or compressed query outputs are inspected. Rank counts linear combinations
    as well as named questions; these need not agree.
    """
    queries, kept, rows = _inputs(graph, Q, keep)
    query_rank, observed_rank = _rank(list(queries.values())), _rank(rows)
    rank = query_rank + observed_rank - _rank(rows + list(queries.values()))
    recovery, collisions = {}, {}
    # Independent coordinate observations support very large record banks.
    disjoint = len(set().union(*(set(r) for r in rows))) == sum(len(r) for r in rows) if rows else True
    if disjoint:
        owner = {i: (k, row) for k, row in zip(kept, rows) for i in row}
        for key, q in queries.items():
            weights, z = {}, None
            for i, value in q.items():
                if i not in owner:
                    z = {i: F(1)}
                    break
                k, row = owner[i]
                if k in weights:
                    continue
                pivot = min(row)
                w = q.get(pivot, F(0)) / row[pivot]
                other = next((j for j in row if q.get(j, F(0)) != w*row[j]), None)
                if other is not None:
                    z = {other: F(1), pivot: -row[other]/row[pivot]}
                    break
                weights[k] = w
            if z is None:
                recovery[key] = {k: w for k, w in weights.items() if w}
            else:
                collisions[key] = {"coordinate_delta": z, "state_delta": graph.lift(z),
                                   "query_difference": _dot(q, z)}
    elif all(len(r) <= 1 for r in rows):
        coordinate = {}
        for k, row in zip(kept, rows):
            if row:
                i, value = next(iter(row.items()))
                coordinate.setdefault(i, (k, value))
        for key, q in queries.items():
            missing = sorted(set(q) - set(coordinate))
            if missing:
                z = {missing[0]: F(1)}
                collisions[key] = {"coordinate_delta": z, "state_delta": graph.lift(z),
                                   "query_difference": q[missing[0]]}
            else:
                recovery[key] = {coordinate[i][0]: v / coordinate[i][1] for i, v in q.items()}
    else:
        cols = sorted(set().union(*(set(r) for r in rows + list(queries.values()))))
        A = [[r.get(i, F(0)) for i in cols] for r in rows]
        # NT2 nullspace has no width for []; handle that explicitly.
        N = nullspace(A) if A else [[F(i == j) for i in range(len(cols))] for j in range(len(cols))]
        for key, q in queries.items():
            target = [q.get(i, F(0)) for i in cols]
            mode = next((v for v in N if sum(a*b for a, b in zip(target, v))), None)
            if mode is not None:
                z = {i: v for i, v in zip(cols, mode) if v}
                collisions[key] = {"coordinate_delta": z, "state_delta": graph.lift(z),
                                   "query_difference": _dot(q, z)}
                continue
            # Solve A.T w=q, allowing arbitrary free weights set to zero.
            aug = [[A[j][i] for j in range(len(A))] + [target[i]] for i in range(len(cols))]
            R, piv = rref(aug)
            weights = [F(0)] * len(A)
            for row, p in zip(R, piv):
                if p == len(A):
                    raise AssertionError("membership and recovery disagree")
                weights[p] = row[-1]
            recovery[key] = {k: w for k, w in zip(kept, weights) if w}
    witness = {"query_rank": query_rank, "observed_rank": observed_rank,
               "intersection_rank": rank, "recovery": recovery, "collisions": collisions,
               "provenance": graph.provenance,
               "arithmetic": "exact rational; floats denote exact binary inputs"}
    return rank, list(collisions), witness


def max_compression(graph, Q, *, max_subsets=100000):
    """Globally largest RECORD deletion retaining the full query rank.

    Independent-coordinate observations use a support proof. General record
    selection enumerates by increasing cardinality. Search exhaustion raises
    SearchBudgetExceeded instead of presenting a greedy set as an optimum.
    None removes the explicit subset budget. Ties use declaration order.
    """
    if not isinstance(Q, Mapping):
        Q = dict(enumerate(Q))
    queries, keys, rows = _inputs(graph, Q, graph.observations)
    full_rank = _rank(list(queries.values()))
    if answerable_before_compression(graph, Q, keys)[0] != full_rank:
        raise MissingLinearContract("full observations cannot answer the declared family")
    tested = 0
    if all(len(r) <= 1 for r in rows):
        coordinate = {}
        for k, row in zip(keys, rows):
            if row:
                coordinate.setdefault(next(iter(row)), k)
        required = sorted(set().union(*(set(q) for q in queries.values()))) if queries else []
        keep = [coordinate[i] for i in required]
        lower_bound, method = len(required), "independent-coordinate support proof"
    else:
        keep = None
        for size in range(full_rank, len(keys)+1):
            for trial in combinations(keys, size):
                tested += 1
                if max_subsets is not None and tested > max_subsets:
                    raise SearchBudgetExceeded("exact minimum-cardinality search exhausted")
                if answerable_before_compression(graph, Q, trial)[0] == full_rank:
                    keep = list(trial)
                    break
            if keep is not None:
                break
        lower_bound, method = len(keep), "exhaustive increasing-cardinality proof"
    keep_set = set(keep)
    return {"keep": keep, "delete": [k for k in keys if k not in keep_set],
            "rank": full_rank, "minimum_kept": lower_bound,
            "maximum_deleted": len(keys)-len(keep), "global_optimum": True,
            "subsets_tested": tested, "method": method,
            "witness": answerable_before_compression(graph, Q, keep)[2]}


def query_reparameterization(graph, Q):
    """J1's quotient with an exact nonorthogonal basis instead of SVD.

    Returns the smallest redesigned linear sketch of z, dimension rank(QT).
    This permits new rows and is distinct from maximum deletion of old records.
    """
    queries, _, _ = _inputs(graph, Q, [])
    cols = sorted(set().union(*(set(r) for r in queries.values()))) if queries else []
    matrix = [[r.get(i, F(0)) for i in cols] for r in queries.values()]
    R, piv = rref(matrix)
    basis = [{i: v for i, v in zip(cols, row) if v} for row in R[:len(piv)]]
    quotient = LinearStateGraph(graph.coordinate_dim, dict(enumerate(basis)))
    witness = answerable_before_compression(quotient, queries, quotient.observations)[2]
    return {"rank": len(piv), "rows": basis, "recovery": witness["recovery"],
            "minimality": "every linear sketch answering Q has rank >= rank(QT)"}


def repair_compression(graph, Q, keep):
    """Append a minimum number of new scalar observations preserving all Q.

    Existing records stay fixed. New exact linear observations are permitted;
    this is a repair/design operation, not deletion of existing records.
    """
    if not isinstance(Q, Mapping):
        Q = dict(enumerate(Q))
    if not isinstance(keep, Projection):
        keep = list(keep)
    queries, _, rows = _inputs(graph, Q, keep)
    rank, _, initial = answerable_before_compression(graph, Q, keep)
    physical_queries = Q if isinstance(Q, Mapping) else dict(enumerate(Q))
    additions = []
    span_rank = _rank(rows)
    for key, q in queries.items():
        candidate_rank = _rank(rows + [q])
        if candidate_rank > span_rank:
            rows.append(q)
            additions.append(physical_queries[key])
            span_rank = candidate_rank
    deficit = initial["query_rank"] - rank
    assert len(additions) == deficit
    old_rows = keep.rows if isinstance(keep, Projection) else [graph.observations[k] for k in keep]
    projection = Projection(list(old_rows) + additions)
    final_rank, lost, witness = answerable_before_compression(graph, Q, projection)
    assert not lost and final_rank == initial["query_rank"]
    return {"additional_rows": additions, "minimum_additional_observations": deficit,
            "rank": final_rank, "witness": witness,
            "minimality": "one additional scalar can increase intersection rank by at most one"}


def harmonic_graph(n, edges, weights, shared, *, include_edge_observations=True):
    """Declared harmonic-potential state using graph_interface.export_edges.

    Conductances follow the exporter's exact-binary64 input contract. Rational
    weights that would be rounded by that exporter are explicitly rejected.
    Interior sources are zero, boundary potentials arbitrary. The exact H from
    the unchanged conductance operator gives x=Tz. Records are node potentials
    and (optionally) edge potential drops. Deletion concerns observations; edge
    conductances and the harmonic law remain required metadata.
    """
    from .graph_interface import GraphInterface
    if weights is not None and any(F(float(w)) != F(w) for w in weights):
        raise ValueError("edge exporter requires exactly representable binary64 conductances")
    interface = GraphInterface.export_edges(n, edges, weights, shared)
    columns = []
    for j, boundary in enumerate(shared):
        col = {int(boundary): F(1)}
        for i, node in enumerate(interface.interior):
            if interface.exact_H[i][j]:
                col[int(node)] = interface.exact_H[i][j]
        columns.append(col)
    observations = {("node", i): {i: F(1)} for i in range(n)}
    if include_edge_observations:
        for j, (a, b) in enumerate(edges):
            observations[("edge", j)] = {} if a == b else {int(a): F(1), int(b): F(-1)}
    return LinearStateGraph(n, observations, columns,
                            "exact edge-native harmonic model; zero interior sources")
