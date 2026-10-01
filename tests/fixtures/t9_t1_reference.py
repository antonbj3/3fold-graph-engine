"""Anytime Thomson/Dirichlet certificates for finite positive resistor graphs.

Edges are (tail, head, conductance). Integers and Fractions are exact; real
floating inputs mean their represented binary value. Returned endpoints are
Fractions (or +inf), never inward-rounded floats. No Laplacian is assembled.

An integer budget limits refinement attempts AFTER mandatory initialization.
ResistanceBudget(seconds=...) instead limits total wall time, including that
initialization. The deadline is soft, checked between arithmetic operations:
a single Fraction operation and final witness copying can overrun. A deadline
before initialization completes returns the universal [0, inf], explicitly
marked uninformative. Verification is a separate O(edges + vertices) operation.

Local witnesses contain private fields. Exchanging only the interval does not
constitute a public proof or implement a multi-owner federation protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import heapq
import math
from numbers import Integral, Real
from time import perf_counter

__all__ = ["ResistanceBudget", "ResistanceWitness", "certified_cross_resistance",
           "verify_resistance_witness", "exact_edge_resistance"]


@dataclass(frozen=True)
class ResistanceBudget:
    seconds: float | None = None
    max_updates: int | None = None
    bits: int = 40
    refinement: str = "cycle"
    certify_every: int = 1

    def __post_init__(self):
        if self.seconds is not None and (not math.isfinite(self.seconds) or self.seconds < 0):
            raise ValueError("seconds must be finite and nonnegative")
        if self.max_updates is not None and (isinstance(self.max_updates, bool) or
                not isinstance(self.max_updates, Integral) or self.max_updates < 0):
            raise ValueError("max_updates must be a nonnegative integer")
        if isinstance(self.bits, bool) or not isinstance(self.bits, Integral) or self.bits < 1:
            raise ValueError("bits must be a positive integer")
        if self.seconds is None and self.max_updates is None:
            raise ValueError("specify a time or update budget")
        if self.refinement not in ("cycle", "tree", "cg"):
            raise ValueError("refinement must be cycle, tree or cg")
        if isinstance(self.certify_every, bool) or not isinstance(self.certify_every, Integral) or self.certify_every < 1:
            raise ValueError("certify_every must be a positive integer")


@dataclass(frozen=True)
class ResistanceWitness:
    kind: str
    flow: tuple = ()
    potential: tuple = ()
    component: tuple = ()
    updates: int = 0
    elapsed_seconds: float = 0.0
    startup_seconds: float | None = None
    budget_exhausted: bool = False
    # True only after all conductances and endpoints were validated.
    input_validated: bool = False


class _Expired(Exception):
    pass


def _node(x):
    if isinstance(x, bool) or not isinstance(x, Integral) or x < 0:
        raise ValueError("nodes must be nonnegative integer indices")
    return int(x)


def _rat(x):
    if isinstance(x, bool):
        raise TypeError("boolean conductance is not accepted")
    if isinstance(x, F):
        q = x
    elif isinstance(x, Integral):
        q = F(int(x))
    elif isinstance(x, Real):
        # Reject extended precision if conversion would lose represented bits.
        y = float(x)
        if not math.isfinite(y) or x != y:
            raise ValueError("conductance must be finite and exactly representable")
        q = F(y)
    else:
        raise TypeError("conductance must be integer, Fraction or binary real")
    if q <= 0:
        raise ValueError("strictly positive conductance required")
    return q


def _edges(edges, a, b, check=lambda: None):
    rows, adj = [], {a: [], b: []}
    for row in edges:
        check()
        if len(row) != 3:
            raise ValueError("edges must contain tail, head and conductance")
        u, v, c = _node(row[0]), _node(row[1]), _rat(row[2])
        i = len(rows)
        rows.append((u, v, c))
        adj.setdefault(u, [])
        adj.setdefault(v, [])
        if u != v:
            adj[u].append((v, i, 1))
            adj[v].append((u, i, -1))
    check()
    return rows, adj


def _snap(x, bits):
    """Exact nearest dyadic proposal; accept using exact energy, not rounding."""
    scale = 1 << bits
    return F((x.numerator * scale * 2 + x.denominator) // (2 * x.denominator), scale)


def _energies(rows, flow, phi, check=lambda: None):
    upper, energy = F(0), F(0)
    for (u, v, c), f in zip(rows, flow):
        check()
        upper += f * f / c
        d = phi[u] - phi[v]
        energy += c * d * d
    return upper, energy


def certified_cross_resistance(edges, a, b, budget):
    """Return (lo, hi, local witness) without elimination or a pseudoinverse.

    ``budget=0`` constructs just the start witnesses; larger integer budgets
    give a deterministic nested sequence. To cap total work by wall time use
    ``ResistanceBudget(seconds=t)``. If the latter cannot pay the initial edge
    scan and certificate construction the interval is [0, inf].
    """
    started = perf_counter()
    a, b = _node(a), _node(b)
    if isinstance(budget, Integral) and not isinstance(budget, bool):
        budget = ResistanceBudget(max_updates=int(budget))
    if not isinstance(budget, ResistanceBudget):
        raise TypeError("budget must be an integer or ResistanceBudget")
    deadline = None if budget.seconds is None else started + budget.seconds

    def check():
        if deadline is not None and perf_counter() >= deadline:
            raise _Expired

    if a == b:
        return F(0), F(0), ResistanceWitness("same_port", elapsed_seconds=perf_counter()-started)
    validated = False
    try:
        rows, adj = _edges(edges, a, b, check)
        validated = True
        # Exact Dijkstra from sink. Parent edges point toward sink and give a
        # shortest-resistance spanning tree of the port component.
        dist, parent, depth = {b: F(0)}, {}, {b: 0}
        heap, done = [(F(0), b)], set()
        while heap:
            check()
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for v, i, sign in adj[u]:
                check()
                candidate = d + 1 / rows[i][2]
                if v not in dist or candidate < dist[v]:
                    dist[v] = candidate
                    parent[v] = (u, i, -sign)
                    depth[v] = depth[u] + 1
                    heapq.heappush(heap, (candidate, v))
        if a not in done:
            component = tuple(sorted(done))
            return math.inf, math.inf, ResistanceWitness(
                "disconnected", component=component, elapsed_seconds=perf_counter()-started,
                startup_seconds=perf_counter()-started, input_validated=True)
        flow = [F(0)] * len(rows)
        u = a
        while u != b:
            check()
            v, i, sign = parent[u]
            flow[i] = F(sign)
            u = v
        phi = {}
        for u in adj:
            check()
            phi[u] = min(F(1), dist[u] / dist[a]) if u in dist else F(0)
        hi, energy = _energies(rows, flow, phi, check)
        lo = 1 / energy
        chord_indices = []
        tree = {i for _, i, _ in parent.values()}
        for i, (u, v, _) in enumerate(rows):
            check()
            if u in done and u != v and i not in tree:
                chord_indices.append(i)
        coordinates = sorted(done - {a, b})
        startup = perf_counter() - started
    except _Expired:
        return F(0), math.inf, ResistanceWitness(
            "uninformative", elapsed_seconds=perf_counter()-started,
            budget_exhausted=True, input_validated=validated)

    def cycle(i):
        u, v, _ = rows[i]
        z = [(i, 1)]
        # close chord u->v along tree v->u
        while u != v:
            check()
            if depth[v] >= depth[u]:
                p, edge, sign = parent[v]
                z.append((edge, sign)); v = p
            else:
                p, edge, sign = parent[u]
                z.append((edge, -sign)); u = p
        return z

    updates, exhausted = 0, False
    width = max(len(chord_indices), len(coordinates), 1)
    bits, sweep_changed = budget.bits, False

    def projected_flow(raw_potential=None):
        # Proposed Ohmic current. Quantization is harmless because its entire
        # divergence residual is corrected exactly on the spanning tree.
        proposed, residual = [], {u: F(int(u == a)-int(u == b)) for u in done}
        for u, v, c in rows:
            check()
            if u not in done:
                value = F(0)
            elif raw_potential is None:
                value = _snap(c * (phi[u]-phi[v]) / energy, bits)
            else:
                value = _snap(c * (raw_potential[u]-raw_potential[v]), bits)
            proposed.append(value)
            if u in done:
                residual[u] -= value
                residual[v] += value
        for u in sorted(parent, key=lambda u: depth[u], reverse=True):
            check()
            p, edge, sign = parent[u]
            proposed[edge] += sign * residual[u]
            residual[p] += residual[u]
        assert residual[b] == 0
        return proposed

    def improve_projected_flow(raw_potential=None):
        nonlocal flow, hi, sweep_changed
        proposed = projected_flow(raw_potential)
        g, h = F(0), F(0)
        for i, (_, _, c) in enumerate(rows):
            check()
            z = proposed[i] - flow[i]
            g += flow[i] * z / c
            h += z * z / c
        if h:
            alpha = _snap(-g / h, bits)
            change = 2*alpha*g + alpha*alpha*h
            if change < 0:
                candidate = []
                for old, new in zip(flow, proposed):
                    check()
                    candidate.append(old + alpha*(new-old))
                # Both fields are feasible, so any exact linear combination
                # with coefficients adding to one is feasible.
                flow, hi = candidate, hi + change
                sweep_changed = True

    def cg_proposals():
        """Untrusted PCG iterates; no direct solver or residual certificate."""
        nonlocal phi, energy, lo, updates
        if budget.max_updates == 0:
            return
        import numpy as np
        nodes = sorted(done - {b}) + [b]
        index = {u:i for i,u in enumerate(nodes)}
        n = len(nodes)-1
        selected = [(u,v,c) for u,v,c in rows if u in done and u != v]
        scale = max(c for _,_,c in selected)
        check()
        ii = np.array([index[u] for u,_,_ in selected])
        jj = np.array([index[v] for _,v,_ in selected])
        weights = np.array([float(c/scale) for _,_,c in selected])
        degree = (np.bincount(ii,weights=weights,minlength=n+1)+
                  np.bincount(jj,weights=weights,minlength=n+1))[:n]
        if np.any(degree <= 0):
            return
        def matvec(x):
            full = np.r_[x,0.0]
            current = weights*(full[ii]-full[jj])
            return (np.bincount(ii,weights=current,minlength=n+1)-
                    np.bincount(jj,weights=current,minlength=n+1))[:n]
        x, residual = np.zeros(n), np.zeros(n)
        residual[index[a]] = 1.0
        z = residual/degree
        p, rz = z.copy(), float(residual@z)
        # A floating breakdown merely stops proposals, leaving exact bounds.
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            while budget.max_updates is None or updates < budget.max_updates:
                check()
                ap = matvec(p); denom = float(p@ap)
                if not math.isfinite(denom) or denom <= 0 or not math.isfinite(rz) or rz <= 0:
                    break
                alpha = rz/denom
                x += alpha*p
                residual -= alpha*ap
                if not np.all(np.isfinite(x)) or x[index[a]] == 0:
                    break
                updates += 1
                if updates % budget.certify_every:
                    z = residual/degree
                    next_rz = float(residual@z)
                    p = z+(next_rz/rz)*p
                    rz = next_rz
                    continue
                # Exact values of the finite proposals, not their approximate
                # residual. Common normalization gives exactly phi[a]=1.
                raw = {b:F(0)}
                for u in nodes[:-1]:
                    check()
                    raw[u] = F(float(x[index[u]]))/scale
                candidate_phi = {}
                for u in adj:
                    check()
                    candidate_phi[u] = raw[u]/raw[a] if u in raw else F(0)
                candidate_energy = F(0)
                for u,v,c in rows:
                    check()
                    d = candidate_phi[u]-candidate_phi[v]
                    candidate_energy += c*d*d
                if candidate_energy < energy:
                    phi,energy = candidate_phi,candidate_energy
                    lo = 1/energy
                improve_projected_flow(raw)
                if lo == hi:
                    break
                z = residual/degree
                next_rz = float(residual@z)
                p = z+(next_rz/rz)*p
                rz = next_rz

    try:
        if budget.refinement == "cg":
            cg_proposals()
        if budget.refinement == "tree":
            improve_projected_flow()
        while budget.refinement != "cg" and (budget.max_updates is None or updates < budget.max_updates):
            check()
            step = updates // 2
            if updates % 2 == 0 and chord_indices and budget.refinement == "cycle":
                z = cycle(chord_indices[step % len(chord_indices)])
                g, h = F(0), F(0)
                for edge, sign in z:
                    check()
                    r = 1 / rows[edge][2]
                    g += sign * flow[edge] * r
                    h += r
                delta = _snap(-g / h, bits)
                change = 2 * delta * g + delta * delta * h
                if change < 0:
                    # Atomic feasible update: interruption is checked before
                    # mutation, never in the middle of a cycle's edge changes.
                    check()
                    for edge, sign in z:
                        flow[edge] += sign * delta
                    hi += change
                    sweep_changed = True
            elif updates % 2 == 1 and coordinates:
                u = coordinates[step % len(coordinates)]
                degree, rhs = F(0), F(0)
                for v, edge, _ in adj[u]:
                    check()
                    c = rows[edge][2]
                    degree += c
                    rhs += c * phi[v]
                delta = _snap(rhs / degree, bits) - phi[u]
                change = 2 * delta * (degree * phi[u] - rhs) + delta * delta * degree
                if change < 0:
                    phi[u] += delta
                    energy += change
                    lo = 1 / energy
                    sweep_changed = True
            updates += 1
            if budget.refinement == "tree" and coordinates and updates % (2*len(coordinates)) == 0:
                improve_projected_flow()
            if lo == hi:
                break
            if updates % (2 * width) == 0:
                if not sweep_changed:
                    # Refining the dyadic lattice prevents a permanent
                    # precision floor; budget remains the stopping rule.
                    bits += 16
                sweep_changed = False
            if not chord_indices and not coordinates:
                break
    except _Expired:
        exhausted = True
    # Final copying is charged in elapsed_seconds. A soft deadline can overrun.
    fs, ps = tuple(flow), tuple(sorted(phi.items()))
    return lo, hi, ResistanceWitness(
        "connected", fs, ps, updates=updates, elapsed_seconds=perf_counter()-started,
        startup_seconds=startup, budget_exhausted=exhausted, input_validated=True)


def verify_resistance_witness(edges, a, b, lo, hi, witness):
    """Recheck the fields/cut against the original graph, without a solve.

    Returns True or raises ValueError. Claimed metadata is never sufficient.
    The universal interval is only conditional on the positive-graph input
    contract, so even an early uninformative return is fully validated here.
    """
    a, b = _node(a), _node(b)
    rows, adj = _edges(edges, a, b)
    if not isinstance(witness, ResistanceWitness):
        raise ValueError("invalid witness type")
    if witness.kind == "same_port":
        if a != b or lo != 0 or hi != 0:
            raise ValueError("invalid same-port witness")
        return True
    if witness.kind == "uninformative":
        if lo != 0 or hi != math.inf:
            raise ValueError("invalid universal interval")
        return True
    if witness.kind == "disconnected":
        cut = set(witness.component)
        if a in cut or b not in cut or lo != math.inf or hi != math.inf:
            raise ValueError("cut must separate the ports")
        if any((u in cut) != (v in cut) for u, v, _ in rows):
            raise ValueError("cut has a positive crossing edge")
        return True
    if witness.kind != "connected" or len(witness.flow) != len(rows):
        raise ValueError("invalid connected witness")
    phi = dict(witness.potential)
    if len(phi) != len(witness.potential) or set(phi) != set(adj):
        raise ValueError("potential must assign every node exactly once")
    if any(not isinstance(x, F) for x in witness.flow) or any(not isinstance(x, F) for x in phi.values()):
        raise ValueError("exact rational witness values required")
    if phi[a] - phi[b] != 1:
        raise ValueError("potential port drop must be exactly one")
    div = {u: F(0) for u in adj}
    for (u, v, _), f in zip(rows, witness.flow):
        div[u] += f
        div[v] -= f
    if any(value != F(int(u == a)-int(u == b)) for u, value in div.items()):
        raise ValueError("flow does not exactly conserve the port injection")
    upper, energy = _energies(rows, witness.flow, phi)
    if energy <= 0 or lo != 1 / energy or hi != upper or lo > hi:
        raise ValueError("claimed endpoints do not equal the witness energies")
    return True


def exact_edge_resistance(edges, a, b):
    """Modest-graph full rational reference, reusing NT2's solve_right.

    Deliberately separate from the anytime operation. No certificate builder
    calls this function; it allocates and solves the full grounded matrix.
    """
    from ._rational_interface import solve_right
    a, b = _node(a), _node(b)
    rows, adj = _edges(edges, a, b)
    if a == b:
        return F(0)
    seen, stack = {b}, [b]
    while stack:
        for v, _, _ in adj[stack.pop()]:
            if v not in seen:
                seen.add(v); stack.append(v)
    if a not in seen:
        return math.inf
    nodes = sorted(seen - {b})
    index = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    matrix = [[F(0)] * n for _ in nodes]
    for u, v, c in rows:
        if u in index:
            matrix[index[u]][index[u]] += c
        if v in index:
            matrix[index[v]][index[v]] += c
        if u in index and v in index:
            matrix[index[u]][index[v]] -= c
            matrix[index[v]][index[u]] -= c
    rhs = [[F(int(u == a))] for u in nodes]
    return solve_right(matrix, rhs)[index[a]][0]
