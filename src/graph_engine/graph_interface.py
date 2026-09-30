#!/usr/bin/env python3
"""
graph_interface.py — combine two graphs through their shared concepts WITHOUT exchanging their interiors.

Two graphs A and B share a concept set S and nothing else. Glue them: L = L_A ⊕ L_B with the rows of S
identified. Order A's nodes as (interior i, shared s):  L_A = [[A_ii, A_is], [A_si, A_ss]].

Each graph exports three things (its INTERFACE on S):
    S_A = A_ss − A_si A_ii⁻¹ A_is        |S|×|S|   Schur complement = Kron reduction of A onto S
    H_A = −A_ii⁻¹ A_is                   n_i×|S|   row a = h_a, the harmonic measure of interior node a:
                                                   h_a[s] = P(random walk from a first reaches S at s); rows sum to 1
    g_A = diag(A_ii⁻¹)                   n_i       resistance from a to S with S shorted together
Then, exactly, for a interior to A and b interior to B:

    R_ab = g_a + g_b + (h_a − h_b)ᵀ (S_A + S_B)⁺ (h_a − h_b)

Proof. Unit current in at a, out at b. Eliminating A's interior (block Gaussian elimination) leaves, on S,
the Laplacian S_A + S_B with injected current h_a − h_b (a's current reaches S distributed as h_a; both rows
sum to 1, so the injection is balanced). The potential drop inside A from a to the S-potentials it induces
is g_a, likewise g_b in B. Same node-to-node formula within one graph with S_A + S_B in place of S_A shows how
B changes distances INSIDE A. This is Kron reduction (Dörfler & Bullo 2013) / Schur substructuring; the use
here is the federation boundary: a private graph exports (S, H, g) and its interior never leaves.

Representation. h_a is a point on the simplex over the shared concepts — a node described by typed
probabilities over a runtime-chosen vocabulary. Two nodes from different graphs become comparable in one
space without any text or embedding model, and the simplex has the same Fisher geometry as the probe value
in regime_posterior (θ = 2·arcsin√h makes it flat).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from ._rational_interface import nullspace, solve_right, rref

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["GraphInterface", "cross_resistance"]


@dataclass
class GraphInterface:
    S: np.ndarray            # |S|×|S|
    H: np.ndarray            # n_interior×|S|
    g: np.ndarray            # n_interior
    interior: np.ndarray     # original indices of the interior nodes

    exact_S: list | None = field(default=None, repr=False)
    exact_H: list | None = field(default=None, repr=False)
    exact_g: list | None = field(default=None, repr=False)

    @classmethod
    def export_edges(cls, n, edges, weights, shared):
        """Exact binary64 conductances before Schur reduction (NT2 branch B).

        Fractions preserve weak positive couplings. Intended as the reference
        exporter for modest graphs; rational elimination can be expensive.
        Every interior component must touch a shared node.
        """
        shared = np.asarray(shared, dtype=int)
        if shared.ndim != 1 or len(set(shared)) != len(shared) or np.any(shared < 0) or np.any(shared >= n):
            raise ValueError("shared nodes must be distinct valid indices")
        edges = np.asarray(edges, dtype=int).reshape(-1, 2)
        weights = np.ones(len(edges)) if weights is None else np.asarray(weights, float)
        if weights.shape != (len(edges),) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("strictly positive finite edge conductances required")
        if np.any(edges < 0) or np.any(edges >= n):
            raise ValueError("edge endpoint outside graph")
        L = [[Fraction(0) for _ in range(n)] for _ in range(n)]
        for (a, b), w in zip(edges, weights):
            w = Fraction(float(w))
            L[a][a] += w; L[b][b] += w
            L[a][b] -= w; L[b][a] -= w
        interior = np.setdiff1d(np.arange(n), shared)
        Aii = [[L[i][j] for j in interior] for i in interior]
        Ais = [[L[i][j] for j in shared] for i in interior]
        Ass = [[L[i][j] for j in shared] for i in shared]
        ni, ns = len(interior), len(shared)
        identity = [[Fraction(int(i == j)) for j in range(ni)] for i in range(ni)]
        try:
            inv = solve_right(Aii, identity)
        except ZeroDivisionError as exc:
            raise ValueError("every interior component must touch the interface") from exc
        X = [[sum(inv[i][r] * Ais[r][j] for r in range(ni)) for j in range(ns)] for i in range(ni)]
        S = [[Ass[i][j] - sum(Ais[r][i] * X[r][j] for r in range(ni)) for j in range(ns)] for i in range(ns)]
        H, g = [[-v for v in row] for row in X], [inv[i][i] for i in range(ni)]
        return cls(np.asarray(S, float).reshape(ns, ns), np.asarray(H, float).reshape(ni, ns),
                   np.asarray(g, float), interior, S, H, g)

    @classmethod
    def export(cls, L: sp.spmatrix, shared: np.ndarray, estimate_g_with: int | None = None) -> "GraphInterface":
        """g = diag(A_ii⁻¹) is computed EXACTLY by default (one solve per interior node, in blocks). `estimate_g_with=k`
        replaces it by a k-sample Hutchinson estimate; then R_ab is no longer exact (measured on a 294-node interior:
        median relative error of g 0.155 at k = 128, 0.054 at k = 1 024)."""
        L = sp.csc_matrix(L); n = L.shape[0]
        shared = np.asarray(shared)
        # For small combinatorial Laplacians recover the edge-native inputs
        # from off-diagonals, before a floating Schur subtraction can lose them.
        if n <= 16 and estimate_g_with is None:
            coo = sp.triu(L, k=1).tocoo()
            weights = -coo.data
            diagonal = np.asarray(L.diagonal())
            degree = -np.asarray(L.sum(axis=1)).ravel() + diagonal
            if np.all(weights > 0) and np.allclose(diagonal, degree, rtol=1e-12, atol=0):
                return cls.export_edges(n, np.column_stack((coo.row, coo.col)), weights, shared)
        interior = np.setdiff1d(np.arange(n), shared)
        Aii, Ais, Ass = L[interior][:, interior].tocsc(), L[interior][:, shared].toarray(), L[shared][:, shared].toarray()
        if not len(interior):
            return cls(Ass, np.empty((0, len(shared))), np.empty(0), interior)
        lu = spla.splu(Aii)
        X = lu.solve(Ais)                                        # A_ii⁻¹ A_is
        ni = len(interior)
        if estimate_g_with:
            g = _diag_inv_hutchinson(lu, ni, k=estimate_g_with)
        else:
            g = np.empty(ni)
            for a in range(0, ni, 512):
                b = min(a + 512, ni); E = np.zeros((ni, b - a)); E[np.arange(a, b), np.arange(b - a)] = 1.0
                g[a:b] = lu.solve(E)[np.arange(a, b), np.arange(b - a)]
        return cls(Ass - Ais.T @ X, -X, g, interior)

    def nbytes(self) -> int:
        import sys
        arrays = self.S.nbytes + self.H.nbytes + self.g.nbytes + self.interior.nbytes
        def size(value):
            if isinstance(value, list):
                return sys.getsizeof(value) + sum(size(x) for x in value)
            if isinstance(value, Fraction):
                return sys.getsizeof(value) + sys.getsizeof(value.numerator) + sys.getsizeof(value.denominator)
            return 0
        return arrays + sum(size(x) for x in (self.exact_S, self.exact_H, self.exact_g))


def _diag_inv_hutchinson(lu, n: int, k: int = 128, seed: int = 0) -> np.ndarray:
    V = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n, k))
    return np.einsum("ij,ij->i", V, lu.solve(V)) / k


def cross_resistance(A: GraphInterface, B: GraphInterface, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """R between interior node a[k] of A and interior node b[k] of B in the glued graph (positions into .interior)."""
    a, b = np.broadcast_arrays(np.atleast_1d(a).astype(int), np.atleast_1d(b).astype(int))
    if A.S.shape != B.S.shape:
        raise ValueError("interfaces must share the same ordered boundary")
    if A.exact_S is not None and B.exact_S is not None:
        S = [[x+y for x,y in zip(ra,rb)] for ra,rb in zip(A.exact_S,B.exact_S)]
        N = nullspace(S)  # all modes, not merely the global constant
        out = []
        for ai, bi in zip(a.flat, b.flat):
            d = [x-y for x,y in zip(A.exact_H[ai],B.exact_H[bi])]
            if any(sum(x*y for x,y in zip(mode,d)) for mode in N):
                out.append(float("inf")); continue
            # A particular solution to S x=d suffices: energy is gauge invariant.
            R, piv = rref([row+[rhs] for row,rhs in zip(S,d)])
            x = [Fraction(0) for _ in d]
            for row, pivot in zip(R, piv):
                x[pivot] = row[-1]
            out.append(float(A.exact_g[ai]+B.exact_g[bi]+sum(u*v for u,v in zip(d,x))))
        return np.asarray(out).reshape(a.shape)
    S = (A.S + B.S + A.S.T + B.S.T) / 2
    lam, V = np.linalg.eigh(S)
    tol = np.finfo(float).eps * max(len(S), 1) * max(float(np.max(np.abs(lam), initial=0)), 1e-300) * 8
    if np.any(lam < -tol):
        raise ValueError("interface Schur form must be positive semidefinite")
    active = lam > tol
    N = V[:, ~active]
    Sp = (V[:, active] / lam[active]) @ V[:, active].T
    d = A.H[a] - B.H[b]
    out = A.g[a] + B.g[b] + np.einsum("...i,ij,...j->...", d, Sp, d)
    # Floating path has a numerical rank tolerance; use export_edges when
    # resolving extremely weak positive couplings is part of the contract.
    residual = np.linalg.norm(d @ N, axis=-1)
    feasible = residual <= 1e-10 * np.maximum(np.linalg.norm(d, axis=-1), 1.0)
    return np.where(feasible, out, np.inf)
