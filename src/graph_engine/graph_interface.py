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

from dataclasses import dataclass

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

    @classmethod
    def export(cls, L: sp.spmatrix, shared: np.ndarray, estimate_g_with: int | None = None) -> "GraphInterface":
        """g = diag(A_ii⁻¹) is computed EXACTLY by default (one solve per interior node, in blocks). `estimate_g_with=k`
        replaces it by a k-sample Hutchinson estimate; then R_ab is no longer exact (measured on a 294-node interior:
        median relative error of g 0.155 at k = 128, 0.054 at k = 1 024)."""
        L = sp.csc_matrix(L); n = L.shape[0]
        shared = np.asarray(shared); interior = np.setdiff1d(np.arange(n), shared)
        Aii, Ais, Ass = L[interior][:, interior].tocsc(), L[interior][:, shared].toarray(), L[shared][:, shared].toarray()
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
        return self.S.nbytes + self.H.nbytes + self.g.nbytes


def _diag_inv_hutchinson(lu, n: int, k: int = 128, seed: int = 0) -> np.ndarray:
    V = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n, k))
    return np.einsum("ij,ij->i", V, lu.solve(V)) / k


def cross_resistance(A: GraphInterface, B: GraphInterface, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """R between interior node a[k] of A and interior node b[k] of B in the glued graph (positions into .interior)."""
    Sp = np.linalg.pinv(A.S + B.S, hermitian=True)
    d = A.H[a] - B.H[b]
    return A.g[a] + B.g[b] + np.einsum("ij,jk,ik->i", d, Sp, d)
