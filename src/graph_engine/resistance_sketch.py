#!/usr/bin/env python3
"""
resistance_sketch.py — the hole field L⁺_ii and effective resistance R_ij without forming L⁺.

graph_hole_engine.kernel.resolvent_leverage takes a dense pseudo-inverse: O(n³) time, O(n²) memory.
This module replaces that with two steps, both data-oblivious:

1. SKETCH (Spielman–Srivastava 2008). With L = Bᵀ W B (B = signed edge incidence, m×n) and Q a k×m
   matrix of ±1/√k signs,
        Z = L⁺ Bᵀ W^{1/2} Qᵀ            (n×k, k sparse Laplacian solves)
   satisfies E[Z Zᵀ] = L⁺ Bᵀ W B L⁺ = L⁺. So L⁺ is the Gram matrix of k-dim node vectors:
        L⁺_ij ≈ ⟨z_i, z_j⟩,   L⁺_ii ≈ ‖z_i‖²,   R_ij ≈ ‖z_i − z_j‖².
   Relative error of each quadratic form is about sqrt(2/k), independent of n.

2. QUANTIZE (the turbovec / TurboQuant recipe, arXiv:2504.19874, applied to node vectors instead of
   text embeddings). A sign-flip + Walsh–Hadamard rotation makes the coordinates of z_i/‖z_i‖
   exchangeable with a near-Gaussian marginal, so one fixed Lloyd–Max codebook for N(0,1/k) is
   near-optimal for every node; each coordinate is stored in `bits` bits. One float per node holds
   the RaBitQ-style scale ‖z‖/⟨u, û⟩ (arXiv:2405.12497) so that ⟨z_i, q⟩ estimated from the code has
   no length bias. turbovec does not implement the paper's 1-bit QJL residual; neither does this.

Memory per node: k·bits/8 + 4 bytes (k=64, 4 bit → 36 B) against 8·n bytes per node for dense L⁺.

Not claimed: numpy here does not reproduce turbovec's SIMD nibble-LUT kernels; the speed numbers in
examples/engine_experiments are about the sketch replacing the dense pinv, not about the scan kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["laplacian_from_edges", "ResistanceSketch", "QuantizedSketch", "lloyd_max_gaussian"]


def laplacian_from_edges(n: int, edges: np.ndarray, weights: np.ndarray | None = None):
    """(L, B, w): sparse Laplacian, signed incidence (m×n) and edge weights. Parallel edges add."""
    edges = np.asarray(edges, dtype=np.int64)
    m = len(edges)
    w = np.ones(m) if weights is None else np.asarray(weights, float)
    rows = np.repeat(np.arange(m), 2)
    cols = edges.reshape(-1)
    vals = np.tile([1.0, -1.0], m)
    B = sp.csr_matrix((vals, (rows, cols)), shape=(m, n))
    L = (B.T @ sp.diags(w) @ B).tocsc()
    return L, B, w


class _GroundedSolver:
    """x = L⁺ b for b ⟂ 1 on a CONNECTED graph. Small graphs: factor L with one node removed and
    re-center. Large graphs: Jacobi-preconditioned CG run on all right-hand sides at once (a sparse
    LU of a citation graph fills in; measured: no result in 10 min at n=27k, CG takes seconds)."""

    def __init__(self, L: sp.csc_matrix, direct_below: int = 6000, tol: float = 1e-6, maxiter: int = 2000):
        self.n = L.shape[0]
        self.L = L.tocsr()
        self.tol, self.maxiter = tol, maxiter
        self.lu = spla.splu(L[:-1, :-1].tocsc()) if self.n <= direct_below else None
        self.dinv = 1.0 / np.maximum(L.diagonal(), 1e-300)

    def solve(self, Bmat: np.ndarray) -> np.ndarray:
        Bmat = np.asarray(Bmat, float)
        if self.lu is not None:
            X = np.zeros((self.n, Bmat.shape[1]))
            X[:-1] = self.lu.solve(np.ascontiguousarray(Bmat[:-1]))
            return X - X.mean(0, keepdims=True)
        X = np.zeros_like(Bmat)
        R = Bmat.copy()
        Zr = R * self.dinv[:, None]
        P = Zr.copy()
        rz = np.einsum("ij,ij->j", R, Zr)
        b2 = np.einsum("ij,ij->j", Bmat, Bmat) + 1e-300
        for _ in range(self.maxiter):
            AP = self.L @ P
            alpha = rz / np.maximum(np.einsum("ij,ij->j", P, AP), 1e-300)
            X += P * alpha
            R -= AP * alpha
            if (np.einsum("ij,ij->j", R, R) / b2).max() < self.tol ** 2:
                break
            Zr = R * self.dinv[:, None]
            rz_new = np.einsum("ij,ij->j", R, Zr)
            P = Zr + P * (rz_new / np.maximum(rz, 1e-300))
            rz = rz_new
        return X - X.mean(0, keepdims=True)


@dataclass
class ResistanceSketch:
    Z: np.ndarray            # n×k, ⟨z_i,z_j⟩ ≈ L⁺_ij
    Y: np.ndarray | None     # n×k, ⟨y_i,y_j⟩ ≈ (L⁺²)_ij  (only if second_order)

    @classmethod
    def build(cls, n: int, edges: np.ndarray, weights: np.ndarray | None = None, k: int = 64,
              seed: int = 0, second_order: bool = False) -> "ResistanceSketch":
        rng = np.random.default_rng(seed)
        L, B, w = laplacian_from_edges(n, edges, weights)
        from scipy.sparse.csgraph import connected_components
        if connected_components(L, directed=False)[0] != 1:
            raise ValueError("resistance_sketch needs a connected graph: take the largest component first "
                             "(effective resistance between components is infinite)")
        solver = _GroundedSolver(L)
        Q = rng.choice([-1.0, 1.0], size=(len(w), k)) / np.sqrt(k)
        rhs = B.T @ (np.sqrt(w)[:, None] * Q)                 # n×k, every column ⟂ 1
        Z = solver.solve(np.asarray(rhs))
        Y = None
        if second_order:
            G = rng.standard_normal((n, k)) / np.sqrt(k)
            G -= G.mean(0, keepdims=True)
            Y = solver.solve(G)
        return cls(Z, Y)

    @classmethod
    def from_laplacian(cls, L, k: int = 64, seed: int = 0) -> "ResistanceSketch | None":
        """Sketch from a (dense or sparse) Laplacian. None if the graph is not connected: pinv then
        acts per component and the caller should keep the dense path."""
        from scipy.sparse.csgraph import connected_components
        L = sp.csr_matrix(L)
        if abs(L.sum(1)).max() > 1e-8 * max(abs(L.diagonal()).max(), 1.0) or (L - sp.diags(L.diagonal())).max() > 1e-12:
            raise ValueError("not a graph Laplacian: rows must sum to 0 and off-diagonal entries must be ≤ 0")
        A = sp.triu(-L, 1).tocoo()
        keep = A.data > 0
        if connected_components(sp.csr_matrix((A.data[keep], (A.row[keep], A.col[keep])), shape=L.shape), directed=False)[0] != 1:
            return None
        return cls.build(L.shape[0], np.stack([A.row[keep], A.col[keep]], 1), A.data[keep], k=k, seed=seed)

    def hole_field(self) -> np.ndarray:
        """≈ diag(L⁺), the same field GraphHoleEngine.hole_field() returns from the dense pinv."""
        return np.einsum("ij,ij->i", self.Z, self.Z)

    def refined_hole_field(self, L: sp.spmatrix) -> np.ndarray:
        """diag(L⁺) with the sketch used only for the OFF-diagonal part. Row i of L L⁺ = I − J/n gives
            L⁺_ii = (1 − 1/n + Σ_j w_ij L⁺_ij) / d_i ,
        so the exact local term 1/d_i carries no sketch noise, and the noisy terms ⟨z_i,z_j⟩ enter
        divided by d_i. The error of ⟨z_i,z_j⟩ scales with sqrt(L⁺_ii L⁺_jj) instead of L⁺_ii, which
        is small exactly where it matters: a weakly attached node next to a well-attached one."""
        n = L.shape[0]
        d = L.diagonal()
        Aw = sp.diags(d) - L
        return (1.0 - 1.0 / n + np.einsum("ij,ij->i", self.Z, Aw @ self.Z)) / d

    def resistance(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        d = self.Z[i] - self.Z[j]
        return np.einsum("ij,ij->i", d, d)

    def kirchhoff_drop(self, i: np.ndarray, j: np.ndarray, w: float = 1.0) -> np.ndarray:
        """Decrease of the Kirchhoff index n·tr(L⁺) if edge (i,j) of weight w were added.
        Sherman–Morrison on L⁺: Δtr = w‖L⁺(e_i−e_j)‖² / (1 + w R_ij). Needs second_order=True."""
        if self.Y is None:
            raise ValueError("build(..., second_order=True) first")
        d = self.Y[i] - self.Y[j]
        num = np.einsum("ij,ij->i", d, d)
        return len(self.Z) * w * num / (1.0 + w * self.resistance(i, j))


# ------------------------------------------------------------------------------------------------
# quantization
# ------------------------------------------------------------------------------------------------
def lloyd_max_gaussian(bits: int, iters: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """(levels, edges) of the MSE-optimal scalar quantizer of N(0,1) with 2**bits levels."""
    from scipy.stats import norm
    nl = 2 ** bits
    lev = norm.ppf((np.arange(nl) + 0.5) / nl)
    for _ in range(iters):
        e = np.concatenate([[-np.inf], (lev[1:] + lev[:-1]) / 2, [np.inf]])
        mass = norm.cdf(e[1:]) - norm.cdf(e[:-1])
        lev = (norm.pdf(e[:-1]) - norm.pdf(e[1:])) / mass     # E[x | bin] for the standard normal
    return lev, (lev[1:] + lev[:-1]) / 2


def _fwht(X: np.ndarray) -> np.ndarray:
    """Normalized Walsh–Hadamard transform along the last axis (length must be a power of two)."""
    X = X.copy()
    k = X.shape[-1]
    h = 1
    while h < k:
        X = X.reshape(*X.shape[:-1], -1, 2, h)
        a, b = X[..., 0, :].copy(), X[..., 1, :].copy()
        X[..., 0, :], X[..., 1, :] = a + b, a - b
        X = X.reshape(*X.shape[:-3], -1)
        h *= 2
    return X / np.sqrt(k)


@dataclass
class QuantizedSketch:
    codes: np.ndarray        # n×k uint8 level indices (packed size = k·bits/8 bytes per node)
    scale: np.ndarray        # n float32, ‖z‖/⟨u,û⟩
    levels: np.ndarray
    signs: np.ndarray
    perm: np.ndarray
    bits: int

    @classmethod
    def build(cls, Z: np.ndarray, bits: int = 4, seed: int = 0) -> "QuantizedSketch":
        n, k = Z.shape
        if k & (k - 1):
            raise ValueError("k must be a power of two for the Hadamard rotation")
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=k)
        perm = rng.permutation(k)
        levels, edges = lloyd_max_gaussian(bits)
        R = _fwht(Z[:, perm] * signs)
        norm = np.linalg.norm(R, axis=1) + 1e-300
        U = R / norm[:, None]
        codes = np.searchsorted(edges, U * np.sqrt(k)).astype(np.uint8)
        Uhat = levels[codes] / np.sqrt(k)
        scale = norm / np.maximum(np.einsum("ij,ij->i", U, Uhat), 1e-12)
        return cls(codes, scale.astype(np.float32), levels, signs, perm, bits)

    def nbytes(self) -> int:
        n, k = self.codes.shape
        return n * (k * self.bits // 8 + 4)

    def rotate(self, Q: np.ndarray) -> np.ndarray:
        return _fwht(np.atleast_2d(Q)[:, self.perm] * self.signs)

    def decode(self) -> np.ndarray:
        """Rotated-frame reconstruction; inner products and norms are rotation invariant."""
        k = self.codes.shape[1]
        return self.levels[self.codes] / np.sqrt(k) * self.scale[:, None]

    def hole_field(self) -> np.ndarray:
        """‖z_i‖² from the stored scale alone: scale·⟨u,û⟩ = ‖z‖, and ⟨u,û⟩ is recomputable from û."""
        k = self.codes.shape[1]
        Uhat = self.levels[self.codes] / np.sqrt(k)
        # ⟨u,û⟩ is not stored; ‖û‖² approximates it (Lloyd–Max centroids satisfy E[u·û]=E[û²]).
        return (self.scale * np.einsum("ij,ij->i", Uhat, Uhat)) ** 2

    def inner(self, q: np.ndarray) -> np.ndarray:
        """⟨z_i, q⟩ for every node i against one float query vector (asymmetric estimate)."""
        k = self.codes.shape[1]
        rq = self.rotate(q)[0]
        return (self.levels[self.codes] @ rq) / np.sqrt(k) * self.scale
