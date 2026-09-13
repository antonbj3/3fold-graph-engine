"""Shared linalg-primitive guards (agent pool linalg-silent-NaN / lstsq-hang class).

np.linalg.lstsq on a non-finite A/b HANGS in some BLAS/thread-pool environments (C/D observed: 'DLASCL parameter had an
illegal value' then hang) and raises LinAlgError + LAPACK spam in others. Either way non-finite input is never valid.
guarded_lstsq does a cheap pre-call isfinite check and raises a CLEAN ValueError, preventing the hang and the spam.
Same return signature as np.linalg.lstsq (solution, residuals, rank, singular_values) so call sites unpack unchanged.
"""
import numpy as np


def guarded_lstsq(A, b, rcond=None):
    A = np.asarray(A, float); b = np.asarray(b, float)
    if not (np.isfinite(A).all() and np.isfinite(b).all()):   # non-finite -> lstsq hangs (some BLAS) / LAPACK-spam+raise
        raise ValueError("guarded_lstsq: A/b must be finite (non-finite input hangs/errors LAPACK DLASCL)")
    return np.linalg.lstsq(A, b, rcond=rcond)


def guarded_eigh(A):
    """eigh silent-NaN guard (agent pool class): np.linalg.eigh on a non-finite matrix does NOT reliably raise -- it
    returns NaN eigenvalues/vectors and any downstream argmax/max/clip on them is SILENTLY WRONG (argmax(NaN)->0,
    clip(NaN)->NaN). Guard the MATRIX isfinite BEFORE the primitive. Same return sig (w, V)."""
    A = np.asarray(A, float)
    if not np.isfinite(A).all():
        raise ValueError("guarded_eigh: matrix must be finite (eigh on NaN silently returns NaN w/V -> wrong argmax/max/clip)")
    w, V = np.linalg.eigh(A)
    # /D agent pool gap: some BLAS return FINITE eigenVALUES but NaN-corrupted eigenVECTORS on a degenerate finite matrix
    # -> a guard that checks only w passes, and V@diag(w)@V.T / rotations / det(V) downstream are silently poisoned.
    # Guard the OUTPUT VECTORS too (BLAS-portable; my numpy 2.2.6 raises instead, but a consumer under another BLAS is covered).
    if not (np.isfinite(w).all() and np.isfinite(V).all()):
        raise np.linalg.LinAlgError("guarded_eigh: non-finite eigenvalues/eigenvectors from a finite matrix "
                                    "(degenerate/ill-conditioned -> BLAS returned NaN w/V; refuse to poison downstream)")
    return w, V


def guarded_eigvalsh(A):
    """eigvalsh silent-NaN guard (/ sibling): eigvalsh on a non-finite matrix returns NaN eigenvalues silently
    -> downstream min/max/clip/argmax WRONG. Guard the matrix isfinite before the primitive. Returns w (ascending)."""
    A = np.asarray(A, float)
    if not np.isfinite(A).all():
        raise ValueError("guarded_eigvalsh: matrix must be finite (eigvalsh on NaN -> silent NaN eigenvalues)")
    w = np.linalg.eigvalsh(A)
    if not np.isfinite(w).all():   # F513/D sibling: finite matrix -> NaN eigenvalues on some BLAS (BLAS-portable output guard)
        raise np.linalg.LinAlgError("guarded_eigvalsh: non-finite eigenvalues from a finite matrix (degenerate -> refuse)")
    return w


def guarded_svd(A, **kw):
    """svd silent-NaN guard (sibling): np.linalg.svd on a non-finite matrix returns NaN singular values/vectors silently
    (svd is silent on NaN AND inf). Guard the matrix isfinite before the primitive. Same return signature as np.linalg.svd."""
    A = np.asarray(A, float)
    if not np.isfinite(A).all():
        raise ValueError("guarded_svd: matrix must be finite (svd on NaN/inf -> silent NaN singular values)")
    out = np.linalg.svd(A, **kw)
    # /D sibling: finite matrix -> NaN singular VECTORS (U/Vt) with finite singular VALUES on some BLAS; a values-only
    # guard misses it and U@diag(s)@Vt / Kabsch rotations (Vt.T@U.T) / top-axis (U[:,0]) downstream are poisoned. Guard outputs.
    parts = out if isinstance(out, tuple) else (out,)   # compute_uv=False returns s alone; True returns (U, s, Vt)
    if not all(np.isfinite(np.asarray(p)).all() for p in parts):
        raise np.linalg.LinAlgError("guarded_svd: non-finite singular values/vectors from a finite matrix "
                                    "(degenerate/ill-conditioned -> BLAS returned NaN; refuse to poison downstream)")
    return out
