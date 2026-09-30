"""Exact kernel reused from NT2/cross04; edge-native Schur algebra from NT2 branch B."""
from fractions import Fraction as F

def rref(M):
    M = [[F(x) for x in row] for row in M]
    m = len(M); n = len(M[0]) if m else 0
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, m) if M[i][c] != 0), None)
        if p is None: continue
        M[r], M[p] = M[p], M[r]
        inv = M[r][c]; M[r] = [x / inv for x in M[r]]
        for i in range(m):
            if i != r and M[i][c] != 0:
                fct = M[i][c]; M[i] = [a - fct * b for a, b in zip(M[i], M[r])]
        piv.append(c); r += 1
        if r == m: break
    return M, piv

def nullspace(M):
    if not M: return []
    R, piv = rref(M)
    n = len(M[0]); cols = []
    for fc in [c for c in range(n) if c not in piv]:
        v = [F(0)] * n; v[fc] = F(1)
        for i, c in enumerate(piv): v[c] = -R[i][fc]
        cols.append(v)
    return cols

def solve_right(M, B):
    A = [r[:] for r in M]; X = [r[:] for r in B]; n = len(A)
    for c in range(n):
        p = next((i for i in range(c, n) if A[i][c] != 0), None)
        if p is None: raise ZeroDivisionError('singular')
        A[c], A[p] = A[p], A[c]; X[c], X[p] = X[p], X[c]
        inv = A[c][c]; A[c] = [x / inv for x in A[c]]; X[c] = [x / inv for x in X[c]]
        for i in range(n):
            if i != c and A[i][c] != 0:
                fct = A[i][c]; A[i] = [x - fct * y for x, y in zip(A[i], A[c])]
                X[i] = [x - fct * y for x, y in zip(X[i], X[c])]
    return X
