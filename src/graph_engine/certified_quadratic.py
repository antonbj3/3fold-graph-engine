"""Certified signs of h.T J^-1 h for symmetric strictly dominant SPD forms.

The precision lower spectral bound alpha is proved from the exact matrix,
never inferred from floating eigenvalues. PCG only proposes x. Verification
recomputes the primal value and residual correction exactly. This is not an
arbitrary PSD or general log-det certificate.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import math
from numbers import Integral
from time import perf_counter

from .certified_decision import _threshold, decision_verdict


def _problem(matrix, h):
    J=tuple(tuple(_threshold(c) for c in row) for row in matrix)
    h=tuple(_threshold(c) for c in h)
    n=len(h)
    if n==0 or len(J)!=n or any(len(row)!=n for row in J):
        raise ValueError('matrix and h shape mismatch')
    if any(J[i][j]!=J[j][i] for i in range(n) for j in range(i)):
        raise ValueError('symmetric matrix required')
    alpha=min(J[i][i]-sum((abs(J[i][j]) for j in range(n) if j!=i),F(0)) for i in range(n))
    if alpha<=0:
        raise ValueError('positive exact Gershgorin bound required')
    digest=hashlib.sha256(b'T7-quadratic-v1\n')
    for row in J+(h,):
        digest.update((';'.join(f'{c.numerator}/{c.denominator}' for c in row)+'\n').encode('ascii'))
    return J,h,alpha,digest.hexdigest()


def _bounds(J,h,x,alpha):
    Jx=[sum((c*y for c,y in zip(row,x)),F(0)) for row in J]
    lower=2*sum((c*y for c,y in zip(h,x)),F(0))-sum((c*y for c,y in zip(x,Jx)),F(0))
    residual2=sum(((c-y)**2 for c,y in zip(h,Jx)),F(0))
    return lower,lower+residual2/alpha


@dataclass(frozen=True)
class QuadraticWitness:
    problem_sha256: str
    x: tuple
    alpha: F
    lower: F
    upper: F
    theta: F
    updates: int
    elapsed_seconds: float


def certified_quadratic_decision(matrix,h,theta,max_updates,*,observer=None):
    started=perf_counter()
    theta=_threshold(theta)
    if isinstance(max_updates,bool) or not isinstance(max_updates,Integral) or max_updates<0:
        raise ValueError('nonnegative update budget required')
    J,h,alpha,digest=_problem(matrix,h)
    exact_x=tuple(F(0) for _ in h)
    lower,upper=_bounds(J,h,exact_x,alpha)
    updates=0
    def finish():
        w=QuadraticWitness(digest,exact_x,alpha,lower,upper,theta,updates,perf_counter()-started)
        return {'verdict':decision_verdict(lower,upper,theta),'updates_used':updates,'witness':w}
    if observer:observer(updates,lower,upper)
    if decision_verdict(lower,upper,theta)!='UNDECIDED' or lower==upper or max_updates==0:
        return finish()
    import numpy as np
    A=np.array(J,dtype=float); rhs=np.array(h,dtype=float)
    scale=float(max(max(abs(c) for c in row) for row in J))
    if not math.isfinite(scale) or scale<=0: return finish()
    A=A/scale;rhs=rhs/scale
    diagonal=np.diag(A)
    x=np.zeros(len(h));r=rhs.copy();z=r/diagonal;p=z.copy();rz=float(r@z)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        while updates<max_updates:
            ap=A@p;denom=float(p@ap)
            if not math.isfinite(denom) or denom<=0 or not math.isfinite(rz) or rz<=0:break
            step=rz/denom;x+=step*p;r-=step*ap
            if not np.all(np.isfinite(x)):break
            proposal=tuple(F(float(c)) for c in x)
            pl,pu=_bounds(J,h,proposal,alpha)
            # One field proves both sides; retain the narrower prior field if
            # the exact interval width deteriorates under floating proposals.
            if pu-pl<=upper-lower:
                exact_x,lower,upper=proposal,pl,pu
            updates+=1
            if observer:observer(updates,lower,upper)
            if decision_verdict(lower,upper,theta)!='UNDECIDED' or lower==upper:break
            z=r/diagonal;next_rz=float(r@z);p=z+(next_rz/rz)*p;rz=next_rz
    return finish()


def verify_quadratic_witness(matrix,h,theta,result):
    J,h,alpha,digest=_problem(matrix,h)
    w=result.get('witness')
    if not isinstance(w,QuadraticWitness):raise ValueError('invalid quadratic witness')
    if digest!=w.problem_sha256:raise ValueError('problem hash mismatch')
    if w.theta!=_threshold(theta):raise ValueError('query binding mismatch')
    if len(w.x)!=len(h) or any(not isinstance(c,F) for c in w.x):raise ValueError('exact field required')
    lower,upper=_bounds(J,h,w.x,alpha)
    if (alpha,lower,upper)!=(w.alpha,w.lower,w.upper):raise ValueError('bounds mismatch')
    if result.get('verdict')!=decision_verdict(lower,upper,w.theta):raise ValueError('verdict mismatch')
    if result.get('updates_used')!=w.updates:raise ValueError('update count mismatch')
    return True
