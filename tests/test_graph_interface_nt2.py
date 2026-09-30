"""All fourteen NT2 branch-B cases, against an independent full edge graph."""
import json
import math
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import pytest
from graph_engine.graph_interface import GraphInterface, cross_resistance
from graph_engine.resistance_sketch import laplacian_from_edges

CASES=json.loads((Path(__file__).parent/'fixtures/interface_nt2.json').read_text())


def full_edge_oracle(edges,weights):
    # Full graph: A interior 3, B interior 4. Traverse exact support, ground 3.
    adj=[set() for _ in range(5)]
    for a,b in edges:adj[a].add(b);adj[b].add(a)
    seen,stack={3},[3]
    while stack:
        for b in adj[stack.pop()]-seen:seen.add(b);stack.append(b)
    if 4 not in seen:return math.inf
    idx={i:k for k,i in enumerate(sorted(seen-{3}))};n=len(idx)
    L=[[F(0) for _ in range(n+1)] for _ in range(n)]
    for (a,b),w in zip(edges,weights):
        w=F(float(w))
        if a in idx:L[idx[a]][idx[a]]+=w
        if b in idx:L[idx[b]][idx[b]]+=w
        if a in idx and b in idx:L[idx[a]][idx[b]]-=w;L[idx[b]][idx[a]]-=w
    L[idx[4]][-1]=F(1)
    # Forward elimination plus back substitution; no interface matrices used.
    for k in range(n):
        p=next(i for i in range(k,n) if L[i][k]);L[k],L[p]=L[p],L[k]
        for i in range(k+1,n):
            factor=L[i][k]/L[k][k]
            L[i]=[a-factor*b for a,b in zip(L[i],L[k])]
    x=[F(0)]*n
    for i in range(n-1,-1,-1):x[i]=(L[i][-1]-sum(L[i][j]*x[j] for j in range(i+1,n)))/L[i][i]
    return float(x[idx[4]])


@pytest.mark.parametrize('case',CASES,ids=[c['case'] for c in CASES])
def test_nt2_against_full_edge_control(case):
    ea=np.asarray(case['ea'],int);eb=np.asarray(case['eb'],int)
    wa=np.asarray(case.get('wa',np.ones(len(ea))));wb=np.asarray(case.get('wb',np.ones(len(eb))))
    ga=ea.copy();ga[ga==3]=4
    oracle=full_edge_oracle(np.vstack([ga,eb]),np.r_[wa,wb])
    A=GraphInterface.export_edges(4,ea,wa,[0,1,2]);B=GraphInterface.export_edges(4,eb,wb,[0,1,2])
    actual=cross_resistance(A,B,[0],[0])[0]
    if math.isinf(oracle):assert math.isinf(actual)
    else:assert actual==pytest.approx(oracle,rel=1e-12)
    # Existing sparse-L entry point must also keep the off-diagonal tiny bridge.
    LA=laplacian_from_edges(4,ea,wa)[0];LB=laplacian_from_edges(4,eb,wb)[0]
    old_api=cross_resistance(GraphInterface.export(LA,np.arange(3)),GraphInterface.export(LB,np.arange(3)),[0],[0])[0]
    if math.isinf(oracle):assert math.isinf(old_api)
    else:assert old_api==pytest.approx(oracle,rel=1e-12)


def test_complete_nullspace_with_three_disconnected_boundary_components():
    A=GraphInterface.export_edges(6,[(0,3),(1,4),(2,5)],[1.,1.,1.],[0,1,2])
    B=GraphInterface.export_edges(6,[(0,3),(1,4),(2,5)],[1.,1.,1.],[0,1,2])
    r=cross_resistance(A,B,[0,0,1,2],[0,1,2,2])
    assert r[0]==r[-1]==2 and np.all(np.isinf(r[1:3]))


def test_floating_interfaces_have_complete_kernel_guard():
    A=GraphInterface(np.diag([0.,0.,1.]),np.array([[1.,0.,0.]]),np.array([1.]),np.array([3]))
    B=GraphInterface(np.zeros((3,3)),np.array([[0.,1.,0.]]),np.array([1.]),np.array([3]))
    assert math.isinf(cross_resistance(A,B,[0],[0])[0])


def test_isolated_interior_is_rejected_and_boundary_only_export_supported():
    with pytest.raises(ValueError,match='touch'):GraphInterface.export_edges(3,[(0,1)],[1.],[0])
    A=GraphInterface.export_edges(2,[(0,1)],[1.],[0,1])
    assert A.H.shape==(0,2) and A.S.shape==(2,2)
