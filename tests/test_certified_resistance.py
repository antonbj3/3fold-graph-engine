"""Exact containment, monotonicity, tamper rejection and no-solve contract."""
import json
import math
import random
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path

import pytest

from graph_engine.certified_resistance import (
    ResistanceBudget, certified_cross_resistance, exact_edge_resistance,
    verify_resistance_witness)
from graph_engine.graph_interface import GraphInterface, cross_resistance


def independent_oracle(edges, a, b):
    """Full edge graph, forward Gauss plus back substitution, no engine solve."""
    adj = {a: set(), b: set()}
    for u, v, _ in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    seen, stack = {b}, [b]
    while stack:
        for u in adj[stack.pop()] - seen:
            seen.add(u); stack.append(u)
    if a not in seen:
        return math.inf
    if a == b:
        return F(0)
    index = {u: i for i, u in enumerate(sorted(seen - {b}))}
    n = len(index)
    matrix = [[F(0)] * (n + 1) for _ in range(n)]
    for u, v, raw in edges:
        c = F(raw)
        if u in index:
            matrix[index[u]][index[u]] += c
        if v in index:
            matrix[index[v]][index[v]] += c
        if u in index and v in index:
            matrix[index[u]][index[v]] -= c
            matrix[index[v]][index[u]] -= c
    matrix[index[a]][-1] = F(1)
    for k in range(n):
        pivot = next(i for i in range(k, n) if matrix[i][k])
        matrix[k], matrix[pivot] = matrix[pivot], matrix[k]
        for i in range(k + 1, n):
            if matrix[i][k]:
                ratio = matrix[i][k] / matrix[k][k]
                for j in range(k, n + 1):
                    matrix[i][j] -= ratio * matrix[k][j]
    x = [F(0)] * n
    for i in reversed(range(n)):
        x[i] = (matrix[i][-1] - sum(matrix[i][j]*x[j] for j in range(i+1, n))) / matrix[i][i]
    return x[index[a]]


def random_cases(count=500):
    rng = random.Random(2026100101)
    for k in range(count):
        n = rng.randint(2, 18)
        edges = []
        disconnected = k % 7 == 0
        split = rng.randint(1, n-1)
        for u in range(n):
            for v in range(u+1, n):
                if disconnected and (u < split) != (v < split):
                    continue
                if v == u+1 or rng.random() < 0.25:
                    pool = [1, 2, 7, F(1, 3), F(3, 7), .1, 1e-18]
                    c = rng.choice(pool)
                    if k % 31 == 0:
                        c = F(c) * F(10) ** (150 if u % 2 else -150)
                    edges.append((u, v, c))
                    if rng.random() < .07:
                        edges.append((v, u, rng.choice(pool)))
        if k % 9 == 0:
            edges.append((0, 0, 13))
        # include isolated nodes through port indices
        yield k, edges, 0, n-1


@pytest.mark.parametrize("k,edges,a,b", list(random_cases()), ids=lambda x: str(x) if isinstance(x, int) else None)
def test_500_random_exact_enclosures_and_nesting(k, edges, a, b):
    exact = independent_oracle(edges, a, b)
    previous = (F(0), math.inf)
    for updates in (0, 16, 80):
        lo, hi, w = certified_cross_resistance(edges, a, b, updates)
        assert verify_resistance_witness(edges, a, b, lo, hi, w)
        assert lo <= exact <= hi
        assert previous[0] <= lo <= hi <= previous[1]
        previous = lo, hi
        if math.isinf(exact):
            assert lo == hi == math.inf and w.kind == "disconnected"
    if k % 50 == 0:
        assert exact_edge_resistance(edges, a, b) == exact


NT2 = json.loads((Path(__file__).parent / "fixtures/interface_nt2.json").read_text())


@pytest.mark.parametrize("case", NT2, ids=[c["case"] for c in NT2])
def test_all_14_nt2_and_export_edges(case):
    ea, eb = case["ea"], case["eb"]
    wa, wb = case.get("wa", [1]*len(ea)), case.get("wb", [1]*len(eb))
    edges = [(4 if u == 3 else u, 4 if v == 3 else v, c) for (u,v),c in zip(ea,wa)]
    edges += [(u,v,c) for (u,v),c in zip(eb,wb)]
    exact = independent_oracle(edges, 4, 3)
    prev = F(0), math.inf
    for steps in (0, 16, 128):
        lo, hi, w = certified_cross_resistance(edges, 4, 3, steps)
        assert verify_resistance_witness(edges, 4, 3, lo, hi, w)
        assert prev[0] <= lo <= exact <= hi <= prev[1]
        prev = lo, hi
    assert exact_edge_resistance(edges, 4, 3) == exact
    A = GraphInterface.export_edges(4, ea, wa, [0,1,2])
    B = GraphInterface.export_edges(4, eb, wb, [0,1,2])
    assert cross_resistance(A,B,[0],[0])[0] == pytest.approx(float(exact), rel=1e-12)


def test_never_calls_full_elimination(monkeypatch):
    import graph_engine._rational_interface as rational
    def forbidden(*args, **kwargs):
        raise AssertionError("candidate attempted a full solve")
    monkeypatch.setattr(rational, "solve_right", forbidden)
    monkeypatch.setattr(rational, "rref", forbidden)
    monkeypatch.setattr(GraphInterface, "export_edges", forbidden)
    edges = [(0,1,1),(1,2,1),(0,2,1)]
    lo, hi, w = certified_cross_resistance(edges, 0, 2, 128)
    assert verify_resistance_witness(edges, 0, 2, lo, hi, w)
    assert lo <= F(2,3) <= hi
    assert hi-lo < F(1,10**9)


def test_witness_tampering_is_rejected():
    edges = [(0,1,1),(1,2,1),(0,2,1)]
    lo, hi, w = certified_cross_resistance(edges, 0, 2, 8)
    with pytest.raises(ValueError, match="conserve"):
        verify_resistance_witness(edges,0,2,lo,hi,replace(w,flow=(w.flow[0]+1,)+w.flow[1:]))
    with pytest.raises(ValueError, match="drop"):
        verify_resistance_witness(edges,0,2,lo,hi,replace(w,potential=((0,F(2)),)+w.potential[1:]))
    with pytest.raises(ValueError, match="energies"):
        verify_resistance_witness(edges,0,2,lo+1,hi,w)
    lo, hi, w = certified_cross_resistance([(0,1,1),(2,3,1)],0,3,0)
    with pytest.raises(ValueError,match="crossing"):
        verify_resistance_witness([(0,1,1),(2,3,1),(1,2,1)],0,3,lo,hi,w)


def test_deadline_and_empty_or_same_ports():
    edges = [(0,1,1)]
    lo, hi, w = certified_cross_resistance(edges,0,1,ResistanceBudget(seconds=0))
    assert (lo,hi,w.kind) == (0,math.inf,"uninformative")
    assert verify_resistance_witness(edges,0,1,lo,hi,w)
    for a,b,result in [(0,0,(0,0)),(0,1,(math.inf,math.inf))]:
        lo,hi,w = certified_cross_resistance([],a,b,0)
        assert (lo,hi) == result
        assert verify_resistance_witness([],a,b,lo,hi,w)


@pytest.mark.parametrize("c", [0,-1,float("nan"),float("inf"),True,"1"])
def test_invalid_conductance(c):
    with pytest.raises((ValueError,TypeError)):
        certified_cross_resistance([(0,1,c)],0,1,0)


def test_rational_series_parallel_extremes_and_exact_integer_input():
    for edges,a,b,r in [
        ([(0,1,F(1,3)),(1,2,F(1,7))],0,2,F(10)),
        ([(0,1,F(1,3)),(0,1,F(1,7))],0,1,F(21,10)),
        ([(0,1,2**53+1)],0,1,F(1,2**53+1)),
        ([(0,1,F(1,10**600)),(1,2,F(10**600))],0,2,F(10**600)+F(1,10**600)),
    ]:
        lo,hi,w = certified_cross_resistance(edges,a,b,80)
        assert verify_resistance_witness(edges,a,b,lo,hi,w)
        assert lo <= r <= hi


@pytest.mark.parametrize("k,edges,a,b", list(random_cases()))
def test_500_tree_corrected_exact_enclosures(k, edges, a, b):
    exact = independent_oracle(edges, a, b)
    previous = F(0), math.inf
    for updates in (0, 16, 80):
        budget = ResistanceBudget(max_updates=updates, refinement="tree")
        lo,hi,w = certified_cross_resistance(edges,a,b,budget)
        assert verify_resistance_witness(edges,a,b,lo,hi,w)
        assert previous[0] <= lo <= exact <= hi <= previous[1]
        previous = lo,hi


@pytest.mark.parametrize("case", NT2, ids=[c["case"] for c in NT2])
def test_all_14_nt2_with_tree_correction(case):
    ea,eb = case['ea'],case['eb']
    wa,wb = case.get('wa',[1]*len(ea)),case.get('wb',[1]*len(eb))
    edges=[(4 if u==3 else u,4 if v==3 else v,c) for (u,v),c in zip(ea,wa)]
    edges += [(u,v,c) for (u,v),c in zip(eb,wb)]
    exact = independent_oracle(edges,4,3)
    lo,hi,w=certified_cross_resistance(edges,4,3,ResistanceBudget(max_updates=128,refinement='tree'))
    assert verify_resistance_witness(edges,4,3,lo,hi,w)
    assert lo <= exact <= hi


@pytest.mark.parametrize("k,edges,a,b", list(random_cases()))
def test_500_cg_proposal_exact_enclosures(k, edges, a, b):
    exact = independent_oracle(edges,a,b)
    previous = F(0), math.inf
    for updates in (0,1,4):
        lo,hi,w=certified_cross_resistance(edges,a,b,ResistanceBudget(max_updates=updates,refinement='cg'))
        assert verify_resistance_witness(edges,a,b,lo,hi,w)
        assert previous[0] <= lo <= exact <= hi <= previous[1]
        previous=lo,hi


@pytest.mark.parametrize("case", NT2, ids=[c["case"] for c in NT2])
def test_all_14_nt2_with_cg_proposals(case):
    ea,eb=case['ea'],case['eb']
    wa,wb=case.get('wa',[1]*len(ea)),case.get('wb',[1]*len(eb))
    edges=[(4 if u==3 else u,4 if v==3 else v,c) for (u,v),c in zip(ea,wa)]
    edges += [(u,v,c) for (u,v),c in zip(eb,wb)]
    exact=independent_oracle(edges,4,3)
    lo,hi,w=certified_cross_resistance(edges,4,3,ResistanceBudget(max_updates=4,refinement='cg'))
    assert verify_resistance_witness(edges,4,3,lo,hi,w)
    assert lo <= exact <= hi


@pytest.mark.parametrize("k,edges,a,b", list(random_cases()))
def test_500_batched_cg_retained_exact_enclosures(k,edges,a,b):
    exact=independent_oracle(edges,a,b)
    previous=F(0),math.inf
    for updates in (0,1,4,16):
        lo,hi,w=certified_cross_resistance(edges,a,b,ResistanceBudget(max_updates=updates,refinement='cg',certify_every=4))
        assert verify_resistance_witness(edges,a,b,lo,hi,w)
        assert previous[0] <= lo <= exact <= hi <= previous[1]
        previous=lo,hi


@pytest.mark.parametrize('method', ['cycle','tree','cg'])
def test_short_deadlines_keep_exact_retained_fields(method):
    edges=[(u,(u+1)%24,1) for u in range(24)]
    edges += [(u,(u+7)%24,F(1,3)) for u in range(24)]
    exact=independent_oracle(edges,0,12)
    for seconds in (0,.0001,.002,.01):
        lo,hi,w=certified_cross_resistance(edges,0,12,ResistanceBudget(seconds=seconds,refinement=method,certify_every=4))
        assert verify_resistance_witness(edges,0,12,lo,hi,w)
        assert lo <= exact <= hi
