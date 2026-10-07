"""Exact algebra, adversarial partitions, native feasibility and compatibility."""
from fractions import Fraction as F
import heapq
import random

import pytest

from graph_engine.certified_resistance import (
    _edges, ResistanceBudget, certified_cross_resistance, verify_resistance_witness)
from graph_engine.instance_slack import distance_quotient_potential
from graph_engine.potential_side import residual_quotient_potential, quotient_solve
from graph_engine._rational_interface import solve_right
from test_certified_resistance import independent_oracle

def dist_adj(edges,a,b):
    rows,adj = _edges(edges,a,b)
    dist,heap,done = {b:F(0)},[(F(0),b)],set()
    while heap:
        d,u = heapq.heappop(heap)
        if u in done:continue
        done.add(u)
        for v,i,_ in adj[u]:
            nd = d+1/rows[i][2]
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap,(nd,v))
    return rows,adj,dist

def energy(edges,phi):
    return sum((F(c)*(phi[u]-phi[v])**2 for u,v,c in edges),F(0))

@pytest.mark.parametrize('seed',range(100))
def test_nested_partition_exact_containment_and_legacy_optimum(seed):
    rng = random.Random(202610011500+seed)
    n = rng.randint(4,16)
    edges = [(u,u+1,F(1)) for u in range(n-1)]
    edges += [(u,v,rng.choice([F(1),F(2),F(1,3)]))
              for u in range(n) for v in range(u+2,n) if rng.random()<.3]
    edges += [(0,0,F(10)),(1,0,F(1,5))]
    a,b = rng.sample(range(n),2)
    rows,adj,dist = dist_adj(edges,a,b)
    reference = independent_oracle(edges,a,b)
    oldphi,_ = distance_quotient_potential(rows,adj,a,b,dist,condition_ports=True)
    last = None
    def check(step,phi,receipt):
        nonlocal last
        e = energy(edges,phi)
        assert e == F(receipt['energy'])
        assert 1/e <= reference
        assert last is None or e <= last
        if step == 0:assert e == energy(edges,oldphi)
        last = e
    phi,_ = residual_quotient_potential(rows,adj,a,b,dist,splits=n,checkpoint=check)
    assert 1/energy(edges,phi) == reference
    lo,hi,w = certified_cross_resistance(edges,a,b,ResistanceBudget(max_updates=32),
                                        initial_potential='residual_quotient',potential_splits=3)
    assert verify_resistance_witness(edges,a,b,lo,hi,w)
    assert lo <= reference <= hi

@pytest.mark.parametrize('n',[2,3,8,32,128])
def test_three_distance_classes_have_unbounded_loss_and_score_misranks(n):
    # s=0, t=1, load-bearing u=2, dead leaves >=3.
    edges = [(0,2,F(n*n)),(2,1,F(1))]+[(v,1,F(1)) for v in range(3,n+2)]
    rows,adj,dist = dist_adj(edges,0,1)
    phi,receipt = residual_quotient_potential(rows,adj,0,1,dist,splits=0)
    r = F(1)+F(1,n*n)
    lo = 1/energy(edges,phi)
    assert receipt['quotient_groups'] == 3
    assert lo == F(1,n)+F(1,n*n)
    assert r/lo == F(n*n+1,n+1)
    _,details = residual_quotient_potential(rows,adj,0,1,dist,splits=1)
    assert details['selected_nodes'][0] != 2
    schur,details = residual_quotient_potential(rows,adj,0,1,dist,splits=1,score_mode='schur')
    assert details['selected_nodes'][0] == 2
    assert 1/energy(edges,schur) == r
    # Break out the bearing node directly, proving one class suffices.
    graph = {0:{2:F(n*n)},1:{2:F(1),3:F(n-1)},
             2:{0:F(n*n),1:F(1)},3:{1:F(n-1)}}
    gp,_ = quotient_solve(graph,0,1)
    fullphi = {0:gp[0],1:gp[1],2:gp[2],**{v:gp[3] for v in range(3,n+2)}}
    assert 1/energy(edges,fullphi) == r

def test_singleton_schur_gain_equals_full_reoptimization():
    edges = [(0,2,F(16)),(2,1,F(1)),(3,1,F(1)),(4,1,F(1)),(5,1,F(1))]
    rows,adj,dist = dist_adj(edges,0,1)
    phi,_ = residual_quotient_potential(rows,adj,0,1,dist,splits=0)
    # Single free class C={2,3,4,5}, A=C^T K C=20.
    A, v, kuu = F(20), F(17), F(17)
    ru = sum((rows[i][2]*(phi[2]-phi[w]) for w,i,_ in adj[2]),F(0))
    h = kuu-v*v/A
    graph = {0:{2:F(16)},1:{2:F(1),3:F(3)},2:{0:F(16),1:F(1)},3:{1:F(3)}}
    gp,_ = quotient_solve(graph,0,1)
    newphi = {0:gp[0],1:gp[1],2:gp[2],3:gp[3],4:gp[3],5:gp[3]}
    assert energy(edges,phi)-energy(edges,newphi) == ru*ru/h
    assert ru*ru/h > ru*ru/kuu

def test_unit_drop_error_energy_identity_against_independent_gauss():
    edges = [(0,2,F(16)),(2,1,F(1)),(3,1,F(1)),(4,1,F(1)),(5,1,F(1))]
    reference = independent_oracle(edges,0,1)
    graph = {u:{} for u in range(6)}
    for u,v,c in edges:graph[u][v]=graph[v][u]=c
    optimal,_ = quotient_solve(graph,0,1)
    assert energy(edges,optimal) == 1/reference
    rows,adj,dist = dist_adj(edges,0,1)
    phi,_ = residual_quotient_potential(rows,adj,0,1,dist,splits=0)
    error = {u:phi[u]-optimal[u] for u in phi}
    assert energy(edges,error) > 0
    assert reference*energy(edges,phi) == 1+reference*energy(edges,error)

@pytest.mark.parametrize('bad',[-1,True,1.5])
def test_invalid_split_count_rejected(bad):
    with pytest.raises(ValueError):
        certified_cross_resistance([(0,1,1)],0,1,0,potential_splits=bad)


@pytest.mark.parametrize('bad', [-1, True, 1.5])
def test_direct_potential_rejects_invalid_split_count(bad):
    rows, adj, dist = dist_adj([(0, 1, 1)], 0, 1)
    with pytest.raises(ValueError, match='splits'):
        residual_quotient_potential(rows, adj, 0, 1, dist, splits=bad)


@pytest.mark.parametrize('bad', [-1, 0, 1, True, 2.5])
def test_direct_potential_rejects_invalid_cap(bad):
    rows, adj, dist = dist_adj([(0, 1, 1)], 0, 1)
    with pytest.raises(ValueError, match='cap'):
        residual_quotient_potential(rows, adj, 0, 1, dist, cap=bad)


def test_class_cap_preserves_a_verified_distance_witness():
    edges = [(i, i + 1, 1) for i in range(65)]
    rows, adj, dist = dist_adj(edges, 0, 65)
    phi, receipt = residual_quotient_potential(rows, adj, 0, 65, dist, splits=5)
    assert phi is None and receipt['quotient_fallback']
    costs = {}
    lo, hi, witness = certified_cross_resistance(
        edges, 0, 65, 0, initial_potential='residual_quotient',
        potential_splits=5, costs=costs)
    assert lo == hi == 65
    assert costs['quotient_fallback'] and costs['quotient_groups'] == 66
    assert verify_resistance_witness(edges, 0, 65, lo, hi, witness)


def test_residual_start_preserves_the_stop_callback():
    edges = [(0, 1, 1), (1, 2, 1), (0, 2, 1), (2, 3, 1)]
    calls = []
    def stop(lo, hi):
        calls.append((lo, hi))
        return True
    lo, hi, witness = certified_cross_resistance(
        edges, 0, 3, 100, initial_potential='residual_quotient', stop=stop)
    assert witness.updates == 0
    assert calls or lo == hi
    assert verify_resistance_witness(edges, 0, 3, lo, hi, witness)
