from dataclasses import replace
from fractions import Fraction as F
import math
import random
import pytest

from graph_engine.certified_resistance import ResistanceBudget
from graph_engine.certified_decision import (
    certified_decision, verify_decision_witness, edge_hash,
    certified_decisions, verify_decision_batch)
from test_certified_resistance import independent_oracle


@pytest.mark.parametrize("mode", ["cycle", "tree", "cg"])

@pytest.mark.parametrize("seed", range(20))
def test_exact_containment_and_decisions(mode, seed):
    rng = random.Random(202610017000+seed)
    n = rng.randint(4, 11)
    edges = [(u, u+1, F(rng.randint(1, 11), rng.randint(1, 5))) for u in range(n-1)]
    edges += [(u, v, F(rng.randint(1, 11), rng.randint(1, 5)))
              for u in range(n) for v in range(u+1, n) if rng.random() < .2]
    edges += [(1, 1, F(7)), edges[0]]
    exact = independent_oracle(edges, 0, n-1)
    for ratio in [F(1, 2), F(99, 100), F(101, 100), F(2)]:
        theta = ratio*exact
        result = certified_decision(iter(edges), 0, n-1, theta,
                  ResistanceBudget(max_updates=40, refinement=mode))
        w = result['witness']
        assert w.lo <= exact <= w.hi
        assert verify_decision_witness(edges, 0, n-1, theta, result)
        if result['verdict'] != 'UNDECIDED':
            assert result['verdict'] == ('GREATER' if exact > theta else 'LESS')


def test_early_stop_and_equality():
    edges = [(0, 1, 1)]
    for theta, verdict in [(F(1, 2), 'GREATER'), (F(2), 'LESS'), (F(1), 'UNDECIDED')]:
        result = certified_decision(edges, 0, 1, theta, 1000)
        assert result['verdict'] == verdict
        assert result['updates_used'] == 0
        assert verify_decision_witness(edges, 0, 1, theta, result)


def test_single_weight_flip_and_irrelevant_loop_rejected():
    edges = [(0, 1, 1), (0, 0, 3)]
    result = certified_decision(edges, 0, 1, F(1, 2), 0)
    assert result['verdict'] == 'GREATER'
    changed = [(0, 1, 4), (0, 0, 3)]
    assert certified_decision(changed, 0, 1, F(1, 2), 0)['verdict'] == 'LESS'
    with pytest.raises(ValueError, match='hash'):
        verify_decision_witness(changed, 0, 1, F(1, 2), result)
    with pytest.raises(ValueError, match='hash'):
        verify_decision_witness([(0, 1, 1), (0, 0, 4)], 0, 1, F(1, 2), result)


def test_graph_and_query_and_result_binding():
    edges = [(0, 1, 2), (1, 2, 3)]
    result = certified_decision(edges, 0, 2, F(1), 0)
    for changed in [edges[::-1], [(1, 0, 2), edges[1]], edges+[(3, 3, 1)]]:
        with pytest.raises(ValueError, match='hash'):
            verify_decision_witness(changed, 0, 2, F(1), result)
    for a, b, t in [(2, 0, F(1)), (0, 2, F(2))]:
        with pytest.raises(ValueError, match='binding'):
            verify_decision_witness(edges, a, b, t, result)
    with pytest.raises(ValueError, match='verdict'):
        verify_decision_witness(edges, 0, 2, F(1), {**result, 'verdict':'GREATER'})
    assert edge_hash([(0, 1, 2)]) == edge_hash([(0, 1, F(4, 2))])


@pytest.mark.parametrize('edges,a,b', [([],0,0),([],0,1), ([(0,1,1)],0,2)])
def test_special_kinds_bound_to_graph(edges,a,b):
    result = certified_decision(edges,a,b,F(1),ResistanceBudget(seconds=0))
    verify_decision_witness(edges,a,b,F(1),result)
    with pytest.raises(ValueError, match='hash'):
        verify_decision_witness(edges+[(8,8,1)],a,b,F(1),result)


def test_no_fullsolve(monkeypatch):
    import graph_engine.certified_resistance as cr
    import graph_engine._rational_interface as ri
    def forbidden(*args, **kwargs):
        raise AssertionError('fullsolve called')
    monkeypatch.setattr(cr, 'exact_edge_resistance', forbidden)
    monkeypatch.setattr(ri, 'solve_right', forbidden)
    edges=[(0,1,1),(1,2,1),(0,2,1)]
    for mode in ['cycle','tree','cg']:
        result=certified_decision(edges,0,2,F(1),ResistanceBudget(max_updates=30,refinement=mode))
        assert result['verdict']=='LESS'
        verify_decision_witness(edges,0,2,F(1),result)


@pytest.mark.parametrize('mode',['cycle','tree','cg'])
def test_shared_threshold_batch(mode):
    edges=[(0,1,2),(1,2,3),(0,2,1)]
    exact=independent_oracle(edges,0,2)
    thetas=[exact/2,exact,exact*2]
    result=certified_decisions(iter(edges),0,2,thetas,ResistanceBudget(max_updates=30,refinement=mode))
    verify_decision_batch(edges,0,2,thetas,result)
    assert result['verdicts']==('GREATER','UNDECIDED','LESS')
    assert result['witness'].lo<=exact<=result['witness'].hi
    with pytest.raises(ValueError,match='hash'):
        verify_decision_batch(edges+[(9,9,1)],0,2,thetas,result)
    with pytest.raises(ValueError,match='binding'):
        verify_decision_batch(edges,0,2,thetas[::-1],result)


def test_batch_rejects_forged_verdict_and_empty_query():
    edges=[(0,1,1)]
    result=certified_decisions(edges,0,1,[F(1,2),F(2)],0)
    assert result['updates_used']==0
    with pytest.raises(ValueError,match='verdicts'):
        verify_decision_batch(edges,0,1,[F(1,2),F(2)],{**result,'verdicts':('LESS','LESS')})
    with pytest.raises(ValueError,match='at least one'):
        certified_decisions(edges,0,1,[],0)
