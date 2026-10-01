"""Independent rational falsification, not a floating SVD regression."""
from fractions import Fraction as F
from itertools import combinations
import random
import sympy as s
import pytest
from graph_engine.answerability import (LinearStateGraph, Projection,
    answerable_before_compression as before, max_compression,
    query_reparameterization, harmonic_graph, repair_compression, MissingLinearContract, SearchBudgetExceeded)


def sm(rows, n):
    return s.Matrix([[s.Rational(x.numerator, x.denominator) if isinstance(x, F) else x for x in r] for r in rows]) if rows else s.zeros(0,n)


def dot(row, x):
    if not isinstance(row, dict): row=dict(enumerate(row))
    return sum(F(v)*x.get(i,F(0)) for i,v in row.items())


def certify(graph, queries, kept, A, Q, T):
    rank, lost, w=before(graph, queries, kept)
    B, C=A*T, Q*T
    expected=C.rank()+B.rank()-C.col_join(B).rank()
    assert rank==expected
    unsafe=0
    for idx, q in enumerate(queries.values()):
        expected_answerable=B.col_join(C[idx,:]).rank()==B.rank()
        assert (list(queries)[idx] not in lost)==expected_answerable
        key=list(queries)[idx]
        if key in w['recovery']:
            weights=w['recovery'][key]
            recovered=s.zeros(1,T.cols)
            for k,value in weights.items(): recovered+=s.Rational(value.numerator,value.denominator)*(sm([graph.observations[k]],graph.state_dim) if not isinstance(graph.observations[k],dict) else s.Matrix([[graph.observations[k].get(i,0) for i in range(graph.state_dim)]]))*T
            assert recovered==C[idx,:]
            # Evaluate actual answers on arbitrary admissible state, separately.
            z=s.Matrix(T.cols,1,[F((i+1)*7,11) for i in range(T.cols)]); x=T*z
            truth=(C[idx,:]*z)[0]
            actual=sum(s.Rational(v.numerator,v.denominator)*(B[kept.index(k),:]*z)[0] for k,v in weights.items())
            unsafe+=int(actual!=truth)
        else:
            col=w['collisions'][key]; x=col['state_delta']; z=col['coordinate_delta']
            assert T*s.Matrix(T.cols,1,[z.get(i,0) for i in range(T.cols)])==s.Matrix(graph.state_dim,1,[x.get(i,0) for i in range(graph.state_dim)])
            assert all(dot(graph.observations[k],x)==0 for k in kept)
            assert dot(q,x)==col['query_difference']!=0
    assert unsafe==0


@pytest.mark.parametrize('seed', range(300))
def test_constructed_truth(seed):
    rng=random.Random(20261001+seed)
    n=1+seed%6; m=seed%8; k=1+seed%5
    raw=[[F(rng.randrange(-2,3),rng.randrange(1,5)) for _ in range(n)] for _ in range(m)]
    if raw and seed%3==0: raw[-1]=raw[0][:]
    if raw and seed%7==0: raw[0]=[F(0)]*n; raw[0][0]=F(1,10**60)
    qs=[[F(rng.randrange(-3,4),rng.randrange(1,6)) for _ in range(n)] for _ in range(k)]
    if raw and seed%2==0: qs[0]=raw[0][:]
    if seed%11==0: qs[-1]=[F(0)]*n
    basis=None; T=s.eye(n)
    if seed%4==0:
        d=seed%(n+1)
        basis=[[F(rng.randrange(-2,3)) for _ in range(n)] for _ in range(d)]
        T=sm(basis,n).T
    graph=LinearStateGraph(n,dict(enumerate(raw)),basis)
    keep=[i for i in range(m) if rng.randrange(2)]
    queries=dict(enumerate(qs))
    certify(graph,queries,keep,sm([raw[i] for i in keep],n),sm(qs,n),T)
    quotient=query_reparameterization(graph,queries)
    assert quotient['rank']==(sm(qs,n)*T).rank()
    repair=repair_compression(graph,queries,keep)
    expected=(sm(qs,n)*T).rank()-before(graph,queries,keep)[0]
    assert repair['minimum_additional_observations']==expected


def test_intersection_is_not_number_of_answerable_named_queries():
    g=LinearStateGraph(2,{'a':[1,0]})
    rank,lost,w=before(g,{'sum':[1,1],'difference':[1,-1]},['a'])
    assert rank==1 and len(lost)==2


@pytest.mark.parametrize('seed',range(30))
def test_global_maximum_against_independent_enumeration(seed):
    rng=random.Random(seed)
    n=1+seed%3; m=3+seed%4
    raw=[[rng.randrange(-2,3) for _ in range(n)] for _ in range(m)]
    g=LinearStateGraph(n,dict(enumerate(raw)))
    Q={'q':[sum(row[i] for row in raw) for i in range(n)]}
    result=max_compression(g,Q)
    A=sm(raw,n); target=sm(list(Q.values()),n)
    best=m
    for count in range(m+1):
        if any((B:=sm([raw[i] for i in subset],n)).col_join(target).rank()==B.rank() for subset in combinations(range(m),count)):
            best=count; break
    assert result['minimum_kept']==best
    assert result['maximum_deleted']==m-best and result['global_optimum']


def test_greedy_is_not_global_optimum():
    # Delete the direct q record first: irreducible remaining pair is not minimal.
    g=LinearStateGraph(2,{'direct':[1,0],'b':[0,1],'sum':[1,1]})
    Q={'q':[1,0]}
    assert not before(g,Q,['b','sum'])[1]
    assert before(g,Q,['b'])[1] and before(g,Q,['sum'])[1]
    assert max_compression(g,Q)['keep']==['direct']


def test_empty_zero_duplicate_and_abstention():
    g=LinearStateGraph(0,{})
    assert before(g,{'zero':[]},[])[0]==0
    assert max_compression(g,[])['minimum_kept']==0
    with pytest.raises(MissingLinearContract): before({},[],[])
    with pytest.raises(MissingLinearContract): before(LinearStateGraph(1,{}),{'prose':None},[])
    with pytest.raises(ValueError): before(LinearStateGraph(1,{'a':[1]}),[],['a','a'])
    with pytest.raises(SearchBudgetExceeded): max_compression(LinearStateGraph(2,{'a':[1,1]}),{'q':[1,1]},max_subsets=0)


def test_explicit_projection_and_harmonic_export():
    with pytest.raises(ValueError, match='representable'):
        harmonic_graph(2, [(0,1)], [F(1,3)], [0,1])
    g=harmonic_graph(3,[(0,1),(1,2)],[1.,1e-30],[0,2])
    q={'middle':[0,1,0]}
    r,l,w=before(g,q,[('node',0),('node',2)])
    assert r==1 and not l
    h=g.basis[0][1]
    assert h==F(1)/(F(1)+F(1e-30))
    r,l,w=before(g,q,Projection([[0,1,0]]))
    assert r==1 and not l
    # A changed source or arbitrary interior forcing is not this harmonic model.
    unrestricted=LinearStateGraph(3,g.observations)
    assert before(unrestricted,q,[('node',0),('node',2)])[1]==['middle']


def test_generators_and_invertible_state_change():
    g=LinearStateGraph(2, {'a':[1,0], 'b':[0,1]})
    mapped=LinearStateGraph(2, g.observations, [[2,1],[3,2]])
    Q={'sum':[1,1],'first':[1,0]}
    for keep in [[],['a'],['b'],['a','b']]:
        r,l,_=before(g,Q,keep)
        rr,ll,_=before(mapped,Q,keep)
        assert (r,l)==(rr,ll)
    assert repair_compression(g, iter([[1,1]]), iter(['a']))['minimum_additional_observations']==1
    assert max_compression(g, iter([[1,0]]))['keep']==['a']
