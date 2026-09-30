import itertools
import math
import numpy as np
import pytest

from graph_engine.precision_form import PrecisionForm, Candidate
from graph_engine.typed_throws import (Part, Port, Claim, set_value_Q_bits, closure_value,
                                      frustrated_cycles, draw_triples, cover_triples)
from graph_engine.claim_federation import Federation
from graph_engine.next_actions import (EngineState, triple_bundle, next_actions, apply,
                                       chain_throw_bundle)


def target_world():
    f = PrecisionForm(np.diag([1., 1e-4, 1e-4, 1e-8]), np.zeros(4))
    cs = [Candidate(name, np.array(h), .01) for name,h in
          [('A',[1.,1.,0.,0.]),('B',[0.,1.,1.,0.]),('C',[0.,0.,1.,0.]),('D',[0.,0.,0.,1.])]]
    return f, cs, np.array([[1.,0.,0.,0.]])


def test_best_set_target_closes_mediator_and_matches_independent_batch_conditioning():
    f, cs, Q = target_world()
    J = f.J.copy()
    picked = f.best_set(cs, 3, Q=Q)
    assert {c.id for c in picked} == {'A','B','C'}
    assert 'D' in {c.id for c in f.best_set(cs,3)}
    for S in itertools.combinations(cs,3):
        H = np.array([c.h for c in S]); sig=np.array([c.sigma for c in S])
        C=f.cov(); after=C-C@H.T@np.linalg.solve(np.diag(sig**2)+H@C@H.T,H@C)
        oracle=.5*math.log2(float((Q@C@Q.T)[0,0]/(Q@after@Q.T)[0,0]))
        assert set_value_Q_bits(f,H,sig,Q) == pytest.approx(oracle,abs=1e-6)
    assert np.array_equal(f.J,J)
    assert max(set_value_Q_bits(f,[c.h for c in S],.01,Q) for S in itertools.combinations(cs[:3],2)) < .001
    assert set_value_Q_bits(f,[c.h for c in cs[:3]],.01,Q)>5
    for perm in itertools.permutations(cs[:3]):
        assert set_value_Q_bits(f,[c.h for c in perm],.01,Q) == pytest.approx(set_value_Q_bits(f,[c.h for c in cs[:3]],.01,Q),abs=1e-6)
    with pytest.raises(ValueError,match='max_sets'):
        f.best_set(cs,3,Q=Q,max_sets=1)


def test_target_cost_objective_and_validation():
    f,cs,Q=target_world();cs[2].cost=1e7
    pool=list(itertools.combinations(cs,3))
    oracle=max(pool,key=lambda S:set_value_Q_bits(f,[c.h for c in S],.01,Q)/sum(c.cost for c in S))
    assert f.best_set(cs,3,per_cost=True,Q=Q)==list(oracle)
    assert f.best_set(cs,0,Q=Q)==[]
    assert set_value_Q_bits(f,[],Q=Q)==0
    with pytest.raises(ValueError):set_value_Q_bits(f,[cs[0].h],0,Q)


def make_part(id,ins,outs,**kw):
    return Part(id,id,'test',id,ins,outs,['low_rank_update'],**kw)


def test_native_triple_actions_and_atomic_gaussian_update():
    f,cs,Q=target_world();state=EngineState(form=f,candidates=cs,
        triple_requests=[{'members':['A','B','C'],'target':'t','valuation':'value_Q','Q':Q}])
    a=next_actions(state,k=10,guard_share=0).by_kind('triple')[0]
    assert a.value_unit=='bits' and a.value_bits>5
    original=f.J.copy()
    with pytest.raises(ValueError):apply(state,a,{'values':[1,2,float('nan')]})
    assert np.array_equal(f.J,original)
    row=apply(state,a,{'values':[0,1,2]})
    assert row['value_realized']==pytest.approx(a.value_bits)
    with pytest.raises(ValueError,match='stale'):apply(state,a,{'values':[0,1,2]})


def test_closure_bundle_is_typed_and_surrogate_currency_preserved():
    A=make_part('A',[Port('x','1',given=True)],[Port('y','1')])
    B=make_part('B',[Port('y','1')],[Port('z','1')])
    C=make_part('C',[Port('z','1')],[Port('w','1')])
    state=EngineState(typed_parts={p.id:p for p in (A,B,C)},triple_given={'x'},
        triple_requests=[{'members':['A','B','C'],'target':'w'}])
    ranked=next_actions(state,k=3)
    assert ranked==[] and len(ranked.other)==1
    a=ranked.by_kind('triple')[0]
    assert a.value_unit=='structural_bits' and a.value_bits>5
    assert apply(state,a,{'combination':{},'evidence':['run1']})['value_realized'] is None
    C.inputs=[Port('z','N')]
    with pytest.raises(ValueError,match='incompatible'):triple_bundle(state,['A','B','C'],'w')
    assert closure_value(state.typed_parts,['A','B','C'],'w',{'x'})==0


def fed_cycle(boxes=None,instances=None,regimes=None):
    f=Federation()
    for k,((s,o),sign) in enumerate(zip([('s','x'),('x','o'),('s','o')],[1,1,-1])):
        f.add_graph({'graph_id':str(k),'sources':[{'id':str(k)}],'claims':[{
            'id':'c','subject':s,'object':o,'sign':sign,'evidence':[str(k)],
            'validity':(boxes or [{},{},{}])[k], 'instance':(instances or ['','',''])[k],
            'regime':(regimes or [{},{},{}])[k]}]})
    return f


def test_asserted_opposition_remains_inferred_and_cycle_is_in_next_experiments():
    f=fed_cycle();links,_=f.inferred_links()
    l=next(l for l in links if (l.subject,l.object)==('s','o'))
    assert l.sign==1 and l.contradicts==('2:c',) and l.status=='INFERRED'
    probes=[p for p in f.stress_points() if p.kind=='CYCLE']
    assert len(probes)==1 and len(probes[0].claims)==3
    assert any(p.kind=='CYCLE' for p in f.next_experiments())
    with pytest.raises(ValueError,match='edge-specific'):f.record(probes[0],1,'new')
    scoped=fed_cycle(instances=['one']*3,regimes=[{'branch':['stick']}]*3)
    state=EngineState(federation=scoped)
    action=next(a for a in next_actions(state).by_kind('probe') if a.meta['kind']=='CYCLE')
    receipt=apply(state,action,{'sign':1,'source':'cycle-measurement'})
    measured=scoped.claims[receipt['claim_id']]
    assert measured['pair']==action.target and measured['instance']=='one'
    assert measured['regime']=={'branch':['stick']}
    assert set(receipt['cycle_witness'])==set(action.meta['probe'].claims)
    assert any(p.kind=='CYCLE' for p in scoped.stress_points())


@pytest.mark.parametrize('kwargs',[
    {'boxes':[{'t':[0,1]},{'t':[0,1]},{'t':[2,3]}]},
    {'instances':['one','one','other']},
    {'regimes':[{'branch':['stick']},{'branch':['stick']},{'branch':['slip']}]},
])
def test_incompatible_claims_do_not_make_cycles_or_asserted_conflicts(kwargs):
    f=fed_cycle(**kwargs)
    assert not any(p.kind=='CYCLE' for p in f.stress_points())
    assert not any(l.contradicts for l in f.inferred_links()[0])


def test_region_specific_tree_finds_cycle_masked_by_incompatible_shortcut():
    cs=[Claim('a','A','sign',subject='s',object='x',sign=1,box={'t':(0,1)}),
        Claim('bad','B','sign',subject='s',object='o',sign=1,box={'t':(2,3)}),
        Claim('b','B','sign',subject='x',object='o',sign=1,box={'t':(0,1)}),
        Claim('c','C','sign',subject='s',object='o',sign=-1,box={'t':(0,1)})]
    assert any(set(x.claims)=={'a','b','c'} and x.cls=='contradiction' for x in frustrated_cycles(cs))


def test_cycle_presence_matches_independent_exhaustive_balance_oracle():
    edges=list(itertools.combinations(range(4),2))
    for signs in itertools.product((0,1,-1),repeat=len(edges)):
        claims=[Claim(str(k),'test','sign',subject=str(a),object=str(b),sign=s)
                for k,((a,b),s) in enumerate(zip(edges,signs)) if s]
        balanced=any(all(not s or colors[a]*colors[b]==s for (a,b),s in zip(edges,signs))
                     for colors in itertools.product((1,-1),repeat=4))
        assert bool(frustrated_cycles(claims)) == (not balanced)


def test_cover_design_and_exact_draw_contract():
    d=cover_triples(7,4,seed=18,n_cand=5)
    assert d.covered==d.total==math.comb(7,3) and d.triple_pi==1
    assert len(set(t for b in d.blocks for t in itertools.combinations(b,3)))==d.total
    short=cover_triples(7,4,seed=18,budget=1,n_cand=5)
    assert short.triple_pi==math.comb(4,3)/math.comb(7,3)
    assert cover_triples(7,4,seed=18,budget=0,n_cand=5).triple_pi==0
    draws=draw_triples(np.arange(10.),3)
    assert len(draws)==3 and len({i for i,p in draws})==3 and all(0<p<=1 for i,p in draws)
    assert draw_triples([],3)==[]


def test_typed_geometric_chain_gate():
    f,_,_=target_world();state=EngineState(form=f)
    A=make_part('A',[Port('a',given=True)],[Port('b')],box={'t':(0,.3)})
    B=make_part('B',[Port('b')],[Port('c')],box={'t':(.2,.6)})
    C=make_part('C',[Port('c')],[Port('d')],box={'t':(.5,1)})
    with pytest.raises(ValueError,match='regime_conf=True'):
        chain_throw_bundle(state,[0,1,2],parts=[A,B,C])
