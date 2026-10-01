"""Independent exact safety of alternative starts and pruned witness extension."""
import importlib.util
from pathlib import Path
from fractions import Fraction as F
import math
import pytest
from graph_engine.certified_resistance import ResistanceBudget,ResistanceWitness,certified_cross_resistance,verify_resistance_witness
from graph_engine.instance_slack import prune_leaves
from test_certified_resistance import random_cases,independent_oracle

@pytest.mark.parametrize('form',['edge_disjoint','retained_sp'])
def test_fresh_exact_graphs(form):
    for _,edges,a,b in random_cases(80):
        exact=independent_oracle(edges,a,b)
        for budget in [0,32,128]:
            lo,hi,w=certified_cross_resistance(edges,a,b,ResistanceBudget(max_updates=budget),initial_flow=form)
            verify_resistance_witness(edges,a,b,lo,hi,w)
            assert lo<=exact<=hi

def test_default_matches_preserved_t1():
    path=Path(__file__).resolve().parent/'fixtures/t9_t1_reference.py'
    spec=importlib.util.spec_from_file_location('graph_engine._t1_reference',path)
    mod=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    for _,edges,a,b in random_cases(30):
        for budget in [0,32,128]:
            lo,hi,w=certified_cross_resistance(edges,a,b,budget)
            old_l,old_h,old_w=mod.certified_cross_resistance(edges,a,b,budget)
            assert (lo,hi,w.flow,w.potential,w.updates)==(old_l,old_h,old_w.flow,old_w.potential,old_w.updates)

def test_star_hub_is_essential_dual_is_defect():
    edges=[(0,u,F(1)) for u in range(1,40)]
    lo,hi,w=certified_cross_resistance(edges,1,2,0,initial_flow='edge_disjoint')
    assert hi==independent_oracle(edges,1,2)==2
    assert lo==F(4,39)
    reduced,kept,forest=prune_leaves(edges,1,2)
    lo,hi,w=certified_cross_resistance(reduced,1,2,0)
    phi=dict(w.potential)
    for u,p in reversed(forest):phi[u]=phi[p] if p is not None else F(0)
    flow=[F(0)]*len(edges)
    for i,f in zip(kept,w.flow):flow[i]=f
    full=ResistanceWitness('connected',tuple(flow),tuple(sorted(phi.items())),input_validated=True)
    verify_resistance_witness(edges,1,2,lo,hi,full)
    assert lo==hi==2

def test_retained_sp_never_worse_at_initialization():
    for _,edges,a,b in random_cases(40):
        lo,hi,_=certified_cross_resistance(edges,a,b,0)
        ll,hh,_=certified_cross_resistance(edges,a,b,0,initial_flow='retained_sp')
        assert ll==lo and hh<=hi

def test_no_reference_solve(monkeypatch):
    import graph_engine._rational_interface as exact
    monkeypatch.setattr(exact,'solve_right',lambda *a,**kw:pytest.fail('candidate called exact solve'))
    edges=[(0,1,F(1)),(1,2,F(2)),(0,3,F(1)),(3,2,F(4))]
    for form in ['shortest','edge_disjoint','retained_sp','routed']:
        lo,hi,w=certified_cross_resistance(edges,0,2,128,initial_flow=form)
        verify_resistance_witness(edges,0,2,lo,hi,w)

@pytest.mark.parametrize("form",["shortest","edge_disjoint"])
@pytest.mark.parametrize("potential",["hub_harmonic","tree_voltage","distance_quotient","port_conditioned_quotient"])
def test_hub_potential_remains_sound(form,potential):
    for _,edges,a,b in random_cases(40):
        exact=independent_oracle(edges,a,b)
        for budget in [0,32,128]:
            lo,hi,w=certified_cross_resistance(edges,a,b,budget,initial_flow=form,initial_potential=potential)
            verify_resistance_witness(edges,a,b,lo,hi,w)
            assert lo<=exact<=hi


def test_frozen_native_router_matches_selected_form():
    from graph_engine.instance_slack import select_initial_flow
    from graph_engine.certified_resistance import _edges
    for _,edges,a,b in random_cases(30):
        rows,adj=_edges(edges,a,b)
        seen={b};stack=[b]
        while stack:
            for v,_,_ in adj[stack.pop()]:
                if v not in seen:seen.add(v);stack.append(v)
        if a not in seen:continue
        chosen,_=select_initial_flow(adj,seen)
        for budget in [0,32,128]:
            lo,hi,w=certified_cross_resistance(edges,a,b,budget,initial_flow="routed")
            ll,hh,ww=certified_cross_resistance(edges,a,b,budget,initial_flow=chosen)
            assert (lo,hi,w.flow,w.potential,w.updates)==(ll,hh,ww.flow,ww.potential,ww.updates)

def test_quotient_lower_bound_dominates_at_zero():
    for _,edges,a,b in random_cases(60):
        lo,hi,w=certified_cross_resistance(edges,a,b,0)
        ll,hh,ww=certified_cross_resistance(edges,a,b,0,initial_potential='distance_quotient')
        assert ll>=lo and hh==hi
        verify_resistance_witness(edges,a,b,ll,hh,ww)


def test_port_conditioning_expands_quotient_and_solves_star():
    for _,edges,a,b in random_cases(40):
        lo,hi,w=certified_cross_resistance(edges,a,b,0,initial_potential="distance_quotient")
        ll,hh,ww=certified_cross_resistance(edges,a,b,0,initial_potential="port_conditioned_quotient")
        assert ll>=lo and hh==hi
        verify_resistance_witness(edges,a,b,ll,hh,ww)
    edges=[(0,u,F(1)) for u in range(1,100)]
    lo,hi,w=certified_cross_resistance(edges,1,2,0,initial_potential="port_conditioned_quotient")
    assert lo==hi==2
