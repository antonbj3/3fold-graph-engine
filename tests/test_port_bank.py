"""Adverse contracts for the bank's native consumer, without source-worker labels."""
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

from graph_engine.port_bank import PortBank


def data():
    coords=['environment','interval','primitive','target']
    quantities=[dict(id=q,unit='1',kind='operator',instance='test') for q in coords]
    def port(q):return dict(q=q,unit='1',kind='operator',instance='test',required=True)
    def node(name,inputs,output):
        return dict(id=name,inputs=[port(q) for q in inputs],outputs=[port(output)],and_inputs=inputs,
                    direction='forward',operator='test relation',operator_tokens=['target_restricted_value'],
                    equation='declared structural relation',assumptions=[],validity={'x':[0,1]},instance='test',
                    provenance=[{'path':'declared_test_spec'}],cost=1.,leaf_status='DERIVED_UNDER_ASSUMPTIONS',reach='operator')
    nodes=[node('meter',['environment'],'interval'),node('arithmetic',['environment'],'primitive'),
           node('consumer',['interval','primitive'],'target')]
    H=[[ -1,1,0,0],[-1,0,1,0],[0,-1,-1,1]]
    return dict(schema='source_port_bank_v1',id='test',version=1,quantities=quantities,nodes=nodes,
                given=['environment'],target='target',sign_domains={'test':{'instance':'signed_test','semantics':'declared_parity',
                       'variables':['a','b','c'],'validity':{'x':[0,1]}}},
                gaussian={'semantics':'uncalibrated_dimensionless_artifact_proxy','coordinates':coords,
                          'prior_precision':np.diag([400.0001,.0001,.0001,.0001]).tolist(),
                          'observations':{n['id']:{'row':h,'sigma':.05} for n,h in zip(nodes,H)},'Q':[[0,0,0,1]]})


def bank(d=None):return PortBank(data() if d is None else d,'testdigest')


def test_information_without_required_port_is_not_a_build():
    b=bank()
    assert b.value(['consumer','meter'])>0
    assert b.value(['consumer','meter'],acquisition=True)==0
    assert b.best_executable_set(['consumer','meter'],2) is None
    with pytest.raises(ValueError,match='closure'):b.action(['consumer','meter'])
    assert b.best_executable_set(list(b.nodes)) is not None
    assert b.action(list(b.nodes)).meta['bank']==b.reference()


def test_order_independent_and_all_required_inputs():
    b=bank()
    av,order,missing=b.execution(['consumer','arithmetic','meter'])
    assert 'target' in av and order[-1]=='consumer' and not missing
    av,_,missing=b.execution(['consumer','meter'])
    assert 'target' not in av and missing['consumer']==['primitive']


def test_inverse_relation_and_cycle_do_not_create_given_inputs():
    d=data();d['given']=[]
    b=bank(d)
    assert b.execution(list(b.nodes))[1]==()
    assert b.value(list(b.nodes),acquisition=True)==0


def test_whole_set_region_is_required():
    d=data();d['nodes'][0]['validity']={'x':[0,.2]};d['nodes'][1]['validity']={'x':[.8,1]}
    b=bank(d)
    assert 'target' not in b.execution(list(b.nodes))[0]
    with pytest.raises(ValueError):b.action(list(b.nodes))


@pytest.mark.parametrize('field,value',[('unit','N'),('kind','scalar'),('instance','other')])
def test_declared_port_type_cannot_be_changed(field,value):
    d=data();d['nodes'][2]['inputs'][0][field]=value
    with pytest.raises(ValueError,match='canonical'):bank(d)


def test_changed_row_or_target_is_rejected():
    d=data();d['gaussian']['observations']['consumer']['row'][1]=1
    with pytest.raises(ValueError,match='observation'):bank(d)
    d=data();d['gaussian']['Q']=[[0,1,0,0]]
    with pytest.raises(ValueError,match='Q'):bank(d)


def test_hash_and_version_bind_worker_manifest(tmp_path):
    path=tmp_path/'bank.json';path.write_text(json.dumps(data()))
    h=sha256(path.read_bytes()).hexdigest();b=PortBank.load(path,h)
    export={'bank':b.reference(),'graph_id':'worker','members':list(b.nodes),'signed_edges':[]}
    assert b.federation(export).claims=={}
    export['bank']['version']=2
    with pytest.raises(ValueError,match='version'):b.worker_graph(export)
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='SHA256'):PortBank.load(path,h)


@pytest.mark.parametrize('sign',[0,True,False,1.0,'+'])
def test_no_implicit_or_zero_signed_export(sign):
    b=bank();e={'bank':b.reference(),'graph_id':'worker','members':list(b.nodes),
              'signed_edges':[{'id':'edge','sign':sign}]}
    with pytest.raises(ValueError,match='explicit sign'):b.worker_graph(e)


def test_same_object_signed_cycle_reaches_federation():
    b=bank();edges=[]
    for i,(a,c,sign) in enumerate([('a','b',1),('b','c',1),('a','c',-1)]):
        edges.append({'id':str(i),'subject':a,'object':c,'sign':sign,'domain':'test','semantics':'declared_parity',
                      'instance':'signed_test','validity':{'x':[0,1]},'evidence':['actual_test_spec']})
    e={'bank':b.reference(),'graph_id':'worker','members':list(b.nodes),'signed_edges':edges}
    assert any(p.kind=='CYCLE' for p in b.federation(e).stress_points())
    e['signed_edges'][0]['instance']='other'
    with pytest.raises(ValueError,match='instance'):b.worker_graph(e)


def test_exhaustive_budget_limit_is_explicit():
    b=bank()
    with pytest.raises(ValueError,match='max_sets'):b.best_executable_set(list(b.nodes),2,max_sets=1)
