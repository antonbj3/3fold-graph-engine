import copy, importlib.util, json
from pathlib import Path
import numpy as np
import pytest
from graph_engine.agreement_admission_gate import equal_selectivity_control, agreement_admission_gate
from graph_engine.claim_federation import source_family_components
from graph_engine.coupling_admission_precheck import source_family_admission
from graph_engine.tools.fold_gate import _match
from graph_engine.tools.fold_gate_v2 import fold_gate_v2, fold_gate_v2_round, planted_fold_catalogue

path=Path(__file__).parents[1]/'examples/engine_experiments/admission_triple_fixture.py'
spec=importlib.util.spec_from_file_location('triple_fixture',path);fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)

def test_equal_k_positive_and_selection_only():
 c=fixture.calibration();r=equal_selectivity_control(**c)
 assert r['validated'] and r['delta_ci'][0]>0 and all(x['k']==r['k'] for x in r['controls'])
 c['scores']=[[2.0 if a else i/128 for i,a in enumerate(c['selected'])]]*2
 r=equal_selectivity_control(**c)
 assert not r['validated'] and r['delta']==0 and r['reason_codes']==['NO_EXCESS_AGREEMENT_GAIN']

@pytest.mark.parametrize('field,value', [('scores',[[float('nan')]*128]),('source_families',['same']*128),('candidate_ids',['same']*128),('selected',[False]*128),('n_bootstrap',float('inf'))])
def test_bad_calibration_abstains(field,value):
 c=fixture.calibration();c[field]=value;assert not equal_selectivity_control(**c)['validated']

def test_ties_do_not_fabricate_signal():
 c=fixture.calibration();c['scores']=[[1.0]*128]*2
 assert not equal_selectivity_control(**c)['validated']

@pytest.mark.parametrize('tol',[float('inf'),float('nan'),-1,True,'10'])
def test_bad_tolerance_is_not_evidence(tol):assert not _match(100.,1.,tol)

def test_numeric_and_boolean_are_distinct():
 assert not _match(True,1,None);assert _match(1.,1.00000001,1e-5)

def test_overlapping_mixed_reports_are_one_unit():
 r=source_family_components([{'verified':True,'source_family':x} for x in [['A'],['B'],['A','B']]])
 assert r['count']==1
 assert not source_family_admission([{'verified':True,'source_family':['A']}]*2)['admit']

def test_coupling_cannot_supply_a_family():
 src=[{'verified':True,'source_family':['A']}]
 for n in [0,1,1000]:assert not source_family_admission(src,n)['admit']
 assert source_family_admission(src,1000)['reason_codes']==['COUPLING_CASCADE']
 assert source_family_admission(src+[{'verified':True,'source_family':['B']}],1000)['admit']

def test_round_runs_real_mutations_and_admits_valid_control(tmp_path):
 good=fixture.template(tmp_path);r=fold_gate_v2_round([good],good,str(tmp_path),verification_registry=fixture.registry())
 assert r['fence_passed'] and r['accepted_indices']==[0]
 assert len(r['probes'])==15 and all(p['passed'] for p in r['probes'])
 assert not list(tmp_path.glob('probe-*.json'))

@pytest.mark.parametrize('guards',[('b',),('a','b'),('b','c')])
def test_exhaustive_fence_stops_incomplete_rounds(tmp_path,guards):
 good=fixture.template(tmp_path);r=fold_gate_v2_round([good],good,str(tmp_path),guards)
 assert not r['fence_passed'] and not r['accepted_indices']

@pytest.mark.parametrize('mode',['accept_all','wrong_reason','exception'])
def test_fence_checks_reasons_not_just_no(tmp_path,mode):
 good=fixture.template(tmp_path)
 def evaluator(s):
  if mode=='exception':raise RuntimeError('broken evaluator')
  return {'decision':'ALLOW' if mode=='accept_all' else 'BLOCK','as':'result','reason_codes':['WRONG_REASON']}
 r=fold_gate_v2_round([good],good,str(tmp_path),evaluator=evaluator)
 assert not r['fence_passed'] and not r['accepted_indices']

def test_fault_labels_are_not_given_to_evaluator(tmp_path):
 good=fixture.template(tmp_path);seen=[]
 def evaluator(s):
  assert 'fault_class' not in s and 'required_reason' not in s
  seen.append(s['cell']);return fold_gate_v2(s,base_dir=str(tmp_path),admission_guards=('a','b','c'),verification_registry=fixture.registry())
 r=fold_gate_v2_round([good],good,str(tmp_path),evaluator=evaluator)
 assert r['fence_passed'] and len(set(seen))==16

def test_existing_fisher_precheck_cannot_add_unsupported_channel():
 from graph_engine.coupling_admission_precheck import precheck
 r=precheck([[0.]],{'leg':[[1.]]},require_source_families=True)
 assert not r['admitted']
 r=precheck([[0.]],{'leg':[[1.]]},source_reports={'leg':[{'verified':True,'source_family':['A']},{'verified':True,'source_family':['B']}]})
 assert r['admitted']==['leg']

def test_strict_agreement_requires_calibration():
 rng=np.random.default_rng(0);y=rng.integers(0,5,8000)
 def good():
  a=y.copy();m=rng.random(len(y))<.2;a[m]=(a[m]+rng.integers(1,5,int(m.sum())))%5;return a
 r=agreement_admission_gate([good(),good()],y,min_competence=.35,require_equal_selectivity=True)
 assert not r['admit'] and not r['agreement_gain_credited']
 assert r['equal_selectivity_control']['reason_codes']==['CALIBRATION_MISSING']

def test_empty_round_is_not_scientific_acceptance(tmp_path):
 good=fixture.template(tmp_path);r=fold_gate_v2_round([],good,str(tmp_path),verification_registry=fixture.registry())
 assert r['fence_passed'] and r['accepted_indices']==[]

def test_hypothesis_is_never_accepted_as_result(tmp_path):
 good=fixture.template(tmp_path);hyp=copy.deepcopy(good);hyp['hypothesis']=True
 r=fold_gate_v2_round([hyp],good,str(tmp_path),verification_registry=fixture.registry())
 assert r['fence_passed'] and not r['accepted_indices']

def test_external_registry_prevents_producer_self_verification(tmp_path):
 good=fixture.template(tmp_path);p=Path(good['rerun']['artifact']);a=json.loads(p.read_text())
 a['source_reports']=[{'source_id':'invented-A','verified':True,'source_family':['A']}, {'source_id':'invented-B','verified':True,'source_family':['B']}];p.write_text(json.dumps(a))
 r=fold_gate_v2(good,base_dir=str(tmp_path),admission_guards=('a','b','c'),verification_registry=fixture.registry())
 assert r['decision']=='BLOCK' and 'PROVENANCE_MISSING' in r['reason_codes']
 r=fold_gate_v2(good,base_dir=str(tmp_path),admission_guards=('a','b','c'))
 assert r['decision']=='BLOCK'

def test_declared_families_cannot_override_registry():
 src=[{'source_id':'source-A','source_family':['invented-A'],'verified':True},{'source_id':'source-A','source_family':['invented-B'],'verified':True}]
 r=source_family_admission(src,verification_registry=fixture.registry(),require_registry=True)
 assert not r['admit'] and r['direct_families']==1

@pytest.mark.parametrize('reports',[None,{},[{'verified':1,'source_family':['A']}],[{'verified':True,'source_family':[]}],[{'verified':True,'source_family':[None]}]])
def test_invalid_family_records_fail_closed(reports):
 assert not source_family_components(reports)['valid']

def test_unresolved_lineage_cycle_cannot_be_independent_sources():
 reg=fixture.registry();reg['source-A']['derives_from']=['source-B'];reg['source-B']['derives_from']=['source-A']
 r=source_family_admission([{'source_id':'source-A'},{'source_id':'source-B'}],verification_registry=reg,require_registry=True)
 assert not r['admit'] and r['reason_codes']==['PROVENANCE_MISSING']
