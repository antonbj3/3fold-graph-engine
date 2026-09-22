"""Live Graph claims/evidence survive export into the actual native learner."""
import importlib
import os
import sys

import numpy as np
import pytest

from graph_engine.compiled_regime import CompiledRegime
from graph_engine.regime_posterior import RegimePosterior
from graph_engine.training_state import CompiledTrainingBatch


@pytest.fixture(scope='module')
def kernel():
    source=os.environ.get('KERNEL_ENGINE_SRC')
    build=os.environ.get('BRIDGE_BUILD_DIR')
    if not source or not build:
        pytest.skip('KERNEL_ENGINE_SRC/BRIDGE_BUILD_DIR unset')
    sys.path.insert(0,source)
    cls=importlib.import_module('kernel_engine.inference.finite_regime').FiniteRegimeKernel
    return cls(build)


def examples(kernel):
    pairs=[];references=[]
    for i in range(3):
        r=RegimePosterior(0,1,n_grid=8,p_flip=.3,p_two=.2)
        r.add_claim(0,.5,1,n_eff=1+i,reliability=.8)
        r.add_claim(.5,1,-1,n_eff=2,reliability=.7)
        r.add_probe(.2,(-1,1,-1)[i],reliability=.9,weight=1.)
        c=CompiledRegime.from_regime(r,kernel,probe_ids=[f'physical:{i}:0'])
        pairs.append((f'example:{i}',c));references.append(r)
    return pairs,references


def test_export_preserves_claims_and_uses_prior_not_posterior(kernel):
    pairs,refs=examples(kernel)
    batch=CompiledTrainingBatch.from_regimes(pairs)
    predictions=kernel.evaluate_batch(batch.natural[:,0],batch.natural[:,1],batch.family_mass)['p_plus']
    for i,(_,compiled) in enumerate(pairs):
        reference=np.array([refs[i].p_plus(float(x)) for x in batch.cells.mean(axis=1)])
        np.testing.assert_allclose(predictions[i],reference,atol=1e-12)
        np.testing.assert_allclose(predictions[i],compiled.p_plus_cells(),atol=1e-12)
    assert not np.allclose(batch.family_mass,pairs[0][1].family_mass())
    assert batch.evidence_ids==(('physical:0:0',),('physical:1:0',),('physical:2:0',))


def test_snapshot_stays_frozen_after_live_evidence(kernel):
    pairs,_=examples(kernel)
    batch=CompiledTrainingBatch.from_regimes(pairs)
    before=batch.natural.copy()
    for array in (batch.natural,batch.cells):
        with pytest.raises(ValueError):array.setflags(write=True)
    pairs[0][1].add_probe(.7,1,.8,evidence_id='new-physical-reading')
    changed=CompiledTrainingBatch.from_regimes(pairs)
    assert changed.fingerprint!=batch.fingerprint
    assert np.array_equal(batch.natural,before)
    assert not np.array_equal(changed.natural,before)


def test_incompatible_geometry_prior_ids_rejected(kernel):
    pairs,_=examples(kernel)
    for bad in [CompiledRegime(0,2,n_grid=8,p_flip=.3,p_two=.2,backend=kernel),
                CompiledRegime(0,1,n_grid=8,p_flip=.4,p_two=.2,backend=kernel)]:
        with pytest.raises(ValueError,match='identical partitions and family priors'):
            CompiledTrainingBatch.from_regimes([pairs[0],('other',bad)])
    for rows in ([],[pairs[0],pairs[0]],[('',pairs[0][1])]):
        with pytest.raises(ValueError):CompiledTrainingBatch.from_regimes(rows)
    batch=CompiledTrainingBatch.from_regimes(pairs)
    assert batch.indices_for(['example:2','example:0','example:2']).tolist()==[2,0,2]
    with pytest.raises(ValueError):batch.indices_for(['missing'])
    with pytest.raises(ValueError):batch.indices_for('example:0')


def test_graph_snapshots_feed_real_forecast_training(kernel):
    torch=pytest.importorskip('torch')
    cls=importlib.import_module('kernel_engine.inference.forecast_training').ProbeDistillation
    pairs,_=examples(kernel);batch=CompiledTrainingBatch.from_regimes(pairs)
    objective=cls(kernel,batch.natural,[1,5],.85,family_mass=batch.family_mass)
    rows=batch.indices_for(['example:2','example:0'])
    student=torch.tensor(batch.natural[rows].copy(),dtype=torch.float64,requires_grad=True)
    exact=objective(student,rows)
    assert exact.combined.abs().max().item()<1e-14
    (exact.combined.sum()).backward()
    assert torch.isfinite(student.grad).all()
    perturbed=student.detach().clone();perturbed[:,1,1]-=.5
    assert objective(perturbed,rows).future.min().item()>0


def test_actual_kernel_audits_named_repeated_learner_rows(kernel):
    pairs,_=examples(kernel);batch=CompiledTrainingBatch.from_regimes(pairs)
    ids=['example:2','example:0','example:2'];indices=batch.indices_for(ids)
    prediction=batch.natural[indices].copy()
    exact=batch.audit_predictions(prediction,ids,backend=kernel,kl_budget_nats=0)
    assert [r['status'] for r in exact['rows']]==['within_budget']*3
    assert all(r['kl_upper_nats']==0 for r in exact['rows'])
    prediction[0,1,1]+=.5
    changed=batch.audit_predictions(prediction,ids,backend=kernel,kl_budget_nats=0)
    assert changed['rows'][0]['status']=='unresolved'
    assert changed['rows'][0]['kl_upper_nats']>0
    assert changed['rows'][1]['status']==changed['rows'][2]['status']=='within_budget'
    assert changed['prediction_fingerprint']!=exact['prediction_fingerprint']
    assert changed['teacher_fingerprint']==batch.fingerprint


def test_audit_owns_input_and_unavailable_bound_means_unresolved(kernel):
    pairs,_=examples(kernel);batch=CompiledTrainingBatch.from_regimes(pairs)
    prediction=batch.natural.copy()
    class InterleavedMutation:
        def drift_bound(self,*args):
            prediction[:]+=10  # Simulates a producer changing its shared buffer.
            return kernel.drift_bound(*args)
    result=batch.audit_predictions(prediction,batch.example_ids,backend=InterleavedMutation())
    assert all(r['kl_upper_nats']==0 for r in result['rows'])
    class Unavailable:
        def drift_bound(self,*args):raise ArithmeticError('energy overflow')
    result=batch.audit_predictions(prediction,batch.example_ids,backend=Unavailable(),kl_budget_nats=.1)
    assert all(r['status']=='unresolved' and 'overflow' in r['reason'] for r in result['rows'])


def test_audit_rejects_missing_identity_wrong_shapes_and_bad_budget(kernel):
    pairs,_=examples(kernel);batch=CompiledTrainingBatch.from_regimes(pairs)
    for data,ids,budget in [(batch.natural,['missing']*3,0),
            (batch.natural,batch.example_ids,-1),(batch.natural,batch.example_ids,np.inf),
            (batch.natural[0],['example:0'],0),(batch.natural,[],0),
            (np.full_like(batch.natural,np.nan),batch.example_ids,0)]:
        with pytest.raises(ValueError):batch.audit_predictions(data,ids,backend=kernel,kl_budget_nats=budget)
