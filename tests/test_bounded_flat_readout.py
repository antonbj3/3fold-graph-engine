"""Optional flat readout against the real native finite-regime kernel."""
from __future__ import annotations

import importlib
import math
import os
import sys

import numpy as np
import pytest

from graph_engine.compiled_regime import CompiledRegime, FrozenPartitionError
from graph_engine.regime_posterior import RegimePosterior

_SRC=os.environ.get("KERNEL_ENGINE_SRC")
_BUILD=os.environ.get("BRIDGE_BUILD_DIR")
pytestmark=pytest.mark.skipif(not (_SRC and _BUILD),reason="native Kernel source/build paths required")


class CountingBackend:
    def __init__(self, real):self.real=real;self.calls=0
    def evaluate(self,*args,**kwargs):
        self.calls+=1
        return self.real.evaluate(*args,**kwargs)


@pytest.fixture(scope="module")
def backend():
    if _SRC not in sys.path:sys.path.insert(0,_SRC)
    native=importlib.import_module("kernel_engine.inference.finite_regime")
    return CountingBackend(native.FiniteRegimeKernel(_BUILD))


def tiny_tail(backend,q=16,tail=2**-20):
    # Existing effective-prior formula then gives masses (1-tail,tail/2,tail/2).
    two=tail/2
    return CompiledRegime(0,1,n_grid=q,p_flip=(tail/2)/(1-two),p_two=two,backend=backend)


def independent_full(c,backend):
    state=c.natural_statistics()
    mass=c.transition_prior()["mass"]
    return backend.real.evaluate(state["kernel_a"],state["kernel_b"],mass)["p_plus"]


def test_first_native_repeat_copy_tightening_and_exact_api(backend):
    c=tiny_tail(backend)
    first=c.approximate_p_plus_cells(.01)
    assert first["mode"]=="native" and first["owner_revision"]==0
    np.testing.assert_allclose(first["p_plus"],independent_full(c,backend),atol=1e-12)
    before=backend.calls;first["p_plus"][:]=0
    again=c.approximate_p_plus_cells(.01)
    assert backend.calls==before and np.all(again["p_plus"]>0)
    c.add_probe(.03,1,.55,evidence_id="one")
    flat=c.approximate_p_plus_cells(.01)
    assert flat["mode"]=="flat" and not flat["numerical_certificate"]
    assert flat["anchor_revision"]==0 and flat["owner_revision"]==1
    assert flat["error_bound_estimate"]>0
    oracle=independent_full(c,backend)
    assert np.max(np.abs(flat["p_plus"]-oracle))<=flat["error_bound_estimate"]+1e-8
    flat["p_plus"][:]=0
    assert np.all(c.approximate_p_plus_cells(.01)["p_plus"]>0)
    tighter=c.approximate_p_plus_cells(flat["error_bound_estimate"]*.5)
    assert tighter["mode"]=="native" and tighter["anchor_revision"]==1
    np.testing.assert_allclose(tighter["p_plus"],oracle,atol=1e-12)
    assert c.approximate_p_plus_cells(0)["mode"]=="native"
    # Public exact methods have unchanged semantics and detached outputs.
    out=c.p_plus_cells();out[:]=0
    np.testing.assert_allclose(c.p_plus_cells(),oracle,atol=1e-12)
    assert c.p_plus(.03)==pytest.approx(oracle[0],abs=1e-12)


def test_multiple_probes_and_revealing_fallback(backend):
    c=tiny_tail(backend,tail=1e-6)
    c.approximate_p_plus_cells(.01)
    saw_flat=saw_native=False
    for i in range(8):
        c.add_probe((i+.5)/16,1 if i%2==0 else -1,.85,evidence_id=f"physical:{i}")
        answer=c.approximate_p_plus_cells(.01)
        oracle=independent_full(c,backend)
        err=float(np.max(np.abs(answer["p_plus"]-oracle)))
        if answer["mode"]=="flat":
            saw_flat=True
            assert err<=answer["error_bound_estimate"]+1e-8
            assert answer["error_bound_estimate"]<=.01
        else:
            saw_native=True
            assert err<1e-11
        assert answer["owner_revision"]==i+1
    # The conditional guard can safely stay flat for all eight mixed reports;
    # the separate rare-branch revival test requires and checks native fallback.
    assert saw_flat


def test_tiny_supported_transition_revives_under_contrary_spatial_evidence(backend):
    c=tiny_tail(backend,tail=1e-6)
    initial_prior=c.transition_prior()["mass"].copy()
    assert initial_prior[1]+initial_prior[2]>0
    assert c.approximate_p_plus_cells(.01)["mode"]=="native"
    c.add_probe(.03,1,.55,evidence_id="prefix")
    assert c.approximate_p_plus_cells(.01)["mode"]=="flat"
    # A transition near the middle explains the two opposed regions while
    # either flat orientation must explain one entire region incorrectly.
    for repeat in range(3):
        for j in range(16):
            c.add_probe((j+.5)/16,1 if j<8 else -1,.99,
                        evidence_id=f"physical:{repeat}:{j}")
            c.approximate_p_plus_cells(.01)
    final=c.approximate_p_plus_cells(.01)
    exact=independent_full(c,backend)
    assert final["mode"]=="native"
    np.testing.assert_allclose(final["p_plus"],exact,atol=1e-12)
    assert c.family_mass()[1:].sum()>.99
    assert float(np.ptp(exact))>.8
    assert len(c.evidence_ids)==49 and c.natural_statistics()["n_probes"]==49
    np.testing.assert_array_equal(c.transition_prior()["mass"],initial_prior)


def test_tempered_probe_contrast_is_weighted_and_owner_state_retained(backend):
    c=tiny_tail(backend,tail=1e-6)
    c.approximate_p_plus_cells(.01)
    c.add_probe(.2,1,.85,weight=2.0,evidence_id="tempered")
    result=c.approximate_p_plus_cells(.01)
    assert result["mode"]=="flat"
    assert result["log_likelihood_contrast"]==pytest.approx(2*math.log(.85/.15))
    oracle=independent_full(c,backend)
    assert np.max(np.abs(result["p_plus"]-oracle))<=result["error_bound_estimate"]+1e-8
    assert c.natural_statistics()["n_probes"]==1 and c.evidence_ids==frozenset({"tempered"})


def test_claims_recalibration_and_failed_mutation_atomicity(backend):
    c=CompiledRegime(0,1,n_grid=4,backend=backend)
    c.approximate_p_plus_cells(.99)
    c.add_probe(.1,1,.55,evidence_id="first")
    assert c.approximate_p_plus_cells(.99)["mode"]=="flat"
    rev=c._owner_revision
    c.add_claim(0,.5,1,n_eff=2,reliability=.8)
    assert c._owner_revision==rev+1
    assert c.approximate_p_plus_cells(.99)["mode"]=="native"
    c.add_probe(.3,-1,.55,evidence_id="second")
    assert c.approximate_p_plus_cells(.99)["mode"]=="flat"
    c.set_claim_reliabilities([.9])
    assert c.approximate_p_plus_cells(.99)["mode"]=="native"
    old=c.approximate_p_plus_cells(.99);rev=c._owner_revision;calls=backend.calls
    with pytest.raises(ValueError):c.add_probe(.7,1,.85,evidence_id="first")
    with pytest.raises(ValueError):c.set_claim_reliabilities([1.0])
    with pytest.raises(FrozenPartitionError):c.add_claim(.11,.5,1)
    assert c._owner_revision==rev and backend.calls==calls
    np.testing.assert_array_equal(c.approximate_p_plus_cells(.99)["p_plus"],old["p_plus"])


def test_constructor_import_default_prior_and_hypothetical_is_not_anchor(backend):
    c=CompiledRegime(0,1,n_grid=16,backend=backend,
                     claims=[(0,.5,1,1,.8)],probes=[(.25,1,.55,1)],probe_ids=["imported"])
    assert c.approximate_p_plus_cells(.99)["mode"]=="native"
    assert c.approximate_p_plus_cells(.99)["owner_revision"]>=2
    default=CompiledRegime(0,1,n_grid=16,backend=backend)
    default.approximate_p_plus_cells(.5)
    default.add_probe(.2,1,.55,evidence_id="default:1")
    assert default.approximate_p_plus_cells(.5)["mode"]=="flat"
    regime=RegimePosterior(0,1,n_grid=8,p_flip=.2,p_two=.1)
    regime.add_probe(.2,1,reliability=.55)
    imported=CompiledRegime.from_regime(regime,backend,probe_ids=["physical:0"])
    assert imported.approximate_p_plus_cells(.9)["mode"]=="native"
    imported.add_probe(.3,1,.55,evidence_id="physical:1")
    assert imported.approximate_p_plus_cells(.9)["mode"]=="flat"
    rev=imported._owner_revision
    gain=imported.expected_gain(.4,reliability=.75)
    assert math.isfinite(gain["gain"]) and imported._owner_revision==rev
    # expected_gain used full current native state; its hypothetical branches
    # did not replace the anchor or create an actual evidence revision.
    assert imported.approximate_p_plus_cells(.9)["mode"]=="native"


def test_actual_default_prior_after_flat_favoring_history(backend):
    c=CompiledRegime(0,1,n_grid=16,backend=backend)
    np.testing.assert_allclose(c.transition_prior()["mass"],(.665,.285,.05),atol=1e-14)
    for i in range(64):
        j=min(15,(i*16)//64)
        c.add_probe((j+.5)/16,1,.99,evidence_id=f"prehistory:{i}")
    anchor=c.approximate_p_plus_cells(.01)
    assert anchor["mode"]=="native" and 0<anchor["tail_mass_at_anchor"]<.01
    c.add_probe(.03,1,.55,evidence_id="physical:0")
    answer=c.approximate_p_plus_cells(.01)
    assert answer["mode"]=="flat"
    error=float(np.max(np.abs(answer["p_plus"]-independent_full(c,backend))))
    assert error<=answer["error_bound_estimate"]+1e-8<=.01+1e-8


def test_conditional_guard_uses_coherent_evidence_and_charges_delta_arrays(backend):
    c=CompiledRegime(0,1,n_grid=16,backend=backend)
    assert c._flat_anchor is None
    for i in range(64):
        j=i//4
        c.add_probe((j+.5)/16,1,.99,evidence_id=f"history:{i}")
    anchor=c.approximate_p_plus_cells(.01)
    assert anchor["mode"]=="native"
    assert anchor["approx_anchor_numeric_bytes"]==16*16
    chosen=[]
    for i in range(8):
        c.add_probe((i+.5)/16,1,.55,evidence_id=f"coherent:{i}")
        answer=c.approximate_p_plus_cells(.01)
        chosen.append(answer["bound_choice"])
        assert answer["mode"]=="flat"
        assert answer["approx_anchor_numeric_bytes"]==16*16
        assert answer["conditional_bound_estimate"]<=answer["generic_bound_estimate"]+1e-14
        assert np.max(np.abs(answer["p_plus"]-independent_full(c,backend)))<=answer["error_bound_estimate"]+1e-8
        state=c.natural_statistics()
        full=backend.real.evaluate(state["kernel_a"],state["kernel_b"],c.transition_prior()["mass"])
        assert float(np.sum(full["family_mass"][1:]))<=answer["error_bound_estimate"]+1e-8
    assert "conditional" in chosen
    c.add_claim(0,.5,1,n_eff=1,reliability=.8)
    assert c._flat_anchor is None
    assert c.approximate_p_plus_cells(.01)["mode"]=="native"


def test_opposed_same_cell_cancels_da_and_tighter_request_refreshes(backend):
    c=tiny_tail(backend,tail=1e-4)
    c.approximate_p_plus_cells(.9)
    c.add_probe(.03,1,.85,evidence_id="plus")
    c.add_probe(.03,-1,.85,evidence_id="minus")
    answer=c.approximate_p_plus_cells(.9)
    assert answer["mode"]=="flat" and answer["bound_choice"]=="conditional"
    assert answer["conditional_bound_estimate"]<answer["generic_bound_estimate"]
    assert abs(c._flat_anchor["delta_a"][0])<1e-14
    assert np.max(np.abs(answer["p_plus"]-independent_full(c,backend)))<=answer["error_bound_estimate"]+1e-8
    state=c.natural_statistics()
    full=backend.real.evaluate(state["kernel_a"],state["kernel_b"],c.transition_prior()["mass"])
    assert float(np.sum(full["family_mass"][1:]))<=answer["error_bound_estimate"]+1e-8
    answer["p_plus"][:]=0
    assert np.all(c.approximate_p_plus_cells(.9)["p_plus"]>0)
    tighter=c.approximate_p_plus_cells(answer["error_bound_estimate"]*.5)
    assert tighter["mode"]=="native" and tighter["bound_choice"]=="native"
    assert tighter["approx_anchor_numeric_bytes"]==16*16


def test_nonfinite_conditional_intermediates_force_native_fallback(backend):
    c=tiny_tail(backend)
    c.approximate_p_plus_cells(.9)
    c._flat_anchor["anchor_h"]=1e308  # adversarial derived-cache corruption
    c.add_probe(.03,1,.55,evidence_id="overflow-2h")
    assert c.approximate_p_plus_cells(.9)["mode"]=="native"
    c._flat_anchor["sum_da"]=1e308
    c._flat_anchor["sum_db"]=1e308
    c.add_probe(.07,1,.55,evidence_id="overflow-log-z")
    assert c.approximate_p_plus_cells(.9)["mode"]=="native"


def test_invalid_tolerances_rounded_zero_and_nonfinite_guard(backend):
    c=tiny_tail(backend)
    for bad in (True,False,-1,1,math.nan,math.inf,-math.inf):
        with pytest.raises(ValueError):c.approximate_p_plus_cells(bad)
    class RoundedZero(CountingBackend):
        def evaluate(self,*args,**kwargs):
            result=super().evaluate(*args,**kwargs)
            result["family_mass"]=np.array([1.,0.,0.])
            return result
    fake=RoundedZero(backend.real)
    z=tiny_tail(fake)
    first=z.approximate_p_plus_cells(.9)
    assert first["mode"]=="native" and not first["tail_resolved"]
    assert first["prior_tail_mass"]>0 and first["tail_mass_at_anchor"]==0
    z.add_probe(.2,1,.55,evidence_id="z")
    assert z.approximate_p_plus_cells(.9)["mode"]=="native"
    huge=tiny_tail(backend)
    huge.approximate_p_plus_cells(.9)
    huge.add_probe(.2,1,.99,weight=1e308,evidence_id="huge")
    with pytest.raises((ValueError,ArithmeticError,OverflowError)):
        huge.approximate_p_plus_cells(.9)
