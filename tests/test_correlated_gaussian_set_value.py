"""Regressions for pricing explicit correlated Gaussian observation bundles."""
import numpy as np
import pytest
from graph_engine.precision_form import PrecisionForm

def prior(d=2):
    j=np.eye(d)*2;return PrecisionForm(j,np.zeros(d))

@pytest.mark.parametrize('scale',[1e-6,1.,1e6])
def test_diagonal_covariance_reproduces_existing_gaussian_value(scale):
    f=prior();h=scale*np.array([[1.,2.],[-1.,.5],[.2,1.]])
    sig=scale*np.array([.7,1.1,2.])
    expected=f.set_value_bits(h,sig,exact_bernoulli=False)
    assert f.gaussian_set_value_bits(h,np.diag(sig**2))==pytest.approx(expected,abs=1e-12)

def test_joint_value_matches_existing_joint_measurement_posterior_logdet():
    f=prior();h=np.array([[1.,2.],[-1.,.5],[.2,1.]])
    r=np.array([[2.,.4,-.1],[.4,1.5,.2],[-.1,.2,1.]])
    out=f.copy();out.add_measurement(h,[.2,.5,-.3],r)
    expected=(np.linalg.slogdet(f.cov())[1]-np.linalg.slogdet(out.cov())[1])/(2*np.log(2))
    assert f.gaussian_set_value_bits(h,r)==pytest.approx(expected,abs=1e-12)

def test_correlated_bundle_changes_the_frozen_choice():
    f=PrecisionForm(np.eye(1),np.zeros(1));m=16;cross=np.ones((m,m))*.05
    r=np.block([[np.eye(m),cross],[cross,np.eye(m)]]);h=np.ones((32,1))
    independent_alternative=f.gaussian_set_value_bits([[1.]],[[1/24]])
    assert f.set_value_bits(h,exact_bernoulli=False)>independent_alternative
    assert f.gaussian_set_value_bits(h,r)<independent_alternative
    assert f.gaussian_set_value_bits(h,r)==pytest.approx(.5*np.log2(1+160/9),abs=1e-12)

def test_invertible_row_change_preserves_information():
    f=prior();h=np.array([[1.,2.],[-1.,.5]])
    r=np.array([[1.,.6],[.6,2.]]);t=np.array([[2.,-.2],[.3,.8]])
    assert f.gaussian_set_value_bits(t@h,t@r@t.T)==pytest.approx(f.gaussian_set_value_bits(h,r),abs=1e-12)

def test_existing_exact_bernoulli_path_is_unchanged():
    f=PrecisionForm.zeros(1).add_bernoulli(0,.3);h=np.array([[.8],[.7]])
    before=f.set_value_bits(h)
    f.gaussian_set_value_bits(h,np.array([[1.,.2],[.2,1.]]))
    assert f.set_value_bits(h)==before

@pytest.mark.parametrize('h,r',[
    ([[1.,0.]],[[np.nan]]),
    ([[np.inf,0.]],[[1.]]),
    ([[1.,0.]],[[0.]]),
    ([[1.,0.],[0.,1.]],[[1.,2.],[2.,1.]]),
    ([[1.,0.],[0.,1.]],[[1.,.2],[.1,1.]]),
    ([[1.]],[[1.]]),
    ([[1.,0.]],[[1.,0.],[0.,1.]]),
    ([[1.+0j,0.]],[[1.]]),
    ([[1.,0.]],[[1.+0j]]),
])
def test_invalid_models_refuse_without_changing_form(h,r):
    f=prior();j=f.J.copy();b=f.b.copy();c=f.cov().copy()
    with pytest.raises(ValueError):f.gaussian_set_value_bits(h,r)
    assert np.array_equal(f.J,j) and np.array_equal(f.b,b) and np.array_equal(f.cov(),c)

def test_empty_set_has_zero_value():
    assert prior().gaussian_set_value_bits(np.empty((0,2)),np.empty((0,0)))==0

def test_valid_inputs_remain_unchanged():
    f=prior();h=np.array([[1.,2.],[-1.,.5]]);r=np.array([[1.,.6],[.6,2.]])
    hh=h.copy();rr=r.copy();jj=f.J.copy();bb=f.b.copy()
    f.gaussian_set_value_bits(h,r)
    assert np.array_equal(h,hh) and np.array_equal(r,rr) and np.array_equal(f.J,jj) and np.array_equal(f.b,bb)
