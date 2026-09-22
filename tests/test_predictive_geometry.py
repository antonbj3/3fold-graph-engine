import numpy as np
import pytest

from graph_engine.predictive_geometry import (
    squared_pair_distances, pair_distortion, fisher_pullback, js_pair_divergence,
)


def bernoulli(p):
    p = np.asarray(p)
    return np.stack([1-p,p],axis=-1)


def test_fisher_distance_matches_independent_binary_flat_coordinate():
    p = np.array([0,1e-14,0.01,0.2,0.5,0.87,1-1e-14,1])
    pairs = np.array([(i,j) for i in range(len(p)) for j in range(len(p))])
    actual = squared_pair_distances(bernoulli(p)[:,None,:], [1], pairs, metric="fisher")
    # theta uses an independent arcsine identity, not the implementation half-angle.
    theta = 2*np.arcsin(np.sqrt(p))
    expected = (theta[pairs[:,0]]-theta[pairs[:,1]])**2
    np.testing.assert_allclose(actual,expected,atol=1e-9)
    assert actual.max() == pytest.approx(np.pi**2)


def test_pair_geometry_cannot_detect_semantic_label_swap():
    p = bernoulli([0.1,0.3,0.7,0.9])[:,None,:]
    swapped = p[:,:,::-1]
    pairs = [[0,1],[0,2],[0,3],[1,2],[2,3]]
    for metric in ["hellinger","fisher"]:
        loss,_ = pair_distortion(swapped,p,[1],pairs,metric=metric)
        assert loss == pytest.approx(0,abs=1e-30)
    # Every classification reverses, although all relations above are exact.
    assert np.all(p.argmax(axis=-1) != swapped.argmax(axis=-1))


def test_fixed_noncollapsed_teacher_rejects_student_collapse():
    p = bernoulli([0.05,0.25,0.75,0.95])[:,None,:]
    collapse = np.full_like(p,0.5)
    loss,_ = pair_distortion(collapse,p,[1],[[0,3],[1,2]])
    assert loss > 0.1


def test_product_fisher_is_not_ambient_sphere_geodesic():
    p = np.array([[[1,0],[1,0]], [[1,0],[0,1]]],float)
    # One query unchanged, one changes between disjoint outcomes.
    actual = squared_pair_distances(p,[0.5,0.5],[[0,1]],metric="fisher")[0]
    assert actual == pytest.approx(0.5*np.pi**2)
    ambient = (2*np.arccos(0.5))**2
    assert actual > ambient + 0.1


def test_pullback_recovers_bernoulli_information_and_a_null_direction():
    p = 0.3
    j = np.array([[[-1,0],[1,0]]])
    g = fisher_pullback([[1-p,p]],j,[1])
    np.testing.assert_allclose(g,[[1/(p*(1-p)),0],[0,0]])
    delta = 1e-5
    distance = squared_pair_distances(bernoulli([p,p+delta])[:,None,:], [1], [[0,1]],
                                      metric="fisher")[0]
    assert distance/delta**2 == pytest.approx(g[0,0],rel=1e-4)


def test_close_points_have_eight_to_one_fisher_over_squared_hellinger_ratio():
    p = bernoulli([0.4,0.400001])[:,None,:]
    h = squared_pair_distances(p,[1],[[0,1]])[0]
    f = squared_pair_distances(p,[1],[[0,1]],metric="fisher")[0]
    assert f/h == pytest.approx(8,rel=1e-10)


def test_same_points_exactly_zero_and_missing_queries_remain_unobserved():
    p = np.array([[[0.2,0.8],[1,0]], [[0.2,0.8],[0,1]]])
    for metric in ["hellinger","fisher"]:
        d = squared_pair_distances(p,[1,0],[[0,1],[0,0]],metric=metric)
        np.testing.assert_array_equal(d,[0,0])


def test_pullback_rejects_non_tangent_derivative():
    with pytest.raises(ValueError,match="tangent"):
        fisher_pullback([[0.4,0.6]],[[[1],[1]]],[1])


def test_gan_optimal_discriminator_matches_js_with_correct_nats_bits_conversion():
    p = np.array([0.13,0.22,0.65])
    q = np.array([0.62,0.31,0.07])
    discriminator = p/(p+q)
    # Direct optimal balanced adversarial classification objective in natural logs.
    objective = p@np.log(discriminator) + q@np.log(1-discriminator)
    js_bits = js_pair_divergence(np.array([p,q])[:,None,:],[1],[[0,1]])[0]
    assert objective == pytest.approx(-np.log(4)+2*np.log(2)*js_bits,abs=1e-14)


def test_conf_js_and_fisher_have_the_predicted_local_scale():
    p = bernoulli([0.4,0.40001])[:,None,:]
    js = js_pair_divergence(p,[1],[[0,1]])[0]
    fisher = squared_pair_distances(p,[1],[[0,1]],metric="fisher")[0]
    assert js/fisher == pytest.approx(1/(8*np.log(2)),rel=1e-5)
    # Boundary, exact coincidence and disjoint support need no arbitrary epsilon.
    endpoints = bernoulli([0,1,0.5])[:,None,:]
    np.testing.assert_allclose(js_pair_divergence(endpoints,[1],[[0,0],[0,1]]),[0,1])
