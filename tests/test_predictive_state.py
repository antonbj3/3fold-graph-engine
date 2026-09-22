"""Independent adversaries for an observable bridge, not implementation echoes."""
import numpy as np
import pytest

from graph_engine.predictive_state import (
    BeliefState, ObservationChannel, PredictiveTask, RegimeSnapshot, value_of_observation,
)
from graph_engine.regime_posterior import RegimePosterior


def binary_task(key, target, loss="log"):
    y = np.asarray(target)
    return PredictiveTask(key, np.stack([1-y, y], axis=-1)[:, None, :], [1.0], loss)


def test_xor_sensors_are_useless_separately_and_decisive_jointly():
    # Four equiprobable worlds (a,b). Neither bit predicts a XOR b on its own.
    p = BeliefState("xor", [0.25]*4)
    task = binary_task("xor", [0, 1, 1, 0])
    a = ObservationChannel("a", "xor", [[1,0], [1,0], [0,1], [0,1]])
    b = ObservationChannel("b", "xor", [[1,0], [0,1], [1,0], [0,1]])
    joint = ObservationChannel("ab", "xor", np.eye(4), cost=2)
    assert value_of_observation(p, task, a).gain == 0
    assert value_of_observation(p, task, b).gain == 0
    assert value_of_observation(p, task, joint).gain == pytest.approx(1)
    assert value_of_observation(p, task, joint).gain_per_cost == pytest.approx(0.5)
    after_a = p.observe(a, 1, "sensor-a:1")
    assert value_of_observation(after_a, task, b).gain == pytest.approx(1)


def test_global_branches_survive_identical_readouts_until_intervention():
    p = BeliefState("mirror-worlds", [0.5, 0.5])
    identical = ObservationChannel("lengths", p.space_key, [[0,1], [0,1]])
    chirality = ObservationChannel("signed-volume", p.space_key, np.eye(2))
    task = binary_task(p.space_key, [0,1])
    for i in range(20):
        p = p.observe(identical, 1, f"length:{i}")
    np.testing.assert_array_equal(p.mass, [0.5,0.5])
    assert task.risk(p) == 1
    np.testing.assert_array_equal(p.observe(chirality, 1, "volume:1").mass, [0,1])


def test_repeated_evidence_does_not_multiply_odds_but_new_evidence_can():
    p = BeliefState("coin", [0.5,0.5])
    ch = ObservationChannel("same-instrument", "coin", [[0.8,0.2], [0.2,0.8]])
    once = p.observe(ch, 1, "physical-reading:1")
    np.testing.assert_allclose(once.mass, [0.2,0.8])
    with pytest.raises(ValueError, match="already consumed"):
        once.observe(ch, 1, "physical-reading:1")
    # New independent measurement from SAME source: odds 4 -> 16.
    twice = once.observe(ch, 1, "physical-reading:2")
    np.testing.assert_allclose(twice.mass, [1/17,16/17])
    # A conditional perfect copy is certain, so its conditional likelihood is 1.
    copy = ObservationChannel("copy", "coin", [[0,1],[0,1]], conditioned_on=once.fingerprint)
    np.testing.assert_array_equal(once.observe(copy, 1, "copy:1").mass, once.mass)
    with pytest.raises(ValueError, match="another evidence state"):
        twice.observe(copy, 1, "copy:1")


def test_task_value_prefers_relevant_evidence_over_more_nuisance_bits():
    # 3-bit nuisance and 1-bit target; complete nuisance reveals 3 bits vs 1.
    p = BeliefState("nuisance", np.full(16, 1/16))
    target = np.tile([0,1],8)
    task = binary_task(p.space_key, target)
    nuisance = ObservationChannel("nuisance", p.space_key, np.eye(8)[np.arange(16)//2])
    useful = ObservationChannel("target", p.space_key, np.eye(2)[target])
    assert value_of_observation(p, task, nuisance).gain == 0
    assert value_of_observation(p, task, useful).gain == pytest.approx(1)


@pytest.mark.parametrize("loss,initial", [("log",1), ("brier",0.5), ("error",0.5)])
def test_exact_measurement_eliminates_bayes_risk(loss, initial):
    p = BeliefState("bit", [0.5,0.5])
    task = binary_task("bit", [0,1], loss)
    v = value_of_observation(p, task, ObservationChannel("read", "bit", np.eye(2)))
    assert v.risk_before == pytest.approx(initial)
    assert v.expected_risk_after == 0


def test_regime_bridge_matches_existing_predictions_update_and_oed():
    r = RegimePosterior(-2, 3, n_grid=9, p_two=0.2)
    r.add_claim(-1.7,-0.4,1,reliability=0.8)
    r.add_claim(1.0,2.8,-1,reliability=0.9)
    r.add_probe(0.3, 1, reliability=0.87)
    snapshot = RegimeSnapshot.from_regime(r, frozenset({"existing-probe"}))
    predicted = snapshot.task().predictions(snapshot.belief)[:,1]
    np.testing.assert_allclose(predicted, [r.p_plus(x) for x in snapshot.cells.mean(axis=1)])
    assert snapshot.belief.mass[snapshot.single_family_count:].sum() == pytest.approx(
        r.collision()["p_two_transitions"])
    x, gain = r.best_probe(reliability=0.91)
    value = value_of_observation(snapshot.belief, snapshot.task(), snapshot.probe(x,0.91))
    # Current regime potential integrates domain length; new task averages demand.
    assert value.gain * 5 == pytest.approx(gain, abs=1e-12)
    updated = snapshot.belief.observe(snapshot.probe(x,0.91), 0, "new-probe")
    r.add_probe(x,-1,reliability=0.91)
    np.testing.assert_allclose(updated.mass, r._with_probes()[3], atol=1e-14)


def test_impossible_observation_preserves_excluded_branch_and_demands_expansion():
    p = BeliefState("excluded", [1,0])
    with pytest.raises(ValueError, match="impossible"):
        p.observe(ObservationChannel("read", "excluded", np.eye(2)), 1, "reading")
    np.testing.assert_array_equal(p.mass,[1,0])


def test_unlikely_branch_recovers_after_more_than_float_probability_dynamic_range():
    p = BeliefState("long-history", [0.5,0.5])
    ch = ObservationChannel("read", p.space_key, [[0.8,0.2], [0.2,0.8]])
    # Probability-space updates would underflow, permanently destroying one branch.
    for i in range(600):
        p = p.observe(ch, 1, f"positive:{i}")
    assert p.mass[0] == 0
    for i in range(600):
        p = p.observe(ch, 0, f"negative:{i}")
    np.testing.assert_allclose(p.mass,[0.5,0.5],atol=1e-10)


def test_arrays_are_copied_and_space_mismatch_is_not_silently_accepted():
    source = np.array([0.4,0.6])
    p = BeliefState("v1",source)
    source[:] = [0.9,0.1]
    np.testing.assert_array_equal(p.mass,[0.4,0.6])
    assert not p.mass.flags.writeable
    with pytest.raises(ValueError,match="different hypothesis"):
        p.predict(ObservationChannel("renamed-worlds","v2",np.eye(2)))


@pytest.mark.parametrize("bad", [[-0.1,1.1], [np.nan,0], [0.2,0.2], [np.inf,0], []])
def test_invalid_prior_is_not_repaired_into_a_different_model(bad):
    with pytest.raises(ValueError):
        BeliefState("bad",bad)
