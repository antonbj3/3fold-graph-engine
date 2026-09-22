"""The inference/measurement return path: typed predictive tasks through EXISTING next_actions, apply and Alarm.

This is not a parallel controller. It binds `predictive_state` (belief, task, explicit channels) into the
existing EngineState/next_actions/apply ledger, and feeds the existing `Alarm` the frozen pre-observation
risk-drop law. P1 units; P2 the apply ledger; P3 stale/changed fail before mutation; P4 evidence identity;
P5 impossible-outcome atomicity; P6 the exact binary risk law (including a negative individual drop); P7 the
Alarm receives the finite law and the residual is a martingale; P8 correlated joint observations need a
caller joint law, not weight 1/K; P9 a calibrated executable instrument plan -> reading -> apply -> Alarm.
"""
import numpy as np
import pytest

from graph_engine.alarm import Alarm
from graph_engine.instruments import Calibration, Instrument as ExecutableInstrument, Reading, calibrated_action, calibrated_outcome
from graph_engine.next_actions import (COST_UNIT, EngineState, PredictiveBinding, apply, binding_of,
                                       next_actions, predictive_action, predictive_expected_drop, update_alarm)
from graph_engine.predictive_state import (BeliefState, ObservationChannel, PredictiveTask, risk_drop_law,
                                           value_of_observation)

KEY = "predictive-test"
Y = np.array([0.0, 1.0])                                   # the readout labels h0, h1


def _task(loss="log", key=KEY):
    return PredictiveTask(key, np.stack([1 - Y, Y], axis=-1)[:, None, :], [1.0], loss)


def _binding(belief=None, channel=None, loss="log", cost_unit=COST_UNIT, task_id="bit"):
    belief = BeliefState(KEY, [0.6, 0.4]) if belief is None else belief
    channel = ObservationChannel("read", KEY, [[0.9, 0.1], [0.15, 0.85]], cost=2.0) if channel is None else channel
    return PredictiveBinding(task_id, belief, _task(loss), (channel,), cost_unit=cost_unit)


def _state(*bindings, **kw):
    return EngineState(predictive_bindings=list(bindings), pair_bundles=False, probe_instruments=[], **kw)


# -- P1: units --------------------------------------------------------------------------------------
def test_log_loss_joins_the_bits_ranking_and_brier_error_stay_out_in_their_own_units():
    for loss, in_bits in (("log", True), ("brier", False), ("error", False)):
        st = _state(_binding(loss=loss))
        acts = next_actions(st, k=5)
        assert (len(acts) == 1) is in_bits
        assert (len(acts.other) == 1) is (not in_bits)
        a = acts[0] if in_bits else acts.other[0]
        assert a.kind == "observe" and a.value_unit == ("bits" if loss == "log" else loss)
        assert a.value_bits == pytest.approx(value_of_observation(st.predictive_bindings[0].belief,
                                                                  st.predictive_bindings[0].task,
                                                                  st.predictive_bindings[0].channels[0]).gain)
        assert a.cost == 2.0 and a.value_per_cost == pytest.approx(a.value_bits / 2.0)


def test_a_cost_declared_in_another_currency_is_never_divided_into_the_bits_ranking():
    st = _state(_binding(cost_unit="joules"))
    acts = next_actions(st, k=5)
    assert not acts                                                   # nothing in the graph's currency
    assert [a.cost_unit for a in acts.other] == ["joules"]
    assert acts.other[0].value_unit == "bits"                         # bits, but a different currency
    # the two currencies are still returned beside each other, never in one ranking
    mixed = _state(_binding(cost_unit=COST_UNIT, task_id="g"), _binding(cost_unit="joules", task_id="j"))
    acts = next_actions(mixed, k=5)
    assert {a.cost_unit for a in acts} == {COST_UNIT} and [a.cost_unit for a in acts.other] == ["joules"]


# -- P2: the apply ledger ---------------------------------------------------------------------------
def test_apply_routes_an_observation_and_returns_the_ledger_plus_risk_and_costs():
    st = _state(_binding())
    binding = st.predictive_bindings[0]
    act = predictive_action(binding, 0)
    before = binding.task.risk(binding.belief)
    row = apply(st, act, {"evidence_id": "physics:run17/reading1", "outcome": 1, "cost": 1.7, "source": "run:17"})
    after = binding.task.risk(st.predictive_bindings[0].belief)
    assert row["action"] == "observe:('bit', 'read')" and row["outcome"] == "observe" and row["supersedes"] == []
    assert row["value_predicted"] == pytest.approx(act.value_bits)
    assert row["value_realized"] == pytest.approx(before - after)
    assert row["risk_before"] == pytest.approx(before) and row["risk_after"] == pytest.approx(after)
    assert row["evidence_id"] == "physics:run17/reading1" and row["source_root"] == "run:17"
    assert row["planned_cost"] == pytest.approx(2.0) and row["actual_cost"] == pytest.approx(1.7)
    assert row["cost_unit"] == COST_UNIT and row["value_unit"] == "bits"
    assert st.predictive_bindings[0].belief.evidence_ids == {"physics:run17/reading1"}


# -- P3: stale belief / changed task / changed channel fail before mutation --------------------------
def test_a_stale_action_a_changed_task_and_a_changed_channel_all_fail_before_mutation():
    st = _state(_binding())
    binding = st.predictive_bindings[0]
    stale = predictive_action(binding, 0)
    apply(st, stale, {"evidence_id": "e1", "outcome": 1})              # belief advances
    snapshot = st.predictive_bindings[0].belief
    with pytest.raises(ValueError, match="stale predictive action"):
        apply(st, stale, {"evidence_id": "e2", "outcome": 1})
    assert st.predictive_bindings[0].belief is snapshot               # untouched

    fresh = predictive_action(binding, 0)
    binding.task = PredictiveTask(KEY, np.stack([1 - Y, Y], axis=-1)[:, None, :], [1.0], "brier")
    with pytest.raises(ValueError, match="predictive task changed"):
        apply(st, fresh, {"evidence_id": "e3", "outcome": 1})
    assert st.predictive_bindings[0].belief is snapshot


def test_a_changed_channel_likelihood_is_refused():
    st = _state(_binding())
    binding = st.predictive_bindings[0]
    act = predictive_action(binding, 0)
    binding.channels = (ObservationChannel("read", KEY, [[0.8, 0.2], [0.3, 0.7]], cost=2.0),)
    with pytest.raises(ValueError, match="predictive channel changed"):
        apply(st, act, {"evidence_id": "e1", "outcome": 1})


def test_an_action_from_another_binding_is_refused():
    st = _state(_binding())
    act = predictive_action(_binding(task_id="elsewhere"), 0)
    with pytest.raises(ValueError, match="not in this state"):
        apply(st, act, {"evidence_id": "e1", "outcome": 1})


# -- P4: evidence identity --------------------------------------------------------------------------
def test_a_copied_physical_observation_is_refused_but_independent_same_source_readings_are_not():
    st = _state(_binding())
    binding = st.predictive_bindings[0]
    apply(st, predictive_action(binding, 0), {"evidence_id": "phys:1", "outcome": 1, "source": "campaign:9"})
    with pytest.raises(ValueError, match="already consumed"):
        apply(st, predictive_action(binding, 0), {"evidence_id": "phys:1", "outcome": 0, "source": "campaign:9"})
    row = apply(st, predictive_action(binding, 0), {"evidence_id": "phys:2", "outcome": 0, "source": "campaign:9"})
    assert row["source_root"] == "campaign:9"
    assert st.predictive_bindings[0].belief.evidence_ids == {"phys:1", "phys:2"}


# -- P5: atomicity ----------------------------------------------------------------------------------
def test_an_impossible_outcome_and_a_bad_index_leave_the_belief_untouched():
    st = _state(_binding())
    binding = st.predictive_bindings[0]
    before = binding.belief
    with pytest.raises(ValueError, match="impossible"):
        apply(st, predictive_action(binding, 0), {"evidence_id": "e1", "outcome": 5})
    with pytest.raises(ValueError, match="integer index"):
        apply(st, predictive_action(binding, 0), {"evidence_id": "e2", "outcome": "positive"})
    assert st.predictive_bindings[0].belief is before
    assert st.predictive_bindings[0].belief.evidence_ids == frozenset()


# -- P6: the exact binary risk/outcome law ----------------------------------------------------------
def test_the_frozen_law_is_exact_and_has_a_negative_individual_drop_while_the_mean_is_positive():
    # A peaked belief and a channel whose likely outcome pushes the belief the WRONG way: that outcome's
    # realized risk drop is negative, yet the Bayes-risk drop is nonnegative in expectation.
    belief = BeliefState(KEY, [0.9, 0.1])
    ch = ObservationChannel("r", KEY, [[0.5, 0.5], [0.9, 0.1]], cost=1.0)
    task = _task()
    law = risk_drop_law(belief, task, ch)
    assert min(d for _q, d in law) < 0 < predictive_expected_drop(law)
    value = value_of_observation(belief, task, ch)
    assert predictive_expected_drop(law) == pytest.approx(value.gain, abs=1e-12)
    # every outcome's drop is exactly the difference apply produces on that outcome
    for index, (q, drop) in enumerate(law):
        b = BeliefState(KEY, [0.9, 0.1])
        st = _state(PredictiveBinding("bit", b, task, (ch,)))
        row = apply(st, predictive_action(st.predictive_bindings[0], 0), {"evidence_id": f"e{index}", "outcome": index})
        assert q * drop == pytest.approx(q * row["value_realized"], abs=1e-12)


# -- P7: the Alarm receives the finite law ----------------------------------------------------------
def test_the_alarm_receives_the_frozen_finite_law_and_the_residual_has_zero_mean_under_the_model():
    st = _state(_binding())
    act = predictive_action(st.predictive_bindings[0], 0)
    law = act.meta["risk_law"]
    # one update runs the EXACT path (not the bounded fallback) and leaves a positive wealth
    assert update_alarm(Alarm(), law, act.value_bits)[0] > 0
    with pytest.raises(ValueError, match="frozen pre-observation"):
        update_alarm(Alarm(), [], 0.0)

    # Sample the outcome from the MODEL's own predictive law; the realized drop must have mean = price.
    rng = np.random.default_rng(7)
    q = st.predictive_bindings[0].belief.predict(st.predictive_bindings[0].channels[0])
    residuals, alarm = [], Alarm()
    for i in range(4000):
        b = _binding()
        s = _state(b)
        a = predictive_action(s.predictive_bindings[0], 0)
        outcome = int(rng.choice(len(q), p=q))
        row = apply(s, a, {"evidence_id": f"mc:{i}", "outcome": outcome})
        _, _al = update_alarm(alarm, law, row["value_realized"])
        residuals.append(row["value_realized"] - act.value_bits)
    se = float(np.std(residuals, ddof=1) / np.sqrt(len(residuals)))
    assert abs(float(np.mean(residuals))) <= 3 * se
    assert alarm.state()["n_bounded"] == 0                            # the exact path, never the fallback
    assert alarm.n == 4000


# -- P8: correlated joint observations --------------------------------------------------------------
def test_correlated_readings_need_a_caller_joint_law_not_the_per_reading_weight_one_over_k():
    # h0: the two bits always agree; h1: they always differ. Each single bit is 50/50 under BOTH
    # hypotheses, so no single reading or product of marginals carries information; the JOINT reading
    # determines the hypothesis. This is the exact situation weight 1/K cannot represent.
    belief = BeliefState(KEY, [0.5, 0.5])
    joint = ObservationChannel("pair", KEY, [[0.5, 0.0, 0.0, 0.5],        # indices (00,01,10,11)
                                             [0.0, 0.5, 0.5, 0.0]], cost=3.0)
    marginal = ObservationChannel("one", KEY, [[0.5, 0.5], [0.5, 0.5]], cost=1.0)
    product = ObservationChannel("product", KEY, np.full((2, 4), 0.25), cost=2.0)   # outer product of marginals
    task = _task()
    assert value_of_observation(belief, task, marginal).gain == pytest.approx(0.0)
    assert value_of_observation(belief, task, product).gain == pytest.approx(0.0)
    assert value_of_observation(belief, task, joint).gain == pytest.approx(1.0)     # exact joint = 1 bit
    # a sweep entered as ONE lineage root at weight 1/K is a separate, unchanged mechanism; it is not
    # this joint law (its own test pins its weights), so the correlated reading must go through `joint`.
    jb = PredictiveBinding("pair-task", belief, task, (joint,))
    st = _state(jb)
    a = next(x for x in next_actions(st, k=5) if x.kind == "observe")
    assert a.value_bits == pytest.approx(1.0) and a.cost == 3.0


# -- P9: a calibrated executable instrument, plan -> reading -> apply -> Alarm -----------------------
class _SignInstrument(ExecutableInstrument):
    """The smallest executable instrument: one sign with one reading probability at one cost."""

    def __init__(self, readings):
        self._readings = list(readings)
        self._i = 0
        super().__init__("sign-probe", cost=2.5, reliability=0.9, lineage_root="cell:sweep-A",
                         derives_from=["campaign:A"])

    def measure(self, pair, x):
        r = self._readings[self._i % len(self._readings)]
        self._i += 1
        return r


def test_a_calibrated_instrument_plan_reading_apply_alarm_path_end_to_end():
    # Caller's calibration: P(reading outcome | hypothesis). The instrument reports sign +1 for outcome 0.
    channel = ObservationChannel("calibrated", KEY, [[0.85, 0.15], [0.2, 0.8]], cost=2.5)
    calibration = Calibration("calibration:sign-v3", channel, lambda r: 0 if r.sign > 0 else 1)
    instrument = _SignInstrument([Reading(+1, 0.95, 2.5)])
    b = _binding(channel=channel, task_id="cal")
    st = _state(b)

    action = calibrated_action(st, "cal", instrument, calibration)
    assert action.meta["calibration_id"] == "calibration:sign-v3"
    assert action.meta["instrument_lineage_root"] == "cell:sweep-A"
    assert action.meta["instrument_sources"] == [{"id": "cell:sweep-A", "derives_from": ["campaign:A"]}]
    # the price-tag reliability is not the value source: the value is the exact calibrated-likelihood drop
    assert action.value_bits == pytest.approx(value_of_observation(b.belief, b.task, channel).gain)

    outcome = calibrated_outcome(instrument, calibration, ("a", "b"), 0.4, "cell:sweep-A/reading1")
    assert outcome["outcome"] == 0 and outcome["evidence_id"] == "cell:sweep-A/reading1"
    assert outcome["reading"] == {"sign": 1, "p": 0.95}
    row = apply(st, action, outcome)
    assert row["channel"] == "calibrated" and row["task_id"] == "cal" and row["risk_after"] < row["risk_before"]
    assert st.predictive_bindings[0].belief.mass[0] > 0.8              # outcome 0 favours h0 (likelihood 0.85)

    wealth, alarm_on = update_alarm(Alarm(), action.meta["risk_law"], row["value_realized"])
    assert wealth > 0 and alarm_on is False


def test_the_reading_probability_is_not_turned_into_reliability_and_the_calibration_identity_is_kept():
    channel = ObservationChannel("calibrated", KEY, [[0.85, 0.15], [0.2, 0.8]], cost=2.5)
    calibration = Calibration("calibration:sign-v3", channel, lambda r: 0 if r.sign > 0 else 1)
    # The same sign at wildly different Reading.p maps to the SAME outcome index: p is never converted.
    assert calibration.outcome_index(Reading(+1, 0.51, 2.5)) == calibration.outcome_index(Reading(+1, 0.99, 2.5)) == 0
    assert calibration.outcome_index(Reading(-1, 0.49, 2.5)) == 1
    with pytest.raises(ValueError, match="outside the channel"):
        Calibration("c", channel, lambda r: 9).outcome_index(Reading(+1, 0.9, 1.0))
    with pytest.raises(TypeError, match="Reading"):
        Calibration("c", channel, 5)
    with pytest.raises(ValueError, match="nonempty identity"):
        Calibration("", channel, lambda r: 0)
    b = _binding(channel=channel, task_id="cal")
    st = _state(b)
    other = ObservationChannel("elsewhere", KEY, [[0.5, 0.5], [0.5, 0.5]], cost=1.0)
    instrument = _SignInstrument([Reading(+1, 0.9, 2.5)])
    with pytest.raises(ValueError, match="not one of the binding's declared channels"):
        calibrated_action(st, "cal", instrument, Calibration("c2", other, lambda r: 0))
    with pytest.raises(ValueError, match="exactly one predictive binding"):
        calibrated_action(st, "absent", instrument, calibration)
    assert binding_of(st, "cal") is b


@pytest.mark.parametrize("bad_cost", [float('nan'), float('inf'), -1., 'not-a-cost'])
def test_invalid_actual_cost_does_not_partially_assimilate(bad_cost):
    binding = _binding()
    state = _state(binding)
    before = binding.belief
    with pytest.raises(ValueError):
        apply(state, predictive_action(binding, 0),
              {"evidence_id": "new-reading", "outcome": 1, "cost": bad_cost})
    assert binding.belief is before


def test_changed_calibration_or_instrument_cannot_relabel_a_reading():
    channel = ObservationChannel("calibrated", KEY, [[.85, .15], [.2, .8]], cost=2.5)
    binding = _binding(channel=channel, task_id="cal")
    state = _state(binding)
    instrument = _SignInstrument([Reading(+1, .9, 2.5)])
    calibration = Calibration("v1", channel, lambda reading: 0)
    action = calibrated_action(state, "cal", instrument, calibration)
    wrong = Calibration("v2", channel, lambda reading: 1)
    outcome = calibrated_outcome(instrument, wrong, ("a", "b"), .4, "new-reading")
    before = binding.belief
    with pytest.raises(ValueError, match="calibration"):
        apply(state, action, outcome)
    assert binding.belief is before


def test_names_indices_and_metadata_cannot_change_the_planned_contract():
    first = _binding(task_id="a/b", channel=ObservationChannel("c", KEY, [[.9, .1], [.1, .9]]))
    second = _binding(task_id="a", channel=ObservationChannel("b/c", KEY, [[.9, .1], [.1, .9]]))
    assert predictive_action(first, 0).id != predictive_action(second, 0).id
    with pytest.raises(ValueError, match="integer"):
        predictive_action(first, .5)
    with pytest.raises(ValueError, match="extra metadata"):
        predictive_action(first, 0, meta_extra={"risk_law": [(1., 100.)]})
    action = predictive_action(first, 0)
    first.cost_unit = "joules"
    with pytest.raises(ValueError, match="cost unit"):
        apply(_state(first), action, {"evidence_id": "fresh", "outcome": 0})
    with pytest.raises(ValueError, match="unique"):
        next_actions(_state(_binding(), _binding()))


def test_zero_probability_outcome_is_rejected_atomically():
    channel = ObservationChannel("impossible-positive", KEY, [[1., 0.], [1., 0.]])
    binding = _binding(channel=channel)
    before = binding.belief
    with pytest.raises(ValueError):
        apply(_state(binding), predictive_action(binding, 0), {"evidence_id": "bad", "outcome": 1})
    assert binding.belief is before
