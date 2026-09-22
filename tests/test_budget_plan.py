"""budget_plan: finite-budget Bellman checked against an INDEPENDENT exhaustive policy-tree
enumerator and against plan_value.settle_cost (the unbounded-budget special case); plus the
three constructed discriminating cases (XOR complementarity, training amortization, ambiguous
global branches) built on the existing predictive_state ObservationChannel / PredictiveTask."""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from graph_engine.budget_plan import Action, plan, freeze_state  # noqa: E402
from graph_engine.plan_value import settle_cost  # noqa: E402
from graph_engine.predictive_state import (  # noqa: E402
    BeliefState, ObservationChannel, PredictiveTask,
)


def test_tiny_improvement_uses_same_policy_as_reported_objective():
    from graph_engine.plan_value import budgeted_plan
    actions = lambda s: [Action("improve", "compute", 1, [(1., "done")])] if s == "start" else []
    loss = lambda s: 5e-13 if s == "start" else 0.
    result = budgeted_plan("start", actions, loss, 1, price=0.)
    assert result.chosen == "improve"
    assert result.objective == result.expected_terminal_loss == 0.


def test_deep_merged_branches_are_evaluated_once_per_state():
    # Recursive policy walking would visit 2**1100 paths; recursive value
    # evaluation would also hit Python's stack limit despite positive costs.
    actions = lambda s: [Action("step", "compute", 1, [(.5, s+1), (.5, s+1)])]
    result = plan(0, actions, lambda s: max(1100-s, 0), 1100, price=0.)
    assert result.expected_terminal_loss == 0.
    assert result.expected_resource_cost == 1100.
    assert result.n_decisions == 1101


def test_roundoff_probability_mass_is_normalized_and_names_unambiguous():
    action = Action("a", "compute", 1, [(0.5-1e-10, "t"), (.5, "t")])
    assert math.fsum(p for p, _ in action.outcomes) == 1.
    with pytest.raises(ValueError, match="reserved"):
        Action("stop", "compute", 1, [(1., "t")])
    with pytest.raises(ValueError, match="unique"):
        plan("s", lambda s: [action, action], lambda s: 2., 1)


# ---------------------------------------------------------------- independent enumerator
def _brute_force_policy_tree(initial, actions_fn, terminal_loss, budget, price):
    """INDEPENDENT of the module's recursion: discover the (state,budget) domain under ANY action,
    enumerate every deterministic assignment of {stop} U affordable-actions to every domain node,
    evaluate each full policy tree exactly, return the minimum objective. Exponential; tiny cases.
    """
    cache = {}

    def acts(state):
        if state not in cache:
            cache[state] = tuple(actions_fn(state))
        return cache[state]

    domain, stack = set(), [(initial, budget)]
    while stack:
        state, remaining = stack.pop()
        if (state, remaining) in domain:
            continue
        domain.add((state, remaining))
        for action in acts(state):
            if action.cost <= remaining:
                for probability, successor in action.outcomes:
                    if probability > 0 and (successor, remaining - action.cost) not in domain:
                        stack.append((successor, remaining - action.cost))
    nodes = sorted(domain, key=lambda t: (str(t[0]), t[1]))
    options = [[None] + [i for i, a in enumerate(acts(s)) if a.cost <= b] for s, b in nodes]
    best = math.inf
    for combo in itertools.product(*options):
        policy = dict(zip(nodes, combo))

        def evaluate(state, remaining):
            choice = policy[(state, remaining)]
            if choice is None:
                return terminal_loss(state), 0.0
            action = acts(state)[choice]
            terminal, resource = 0.0, float(action.cost)
            for probability, successor in action.outcomes:
                if probability <= 0:
                    continue
                t, r = evaluate(successor, remaining - action.cost)
                terminal += probability * t
                resource += probability * r
            return terminal, resource

        t, r = evaluate(initial, budget)
        best = min(best, t + price * r)
    return best


# ---------------------------------------------------------------- predictive_state helpers
def _hypothetical(state, channel, outcome):
    """Posterior for a PLANNED outcome without consuming an evidence identity.
    Mirrors predictive_state.value_of_observation's own hypothetical update."""
    likelihood = channel.likelihood[:, outcome]
    loglike = np.full_like(likelihood, -np.inf)
    np.log(likelihood, out=loglike, where=likelihood > 0)
    return BeliefState.from_log_mass(state.space_key, state._log_mass + loglike, state.evidence_ids)


def _belief_planner(root, task, specs, budget, price, stake=1.0, extra_actions=lambda state: ()):
    """specs: (name, kind, channel, cost). Planner state is the belief fingerprint (content key)."""
    beliefs = {root.fingerprint: root}

    def register(state):
        beliefs.setdefault(state.fingerprint, state)

    def actions_fn(fingerprint):
        state = beliefs[fingerprint]
        result = []
        for name, kind, channel, cost in specs:
            outcomes = []
            for outcome, probability in enumerate(state.predict(channel)):
                if probability <= 0:
                    outcomes.append((0.0, fingerprint))
                    continue
                successor = _hypothetical(state, channel, outcome)
                register(successor)
                outcomes.append((float(probability), successor.fingerprint))
            result.append(Action(name, kind, cost, tuple(outcomes)))
        result.extend(extra_actions(state))
        return result

    def terminal_loss(fingerprint):
        return stake * task.risk(beliefs[fingerprint])

    result = plan(root.fingerprint, actions_fn, terminal_loss, budget, price)
    return result, beliefs


def _binary_task(key, target, loss="log"):
    y = np.asarray(target, dtype=float)
    return PredictiveTask(key, np.stack([1 - y, y], axis=-1)[:, None, :], [1.0], loss)


# ---------------------------------------------------------------- malformed inputs
def test_malformed_inputs_are_rejected_not_repaired():
    with pytest.raises(ValueError):
        Action("a", "observe", 0, [(1.0, "s")])                 # non-positive cost
    with pytest.raises(ValueError):
        Action("a", "observe", 1.5, [(1.0, "s")])               # non-integer cost
    with pytest.raises(ValueError):
        Action("a", "observe", 1, [(0.5, "s")])                 # probabilities do not sum to one
    with pytest.raises(ValueError):
        Action("a", "observe", 1, [(float("nan"), "s")])
    with pytest.raises(ValueError):
        Action("a", "fly", 1, [(1.0, "s")])                     # unknown kind
    with pytest.raises(ValueError):
        Action("a", "observe", 1, [])                           # no outcomes
    ok = lambda s: [Action("a", "compute", 1, [(1.0, s)])]      # noqa: E731
    with pytest.raises(ValueError):
        plan("s", ok, lambda s: -1.0, 3)                        # negative terminal loss
    with pytest.raises(ValueError):
        plan("s", ok, lambda s: float("inf"), 3)                # non-finite terminal loss
    with pytest.raises(ValueError):
        plan("s", ok, lambda s: 0.0, -1)                        # negative budget
    with pytest.raises(ValueError):
        plan("s", ok, lambda s: 0.0, 3, price=-1.0)             # negative price
    with pytest.raises(ValueError):
        plan(["not", "hashable"], ok, lambda s: 0.0, 3)         # unhashable state
    with pytest.raises(ValueError):
        plan("s", lambda s: ["not-an-action"], lambda s: 0.0, 3)


def test_zero_probability_branch_is_never_evaluated():
    def terminal_loss(state):
        if state == "IMPOSSIBLE":
            raise AssertionError("planner evaluated an impossible (p=0) successor state")
        return 0.0 if state == "reachable" else 5.0

    actions = lambda s: [Action("act", "compute", 1, [(0.0, "IMPOSSIBLE"), (1.0, "reachable")])]  # noqa: E731
    result = plan("start", actions, terminal_loss, budget=1, price=1.0)
    assert result.chosen == "act" and result.objective == pytest.approx(1.0)
    assert result.expected_resource_cost == pytest.approx(1.0)


# ---------------------------------------------------------------- core invariants
def test_accounting_identity_and_price_units():
    actions = lambda s: ([Action("work", "compute", 2, [(1.0, "done")])] if s != "done" else [])  # noqa: E731
    loss = lambda s: 0.0 if s == "done" else 7.0  # noqa: E731
    for price in (0.0, 0.5, 1.0, 3.0):
        res = plan("start", actions, loss, budget=2, price=price)
        assert res.objective == pytest.approx(res.expected_terminal_loss + price * res.expected_resource_cost)
    assert plan("start", actions, loss, 2, 0.0).objective == pytest.approx(0.0)   # free budget: work
    assert plan("start", actions, loss, 2, 10.0).objective == pytest.approx(7.0)  # dear budget: stop
    assert plan("start", actions, loss, 2, 10.0).chosen == "stop"


def test_budget_monotonicity_and_hard_branch_budget():
    # two steps needed; at budget 1 the second is unaffordable on EVERY branch, so step one is not taken.
    actions = lambda s: ([Action("step1", "compute", 1, [(1.0, "mid")])] if s == "start" else  # noqa: E731
                         ([Action("step2", "compute", 1, [(1.0, "done")])] if s == "mid" else []))
    loss = lambda s: 0.0 if s == "done" else 5.0  # noqa: E731
    values = [plan("start", actions, loss, b, 1.0).objective for b in range(5)]
    assert all(values[i] >= values[i + 1] - 1e-12 for i in range(len(values) - 1))   # nonincreasing
    assert plan("start", actions, loss, 1, 1.0).chosen == "stop"
    assert plan("start", actions, loss, 2, 1.0).chosen == "step1"
    assert plan("start", actions, loss, 2, 1.0).objective == pytest.approx(2.0)
    # a branch whose successor could not afford step2 would be charged only step1: check the
    # objective sequence exactly (stop 5, b1: 1+5=6? no: b1 step1->mid budget0 -> stop 5 => 6>5 stop).
    assert values == pytest.approx([5.0, 5.0, 2.0, 2.0, 2.0])


def test_copied_branches_are_memoised_and_correlation_must_be_supplied_jointly():
    # Two COPIED (perfectly correlated) sensors share one noise event. The explicit joint channel
    # carries only one sensor's worth of information; an independence-assumed product would wrongly
    # multiply the evidence. The planner consumes the supplied joint, never an assumed independence.
    root = BeliefState("bit", [0.5, 0.5])
    task = _binary_task("bit", [0, 1])
    # joint outcome = the shared readout (0/1); both sensors agree always.
    correlated = ObservationChannel("copied-pair", "bit", [[0.7, 0.3], [0.3, 0.7]], cost=2)
    from graph_engine.predictive_state import value_of_observation
    explicit = value_of_observation(root, task, correlated)            # one event, one likelihood
    risk_correlated = explicit.expected_risk_after
    # independent two-vote model: odds squared -> overconfident.
    p = 0.7
    posterior_independent = p * p / (p * p + (1 - p) ** 2)
    risk_independent = float(
        -(posterior_independent * np.log2(posterior_independent)
          + (1 - posterior_independent) * np.log2(1 - posterior_independent)))
    assert risk_correlated > risk_independent + 0.1                     # correlation reduces info
    plan_corr, _ = _belief_planner(root, task, [("pair", "observe", correlated, 2)],
                                   budget=2, price=0.0, stake=10.0)
    assert plan_corr.expected_terminal_loss == pytest.approx(10.0 * risk_correlated)
    assert plan_corr.expected_terminal_loss != pytest.approx(10.0 * risk_independent)
    # Two actions reaching the SAME successor are memoised, not double-evaluated.
    same = lambda s: [Action("x", "compute", 1, [(1.0, "t")]), Action("y", "compute", 1, [(1.0, "t")])]  # noqa: E731
    res = plan("s", same, lambda s: 0.0 if s == "t" else 4.0, budget=1, price=1.0)
    assert res.chosen == "x" and res.expected_resource_cost == pytest.approx(1.0)


def test_independent_policy_tree_enumerator_on_random_finite_cases():
    rng = np.random.default_rng(20260922)
    state_count = 3
    for _ in range(60):
        budget = int(rng.integers(1, 4))
        price = float(rng.choice([0.0, 0.5, 1.0, 2.0]))
        action_count = int(rng.integers(1, 3))
        n_outcomes = int(rng.integers(1, 3))
        loss = rng.uniform(0.0, 2.0, size=state_count)
        built = []
        for ai in range(action_count):
            cost = int(rng.integers(1, budget + 1))
            probs = rng.uniform(0.05, 1.0, size=n_outcomes)
            probs = probs / probs.sum()
            successors = rng.integers(0, state_count, size=n_outcomes)
            built.append(Action(f"a{ai}", "compute", cost,
                                tuple((float(p), int(s)) for p, s in zip(probs, successors))))
        actions_fn = lambda s, built=built: tuple(built)  # noqa: E731
        loss_fn = lambda s, loss=loss: float(loss[int(s)])  # noqa: E731
        got = plan(0, actions_fn, loss_fn, budget, price)
        want = _brute_force_policy_tree(0, actions_fn, loss_fn, budget, price)
        assert got.objective == pytest.approx(want, abs=1e-9)
        assert got.objective == pytest.approx(got.expected_terminal_loss + price * got.expected_resource_cost)


# ---------------------------------------------------------------- CASE 1: complementarity
def test_case1_xor_specialists_compose_and_myopic_selector_fails():
    root = BeliefState("xor", [0.25] * 4)
    task = _binary_task("xor", [0, 1, 1, 0])
    a = ObservationChannel("a", "xor", [[1, 0], [1, 0], [0, 1], [0, 1]])
    b = ObservationChannel("b", "xor", [[1, 0], [0, 1], [1, 0], [0, 1]])
    stake = 10.0
    # single-action value is exactly zero for each specialist.
    from graph_engine.predictive_state import value_of_observation
    assert value_of_observation(root, task, a).gain == 0
    assert value_of_observation(root, task, b).gain == 0
    # myopic selector: gain - price*cost < 0 for both -> stops.
    myopic = {ch.name: stake * value_of_observation(root, task, ch).gain - 1.0 for ch in (a, b)}
    assert all(v < 0 for v in myopic.values())
    # two-step planner composes.
    res, _ = _belief_planner(root, task, [("a", "observe", a, 1), ("b", "observe", b, 1)],
                             budget=2, price=1.0, stake=stake)
    assert res.chosen in {"a", "b"} and res.expected_terminal_loss == pytest.approx(0.0)
    assert res.objective == pytest.approx(2.0)


# ---------------------------------------------------------------- CASE 2: training amortization
def _training_actions(cost_train=10, p_success=0.9, cost_direct=4, cost_cheap=1):
    """Constructed transitions (NOT measured neural training): state = (model_version, remaining)."""
    def actions_fn(state):
        version, remaining = state
        if remaining == 0:
            return []
        acts = [Action("direct_infer", "compute", cost_direct, [(1.0, (version, remaining - 1))])]
        if version == "v2":
            acts.append(Action("cheap_infer", "compute", cost_cheap, [(1.0, ("v2", remaining - 1))]))
        else:
            acts.append(Action("train", "train", cost_train,
                               [(p_success, ("v2", remaining)), (1.0 - p_success, ("v1", remaining))]))
        return acts

    def terminal_loss(state):
        return float(state[1]) * 5.0        # unserved task risk, loss units

    return actions_fn, terminal_loss


def test_case2_training_amortization_short_vs_long_horizon_and_closed_form():
    actions_fn, terminal_loss = _training_actions()
    price, budget = 1.0, 400
    short = plan(("v1", 1), actions_fn, terminal_loss, budget, price)
    long = plan(("v1", 8), actions_fn, terminal_loss, budget, price)
    assert short.chosen == "direct_infer"                       # short horizon: direct inference wins
    assert long.chosen == "train"                               # long horizon: train first
    # closed-form independent candidate enumeration (geometric train retries, then cheap inference).
    # The planner is finite-budget; budget 400 truncates retry chains beyond ~0.1**40.
    p = 0.9
    train_expected = 10.0 / p                                    # expected spend to reach v2
    for horizon in (1, 2, 4, 8, 12):
        direct = horizon * 4.0
        trained = train_expected + horizon * 1.0
        expected = min(direct, trained)
        got = plan(("v1", horizon), actions_fn, terminal_loss, budget, price)
        assert got.objective == pytest.approx(expected, abs=1e-6)
    assert short.objective == pytest.approx(4.0)
    assert long.objective == pytest.approx(train_expected + 8.0, abs=1e-6)
    kinds = dict(long.expected_cost_by_kind)
    assert kinds["train"] > 0 and kinds["compute"] > 0
    # training-failure uncertainty is exercised: at least one reachable branch stays on v1.
    assert any(state[0] == "v1" for state, _b, _a in long.policy)


# ---------------------------------------------------------------- CASE 3: ambiguous branches
def test_case3_ambiguous_branches_refinement_and_self_distillation_get_no_credit():
    root = BeliefState("mirror", [0.5, 0.5])
    task = _binary_task("mirror", [0, 1])
    identical = ObservationChannel("lengths", "mirror", [[0, 1], [0, 1]])    # identical readouts
    self_distill = ObservationChannel("self-teacher", "mirror", [[0, 1], [0, 1]])
    chirality = ObservationChannel("chirality", "mirror", np.eye(2), cost=3)
    stake = 10.0
    # refinement / self-distillation cannot distinguish the branches: posterior fingerprint unchanged.
    for channel in (identical, self_distill):
        for outcome, probability in enumerate(root.predict(channel)):
            if probability > 0:
                assert _hypothetical(root, channel, outcome).fingerprint == root.fingerprint
    # (a) informative calibration is affordable -> planner buys it and resolves the branch.
    cheap_anchor = ObservationChannel("calibrate-cheap", "mirror", np.eye(2), cost=1)
    res, _ = _belief_planner(root, task,
                             [("refine", "compute", identical, 1),
                              ("distill", "train", self_distill, 1),
                              ("calibrate", "observe", cheap_anchor, 1)],
                             budget=1, price=1.0, stake=stake)
    assert res.chosen == "calibrate"
    assert res.expected_terminal_loss == pytest.approx(0.0)
    # (b) NEGATIVE: anchor too expensive (cost 15 > stake*1bit) -> stop is right.
    dear_anchor = ObservationChannel("calibrate-dear", "mirror", np.eye(2), cost=15)
    res_stop, _ = _belief_planner(root, task,
                                  [("refine", "compute", identical, 1),
                                   ("distill", "train", self_distill, 1),
                                   ("calibrate", "observe", dear_anchor, 15)],
                                  budget=15, price=1.0, stake=stake)
    assert res_stop.chosen == "stop"
    assert res_stop.objective == pytest.approx(10.0)            # stake * 1 bit


def test_exact_posterior_outcomes_from_predictive_state_match_value_of_observation():
    root = BeliefState("mirror", [0.5, 0.5])
    task = _binary_task("mirror", [0, 1])
    chirality = ObservationChannel("chirality", "mirror", np.eye(2))
    from graph_engine.predictive_state import value_of_observation
    reference = value_of_observation(root, task, chirality).expected_risk_after
    outcomes = root.predict(chirality)
    manual = 0.0
    for outcome, probability in enumerate(outcomes):
        if probability > 0:
            manual += float(probability) * task.risk(_hypothetical(root, chirality, outcome))
    assert manual == pytest.approx(reference, abs=1e-12)


# ---------------------------------------------------------------- ancestry
def test_settle_cost_is_the_unbounded_budget_special_case():
    # plan_value.settle_cost, single perfect instrument: settles in one step at cost c.
    p, tau, cost = 0.5, 0.9, 20
    s_cost, first = settle_cost(p, [(float(cost), 1.0, "cell")], tau)
    assert (s_cost, first) == (float(cost), "cell")
    actions = lambda s: ([Action("cell", "observe", cost, [(1.0, "settled")])] if s == "open" else [])  # noqa: E731
    loss = lambda s: 0.0 if s == "settled" else 1e6  # noqa: E731
    res = plan("open", actions, loss, budget=cost, price=1.0)
    assert res.expected_resource_cost == pytest.approx(s_cost)
    assert res.expected_terminal_loss == pytest.approx(0.0)


def test_settle_cost_two_instrument_cascade_matches_planner_expected_spend():
    # The expensive-only vs cheap-then-expensive cascade of plan_value, mapped to observe actions.
    p, tau = 0.2, 0.9
    instruments = [(1.0, 0.95, "judge"), (20.0, 1.0, "cell")]
    expected_spend, _ = settle_cost(p, instruments, tau)

    def posterior(prob, r, yes):
        q = prob * r + (1 - prob) * (1 - r)
        return (prob * r / q) if yes else (prob * (1 - r) / (1 - q))

    def actions_fn(state):
        if state >= tau or state <= 1 - tau:
            return []
        out = []
        for cost, r, name in instruments:
            q = state * r + (1 - state) * (1 - r)
            outcomes = ((q, round(posterior(state, r, True), 12)),
                        (1 - q, round(posterior(state, r, False), 12)))
            out.append(Action(name, "observe", int(cost), outcomes))
        return out

    loss = lambda s: 0.0 if (s >= tau or s <= 1 - tau) else 1e6  # noqa: E731
    res = plan(round(p, 12), actions_fn, loss, budget=200, price=1.0)
    assert res.expected_resource_cost == pytest.approx(expected_spend, abs=1e-4)


def test_freeze_state_is_a_hashable_snapshot():
    key = freeze_state({"model": "v2", "remaining": 3})
    assert isinstance(key, tuple) and hash(key)
    assert key == (("model", "v2"), ("remaining", 3))
