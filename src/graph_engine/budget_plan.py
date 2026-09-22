#!/usr/bin/env python3
"""
budget_plan.py — finite-budget Bellman over caller-supplied typed transitions and a task loss.

ANCESTRY (no novel Bellman algorithm is claimed; Bellman 1957 is standard). `plan_value.settle_cost`
is the unbounded-budget special case of this module: its instruments are typed `observe` actions,
its "settled" event is an absorbing successor with zero terminal task loss, and it searches for the
smallest expected spend while this module searches for the smallest expected (task loss + price x
resource spend) under a caller-supplied HARD budget. `unlock_value.rank` / `plan_value.plan` order
nodes; this module plans the inside of ONE decision problem. `next_actions.bundle_value_bits`
prices a joint purchase (e.g. two probes whose single value is zero); the same complementarity is
representable here as a two-step sequence of typed actions. `identifiability_oed` and
`predictive_state` supply the observation channels whose expected task-risk drop defines relevance.

WHAT THIS MODULE IS EXACT FOR. Only the finite transition model the caller supplies:
    * an immutable/hashable state (the planner memoises on it),
    * a finite set of available typed actions per state (`observe`/`compute`/`train`),
      each with a strictly positive INTEGER budget cost and a normalized finite distribution
      over successor states,
    * a terminal nonnegative finite expected task loss per state,
    * an optional nonnegative `price` converting budget units to loss units.
The Bellman choice at (state, remaining budget b) is
    V(s, b) = min( terminal_loss(s),
                   min over actions a with a.cost <= b of
                       price * a.cost + SUM_o p_o * V(s_o, b - a.cost) ) .
Stop is ALWAYS allowed (the first term). Positive integer costs make b strictly decrease on every
action, so the recursion terminates; there are no real-valued costs, so nothing is rounded silently.
The hard budget applies to EVERY branch: an action is only considered when its full cost fits in the
remaining budget of the branch it is taken on, never on average.

WHAT IT IS NOT. It is NOT global-optimal reality, a measured neural training result, or a statement
that training is a Bayesian observation update. A `train` action's success/failure probabilities are
CALLER-SUPPLIED uncertainty in a constructed finite model. Exact only for the model supplied.

CALLABLE CONTRACTS (mutable-callable / cache correctness). `actions_fn(state)` and
`terminal_loss(state)` MUST be pure deterministic functions of their arguments; the planner caches
them by state and assumes a later call cannot return different actions. Callers using a mutable
object as state must snapshot it into an immutable hashable key first (see `freeze_state`), because
the memo and the returned policy are keyed by that value. Actions returned for a state are used as
given; the planner does not mutate them.

DETERMINISTIC TIES. When the best action's value is not strictly less than `terminal_loss(state)`
(within 1e-12), the planner STOPS. Among actions, the winner minimizes (value, cost, name). This is
documented, reproducible behaviour, not an arbitrary `min` over a set.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Hashable, Sequence

__all__ = ["Action", "PlanResult", "plan", "freeze_state", "ACTION_KINDS"]

ACTION_KINDS = ("observe", "compute", "train")
_TIE = 1e-12


@dataclass(frozen=True)
class Action:
    """A typed caller-supplied transition.

    `outcomes` is a tuple of (probability, successor_state) pairs; probabilities are finite,
    nonnegative and sum to one (atol 1e-9), and every successor state is hashable. `cost` is a
    strictly positive integer in budget units. Zero-probability outcomes are permitted and are
    PRUNED: their successor state is never evaluated by the planner.
    """

    name: str
    kind: str
    cost: int
    outcomes: tuple[tuple[float, Hashable], ...]
    provenance: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("action needs a nonempty string name")
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unknown action kind {self.kind!r}; allowed {ACTION_KINDS}")
        if isinstance(self.cost, bool) or not isinstance(self.cost, int):
            raise ValueError("action cost must be an integer number of budget units")
        if self.cost <= 0:
            raise ValueError("action cost must be strictly positive (positive costs guarantee termination)")
        outcomes = tuple(self.outcomes)
        if not outcomes:
            raise ValueError("action needs at least one outcome")
        total = 0.0
        for outcome in outcomes:
            if not (isinstance(outcome, tuple) and len(outcome) == 2):
                raise ValueError("each outcome must be a (probability, successor_state) pair")
            probability, successor = outcome
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                raise ValueError("outcome probability must be a real number")
            probability = float(probability)
            if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
                raise ValueError("outcome probability must be finite and in [0, 1]")
            hash(successor)
            total += probability
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"outcome probabilities must sum to one (got {total!r})")
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "provenance", tuple(self.provenance))


@dataclass(frozen=True)
class PlanResult:
    """Result of a finite-budget plan. All money-like fields are in the caller's declared units.

    objective == expected_terminal_loss + price * expected_resource_cost, by construction.
    expected_resource_cost is the expected SUM over the policy of budget units spent. Runtime of the
    planner itself is NOT here; measure it in the caller (demonstrations report it separately).
    """

    chosen: str                                   # selected action name at the root, or "stop"
    objective: float                              # V(root, budget), loss units
    expected_terminal_loss: float
    expected_resource_cost: float                 # budget units
    expected_cost_by_kind: tuple[tuple[str, float], ...]
    policy: tuple[tuple[Any, int, str], ...]      # ((state, budget), action-or-stop) reachable
    n_decisions: int                              # distinct (state, budget) pairs reached
    price: float


def freeze_state(mapping: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Snapshot a flat dict into an immutable, hashable state key (values must themselves be hashable)."""
    return tuple(sorted(mapping.items()))


def _validate_state(state: Any) -> None:
    try:
        hash(state)
    except TypeError as exc:
        raise ValueError(f"state must be hashable/immutable for memoisation: {exc}") from None


def plan(
    initial_state: Hashable,
    actions_fn: Callable[[Hashable], Sequence[Action]],
    terminal_loss: Callable[[Hashable], float],
    budget: int,
    price: float = 1.0,
) -> PlanResult:
    """Exact finite-budget Bellman plan under a caller-supplied finite transition model.

    Raises ValueError on malformed inputs: non-integer/negative budget, non-finite/negative price,
    unhashable state, a non-`Action` returned by `actions_fn`, or a terminal loss that is negative
    or non-finite. Raises nothing on impossible (p=0) successor states because they are pruned.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(float(price)) or price < 0:
        raise ValueError("price must be a finite nonnegative number")
    price = float(price)
    _validate_state(initial_state)

    action_cache: dict[Hashable, tuple[Action, ...]] = {}

    def actions_of(state: Hashable) -> tuple[Action, ...]:
        cached = action_cache.get(state)
        if cached is None:
            supplied = tuple(actions_fn(state))
            for action in supplied:
                if not isinstance(action, Action):
                    raise ValueError(f"actions_fn must return Action instances, got {type(action).__name__}")
            action_cache[state] = supplied
            return supplied
        return cached

    def stop_of(state: Hashable) -> float:
        value = float(terminal_loss(state))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"terminal task loss must be nonnegative and finite (got {value!r})")
        return value

    memo: dict[tuple[Hashable, int], float] = {}

    def value(state: Hashable, remaining: int) -> float:
        key = (state, remaining)
        if key in memo:
            return memo[key]
        best = stop_of(state)
        for action in actions_of(state):
            if action.cost > remaining:
                continue
            total = price * action.cost
            for probability, successor in action.outcomes:
                if probability <= 0.0:            # prune impossible branches: never evaluate them
                    continue
                total += probability * value(successor, remaining - action.cost)
            if total < best:
                best = total
        memo[key] = best
        return best

    # value(root) fills the memo for every reachable (state, remaining) pair before decisions.
    objective = value(initial_state, budget)

    chosen_cache: dict[tuple[Hashable, int], Action | None] = {}

    def chosen(state: Hashable, remaining: int) -> Action | None:
        key = (state, remaining)
        if key in chosen_cache:
            return chosen_cache[key]
        candidate: Action | None = None
        candidate_rank: tuple[float, int, str] | None = None
        for action in actions_of(state):
            if action.cost > remaining:
                continue
            total = price * action.cost
            for probability, successor in action.outcomes:
                if probability <= 0.0:
                    continue
                total += probability * value(successor, remaining - action.cost)
            rank = (total, action.cost, action.name)
            if candidate_rank is None or rank < candidate_rank:
                candidate, candidate_rank = action, rank
        stop = stop_of(state)
        result = candidate if (candidate is not None and candidate_rank[0] < stop - _TIE) else None
        chosen_cache[key] = result
        return result

    policy: dict[tuple[Hashable, int], str] = {}
    kind_cost: dict[str, float] = {kind: 0.0 for kind in ACTION_KINDS}
    accounting = {"terminal": 0.0, "resource": 0.0}

    def walk(state: Hashable, remaining: int, probability: float) -> None:
        if probability <= 0.0:
            return
        key = (state, remaining)
        action = chosen(state, remaining)
        if action is None:
            policy[key] = "stop"
            accounting["terminal"] += probability * stop_of(state)
            return
        policy[key] = action.name
        accounting["resource"] += probability * action.cost
        kind_cost[action.kind] += probability * action.cost
        for outcome_probability, successor in action.outcomes:
            if outcome_probability <= 0.0:
                continue
            walk(successor, remaining - action.cost, probability * outcome_probability)

    walk(initial_state, budget, 1.0)

    root_action = chosen(initial_state, budget)
    return PlanResult(
        chosen=root_action.name if root_action is not None else "stop",
        objective=objective,
        expected_terminal_loss=accounting["terminal"],
        expected_resource_cost=accounting["resource"],
        expected_cost_by_kind=tuple((kind, kind_cost[kind]) for kind in ACTION_KINDS),
        policy=tuple(sorted(((k[0], k[1], v) for k, v in policy.items()), key=lambda t: (str(t[0]), t[1]))),
        n_decisions=len(policy),
        price=price,
    )
