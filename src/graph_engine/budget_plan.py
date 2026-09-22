#!/usr/bin/env python3
"""
budget_plan.py — finite-budget Bellman over caller-supplied typed transitions and a task loss.

ANCESTRY (no novel Bellman algorithm is claimed; Bellman 1957 is standard). `plan_value.settle_cost`
motivates this finite-budget extension: its instruments are typed `observe` actions,
its "settled" event is an absorbing successor with zero terminal task loss. Its requirement to settle is stronger
than the optional finite-penalty stopping contract here; it searches for the
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
action, so bottom-up evaluation terminates; there are no real-valued costs, so nothing is rounded silently.
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
(exact floating-point tie), the planner STOPS. Among actions, the winner minimizes (value, cost, name). This is
documented, reproducible behaviour, not an arbitrary `min` over a set.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Hashable, Sequence

__all__ = ["Action", "PlanResult", "plan", "freeze_state", "ACTION_KINDS"]

ACTION_KINDS = ("observe", "compute", "train")


@dataclass(frozen=True)
class Action:
    """A typed caller-supplied transition.

    `outcomes` is a tuple of (probability, successor_state) pairs; probabilities are finite,
    nonnegative and sum to one (atol 1e-9; roundoff within this tolerance is normalized), and every successor state is hashable. `cost` is a
    strictly positive integer in budget units. Zero-probability outcomes are permitted and are
    PRUNED: their successor state is never evaluated by the planner.
    """

    name: str
    kind: str
    cost: int
    outcomes: tuple[tuple[float, Hashable], ...]
    provenance: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or self.name == "stop":
            raise ValueError("action needs a nonempty string name other than reserved stop")
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unknown action kind {self.kind!r}; allowed {ACTION_KINDS}")
        if isinstance(self.cost, bool) or not isinstance(self.cost, int):
            raise ValueError("action cost must be an integer number of budget units")
        if self.cost <= 0:
            raise ValueError("action cost must be strictly positive (positive costs guarantee termination)")
        outcomes = tuple(self.outcomes)
        if not outcomes:
            raise ValueError("action needs at least one outcome")
        probabilities = []
        for outcome in outcomes:
            if not (isinstance(outcome, tuple) and len(outcome) == 2):
                raise ValueError("each outcome must be a (probability, successor_state) pair")
            probability, successor = outcome
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                raise ValueError("outcome probability must be a real number")
            probability = float(probability)
            if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
                raise ValueError("outcome probability must be finite and in [0, 1]")
            _validate_state(successor)
            probabilities.append(probability)
        total = math.fsum(probabilities)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"outcome probabilities must sum to one (got {total!r})")
        object.__setattr__(self, "outcomes", tuple((p/total, outcome[1]) for p, outcome in zip(probabilities, outcomes)))
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
    loss_cache: dict[Hashable, float] = {}

    def actions_of(state):
        if state not in action_cache:
            supplied = tuple(actions_fn(state))
            if any(not isinstance(a, Action) for a in supplied):
                raise ValueError("actions_fn must return Action instances")
            if len({a.name for a in supplied}) != len(supplied):
                raise ValueError("action names must be unique within each state")
            action_cache[state] = supplied
        return action_cache[state]

    def stop_of(state):
        if state not in loss_cache:
            value = float(terminal_loss(state))
            if not math.isfinite(value) or value < 0:
                raise ValueError("terminal task loss must be nonnegative and finite")
            loss_cache[state] = value
        return loss_cache[state]

    # Discover only positive-probability, affordable successors. The resource
    # coordinate strictly decreases, so it supplies an ordering without a
    # recursion-depth limit, including models with cycles in physical state.
    root = (initial_state, budget)
    domain, pending = set(), [root]
    while pending:
        key = pending.pop()
        if key in domain:
            continue
        domain.add(key)
        state, remaining = key
        stop_of(state)
        if remaining == 0:
            continue
        for action in actions_of(state):
            if action.cost <= remaining:
                pending.extend((successor, remaining-action.cost)
                               for p, successor in action.outcomes if p > 0)

    # Moments are memoized as well as values. Walking every outcome path to
    # account for cost would be exponential even when paths merge into one state.
    moments, values, choices = {}, {}, {}
    for state, remaining in sorted(domain, key=lambda key: key[1]):
        key = (state, remaining)
        best = stop_of(state)
        chosen_action = None
        best_rank = None
        best_moments = (best, 0.0, (0.0,)*len(ACTION_KINDS))
        for action in actions_of(state) if remaining else ():
            if action.cost > remaining:
                continue
            children = [(p, moments[(s, remaining-action.cost)])
                        for p, s in action.outcomes if p > 0]
            terminal = math.fsum(p*m[0] for p, m in children)
            resource = action.cost + math.fsum(p*m[1] for p, m in children)
            by_kind = tuple((action.cost if kind == action.kind else 0.0)
                            + math.fsum(p*m[2][i] for p, m in children)
                            for i, kind in enumerate(ACTION_KINDS))
            total = terminal + price*resource
            rank = (total, action.cost, action.name)
            # Stop wins exact ties; among improving actions use cost then name.
            if total < best or (total == best and chosen_action is not None and rank < best_rank):
                best, best_rank, chosen_action = total, rank, action
                best_moments = (terminal, resource, by_kind)
        values[key], moments[key], choices[key] = best, best_moments, chosen_action

    policy, pending = {}, [root]
    while pending:
        key = pending.pop()
        if key in policy:
            continue
        state, remaining = key
        action = choices[key]
        policy[key] = action.name if action is not None else "stop"
        if action is not None:
            pending.extend((s, remaining-action.cost) for p, s in action.outcomes if p > 0)
    terminal, resource, by_kind = moments[root]
    return PlanResult(
        chosen=policy[root], objective=values[root],
        expected_terminal_loss=terminal, expected_resource_cost=resource,
        expected_cost_by_kind=tuple(zip(ACTION_KINDS, by_kind)),
        policy=tuple(sorted(((s, b, name) for (s, b), name in policy.items()),
                            key=lambda item: (str(item[0]), item[1]))),
        n_decisions=len(policy), price=price,
    )
