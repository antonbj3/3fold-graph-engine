# budget_plan — a finite-budget extension of `plan_value`

`src/graph_engine/budget_plan.py` implements one small generic finite-budget Bellman plan over
caller-supplied typed transitions and a caller-supplied terminal task loss. **No novel Bellman
algorithm is claimed.** The dynamic programme is the standard one (Bellman 1957); what is new here is
only the contract that lets the existing engine value *compound* computations — including training —
without pretending training is a Bayesian observation update.

## The contract

A caller supplies

* an **immutable/hashable state** (`freeze_state` snapshots a flat dict). The state carries whatever
  the caller wants: belief fingerprint, model version, cache state, remaining demand.
* `actions_fn(state)` → a finite tuple of `Action(name, kind, cost, outcomes)` with
  `kind ∈ {observe, compute, train}`, `cost` a **strictly positive integer** in budget units, and
  `outcomes` a tuple of `(probability, successor_state)` summing to one (atol 1e-9).
* `terminal_loss(state)` → nonnegative finite expected task loss.
* a hard integer `budget` and an optional nonnegative `price` converting budget units to loss units.

The choice at `(state, b)` is

```
V(s, b) = min( terminal_loss(s),
               min over actions a with a.cost <= b of
                   price * a.cost + SUM_o p_o * V(s_o, b - a.cost) )
```

Stop is always allowed (the first term). The returned `PlanResult` carries `chosen`, `objective`
(= `V`), `expected_terminal_loss`, `expected_resource_cost`, `expected_cost_by_kind`, the reachable
`policy`, and `n_decisions`. `objective == expected_terminal_loss + price * expected_resource_cost`
by construction.

## What it is exact for, and what it is not

Exact only for the **supplied finite transition model**: positive integer costs make the remaining
budget strictly decrease, so the recursion terminates; the hard budget applies to **every branch**
(an action is considered only when its full cost fits the branch's remainder, never on average).
`price = 0` prices budget as free. It is **not** global-optimal reality, and a `train` action's
success/failure probabilities are **caller-supplied uncertainty** in a constructed model — never a
measured neural training result and never a likelihood the planner derives. Zero-probability outcomes
are pruned, so impossible successor states are never evaluated. Deterministic tie behaviour: when the
best action is not strictly better than stopping (on an exact value tie) the planner stops; among actions the
winner minimizes `(value, cost, name)`.

## Ancestry in this engine

* **`plan_value.settle_cost` is the unbounded-budget special case.** Its instruments are `observe`
  actions, its "settled" event is an absorbing state with zero terminal loss, and it minimizes
  expected spend `S(p) = min_i [c_i + Σ_o q_o S(p_o)]`. `budget_plan` generalizes the objective to
  `terminal task loss + price × resource spend` under a hard budget. `tests/test_budget_plan.py`
  maps a two-instrument cascade to `observe` actions and recovers `settle_cost` exactly
  (abs error 3.3e-11); `plan_value.py` itself is **not modified**.
* **`next_actions.bundle_value_bits` / joint bundles.** A purchase with increasing returns (two
  probes worth ≈ 0 alone, a lot together) is representable here as a two-step sequence of typed
  actions or as one `observe` action whose outcomes are the caller's explicit joint distribution.
  The planner never assumes independence: the XOR case shows that supplying the explicit joint
  solves the task (risk 0) while an independence-assumed product does not (risk 0.904).
* **`identifiability_oed` / `predictive_state`.** Relevance is the expected task-risk drop under the
  supplied observation channels (`ObservationChannel` / `PredictiveTask`), not an information proxy.
  `predictive_state.value_of_observation` is the one-step special case; `budget_plan` adds the
  budget and the multi-step composition. Planning uses hypothetical posteriors without consuming an
  evidence identity, matching `value_of_observation`'s own convention.

## Tests

`tests/test_budget_plan.py` (13 tests) checks malformed-input rejection, zero-probability pruning,
the accounting identity and price units, budget monotonicity and per-branch hard budgets, copied
branches / correlation, the three constructed cases, and two independent checks: an exhaustively
enumerated **policy-tree** oracle on 120 random finite cases (max abs error 2.2e-16) and the
`settle_cost` special case. The existing `tests/test_plan_value.py` is untouched and still passes.

## Root integration review

Call `plan_value.budgeted_plan` from the existing planning API, with transition
types from `budget_plan`. Root review replaced recursive path accounting with
bottom-up values and expected moments: merged branches are evaluated once per
(state,budget), including a tested1100-step case. Stop wins exact value ties;
small real improvements are reflected in both the selected policy and reported
objective. Accepted probability roundoff is normalized, and action names are
unique within a state; `stop` is reserved. A supplied finite stopping penalty is
not identical to the legacy requirement to eventually settle a node.
