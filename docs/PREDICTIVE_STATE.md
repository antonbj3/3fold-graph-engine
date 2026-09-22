# Predictive state and shared geometry

This experimental extension gives Graph Engine's explicit beliefs a common
observable interface for inference, experiment selection and model training.
It creates no graph store. A working hypothesis is that graph and model encode
the same underlying latent objects; correspondence is expressed through typed
predictions and their geometry, rather than assumed equality of raw vectors.

`predictive_state.py` supplies a finite Bayesian reference and a frozen bridge
to the existing `RegimePosterior`. `predictive_geometry.py` supplies selected
pair distances, relational distortion, Jensen–Shannon divergence and a local
Fisher pullback. These mathematical operations are established methods. This
implementation is infrastructure for testing combinations, not a claim that
Bayes updates, geometric distillation or Fisher information are new.

```python
from graph_engine.regime_posterior import RegimePosterior
from graph_engine.predictive_state import RegimeSnapshot, value_of_observation

regime = RegimePosterior(0, 1, n_grid=12, p_two=0.2)
regime.add_claim(0.1, 0.3, +1, reliability=0.8)
regime.add_claim(0.7, 0.9, -1, reliability=0.8)
view = RegimeSnapshot.from_regime(regime)
task = view.task(loss="log")
channel = view.probe(0.5, reliability=0.95, cost=2.0)
value = value_of_observation(view.belief, task, channel)
# Run the actual instrument before consuming an outcome (0=negative, 1=positive).
updated = view.belief.observe(channel, outcome=1, evidence_id="instrument/run17/reading1")
predictions = task.predictions(updated)
```

Observation likelihoods have shape `[hypothesis, outcome]`. A task readout has
shape `[hypothesis, query, outcome]`; its query weights express deployment
demand. Log loss uses bits, Brier loss uses the full multiclass squared-error
convention, and error loss uses zero-one classification risk. Experiment values
are expected reductions of the selected task risk, under the supplied model.
This is one-action planning, not a general multi-step optimal policy.

Each hypothesis space has a key defining its meaning and ordering. A changed
Regime partition changes that key. The bridge normalizes query demand to one:
multiply its log/error risk by domain length to recover the existing Regime
integrated potential. It preserves both one- and two-transition families.
The bridge is isolated around Regime's private snapshot API and covered by a
numerical parity test.

Evidence identities prevent consuming a known duplicate. They cannot discover
undeclared copies or source correlations. A channel with no `conditioned_on`
explicitly declares conditional independence from prior observations given the
hypotheses. For dependence, supply a joint channel or a conditional channel tied
to the input state's fingerprint. The same source may produce multiple genuine
observations. Generated teacher labels are training material, not new independent
measurements of the external world.

The state retains log probabilities so a temporarily tiny branch can recover
after contrary evidence. Exactly zero prior support stays excluded. An impossible
observation raises an error rather than fabricating a posterior. Predictions and
expected values use float64: this is a numerical reference, not exact symbolic
arithmetic. The existing Regime snapshot can only preserve the probabilities it
receives from the existing solver.

For a bank of node predictions `[node, query, outcome]`, select the pairs to
compare explicitly:

```python
from graph_engine.predictive_geometry import squared_pair_distances, pair_distortion

# graph_predictions and model_predictions use the same node/query/outcome meanings.
pairs = [[0, 1], [0, 2], [1, 2]]
d2 = squared_pair_distances(graph_predictions, query_weights, pairs, metric="fisher")
loss, residual = pair_distortion(model_predictions, graph_predictions,
                                query_weights, pairs, metric="hellinger")
```

Hellinger uses the standard squared convention `1-sum(sqrt(p*q))`. Fisher uses
the weighted intrinsic product of query-simplex distances, computed with a stable
half-angle identity. Their local squared-distance ratio is eight. Query-weighted
JS in bits has local scale `Fisher_squared/(8*ln(2))`; the optimal balanced GAN
objective is `-ln(4)+2*ln(2)*JS_bits`.

Pairs preserve relations but cannot alone fix semantic label permutations.
Training therefore also needs anchored node outcomes. The local Fisher tensor
can have null directions and does not certify global identifiability. A finite
query bank can miss distinguishing interventions. Smaller pair distortion is a
training signal to evaluate against held-out task loss, not proof of improvement.

## The return path through the existing action layer

`predictive_state` is wired into the existing `next_actions` / `apply` / `alarm`
loop rather than a parallel controller. A caller attaches one or more
`next_actions.PredictiveBinding(task_id, belief, task, channels, cost_unit)` to
`EngineState.predictive_bindings`. Collection prices each explicit channel with
the exact `value_of_observation`:

* `task.loss == "log"` is bits and joins the one bits-per-cost list, but only when
  the binding's declared `cost_unit` is the graph's common currency. `brier` and
  `error` are separate currencies, and a cost declared in any other unit is never
  divided into the bits ranking: it is returned in `Actions.other`.
* `Action.kind == "observe"` carries the belief/task/channel fingerprints taken at
  planning time plus the frozen pre-observation `risk_law`. `apply` refuses a stale
  belief, a changed task or channel, a duplicate physical `evidence_id`, or an
  impossible outcome index BEFORE it replaces the immutable belief. Two genuinely
  new readings from one source root are allowed; a copied reading reuses its id and
  is refused.
* The ledger row adds `evidence_id`, `risk_before`, `risk_after`, `planned_cost`
  and `actual_cost` (when the caller supplies it), alongside the existing
  `value_predicted` / `value_realized` pair.

`risk_drop_law(belief, task, channel)` is the model's own conditional law of the
realized task-risk drop. An individual outcome's drop can be NEGATIVE even when
the expected drop is nonnegative (Bayes risk is concave in the belief). The narrow
adapter `next_actions.update_alarm(alarm, law, realized)` feeds that frozen law to
the EXISTING `Alarm.update(predicted, realized, outcomes=...)`; no new alarm is
constructed, and an empty law is refused rather than silently downgraded to the
bounded path.

An executable `instruments.Instrument` is connected through
`instruments.Calibration(identity, channel, reading_to_outcome)`: the caller
supplies `P(reading outcome | hypothesis)` and the map from the actual `Reading`
to the finite outcome index. `Reading.p` is the instrument's posterior confidence
and is never converted into the channel likelihood or a reliability. A correlated
multi-reading sweep needs a caller-defined JOINT likelihood; the legacy
per-reading weight `1/K` of `Instrument.enter` is one lineage root, not a joint
law, and is unchanged.

The current tests exercise XOR observation synergy, mirrored global branches,
duplicate evidence, conditional copies, relevant versus nuisance information,
long log-tail recovery, Regime parity, geometry's label-swap ambiguity, product
versus ambient metrics, Fisher null directions and the GAN/JS scale identity. The
return-path tests (`tests/test_predictive_actions.py`) add units, the ledger,
stale/changed refusals, evidence identity, impossible-outcome atomicity, the exact
binary risk law with a negative individual drop, the finite-law Alarm martingale,
correlated joint observations, and a calibrated executable instrument path.

```bash
python3 -m pytest tests/test_predictive_state.py tests/test_predictive_geometry.py -q
```

No neural-model quality, context-capacity or 10× throughput claim follows from
these reference tests. Training and budgeted execution experiments must report
their own data, full costs, calibration protocol and independent evaluation.

### Integrated outcome contract

Root review adds atomic validation of reported cost and executable calibration
identity before belief replacement. `calibrated_outcome` carries calibration
version, channel fingerprint, executable instrument and source root; `apply`
checks these against the planned action. The declared channel cost prices the
whole planned acquisition, while `planned_instrument_cost` records the executable
instrument's own price. These costs need not coincide when setup/verification is
included; the caller must declare the complete acquisition cost.

Predictive action targets are `(task_id, channel_name)` tuples, preventing slash
collisions. Channel indices are integers, task IDs are unique in EngineState,
extra metadata cannot replace the planned contract, and risk-drop laws are
frozen tuples. A changed binding identity or cost currency invalidates the plan.
