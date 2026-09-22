# Predictive federation integration

Run the existing planner, executable instrument, predictive action ledger, Alarm
and injected native Kernel backend together on a small constructed problem:

```bash
python3 examples/predictive_federation/run.py \
  --kernel-src /path/to/3fold-kernel-engine/src \
  --build-dir /tmp/predictive-federation-build \
  --out /tmp/predictive-federation-result.json
```

The Kernel checkout must include its experimental `inference` module. Building
the backend requires `g++`; compilation is explicit in the chosen build directory.
The output file must be new, so rerunning does not replace previous evidence.

Each iteration uses `plan_value.budgeted_plan` to price a two-reading policy, then
executes its first action and replans. The reference belief prices outcomes;
`CompiledRegime` independently assimilates the same physical reading IDs. Every
iteration verifies decoded predictions against the reference to `1e-12`, records
planned/actual acquisition cost, and sends the pre-observation risk-drop law to
the existing Alarm. Reported instrument confidence is deliberately distinct from
the calibrated observation likelihood.

The example is a simulated finite-family integration, with six candidate probe
locations. Its acquisition costs count readings; planning wall time is reported
separately. It is not a neural training run, a claim of general optimality, or a
speed comparison against an LLM. Training transitions and their finite-demand
example are documented in [BUDGET_PLAN](../../docs/BUDGET_PLAN.md).

## Export existing beliefs into a training objective

`training_state.CompiledTrainingBatch` snapshots named `CompiledRegime` instances
with one common partition and transition prior. It folds existing claim
likelihoods into the natural parameters and retains probe evidence identities.
The exported family mass is the prior; feeding the updated posterior mass back
as a prior would apply evidence twice. Rows with different geometry or priors
are refused, even when their numeric array dimensions happen to agree.

```python
from graph_engine.training_state import CompiledTrainingBatch
from kernel_engine.inference.forecast_training import ProbeDistillation

batch = CompiledTrainingBatch.from_regimes([
    ("history-a", compiled_a), ("history-b", compiled_b),
])
objective = ProbeDistillation(kernel, batch.natural, cells=[1, 3],
                             family_mass=batch.family_mass, reliability=0.85)
rows = batch.indices_for(["history-b", "history-a"])
# Keep feature rows paired with the same IDs; head output has shape (B,2,Q).
loss = objective(head(features_for_these_rows), rows).combined.mean()
loss.backward()
```

This is a frozen numeric snapshot; adding evidence to a live regime does not
change saved training targets. Create a new snapshot and objective after such
an update. The snapshot fingerprint covers row IDs, numeric state, geometry,
prior and retained probe IDs. It does not establish independence across
training examples or replace the raw evidence needed for recalibration.
Graph imports no PyTorch or cross-repository model framework at module load.
The injected Kernel objective computes teacher-weighted current/future query
losses, with costs and limitations documented in Kernel's predictive-inference
guide. Its implementation is validated; a general model-quality improvement
has not been established.

## Audit learner states against named Graph beliefs

The snapshot can also compare predicted natural states on its fixed partition
and prior with explicitly named teacher rows:

```python
audit = batch.audit_predictions(
    predicted_natural, example_ids, backend=kernel, kl_budget_nats=0.01
)
```

Each row reports a global mathematical KL upper bound in both directions. The
bound survives the same future likelihood updates in exact arithmetic; it is
stronger in scope than agreement on the currently selected queries. A row whose
bound exceeds the budget is `unresolved`, since a loose upper bound does not
prove a bad prediction. This audit owns a frozen copy of the learner buffer,
preserves teacher evidence IDs and fingerprints, and changes no live belief.
It does not establish calibration, identical physical meanings of the learner's
coordinates, or the numerical accuracy of a cached/readout answer. The supplied
Kernel needs the `drift_bound` API documented in `GLOBAL_PREDICTIVE_DRIFT.md`.
