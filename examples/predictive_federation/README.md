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
