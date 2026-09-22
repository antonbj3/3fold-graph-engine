"""Constructed integration: budgeted planning -> instrument -> Graph -> native Kernel.

The finite reference posterior is used to price multi-step observations; the
compiled backend independently assimilates the same real simulated readings.
This is an executable contract example, not an end-to-end LLM speed benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.alarm import Alarm
from graph_engine.budget_plan import Action as Transition
from graph_engine.compiled_regime import CompiledRegime
from graph_engine.instruments import Calibration, Instrument, Reading, calibrated_action, calibrated_outcome
from graph_engine.next_actions import EngineState, PredictiveBinding, apply, next_actions, update_alarm
from graph_engine.plan_value import budgeted_plan
from graph_engine.predictive_state import BeliefState, RegimeSnapshot
from graph_engine.regime_posterior import RegimePosterior


class SimulatedInstrument(Instrument):
    def __init__(self, snapshot, seed, reliability):
        super().__init__("simulated-sensor", 1., reliability, "simulation:one-world")
        self.snapshot = snapshot
        self.rng = np.random.default_rng(seed)
        self.hidden = self.rng.choice(len(snapshot.belief.mass), p=snapshot.belief.mass)

    def measure(self, pair, x):
        cell = int(np.searchsorted(self.snapshot.cells[:, 1], x, side="left"))
        f = self.snapshot.positive[self.hidden, cell]
        p = 1-self.reliability+(2*self.reliability-1)*f
        sign = 1 if self.rng.random() < p else -1
        # Deliberately distinct from calibrated reliability. The adapter uses
        # the supplied likelihood, not this reported confidence.
        return Reading(sign, .95 if sign > 0 else .05, 1.)


def plan_observation(binding):
    beliefs = {binding.belief.fingerprint: binding.belief}

    def available(key):
        belief = beliefs[key]
        options = []
        for channel in binding.channels:
            successors = []
            for outcome, probability in enumerate(belief.predict(channel)):
                if probability == 0:
                    continue
                likelihood = channel.likelihood[:, outcome]
                loglike = np.full_like(likelihood, -np.inf)
                np.log(likelihood, out=loglike, where=likelihood > 0)
                hypothetical = BeliefState.from_log_mass(
                    belief.space_key, belief._log_mass+loglike, belief.evidence_ids)
                beliefs[hypothetical.fingerprint] = hypothetical
                successors.append((float(probability), hypothetical.fingerprint))
            options.append(Transition(channel.name, "observe", 1, tuple(successors)))
        return options

    return budgeted_plan(binding.belief.fingerprint, available,
                         lambda key: binding.task.risk(beliefs[key]), budget=2, price=.03)


def run(kernel_src, build_dir, seed):
    sys.path.insert(0, str(kernel_src))
    from kernel_engine.inference.finite_regime import FiniteRegimeKernel
    kernel = FiniteRegimeKernel(build_dir)
    regime = RegimePosterior(0, 1, n_grid=6, p_flip=.3, p_two=.2)
    snapshot = RegimeSnapshot.from_regime(regime)
    compiled = CompiledRegime.from_regime(regime, kernel)
    reliability = .85
    positions = snapshot.cells.mean(axis=1)
    channels = tuple(snapshot.probe(float(x), reliability, 1.) for x in positions)
    binding = PredictiveBinding("field-sign", snapshot.belief, snapshot.task(), channels)
    engine = EngineState(predictive_bindings=[binding], pair_bundles=False, probe_instruments=[])
    instrument = SimulatedInstrument(snapshot, seed, reliability)
    alarm, trajectory = Alarm(), []
    for step in range(4):
        started = time.perf_counter()
        plan = plan_observation(binding)
        planning_seconds = time.perf_counter()-started
        if plan.chosen == "stop":
            break
        index = next(i for i, c in enumerate(channels) if c.name == plan.chosen)
        channel, x = channels[index], float(positions[index])
        ranked = next_actions(engine, k=len(channels))
        calibration = Calibration("known-simulation-law:r=.85:v1", channel,
                                  lambda reading: int(reading.sign > 0))
        action = calibrated_action(engine, binding.task_id, instrument, calibration)
        outcome = calibrated_outcome(instrument, calibration, ("input", "response"), x,
                                     f"simulated:{seed}:reading:{step}")
        row = apply(engine, action, outcome)
        sign = outcome["reading"]["sign"]
        compiled.add_probe(x, sign, reliability=reliability, evidence_id=outcome["evidence_id"])
        p_reference = binding.task.predictions(binding.belief)[:, 1]
        error = float(np.max(np.abs(p_reference-compiled.p_plus_cells())))
        if error > 1e-12:
            raise AssertionError(f"native/reference assimilation mismatch: {error}")
        wealth, alarming = update_alarm(alarm, action.meta["risk_law"], row["value_realized"])
        trajectory.append({"step": step, "probe_x": x, "sign": sign,
                           "myopic_first": ranked[0].meta["channel"].name,
                           "budget_plan_first": plan.chosen, "budget_plan_objective": plan.objective,
                           "budget_plan_expected_cost": plan.expected_resource_cost,
                           "planning_seconds": planning_seconds,
                           "native_reference_max_abs": error,
                           "alarm_wealth": wealth, "alarm": alarming, "ledger": row})
    return {"scope": "constructed integration, no LLM or speed claim", "seed": seed,
            "cells": len(positions), "trajectory": trajectory,
            "final_native_predictions": compiled.p_plus_cells().tolist(),
            "distinct_physical_readings": len(binding.belief.evidence_ids)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-src', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260922)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('refusing to overwrite an existing evidence artifact')
    result = run(args.kernel_src, args.build_dir, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
