"""A finite reference for shared predictive state, evidence updates and task OED.

The hypotheses are supplied by the caller: this creates no graph or node store.
Bayes updates and proper-score Bayes risks are established methods. This module
gives Graph's explicit hypotheses and learned approximations a common observable
contract, including unresolved branches, observation identity and task cost.

A likelihood without ``conditioned_on`` DECLARES conditional independence from
past observations given the hypotheses. We cannot infer that independence from
distinct source names. Correlated observations require an explicit joint channel
or a conditional channel bound to the exact preceding state fingerprint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Literal

import numpy as np


def _probabilities(value, ndim: int) -> np.ndarray:
    a = np.array(value, dtype=np.float64, copy=True)
    if a.ndim != ndim or 0 in a.shape:
        raise ValueError(f"expected a nonempty {ndim}-dimensional probability array")
    if not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a > 1):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.allclose(a.sum(axis=-1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("probabilities must sum to one on the outcome axis")
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class BeliefState:
    """Mass on fixed hypotheses, retaining log mass for long evidence histories.

    A zero supplied in the prior excludes that hypothesis. A displayed ``mass``
    can underflow for a very unlikely branch; its finite log mass is retained so
    contrary later evidence can recover it. Predictive expectations use float64.
    """

    space_key: str
    mass: np.ndarray
    evidence_ids: frozenset[str] = field(default_factory=frozenset)
    _log_mass: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.space_key, str) or not self.space_key:
            raise ValueError("space_key must identify hypothesis meaning AND ordering")
        object.__setattr__(self, "mass", _probabilities(self.mass, 1))
        logs = np.full_like(self.mass, -np.inf)
        np.log(self.mass, out=logs, where=self.mass > 0)
        logs.setflags(write=False)
        object.__setattr__(self, "_log_mass", logs)
        ids = frozenset(self.evidence_ids)
        if any(not isinstance(i, str) or not i for i in ids):
            raise ValueError("evidence identities must be nonempty strings")
        object.__setattr__(self, "evidence_ids", ids)

    @classmethod
    def from_log_mass(cls, space_key: str, log_mass, evidence_ids=frozenset()) -> BeliefState:
        """Normalize log weights while preserving representable log-tail support."""
        logs = np.array(log_mass, dtype=float, copy=True)
        if (logs.ndim != 1 or not len(logs) or np.any(np.isnan(logs))
                or np.any(np.isposinf(logs))):
            raise ValueError("log mass must be finite or negative infinity")
        peak = float(logs.max())
        if not np.isfinite(peak):
            raise ValueError("outcome impossible under supported hypotheses; expand the model")
        logs -= peak
        logs -= np.log(np.exp(logs).sum())
        state = cls(space_key, np.exp(logs), evidence_ids)
        logs.setflags(write=False)
        object.__setattr__(state, "_log_mass", logs)
        return state

    @property
    def fingerprint(self) -> str:
        metadata = json.dumps([self.space_key, sorted(self.evidence_ids)],
                              ensure_ascii=True, separators=(",", ":")).encode()
        return hashlib.sha256(metadata + b"\0" + self._log_mass.astype("<f8").tobytes()).hexdigest()

    def predict(self, channel: ObservationChannel) -> np.ndarray:
        channel.check(self)
        return self.mass @ channel.likelihood

    def observe(self, channel: ObservationChannel, outcome: int,
                evidence_id: str) -> BeliefState:
        """Consume a NEW physical/simulated observation, never a copied answer.

        Evidence identity is supplied externally, not guessed from answer text.
        Simulation histories can train an approximation; they are not additional
        independent evidence about the external world.
        """
        channel.check(self)
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("an observation needs a stable nonempty evidence identity")
        if evidence_id in self.evidence_ids:
            raise ValueError("this observation was already consumed")
        if isinstance(outcome, (bool, np.bool_)) or not isinstance(outcome, (int, np.integer)):
            raise ValueError("outcome must be an integer index")
        if not 0 <= outcome < channel.likelihood.shape[1]:
            raise ValueError("outcome index outside channel")
        likelihood = channel.likelihood[:, outcome]
        loglike = np.full_like(likelihood, -np.inf)
        np.log(likelihood, out=loglike, where=likelihood > 0)
        return BeliefState.from_log_mass(self.space_key, self._log_mass + loglike,
                                         self.evidence_ids | {evidence_id})


@dataclass(frozen=True, eq=False)
class ObservationChannel:
    """P(outcome | hypothesis, declared history); rows must sum to one.

    A bundle is ONE joint channel, including correlations between its outcomes.
    Cost includes acquisition and inference work in caller-declared common units.
    ``conditioned_on`` is either the exact input-state fingerprint or None, which
    explicitly assumes conditional independence from previous evidence.
    """

    name: str
    space_key: str
    likelihood: np.ndarray
    cost: float = 1.0
    conditioned_on: str | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("channel needs a name")
        if not isinstance(self.space_key, str) or not self.space_key:
            raise ValueError("channel needs a hypothesis-space identity")
        object.__setattr__(self, "likelihood", _probabilities(self.likelihood, 2))
        if not np.isfinite(self.cost) or self.cost <= 0:
            raise ValueError("cost must be finite and strictly positive")

    def check(self, state: BeliefState) -> None:
        if self.space_key != state.space_key or len(self.likelihood) != len(state.mass):
            raise ValueError("channel and belief use different hypothesis spaces")
        if self.conditioned_on is not None and self.conditioned_on != state.fingerprint:
            raise ValueError("conditional channel belongs to another evidence state")

    @property
    def fingerprint(self) -> str:
        """Content identity of the channel: a change to the likelihood, cost or conditioning is a new channel."""
        metadata = json.dumps([self.name, self.space_key, self.conditioned_on],
                              ensure_ascii=True, separators=(",", ":")).encode()
        return hashlib.sha256(metadata + b"\0" + self.likelihood.astype("<f8").tobytes()
                              + b"\0" + np.float64(self.cost).tobytes()).hexdigest()


@dataclass(frozen=True, eq=False)
class PredictiveTask:
    """Weighted queries with Bayes risk under log, Brier or classification loss.

    ``readout[h,q,y]`` is P(y | hypothesis h, query q), without measurement
    noise unless that noise is part of the deployment target. Query weights
    describe deployment demand. This bank does not prove global equivalence.
    Log risk uses bits; Brier is the full multiclass squared-error convention.
    """

    space_key: str
    readout: np.ndarray
    weights: np.ndarray
    loss: Literal["log", "brier", "error"] = "log"

    def __post_init__(self):
        a = _probabilities(self.readout, 3)
        weights = _probabilities(self.weights, 1)
        if a.shape[1] != len(weights):
            raise ValueError("one deployment weight is required per query")
        if self.loss not in {"log", "brier", "error"}:
            raise ValueError("unknown task loss")
        object.__setattr__(self, "readout", a)
        object.__setattr__(self, "weights", weights)

    def predictions(self, state: BeliefState) -> np.ndarray:
        if self.space_key != state.space_key or len(self.readout) != len(state.mass):
            raise ValueError("task and belief use different hypothesis spaces")
        return np.einsum("h,hqy->qy", state.mass, self.readout)

    def risk(self, state: BeliefState) -> float:
        p = self.predictions(state)
        if self.loss == "log":
            logp = np.zeros_like(p)
            np.log2(p, out=logp, where=p > 0)
            risks = -(p * logp).sum(axis=1)
        elif self.loss == "brier":
            risks = 1 - (p * p).sum(axis=1)
        else:
            risks = 1 - p.max(axis=1)
        return float(self.weights @ risks)

    @property
    def fingerprint(self) -> str:
        """Content identity of the task: space, loss, readout and query weights all change the ranking."""
        metadata = json.dumps([self.space_key, self.loss, list(self.readout.shape), list(self.weights.shape)],
                              ensure_ascii=True, separators=(",", ":")).encode()
        return hashlib.sha256(metadata + b"\0" + self.readout.astype("<f8").tobytes()
                              + b"\0" + self.weights.astype("<f8").tobytes()).hexdigest()


@dataclass(frozen=True)
class ActionValue:
    name: str
    risk_before: float
    expected_risk_after: float
    gain: float
    cost: float

    @property
    def gain_per_cost(self) -> float:
        return self.gain / self.cost


def risk_drop_law(state: BeliefState, task: PredictiveTask,
                  channel: ObservationChannel) -> list[tuple[float, float]]:
    """[(P(outcome), realized task-risk drop)] under the PRE-observation belief — freeze it before observing.

    The drop on an individual outcome can be NEGATIVE even though the probability-weighted mean is
    nonnegative: Bayes risk is concave in the belief, so an outcome that makes the belief less certain
    about the deployed task can raise the realized risk. A martingale guarantee for the residual
    ``realized − predicted`` needs this exact conditional law of the outcome, not merely the mean.
    """
    channel.check(state)
    outcomes = state.predict(channel)
    before = task.risk(state)
    law: list[tuple[float, float]] = []
    for outcome, probability in enumerate(outcomes):
        if probability <= 0:
            continue
        likelihood = channel.likelihood[:, outcome]
        loglike = np.full_like(likelihood, -np.inf)
        np.log(likelihood, out=loglike, where=likelihood > 0)
        hypothetical = BeliefState.from_log_mass(state.space_key, state._log_mass + loglike,
                                                 state.evidence_ids)
        law.append((float(probability), float(before - task.risk(hypothetical))))
    return law


def value_of_observation(state: BeliefState, task: PredictiveTask,
                         channel: ObservationChannel) -> ActionValue:
    """Exact one-action expected task-risk reduction under the supplied model.

    This is neither a guarantee for a misspecified world nor a multi-step optimal
    policy. In particular a useful pair can have two zero-value single actions.
    """
    outcomes = state.predict(channel)
    before = task.risk(state)
    after = 0.0
    for outcome, probability in enumerate(outcomes):
        if probability <= 0:
            continue
        # Planning does not insert fictional observation identities into history.
        likelihood = channel.likelihood[:, outcome]
        loglike = np.full_like(likelihood, -np.inf)
        np.log(likelihood, out=loglike, where=likelihood > 0)
        hypothetical = BeliefState.from_log_mass(state.space_key, state._log_mass + loglike,
                                                  state.evidence_ids)
        after += float(probability) * task.risk(hypothetical)
    gain = before - after
    # Concavity gives nonnegative gain; expose numerical failures, hide only roundoff.
    if gain < -1e-10:
        raise ArithmeticError("expected Bayes risk increased; check numerical conditioning")
    return ActionValue(channel.name, before, after, max(0.0, gain), float(channel.cost))


@dataclass(frozen=True, eq=False)
class RegimeSnapshot:
    """Frozen bridge to RegimePosterior; keeps both global transition families."""

    belief: BeliefState
    cells: np.ndarray
    positive: np.ndarray
    single_family_count: int

    @classmethod
    def from_regime(cls, regime, evidence_ids: frozenset[str] = frozenset()) -> RegimeSnapshot:
        # Isolated private-API bridge, pinned by parity tests. Claims can change
        # partition boundaries; the fingerprint then changes rather than silently
        # treating hypothesis indices as interchangeable.
        cells, _, positive, mass = regime._with_probes()
        cells, positive = np.array(cells, copy=True), np.array(positive, copy=True)
        identity = hashlib.sha256(b"regime-grid-v1\0" + cells.astype("<f8").tobytes()
                                  + positive.astype("<f8").tobytes()).hexdigest()
        cells.setflags(write=False)
        positive.setflags(write=False)
        belief = BeliefState("regime:" + identity, mass, evidence_ids)
        return cls(belief, cells, positive, regime._n_one)

    def task(self, loss: Literal["log", "brier", "error"] = "log") -> PredictiveTask:
        weights = self.cells[:, 1] - self.cells[:, 0]
        weights = weights / weights.sum()
        readout = np.stack([1 - self.positive, self.positive], axis=-1)
        return PredictiveTask(self.belief.space_key, readout, weights, loss)

    def probe(self, x: float, reliability: float = 0.95, cost: float = 1.0) -> ObservationChannel:
        if not np.isfinite(x) or not self.cells[0, 0] <= x <= self.cells[-1, 1]:
            raise ValueError("probe is outside the declared regime domain")
        if not np.isfinite(reliability) or not 0.5 <= reliability <= 1:
            raise ValueError("binary probe reliability must be in [0.5, 1]")
        cell = min(int(np.searchsorted(self.cells[:, 1], x, side="left")), len(self.cells) - 1)
        f = self.positive[:, cell]
        p = f * reliability + (1 - f) * (1 - reliability)
        return ObservationChannel(f"regime@{x:.17g}", self.belief.space_key,
                                  np.stack([1 - p, p], axis=-1), cost)
