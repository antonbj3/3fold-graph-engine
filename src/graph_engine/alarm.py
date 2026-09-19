#!/usr/bin/env python3
"""alarm.py — a sequential test on the loop's own prediction residuals, with a false-alarm guarantee.

WHAT IS TESTED. Every probe the engine buys is priced before it is answered: `regime_posterior.best_probe` /
`total_value_probe` / `bundle_value` return the EXPECTED drop of a potential over the answer's outcomes. e33's identity
says the potential is a martingale and the price is its conditional expectation, so under a CORRECT model

    d_t = realized_drop_t − predicted_drop_t ,          E[d_t | F_{t−1}] = 0

EXACTLY, not asymptotically: the update is exact Bayes on the fixed partition and the world answers through the same
likelihood the price was computed with. Under MODEL error the identity fails and the residuals drift — a second
transition the reader does not represent (no mass where the truth is), box claims read pointwise when the source
reported a majority sign, a reliability that is wrong. Those are exactly the three defects measured in this package
(e21: 0 of 25 collisions; +reliability: the pointwise box likelihood, gap 0.21 → 0.56; e21f: priced bundles
over-confident, gap +0.09…+0.34 against −0.08 for the guard configuration). The alarm does not say WHICH defect it is;
it says the price is no longer the expectation of what the answer does.

THE FORM, AND WHY THIS ONE. Probes are chosen adaptively by the loop, so the residuals are neither i.i.d. nor
exchangeable: no permutation or conformal test applies. A test martingale (Vovk; Shafer's game-theoretic testing) needs
neither. Here the null model supplies its OWN predictive distribution of the residual: before the answer, the drop takes
one of the outcome values d^(1)…d^(K) (K = 2 for a single probe) with the model's own probabilities q^(1)…q^(K). So the
exact conditional cumulant generating function is available,

    κ_t(λ) = log Σ_k q^(k) exp(λ (d^(k) − predicted_t)) ,

and for any PREDICTABLE λ_t the product  W_t = Π_{i ≤ t} exp(λ_i d̃_i − κ_i(λ_i)),  d̃ = realized − predicted, is a
nonnegative martingale with W_0 = 1 — exactly, with no boundedness, clipping, tail or variance assumption. The bet size
is set scale-free, λ = c / σ_t with σ_t the model's own predictive SD of the drop, and a uniform MIXTURE over
c ∈ ±(0.25, 0.5, 1, 2) is taken (a mixture of nonnegative martingales is one, so no α is spent on choosing the
direction or the size of the bet; the two signs cover "the loop over-prices" and "the loop under-prices"). Ville's
inequality applied to W gives, for the whole infinite sequence at once:

    GUARANTEED — exact, non-asymptotic, for any adaptive probe choice, no exchangeability and no distributional
    assumption beyond the model's own predictive law:        P( ∃t : W_t ≥ 1/α )  ≤  α.

WHAT IS NOT GUARANTEED.
  • POWER. Nothing is claimed. The alarm can only see a defect the probes actually visit: measured in tests, a planted
    second transition with 12 probes SWEPT over the domain fires in 146 of 200 sequences (r = 0.95; 196 of 200 at
    r = 0.99), but with the 12 probes placed by the engine's own value rule only 21 of 200 — the value rule buys probes
    where the belief is uncertain, and a confidently wrong region is not that. That is the same allocation limit e21b/e28
    measured, seen from the test side.
  • The α covers the event "the wealth ever reaches 1/α", i.e. the `hit` field. The `alarm` field LATCHES (on at 1/α,
    off again when the wealth falls back below 1) because the loop uses it as a switch; the number of on/off cycles is
    not covered by α, only the first crossing is.
  • The null is "the price equals the conditional expectation of the realized drop". A model error that leaves that
    identity intact is invisible here.
  • FALLBACK PATH. `update(predicted, realized)` without the outcome list cannot compute κ and falls back to a bounded
    betting martingale on the residual standardized by a PREDICTABLE running scale and clipped to ±m: that one is exact
    only for the clipped sequence (if the clip binds, the clipped residual's conditional mean need not be 0), and it is
    much weaker — a single huge surprise can move it by at most a factor 1 + lam. Measured on the same planted worlds:
    0 of 200. Use `drop_outcomes` and pass the outcomes whenever the posterior is at hand.

THE LAW IS REQUIRED, AND `update` REFUSES WITHOUT ONE (`require_law`, default True). The α above is a statement about
the model's own predictive law of the drop; a caller that has no such law does not have a weaker guarantee, it has no
guarantee of the stated kind, so the default is a refusal rather than a silent downgrade to the fallback path.
MEASURED, on a domain that has no predictive law at all — an iterative solver's per-sweep residual drop, which is a
deterministic function of the iterate rather than a draw from a model: fitted to a trailing rate estimate the residual
is predictable by construction, the null is violated only during the transition, and the wealth never leaves its peak
1.00 in any of three scenes; with the rate frozen instead, the alarm fires at sweep 247 against a one-line stagnation
heuristic's sweep 104 with the SAME final error 2.384e-7; and on the two ill-conditioned scenes, where a stagnation
rule stops at sweep 6 with 99.6 % / 99.7 % relative error, the alarm never fires at all. It correctly refuses to be
fooled there, and it buys nothing — the exact-α path was running either the fallback or an INVENTED two-point law,
for which α is exact for that invention and not for the sequence. Pass `require_law=False` to take the bounded path
deliberately; `state()["n_bounded"]` then counts the updates that ran without a law.

USE.
    out = drop_outcomes(post, x, r)          # BEFORE the probe: the model's own law of the drop (required)
    before = post.potential_value()
    post.add_probe(x, answer, r)
    wealth, alarm = al.update(out_mean(out), before - post.potential_value(), out)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Alarm", "drop_outcomes", "out_mean", "expected_drop"]

BETS = (-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0)   # bet sizes in units of 1/σ; mixed uniformly


def drop_outcomes(post, x: float, reliability: float) -> list[tuple[float, float]]:
    """[(probability, drop)] — the model's own predictive law of the potential drop of the probe (x, reliability) that
    is ACTUALLY going to be executed, exact over the two answers on the fixed partition. This is `best_probe`'s
    computation evaluated at a given x instead of at its argmax (the loop executes probes chosen by several rules, and
    the residual must be taken against the price of the probe that was bought). Call BEFORE `add_probe`."""
    cells, w, F, p = post._with_probes()
    c = int(min(np.searchsorted(cells[:, 1], x, side="left"), len(cells) - 1))
    f = F[:, c]
    now = float(post._u(p @ F) @ w)
    out = []
    for sg in (1, -1):
        ff = f if sg > 0 else 1 - f
        like = ff * reliability + (1 - ff) * (1 - reliability)
        q = float(p @ like)
        if q <= 0:
            continue
        out.append((q, now - float(post._u((p * like / q) @ F) @ w)))
    return out


def out_mean(outcomes) -> float:
    """The price: Σ q·drop — equal to `best_probe`'s gain at its own argmax."""
    return float(sum(q * d for q, d in outcomes))


def expected_drop(post, x: float, reliability: float) -> float:
    return out_mean(drop_outcomes(post, x, reliability))


@dataclass
class Alarm:
    """A test martingale on the prediction residuals. P(ever alarm | correct model) ≤ alpha, exactly (Ville)."""

    alpha: float = 0.05
    bets: tuple = BETS
    lam: float = 0.5                  # fallback path only: the bounded bet, factors in [1 − lam, 1 + lam]
    m: float = 4.0                    # fallback path only: clip of the standardized residual
    scale0: float = 0.05              # fallback path only: scale before any residual is seen
    require_law: bool = True          # refuse an update with no predictive law instead of downgrading silently
    n: int = 0
    hit: bool = False                 # has the wealth EVER reached 1/alpha — this is what alpha covers
    alarm: bool = False               # latched switch: on at 1/alpha, off again below 1
    n_alarms: int = 0
    n_clipped: int = 0
    n_bounded: int = 0                # updates taken on the fallback path (require_law=False)
    peak: float = 1.0
    legs: np.ndarray | None = None
    _wp: float = 1.0
    _wm: float = 1.0
    _ss: float = 0.0
    history: list = field(default_factory=list)

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1): {self.alpha}")
        if not 0 < self.lam < 1:
            raise ValueError(f"lam must be in (0, 1): {self.lam}")
        self.legs = np.full(len(self.bets), 1.0 / len(self.bets))

    @property
    def wealth(self) -> float:
        return float(self.legs.sum()) if self.n or self.legs is not None else 1.0

    @property
    def threshold(self) -> float:
        return 1.0 / self.alpha

    # -- the two paths -----------------------------------------------------------------------------
    def _exact(self, resid: float, outcomes) -> None:
        """Mixture of exp(λ d̃ − κ(λ)) over the bet grid: exact martingale, no clipping, no assumption."""
        q = np.array([o[0] for o in outcomes], float)
        d = np.array([o[1] for o in outcomes], float)
        q = q / q.sum()
        mu = float(q @ d)
        sd = float(np.sqrt(max(float(q @ (d - mu) ** 2), 0.0)))
        if sd <= 1e-12:                                   # a degenerate probe says nothing about the model
            return
        for i, c in enumerate(self.bets):
            lam = c / sd
            kap = float(np.log(float(q @ np.exp(np.clip(lam * (d - mu), -60.0, 60.0)))))
            self.legs[i] *= float(np.exp(np.clip(lam * resid - kap, -60.0, 60.0)))

    def _bounded(self, resid: float) -> None:
        """Fallback: a clipped bet on the residual standardized by a predictable running scale."""
        s = self.scale0 if (self.n < 2 or self._ss <= 0) else max(float(np.sqrt(self._ss / self.n)), 1e-9)
        z = resid / s
        if abs(z) > self.m:
            self.n_clipped += 1
        u = float(np.clip(z, -self.m, self.m)) / self.m
        self._wp *= 1.0 + self.lam * u
        self._wm *= 1.0 - self.lam * u
        self.legs = np.array([0.5 * self._wp, 0.5 * self._wm])

    def update(self, predicted: float, realized: float, outcomes=None) -> tuple[float, bool]:
        """One action: its price, the drop it actually produced, and (if available) the model's own law of that drop.
        Returns (wealth, alarm)."""
        d = float(realized) - float(predicted)
        if outcomes:
            self._exact(d, outcomes)
        elif self.require_law:
            raise ValueError(
                "Alarm.update: no predictive law was supplied for this drop, so the exact-alpha martingale cannot be "
                "formed and the alpha guarantee does not hold. Pass the model's own law of the drop as `outcomes` "
                "([(probability, drop)], e.g. from drop_outcomes(post, x, r)); if the quantity has no predictive law "
                "(a deterministic residual, a rate fitted to the sequence itself), construct the Alarm with "
                "require_law=False to take the documented-weaker bounded path deliberately.")
        else:
            self._bounded(d)
            self.n_bounded += 1
        self.n += 1
        self._ss += d * d
        w = self.wealth
        self.peak = max(self.peak, w)
        if w >= self.threshold:
            if not self.alarm:
                self.n_alarms += 1
            self.alarm, self.hit = True, True
        elif w < 1.0:
            self.alarm = False
        self.history.append((float(d), float(w), bool(self.alarm)))
        return w, self.alarm

    def state(self) -> dict:
        return {"n": self.n, "wealth": self.wealth, "peak": self.peak, "alarm": self.alarm, "hit": self.hit,
                "n_alarms": self.n_alarms, "n_clipped": self.n_clipped, "n_bounded": self.n_bounded,
                "require_law": self.require_law}
