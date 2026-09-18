#!/usr/bin/env python3
"""
decision_cert.py — certificates that are relative to a DECISION, and the measurement that can flip one.

WHAT A DECISION IS HERE. A `Decision` is a named predicate over the beliefs the engine already holds, with a
required confidence 1 − α:

    sign    "the sign of pair P is σ on the whole box [a, b]"          — read from regime_posterior
    margin  "edge E holds with z ≥ z_req"                             — read from margin_net
    chain   "every node of the goal chain G holds (p ≥ τ)"            — read from the unlock graph's beliefs

`certify(decision, state)` returns {"holds", "p_flip", "credible_set_checked", "kind": "decision-relative", ...}.
`p_flip` is the posterior probability that the PREDICATE is false, computed EXACTLY where the channel allows it:

    regime  the posterior mass of the hypotheses under which the predicate fails. The hypothesis space is
            regime_posterior's own fixed partition (all+, all−, one transition per cell, and the two-transition
            family when p_two > 0), so this is a finite sum, not a bound. A hypothesis whose TRANSITION sits in a
            cell that overlaps the box counts as FAILING the predicate: that cell carries both signs, and the
            partition does not resolve where inside it the transition sits. The rule is conservative by exactly
            one cell width and is declared, not tuned.
    margin  Φ(−(z − z_req)) with z = m̂/s from margin_net's GLS estimate — exact under that channel's declared
            Gaussian error model, and under nothing else.
    chain   1 − Π p_i, the complement of the product — this ASSUMES the nodes fail independently. Said plainly:
            the number is exact arithmetic on a declared assumption; unlock_value's own docstring records that
            shared causes break it. No correlation is estimated here.

`holds` is `p_flip ≤ α` — the decision's OWN level. There is no global threshold in this module: nothing is
compared to a repository-wide τ, and no number here is a property of a node.

WHAT A DECISION-RELATIVE CERTIFICATE DOES NOT SAY. It says nothing about the node, pair or edge globally. A pair
whose transition point is completely unknown (wide posterior, high entropy) can still certify "the sign is + on
[0, 0.2]" if every hypothesis with any posterior mass is + there; and a pair that is narrow everywhere else can
fail to certify a decision whose box straddles its transition. Trust is (twin, DECISION)-relative. Two decisions
on the same pair can hold and fail at the same time, and that is the point, not an inconsistency. A certificate
here binds (decision, box/edge/chain, α) and nothing else.

FLIP ATTRIBUTION — the next measurement is the one that can CHANGE A DECISION. `flip_attribution(decision, state,
instruments)` enumerates the candidate observations the decision's own channels offer, and for each returns the
probability that its outcome flips the certificate's status (holds → not holds, or back):

    regime probe at x     exact over the TWO outcomes: P(flip) = Σ over sign ∈ {+, −} of P(sign) · [status after
                          the Bayes update ≠ status now]. The instrument's reliability enters the likelihood.
    margin measurement    Gaussian and exact: before it is taken the posterior mean is μ' ~ N(m̂, s² − s'²) with
                          s'⁻² = s⁻² + σ_new⁻² (margin_net's own pre-posterior). The status flips across the
                          single point μ* = s'·(z_req + z_{1−α}), so P(flip) is one normal tail.
    chain instrument      exact over the two answers of a yes/no instrument on one node (plan_value's Bayes update).

The candidates are restricted to what the decision READS. A probe on another pair cannot change this decision's
certificate, whatever its entropy drop is, so it is not in the list — which is the whole difference from
`next_actions`, where the currency is the expected entropy drop of a channel's own potential.

FLIPS ARE RELATIVE TO THE CURRENT α. Every P(flip) in this module is the probability of crossing the boundary
p_flip = α of THIS decision. Change α and the ranking changes; a decision certified with p_flip far below α has
zero-value candidates (no single answer can carry it back across), and the same instrument on the same pair can be
the top candidate for a decision whose p_flip sits near its α. A zero here means "cannot flip THIS decision at
THIS α", never "uninformative".

`flip_actions` is the adapter used by `next_actions(..., decisions=[...])`: it wraps the ranking as Actions of kind
"flip" with value_unit "p_flip", which keeps them out of the bits ranking (a probability is not a bit) and in
`.other`, where the non-bits channels already live.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm

__all__ = ["Decision", "certify", "flip_attribution", "flip_actions"]

KINDS = ("sign", "margin", "chain")
_TOL = 1e-12


@dataclass(frozen=True)
class Decision:
    """A named predicate over the engine's beliefs, with the confidence it is required to hold at (1 − alpha).

    kind="sign":   `pair` (the regime_posterior key), `box` = (a, b), `sign` = +1 / −1.
    kind="margin": `edge` (the margin_net edge id), `z_req` (the required distance to the boundary, in σ).
    kind="chain":  `nodes` (ids whose beliefs live in EngineState.p_holds), `tau` (a node holds at p ≥ tau).
    """
    name: str
    kind: str
    alpha: float = 0.05
    pair: Any = None
    box: tuple = (0.0, 1.0)
    sign: int = 1
    edge: str | None = None
    z_req: float = 0.0
    nodes: tuple = ()
    tau: float = 0.9

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown decision kind {self.kind!r}; expected one of {list(KINDS)}")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1): {self.alpha}")
        if self.kind == "sign":
            if self.pair is None:
                raise ValueError(f"decision {self.name!r}: a 'sign' decision needs a pair")
            a, b = self.box
            if not b >= a:
                raise ValueError(f"decision {self.name!r}: box must have b ≥ a, got {self.box}")
            if self.sign not in (1, -1):
                raise ValueError(f"decision {self.name!r}: sign must be +1 or −1, got {self.sign!r}")
        elif self.kind == "margin":
            if not self.edge:
                raise ValueError(f"decision {self.name!r}: a 'margin' decision needs an edge id")
        elif not self.nodes:
            raise ValueError(f"decision {self.name!r}: a 'chain' decision needs at least one node")

    @property
    def targets(self) -> list:
        """What the decision READS — the only places a measurement can flip it."""
        return [self.pair] if self.kind == "sign" else [self.edge] if self.kind == "margin" else list(self.nodes)

    def __str__(self) -> str:
        return f"'{self.name}' ({self.kind}, 1 − α = {1 - self.alpha:g})"


# -- the predicate, per channel -----------------------------------------------------------------------
def _regime_pieces(decision: Decision, post) -> tuple:
    """(posterior, mask of hypotheses that FAIL the predicate, cells, F). The mask does not depend on data."""
    cells, _w, F, p = post._with_probes()
    a, b = float(decision.box[0]), float(decision.box[1])
    ov = np.minimum(cells[:, 1], b) - np.maximum(cells[:, 0], a)
    sel = ov > _TOL
    if not sel.any():                                   # zero-width box: the cell that contains it
        sel = (cells[:, 0] <= a) & (a <= cells[:, 1])
        if not sel.any():
            sel = np.zeros(len(cells), bool); sel[0 if a < cells[0, 0] else len(cells) - 1] = True
    want = 1.0 if decision.sign > 0 else 0.0
    ok = np.all(np.abs(F[:, sel] - want) < 1e-9, axis=1)  # a transition cell (F = 0.5) fails: see docstring
    return p, ~ok, cells, F


def _margin_pieces(decision: Decision, state) -> tuple:
    net = getattr(state, "margins", None)
    if net is None or decision.edge not in getattr(net, "edges", {}):
        raise ValueError(f"decision {decision.name!r}: no margin edge {decision.edge!r} in the state")
    return net, net.estimate(decision.edge)


def _chain_ps(decision: Decision, state) -> dict:
    graph = getattr(state, "unlock_graph", None)
    nodes = {n["id"]: n for n in (graph or {}).get("nodes", [])}
    out = {}
    for i in decision.nodes:
        if i in nodes:
            from .unlock_value import p_of
            out[i] = float(p_of(nodes[i], dict(state.p_holds)))
        elif i in state.p_holds:
            out[i] = float(state.p_holds[i])
        else:
            raise ValueError(f"decision {decision.name!r}: no belief for node {i!r} (not in p_holds, not in the graph)")
    return out


def certify(decision: Decision, state) -> dict:
    """The (decision, state)-relative certificate. `p_flip` = P(the predicate is false); `holds` = p_flip ≤ α.

    Nothing in the returned dict is a statement about a node, pair or edge on its own (module docstring)."""
    if decision.kind == "sign":
        post = getattr(state, "regimes", {}).get(decision.pair)
        if post is None:
            raise ValueError(f"decision {decision.name!r}: no regime posterior for pair {decision.pair!r}")
        p, fail, cells, _F = _regime_pieces(decision, post)
        p_flip = float(p[fail].sum())
        checked = (f"exact sum over the {len(p)} hypotheses of regime_posterior's fixed partition "
                   f"({len(cells)} cells); the failing set is the {int(fail.sum())} hypotheses whose sign on "
                   f"[{decision.box[0]:g}, {decision.box[1]:g}] is not {'+' if decision.sign > 0 else '−'} "
                   f"everywhere (a transition inside an overlapping cell counts as failing)")
        detail = {"pair": decision.pair, "box": [float(decision.box[0]), float(decision.box[1])],
                  "sign": int(decision.sign), "n_hypotheses": int(len(p)), "n_failing": int(fail.sum())}
        channel = "regime_posterior"
    elif decision.kind == "margin":
        _net, est = _margin_pieces(decision, state)
        p_flip = float(norm.cdf(decision.z_req - est.z))
        checked = (f"Gaussian posterior N(m̂ = {est.m:g}, s = {est.s:g}) from margin_net's GLS estimate on edge "
                   f"'{decision.edge}'; p_flip = Φ(−(z − z_req)) with z = {est.z:g}, z_req = {decision.z_req:g} — "
                   f"exact under that channel's declared error model and under nothing else")
        detail = {"edge": decision.edge, "z": float(est.z), "z_req": float(decision.z_req), "m": float(est.m),
                  "s": float(est.s), "edge_kind": est.kind}
        channel = "margin_net"
    else:
        ps = _chain_ps(decision, state)
        p_all = 1.0
        for v in ps.values():
            p_all *= v
        p_flip = float(1.0 - p_all)
        checked = (f"complement of the product over {len(ps)} node beliefs, under DECLARED independence "
                   f"(not verified here): p_flip = 1 − Π p_i, p_i = P(node holds at τ = {decision.tau:g})")
        detail = {"nodes": dict(ps), "tau": float(decision.tau), "p_all_hold": float(p_all),
                  "weakest_node": min(ps, key=ps.get) if ps else None}
        channel = "unlock_value"
    return {"kind": "decision-relative", "decision": decision.name, "decision_kind": decision.kind,
            "holds": bool(p_flip <= decision.alpha), "p_flip": p_flip, "alpha": float(decision.alpha),
            "credible_set_checked": checked, "channel": channel,
            "scope": (f"binds ({decision.name}, {decision.kind} target {decision.targets}, α = {decision.alpha:g}) "
                      f"only; it says nothing about the target globally, nor about any other decision"),
            **detail}


# -- flip attribution ---------------------------------------------------------------------------------
def _instruments_for(decision: Decision, state, instruments):
    if instruments is not None:
        return list(instruments)
    from .next_actions import Instrument
    if decision.kind == "sign":
        return list(getattr(state, "probe_instruments", []) or [Instrument("judge", 1.0, 0.8)])
    if decision.kind == "margin":
        ins = getattr(state, "measure_instrument", None)
        return [ins] if ins is not None else [Instrument("measurement", 1.0, 1.0, sigma=state.margins.default_sigma)]
    return []


def _regime_attribution(decision: Decision, state, instruments) -> list[dict]:
    post = state.regimes[decision.pair]
    p, fail, cells, F = _regime_pieces(decision, post)
    now = bool(float(p[fail].sum()) <= decision.alpha)
    var = state.variable(decision.pair) if hasattr(state, "variable") else "x"
    out = []
    for ins in instruments:
        r = float(ins.reliability)
        for c in range(len(cells)):
            f0 = F[:, c]
            p_flip, flipping = 0.0, []
            for sg in (1, -1):
                f = f0 if sg > 0 else 1 - f0
                like = f * r + (1 - f) * (1 - r)
                pout = float(p @ like)
                if pout <= 0:
                    continue
                q = p * like / pout
                after = bool(float(q[fail].sum()) <= decision.alpha)
                if after != now:
                    p_flip += pout
                    flipping.append({"sign": sg})
            x = float(cells[c].mean())
            out.append({"channel": "regime_posterior", "action_kind": "probe", "target": decision.pair,
                        "instrument": ins, "x": x, "cost": float(ins.cost),
                        "p_flip_status": float(p_flip), "value_per_cost": float(p_flip) / float(ins.cost),
                        "flipping_outcome": flipping[0] if len(flipping) == 1 else (flipping or None),
                        "exact": "over the two outcomes",
                        "how": f"observe the sign at {var} = {x:g} with {ins}"})
    return out


def _margin_attribution(decision: Decision, state, instruments) -> list[dict]:
    net, est = _margin_pieces(decision, state)
    now = bool(float(norm.cdf(decision.z_req - est.z)) <= decision.alpha)
    z_req_eff = float(decision.z_req) + float(norm.ppf(1.0 - decision.alpha))   # p_flip ≤ α ⟺ μ' ≥ s'·z_req_eff
    out = []
    for ins in instruments:
        sn = float(ins.sigma if getattr(ins, "sigma", None) is not None else net.default_sigma)
        if not math.isfinite(est.s):                       # NO-DATA edge: the first report is the whole estimate
            s1, spread, mu = sn, sn, 0.0
        else:
            s1 = math.sqrt(1.0 / (1.0 / est.s ** 2 + 1.0 / sn ** 2))
            spread = math.sqrt(max(est.s ** 2 - s1 ** 2, 0.0))
            mu = float(est.m)
        mu_star = s1 * z_req_eff
        if spread <= 0:
            p_flip, flipping = 0.0, None
        elif now:                                          # certified now: it flips if the mean lands too low
            p_flip = float(norm.cdf((mu_star - mu) / spread))
            flipping = {"posterior_mean_below": float(mu_star)}
        else:
            p_flip = float(1.0 - norm.cdf((mu_star - mu) / spread))
            flipping = {"posterior_mean_at_or_above": float(mu_star)}
        out.append({"channel": "margin_net", "action_kind": "measure", "target": decision.edge, "instrument": ins,
                    "cost": float(ins.cost), "sigma_new": sn, "p_flip_status": p_flip,
                    "value_per_cost": p_flip / float(ins.cost), "flipping_outcome": flipping,
                    "exact": "Gaussian pre-posterior μ' ~ N(m̂, s² − s'²)",
                    "how": f"measure the margin on edge '{decision.edge}' with {ins} (sigma {sn:g})"})
    return out


def _chain_attribution(decision: Decision, state, instruments) -> list[dict]:
    ps = _chain_ps(decision, state)
    base = 1.0
    for v in ps.values():
        base *= v
    now = bool(1.0 - base <= decision.alpha)
    per_node = {}
    for i in decision.nodes:
        if instruments:
            per_node[i] = [(float(x.cost), float(x.reliability), x.name) for x in instruments]
        else:
            per_node[i] = [(float(t[0]), float(t[1]), t[2] if len(t) > 2 else f"{i}:i{k}")
                           for k, t in enumerate((getattr(state, "instruments", {}) or {}).get(i, []))]
    out = []
    for i, ins_list in per_node.items():
        others = 1.0
        for j, v in ps.items():
            if j != i:
                others *= v
        for cost, r, name in ins_list:
            p_flip, flipping = 0.0, []
            for yes in (True, False):
                pi = ps[i]
                q = pi * r + (1 - pi) * (1 - r)
                q = q if yes else 1 - q
                if q <= 0:
                    continue
                pi_new = pi * r / (pi * r + (1 - pi) * (1 - r)) if yes else pi * (1 - r) / (1 - (pi * r + (1 - pi) * (1 - r)))
                after = bool(1.0 - pi_new * others <= decision.alpha)
                if after != now:
                    p_flip += q
                    flipping.append({"answer": "holds" if yes else "fails"})
            out.append({"channel": "unlock_value", "action_kind": "work_node", "target": i,
                        "instrument": name, "cost": float(cost), "p_flip_status": float(p_flip),
                        "value_per_cost": float(p_flip) / float(cost),
                        "flipping_outcome": flipping[0] if len(flipping) == 1 else (flipping or None),
                        "exact": "over the two answers of the instrument",
                        "how": f"settle node '{i}' with instrument '{name}' (reliability {r:g})"})
    return out


def flip_attribution(decision: Decision, state, instruments=None) -> list[dict]:
    """Candidate observations ranked by P(they flip THIS decision's certificate status) / cost.

    Only the channels the decision reads contribute: a probe on another pair, however large its entropy drop,
    cannot move this certificate. `p_flip_status` is exact over the two outcomes for regime probes and over the
    two answers of a chain instrument, and Gaussian-exact for a margin report. Every number is relative to the
    decision's current α: raise α and candidates appear, lower it and they vanish (module docstring)."""
    ins = _instruments_for(decision, state, instruments)
    if decision.kind == "sign":
        rows = _regime_attribution(decision, state, ins)
    elif decision.kind == "margin":
        rows = _margin_attribution(decision, state, ins)
    else:
        rows = _chain_attribution(decision, state, ins)
    for r in rows:
        r["decision"] = decision.name
        r["alpha"] = float(decision.alpha)
    return sorted(rows, key=lambda r: (-r["value_per_cost"], -r["p_flip_status"]))


def flip_actions(state, decisions, per_decision: int = 3) -> list:
    """The top `per_decision` flip candidates per decision, as next_actions.Action of kind "flip".

    value_unit is "p_flip", never bits: a flip probability is not an entropy drop and is not comparable to one, so
    these are returned in `Actions.other` beside the other non-bits channels. `how` names the decision, the
    certificate's current status and the outcome that would flip it. The answer is routed back through the
    UNDERLYING channel action (a probe on the pair, a measurement on the edge), not through a "flip" action."""
    out = []
    for d in decisions:
        cert = certify(d, state)
        status = "certified" if cert["holds"] else "not certified"
        for row in flip_attribution(d, state)[:per_decision]:
            if row["p_flip_status"] <= 0.0:
                continue
            from .next_actions import Action, Instrument
            ins = row["instrument"] if isinstance(row["instrument"], Instrument) else None
            box = None
            if d.kind == "sign" and "x" in row:
                var = state.variable(d.pair) if hasattr(state, "variable") else "x"
                box = {var: [row["x"], row["x"]]}
            out.append(Action(
                "flip", row["target"], ins, float(row["p_flip_status"]), float(row["cost"]),
                float(row["value_per_cost"]), row["channel"],
                f"decision {d}: {status} (p_flip {cert['p_flip']:.3g} vs α {d.alpha:g}); {row['how']} — "
                f"P(this flips the decision) {row['p_flip_status']:.3g}; flipping outcome "
                f"{row['flipping_outcome']}; route the answer back as a '{row['action_kind']}' on "
                f"{row['target']!r}",
                box=box, value_unit="p_flip",
                meta={"decision": d.name, "decision_kind": d.kind, "p_flip_decision": cert["p_flip"],
                      "holds": cert["holds"], "apply_as": row["action_kind"], **({"x": row["x"]} if "x" in row else {})}))
    return sorted(out, key=lambda a: -a.value_per_cost)
