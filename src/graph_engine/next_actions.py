#!/usr/bin/env python3
"""
next_actions.py — ONE list of next actions across the channels, and the route back for an answer.

Every channel in this package already ranks its own candidates: `regime_posterior.best_probe` (a probe on a pair),
`regime_posterior.model_check_probe` (a probe against the model error, not the sign), `margin_net.next_measurements`
(a measurement on an edge), `precision_form.rank` (an observation of a contrast), `hidden_variable.candidates` (a
variable to declare), `unlock_value.rank` / `plan_value.plan` (a node to work), `claim_federation.next_experiments`
(a stress point between two graphs). Nothing assembled them, and nothing took an answer back. This module does both.

WHAT THE AGENT DECIDES. How to execute an action: which apparatus, which prompt, which run, who is asked, in what
order inside one action, and whether it abstains. It may also PROPOSE: a new probe point, a new edge, a new
candidate contrast, a new attribute. A proposal enters as a candidate on its channel with its lineage attached (a
source id, its `derives_from`), and is then valued by that channel like any other candidate. Its own opinion of its
value is never used.

WHAT THE AGENT NEVER DECIDES. The value of an action (the channel computes the expected potential drop from the
posterior it owns), the reliability of an instrument or of itself (`source_reliability` estimates it from agreement
against pinned truths), and whether anything is settled (`plan_value`'s admission level τ, `record_guarantee`'s
threshold). An agent that reports its own confidence has reported an observation, not a weight.

ONE CURRENCY, AND WHAT IS NOT IN IT. The ranking is in BITS per unit cost: the expected drop of a potential that is
an entropy. Three channels give bits (regime potential with `potential="entropy"`, margin_net's pre-posterior
entropy drop on P(violated), precision_form's ½log₂ det drop) and are ranked together. Three do not: `unlock_value`
gives a priority (expected-cost ratio), `hidden_variable` a χ² drop, `claim_federation` an EVPI × extent. Those are
returned beside the ranking, never inside it — adding a χ² to a bit is the kind of arithmetic this repository's
cert modules exist to refuse.

GUARD SHARE. `closed_loop` spends a fixed share of the budget on `model_check_probe` because the value rule never
buys the probes that would expose a second transition (e21: 0 of 25 two-transition pairs flagged without it). The
same share (EngineProfile.guard_share, default 0.2) reserves slots for `model_check` actions here.

`apply` updates the channel's own object and returns an append-only ledger row in the FOLD_LEDGER shape
(consumers / evidence / supersedes), with the potential drop the update actually realized next to the one that was
predicted. Nothing is written to disk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Instrument", "Action", "EngineState", "Actions", "next_actions", "apply"]

BITS = "bits"


def _h(p: float) -> float:
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


@dataclass(frozen=True)
class Instrument:
    """What an action is executed with. `reliability` is P(it reports the true sign); cost is in the graph's units."""
    name: str
    cost: float = 1.0
    reliability: float = 0.8
    sigma: float | None = None            # for measurements: the σ the instrument delivers

    def __str__(self) -> str:
        return f"'{self.name}' (r {self.reliability:g}, cost {self.cost:g})"


@dataclass
class Action:
    kind: str                              # probe | measure | throw | declare_variable | model_check | work_node
    target: Any                            # pair | edge id | candidate id | attribute | node id
    instrument: Instrument | None
    value_bits: float                      # the channel's own number, in `value_unit`
    cost: float
    value_per_cost: float
    source_channel: str
    how: str                               # one line, domain-free, in the contract's terms
    box: dict | None = None                # validity where applicable
    value_unit: str = BITS                 # bits | priority | chi2_drop | evpi
    meta: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.target}"


KINDS = ("probe", "measure", "throw", "declare_variable", "model_check", "work_node")


@dataclass
class EngineState:
    """Whatever of the engine is present. Every field is optional; a channel that is absent contributes nothing."""
    federation: Any = None                                   # claim_federation.Federation
    regimes: dict = field(default_factory=dict)              # {pair: RegimePosterior}
    regime_variables: dict = field(default_factory=dict)     # {pair: name of the condition variable}
    margins: Any = None                                      # margin_net.MarginNet
    form: Any = None                                         # precision_form.PrecisionForm
    candidates: list = field(default_factory=list)           # [precision_form.Candidate]
    unlock_graph: dict | None = None                         # ANCHOR_GRAPH-shaped {"nodes": [...]}
    p_holds: dict = field(default_factory=dict)
    instruments: dict = field(default_factory=dict)          # {node id: [(cost, reliability[, name])]} for plan_value
    reports_by_edge: dict = field(default_factory=dict)      # {edge id: [report]} for hidden_variable
    probe_instruments: list = field(default_factory=lambda: [Instrument("judge", 1.0, 0.8)])
    measure_instrument: Instrument | None = None             # None = margin_net.default_sigma at cost 1

    def variable(self, pair) -> str:
        return self.regime_variables.get(pair, "x")


class Actions(list):
    """The ranked list (bits per cost). `.other` holds the channels whose value is not in bits."""

    def __init__(self, ranked=(), other=()):
        super().__init__(ranked)
        self.other = list(other)

    def by_kind(self, kind: str) -> list:
        return [a for a in self if a.kind == kind] + [a for a in self.other if a.kind == kind]


# -- candidate collection ---------------------------------------------------------------------------
def _pair_str(pair) -> str:
    return f"{pair[0]} → {pair[1]}" if isinstance(pair, (tuple, list)) and len(pair) == 2 else str(pair)


def _regime_actions(state: EngineState) -> list[Action]:
    out: list[Action] = []
    for pair, post in state.regimes.items():
        var = state.variable(pair)
        unit = BITS if getattr(post, "potential", "entropy") == "entropy" else "error_measure"
        subj = pair[0] if isinstance(pair, (tuple, list)) else str(pair)
        obj = pair[1] if isinstance(pair, (tuple, list)) and len(pair) == 2 else "?"
        for ins in state.probe_instruments:
            x, gain = post.best_probe(ins.reliability)
            out.append(Action(
                "probe", pair, ins, float(gain), float(ins.cost), float(gain) / float(ins.cost), "regime_posterior",
                f"observe the sign of d({obj})/d({subj}) at {var} = {x:g} with instrument {ins}; return {{sign: ±1}}",
                box={var: [float(x), float(x)]}, value_unit=unit, meta={"x": float(x)}))
            xm, gm = post.model_check_probe(ins.reliability)
            if gm > 0:
                out.append(Action(
                    "model_check", pair, ins, float(gm), float(ins.cost), float(gm) / float(ins.cost),
                    "regime_posterior",
                    f"observe the sign of d({obj})/d({subj}) at {var} = {xm:g} with instrument {ins}; this probe is "
                    f"bought against the assumption of at most one transition, not against the sign; return {{sign: ±1}}",
                    box={var: [float(xm), float(xm)]}, value_unit=BITS, meta={"x": float(xm), "guard": True}))
    return _best_per(out)


def _best_per(actions: list[Action]) -> list[Action]:
    """One action per (kind, target): the instrument with the largest value per cost."""
    best: dict[tuple, Action] = {}
    for a in actions:
        key = (a.kind, str(a.target))
        if key not in best or a.value_per_cost > best[key].value_per_cost:
            best[key] = a
    return list(best.values())


def _margin_actions(state: EngineState) -> list[Action]:
    net = state.margins
    if net is None:
        return []
    ins = state.measure_instrument or Instrument("measurement", 1.0, 1.0, sigma=net.default_sigma)
    sigma_new = ins.sigma if ins.sigma is not None else net.default_sigma
    out: list[Action] = []
    for eid, v in net.next_measurements(sigma_new):
        e = net.edges[eid]
        if not math.isfinite(v):                              # NO-DATA edge: measurement_value is +inf, not a number
            continue
        # measurement_value is (entropy drop) × weight / cost; the channel's bits are the drop itself
        bits = v * float(e["cost"]) / float(e["weight"]) if e["weight"] else 0.0
        cost = float(e["cost"])
        a, b = (list(e["between"]) + ["?", "?"])[:2]
        out.append(Action(
            "measure", eid, ins, float(bits), cost, float(bits) / cost if cost else 0.0, "margin_net",
            f"measure the margin on edge '{eid}' between {a} and {b} with instrument {ins} (sigma {sigma_new:g}); "
            f"return {{margin: float, sigma: float, sources: [source ids]}}",
            box=None, value_unit=BITS, meta={"sigma_new": float(sigma_new), "weight": float(e["weight"])}))
    return out


def _link_of(h) -> tuple[int, int] | None:
    nz = [(i, float(x)) for i, x in enumerate(h) if abs(float(x)) > 1e-12]
    if len(nz) == 2 and abs(nz[0][1] + nz[1][1]) < 1e-12:
        return (nz[0][0], nz[1][0]) if nz[0][1] > 0 else (nz[1][0], nz[0][0])
    return None


def _form_actions(state: EngineState) -> list[Action]:
    if state.form is None or not state.candidates:
        return []
    by_id = {c.id: c for c in state.candidates}
    out = []
    for cid, bits, per_cost, kind in state.form.rank(list(state.candidates)):
        c = by_id[cid]
        link = _link_of(c.h)
        what = (f"the contrast x[{link[0]}] − x[{link[1]}]" if link else f"the contrast '{cid}'")
        ins = Instrument(kind or "observation", float(c.cost), 1.0, sigma=float(c.sigma))
        out.append(Action(
            "throw", cid, ins, float(bits), float(c.cost), float(per_cost), "precision_form",
            f"observe {what} with instrument {ins} (sigma {c.sigma:g}); "
            f"return {{confirmed_links: [[i, j, w]]}} for the links the observation confirms",
            box=None, value_unit=BITS, meta={"link": link}))
    return out


def _hidden_variable_actions(state: EngineState) -> list[Action]:
    if not state.reports_by_edge:
        return []
    from .hidden_variable import candidates as hv_candidates
    out = []
    for eid, reports in state.reports_by_edge.items():
        for c in hv_candidates(reports):
            cost = 1.0
            out.append(Action(
                "declare_variable", c.attribute, None, float(c.score), cost, float(c.score) / cost, "hidden_variable",
                f"declare the variable '{c.attribute}' on edge '{eid}' and split its reports at {c.split!r} "
                f"({c.n_left} / {c.n_right} reports, permutation p {c.p_value:g}); "
                f"return {{attribute: '{c.attribute}'}}",
                box=dict(c.boxes), value_unit="chi2_drop", meta={"edge": eid, "split": c.split, "p_value": c.p_value}))
    return sorted(out, key=lambda a: -a.value_per_cost)


def _unlock_actions(state: EngineState) -> list[Action]:
    if not state.unlock_graph:
        return []
    from .unlock_value import rank as unlock_rank
    settle: dict[str, float] = {}
    first_instrument: dict[str, str] = {}
    if state.instruments:
        from .plan_value import plan
        p = plan(state.unlock_graph, state.p_holds, state.instruments)
        order, settle = p["order"], p["settle_cost"]
        first_instrument = {p["first_node"]: p["first_instrument"]} if p.get("first_node") else {}
    else:
        order = unlock_rank(state.unlock_graph, state.p_holds)
    out = []
    for r in order:
        if not r["actionable"]:
            continue
        nid = r["id"]
        cost = float(settle.get(nid, r["cost"]))
        ins_name = first_instrument.get(nid)
        ins = Instrument(ins_name, cost, 1.0) if ins_name else None
        out.append(Action(
            "work_node", nid, ins, float(r["score"]), cost, float(r["score"]), "unlock_value",
            f"work node '{nid}' (P(holds) {r['p_holds']:g}, settle cost {cost:g})"
            + (f" starting with instrument '{ins_name}'" if ins_name else "")
            + "; return the node's own evidence: {sign: ±1} or {margin, sigma, sources}",
            box=None, value_unit="priority", meta={"p_holds": r["p_holds"], "stake": r["stake"]}))
    return out


def _federation_actions(state: EngineState) -> list[Action]:
    fed = state.federation
    if fed is None or not getattr(fed, "claims", None):
        return []
    out = []
    for pr in fed.next_experiments():
        point = ", ".join(f"{v} = {x:g}" for v, x in sorted(pr.point.items())) or "the claims' common point"
        ins = state.probe_instruments[0] if state.probe_instruments else None
        cost = float(pr.cost)
        out.append(Action(
            "probe", tuple(pr.pair), ins, float(pr.value * cost), cost, float(pr.value), "claim_federation",
            f"observe the sign of d({pr.pair[1]})/d({pr.pair[0]}) at {point} "
            f"({pr.kind}, from claims {', '.join(pr.claims)})"
            + (f" with instrument {ins}" if ins else "") + "; return {sign: ±1}",
            box={v: [x, x] for v, x in pr.point.items()}, value_unit="evpi", meta={"probe": pr, "kind": pr.kind}))
    return out


# -- the list ---------------------------------------------------------------------------------------
def next_actions(state: EngineState, profile: Any = None, k: int = 10, guard_share: float | None = None) -> Actions:
    """Top-k actions by value per cost, in bits, with `guard_share` of the slots reserved for `model_check`
    actions when any exist. Channels whose value is not in bits (unlock_value: priority; hidden_variable: χ² drop;
    claim_federation: EVPI) are returned in `.other`, in their own order, never mixed into the ranking."""
    if k <= 0:
        raise ValueError(f"k must be ≥ 1: {k}")
    if guard_share is None:
        guard_share = float(getattr(profile, "guard_share", 0.2)) if profile is not None else 0.2
    if not 0.0 <= guard_share <= 1.0:
        raise ValueError(f"guard_share must be in [0, 1]: {guard_share}")

    collected = (_regime_actions(state) + _margin_actions(state) + _form_actions(state)
                 + _hidden_variable_actions(state) + _unlock_actions(state) + _federation_actions(state))
    in_bits = sorted((a for a in collected if a.value_unit == BITS), key=lambda a: -a.value_per_cost)
    other = [a for a in collected if a.value_unit != BITS]

    guards = [a for a in in_bits if a.kind == "model_check"]
    rest = [a for a in in_bits if a.kind != "model_check"]
    n_guard = 0
    if guards and guard_share > 0:
        n_guard = min(len(guards), max(1, int(round(guard_share * k))), k)
    chosen = guards[:n_guard] + rest[:k - n_guard]
    if len(chosen) < k:                                    # unfilled slots go to whatever is left
        taken = {id(a) for a in chosen}
        chosen += [a for a in in_bits if id(a) not in taken][:k - len(chosen)]
    return Actions(sorted(chosen, key=lambda a: -a.value_per_cost), other)


# -- the route back ---------------------------------------------------------------------------------
def _require(outcome: Any, keys: tuple[str, ...], kind: str) -> None:
    if not isinstance(outcome, dict):
        raise ValueError(f"outcome for a {kind} action must be a dict, got {type(outcome).__name__}")
    missing = [x for x in keys if x not in outcome]
    if missing:
        raise ValueError(f"outcome for a {kind} action must carry {list(keys)}; missing {missing}: {outcome}")


def _sign(outcome: dict) -> int:
    s = outcome["sign"]
    if s not in (1, -1, 1.0, -1.0):
        raise ValueError(f"sign must be +1 or -1, got {s!r}")
    return 1 if s > 0 else -1


def _family_entropy(post) -> float:
    """H(P(two transitions)): the potential `model_check_probe` buys against."""
    _, _, _, p = post._with_probes()
    return _h(float(p[post._n_one:].sum()))


def apply(state: EngineState, action: Action, outcome: dict) -> dict:
    """Route one agent's answer back into the channel that valued it, and return the ledger row.

    Row: {"action", "outcome", "consumers", "evidence", "value_predicted", "value_realized", ...} in the
    FOLD_LEDGER shape (append-only; `supersedes` empty — an outcome adds evidence, it does not invalidate a row).
    `value_realized` is the potential drop the update actually produced, in the same unit as `value_predicted`;
    None when the channel has no potential to re-read (a declaration).
    """
    if action.kind not in KINDS:
        raise ValueError(f"unknown action kind {action.kind!r}; expected one of {list(KINDS)}")
    realized: float | None = None
    extra: dict[str, Any] = {}

    if action.kind in ("probe", "model_check"):
        _require(outcome, ("sign",), action.kind)
        sign = _sign(outcome)
        if action.source_channel == "claim_federation":
            fed = state.federation
            if fed is None:
                raise ValueError("no federation in the state to record this probe against")
            src = outcome.get("source") or (action.instrument.name if action.instrument else "agent")
            extra["claim_id"] = fed.record(action.meta["probe"], sign, src)
        else:
            post = state.regimes.get(action.target)
            if post is None:
                raise ValueError(f"no regime posterior for pair {action.target!r}")
            rel = action.instrument.reliability if action.instrument else post.reliability
            pot = _family_entropy if action.kind == "model_check" else (lambda p: p.potential_value())
            before = pot(post)
            post.add_probe(action.meta["x"], sign, reliability=rel, weight=float(outcome.get("weight", 1.0)))
            realized = before - pot(post)

    elif action.kind == "measure":
        _require(outcome, ("margin", "sigma", "sources"), action.kind)
        net = state.margins
        if net is None or action.target not in net.edges:
            raise ValueError(f"no margin edge {action.target!r} in the state")
        e = net.edges[action.target]
        before = _h(net.estimate(action.target).p_violated) if e["reports"] else None
        report = {"margin": float(outcome["margin"]), "sigma": float(outcome["sigma"]),
                  "sources": list(outcome["sources"])}
        if outcome.get("validity"):
            report["validity"] = outcome["validity"]
        if outcome.get("attributes"):
            report["attributes"] = outcome["attributes"]
        net.add_edge(action.target, list(e["between"]), list(e["reports"]) + [report], e["weight"], e["cost"])
        after = _h(net.estimate(action.target).p_violated)
        realized = None if before is None else before - after

    elif action.kind == "throw":
        _require(outcome, ("confirmed_links",), action.kind)
        if state.form is None:
            raise ValueError("no precision form in the state")
        links = outcome["confirmed_links"]
        if not isinstance(links, (list, tuple)):
            raise ValueError(f"confirmed_links must be a list of (i, j, w), got {type(links).__name__}")
        realized = 0.0
        for l in links:
            if len(l) != 3:
                raise ValueError(f"each confirmed link must be (i, j, w), got {l!r}")
            i, j, w = l
            realized += float(state.form.observe_link(int(i), int(j), float(w)))

    elif action.kind == "declare_variable":
        _require(outcome, ("attribute",), action.kind)
        if outcome["attribute"] != action.target:
            raise ValueError(f"outcome declares {outcome['attribute']!r}, the action is about {action.target!r}")
        extra["boxes"] = dict(action.box or {})               # what the declaration would add to every box on the edge

    else:                                                     # work_node
        raise ValueError("a work_node outcome settles a node; route it to the channel that holds its evidence "
                         "(a probe on its pair, or a measurement on its edge), not to apply()")

    row = {"action": action.id, "outcome": action.kind, "consumers": [str(action.target)], "evidence": dict(outcome),
           "value_predicted": float(action.value_bits), "value_realized": realized,
           "value_unit": action.value_unit, "source": action.source_channel,
           "instrument": action.instrument.name if action.instrument else None, "supersedes": []}
    row.update(extra)
    return row
