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

BUNDLES, AND WHY A ONE-STEP PRICE CANNOT SEE THEM. A purchase with INCREASING returns is invisible to a price that
asks what one step is worth: the probe toward a second transition is worth ≈ 0 alone and a lot given a probe on the
other side (e33's non-submodular case). e21f measured the consequence: pairs of probes priced at their own depth find
7 of 25 collisions with 0 false flags and NO reserved share, where every one-step rule with no share found 0. A
`kind = "bundle"` action is that generalization: `members` (atomic actions bought together), `value_bits` = the EXACT
joint value of the members, `cost` = the sum of the member costs, ranked in the same bits-per-cost list as singles.
Three joint values, one per group of members:
  probes on ONE pair      the exact drop of U + λ·H_family over the 2^K joint outcomes, probe k updating the
                          posterior for probe k+1 (the chain rule of e35, which telescopes in any order). K = 2 is
                          `RegimePosterior.bundle_value`'s pair; a SWEEP (one computation cell answering K points on
                          one pair, entering as ONE lineage root, each answer with weight 1/K — `disagreement_field.
                          model_probes`) is the same computation with the tempered likelihood. Exact for K ≤ 8;
                          above it, the sum of conditionals along `n_sample` sampled outcome paths (meta["exact"]
                          is False and meta["subsample"] carries the count — say so when you report it).
  links of a CHAIN THROW  `precision_form.set_value_bits` of the rows e_i − e_j of the decoded path's links, which
                          is the joint log-det value of observing them together, not the sum of the singles.
  members on DIFFERENT    the sum of the groups' values: the beliefs are independent, so the joint value factorizes
  pairs / targets         exactly (e35 (a)).
Nothing is reserved for a bundle and nothing is reserved against one; a bundle wins a slot only by bits per cost.

`apply` updates the channel's own object and returns an append-only ledger row in the FOLD_LEDGER shape
(consumers / evidence / supersedes), with the potential drop the update actually realized next to the one that was
predicted. Nothing is written to disk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Instrument", "Action", "EngineState", "Actions", "next_actions", "apply",
           "bundle_action", "bundle_value_bits", "probe_pair_bundle", "sweep_bundle", "chain_throw_bundle"]

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


KINDS = ("probe", "measure", "throw", "declare_variable", "model_check", "work_node", "bundle")


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
    bundles: list = field(default_factory=list)              # [Action(kind="bundle")] built by the helpers below
    pair_bundles: bool = True                                # offer the exactly priced probe PAIR on every pair (e21f)
    bundle_lam: float = 1.0                                  # λ on the family entropy in a regime bundle's price

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


# -- bundles: a purchase priced at its own depth -----------------------------------------------------
def _regime_total(post, lam: float = 1.0) -> float:
    """U + λ·H(P(two transitions)) in bits: the potential a regime bundle is priced against (`bundle_value`)."""
    import numpy as np
    _, w, F, p = post._with_probes()
    u = float(post._u(p @ F) @ w)
    if len(p) <= post._n_one:
        return u
    two = np.zeros(len(p)); two[post._n_one:] = 1.0
    return u + lam * _h(float(p @ two))


def _joint_probe_gain(post, xs, rels, wts, lam: float = 1.0, max_exact: int = 8, n_sample: int = 64,
                      seed: int = 0) -> tuple[float, list, bool]:
    """(gain, outcome table, exact) for K probes bought TOGETHER on one RegimePosterior.

    gain = U₀ − E[U_K] of the total potential U + λ·H_family, the expectation taken over the 2^K joint outcomes with
    the posterior updated between probes — the chain-rule set value of e35, which telescopes to the same number in any
    order. Each probe enters with its own reliability r and weight wt exactly as `add_probe` does (likelihood^wt), so a
    SWEEP of K answers from one lineage root (wt = 1/K, `disagreement_field.model_probes`) is priced as the one answer's
    worth of evidence it is; the outcome law is that tempered likelihood normalized, which for wt = 1 is the exact
    predictive law and makes the predicted gain the expectation of the realized one (test).

    For K > `max_exact` the enumeration is replaced by the sum of conditionals along `n_sample` outcome paths sampled
    from the same law; the third return value is then False and the number must be reported as a subsample estimate.
    The table is [(signs, probability), ...] for the exact route, [] otherwise."""
    import numpy as np
    cells, w, F, p0 = post._with_probes()
    has_fam = len(p0) > post._n_one
    two = np.zeros(len(p0))
    if has_fam:
        two[post._n_one:] = 1.0

    def tot(q) -> float:
        return float(post._u(q @ F) @ w) + (lam * _h(float(q @ two)) if has_fam else 0.0)

    cix = [min(int(np.searchsorted(cells[:, 1], float(x), side="left")), len(cells) - 1) for x in xs]

    def like(k: int, sg: int):
        f = F[:, cix[k]]
        ff = f if sg > 0 else 1 - f
        return (ff * rels[k] + (1 - ff) * (1 - rels[k])) ** wts[k]

    now = tot(p0)
    K = len(xs)
    if K <= max_exact:
        table: list = []

        def walk(k: int, q, pr: float, signs: tuple):
            if k == K:
                table.append((signs, pr))
                return pr * tot(q)
            ls = [like(k, +1), like(k, -1)]
            ms = [float(q @ l) for l in ls]
            s = sum(ms)
            acc = 0.0
            for j, sg in enumerate((+1, -1)):
                if ms[j] <= 0 or s <= 0:
                    continue
                acc += walk(k + 1, q * ls[j] / ms[j], pr * ms[j] / s, signs + (sg,))
            return acc

        return now - walk(0, p0, 1.0, ()), table, True

    rng = np.random.default_rng(seed)
    after = 0.0
    for _ in range(n_sample):
        q = p0
        for k in range(K):
            ls = [like(k, +1), like(k, -1)]
            ms = [float(q @ l) for l in ls]
            s = sum(ms)
            j = 0 if (s <= 0 or rng.random() < ms[0] / s) else 1
            if ms[j] > 0:
                q = q * ls[j] / ms[j]
        after += tot(q)
    return now - after / n_sample, [], False


def _links_of(members) -> list[tuple[int, int, float]]:
    out = []
    for m in members:
        link = m.meta.get("link")
        if link is None or len(link) < 2:
            raise ValueError(f"a throw member of a bundle must carry meta['link'] = (i, j[, w]): {m.id}")
        i, j = int(link[0]), int(link[1])
        out.append((i, j, float(link[2]) if len(link) > 2 else float(m.meta.get("w", 1.0))))
    return out


def _check_members(state: EngineState, members) -> list[tuple[tuple, list]]:
    """Validate and group the members: [((group kind, target), [members])]. Raises on a malformed bundle."""
    if not isinstance(members, (list, tuple)) or len(members) == 0:
        raise ValueError("a bundle must carry a non-empty list of member actions")
    groups: dict[tuple, list] = {}
    for m in members:
        if not isinstance(m, Action):
            raise ValueError(f"every bundle member must be an Action, got {type(m).__name__}")
        if m.kind == "bundle":
            raise ValueError(f"a bundle cannot contain a bundle: {m.id}")
        if m.kind in ("probe", "model_check"):
            if m.source_channel == "claim_federation":
                raise ValueError(f"a federation probe is valued in EVPI and cannot enter a bits bundle: {m.id}")
            if m.target not in state.regimes:
                raise ValueError(f"no regime posterior for pair {m.target!r} in the state")
            if "x" not in m.meta:
                raise ValueError(f"a probe member of a bundle must carry meta['x']: {m.id}")
            groups.setdefault(("regime", m.target), []).append(m)
        elif m.kind == "throw":
            if state.form is None:
                raise ValueError("no precision form in the state to value a throw bundle against")
            groups.setdefault(("form", None), []).append(m)
        else:
            raise ValueError(f"a bundle member must be a probe, a model_check or a throw, got {m.kind!r}: {m.id}")
    return list(groups.items())


def bundle_value_bits(state: EngineState, members, lam: float | None = None, max_exact: int = 8,
                      n_sample: int = 64) -> tuple[float, dict]:
    """(joint value in bits, meta) of buying `members` together — the EXACT joint value, not the sum of the singles.

    Probes on one pair are valued by `_joint_probe_gain` (the 2^K-outcome chain rule on U + λ·H_family), the links of
    a chain throw by `precision_form.set_value_bits` of their rows, and members on DIFFERENT targets by the sum of
    their groups: independent beliefs, so the joint value factorizes exactly (e35 (a)). meta carries "members"
    (the member value ALONE, in the same convention, one per member), "groups", "exact" and "subsample"."""
    import numpy as np
    lam = state.bundle_lam if lam is None else float(lam)
    groups = _check_members(state, members)
    total, exact, per_group = 0.0, True, []
    for (gk, target), ms in groups:
        if gk == "regime":
            post = state.regimes[target]
            xs = [float(m.meta["x"]) for m in ms]
            rels = [float(m.instrument.reliability) if m.instrument else float(post.reliability) for m in ms]
            wts = [float(m.meta.get("weight", 1.0)) for m in ms]
            g, table, ex = _joint_probe_gain(post, xs, rels, wts, lam, max_exact, n_sample)
            law = [(sgs, float(pr)) for sgs, pr in table]
        else:
            law = []
            links = _links_of(ms)
            H = np.zeros((len(links), state.form.d))
            sig = np.empty(len(links))
            for r, (i, j, w) in enumerate(links):
                H[r, i], H[r, j], sig[r] = 1.0, -1.0, float(w) ** -0.5
            g, ex = float(state.form.set_value_bits(H, sig)), True
        total += float(g)
        exact = exact and ex
        per_group.append({"target": target, "kind": gk, "bits": float(g), "n": len(ms), "outcome_law": law})

    alone = [total] if len(members) == 1 else [
        bundle_value_bits(state, [m], lam=lam, max_exact=max_exact, n_sample=n_sample)[0] for m in members]
    out = {"members": alone, "groups": per_group, "exact": exact,
           "subsample": None if exact else int(n_sample), "lam": float(lam)}
    if len(per_group) == 1 and per_group[0]["outcome_law"]:   # the joint outcome law the price is an expectation over
        out["outcome_law"] = per_group[0]["outcome_law"]
    return float(total), out


def bundle_action(state: EngineState, members, cost: float | None = None, how: str | None = None,
                  lam: float | None = None, source_channel: str = "next_actions", label: str = "bundle",
                  max_exact: int = 8, n_sample: int = 64, meta: dict | None = None) -> Action:
    """One `kind = "bundle"` Action: `members` bought together, valued jointly, cost = Σ member costs by default
    (pass `cost` for a purchase with its own price — a sweep is one computation cell, not K judge calls)."""
    bits, m = bundle_value_bits(state, members, lam=lam, max_exact=max_exact, n_sample=n_sample)
    c = float(sum(float(x.cost) for x in members)) if cost is None else float(cost)
    if c <= 0:
        raise ValueError(f"a bundle's cost must be > 0: {c}")
    targets = []
    for x in members:
        if str(x.target) not in targets:
            targets.append(str(x.target))
    ins = members[0].instrument
    if any((x.instrument.name if x.instrument else None) != (ins.name if ins else None) for x in members):
        ins = None
    m.update({"members": m["members"], "bundle": label, "n_members": len(members)})
    if meta:
        m.update(meta)
    m["member_actions"] = list(members)
    if how is None:
        how = (f"buy these {len(members)} actions TOGETHER on {', '.join(targets)} "
               + "; ".join(f"({i + 1}) {x.how.split(';')[0].strip()}" for i, x in enumerate(members))
               + f" — priced jointly at {bits:g} bits for cost {c:g}"
               + ("" if m["exact"] else f" (subsample of {m['subsample']} outcome paths)")
               + "; return {outcomes: [one outcome per member, in this order]}")
    if not m["exact"] and "subsample" not in how:
        head, _, tail = how.partition("; return ")
        how = f"{head} (value {bits:g} bits, a subsample of {m['subsample']} outcome paths, not the 2^K enumeration)" \
              + ("; return " + tail if tail else "")
    return Action("bundle", " + ".join(targets), ins, float(bits), c, float(bits) / c, source_channel,
                  " ".join(how.split()), box=None, value_unit=BITS, meta=m)


def probe_pair_bundle(state: EngineState, pair, instrument: Instrument | None = None, lam: float | None = None,
                      top: int = 12) -> Action | None:
    """The exactly priced PROBE PAIR on one pair (e21f): `RegimePosterior.bundle_value`'s argmax over pairs of cells,
    as a bundle of two probe members at cost 2 × the instrument's. None when the pair has no pair to offer."""
    post = state.regimes.get(pair)
    if post is None:
        raise ValueError(f"no regime posterior for pair {pair!r}")
    ins = instrument or (state.probe_instruments[0] if state.probe_instruments else Instrument("judge"))
    lam = state.bundle_lam if lam is None else float(lam)
    bv = post.bundle_value(ins.reliability, k=2, lam=lam, top=top)
    if not bv.get("pair"):
        return None
    (x1, x2), gain, _n = bv["pair"]
    var = state.variable(pair)
    subj = pair[0] if isinstance(pair, (tuple, list)) else str(pair)
    obj = pair[1] if isinstance(pair, (tuple, list)) and len(pair) == 2 else "?"
    members = []
    for x in (x1, x2):
        g1, _t, _e = _joint_probe_gain(post, [x], [ins.reliability], [1.0], lam)
        members.append(Action("probe", pair, ins, float(g1), float(ins.cost), float(g1) / float(ins.cost),
                              "regime_posterior",
                              f"observe the sign of d({obj})/d({subj}) at {var} = {x:g} with instrument {ins}; "
                              f"return {{sign: ±1}}",
                              box={var: [float(x), float(x)]}, meta={"x": float(x), "weight": 1.0}))
    how = (f"buy BOTH probes on {_pair_str(pair)} together with instrument {ins}: the sign of d({obj})/d({subj}) at "
           f"{var} = {x1:g} and at {var} = {x2:g}, priced by their exact two-step value over the four joint outcomes "
           f"({gain:g} bits, cost {2 * ins.cost:g}) — a second transition is only exposed by probes on both sides of "
           f"it, which no one-step price can see; return {{outcomes: [{{sign: ±1}}, {{sign: ±1}}]}}")
    a = bundle_action(state, members, cost=2.0 * float(ins.cost), how=how, lam=lam,
                      source_channel="regime_posterior", label="probe_pair",
                      meta={"pair": pair, "x": [float(x1), float(x2)], "channel_gain": float(gain)})
    return a


def sweep_bundle(state: EngineState, pair, xs, instrument: Instrument, lam: float | None = None,
                 max_exact: int = 8, n_sample: int = 64) -> Action:
    """A SWEEP: one computation cell that answers K points on one pair at the instrument's reliability and enters as
    ONE lineage root — each answer with weight 1/K (`disagreement_field.model_probes`: resample the cell and it still
    tells you one thing, so N_eff = 1). Cost is the cell's own cost (the instrument's), not K probe costs."""
    post = state.regimes.get(pair)
    if post is None:
        raise ValueError(f"no regime posterior for pair {pair!r}")
    xs = [float(x) for x in xs]
    if not xs:
        raise ValueError("a sweep must answer at least one point")
    K = len(xs)
    var = state.variable(pair)
    subj = pair[0] if isinstance(pair, (tuple, list)) else str(pair)
    obj = pair[1] if isinstance(pair, (tuple, list)) and len(pair) == 2 else "?"
    members = []
    for x in xs:
        g1, _t, _e = _joint_probe_gain(post, [x], [instrument.reliability], [1.0 / K], lam if lam is not None
                                       else state.bundle_lam)
        members.append(Action("probe", pair, instrument, float(g1), float(instrument.cost) / K,
                              float(g1) * K / float(instrument.cost), "regime_posterior",
                              f"observe the sign of d({obj})/d({subj}) at {var} = {x:g} with instrument {instrument}; "
                              f"return {{sign: ±1}}",
                              box={var: [float(x), float(x)]}, meta={"x": float(x), "weight": 1.0 / K}))
    how = (f"run the sweep {instrument} once on {_pair_str(pair)}: the sign of d({obj})/d({subj}) at {var} = "
           + ", ".join(f"{x:g}" for x in xs)
           + f" — K = {K} answers from ONE lineage root, each entering with weight 1/{K}; "
             f"return {{outcomes: [{{sign: ±1}} × {K}]}}")
    return bundle_action(state, members, cost=float(instrument.cost), how=how, lam=lam,
                         source_channel="regime_posterior", label="sweep", max_exact=max_exact, n_sample=n_sample,
                         meta={"pair": pair, "x": xs, "K": K, "weight": 1.0 / K})


def chain_throw_bundle(state: EngineState, nodes, w: float = 1.0, cost_per_link: float = 1.0,
                       instrument: Instrument | None = None) -> Action:
    """A CHAIN THROW (`throws.chain_throw`: the decoded path a → · → · → b) as a bundle of its links, valued by
    `precision_form.set_value_bits` of the rows e_i − e_j — the joint log-det value of the path, which is not the sum
    of the links' own values."""
    nodes = [int(n) for n in nodes]
    if len(nodes) < 2:
        raise ValueError(f"a chain throw needs at least two nodes: {nodes}")
    if state.form is None:
        raise ValueError("no precision form in the state to value a chain throw against")
    ins = instrument or Instrument("chain_throw", float(cost_per_link), 1.0, sigma=float(w) ** -0.5)
    members = []
    for i, j in zip(nodes[:-1], nodes[1:]):
        gi = float(state.form.value_bits(_link_row(state.form.d, i, j), float(w) ** -0.5))
        members.append(Action("throw", f"link:{i}-{j}", ins, gi, float(cost_per_link),
                              gi / float(cost_per_link), "precision_form",
                              f"observe the contrast x[{i}] − x[{j}] with instrument {ins}; "
                              f"return {{confirmed_links: [[{i}, {j}, {w:g}]]}}",
                              meta={"link": (i, j, float(w))}))
    how = ("observe the whole chain " + " → ".join(str(n) for n in nodes)
           + f" with instrument {ins}: its {len(members)} links together, priced by the joint log-det value of their "
             f"rows; return {{outcomes: [{{confirmed_links: [[i, j, w]]}} per link, in path order]}}")
    return bundle_action(state, members, cost=float(cost_per_link) * len(members), how=how,
                         source_channel="precision_form", label="chain_throw",
                         meta={"nodes": nodes, "w": float(w)})


def _link_row(d: int, i: int, j: int):
    import numpy as np
    v = np.zeros(int(d)); v[int(i)], v[int(j)] = 1.0, -1.0
    return v


def _bundle_actions(state: EngineState) -> list[Action]:
    out: list[Action] = []
    if state.pair_bundles:
        for pair in state.regimes:
            for ins in (state.probe_instruments or [Instrument("judge")]):
                a = probe_pair_bundle(state, pair, ins)
                if a is not None and a.value_bits > 0:
                    out.append(a)
    out = _best_per(out)
    return out + [a for a in state.bundles if isinstance(a, Action)]


# -- the list ---------------------------------------------------------------------------------------
def next_actions(state: EngineState, profile: Any = None, k: int = 10, guard_share: float | None = None,
                 decisions: Any = None) -> Actions:
    """Top-k actions by value per cost, in bits, with `guard_share` of the slots reserved for `model_check`
    actions when any exist. Channels whose value is not in bits (unlock_value: priority; hidden_variable: χ² drop;
    claim_federation: EVPI) are returned in `.other`, in their own order, never mixed into the ranking.

    `decisions` (a list of decision_cert.Decision) adds "flip" actions to `.other`, ranked by P(the outcome flips
    that decision's certificate) / cost — the measurement that can change a DECISION, which is not in general the
    one with the largest entropy drop. Their value_unit is "p_flip", so they never enter the bits ranking."""
    if k <= 0:
        raise ValueError(f"k must be ≥ 1: {k}")
    if guard_share is None:
        guard_share = float(getattr(profile, "guard_share", 0.2)) if profile is not None else 0.2
    if not 0.0 <= guard_share <= 1.0:
        raise ValueError(f"guard_share must be in [0, 1]: {guard_share}")

    collected = (_regime_actions(state) + _margin_actions(state) + _form_actions(state)
                 + _hidden_variable_actions(state) + _unlock_actions(state) + _federation_actions(state)
                 + _bundle_actions(state))
    in_bits = sorted((a for a in collected if a.value_unit == BITS), key=lambda a: -a.value_per_cost)
    other = [a for a in collected if a.value_unit != BITS]
    if decisions:
        from .decision_cert import flip_actions
        other = other + flip_actions(state, decisions)

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

    elif action.kind == "bundle":
        _require(outcome, ("outcomes",), action.kind)
        members = action.meta.get("member_actions") or []
        outs = outcome["outcomes"]
        if not isinstance(outs, (list, tuple)) or len(outs) != len(members):
            raise ValueError(f"a bundle outcome must carry one outcome per member, in order: "
                             f"{len(members)} expected, got {outs!r}")
        lam = float(action.meta.get("lam", 1.0))
        posts = {str(m.target): state.regimes[m.target] for m in members if m.kind in ("probe", "model_check")}
        before = sum(_regime_total(p, lam) for p in posts.values())
        realized = 0.0
        for m, o in zip(members, outs):
            if m.kind in ("probe", "model_check"):
                _require(o, ("sign",), m.kind)
                post = state.regimes[m.target]
                rel = m.instrument.reliability if m.instrument else post.reliability
                wt = float(o.get("weight", m.meta.get("weight", 1.0)))
                post.add_probe(m.meta["x"], _sign(o), reliability=rel, weight=wt)
            else:                                             # throw: the link the member names, confirmed or not
                _require(o, ("confirmed_links",), m.kind)
                for l in o["confirmed_links"]:
                    if len(l) != 3:
                        raise ValueError(f"each confirmed link must be (i, j, w), got {l!r}")
                    realized += float(state.form.observe_link(int(l[0]), int(l[1]), float(l[2])))
        realized += before - sum(_regime_total(p, lam) for p in posts.values())
        extra["members"] = [m.id for m in members]

    elif action.kind == "declare_variable":
        _require(outcome, ("attribute",), action.kind)
        if outcome["attribute"] != action.target:
            raise ValueError(f"outcome declares {outcome['attribute']!r}, the action is about {action.target!r}")
        extra["boxes"] = dict(action.box or {})               # what the declaration would add to every box on the edge

    else:                                                     # work_node
        raise ValueError("a work_node outcome settles a node; route it to the channel that holds its evidence "
                         "(a probe on its pair, or a measurement on its edge), not to apply()")

    consumers = [str(action.target)]
    if action.kind == "bundle":
        consumers = list(dict.fromkeys(str(m.target) for m in action.meta.get("member_actions") or []))
    row = {"action": action.id, "outcome": action.kind, "consumers": consumers, "evidence": dict(outcome),
           "value_predicted": float(action.value_bits), "value_realized": realized,
           "value_unit": action.value_unit, "source": action.source_channel,
           "instrument": action.instrument.name if action.instrument else None, "supersedes": []}
    row.update(extra)
    return row
