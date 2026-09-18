#!/usr/bin/env python3
"""
profile.py — one place for every assumption a graph's owner makes, validated, handed to the modules.

The modules read one data contract (claims = pair + sign + validity box + sources with lineage; margins = value + σ +
sources; graphs = nodes with depends_on) and know nothing about the domain. What they DO carry is about forty numeric
defaults — claim reliability 0.75, prior on a regime transition 0.3, stressed below z = 2, disagreement at p < 0.01,
admission level τ = 0.9, guarantee level α = 0.1 … — set from synthetic experiments (RESULTS.md) and spread over ten
modules. A graph with other requirements (a clinical net that wants z = 3 and p < 0.001; a materials net whose sources
are reliable and rarely copied; a corpus in another language with its own direction lexicon) had to find them one by
one. e21 measured what an unseen default costs: the loop believed 0.97 wrongly signed and had 1.20, because every claim
entered at the fixed 0.75.

An EngineProfile holds them together, validates them once, and builds the modules:

    prof = EngineProfile(claim_reliability=0.85, stressed_below_z=3.0, disagree_p=0.001)
    prof.validate()
    fed = prof.federation();  net = prof.margin_net();  rp = prof.regime_posterior(lo, hi)

Every field keeps the module's current default, so nothing changes for code that does not use a profile. Fields are
grouped by which module reads them; `describe()` prints them with the module that consumes each. `to_json` / `from_json`
so a graph can ship its profile next to its data. The profile is not a config language: it holds numbers and a lexicon,
never code paths.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

__all__ = ["EngineProfile"]


@dataclass
class EngineProfile:
    # -- claims and their origins (claim_federation, regime_posterior, source_reliability) ---------
    claim_reliability: float = 0.75          # prior P(an independent origin reports the true sign)
    transitivity: float = 0.6                # prior P(a two-step chain composes); scores inferred links only
    use_lineage: bool = True                 # copies of one origin are one origin
    use_validity: bool = True                # validity boxes bound what a claim says
    reliability_by_root: dict = field(default_factory=dict)   # per-origin overrides, e.g. from source_reliability.estimate
    reliability_prior: tuple = (3.0, 1.5)    # Beta prior for source_reliability.estimate
    # -- regimes along a condition variable (regime_posterior) ------------------------------------
    p_transition: float = 0.3                # prior P(the sign changes once inside the domain)
    p_two_transitions: float = 0.05          # prior P(twice): the collision family; 0 = assume it away
    regime_grid: int = 48                    # fixed cells; probes never add cells
    potential: str = "entropy"               # "entropy" or "error" (= min(p, 1−p), piecewise linear)
    guard_share: float = 0.2                 # share of a probing budget spent against the model error (closed_loop)
    # -- margins (margin_net) --------------------------------------------------------------------
    default_sigma: float = 0.1
    stressed_below_z: float = 2.0
    disagree_p: float = 0.01
    copy_tolerance: float = 0.5              # declared copies may differ by this many σ
    # -- admission and guarantees (plan_value, record_guarantee) ---------------------------------
    admission_tau: float = 0.9               # a node is settled when P ≥ τ or ≤ 1 − τ
    guarantee_alpha: float = 0.1             # ≥ 1 − α of admitted records entirely right
    guarantee_delta: float = 0.05            # the guarantee holds with probability ≥ 1 − δ
    guarantee_grid: int = 20
    # -- throws (throws) ---------------------------------------------------------------------------
    throw_temperature: float = 1.0
    throw_floor: float = 0.05                # minimum inclusion probability, keeps 1/π weights bounded
    # -- reading text (polarity_rules, lens_pooling, typed_extraction) ---------------------------
    direction_lexicon: dict | None = None    # {"up": [...], "down": [...]} to replace the English lexicon; None = default
    abstain_below: float = 0.6               # extraction confidence under which the pipeline ABSTAINS
    # -- the graph's own declaration ---------------------------------------------------------------
    log_scale_variables: list = field(default_factory=list)   # variables whose boxes are in log units
    notes: str = ""

    # -- validation ---------------------------------------------------------------------------------
    def validate(self) -> "EngineProfile":
        def between(name, lo, hi, closed_lo=True, closed_hi=True):
            v = getattr(self, name)
            ok = (lo <= v if closed_lo else lo < v) and (v <= hi if closed_hi else v < hi)
            if not ok:
                raise ValueError(f"{name} = {v} must be in {'[' if closed_lo else '('}{lo}, {hi}{']' if closed_hi else ')'}")
        between("claim_reliability", 0.5, 1.0, closed_lo=False, closed_hi=False)
        between("transitivity", 0.0, 1.0)
        between("p_transition", 0.0, 1.0)
        between("p_two_transitions", 0.0, 1.0)
        if self.p_transition + self.p_two_transitions > 1.0:
            raise ValueError("p_transition + p_two_transitions must be ≤ 1")
        if self.regime_grid < 4:
            raise ValueError("regime_grid must be ≥ 4")
        if self.potential not in ("entropy", "error"):
            raise ValueError("potential must be 'entropy' or 'error'")
        between("guard_share", 0.0, 1.0)
        if self.default_sigma <= 0:
            raise ValueError("default_sigma must be > 0 (a report without uncertainty cannot be combined)")
        if self.stressed_below_z < 0:
            raise ValueError("stressed_below_z must be ≥ 0")
        between("disagree_p", 0.0, 1.0, closed_lo=False, closed_hi=False)
        if self.copy_tolerance <= 0:
            raise ValueError("copy_tolerance must be > 0")
        between("admission_tau", 0.5, 1.0, closed_lo=False, closed_hi=False)
        between("guarantee_alpha", 0.0, 1.0, closed_lo=False, closed_hi=False)
        between("guarantee_delta", 0.0, 1.0, closed_lo=False, closed_hi=False)
        if self.guarantee_grid < 2:
            raise ValueError("guarantee_grid must be ≥ 2")
        if self.throw_temperature <= 0:
            raise ValueError("throw_temperature must be > 0")
        between("throw_floor", 0.0, 1.0)
        between("abstain_below", 0.0, 1.0)
        for r, v in self.reliability_by_root.items():
            if not 0.0 < v < 1.0:
                raise ValueError(f"reliability_by_root[{r!r}] = {v} must be in (0, 1)")
        a, b = self.reliability_prior
        if a <= 0 or b <= 0:
            raise ValueError("reliability_prior must be positive")
        if self.direction_lexicon is not None:
            if set(self.direction_lexicon) != {"up", "down"} or not all(isinstance(w, str) for k in ("up", "down") for w in self.direction_lexicon[k]):
                raise ValueError("direction_lexicon must be {'up': [words], 'down': [words]}")
            if set(self.direction_lexicon["up"]) & set(self.direction_lexicon["down"]):
                raise ValueError("a word cannot be in both 'up' and 'down'")
        return self

    # -- builders ------------------------------------------------------------------------------------
    def federation(self, **kw):
        from .claim_federation import Federation
        return Federation(reliability=self.claim_reliability, transitivity=self.transitivity, use_lineage=self.use_lineage,
                          use_validity=self.use_validity, reliability_by_root=dict(self.reliability_by_root), **kw)

    def margin_net(self, **kw):
        from .margin_net import MarginNet
        return MarginNet(default_sigma=self.default_sigma, stressed_below_z=self.stressed_below_z, disagree_p=self.disagree_p,
                         copy_tolerance=self.copy_tolerance, **kw)

    def regime_posterior(self, lo: float, hi: float, **kw):
        from .regime_posterior import RegimePosterior
        return RegimePosterior(lo, hi, reliability=self.claim_reliability, p_flip=self.p_transition, potential=self.potential,
                               n_grid=self.regime_grid, p_two=self.p_two_transitions, **kw)

    def guarantee_threshold(self, scores, all_right):
        from .record_guarantee import fit_threshold
        return fit_threshold(scores, all_right, alpha=self.guarantee_alpha, delta=self.guarantee_delta, n_grid=self.guarantee_grid)

    def lexicon(self) -> dict:
        """The direction lexicon in the form polarity_rules uses (its default when none is declared)."""
        from . import polarity_rules as pr
        if self.direction_lexicon is None:
            return {"up": sorted(pr.UP), "down": sorted(pr.DOWN)}
        return {"up": sorted(self.direction_lexicon["up"]), "down": sorted(self.direction_lexicon["down"])}

    # -- estimation from data, and merging two graphs' profiles ------------------------------------------
    ESTIMABLE = ("claim_reliability", "reliability_by_root", "p_transition")
    REQUIREMENTS = ("stressed_below_z", "disagree_p", "admission_tau", "guarantee_alpha", "guarantee_delta", "copy_tolerance")
    # On a conflict between two graphs the value that LEAVES MORE OPEN governs: settles fewer nodes, averages away fewer
    # disagreements, trusts less. A first draft called this "stricter" and took the LOWER disagreement p — which flags fewer
    # pairs as contradictions and averages the rest, i.e. concludes more. Wrong direction; a contradiction is a node to
    # expand, not an alarm to suppress. So disagree_p takes the higher value.
    LEAVES_OPEN = {"stressed_below_z": max, "disagree_p": max, "admission_tau": max, "guarantee_alpha": min, "guarantee_delta": min,
                   "copy_tolerance": min, "claim_reliability": min, "abstain_below": max, "p_two_transitions": max}

    def estimate_from(self, fed, known: dict | None = None, min_questions: int = 20) -> dict:
        """Fill the ESTIMABLE fields from a claim_federation.Federation: per-root reliability from agreement
        (source_reliability, with `known` truths to break its mirror symmetry), the default claim reliability as their
        median, and p_transition as the share of pairs whose claims carry both signs. Requirements are never touched.
        Returns {field: (old, new)} for what changed; nothing changes when there are fewer than `min_questions`."""
        from .source_reliability import estimate, root_votes
        import statistics
        votes = root_votes(fed)
        changed = {}
        if len(votes) >= min_questions:
            est = estimate(votes, prior=self.reliability_prior, known=known)
            rb = {r: min(max(e["r"], 0.5), 0.98) for r, e in est.items()}
            if rb:
                changed["reliability_by_root"] = (dict(self.reliability_by_root), rb); self.reliability_by_root = rb
                med = float(statistics.median(rb.values()))
                changed["claim_reliability"] = (self.claim_reliability, med); self.claim_reliability = med
        signs: dict[tuple, set] = {}
        for c in fed.claims.values():
            signs.setdefault(c["pair"], set()).add(c["sign"])
        if len(signs) >= min_questions:
            pt = sum(len(v) == 2 for v in signs.values()) / len(signs)
            changed["p_transition"] = (self.p_transition, pt); self.p_transition = pt
        self.validate()
        return changed

    def merge(self, other: "EngineProfile") -> tuple["EngineProfile", list[dict]]:
        """Profile for the federation of two graphs. Per-root reliabilities and log-scale variables are unioned (a root
        keeps its own number; the same root declared twice with different numbers is a conflict, the lower wins).
        Requirements are taken per field by the LEAVES_OPEN rule and every difference is reported, so the owner sees which
        requirement now governs a question that used to be governed by the other graph's. Lexicons are unioned; a word
        that is 'up' in one and 'down' in the other is a conflict and is dropped from both."""
        m = EngineProfile(**{f.name: getattr(self, f.name) for f in fields(self)})
        conflicts = []
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if f.name == "reliability_by_root":
                out = dict(a)
                for r, v in b.items():
                    if r in out and out[r] != v:
                        conflicts.append({"field": f.name, "root": r, "values": (out[r], v), "took": min(out[r], v)})
                    out[r] = min(out[r], v) if r in out else v
                m.reliability_by_root = out
            elif f.name == "log_scale_variables":
                m.log_scale_variables = sorted(set(a) | set(b))
            elif f.name == "direction_lexicon":
                if a is None and b is None:
                    continue
                la, lb = self.lexicon(), other.lexicon()
                up, down = set(la["up"]) | set(lb["up"]), set(la["down"]) | set(lb["down"])
                both = up & down
                if both:
                    conflicts.append({"field": f.name, "words": sorted(both), "took": "dropped from both"})
                m.direction_lexicon = {"up": sorted(up - both), "down": sorted(down - both)}
            elif f.name == "notes":
                m.notes = "; ".join(x for x in (a, b) if x)
            elif f.name in self.LEAVES_OPEN:
                if a != b:
                    took = self.LEAVES_OPEN[f.name](a, b)
                    conflicts.append({"field": f.name, "values": (a, b), "took": took}); setattr(m, f.name, took)
            elif a != b:
                conflicts.append({"field": f.name, "values": (a, b), "took": a})
        m.validate()
        return m, conflicts

    # -- inspection and transport --------------------------------------------------------------------
    READERS = {
        "claim_reliability": "claim_federation, regime_posterior", "transitivity": "claim_federation", "use_lineage": "claim_federation",
        "use_validity": "claim_federation", "reliability_by_root": "claim_federation", "reliability_prior": "source_reliability",
        "p_transition": "regime_posterior", "p_two_transitions": "regime_posterior", "regime_grid": "regime_posterior",
        "potential": "regime_posterior", "guard_share": "closed_loop", "default_sigma": "margin_net", "stressed_below_z": "margin_net",
        "disagree_p": "margin_net", "copy_tolerance": "margin_net", "admission_tau": "plan_value", "guarantee_alpha": "record_guarantee",
        "guarantee_delta": "record_guarantee", "guarantee_grid": "record_guarantee", "throw_temperature": "throws", "throw_floor": "throws",
        "direction_lexicon": "polarity_rules", "abstain_below": "paper_graph.pipeline", "log_scale_variables": "claim_federation (boxes)",
        "notes": "—",
    }

    def describe(self) -> str:
        d = EngineProfile()
        rows = []
        for f in fields(self):
            v, dv = getattr(self, f.name), getattr(d, f.name)
            mark = "" if v == dv else "   ← changed from " + repr(dv)
            rows.append(f"{f.name:22s} {v!r:<40} {self.READERS.get(f.name, '?')}{mark}")
        return "\n".join(rows)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "EngineProfile":
        d = json.loads(s)
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown profile fields: {sorted(unknown)}")
        if "reliability_prior" in d:
            d["reliability_prior"] = tuple(d["reliability_prior"])
        return cls(**d).validate()
