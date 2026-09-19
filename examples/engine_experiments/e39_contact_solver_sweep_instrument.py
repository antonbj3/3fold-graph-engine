#!/usr/bin/env python3
"""
e39 — the cells-are-instruments contract on measured contact-solver data: which sweep does the loop buy?

WHAT IS MEASURED. `instruments.Instrument` says an instrument has a cost, a reliability, one lineage root and a
`probe(pair, x) -> (sign, p, cost)` reduction. `instruments.ContactSolverSweep` is the first instance: a sweep
over solver settings on a scene family, whose claim is "method A converges faster than method B on regime R",
with the sign and the probability read off the MEASURED iteration ratios and their spread. This script wires it
to `next_actions` and runs the loop.

THREE FAMILIES, ALL RECORDED, NOTHING RE-RUN (`e39_contact_solver_data.json`):

    router     21 benchmark scenes, plain PGS against the multi-contact ADMM ladder, in sweep-equivalents.
               Condition variable: log10(n_c). Known answer: the 4 scenes the router misroutes (> 3x its own
               per-scene oracle) — cube, and three small-n_c floating-base robot stances.
    rho        14 scenes x a 10-point rho ladder, iterations at rho against iterations at rho_B = sqrt(lmin*lmax).
               Condition variable: log10(rho / rho_B). Known answer: on the three jammed scenes where rho_B does
               not converge, the measured optimum sits at 2.3-3.1 rho_B.
    channels   448 scenes x 5 solver channels. The exact-cone channel against one fixed anchor channel, scored
               against the analytic slip/tipping anchor. Condition variable: log10(mass ratio). Known answer: the
               39 scenes where the exact channel loses.

THE LOOP. `next_actions` does the ranking: every candidate is put on `EngineState.bundles` and the engine
returns its own bits-per-cost list, from which the top action is bought. Each pair's posterior is seeded with the PUBLISHED RULE as one box claim (the router's default path,
h18's rho_B, "the exact cone is closer to the anchor") — what is believed before any sweep is bought. Candidates
are then, per round: one sweep per axis window (the cell restricted to the scenes in that window, priced through
`next_actions.sweep_bundle` at the window's own recorded wall cost), and one single solve at each pair's
currently most-informative point (the same code path at K = 1, at one scene's cost). They are ranked in ONE
bits-per-cost list across all three pairs. The top action is bought, the instrument supplies the outcome from
the recorded measurements, `next_actions.apply` routes it back and returns the ledger row with the realized
potential drop beside the predicted one.

THE HIT RATE. A KNOWN-ANSWER WINDOW is one whose density of known answers is above the family's own overall
rate: "contains at least one" is worthless where 39 lost scenes are spread over all 8 windows. A purchase hits
when it lands in such a window. The base rate is the share of non-empty windows that qualify — what a window
picked at random scores. Two controls: COST-BLIND ranks the same candidates by bits alone, and RANDOM picks
uniformly among them. Reported beside the pooled rate, because a cross-family bits-per-cost list is dominated by
whichever family's measurements were cheapest: the rank of each family's known-answer window INSIDE its own
first-round list, which does not depend on that cost scale.

WHAT THIS DOES NOT SHOW. The instrument is calibrated on the same recorded campaign it then reports from, so the
reliability is in-sample; the hit rate is against known answers from those same campaigns, so it measures
whether the engine's price points at the places the measurements already know are hard, not whether it would
find a new one. Every claim enters under one lineage root per cell; the shared-error state (same solver binary
across windows) is not exercised here.

    python3 examples/engine_experiments/e39_contact_solver_sweep_instrument.py
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from graph_engine.instruments import ContactSolverSweep, SweepFamily          # noqa: E402
from graph_engine.next_actions import EngineState, apply, next_actions        # noqa: E402
from graph_engine.regime_posterior import RegimePosterior                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "e39_contact_solver_data.json")
OUT = os.path.join(HERE, "e39_results.json")

ROUNDS = 12
K_MAX = 8                        # one purchase answers at most 8 points: the regime sweep_bundle prices EXACTLY
SEED_RELIABILITY = 0.75          # a published rule is a claim, not a measurement
CAP_SWEEPS = 30000.0             # a run that did not converge inside its cap is recorded AT the cap (censored)


# -- the three families ------------------------------------------------------------------------------
def router_family(d):
    """PGS against the ADMM ladder over log10(n_c). A = plain PGS, B = the multi-contact ADMM ladder."""
    rows, facit = [], set()
    for r in d["router"]["scenes"]:
        ms = [v for v in (r.get("pgs_ms"), r.get("admm_ms")) if v is not None and math.isfinite(v)]
        cost = sum(ms) if ms else float(r["router_ms"])
        rows.append(SweepFamily(label=r["scene"], x=math.log10(r["n_c"]),
                                a=min(float(r["pgs_sweeps"]), CAP_SWEEPS),
                                b=min(float(r["admm_sweeps"]), CAP_SWEEPS),
                                cost=max(cost, 1e-3), tag=r["router_path"]))
        if r["misrouted"]:
            facit.add(r["scene"])
    return rows, facit, ("contact_count", "pgs_advantage_over_admm_ladder"), "log10_n_c", -1


def rho_family(d):
    """One ladder point against rho_B, over log10(rho / rho_B). A = the ladder point, B = rho_B."""
    rows, facit = [], set()
    best_of = {}
    for scene, v in d["rho_ladders"].items():
        it_b = float(v["it_B"]) if v["conv_B"] else float(v["cap"])
        conv = [(p, float(p["iters"])) for p in v["sweep"] if p["conv"]]
        if conv:
            best_of[scene] = min(conv, key=lambda t: t[1])
        for p in v["sweep"]:
            it = float(p["iters"]) if p["conv"] else float(v["cap"])
            rows.append(SweepFamily(label=f"{scene}@j{p['rho']:.4g}", x=math.log10(p["rho"] / v["rho_B"]),
                                    a=max(it, 1.0), b=max(it_b, 1.0),
                                    cost=max(float(p["wall"]) * 1e3, 1e-3), tag=scene))
    # known answer: the jammed scenes, where rho_B itself does not converge and the measured optimum is elsewhere
    for scene in ("dem_step", "pack200", "pack800"):
        if scene in best_of:
            p, _ = best_of[scene]
            facit.add(f"{scene}@j{p['rho']:.4g}")
    return rows, facit, ("rho_over_rho_B", "ladder_advantage_over_rho_B"), "log10_rho_over_rho_B", -1


def channel_family(d):
    """The exact-cone channel against one fixed anchor channel, over log10(mass ratio)."""
    rows, facit = [], set()
    for i, s in enumerate(d["disagreement"]["scenes"]):
        a = float(s["anchor_score"]["gpu_exact"])
        b = float(s["anchor_score"]["box3d"])
        ms = float(s["ms"].get("gpu_exact", 1.0)) + float(s["ms"].get("box3d", 1.0))
        lab = f"s{i:03d}"
        rows.append(SweepFamily(label=lab, x=math.log10(max(s["r"], 1.0)), a=max(a, 1e-6), b=max(b, 1e-6),
                                cost=max(ms, 1e-3), tag=s["src"]))
        if s["exact_lost"]:
            facit.add(lab)
    return rows, facit, ("mass_ratio", "exact_advantage_over_anchor"), "log10_mass_ratio", +1


# -- windows -----------------------------------------------------------------------------------------
def windows(rows, n):
    lo = min(r.x for r in rows)
    hi = max(r.x for r in rows)
    step = (hi - lo) / n
    out = []
    for i in range(n):
        a = lo + i * step
        b = hi if i == n - 1 else lo + (i + 1) * step
        inside = [r for r in rows if (a <= r.x < b) or (i == n - 1 and r.x == hi)]
        if inside:
            out.append((i, a, b, inside))
    return out


def _grid(xs, k):
    """At most k of the window's own condition values, spread evenly over them — the sweep's design, declared."""
    if len(xs) <= k:
        return list(xs)
    step = (len(xs) - 1) / (k - 1)
    return [xs[int(round(i * step))] for i in range(k)]


def window_of(P, x):
    """Which of the family's windows a condition value falls in; None outside every non-empty one."""
    hi = max(r.x for r in P["rows"])
    for (i, a, b, inside) in P["windows"]:
        if a <= x < b or (b == hi and x == b):
            return i
    return None


def is_hit(P, act):
    """A purchase hits when it lands in a window whose density of known answers is above the family's own rate."""
    w = act.meta["window"]
    if w < 0:
        w = window_of(P, float(act.meta["span"][0]))
    return w in P["facit_windows"]


def candidates(state, pairs, round_i):
    """Every purchase on offer this round, priced in one bits-per-cost list."""
    out = []
    for P in pairs:
        for (i, a, b, inside) in P["windows"]:
            ins = ContactSolverSweep(f"{P['name']}:w{i}", inside, f"{P['campaign']}:w{i}",
                                     method_a=P["method_a"], method_b=P["method_b"],
                                     quantity=P["quantity"], window=P["half_width"])
            xs = _grid(sorted({round(r.x, 9) for r in inside}), K_MAX)
            act = ins.action(state, P["pair"], xs)
            act.meta.update({"family": P["name"], "window": i, "span": [a, b],
                             "labels": [r.label for r in inside], "K": len(xs)})
            out.append((act, ins, P, xs))
        # the fair comparison: one more single solve at the pair's currently most-informative point
        post = state.regimes[P["pair"]]
        x, _ = post.best_probe(P["single"].reliability)
        near = P["single"].rows_near(x)
        act = P["single"].action(state, P["pair"], [x])
        act.meta.update({"family": P["name"], "window": -1, "span": [x, x],
                         "labels": [r.label for r in near], "K": 1, "single": True})
        out.append((act, P["single"], P, [x]))
    return out


def main():
    d = json.load(open(DATA))
    rng = random.Random(20260920)
    specs = [
        dict(build=router_family, name="router", campaign="campaign:router_benchmark", n_windows=12,
             method_a="plain PGS", method_b="ADMM ladder", quantity="sweep-equivalents", half_width=0.14),
        dict(build=rho_family, name="rho", campaign="campaign:rho_ladder", n_windows=12,
             method_a="the ladder point", method_b="rho_B", quantity="iterations", half_width=0.30),
        dict(build=channel_family, name="channels", campaign="campaign:channel_sweep", n_windows=8,
             method_a="exact cone", method_b="anchor channel", quantity="anchor score", half_width=0.22),
    ]
    state = EngineState(pair_bundles=False, probe_instruments=[])
    pairs = []
    for sp in specs:
        rows, facit, pair, var, seed_sign = sp["build"](d)
        lo = min(r.x for r in rows) - 1e-9
        hi = max(r.x for r in rows) + 1e-9
        post = RegimePosterior(lo=lo, hi=hi, reliability=SEED_RELIABILITY, p_flip=0.3)
        post.add_claim(lo, hi, seed_sign, n_eff=1.0, reliability=SEED_RELIABILITY)   # the published rule
        state.regimes[pair] = post
        state.regime_variables[pair] = var
        full = ContactSolverSweep(sp["name"], rows, sp["campaign"], method_a=sp["method_a"],
                                  method_b=sp["method_b"], quantity=sp["quantity"], window=sp["half_width"])
        single = ContactSolverSweep(f"{sp['name']}:single", rows, f"{sp['campaign']}:single",
                                    method_a=sp["method_a"], method_b=sp["method_b"], quantity=sp["quantity"],
                                    window=sp["half_width"],
                                    cost=sorted(r.cost for r in rows)[len(rows) // 2])
        wins = windows(rows, sp["n_windows"])
        # A KNOWN-ANSWER WINDOW is one whose density of known answers is above the family's own overall rate.
        # "Contains at least one" is useless where the known answers are spread thin over every window (the 39
        # lost channel scenes sit in all 8), so the facit is a concentration, and the base rate follows from it.
        overall = len(facit) / len(rows)
        facit_windows = [i for (i, a, b, ins) in wins
                         if sum(r.label in facit for r in ins) / len(ins) > overall]
        pairs.append(dict(name=sp["name"], campaign=sp["campaign"], pair=pair, var=var, rows=rows,
                          facit=facit, windows=wins, full=full, single=single, method_a=sp["method_a"],
                          method_b=sp["method_b"], quantity=sp["quantity"], half_width=sp["half_width"],
                          facit_windows=facit_windows, n_windows=len(wins)))

    report = {"families": [], "rounds": [], "hit_rate": {}, "controls": {}}
    print("# e39 — cells as instruments, the first contact-solver sweep\n")
    print(f"{'family':>9} {'scenes':>7} {'windows':>8} {'facit':>6} {'facit win':>10} {'base rate':>10} "
          f"{'r(instr)':>9} {'cost(ms)':>10}")
    for P in pairs:
        base = len(P["facit_windows"]) / P["n_windows"]
        print(f"{P['name']:>9} {len(P['rows']):>7} {P['n_windows']:>8} {len(P['facit']):>6} "
              f"{len(P['facit_windows']):>10} {base:>10.3f} {P['full'].reliability:>9.4f} "
              f"{P['full'].cost:>10.1f}")
        report["families"].append({
            "family": P["name"], "pair": list(P["pair"]), "variable": P["var"], "n_rows": len(P["rows"]),
            "n_windows": P["n_windows"], "n_facit_rows": len(P["facit"]), "facit_windows": P["facit_windows"],
            "base_rate": base, "instrument_reliability": P["full"].reliability,
            "campaign_cost_ms": P["full"].cost, "lineage_root": P["campaign"],
            "claim": f"{P['method_a']} converges faster than {P['method_b']} on {P['var']}",
            "span": [min(r.x for r in P["rows"]), max(r.x for r in P["rows"])],
        })

    # what the instrument reads at each family's own windows, before anything is bought
    print("\n## the reduction: sign and p per window (measurement -> claim form)")
    for P in pairs:
        print(f"\n{P['name']}  ({P['method_a']} vs {P['method_b']}, {P['quantity']}; "
              f"+1 = {P['method_a']} faster)")
        for (i, a, b, inside) in P["windows"]:
            ins = ContactSolverSweep(f"{P['name']}:w{i}", inside, f"{P['campaign']}:w{i}",
                                     method_a=P["method_a"], method_b=P["method_b"],
                                     quantity=P["quantity"], window=P["half_width"])
            x = 0.5 * (a + b)
            sg, p, c = ins.probe(P["pair"], x)
            mark = "  <- known answer" if i in P["facit_windows"] else ""
            print(f"  w{i:<2} x [{a:+.2f}, {b:+.2f})  n {len(inside):>3}  sign {sg:+d}  "
                  f"p {max(p, 1 - p):.3f}  ratio {math.exp(abs(ins.advantage(x))):>8.2f}x  "
                  f"cost {ins.cost:>9.1f} ms{mark}")

    # -- where the known-answer window ranks in each family's own first-round list ---------------------
    print("\n## round 1, inside each family: where does the known-answer window rank?")
    print(f"{'family':>9} {'windows':>8} {'by bits/cost':>13} {'by bits':>9} {'p@1':>5} {'p@3':>5} {'base':>6}")
    cand0 = candidates(state, pairs, 0)
    for P in pairs:
        mine = [(a, i, Q, xs) for (a, i, Q, xs) in cand0 if Q is P and a.meta["window"] >= 0]
        by_vpc = sorted(mine, key=lambda c: -c[0].value_per_cost)
        by_bits = sorted(mine, key=lambda c: -c[0].value_bits)
        rank_vpc = [j + 1 for j, c in enumerate(by_vpc) if c[0].meta["window"] in P["facit_windows"]]
        rank_bits = [j + 1 for j, c in enumerate(by_bits) if c[0].meta["window"] in P["facit_windows"]]
        p1 = sum(c[0].meta["window"] in P["facit_windows"] for c in by_vpc[:1])
        p3 = sum(c[0].meta["window"] in P["facit_windows"] for c in by_vpc[:3]) / 3.0
        base = len(P["facit_windows"]) / P["n_windows"]
        print(f"{P['name']:>9} {P['n_windows']:>8} {str(rank_vpc):>13} {str(rank_bits):>9} "
              f"{p1:>5} {p3:>5.2f} {base:>6.3f}")
        report["families"][[q["name"] for q in pairs].index(P["name"])].update(
            {"rank_of_known_answer_window_by_bits_per_cost": rank_vpc,
             "rank_of_known_answer_window_by_bits": rank_bits, "precision_at_1": p1, "precision_at_3": p3})

    # -- the loop ------------------------------------------------------------------------------------
    print(f"\n## the next-actions loop ({ROUNDS} rounds, one bits-per-cost list across all three pairs)\n")
    print(f"{'#':>2} {'family':>9} {'win':>4} {'K':>3} {'bits':>9} {'cost(ms)':>10} {'bits/cost':>11} "
          f"{'realized':>9} {'hit':>4}")
    hits, bought = 0, []
    spent = 0.0
    taken = set()
    for t in range(ROUNDS):
        cand = [c for c in candidates(state, pairs, t) if id(c[0]) is not None]
        cand = [(a, i, P, xs) for (a, i, P, xs) in cand
                if (P["name"], a.meta["window"]) not in taken and a.value_bits > 0]
        if not cand:
            break
        by_id = {id(a): (a, i, Q, xs) for (a, i, Q, xs) in cand}
        state.bundles = [a for (a, _i, _Q, _xs) in cand]
        ranked = next_actions(state, k=len(cand))              # the engine's own list, in bits per cost
        act, ins, P, xs = by_id[id(ranked[0])]
        outcome = ins.outcome(P["pair"], xs)
        row = apply(state, act, outcome)
        hit = is_hit(P, act)
        hits += int(hit)
        spent += act.cost
        taken.add((P["name"], act.meta["window"]))
        print(f"{t + 1:>2} {P['name']:>9} {act.meta['window']:>4} {act.meta['K']:>3} {act.value_bits:>9.4f} "
              f"{act.cost:>10.1f} {act.value_per_cost:>11.6f} {row['value_realized']:>9.4f} "
              f"{'YES' if hit else 'no':>4}")
        bought.append({"round": t + 1, "family": P["name"], "window": act.meta["window"], "K": act.meta["K"],
                       "span": act.meta["span"], "value_bits": act.value_bits, "cost_ms": act.cost,
                       "value_per_cost": act.value_per_cost, "realized_bits": row["value_realized"],
                       "lineage_root": act.meta["lineage_root"], "sources": act.meta["sources"],
                       "labels": act.meta["labels"], "hit": bool(hit),
                       "instrument_reliability": ins.reliability})
    report["rounds"] = bought

    pooled_base = sum(len(P["facit_windows"]) for P in pairs) / sum(P["n_windows"] for P in pairs)
    hr = hits / len(bought) if bought else 0.0
    report["hit_rate"] = {"hits": hits, "purchases": len(bought), "hit_rate": hr,
                          "base_rate_pooled": pooled_base, "lift": hr / pooled_base if pooled_base else None,
                          "cost_ms_spent": spent}
    print(f"\nhit rate {hits}/{len(bought)} = {hr:.3f}  against a pooled base rate of {pooled_base:.3f} "
          f"({hr / pooled_base:.2f}x)  for {spent:.1f} ms of measurement")

    # -- controls ------------------------------------------------------------------------------------
    for label, key in (("cost-blind (bits only)", "bits"), ("random", "random")):
        st2 = EngineState(pair_bundles=False, probe_instruments=[])
        for P in pairs:
            lo = min(r.x for r in P["rows"]) - 1e-9
            hi = max(r.x for r in P["rows"]) + 1e-9
            p2 = RegimePosterior(lo=lo, hi=hi, reliability=SEED_RELIABILITY, p_flip=0.3)
            p2.add_claim(lo, hi, -1 if P["name"] != "channels" else +1, 1.0, SEED_RELIABILITY)
            st2.regimes[P["pair"]] = p2
            st2.regime_variables[P["pair"]] = P["var"]
        h2, n2, c2 = 0, 0, 0.0
        tk = set()
        for t in range(ROUNDS):
            cand = [(a, i, P, xs) for (a, i, P, xs) in candidates(st2, pairs, t)
                    if (P["name"], a.meta["window"]) not in tk and a.value_bits > 0]
            if not cand:
                break
            pick = rng.choice(cand) if key == "random" else max(cand, key=lambda c: c[0].value_bits)
            act, ins, P, xs = pick
            apply(st2, act, ins.outcome(P["pair"], xs))
            h2 += int(is_hit(P, act)); n2 += 1; c2 += act.cost
            tk.add((P["name"], act.meta["window"]))
        report["controls"][key] = {"hits": h2, "purchases": n2, "hit_rate": h2 / n2 if n2 else 0.0,
                                   "cost_ms_spent": c2}
        print(f"control {label:>22}: {h2}/{n2} = {h2 / n2 if n2 else 0:.3f}  for {c2:.1f} ms")

    # -- is the router miss an artefact of the window count, the seed, or the instrument's own sharpness? -----
    print("\n## sensitivity: where the router's known-answer window ranks, by bits per cost")
    print(f"{'windows':>8} {'seed r':>7} {'measured r':>12} {'uniform r':>11} {'n windows':>10}")
    sens = []
    rows_r, facit_r, pair_r, var_r, _sg = router_family(d)
    for nw in (8, 10, 12, 16):
        wins = windows(rows_r, nw)
        overall = len(facit_r) / len(rows_r)
        fw = [i for (i, a, b, ins) in wins if sum(r.label in facit_r for r in ins) / len(ins) > overall]
        for seed_r in (0.60, 0.75, 0.90):
            ranks = {}
            for mode in ("measured", "uniform"):
                st = EngineState(pair_bundles=False, probe_instruments=[])
                lo, hi = min(r.x for r in rows_r) - 1e-9, max(r.x for r in rows_r) + 1e-9
                rp = RegimePosterior(lo=lo, hi=hi, reliability=seed_r, p_flip=0.3)
                rp.add_claim(lo, hi, -1, 1.0, seed_r)
                st.regimes[pair_r] = rp
                st.regime_variables[pair_r] = var_r
                fam_r = ContactSolverSweep("router", rows_r, "campaign:router_benchmark", window=0.14).reliability
                acts = []
                for (i, a, b, inside) in wins:
                    kw = {"reliability": fam_r} if mode == "uniform" else {}
                    ins = ContactSolverSweep(f"w{i}", inside, f"campaign:router_benchmark:w{i}",
                                             window=0.14, **kw)
                    act = ins.action(st, pair_r, _grid(sorted({round(r.x, 9) for r in inside}), K_MAX))
                    acts.append((i, act))
                order = [i for i, _a in sorted(acts, key=lambda t: -t[1].value_per_cost)]
                ranks[mode] = [order.index(i) + 1 for i in fw]
            print(f"{nw:>8} {seed_r:>7.2f} {str(ranks['measured']):>12} {str(ranks['uniform']):>11} "
                  f"{len(wins):>10}")
            sens.append({"n_windows": nw, "seed_reliability": seed_r, "n_nonempty": len(wins),
                         "facit_windows": fw, "rank_measured_reliability": ranks["measured"],
                         "rank_uniform_reliability": ranks["uniform"]})
    report["router_sensitivity"] = sens

    # -- the N_eff = 1 rule, on the bought sweeps -----------------------------------------------------
    roots = [b["lineage_root"] for b in bought]
    report["lineage"] = {"purchases": len(bought), "distinct_roots": len(set(roots)),
                         "answers_entered": sum(b["K"] for b in bought),
                         "weight_per_answer": [1.0 / b["K"] for b in bought]}
    print(f"\nlineage: {sum(b['K'] for b in bought)} answers entered from {len(set(roots))} roots "
          f"({len(bought)} purchases, one root each, weight 1/K per answer)")

    json.dump(report, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
