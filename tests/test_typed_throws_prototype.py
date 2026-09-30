"""Small tests: a constructed case with a KNOWN mediator, a constructed KNOWN triple conf (each kind), chain regime
conf, AND-inputs, exact draw probabilities, and the port signature separating GRAPH-01's deceptive neighbours.

Run:  OMP_NUM_THREADS=2 nice -n 10 /home/anton/projects/CADtoSIMReady/.venv-newton/bin/python -m pytest -q \
      /home/anton/research/TRIPLE_THROWS_20260930/triple_throws/tests/test_triple_throws.py
"""
import math
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph_engine.typed_throw_ops.parts import Part, Port, Claim                                  # noqa: E402
from graph_engine.typed_throw_ops.compose import link, evaluate_chain, enumerate_chains           # noqa: E402
from graph_engine.typed_throw_ops.value import StructuralImage, triple_terms                      # noqa: E402
from graph_engine.typed_throw_ops.triples import (mediation, sign_cycles, interval_triples, order_cycles,  # noqa: E402
                                   transition_triples, federation_sees, Relation, satisfiable, frustrated_cycles)
from graph_engine.typed_throw_ops.value import closure_value  # noqa: E402
from graph_engine.typed_throw_ops.signature import port_signature, distance                       # noqa: E402
from graph_engine.typed_throw_ops.draw import draw_from_universe, all_inclusion_probabilities     # noqa: E402


def P(id, ins, outs, ops=("schur_reduction",), **kw):
    return Part(id=id, report=kw.pop("report", id), family=kw.pop("family", "test"), does=kw.pop("does", id),
                inputs=[Port(*x) if isinstance(x, tuple) else x for x in ins],
                outputs=[Port(*x) if isinstance(x, tuple) else x for x in outs], operator=list(ops), **kw)


# ---------------------------------------------------------------------------------------------- known mediator
def mediator_world():
    A = P("A", [Port("x", "N", given=True)], [("y", "N s")])
    B = P("B", [("y", "N s")], [("z", "m/s")], ops=("delassus_contraction",))
    C = P("C", [("z", "m/s")], [("w", "1")], ops=("fisher_information",))
    D = P("D", [Port("x", "N", given=True)], [("w2", "1")], ops=("greedy_selection",))   # decoy, unrelated output
    return A, B, C, D


def test_known_mediator_ports():
    A, B, C, D = mediator_world()
    m = mediation(A, B, C)
    assert m is not None and m.direction == "a>b>c" and not m.direct_ac and m.regime_ok
    assert m.shared == ("y", "z")
    assert mediation(A, D, C) is None                       # D does not open a port from A to C


def test_known_mediator_value_is_not_a_sum_of_pairs():
    A, B, C, D = mediator_world()
    img = StructuralImage([A, B, C, D])
    t = triple_terms(lambda S: img.value_Q(S, "w"), "A", "B", "C")
    v = t["v"]
    assert v["A+B+C"] > 5.0                                  # the triple pins w from the given x
    assert max(v["A+B"], v["A+C"], v["B+C"]) < 1.0           # no pair does
    assert t["excess_over_pairs"] > 5.0 and t["max_complementarity"] > 5.0
    # the decoy triple has no excess
    t2 = triple_terms(lambda S: img.value_Q(S, "w"), "A", "D", "C")
    assert t2["excess_over_pairs"] < 0.5


def test_submodular_value_cannot_see_the_mediator():
    """best_set's valuation (unrestricted log det) is submodular: complementarity ≤ 0 for every labelling."""
    A, B, C, D = mediator_world()
    img = StructuralImage([A, B, C, D])
    for trio in combinations(["A", "B", "C", "D"], 3):
        t = triple_terms(lambda S: img.value_full(S), *trio)
        assert t["max_complementarity"] <= 1e-6, (trio, t)
    # random structural worlds: same property
    rng = np.random.default_rng(0)
    for rep in range(20):
        qs = [f"q{i}" for i in range(6)]
        parts = []
        for k in range(5):
            ins = rng.choice(qs, size=rng.integers(1, 3), replace=False)
            out = rng.choice([q for q in qs if q not in ins])
            parts.append(P(f"R{k}", [Port(str(q), None, given=bool(rng.random() < .3)) for q in ins], [(str(out), None)]))
        img = StructuralImage(parts)
        for trio in combinations([p.id for p in parts], 3):
            assert triple_terms(lambda S: img.value_full(S), *trio)["max_complementarity"] <= 1e-6


def test_link_rejects_unit_and_regime_and_assumption():
    A = P("A", [Port("x", "N", given=True)], [("y", "N s")])
    Bu = P("Bu", [("y", "m/s")], [("z", "1")])              # same name, wrong dimension
    assert not link(A, Bu)[0].ok and any("unit_dimension_mismatch" in r for r in link(A, Bu)[0].reasons)
    Ar = P("Ar", [Port("x", "N", given=True)], [("y", "N s")], regime={"branch": ["stick"]})
    Br = P("Br", [("y", "N s")], [("z", "1")], regime={"branch": ["slip"]})
    assert not link(Ar, Br)[0].ok and any("regime_disjoint" in r for r in link(Ar, Br)[0].reasons)
    Aa = P("Aa", [Port("x", "N", given=True)], [("y", "N s")], assumptions=["rigid_contact"])
    Ba = P("Ba", [("y", "N s")], [("z", "1")], assumptions=["compliant_contact"])
    assert not link(Aa, Ba)[0].ok and any("assumption_conflict" in r for r in link(Aa, Ba)[0].reasons)
    Ab = P("Ab", [Port("x", "N", given=True)], [("y", "N s")], box={"mu": (0, .2)})
    Bb = P("Bb", [("y", "N s")], [("z", "1")], box={"mu": (.3, 1)})
    assert not link(Ab, Bb)[0].ok and any("disjoint_validity" in r for r in link(Ab, Bb)[0].reasons)


# ---------------------------------------------------------------------------------------------- chains
def test_chain_regime_conf_nonadjacent_boxes():
    P1 = P("P1", [Port("a", "1", given=True)], [("b", "1")], box={"mu": (0, .3)})
    P2 = P("P2", [("b", "1")], [("c", "1")], box={"mu": (.2, .6)})
    P3 = P("P3", [("c", "1")], [("d", "1")], box={"mu": (.5, 1)})
    ch = evaluate_chain([P1, P2, P3])
    assert ch.ok_adjacent and ch.inputs_closed and ch.regime_conf and not ch.runs


def test_chain_regime_conf_categorical_helly_failure():
    P1 = P("P1", [Port("a", "1", given=True)], [("b", "1")], regime={"band": ["gap", "passband"]})
    P2 = P("P2", [("b", "1")], [("c", "1")], regime={"band": ["gap", "static"]})
    P3 = P("P3", [("c", "1")], [("d", "1")], regime={"band": ["passband", "static"]})
    ch = evaluate_chain([P1, P2, P3])
    assert ch.ok_adjacent and ch.regime_conf


def test_chain_and_input_needs_second_supplier():
    S1 = P("S1", [Port("e0", "1", given=True)], [("e", "1")])
    PHI = P("PHI", [("e", "1"), ("r", "1")], [("Phi", "1")], ops=("coherence_certificate",))
    USE = P("USE", [("Phi", "1")], [("decision", "1")])
    ch = evaluate_chain([S1, PHI, USE])
    assert ch.ok_adjacent and not ch.inputs_closed and ch.missing_inputs == [("PHI", "r")]
    S2 = P("S2", [Port("r0", "1", given=True)], [("r", "1")])
    img = StructuralImage([S1, S2, PHI, USE])
    t = triple_terms(lambda S: img.value_Q(S, "Phi"), "S1", "S2", "PHI")
    assert t["v"]["S1+S2+PHI"] > 5 and max(t["v"]["S1+PHI"], t["v"]["S2+PHI"]) < 1.0   # AND node: both suppliers
    chains = enumerate_chains([S1, PHI, USE], max_len=3)
    assert any(c.parts == ["S1", "PHI", "USE"] for c in chains)


# ---------------------------------------------------------------------------------------------- triple confs
PHI_REL = [Relation("U586", "map", out="Phi", ins=("e", "r"),
                    f=lambda e, r: math.tanh(2 * math.atanh(min(max(e, 0), .999999)) + math.atanh(min(max(r, 0), .999999))),
                    mono=(1, 1))]


def test_known_interval_triple_conf_contradiction():
    c1 = Claim("c1", "X", "bound", subject="e", lo=0, hi=.01)
    c2 = Claim("c2", "Y", "bound", subject="r", lo=0, hi=.01)
    c3 = Claim("c3", "Z", "bound", subject="Phi", lo=.05, hi=1)
    for a, b in combinations([c1, c2, c3], 2):
        assert satisfiable([a, b], PHI_REL)
    assert not satisfiable([c1, c2, c3], PHI_REL)
    got = interval_triples([c1, c2, c3], PHI_REL)
    assert len(got) == 1 and got[0].cls == "contradiction"


def test_known_interval_triple_conf_regime_boundary():
    c1 = Claim("c1", "X", "bound", subject="e", lo=0, hi=.01, regime={"band": ["gap", "passband"]})
    c2 = Claim("c2", "Y", "bound", subject="r", lo=0, hi=.01, regime={"band": ["gap", "static"]})
    c3 = Claim("c3", "Z", "bound", subject="Phi", lo=.05, hi=1, regime={"band": ["passband", "static"]})
    got = interval_triples([c1, c2, c3], PHI_REL)
    assert len(got) == 1 and got[0].cls == "regime_boundary"


def test_sign_cycle_is_visible_in_federation():
    a = Claim("a", "A", "sign", subject="s", object="x", sign=1)
    b = Claim("b", "B", "sign", subject="x", object="o", sign=1)
    c = Claim("c", "C", "sign", subject="s", object="o", sign=-1)
    got = sign_cycles([a, b, c])
    assert len(got) == 1 and got[0].cls == "contradiction"
    fs = federation_sees([a, b, c])
    assert fs["stress_points"] == [] or all(k != "CONTRADICTION" for k, _ in fs["stress_points"])
    assert any(k == "CYCLE" for k, _ in fs["stress_points"])
    assert any(s == "s" and o == "o" for s, _, o, _ in fs["inferred"])


def test_order_cycle():
    a = Claim("a", "A", "order", subject="p", object="q")
    b = Claim("b", "B", "order", subject="q", object="r")
    c = Claim("c", "C", "order", subject="r", object="p")
    assert len(order_cycles([a, b, c])) == 1 and len(order_cycles([a, b])) == 0


def test_transition_triple():
    obs = [Claim(f"o{k}", f"S{k}", "sign_at", subject="u", object="v", sign=s, axis="f", at=x)
           for k, (s, x) in enumerate([(1, .1), (-1, .5), (1, .9)])]
    rule = Claim("rule", "M", "sign_at", subject="u", object="v", axis="f", max_transitions=1)
    got = transition_triples(obs + [rule])
    assert len(got) == 1 and got[0].cls == "ordering"
    assert transition_triples(obs[:2] + [rule]) == []


# ---------------------------------------------------------------------------------------------- draw
def test_draw_probabilities_are_exact():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=40)
    k = 5
    pi = all_inclusion_probabilities(scores, k, temperature=1.0, floor=.1)
    assert abs(pi.sum() - k) < 1e-9 and pi.min() > 0
    counts = np.zeros(40)
    n = 3000
    for s in range(n):
        for t, p in draw_from_universe(scores, k, 1.0, .1, seed=s):
            counts[t] += 1
            assert abs(p - pi[t]) < 1e-12
    assert np.max(np.abs(counts / n - pi)) < 0.035


# ---------------------------------------------------------------------------------------------- port signature
def test_port_signature_separates_graph01_deceptive_neighbours():
    base = dict(ins=[("q_power", "W"), ("sensor_set", None)], outs=[("rho", "1")], ops=("vertex_maximization",))
    ind = P("IND", **base, regime={"uncertainty_class": ["independent_box"]})
    tied = P("TIED", **base, regime={"uncertainty_class": ["tied_gain"]})
    cond = P("COND", **base, regime={"uncertainty_class": ["conditioned_schur"]})
    d1 = distance(port_signature(ind), port_signature(tied))
    d2 = distance(port_signature(ind), port_signature(cond))
    assert d1 > 0 and d2 > 0 and distance(port_signature(ind), port_signature(ind)) == 0
    words = P("W", **base, reach="words")
    assert math.isnan(distance(port_signature(ind), port_signature(words)))


def test_instances_do_not_mix():
    c1 = Claim("c1", "X", "bound", subject="e", lo=0, hi=.01, instance="mountA")
    c2 = Claim("c2", "Y", "bound", subject="r", lo=0, hi=.01, instance="mountA")
    c3 = Claim("c3", "Z", "bound", subject="Phi", lo=.05, hi=1, instance="gait")
    assert interval_triples([c1, c2, c3], PHI_REL) == []
    c3.instance = "mountA"
    assert len(interval_triples([c1, c2, c3], PHI_REL)) == 1


def test_frustrated_chain_of_four():
    cs = [Claim("a", "A", "sign", subject="s", object="x", sign=1), Claim("b", "B", "sign", subject="x", object="y", sign=-1),
          Claim("c", "C", "sign", subject="y", object="o", sign=1), Claim("d", "D", "sign", subject="s", object="o", sign=1)]
    got = frustrated_cycles(cs)
    assert len(got) == 1 and len(got[0].claims) == 4 and got[0].cls == "contradiction"
    cs[3].sign = -1                                          # balanced: product of chain = −1 = asserted
    assert frustrated_cycles(cs) == []
    assert sign_cycles(cs) == []                             # a triangle detector cannot see a 4-chain


def test_closure_value_known_mediator_and_direction():
    A, B, C, D = mediator_world()
    Pd = {p.id: p for p in (A, B, C, D)}
    given = {"x"}
    t = triple_terms(lambda S: closure_value(Pd, S, "w", given), "A", "B", "C")
    assert t["v"]["A+B+C"] > 5 and max(t["v"]["A+B"], t["v"]["A+C"], t["v"]["B+C"]) == 0.0
    assert t["max_complementarity"] > 5
    # direction: giving w does not pin x (no inverse use of a port)
    assert closure_value(Pd, ["A", "B", "C"], "x", {"w"}) == 0.0
