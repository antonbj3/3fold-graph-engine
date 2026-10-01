"""Execution needs a typed supply for every AND input, including skip edges."""
import itertools

import pytest

from graph_engine.claim_types import dimension, same_dimension
from graph_engine.next_actions import EngineState, chain_throw_bundle
from graph_engine.precision_form import PrecisionForm
from graph_engine.typed_throws import Part, Port, evaluate_chain


def part(name, inputs, outputs):
    return Part(name, name, "test", name, inputs, outputs, ["low_rank_update"])


def chain(extra_unit="N", extra_kind="scalar"):
    return [
        part("A", [Port("u", "1", given=True)], [Port("y", "1"), Port("x", "N")]),
        part("B", [Port("y", "1")], [Port("z", "1")]),
        part("C", [Port("z", "1"), Port("x", extra_unit, kind=extra_kind)], [Port("w", "1")]),
    ]


@pytest.mark.parametrize("unit,kind", [("m", "scalar"), ("N", "operator")])
def test_nonadjacent_required_supply_must_match_types(unit, kind):
    parts = chain(unit, kind)
    checked = evaluate_chain(parts)
    assert checked.ok_adjacent
    assert not checked.runs
    assert checked.missing_inputs == [("C", "x")]
    with pytest.raises(ValueError, match="missing=.*C.*x"):
        chain_throw_bundle(EngineState(form=PrecisionForm.zeros(4)), [0, 1, 2], parts=parts)


def test_one_good_adjacent_link_cannot_close_another_bad_required_input():
    parts = chain("m")
    parts[1].outputs.append(Port("x", "N"))
    assert evaluate_chain(parts).ok_adjacent
    assert evaluate_chain(parts).missing_inputs == [("C", "x")]


def test_matching_alternate_supply_and_explicit_given_input():
    parts = chain("m")
    parts[1].outputs.append(Port("x", "m"))
    assert evaluate_chain(parts).runs
    parts = chain("m")
    parts[2].inputs[1].given = True
    assert evaluate_chain(parts).runs


def test_head_external_supply_keeps_its_type():
    parts = chain("m")
    parts[0].outputs.pop()
    parts[0].inputs.append(Port("x", "N"))
    assert evaluate_chain(parts).missing_inputs == [("C", "x")]
    parts[2].inputs[1].unit = "kg*m/s^2"
    assert evaluate_chain(parts).runs


def forward_oracle(parts):
    """Separate Port-record reachability, without link/evaluate_chain calls."""
    def compatible(a, b):
        return (a.q == b.q and a.kind == b.kind
                and (not a.unit or not b.unit
                     or same_dimension(dimension(a.unit), dimension(b.unit))))

    if any(not any(compatible(o, i) for o in a.outputs for i in b.inputs)
           for a, b in zip(parts, parts[1:])):
        return False
    available = list(parts[0].inputs)
    for position, p in enumerate(parts):
        if position and any(i.required and not i.given
                            and not any(compatible(o, i) for o in available) for i in p.inputs):
            return False
        available.extend(p.outputs)
    return True


def test_typed_closure_matches_forward_port_oracle_over_orders_and_units():
    # Eight independently compatible/incompatible required occurrences, all
    # six operator orders. The oracle has the same external-head convention.
    count = 0
    for match_y, match_z, match_x in itertools.product((False, True), repeat=3):
        parts = chain("N" if match_x else "m")
        parts[1].inputs[0].unit = "1" if match_y else "m"
        parts[2].inputs[0].unit = "1" if match_z else "m"
        for order in itertools.permutations(parts):
            assert evaluate_chain(list(order)).runs == forward_oracle(order)
            count += 1
    assert count == 48
