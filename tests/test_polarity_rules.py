"""polarity_rules on hand-written sentences that are NOT the templates of the e7/e14 experiments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from graph_engine.polarity_rules import asserted_sign  # noqa: E402

CASES = [
    ("Raising the sintering temperature increases the density of the part.", "sintering temperature", "density", 1),
    ("Lowering the feed rate reduces surface roughness.", "feed rate", "surface roughness", 1),
    ("Lowering the feed rate raises tool life.", "feed rate", "tool life", -1),
    ("Tool life falls as the cutting speed rises.", "cutting speed", "tool life", -1),
    ("A thicker wall does not increase the buckling load; the buckling load decreases with wall thickness in this regime.", "wall thickness", "buckling load", -1),
    ("Drag is inversely proportional to the gap width.", "gap width", "drag", -1),
    ("The yield is not inversely proportional to dose.", "dose", "yield", 0),           # negated proportionality asserts no direction
    ("With more dopant the carrier mobility drops.", "dopant", "carrier mobility", -1),
    ("Smaller grains, higher strength.", "grains", "strength", -1),
    ("The lower bound of the interval was tightened.", "interval", "bound", 0),          # no relation between the two: must abstain or not matter
    ("Ökad temperatur ger lägre viskositet.", "temperatur", "viskositet", 0),           # Swedish: outside the lexicon → abstain
    ("The coating was applied in two passes.", "coating", "passes", 0),
]


def test_composition_negation_and_abstention():
    wrong = [(t, asserted_sign(t, x, y), s) for t, x, y, s in CASES if s != 0 and asserted_sign(t, x, y) != s]
    assert not wrong, wrong
    assert all(asserted_sign(t, x, y) == 0 for t, x, y, s in CASES if s == 0 and x != "interval")


# review cases: each answered WRONGLY before; the expected value is the truth, or 0 where abstention is the honest answer
REVIEW = [
    ("The pressure increases. The volume decreases.", "pressure", "volume", 0),        # two sentences: no relation asserted
    ("Higher pressure, higher yield, but lower purity.", "pressure", "yield", 1),      # "lower" names a THIRD quantity
    ("No increase in pressure was observed; the volume increases.", "pressure", "volume", 1),  # "no increase" ≠ a decrease

    ("A larger grain does not, in any way we could measure, increase the strength.", "grain", "strength", -1),
    ("The pressure does not not increase the volume.", "pressure", "volume", 0),       # double negation: abstain, never flip once
]


def test_review_cases_abstain_instead_of_answering_wrongly():
    got = [(t, asserted_sign(t, x, y), s) for t, x, y, s in REVIEW]
    assert [g[1] for g in got] == [g[2] for g in got], got


def test_a_direction_word_inside_a_quantity_name_is_not_a_direction():
    assert asserted_sign("The lower bound rises with sample size.", "sample size", "lower bound") == 1
