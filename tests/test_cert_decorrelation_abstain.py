"""cert_decorrelation.abstain_propagation: a downstream result that states nothing must abstain.

The function raises a ValueError on non-parallel flag lists precisely to avoid "neither certified nor
flagged = fail-open", and its own rule is that a passing downstream cert does not rescue an abstained
upstream item. A missing `pass` key is UNKNOWN, and for an abstain decision unknown has to abstain;
defaulting it to passing suppressed the abstain on exactly the input that says least.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from graph_engine.cert_decorrelation import abstain_propagation  # noqa: E402

ITEMS = ["a", "b"]
ALL_GROUNDED = [True, True]


def test_a_downstream_result_without_a_verdict_abstains():
    out = abstain_propagation(ITEMS, ALL_GROUNDED, lambda items: {"value": 0.42})
    assert out["model_abstains"] is True


def test_an_explicit_failure_abstains():
    out = abstain_propagation(ITEMS, ALL_GROUNDED, lambda items: {"pass": False})
    assert out["model_abstains"] is True


def test_an_explicit_pass_with_nothing_abstained_does_not_abstain():
    out = abstain_propagation(ITEMS, ALL_GROUNDED, lambda items: {"pass": True})
    assert out["model_abstains"] is False
    assert out["false_certify"] is False


def test_a_passing_downstream_does_not_rescue_an_abstained_item():
    """The module's stated rule, which is the reason false_certify exists."""
    out = abstain_propagation(ITEMS, [True, False], lambda items: {"pass": True},
                              naive_downstream_fn=lambda items: {"pass": True})
    assert out["abstained"] == ["b"]
    assert out["model_abstains"] is True
    assert out["false_certify"] is True


def test_a_silent_naive_result_does_not_manufacture_an_accusation():
    """The two defaults are opposite on purpose, and both fail closed."""
    out = abstain_propagation(ITEMS, [True, False], lambda items: {"pass": True},
                              naive_downstream_fn=lambda items: {"value": 1.0})
    assert out["model_abstains"] is True
    assert out["false_certify"] is False


def test_non_parallel_flags_still_raise():
    with pytest.raises(ValueError, match="parallel"):
        abstain_propagation(ITEMS, [True], lambda items: {"pass": True})
