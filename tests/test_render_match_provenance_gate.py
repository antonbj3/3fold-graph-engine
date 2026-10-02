"""multi_anchor_scale_gate: declaring a doubt is not acting on it.

The docstring promises that with no provenances supplied the gate "DECLARES provenance_unverified
(it cannot rule out a shared-ruler common-mode), rather than silently certifying". It did both: the
result carried CERTIFY and provenance_unverified=True together. Only KNOWN-too-weak provenance
abstained, while UNKNOWN provenance certified.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from graph_engine.render_match_decorrelation_judge import multi_anchor_scale_gate  # noqa: E402

TWO_CORRECT = [(1.0, 1.0), (1.0, 1.0)]          # uniform and matching the known lengths


def test_a_shared_ruler_abstains():
    out = multi_anchor_scale_gate(TWO_CORRECT, provenances=["ruler_x", "ruler_x"])
    assert out["verdict"] == "ABSTAIN"
    assert out["provenance_neff"] == 1.0
    assert out["provenance_unverified"] is False


def test_decorrelated_rulers_certify():
    out = multi_anchor_scale_gate(TWO_CORRECT, provenances=["ruler_x", "ruler_y"])
    assert out["verdict"] == "CERTIFY"
    assert out["provenance_neff"] == 2.0
    assert out["provenance_unverified"] is False


def test_absent_provenance_abstains_rather_than_certifying():
    """The case the docstring names, and the one that used to certify."""
    out = multi_anchor_scale_gate(TWO_CORRECT)
    assert out["verdict"] == "ABSTAIN"
    assert out["provenance_unverified"] is True
    assert "shared ruler" in out["reason"]


def test_a_wrong_but_uniform_scale_still_rejects():
    """The rejection must not be swallowed by the new abstain branch."""
    out = multi_anchor_scale_gate([(2.0, 1.0), (2.0, 1.0)])
    assert out["verdict"] == "REJECT"


def test_a_drifting_scale_still_abstains_for_its_own_reason():
    out = multi_anchor_scale_gate([(1.0, 1.0), (2.0, 1.0)], provenances=["x", "y"])
    assert out["verdict"] == "ABSTAIN"
    assert out["scale_nonuniformity"] > 0.05


def test_the_single_anchor_path_now_carries_the_field():
    """One anchor certifies only locally, which is scoped, but the field must still be visible."""
    out = multi_anchor_scale_gate([(1.0, 1.0)])
    assert out["single_anchor_uniformity_unobservable"] is True
    assert out["provenance_unverified"] is True
