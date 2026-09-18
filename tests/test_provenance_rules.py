"""Hand cases for graph_engine.provenance_rules: what the rules must read, what they must NOT guess, and that their
output is a valid margin_net lineage declaration (three copies of one trial ⇒ N_eff 1; three trials ⇒ N_eff 3)."""
import pytest

from graph_engine.margin_net import MarginNet
from graph_engine.provenance_rules import (KIND_TO_LINEAGE, extract_provenance, provenance_abstentions,
                                           provenance_sources)


def _kinds(text):
    return {(r["kind"], r["target"]) for r in extract_provenance(text)}


# -- 15 hand cases: 10 that must extract, 5 that must abstain ------------------------------------
EXTRACT = [
    ("Blood pressure targets were set according to the 2018 ESC/ESH guidelines for arterial hypertension.",
     {("copy", "ESC/ESH guideline")}),
    ("Treatment thresholds followed the NICE guideline NG136.", {("copy", "NICE guideline")}),
    ("Intensive control is recommended by the 2017 ACC/AHA hypertension guideline.",
     {("copy", "ACC/AHA guideline")}),
    ("This is a secondary analysis of the SPRINT trial.", {("derived", "SPRINT")}),
    ("We report a post hoc analysis of the ACCORD study.", {("derived", "ACCORD")}),
    ("A pooled analysis of the ACCORD and SPRINT trials was performed.",
     {("derived", "ACCORD"), ("derived", "SPRINT")}),
    ("Using data from the ARIC cohort, we estimated the hazard ratio for stroke.", {("shared", "ARIC")}),
    ("Participants enrolled in the HYVET trial were followed for five years.", {("shared", "HYVET")}),
    ("Our estimate is consistent with the ALLHAT trial.", {("cites", "ALLHAT")}),
    ("The SPRINT trial showed a 25% reduction in cardiovascular events.", {("cites", "SPRINT")}),
]

ABSTAIN = [
    "We performed a meta-analysis of 42 randomised trials of blood pressure lowering.",
    "Our results are consistent with previous reports.",
    "Previous studies have suggested a benefit of intensive blood pressure control.",
    "Blood pressure was measured with an oscillometric device in a quiet room.",
    "According to our findings, the absolute risk reduction is small.",
]


@pytest.mark.parametrize("text,want", EXTRACT)
def test_hand_cases_extract(text, want):
    assert _kinds(text) == want


@pytest.mark.parametrize("text", ABSTAIN)
def test_hand_cases_abstain(text):
    assert extract_provenance(text) == []


def test_marker_without_a_name_is_reported_as_an_abstention_not_a_record():
    text = "We performed a meta-analysis of 42 randomised trials."
    assert extract_provenance(text) == []
    ab = provenance_abstentions(text)
    assert len(ab) == 1 and ab[0]["target"] is None and "meta-analysis" in ab[0]["marker"].lower()


def test_statistics_and_disease_abbreviations_are_not_trial_names():
    text = "In the intensive arm the HR for CVD was 0.75 (95% CI 0.64-0.89) on ABPM."
    assert extract_provenance(text) == []


def test_ambivalent_marker_splits_on_what_is_named():
    got = _kinds("Targets were based on the SPRINT trial and on the 2017 ACC/AHA guideline.")
    assert got == {("derived", "SPRINT"), ("copy", "ACC/AHA guideline")}


def test_one_record_per_target_strongest_kind_wins():
    text = ("According to the SPRINT trial, intensive control helps. Our findings are consistent with the "
            "SPRINT trial.")
    recs = extract_provenance(text)
    assert [r["target"] for r in recs] == ["SPRINT"] and recs[0]["kind"] == "copy"
    assert len(extract_provenance(text, dedupe=False)) >= 2


def test_sources_records_are_what_margin_net_accepts():
    st = extract_provenance("This is a secondary analysis of the SPRINT trial performed according to the "
                            "2018 ESC/ESH guidelines.")
    recs = provenance_sources("paper1", st, rho=0.5)
    assert recs == [{"id": "paper1", "derives_from": ["ESC/ESH guideline"], "shares": {"SPRINT": 0.5}}]
    MarginNet().add_sources(recs)                       # must not raise
    assert provenance_sources("paper2", [{"kind": "cites", "target": "SPRINT", "confidence": 0.7}]) == []
    with pytest.raises(ValueError):
        provenance_sources("paper3", st, rho=1.0)
    assert set(KIND_TO_LINEAGE.values()) == {"derives_from", "shares", None}


def test_three_copies_of_one_trial_collapse_to_one_effective_report():
    """The owner's case: three guideline-style abstracts restating SPRINT are one measurement, not three."""
    copies = ["Management was in accordance with the SPRINT trial protocol.",
              "Thresholds were set according to the SPRINT trial.",
              "We applied the targets recommended by the SPRINT trial."]
    net = MarginNet()
    reports = []
    for i, t in enumerate(copies):
        st = [r for r in extract_provenance(t) if r["kind"] == "copy"]
        assert [r["target"] for r in st] == ["SPRINT"]
        net.add_sources(provenance_sources(f"a{i}", st))
        reports.append({"margin": 0.75, "sigma": 0.1, "sources": [f"a{i}"]})
    net.add_edge("hr", ["hr"], reports)
    e = net.estimate("hr")
    assert e.n_reports == 3 and e.n_eff == pytest.approx(1.0)
    assert e.s == pytest.approx(0.1)                    # three copies do not narrow the estimate

    indep = MarginNet()
    indep.add_edge("hr", ["hr"], [{"margin": 0.75, "sigma": 0.1, "sources": [f"b{i}"]} for i in range(3)])
    ei = indep.estimate("hr")
    assert ei.n_eff == pytest.approx(3.0) and ei.s == pytest.approx(0.1 / 3 ** 0.5)


def test_copies_that_disagree_are_a_contradiction_shared_ones_are_not():
    net = MarginNet()
    for i in range(3):
        net.add_sources(provenance_sources(f"a{i}", [{"kind": "copy", "target": "SPRINT", "confidence": 0.9}]))
    net.add_edge("hr", ["hr"], [{"margin": m, "sigma": 0.1, "sources": [f"a{i}"]}
                                for i, m in enumerate([0.60, 0.75, 0.90])])
    assert net.estimate("hr").kind == "CONTRADICTION"   # declared copies must agree

    part = MarginNet()
    for i in range(3):
        part.add_sources(provenance_sources(f"c{i}", [{"kind": "derived", "target": "SPRINT", "confidence": 0.9}],
                                            rho=0.5))
    part.add_edge("hr", ["hr"], [{"margin": m, "sigma": 0.1, "sources": [f"c{i}"]}
                                 for i, m in enumerate([0.60, 0.75, 0.90])])
    est = part.estimate("hr")
    assert est.kind != "CONTRADICTION" and 1.0 < est.n_eff < 3.0
