#!/usr/bin/env python3
"""Hand-written cases for graph_engine.numeric_rules, including the abstentions, the unit table, and the hand-off
to margin_net: two extracted reports become two edge reports and the χ² disagreement test must fire on a planted
contradiction and must not fire on two consistent measurements."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graph_engine.margin_net import MarginNet                      # noqa: E402
from graph_engine.numeric_rules import extract_numeric_claims, normalize_unit  # noqa: E402


def one(text):
    r = extract_numeric_claims(text)
    assert len(r) == 1, f"expected exactly one claim, got {r}"
    return r[0]


def test_plain_pm_with_unit():
    r = one("The top quark mass is m_t = 172.52 ± 0.33 GeV.")
    assert r["value"] == pytest.approx(172.52) and r["sigma"] == pytest.approx(0.33)
    assert r["unit"] == "GeV" and r["quantity"] == "m_t" and r["flags"] == []
    assert text_of(r, "The top quark mass is m_t = 172.52 ± 0.33 GeV.").startswith("172.52")


def text_of(rec, text):
    a, b = rec["span"]
    return text[a:b]


def test_ascii_plus_minus_and_gev_over_c2():
    r = one("A Higgs boson mass of 125.09 +/- 0.24 GeV/c^2 is obtained.")
    assert r["value"] == pytest.approx(125.09) and r["unit"] == "GeV"
    assert r["quantity"] == "Higgs boson mass"


def test_mev_normalized_to_gev():
    r = one("The pion mass is 139.57 ± 0.02 MeV.")
    assert r["value"] == pytest.approx(0.13957) and r["sigma"] == pytest.approx(2e-5)
    assert r["unit"] == "GeV"


def test_ms_normalized_to_seconds_and_range_is_uniform():
    r = one("The drift time lies in the range 2.0-8.0 ms in this chamber.")
    assert r["unit"] == "s" and r["value"] == pytest.approx(5e-3)
    assert r["sigma"] == pytest.approx(6e-3 / (2 * math.sqrt(3)))
    assert "range_uniform" in r["flags"]


def test_percent_becomes_a_fraction():
    r = one("The selection efficiency is 92.5 ± 1.5 %.")
    assert r["unit"] == "1" and r["value"] == pytest.approx(0.925) and r["sigma"] == pytest.approx(0.015)


def test_stat_syst_combined_in_quadrature():
    r = one("We measure the top-quark mass 172.5 ± 0.7 (stat) ± 1.0 (syst) GeV.")
    assert r["sigma"] == pytest.approx(math.hypot(0.7, 1.0))
    assert "quadrature" in r["flags"] and r["quantity"] == "top-quark mass"


def test_latex_asymmetric_takes_the_larger_side_and_flags_it():
    r = one("The lifetime is 1.52 ^{+0.08}_{-0.11} ps.")
    assert r["value"] == pytest.approx(1.52e-12) and r["sigma"] == pytest.approx(1.1e-13)
    assert "asymmetric" in r["flags"] and r["unit"] == "s"


def test_parenthesised_value_with_common_power_of_ten():
    r = one("The branching ratio is (1.2 ± 0.3) × 10^{-5}.")
    assert r["value"] == pytest.approx(1.2e-5) and r["sigma"] == pytest.approx(3e-6)
    assert r["quantity"] == "branching ratio" and "power_of_ten" in r["flags"] and r["unit"] is None


def test_e_notation_on_both_value_and_sigma():
    r = one("The decay time is 1.2e-5 ± 3.0e-6 s.")
    assert r["value"] == pytest.approx(1.2e-5) and r["sigma"] == pytest.approx(3.0e-6) and r["unit"] == "s"


def test_confidence_interval_bracket_form():
    r = one("The odds ratio was 1.34 (95% CI [1.10, 1.63]).")
    assert r["value"] == pytest.approx(1.34)
    assert r["sigma"] == pytest.approx((1.63 - 1.10) / (2 * 1.959963984540054))
    assert "ci" in r["flags"]


def test_pm_annotated_as_a_confidence_interval_is_a_half_width():
    r = one("The mean shift is 12.0 ± 1.5 (95% CI) mm.")
    assert r["sigma"] == pytest.approx(1.5e-3 / 1.959963984540054)
    assert "ci_halfwidth" in r["flags"] and r["unit"] == "m"


def test_two_claims_in_one_sentence_keep_their_own_quantities():
    rs = extract_numeric_claims("sin^2(theta_W) = 0.23153 ± 0.00016 and the Z width 2.4952 ± 0.0023 GeV.")
    assert [r["quantity"] for r in rs] == ["sin^2(theta_W)", "Z width"]
    assert rs[0]["unit"] is None and rs[1]["unit"] == "GeV"
    assert rs[0]["span"][1] <= rs[1]["span"][0]


def test_abstain_when_no_quantity_phrase_precedes_the_number():
    assert extract_numeric_claims("172.5 ± 0.7 GeV was the result of the fit.") == []


def test_abstain_when_no_uncertainty_is_stated():
    assert extract_numeric_claims("A new boson of mass 125 GeV was observed at the LHC.") == []


def test_abstain_on_an_upper_limit():
    assert extract_numeric_claims("We set an upper limit of 1.2 × 10^{-9} at 95% CL.") == []


def test_unknown_unit_is_kept_verbatim_and_flagged():
    r = one("The cross section is 12.0 ± 1.5 fb in this channel.")
    assert r["value"] == pytest.approx(12.0) and r["unit"] == "fb" and "unknown_unit" in r["flags"]


def test_latex_source_as_arxiv_abstracts_actually_write_it():
    r = one(r"The top quark mass is measured to be $m_t = 172.95 \pm 0.53$ GeV, the most precise ATLAS result.")
    assert r["value"] == pytest.approx(172.95) and r["sigma"] == pytest.approx(0.53)
    assert r["unit"] == "GeV" and r["quantity"] == "m_t"


def test_latex_macro_unit_and_braced_symbol():
    r = one(r"a top mass of $m_{top} = 176.1\pm 5.1~(\text{stat.})\pm 5.3~(\text{syst.}) \gevcc$ is obtained")
    assert r["quantity"] == "m_top" and r["unit"] == "GeV"
    assert r["sigma"] == pytest.approx(math.hypot(5.1, 5.3)) and "quadrature" in r["flags"]


def test_slash_form_asymmetric_uncertainty():
    r = one("The method yields a top quark mass of 177.8+-4.5/5.0(stat)+-6.2(sys)GeV/c^2.")
    assert r["value"] == pytest.approx(177.8) and r["sigma"] == pytest.approx(math.hypot(5.0, 6.2))
    assert "asymmetric" in r["flags"] and r["quantity"] == "top quark mass"


def test_mathrm_unit_is_a_unit_and_subscript_stat_syst_are_labels():
    r = one(r"a resonance reported with mass $1721.0\pm5.2_{\mathrm{stat.}}\pm3.4_{\mathrm{syst.}}~\mathrm{MeV}$ and")
    assert r["unit"] == "GeV" and r["value"] == pytest.approx(1.7210)
    assert r["sigma"] == pytest.approx(math.hypot(5.2, 3.4) * 1e-3) and "quadrature" in r["flags"]


def test_rm_braced_labels_and_trailing_mathrm_unit():
    r = one(r"The top quark mass is found to be $172.25\pm 0.08\,\rm{(stat+JSF)} \pm 0.62\,\rm{(syst)}~\mathrm{GeV}$.")
    assert r["quantity"] == "top quark mass" and r["unit"] == "GeV"
    assert r["sigma"] == pytest.approx(math.hypot(0.08, 0.62))


def test_normalize_unit_directly():
    assert normalize_unit(1.0, 0.1, "TeV")[:3] == (1000.0, 100.0, "GeV")
    assert normalize_unit(1.0, 0.1, None)[2] is None
    assert normalize_unit(1.0, 0.1, "furlong")[2] == "furlong"


# -- the hand-off to margin_net --------------------------------------------------------------------
def _edge(net, texts, sources_shared=False):
    reports = []
    for k, t in enumerate(texts):
        c = one(t)
        src = "collab" if sources_shared else f"paper{k}"
        reports.append({"margin": c["value"], "sigma": c["sigma"], "sources": [f"{src}#{k}" if sources_shared else src]})
    if sources_shared:
        net.add_sources([{"id": f"collab#{k}", "derives_from": ["collab"]} for k in range(len(texts))])
    net.add_edge("m_t", ["m_t"], reports)
    return net.estimate("m_t")


AGREE = ["ATLAS measures the top quark mass m_t = 172.52 ± 0.33 GeV.",
         "CMS measures the top-quark mass 172.44 ± 0.48 GeV.",
         "The combined top quark mass is 172.69 ± 0.30 GeV."]
CONTRADICT = ["ATLAS measures the top quark mass m_t = 172.52 ± 0.33 GeV.",
              "CMS measures the top-quark mass 176.90 ± 0.40 GeV."]


def test_extracted_reports_feed_margin_net_and_agree():
    est = _edge(MarginNet(), AGREE)
    assert est.n_reports == 3 and est.n_eff == pytest.approx(3.0, abs=1e-9)
    assert 172.3 < est.m < 172.8 and est.s < 0.33
    assert est.p_agree > 0.01 and est.kind == "OK"


def test_chi2_fires_on_a_planted_contradiction_between_two_extractions():
    est = _edge(MarginNet(), CONTRADICT)
    assert est.n_reports == 2 and est.dof == 1
    assert est.q > 50 and est.p_agree < 1e-10
    assert est.kind == "CONTRADICTION"


def test_declared_common_lineage_catches_the_same_contradiction_off_the_chi2_axis():
    """Reports the lineage declares to be copies of one origin must agree; margin_net's part (b) sees it."""
    est = _edge(MarginNet(), CONTRADICT, sources_shared=True)
    assert est.n_eff == pytest.approx(1.0, abs=1e-9)
    assert est.p_agree == 0.0 and est.kind == "CONTRADICTION"
