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
    # an odds ratio is a RATIO: the interval is symmetric in the log, so σ is the log-scale one (see log-scale test)
    r = one("The odds ratio was 1.34 (95% CI [1.10, 1.63]).")
    assert r["value"] == pytest.approx(1.34)
    assert r["sigma"] == pytest.approx((math.log(1.63) - math.log(1.10)) / (2 * 1.959963984540054))
    assert "ci" in r["flags"] and r["log_scale"] is True and r["quantity"] == "odds ratio"


def test_confidence_interval_linear_when_the_quantity_is_not_a_ratio():
    r = one("The measured cross section is 8.6 (95% CI 7.9, 9.3) pb.")
    assert r["sigma"] == pytest.approx((9.3 - 7.9) / (2 * 1.959963984540054))
    assert r["log_scale"] is False and "log_scale" not in r["flags"]


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


# -- medical reporting surfaces (paraphrased fragments of PubMed abstracts on blood-pressure trials) ------------
# numeric_rules used to need the value glued to the interval; medicine puts a comma or a semicolon in between and
# the label ("HR", "adjusted hazard ratio for stroke") before it. These are the forms measured on the cached corpus.
Z95 = 1.959963984540054


def log_sigma(lo, hi):
    return (math.log(hi) - math.log(lo)) / (2 * Z95)


def test_semicolon_before_the_interval_and_a_worded_label():
    r = one("The hazard ratio for cardiovascular events was 0.79; 95% CI, 0.70 to 0.89.")
    assert r["quantity"] == "hazard ratio for cardiovascular events"
    assert r["value"] == pytest.approx(0.79) and r["sigma"] == pytest.approx(log_sigma(0.70, 0.89))
    assert r["log_scale"] is True and "ci" in r["flags"]


def test_comma_before_the_interval():
    r = one("Death was less frequent with intensive therapy (relative risk 0.75, 95% CI 0.64-0.89).")
    assert r["quantity"] == "relative risk" and r["value"] == pytest.approx(0.75)
    assert r["sigma"] == pytest.approx(log_sigma(0.64, 0.89)) and r["log_scale"] is True


def test_interval_without_a_stated_value_takes_the_geometric_mean_for_a_ratio():
    r = one("The pooled hazard ratio, 95% CI 0.64 to 0.89, excluded unity.")
    assert r["value"] == pytest.approx(math.sqrt(0.64 * 0.89)) and r["log_scale"] is True


def test_parenthesised_label_value_interval_and_p_value():
    r = one("Intensive control reduced major cardiovascular events (HR, 0.75; 95% CI, 0.64-0.89; P<0.001).")
    assert r["quantity"] == "HR" and r["value"] == pytest.approx(0.75)
    assert r["sigma"] == pytest.approx(log_sigma(0.64, 0.89)) and r["log_scale"] is True


def test_equals_sign_spaced_percent_and_the_spelled_out_interval():
    r = one("RR = 0.79 (95 % confidence interval: 0.70 to 0.89) for the primary outcome.")
    assert r["quantity"] == "RR" and r["value"] == pytest.approx(0.79)
    assert r["sigma"] == pytest.approx(log_sigma(0.70, 0.89)) and r["log_scale"] is True


def test_level_free_bracket_behind_a_label_is_read_as_95_percent():
    r = one("Adjusted odds ratio 1.28 [1.05, 1.56] for incident heart failure.")
    assert r["quantity"] == "Adjusted odds ratio" and r["value"] == pytest.approx(1.28)
    assert r["sigma"] == pytest.approx(log_sigma(1.05, 1.56))
    assert "ci_implied" in r["flags"] and r["log_scale"] is True


def test_level_free_bracket_without_a_label_is_an_abstention():
    assert extract_numeric_claims("A total of 4733 participants [2371, 2362] were randomised.") == []


def test_mean_difference_keeps_the_linear_scale_and_its_unit():
    r = one("The between-group difference in systolic pressure was -14.8 mmHg (95% CI -16.6 to -13.0).")
    assert r["value"] == pytest.approx(-14.8) and r["log_scale"] is False
    assert r["sigma"] == pytest.approx((-13.0 - -16.6) / (2 * Z95))
    assert r["unit"] == "mmHg" and "unknown_unit" in r["flags"]


def test_absolute_risk_reduction_in_percent_is_a_fraction_and_not_a_ratio():
    r = one("The absolute risk reduction was 2.1%, 95% CI 0.9-3.3.")
    assert r["quantity"] == "absolute risk reduction" and r["log_scale"] is False
    assert r["value"] == pytest.approx(0.021) and r["sigma"] == pytest.approx(0.024 / (2 * Z95))


def test_a_value_outside_its_own_interval_is_not_the_point_estimate():
    """"...in 9361 participants, 95% CI 0.70-0.89" — the number across the separator is an n, not the estimate."""
    r = one("Fewer events occurred among the 9361 adults, 95% CI 0.70-0.89, in the intensive arm.")
    assert r["value"] == pytest.approx((0.70 + 0.89) / 2) and "value_outside_ci" in r["flags"]


def test_log_scale_record_hands_margin_net_the_log_of_the_value():
    """Two ratio reports of the same effect: combined in the log, as the record's log_scale key instructs."""
    a = one("The hazard ratio for death was 0.75; 95% CI, 0.64 to 0.89.")
    b = one("A second trial reported an adjusted hazard ratio of 0.83 (95% CI 0.72-0.96).")
    assert a["log_scale"] and b["log_scale"]
    net = MarginNet()
    net.add_edge("hr", ["hr"], [{"margin": math.log(r["value"]), "sigma": r["sigma"], "sources": [str(i)]}
                                for i, r in enumerate((a, b))])
    est = net.estimate("hr")            # margin_net's kind reads a negative margin as VIOLATED; log HR < 0 here,
    assert est.p_agree > 0.05           # so the question asked of it is agreement, not the sign
    assert 0.75 < math.exp(est.m) < 0.83 and est.s < min(a["sigma"], b["sigma"])


def test_decimal_comma_interval_is_an_abstention_not_a_garbage_claim():
    """Found in the cached corpus: an author writing "0,123-0,594" means 0.123-0.594, and reading the commas as
    the interval separator produced the claim 81 ± 41. No claim is the only honest reading here."""
    assert extract_numeric_claims("Previous disease was a predictor (0,270, 95% CI 0,123-0,594, P<.001).") == []
