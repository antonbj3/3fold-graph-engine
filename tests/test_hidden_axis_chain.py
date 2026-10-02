"""hidden_axis_chain: every statement in the module docstring that can be checked numerically is checked here."""
import math
import sys
from pathlib import Path

import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from graph_engine.hidden_axis_chain import (  # noqa: E402
    coverage_audit,
    finite_delivery_certificate,
    moment_feasibility,
    instrument_can_decide,
    required_new_measurement,
    run_chain,
)


# ---- moment_feasibility: the Bhatia-Davis bound, which is exact and not an approximation ----

def test_bhatia_davis_bound_is_the_exact_product():
    """Var <= (hi - mean)(mean - lo). With support [0, 10] and mean 2 that is exactly 16."""
    r = moment_feasibility((0.0, 10.0), 2.0, 1.0)
    assert r["variance_upper_bound"] == pytest.approx(16.0, abs=0.0)
    assert r["sd_upper_bound"] == pytest.approx(4.0)
    assert r["moments_feasible"] is True


def test_bound_rejects_an_sd_just_above_it():
    assert moment_feasibility((0.0, 10.0), 2.0, 4.0)["moments_feasible"] is True
    assert moment_feasibility((0.0, 10.0), 2.0, 4.001)["moments_feasible"] is False


def test_mean_outside_support_gives_zero_bound_not_a_large_one():
    """A mean outside the declared support must not yield a permissive bound."""
    r = moment_feasibility((0.0, 10.0), 11.0, 0.1)
    assert r["mean_feasible"] is False
    assert r["variance_upper_bound"] == 0.0
    assert r["moments_feasible"] is False


def test_sample_sd_correction_is_n_over_n_minus_one():
    r = moment_feasibility((0.0, 10.0), 2.0, 1.0, sample_n=5)
    assert r["finite_sample_variance_correction"] == pytest.approx(1.25)
    assert r["variance_upper_bound"] == pytest.approx(20.0)


def test_sample_sd_needs_at_least_two_observations():
    with pytest.raises(ValueError):
        moment_feasibility((0.0, 10.0), 2.0, 1.0, sample_n=1)


def test_identification_is_never_claimed():
    """An attainable pair of moments does not identify a mechanism; the field is fixed False."""
    assert moment_feasibility((0.0, 10.0), 2.0, 1.0)["identified"] is False


# ---- coverage_audit: the reasons an empty candidate list can have ----

def _rep(i, **attrs):
    return dict(id=f"r{i}", margin=1.0, sigma=0.5, sources=[f"s{i}"], attributes=attrs)


def test_empty_is_never_read_as_absence():
    assert coverage_audit([_rep(i, a=1) for i in range(4)])["empty_means_absence"] is False


def test_n_too_small_is_named():
    audit = coverage_audit([_rep(0, a=1), _rep(1, a=2)])
    assert audit["n"] == 2 and audit["n_required"] == 4
    assert "N_TOO_SMALL" in audit["attributes"]["a"]["reasons"]


def test_coverage_too_low_is_named():
    """Three of four reports leave the attribute unknown, so coverage is 0.25 < 0.7."""
    reports = [_rep(0, a=1), _rep(1, a=None), _rep(2, a=None), _rep(3, a=None)]
    row = coverage_audit(reports)["attributes"]["a"]
    assert row["n_known"] == 1
    assert row["coverage"] == pytest.approx(0.25)
    assert "COVERAGE_TOO_LOW" in row["reasons"]
    assert row["eligible"] is False


def test_no_two_vs_two_split_is_named():
    """Four reports, but one value sits alone on its side, so no split has >= 2 per side."""
    reports = [_rep(0, a=1), _rep(1, a=1), _rep(2, a=1), _rep(3, a=2)]
    row = coverage_audit(reports)["attributes"]["a"]
    assert row["n_known"] == 4
    assert row["admissible_splits"] == 0
    assert "NO_TWO_VS_TWO_SPLIT" in row["reasons"]


def test_an_eligible_attribute_has_no_reasons():
    reports = [_rep(0, a=1), _rep(1, a=1), _rep(2, a=2), _rep(3, a=2)]
    row = coverage_audit(reports)["attributes"]["a"]
    assert row["admissible_splits"] == 1
    assert row["reasons"] == []
    assert row["eligible"] is True


def test_categorical_attributes_are_split_by_level():
    reports = [_rep(0, a="x"), _rep(1, a="x"), _rep(2, a="y"), _rep(3, a="y")]
    assert coverage_audit(reports)["attributes"]["a"]["admissible_splits"] == 2


# ---- required_new_measurement: the threshold is the Gaussian certify() boundary ----

def test_threshold_is_exactly_where_the_posterior_z_reaches_the_critical_value():
    est_m, est_s, sigma_new, alpha = 0.4, 0.5, 0.3, 0.05
    out = required_new_measurement(est_m, est_s, sigma_new, alpha=alpha)
    post_s = out["posterior_sd"]
    assert post_s == pytest.approx((1 / est_s**2 + 1 / sigma_new**2) ** -0.5)
    observed = out["observed_margin_threshold"]
    post_m = post_s**2 * (est_m / est_s**2 + observed / sigma_new**2)
    assert post_m / post_s == pytest.approx(norm.ppf(1 - alpha))


def test_a_sharper_new_instrument_lowers_the_required_margin():
    loose = required_new_measurement(0.4, 0.5, 1.0)["observed_margin_threshold"]
    sharp = required_new_measurement(0.4, 0.5, 0.1)["observed_margin_threshold"]
    assert sharp < loose


# ---- finite_delivery_certificate: a set certificate, so the bounds are the whole claim ----

def test_ratio_bounds_are_the_product_of_the_input_bounds():
    r = finite_delivery_certificate(content_ratio_bounds=(1.1, 1.2), flow_ratio_bounds=(1.3, 1.4))
    assert r["delivery_ratio_bounds"] == pytest.approx([1.3 * 1.1, 1.4 * 1.2])
    assert r["decision"] == "FASTER"
    assert r["faster_margin"] == pytest.approx(1.3 * 1.1 - 1.0)


def test_an_interval_straddling_one_is_unknown_not_a_coin_flip():
    r = finite_delivery_certificate(content_ratio_bounds=(0.8, 1.2), flow_ratio_bounds=(0.9, 1.1))
    assert r["delivery_ratio_bounds"][0] < 1.0 < r["delivery_ratio_bounds"][1]
    assert r["decision"] == "UNKNOWN"


def test_upper_bound_below_one_is_slower():
    assert finite_delivery_certificate(content_ratio_bounds=(0.5, 0.6),
                                       flow_ratio_bounds=(0.7, 0.8))["decision"] == "SLOWER"


def test_a_broken_assumption_forces_unknown_even_with_a_decisive_interval():
    r = finite_delivery_certificate(content_ratio_bounds=(1.1, 1.2), flow_ratio_bounds=(1.3, 1.4),
                                    same_capacity=False)
    assert r["delivery_ratio_bounds"][0] > 1.0
    assert r["decision"] == "UNKNOWN"


@pytest.mark.parametrize("content,flow", [((0.0, 1.0), (1.0, 2.0)), ((1.0, 2.0), (0.0, 1.0)),
                                          ((2.0, 1.0), (1.0, 2.0)), ((1.0, 2.0), (2.0, 1.0))])
def test_non_positive_or_unordered_bounds_raise(content, flow):
    with pytest.raises(ValueError):
        finite_delivery_certificate(content_ratio_bounds=content, flow_ratio_bounds=flow)


# ---- run_chain: the composition itself ----

def _chain(reports, sources, **kw):
    return run_chain(name="observable", model=lambda x, a: a * x + 1.0, x_range=(1.0, 2.0),
                     params={"a": (1.0, 2.0)}, reports=reports, sources=sources,
                     n_x=11, n_p=5, n_perm=50, **kw)


def _four_reports():
    return [dict(id=f"r{i}", margin=0.5 + 0.1 * i, sigma=0.2, sources=[f"s{i}"],
                 attributes={"grade": i < 2}, validity={}, locator=f"doc{i}")
            for i in range(4)]


def test_zero_sign_flips_is_a_result_not_a_failure():
    """a*x + 1 has a positive slope for every a in [1, 2], so no parameter can flip the sign."""
    out = _chain(_four_reports(), [dict(id=f"s{i}") for i in range(4)])
    assert all(row["score"] == 0 for row in out["mechanism_sweep"])
    assert out["no_flip_gate"] == "NO_ADDED_SIGN_INFORMATION"


def test_root_overlapping_reports_are_omitted_from_the_certificate_subset():
    reports = _four_reports()
    reports[1]["sources"] = ["s0"]                      # same root as r0
    out = _chain(reports, [dict(id=f"s{i}") for i in range(4)])
    omitted = {o["id"]: o["reason"] for o in out["data"]["omitted"]}
    assert omitted == {"r1": "ROOT_OVERLAP"}
    assert "r1" not in out["certificate_report_subset"]
    assert out["federation"]["n_reports"] == 4          # federation keeps it; only GLS thins


def test_identification_stays_unknown_without_the_explicit_gates():
    """measured_axes empty and predictive_inputs_external False must not yield a candidate."""
    out = _chain(_four_reports(), [dict(id=f"s{i}") for i in range(4)])
    assert out["physical_identification"] == "UNKNOWN"


def test_a_report_without_uncertainty_is_refused():
    reports = _four_reports()
    reports[0]["sigma"] = 0.0
    with pytest.raises(ValueError):
        _chain(reports, [dict(id=f"s{i}") for i in range(4)])


def test_a_report_without_sources_is_refused():
    reports = _four_reports()
    reports[0]["sources"] = []
    with pytest.raises(ValueError):
        _chain(reports, [dict(id=f"s{i}") for i in range(4)])


def test_the_result_is_json_ready():
    import json
    out = _chain(_four_reports(), [dict(id=f"s{i}") for i in range(4)])
    json.dumps(out)                                     # raises if any numpy or dataclass leaked
    assert math.isfinite(out["costs"]["wall_s"])


# ---- instrument_can_decide: two refusals that are different failures ----

Z95 = norm.ppf(0.95)


def test_with_no_prior_the_required_observation_is_exactly_sigma_times_z():
    """The closed form the whole precision test rests on, so it is pinned rather than assumed."""
    out = instrument_can_decide(est_m=0.5, est_s=1e9, sigma_new=3.96,
                                support=(0.0, 20.0), decision_band=(1.0, 5.0))
    assert out["required_observation"] == pytest.approx(3.96 * Z95, rel=1e-9)


def test_a_wide_support_makes_the_precision_test_non_binding():
    """Measured negative: sigma 3.96 needs an observation of 6.51, which [0, 20] contains."""
    out = instrument_can_decide(est_m=0.5, est_s=2.0, sigma_new=3.96,
                                support=(0.0, 20.0), decision_band=(1.0, 5.0))
    assert out["required_observation_in_support"] is True
    assert "REQUIRED_OBSERVATION_OUTSIDE_SUPPORT" not in out["refusal_reasons"]


def test_a_large_resolution_ratio_alone_does_not_refuse():
    """The ratio is reported for the reader and decides nothing: no new threshold is introduced."""
    out = instrument_can_decide(est_m=0.5, est_s=2.0, sigma_new=3.96,
                                support=(0.0, 20.0), decision_band=(1.0, 5.0))
    assert out["resolution_ratio"] == pytest.approx(3.96)
    assert out["verdict"] == "CAN_DECIDE"


def test_the_precision_refusal_fires_when_sigma_times_z_leaves_the_support():
    out = instrument_can_decide(est_m=0.5, est_s=2.0, sigma_new=3.96,
                                support=(0.0, 6.0), decision_band=(1.0, 5.0))
    assert 3.96 * Z95 > 6.0
    assert out["required_observation_in_support"] is False
    assert out["verdict"] == "CANNOT_DECIDE"


def test_the_refusal_names_a_sigma_that_would_decide_it():
    out = instrument_can_decide(est_m=0.5, est_s=2.0, sigma_new=3.96,
                                support=(0.0, 6.0), decision_band=(1.0, 5.0))
    better = out["sigma_required"]
    assert better is not None and better < 3.96
    again = instrument_can_decide(est_m=0.5, est_s=2.0, sigma_new=better,
                                  support=(0.0, 6.0), decision_band=(1.0, 5.0))
    assert again["required_observation_in_support"] is True


def test_sampling_part_of_the_declared_domain_refuses_however_precise_the_reading():
    """Coverage is not precision: the instrument reads only where the probe touched."""
    out = instrument_can_decide(est_m=0.5, est_s=2.0, sigma_new=1e-9,
                                support=(0.0, 20.0), decision_band=(1.0, 5.0),
                                declared_domain=range(40), sampled_domain=range(6))
    assert out["required_observation_in_support"] is True        # precision is fine ...
    assert out["coverage"] == "SUBSET_OF_DECLARED_DOMAIN"        # ... and it still cannot decide
    assert out["refusal_reasons"] == ["INSTRUMENT_SAMPLES_A_SUBSET_OF_THE_DECLARED_DOMAIN"]
    assert out["verdict"] == "CANNOT_DECIDE"


def test_full_coverage_passes():
    out = instrument_can_decide(est_m=0.5, est_s=2.0, sigma_new=1e-9,
                                support=(0.0, 20.0), decision_band=(1.0, 5.0),
                                declared_domain=range(6), sampled_domain=range(6))
    assert out["coverage"] == "FULL" and out["verdict"] == "CAN_DECIDE"


def test_an_unasked_coverage_question_is_unchecked_not_passed():
    out = instrument_can_decide(est_m=0.5, est_s=2.0, sigma_new=3.96,
                                support=(0.0, 20.0), decision_band=(1.0, 5.0))
    assert out["coverage"] == "UNCHECKED"


def test_a_sampled_point_outside_the_declared_domain_raises():
    with pytest.raises(ValueError):
        instrument_can_decide(est_m=0.5, est_s=2.0, sigma_new=1.0, support=(0.0, 20.0),
                              decision_band=(1.0, 5.0), declared_domain=range(3), sampled_domain=range(5))


def test_a_derived_sigma_is_reported_as_derived():
    out = instrument_can_decide(est_m=0.5, est_s=2.0, sigma_new=3.96, support=(0.0, 20.0),
                                decision_band=(1.0, 5.0), sigma_source="derived_from_R2")
    assert out["sigma_source"] == "derived_from_R2"


@pytest.mark.parametrize("support,band", [((20.0, 0.0), (1.0, 5.0)), ((0.0, 20.0), (5.0, 1.0)),
                                          ((0.0, 20.0), (0.0, 5.0))])
def test_malformed_support_or_band_raises(support, band):
    with pytest.raises(ValueError):
        instrument_can_decide(est_m=0.5, est_s=2.0, sigma_new=1.0, support=support, decision_band=band)


# ---- the sign gate must say which sweep earned it ----

def _chain_model(model, params):
    return run_chain(name="observable", model=model, x_range=(1.0, 2.0), params=params,
                     reports=_four_reports(), sources=[dict(id=f"s{i}") for i in range(4)],
                     n_x=11, n_p=5, n_perm=20)


def test_a_diagonal_flip_is_not_reported_as_no_information():
    """The axis sweep sees nothing; the joint scan over the product does, and the verdict says which."""
    out = _chain_model(lambda x, a, b: (a * b - 1.38) * x, {"a": (1.0, 1.2), "b": (1.0, 1.2)})
    assert [r["score"] for r in out["mechanism_sweep"]] == [0.0, 0.0]
    assert out["no_flip_gate"] == "SIGN_REVERSED_IN_JOINT_BOX"
    assert out["joint_box_certificate"]["ok"] is False
    assert out["midpoint_sign"] == -1


def test_the_strong_verdict_survives_for_a_genuinely_sign_free_box():
    out = _chain_model(lambda x, a: a * x + 1.0, {"a": (1.0, 2.0)})
    assert out["no_flip_gate"] == "NO_ADDED_SIGN_INFORMATION"
    assert out["joint_box_certificate"]["ok"] is True


def test_an_axis_sweep_that_finds_the_flip_itself_still_reports_it():
    out = _chain_model(lambda x, a: (a - 1.5) * x, {"a": (1.0, 2.0)})
    assert any(r["score"] > 0 for r in out["mechanism_sweep"])
    assert out["no_flip_gate"] == "SIGN_REVERSAL_PRESENT"


def test_a_flat_box_is_degenerate_not_reversed():
    """Slope a - 1.0 over a in [1, 2] is zero only at the lower endpoint: no sign runs backwards."""
    out = _chain_model(lambda x, a: (a - 1.0) * x, {"a": (1.0, 2.0)})
    cert = out["joint_box_certificate"]
    assert cert["ok"] is False
    assert cert["n_opposite_sign"] == 0 and cert["n_zero_slope"] > 0
    assert out["no_flip_gate"] == "SIGN_DEGENERATE_IN_JOINT_BOX"


def test_the_diagonal_box_is_reversed_and_the_counts_say_so():
    """The first witness is a zero, so the verdict must come from the counts and not from the witness."""
    out = _chain_model(lambda x, a, b: (a * b - 1.38) * x, {"a": (1.0, 1.2), "b": (1.0, 1.2)})
    cert = out["joint_box_certificate"]
    assert cert["counterexample"]["dydx"] == 0.0            # the witness alone looks merely degenerate
    assert cert["n_opposite_sign"] > 0                      # but a strict reversal does exist
    assert cert["failure"] == "opposite_sign"
    assert out["no_flip_gate"] == "SIGN_REVERSED_IN_JOINT_BOX"
