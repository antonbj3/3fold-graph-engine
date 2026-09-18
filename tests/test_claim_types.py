"""claim_types: the checker layer — a unit grammar and dimension check on numeric records, deductive and statistical
certificates with the reliability a reader may use, and the typecheck a two-step inference has to pass."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from graph_engine.claim_types import (  # noqa: E402
    check_numeric, canonical_dimension, deductive_certificate, dimension, reliability_of, same_dimension,
    statistical_certificate, typecheck_link,
)
from graph_engine.numeric_rules import extract_numeric_claims  # noqa: E402
from graph_engine.record_guarantee import fit_threshold  # noqa: E402


# -- 1. dimensions ------------------------------------------------------------------------------
def test_si_prefixes_and_derived_units():
    assert str(dimension("kg")) == "M" and str(dimension("mg")) == "M" and str(dimension("g")) == "M"
    assert str(dimension("ms")) == "T" and str(dimension("km")) == "L"
    assert str(dimension("Pa")) == str(dimension("mmHg")) == "M·L^-1·T^-2"
    assert str(dimension("N")) == "M·L·T^-2" and str(dimension("Hz")) == "T^-1"
    assert dimension("%").kind == "dimensionless" and dimension("1").kind == "dimensionless"
    assert dimension("events").kind == "count"
    assert dimension("IU") is None and dimension("furlong") is None and dimension("") is None


def test_compound_units_parse_with_exponents():
    assert same_dimension(dimension("kg/m^2"), dimension("kg/m2"))
    assert same_dimension(dimension("mg/dL"), dimension("g/L"))          # mass per volume either way
    assert not same_dimension(dimension("mg/dL"), dimension("mmol/L"))   # mass ≠ amount: a real type error
    assert str(dimension("mL/min")) == "L^3·T^-1"
    assert same_dimension(dimension("GeV/c^2"), dimension("kg"), allow_natural=False)
    assert same_dimension(dimension("fb^-1"), dimension("mb^-1"))


def test_natural_units_only_collapse_mass_and_energy_when_declared():
    gev, kg = dimension("GeV"), dimension("kg")
    assert same_dimension(gev, kg) and not same_dimension(gev, kg, allow_natural=False)
    assert same_dimension(dimension("GeV^-2"), dimension("mb"))          # a cross section in natural units
    assert not same_dimension(gev, dimension("s"), allow_natural=True)   # E vs E^-1: still different
    assert not same_dimension(dimension("events"), dimension("1"))       # a count is not a dimensionless real


def test_canonical_table_resolves_and_abstains():
    assert canonical_dimension("adjusted hazard ratio for death").ratio
    assert canonical_dimension("the top quark mass").name == "mass"
    assert canonical_dimension("systolic blood pressure").name == "blood pressure"
    assert canonical_dimension("median follow-up").name == "duration"
    assert canonical_dimension("the fraction of electroweak events").name == "fraction"
    assert canonical_dimension("the Zeta parameter of the model") is None   # unchecked, not rejected


def test_check_numeric_accepts_well_typed_records():
    ok = [{"quantity": "the top quark mass", "unit": "GeV", "value": 172.5, "sigma": 0.3, "log_scale": False},
          {"quantity": "systolic blood pressure", "unit": "mmHg", "value": -3.2, "sigma": 0.8, "log_scale": False},
          {"quantity": "the production cross section", "unit": "pb", "value": 830.0, "sigma": 40.0, "log_scale": False},
          {"quantity": "median follow-up", "unit": "yr", "value": 3.3, "sigma": 0.2, "log_scale": False},
          {"quantity": "hazard ratio", "unit": None, "value": 0.75, "sigma": 0.06, "log_scale": True},
          {"quantity": "the branching ratio", "unit": "%", "value": 2.1, "sigma": 0.3, "log_scale": False},
          {"quantity": "the Zeta parameter", "unit": None, "value": 1.0, "sigma": 0.1, "log_scale": False}]
    for r in ok:
        c = check_numeric(r)
        assert c["ok"], (r["quantity"], c)


def test_ratio_quantity_with_a_unit_is_a_type_error():
    c = check_numeric({"quantity": "hazard ratio for stroke", "unit": "mmHg", "value": 0.79, "sigma": 0.05,
                       "log_scale": True})
    assert not c["ok"] and c["reason"] == "ratio_quantity_with_unit"
    # log_scale alone (a ratio record from numeric_rules) is enough, whatever the phrase says
    c2 = check_numeric({"quantity": "the effect", "unit": "mm", "value": 1.2, "sigma": 0.1, "log_scale": True})
    assert not c2["ok"] and c2["reason"] == "ratio_quantity_with_unit"


def test_value_with_a_unit_but_no_quantity_is_rejected():
    c = check_numeric({"quantity": None, "unit": "GeV", "value": 125.0, "sigma": 0.2, "log_scale": False})
    assert not c["ok"] and c["reason"] == "value_with_unit_without_quantity"
    assert check_numeric({"quantity": "  ", "unit": None, "value": 1.0, "sigma": 0.1})["reason"] == "no_quantity"


def test_dimension_mismatch_and_missing_unit_are_caught():
    bad = check_numeric({"quantity": "the top quark mass", "unit": "s", "value": 172.5, "sigma": 0.3, "log_scale": False})
    assert not bad["ok"] and bad["reason"].startswith("dimension_mismatch:mass")
    bp = check_numeric({"quantity": "diastolic blood pressure", "unit": "mg/dL", "value": 80.0, "sigma": 2.0})
    assert not bp["ok"] and bp["reason"].startswith("dimension_mismatch:blood pressure")
    nu = check_numeric({"quantity": "the top quark mass", "unit": None, "value": 172.5, "sigma": 0.3})
    assert not nu["ok"] and nu["reason"] == "missing_unit_for_mass"
    uu = check_numeric({"quantity": "the exposure", "unit": "IU", "value": 3.0, "sigma": 0.2})
    assert not uu["ok"] and uu["reason"] == "unknown_unit"


def test_checker_runs_on_real_numeric_rules_output():
    text = ("We measure the top quark mass to be 172.52 $\\pm$ 0.33 GeV. "
            "The adjusted hazard ratio was 0.79 (95% CI, 0.70 to 0.89).")
    recs = extract_numeric_claims(text)
    assert len(recs) >= 2
    assert all(check_numeric(r)["ok"] for r in recs), [(r["quantity"], r["unit"], check_numeric(r)) for r in recs]


# -- 2. certificates ----------------------------------------------------------------------------
def _logistic_growth(h, K=1.0, r=1.0):
    """Y = h·K·(1 − h/r): dY/dh = K(1 − 2h/r) > 0 for h < r/2."""
    return h * K * (1.0 - h / r)


def test_deductive_certificate_certifies_the_box_it_scanned():
    cert = deductive_certificate(_logistic_growth, (0.2, 0.4), {"K": (0.5, 2.0), "r": (1.0, 3.0)}, sign=+1)
    assert cert["ok"] and cert["kind"] == "deductive"
    assert cert["x_range"] == [0.2, 0.4] and cert["n_points"] > 100 and cert["min_abs_slope"] > 0


def test_deductive_certificate_returns_a_counterexample_outside_the_box():
    cert = deductive_certificate(_logistic_growth, (0.2, 0.8), {"K": (0.5, 2.0), "r": (1.0, 3.0)}, sign=+1)
    assert not cert["ok"]
    ce = cert["counterexample"]
    # the first point where K(1 − 2h/r) stops being strictly positive: h ≥ r/2 with r at its lower edge
    assert ce["x"] >= 0.5 and ce["params"]["r"] <= 2 * ce["x"] and ce["dydx"] <= 0
    # what it did verify is a strict sub-interval from the low end, never the requested range
    assert cert["x_range"] is not None and cert["x_range"][0] == 0.2 and cert["x_range"][1] < 0.8
    assert reliability_of(cert, default=0.8, point=0.3) > 0.99
    assert reliability_of(cert, default=0.8, point=0.75) == 0.8


def test_deductive_certificate_on_a_monotone_model_with_no_parameters():
    cert = deductive_certificate(lambda x: -2.0 * x + 1.0, (0.0, 10.0), {}, sign=-1)
    assert cert["ok"] and cert["x_range"] == [0.0, 10.0]
    assert deductive_certificate(lambda x: -2.0 * x + 1.0, (0.0, 10.0), {}, sign=+1)["ok"] is False
    with pytest.raises(ValueError):
        deductive_certificate(lambda x: x, (0.0, 1.0), {}, sign=0)


def test_statistical_certificate_wraps_a_fitted_threshold():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 0.5, 1000)
    err_rate = np.where(scores < 0.25, 0.01, 0.5)        # certain records are almost always entirely right
    all_right = rng.uniform(size=1000) > err_rate
    fit = fit_threshold(scores, all_right, alpha=0.1, delta=0.05)
    assert fit["threshold"] > 0
    cert = statistical_certificate(0.1, 0.05, fit, score=float(fit["threshold"] / 2))
    assert cert["ok"] and cert["threshold"] == fit["threshold"]
    assert reliability_of(cert, default=0.7) == pytest.approx(0.9)
    out = statistical_certificate(0.1, 0.05, fit, score=float(fit["threshold"]) + 0.01)
    assert not out["ok"] and out["reason"] == "record_not_admitted"
    assert reliability_of(out, default=0.7) == 0.7
    # a threshold fitted at a looser level does not certify a stricter one
    loose = statistical_certificate(0.01, 0.05, fit, score=0.0)
    assert not loose["ok"] and loose["reason"].startswith("threshold_fitted_at_looser_level")


def test_reliability_of_defaults_and_no_certificate():
    assert reliability_of(None, 0.85) == 0.85
    assert reliability_of({"kind": "none"}, 0.85) == 0.85
    cert = deductive_certificate(_logistic_growth, (0.2, 0.4), {"K": (0.5, 2.0), "r": (1.0, 3.0)}, sign=+1)
    assert reliability_of(cert, 0.85) == pytest.approx(1 - 1e-6)
    assert reliability_of(cert, 0.85, point={"x": 0.3, "r": 9.0}) == 0.85     # outside the parameter box


# -- 3. link typecheck --------------------------------------------------------------------------
CONCEPTS = [{"id": "blood_pressure", "aliases": ["SBP", "systolic blood pressure"]},
            {"id": "stroke_risk", "aliases": ["stroke"]},
            {"id": "salt_intake", "aliases": ["sodium intake"]}]


def _a(**kw):
    c = {"id": "a", "subject": "salt_intake", "object": "SBP", "sign": +1,
         "validity": {"age_y": (40, 80)}, "units": {"SBP": "mmHg"}}
    c.update(kw); return c


def _b(**kw):
    c = {"id": "b", "subject": "systolic blood pressure", "object": "stroke", "sign": +1,
         "validity": {"age_y": (60, 90)}, "units": {"systolic blood pressure": "mmHg"}}
    c.update(kw); return c


def test_typecheck_accepts_a_composable_two_step():
    r = typecheck_link(_a(), _b(), CONCEPTS)
    assert r["ok"] and r["concept"] == "blood_pressure" and r["sign"] == 1
    assert r["validity"]["age_y"] == (60, 80)


def test_typecheck_rejects_unresolved_and_mismatched_concepts():
    r = typecheck_link(_a(object="pulse pressure"), _b(), CONCEPTS)
    assert not r["ok"] and any(x.startswith("unresolved_concept:pulse") for x in r["reasons"])
    r2 = typecheck_link(_a(object="sodium intake"), _b(), CONCEPTS)
    assert not r2["ok"] and any(x.startswith("shared_concept_mismatch") for x in r2["reasons"])
    amb = CONCEPTS + [{"id": "other_bp", "aliases": ["SBP"]}]
    r3 = typecheck_link(_a(), _b(), amb)
    assert not r3["ok"] and any(x.startswith("ambiguous_alias") for x in r3["reasons"])


def test_typecheck_rejects_disjoint_validity_boxes():
    r = typecheck_link(_a(validity={"age_y": (40, 55)}), _b(), CONCEPTS)
    assert not r["ok"] and any(x.startswith("disjoint_validity:age_y") for x in r["reasons"])


def test_typecheck_rejects_log_against_linear_unless_declared():
    # a's box on the shared variable is in log10(mmHg), b's is linear: the intervals overlap numerically and mean
    # nothing in common. Declaring the same scale on both sides makes the link legal again.
    a = _a(validity={"SBP": (2.0, 2.3)}, scales={"SBP": "log"})
    b = _b(validity={"systolic blood pressure": (120.0, 180.0)})
    r = typecheck_link(a, b, CONCEPTS)
    assert not r["ok"] and any(x.startswith("scale_mismatch") for x in r["reasons"])
    # graph-level declaration alone does not help: a's box is log, b's is not, and b declares linear implicitly
    r2 = typecheck_link(a, b, CONCEPTS, scales={"SBP": "log"})
    assert not r2["ok"] and any(x.startswith("scale_mismatch") for x in r2["reasons"])
    # both sides log, both boxes in log units → ok
    b_log = _b(validity={"systolic blood pressure": (2.08, 2.26)}, scales={"systolic blood pressure": "log"})
    r3 = typecheck_link(a, b_log, CONCEPTS)
    assert r3["ok"], r3["reasons"]


def test_typecheck_rejects_incompatible_units_on_the_shared_variable():
    r = typecheck_link(_a(units={"SBP": "mmol/L"}), _b(), CONCEPTS)
    assert not r["ok"] and any(x.startswith("unit_dimension_mismatch") for x in r["reasons"])
    r2 = typecheck_link(_a(units={"SBP": "kPa"}), _b(), CONCEPTS)     # same dimension, different scale of unit
    assert r2["ok"], r2["reasons"]
    r3 = typecheck_link(_a(units={"SBP": "IU"}), _b(), CONCEPTS)
    assert not r3["ok"] and any(x.startswith("unknown_unit") for x in r3["reasons"])


def test_typecheck_rejects_a_sign_composed_with_a_number():
    num = _b(sign=None, value=0.02, units={"systolic blood pressure": "mmHg", "stroke": "1"})
    r = typecheck_link(_a(), num, CONCEPTS)
    assert not r["ok"] and any(x.startswith("mixed_statement_kinds") for x in r["reasons"])


def test_typecheck_composes_two_numbers_into_one_dimension():
    a = _a(sign=None, value=1.5, units={"salt_intake": "g/day", "SBP": "mmHg"})
    b = _b(sign=None, value=0.02, units={"systolic blood pressure": "mmHg", "stroke": "1"})
    r = typecheck_link(a, b, CONCEPTS)
    assert r["ok"] and r["composed_dimension"] == str(dimension("day/g"))
    bad = _b(sign=None, value=0.02, units={"stroke": "1"})               # no unit for the shared variable
    r2 = typecheck_link(a, bad, CONCEPTS)
    assert not r2["ok"] and any("needs_known_units" in x for x in r2["reasons"])
