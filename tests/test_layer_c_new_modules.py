"""Every newly shipped module that carries a selftest, run under pytest."""
import importlib

import pytest

MODULES = [
    "graph_engine.agreement_admission_gate",
    "graph_engine.admission_bands",
    "graph_engine.black_box_sigma_min_detector",
    "graph_engine.cert_grade_engine",
    "graph_engine.certifying_power_cert",
    "graph_engine.competence_decorrelation_gate",
    "graph_engine.coverage_strata_guard",
    "graph_engine.decorrelated_abstain_field",
    "graph_engine.decorrelated_probe_value",
    "graph_engine.decorrelated_recovery",
    "graph_engine.decorrelation_channel_cert",
    "graph_engine.decorrelation_validity_cert",
    "graph_engine.fluctuation_oed",
    "graph_engine.gauge_aware_admit",
    "graph_engine.gauge_completeness_cert",
    "graph_engine.gauge_kind_cert",
    "graph_engine.generative_leg_admission",
    "graph_engine.identifiability_oed",
    "graph_engine.leg_admission",
    "graph_engine.leg_decorrelation_lineage_gate",
    "graph_engine.member_decorrelation_cert",
    "graph_engine.monotone_agnostic_admission",
    "graph_engine.neff_channel_diversity",
    "graph_engine.neff_form",
    "graph_engine.observability_sigma_min_cert",
    "graph_engine.overdet_budget",
    "graph_engine.overdet_neff",
    "graph_engine.overdet_provenance_check",
    "graph_engine.probe_gauge_cert",
    "graph_engine.render_match_decorrelation_judge",
    "graph_engine.scene_eyes_coverage_audit",
    "graph_engine.stage_decorrelation_verifier",
    "graph_engine.video_admission_agreement_cert",
    "graph_engine.video_admission_cert",
    "graph_engine.video_admission_identifiability_pregate",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_selftest(name):
    mod = importlib.import_module(name)
    fn = getattr(mod, "selftest", None) or getattr(mod, "_selftest", None)
    assert fn is not None, f"{name} has no selftest()"
    r = fn()
    assert r is not False, f"{name} selftest reported a failure"
    if isinstance(r, dict):
        for key in ("all_pass", "ALL_PASS"):
            if key in r:
                assert r[key], f"{name} selftest reported {key}=False"
