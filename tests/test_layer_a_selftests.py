"""Each Layer A module carries its own selftest(); run them all under pytest."""
import importlib

import pytest

MODULES = [
    "graph_engine.cert_decorrelation",
    "graph_engine.fleet_oed",
    "graph_engine.overdet_precondition_gate",
    "graph_engine.scene_eyes_neff_leg",
    "graph_engine.reliability_gate_decorr_check",
    "graph_engine.gauge_detector",
    "graph_engine.gauge_catalog",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_selftest(name):
    mod = importlib.import_module(name)
    fn = getattr(mod, "selftest", None) or getattr(mod, "_selftest", None)
    assert fn is not None, f"{name} has no selftest()"
    assert fn() is not False, f"{name} selftest reported a failure"
