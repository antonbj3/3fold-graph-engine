"""EngineProfile: defaults equal the modules' own, validation catches what the modules would silently accept, the
estimable fields come from data, and merging two graphs' profiles keeps per-root numbers and takes the stricter requirement."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from graph_engine.profile import EngineProfile  # noqa: E402
from graph_engine.margin_net import MarginNet  # noqa: E402
from graph_engine.regime_posterior import RegimePosterior  # noqa: E402
from graph_engine.claim_federation import Federation  # noqa: E402


def test_defaults_equal_the_modules_defaults():
    p = EngineProfile().validate()
    net, rp, fed = p.margin_net(), p.regime_posterior(0.0, 1.0), p.federation()
    d = MarginNet(); assert (net.default_sigma, net.stressed_below_z, net.disagree_p, net.copy_tolerance) == (d.default_sigma, d.stressed_below_z, d.disagree_p, d.copy_tolerance)
    r = RegimePosterior(0.0, 1.0); assert (rp.reliability, rp.p_flip, rp.n_grid, rp.p_two, rp.potential) == (r.reliability, r.p_flip, r.n_grid, r.p_two, r.potential)
    f = Federation(); assert (fed.reliability, fed.transitivity, fed.use_lineage) == (f.reliability, f.transitivity, f.use_lineage)


def test_validation_rejects_what_modules_would_accept():
    for bad in [dict(claim_reliability=0.4), dict(default_sigma=0.0), dict(disagree_p=0.0), dict(admission_tau=1.0),
                dict(p_transition=0.9, p_two_transitions=0.2), dict(direction_lexicon={"up": ["rise"], "down": ["rise"]})]:
        try:
            EngineProfile(**bad).validate(); assert False, bad
        except ValueError:
            pass


def test_json_round_trip_and_describe_marks_changes():
    p = EngineProfile(stressed_below_z=3.0, notes="clinical")
    q = EngineProfile.from_json(p.to_json())
    assert q == p and "changed from 2.0" in q.describe()
    try:
        EngineProfile.from_json(json.dumps({"no_such_field": 1})); assert False
    except ValueError:
        pass


def test_estimate_from_data_fills_reliabilities_not_requirements():
    rng = np.random.default_rng(0); rel = {"a": 0.95, "b": 0.85, "c": 0.6}
    fed = Federation(); claims = []
    for q in range(60):
        t = int(rng.choice([-1, 1]))
        for s, r in rel.items():
            claims.append({"id": f"{q}-{s}", "subject": "s", "object": f"o{q}", "sign": t if rng.random() < r else -t, "validity": {}, "evidence": [s]})
    fed.add_graph({"graph_id": "G", "claims": claims, "sources": [{"id": s} for s in rel]})
    p = EngineProfile(stressed_below_z=3.0); changed = p.estimate_from(fed)
    assert "reliability_by_root" in changed and p.reliability_by_root["a"] > p.reliability_by_root["c"]
    assert p.stressed_below_z == 3.0                                  # a requirement, untouched


def test_merge_manufacturing_with_medicine_takes_the_stricter_requirement_and_keeps_roots():
    manuf = EngineProfile(claim_reliability=0.85, reliability_by_root={"lab-A": 0.9}, log_scale_variables=["grain size"],
                          direction_lexicon={"up": ["increase", "harden"], "down": ["decrease", "soften"]}, notes="manufacturing")
    med = EngineProfile(stressed_below_z=3.0, disagree_p=0.001, guarantee_alpha=0.02, reliability_by_root={"trial-1": 0.8, "lab-A": 0.7},
                        direction_lexicon={"up": ["increase", "elevate"], "down": ["decrease", "harden"]}, notes="medicine")
    m, conflicts = manuf.merge(med)
    assert m.stressed_below_z == 3.0 and m.disagree_p == 0.001 and m.guarantee_alpha == 0.02      # stricter governs
    assert m.reliability_by_root == {"lab-A": 0.7, "trial-1": 0.8}                                  # roots kept, lower wins on conflict
    assert m.log_scale_variables == ["grain size"]
    assert "harden" not in m.direction_lexicon["up"] and "harden" not in m.direction_lexicon["down"]  # contradictory word dropped
    fields = {c["field"] for c in conflicts}
    assert {"stressed_below_z", "disagree_p", "guarantee_alpha", "reliability_by_root", "direction_lexicon", "claim_reliability"} <= fields
    assert m.notes == "manufacturing; medicine"
