"""hidden_variable: an overlapping-box disagreement is read as a candidate undeclared variable — from the reports' attributes
(data side) and from a first-principles model (mechanism side); chance-level splits get p ≈ 1."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from graph_engine.hidden_variable import candidates, from_mechanism  # noqa: E402


def _reports(seed=0, n=16, effect=True):
    rng = np.random.default_rng(seed); out = []
    for i in range(n):
        age = float(rng.uniform(20, 80)); dose = float(rng.uniform(1, 10)); site = rng.choice(["A", "B"])
        m = (0.6 if age > 50 else -0.6) if effect else 0.0
        out.append({"margin": m + rng.normal(0, 0.15), "sigma": 0.15, "attributes": {"age": age, "dose": dose, "site": str(site)}})
    return out


def test_planted_modifier_is_found_with_small_p_and_others_are_not():
    c = candidates(_reports(), n_perm=300)
    assert c[0].attribute == "age" and 45 <= c[0].split <= 55 and c[0].p_value < 0.02
    assert c[0].left["mean"] < 0 < c[0].right["mean"]
    others = {x.attribute: x.p_value for x in c[1:]}
    assert all(p > 0.05 for p in others.values()), others


def test_no_effect_gives_chance_level_p():
    c = candidates(_reports(seed=1, effect=False), n_perm=300)
    assert all(x.p_value > 0.05 for x in c), [(x.attribute, x.p_value) for x in c]


def test_missing_attribute_cannot_score_and_signs_work():
    r = _reports()
    for x in r[:10]:
        x["attributes"].pop("dose")
    assert "dose" not in {x.attribute for x in candidates(r, n_perm=50)}
    signed = [{"sign": 1 if x["margin"] > 0 else -1, "attributes": x["attributes"]} for x in r]
    c = candidates(signed, n_perm=100)
    assert c[0].attribute == "age" and c[0].p_value < 0.05


def test_mechanism_side_nominates_the_parameter_that_flips_the_sign():
    """Harvested yield Y = h·K(1 − h/r) against effort h: dY/dh flips at h = r/2. With h ∈ [0.2, 0.8], r ∈ [0.5, 2.0] the
    growth rate r is a hidden axis (flip inside the box); the carrying capacity K only scales Y and never flips it."""
    Y = lambda h, r, K: h * K * (1 - h / r)
    c = from_mechanism(Y, (0.2, 0.8), {"r": (0.5, 2.0), "K": (10.0, 100.0)})
    assert c[0]["attribute"] == "r" and c[0]["score"] > 0.1 and c[0]["flip_threshold"] is not None
    assert c[1]["attribute"] == "K" and c[1]["score"] == 0.0
