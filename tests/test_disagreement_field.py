import numpy as np
import pytest

from graph_engine.claim_types import deductive_certificate
from graph_engine.disagreement_field import (
    LinkedWorld, SignField, claims_sign_field, disagreement_field, gap_correlation, harvest,
    link_scores, model_probes, model_sign_field, resonance, stress_probe,
)
from graph_engine.regime_posterior import RegimePosterior


def _post(claims, **kw):
    rp = RegimePosterior(0.0, 1.0, n_grid=24, p_two=0.0, **kw)
    for a, b, s in claims:
        rp.add_claim(a, b, s)
    return rp


def test_model_field_is_one_or_zero_where_the_sign_is_certified():
    up = lambda x, a=1.0: a * x                      # dy/dx = a > 0 on the whole box
    down = lambda x, a=1.0: -a * x
    box = {"a": (0.5, 2.0)}
    cu = deductive_certificate(up, (0.0, 1.0), box, sign=1)
    cd = deductive_certificate(down, (0.0, 1.0), box, sign=-1)
    assert cu["ok"] and cd["ok"]
    fu = model_sign_field(up, (0.0, 1.0), box, n=40, n_p=7)
    fd = model_sign_field(down, (0.0, 1.0), box, n=40, n_p=7)
    assert np.allclose(fu.p_plus, 1.0)
    assert np.allclose(fd.p_plus, 0.0)
    assert fu.source == "model:<lambda>"


def test_uncertified_theta_box_gives_the_ramp_and_the_two_families_agree():
    box = {"theta": (0.2, 0.8)}
    assert deductive_certificate(harvest, (0.0, 1.0), box, sign=1)["ok"] is False
    fh = model_sign_field(harvest, (0.0, 1.0), box, n=101, n_p=61)
    fr = model_sign_field(resonance, (0.0, 1.0), box, n=101, n_p=61)
    # P₊(x) = share of the θ grid above x: 1 below 0.2, 0 above 0.8, a decreasing ramp between
    assert fh.p_plus[0] == pytest.approx(1.0)
    assert fh.p_plus[-1] == pytest.approx(0.0)
    assert np.all(np.diff(fh.p_plus) <= 1e-12)
    assert fh.p_plus[50] == pytest.approx(1.0 - (0.5 - 0.2) / 0.6, abs=0.03)
    # same sign structure, different formula — except at x = 0, where the Lorentzian is even and its derivative is
    # exactly 0, so the field takes the undeclared value ½ there (the rule that keeps a peak's apex from biasing a side)
    assert fr.p_plus[0] == 0.5
    assert np.max(np.abs(fh.p_plus[1:] - fr.p_plus[1:])) < 0.05


def test_disagreement_is_zero_when_the_model_equals_the_claims():
    rp = _post([(0.0, 0.5, +1), (0.5, 1.0, -1)])
    fc = claims_sign_field(rp, n=60)
    fm = SignField(fc.xs, fc.p_plus.copy(), "model:copy")
    d = disagreement_field(fm, fc)
    assert d.integral_abs == pytest.approx(0.0, abs=1e-12)
    assert d.integral_js == pytest.approx(0.0, abs=1e-12)
    assert stress_probe(d)[1] == pytest.approx(0.0, abs=1e-12)


def test_js_is_in_bits_and_bounded_and_grows_with_confident_opposition():
    xs = np.linspace(0.0, 1.0, 51)
    opposed = disagreement_field(np.full(51, 0.02), np.full(51, 0.98), xs=xs)
    mild = disagreement_field(np.full(51, 0.30), np.full(51, 0.70), xs=xs)
    assert 0.0 <= opposed.integral_js <= 1.0
    assert opposed.integral_js == pytest.approx(0.858, abs=0.01)   # 1 bit domain × JS(0.02, 0.98)
    assert opposed.integral_abs == pytest.approx(0.96, abs=1e-9)
    assert opposed.integral_js > 4 * mild.integral_js              # |Δ| only 2.4× larger
    with pytest.raises(ValueError):
        disagreement_field(np.zeros(5), np.zeros(4), xs=np.zeros(5))


def test_stress_probe_lands_in_the_gap_between_a_plus_claim_and_a_minus_model():
    # claims: a confident + on [0, 0.5], nothing above; model: certified − everywhere
    rp = _post([(0.0, 0.5, +1)], reliability=0.95)
    fc = claims_sign_field(rp, n=201)
    fm = model_sign_field(lambda x, a=1.0: -a * x, (0.0, 1.0), {"a": (0.5, 2.0)}, n=201, n_p=5)
    d = disagreement_field(fm, fc)
    x, v = stress_probe(d)
    assert v > 0.5                                    # confident + against a certified −: most of a bit
    assert 0.0 <= x <= 0.5                            # inside the + claim, not in the unclaimed span
    assert d.js[np.argmin(np.abs(d.xs - x))] == pytest.approx(v)
    assert stress_probe(d, measure="abs")[1] == pytest.approx(float(d.abs_gap.max()))
    with pytest.raises(ValueError):
        stress_probe(d, measure="nope")


def test_k_model_probes_carry_the_evidence_of_one():
    claims = [(0.0, 0.4, +1), (0.6, 1.0, -1)]
    fm = model_sign_field(harvest, (0.0, 1.0), {"theta": (0.2, 0.8)}, n=41, n_p=21)
    many = _post(claims)
    one = _post(claims)
    recs = model_probes(many, fm, xs=np.full(64, 0.1), r_model=0.9, name="harvest")
    one.add_probe(0.1, +1, reliability=0.9, weight=1.0)
    assert len(recs) == 64 and sum(r["weight"] for r in recs) == pytest.approx(1.0)
    assert all(r["lineage"] == "model:harvest" for r in recs)
    assert len(many.model_probe_lineage) == 64
    for x in (0.05, 0.3, 0.5, 0.75, 0.95):
        assert many.p_plus(x) == pytest.approx(one.p_plus(x), abs=1e-12)
    # and K answers are strictly weaker than K INDEPENDENT answers at the same point
    naive = _post(claims)
    for _ in range(64):
        naive.add_probe(0.1, +1, reliability=0.9, weight=1.0)
    assert naive.p_plus(0.1) > many.p_plus(0.1)


def test_planted_hidden_link_raises_the_field_correlation_above_unlinked_pairs():
    fm = model_sign_field(harvest, (0.0, 1.0), {"theta": (0.2, 0.8)}, n=80, n_p=41)
    linked, unlinked = [], []
    for seed in range(8):
        w = LinkedWorld(n_pairs=8, n_links=2, seed=seed)
        assert w.true_links() == {(0, 1), (2, 3)}
        assert w.is_link(1, 0) and not w.is_link(0, 2)
        assert abs(w.theta[0] - w.theta[1]) < 1e-12
        posts = [RegimePosterior(0.0, 1.0, n_grid=24, p_two=0.0) for _ in range(w.n_pairs)]
        for p, a, b, s, _roots in w.claims:
            posts[p].add_claim(a, b, s)
        ds = [disagreement_field(fm, claims_sign_field(rp, n=80)) for rp in posts]
        C = link_scores(ds)
        base = np.mean([d.gap for d in ds], axis=0)
        for p in range(w.n_pairs):
            for q in range(p + 1, w.n_pairs):
                assert C[p, q] == pytest.approx(gap_correlation(ds[p], ds[q], baseline=base), abs=1e-9)
                (linked if w.is_link(p, q) else unlinked).append(C[p, q])
    assert len(linked) == 16
    assert np.mean(linked) > np.mean(unlinked) + 0.15
    # the RAW correlation (no cohort baseline) does not separate: every pair of the family has the same + → − shape
    raw = [gap_correlation(ds[0], ds[q]) for q in range(1, 8)]
    assert min(raw) > 0.85


def test_linked_world_keeps_the_claim_structure_and_the_planted_truth():
    base = LinkedWorld(n_pairs=8, n_links=2, seed=3)
    assert all(len(t) == 1 for t in base.transitions)
    assert all(s == 1 for s in base.start_sign)
    for p in range(base.n_pairs):
        t = base.transitions[p][0]
        assert base.sign(p, max(t - 0.05, 0.0)) == 1 and base.sign(p, min(t + 0.05, 1.0)) == -1
    keys = [(p, round(a, 9), round(b, 9), tuple(sorted(roots))) for p, a, b, _s, roots in base.claims]
    assert len(keys) > len(set(keys))                               # the copy structure of World survives the replant
    for k in set(keys):                                             # and a copy still repeats its parent's SIGN
        assert len({s for (p, a, b, s, r), kk in zip(base.claims, keys) if kk == k}) == 1
    agree = np.mean([s == base.majority_sign(p, a, b) for p, a, b, s, _ in base.claims])
    assert agree > 0.7                                              # claims follow the PLANTED truth, up to source noise
    assert base.model_of(0) in (harvest, resonance)
    with pytest.raises(ValueError):
        LinkedWorld(n_pairs=3, n_links=2, seed=0)
