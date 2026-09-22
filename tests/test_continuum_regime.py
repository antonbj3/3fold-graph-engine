"""Contracts for the resolution-free regime backend (``CompiledRegime(n_grid=None)``).

References are independent of the closed forms: brute-force midpoint integration of the raw
model definition (G evaluated pointwise from claims/probes, transitions integrated numerically),
and, when KERNEL_ENGINE_SRC/BRIDGE_BUILD_DIR are set, the native kernel at a fine grid with a
declared O(1/n) tolerance.
"""
from __future__ import annotations

import importlib
import math
import os
import sys

import numpy as np
import pytest

from graph_engine.compiled_regime import CompiledRegime, FrozenPartitionError
from graph_engine.continuum_regime import ContinuumRegime

_KERNEL_SRC = os.environ.get("KERNEL_ENGINE_SRC")
_BUILD = os.environ.get("BRIDGE_BUILD_DIR")
needs_kernel = pytest.mark.skipif(not (_KERNEL_SRC and _BUILD),
                                  reason="KERNEL_ENGINE_SRC/BRIDGE_BUILD_DIR unset")


def _instance(seed, p_two=0.05):
    rng = np.random.default_rng(seed)
    claims = []
    for _ in range(int(rng.integers(1, 4))):
        a = float(rng.uniform(0.0, 0.7))
        claims.append((a, float(rng.uniform(a + 0.05, 1.0)), int(rng.choice([-1, 1])),
                       float(rng.uniform(0.5, 4.0)), float(rng.uniform(0.6, 0.9))))
    probes = [(float(rng.uniform(0, 1)), int(rng.choice([-1, 1])), float(rng.uniform(0.6, 0.9)),
               float(rng.uniform(0.5, 2.0))) for _ in range(int(rng.integers(0, 5)))]
    return dict(claims=claims, probes=probes, p_flip=0.3, p_two=p_two)


def _cont(inst, **kw):
    return CompiledRegime(0.0, 1.0, backend=None, n_grid=None, p_flip=inst["p_flip"],
                          p_two=inst["p_two"], claims=inst["claims"], probes=inst["probes"],
                          probe_ids=[f"p{i}" for i in range(len(inst["probes"]))], **kw)


def _brute(inst, xs, n=1200):
    """Raw-definition reference: midpoint rule over transition positions (k<=2), G pointwise."""
    t = (np.arange(n) + 0.5) / n

    def G(pts):
        v = np.zeros_like(pts)
        for a, b, s, ne, r in inst["claims"]:
            d = ne * s * math.log(r / (1 - r)) / (2 * (b - a))
            v += d * np.clip(np.minimum(pts, b) - a, 0, None)
        for x, s, r, w in inst["probes"]:
            v += (pts > x) * w * s * math.log(r / (1 - r)) / 2
        return v

    Gh = float(G(np.array([1.0 + 1e-12]))[0])
    two = inst["p_two"]
    m = np.array([(1 - inst["p_flip"]) * (1 - two), inst["p_flip"] * (1 - two), two])
    m = m / m.sum()
    gt = G(t)
    ep, em = np.exp(2 * gt), np.exp(-2 * gt)
    W0 = m[0] / 2 * (math.exp(Gh) + math.exp(-Gh))
    W1 = m[1] / 2 * (math.exp(-Gh) * ep.mean() + math.exp(Gh) * em.mean())
    upper = np.triu(np.ones((n, n)), 1) / n ** 2
    W2 = m[2] * (math.exp(Gh) * ep @ upper @ em + math.exp(-Gh) * em @ upper @ ep)
    Z = W0 + W1 + W2
    pp = []
    for x in xs:
        L_ = t < x
        R_ = ~L_
        N0 = m[0] / 2 * math.exp(Gh)
        N1 = m[1] / 2 * (math.exp(-Gh) * (ep * R_).sum() / n + math.exp(Gh) * (em * L_).sum() / n)
        same = ((ep * L_) @ upper @ (em * L_)) + ((ep * R_) @ upper @ (em * R_))
        cross = (em * L_).sum() / n * (ep * R_).sum() / n
        N2 = m[2] * (math.exp(Gh) * same + math.exp(-Gh) * cross)
        pp.append((N0 + N1 + N2) / Z)
    return np.array(pp), np.array([W0, W1, W2]) / Z


def test_dispatch_returns_continuum_without_backend():
    c = CompiledRegime(0.0, 1.0, backend=None, n_grid=None)
    assert isinstance(c, ContinuumRegime) and c.n_grid is None
    assert c.log_evidence(normalized_prior=True) == pytest.approx(0.0, abs=1e-12)
    assert c.p_plus(0.3) == pytest.approx(0.5, abs=1e-12)
    with pytest.raises(ValueError):
        ContinuumRegime(0.0, 1.0, n_grid=48)
    with pytest.raises(ValueError):
        ContinuumRegime(0.0, 1.0, query_weights=[1.0])


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_matches_raw_definition_reference(seed):
    inst = _instance(seed)
    xs = np.array([0.05, 0.31, 0.5, 0.77, 0.98])
    ref_p, ref_f = _brute(inst, xs)
    c = _cont(inst)
    assert np.max(np.abs(c.p_plus_at(xs) - ref_p)) < 5e-4
    assert np.max(np.abs(c.family_mass() - ref_f)) < 5e-4


def test_new_edges_accepted_and_evidence_free_edges_change_nothing():
    inst = _instance(5)
    c = _cont(inst)
    xs = np.linspace(0, 1, 9)
    before = c.p_plus_at(xs)
    c.add_claim(0.123456789, 0.87654321, 1, 0.0, 0.8)   # n_eff = 0: new edges, no evidence
    assert np.max(np.abs(c.p_plus_at(xs) - before)) < 1e-14
    c.add_claim(0.123456789, 0.87654321, 1, 1.0, 0.8)   # real evidence on new edges
    assert np.max(np.abs(c.p_plus_at(xs) - before)) > 1e-4


def test_order_and_incremental_invariance():
    inst = _instance(17)
    base = _cont(inst)
    xs = np.linspace(0, 1, 13)
    rng = np.random.default_rng(0)
    for _ in range(3):
        inc = CompiledRegime(0.0, 1.0, backend=None, n_grid=None, p_flip=0.3, p_two=0.05)
        ev = [("c", c) for c in inst["claims"]] + [("p", i) for i in range(len(inst["probes"]))]
        for j in rng.permutation(len(ev)):
            kind, e = ev[j]
            if kind == "c":
                inc.add_claim(*e)
            else:
                inc.add_probe(*inst["probes"][e], evidence_id=f"p{e}")
            inc.family_mass()
        assert np.max(np.abs(inc.p_plus_at(xs) - base.p_plus_at(xs))) < 1e-12
        assert abs(inc.log_evidence() - base.log_evidence()) < 1e-12


def test_ledger_round_trip_and_guards():
    inst = _instance(23)
    c = _cont(inst)
    d = ContinuumRegime.from_ledger(0.0, 1.0, c.evidence_ledger(), p_flip=0.3, p_two=0.05)
    assert d.log_evidence() == pytest.approx(c.log_evidence(), abs=1e-12)
    with pytest.raises(ValueError):
        c.add_probe(0.5, 1, 0.9, evidence_id="p0")
    with pytest.raises(ValueError):
        c.add_probe(1.5, 1, 0.9, evidence_id="new")
    with pytest.raises(ValueError):
        c.add_claim(0.4, 0.4, 1)
    with pytest.raises(ValueError):
        c.p_plus(-0.1)


def test_location_posterior_prior_moments_closed_form():
    # No evidence: T = lo or hi (k=0, each m0/2), U(0,1) (k=1), min of two uniforms (k=2).
    c = ContinuumRegime(0.0, 1.0, p_flip=0.3, p_two=0.05)
    m0, m1, m2 = c.transition_prior()["mass"]
    mean = m0 / 2 * 1.0 + m1 * 0.5 + m2 / 3
    second = m0 / 2 * 1.0 + m1 / 3 + m2 / 6
    loc = c.location_posterior()
    assert loc["mean"] == pytest.approx(mean, abs=1e-12)
    assert loc["var"] == pytest.approx(second - mean ** 2, abs=1e-12)
    assert loc["quadrature_mass_error"] < 1e-12


def test_location_oed_gain_nonnegative_and_prefers_uncertain_region():
    c = ContinuumRegime(0.0, 1.0, p_flip=0.9, p_two=0.0)
    c.add_probe(0.2, -1, 0.95, evidence_id="a")
    c.add_probe(0.8, 1, 0.95, evidence_id="b")
    x, gain = c.best_location_probe(0.9)
    assert gain > 0 and 0.2 < x < 0.8
    for xv in (0.0, 0.1, 0.5, 0.95):
        assert c.expected_location_loss(xv, 0.9)["gain"] >= -1e-12
    x2, g2 = c.best_probe(0.9)
    assert 0.0 <= x2 <= 1.0 and g2 >= 0.0


@needs_kernel
def test_parity_with_native_at_fine_grid():
    if _KERNEL_SRC not in sys.path:
        sys.path.insert(0, _KERNEL_SRC)
    kern = importlib.import_module("kernel_engine.inference.finite_regime").FiniteRegimeKernel(_BUILD)
    for seed in (2, 7, 13):
        inst = _instance(seed)
        c = _cont(inst)
        errs = []
        for n in (3072, 12288):
            g = CompiledRegime(0.0, 1.0, backend=kern, n_grid=n, p_flip=0.3, p_two=inst["p_two"],
                               claims=inst["claims"], probes=inst["probes"],
                               probe_ids=[f"p{i}" for i in range(len(inst["probes"]))])
            xs = [0.1, 0.45, 0.9]
            err = max(abs(g.p_plus(x) - c.p_plus(x)) for x in xs)
            errs.append(err)
            assert err <= 40.0 / n                       # declared O(1/n) tolerance
            assert np.max(np.abs(g.family_mass() - c.family_mass())) <= 40.0 / n
            with pytest.raises(FrozenPartitionError):
                g.add_claim(0.1234567, 0.7654321, 1)
        assert errs[1] < errs[0] or errs[0] < 1e-9
