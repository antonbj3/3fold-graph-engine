"""Contracts for the native Graph-to-Kernel regime bridge.

These are semantic/parity tests against the LIVE RegimePosterior and an independent
direct-enumeration log-evidence oracle — not mirrors of the implementation. The
kernel backend is injected; set KERNEL_ENGINE_SRC (dir containing `kernel_engine`)
and BRIDGE_BUILD_DIR (writable) to run them, else they skip.
"""
from __future__ import annotations

import importlib
import math
import os
import sys

import numpy as np
import pytest

from graph_engine.compiled_regime import (
    CompiledRegime, FrozenPartitionError, majority_unsupported)
from graph_engine.regime_posterior import RegimePosterior

_KERNEL_SRC = os.environ.get("KERNEL_ENGINE_SRC")
_BUILD = os.environ.get("BRIDGE_BUILD_DIR")
pytestmark = pytest.mark.skipif(
    not (_KERNEL_SRC and _BUILD), reason="KERNEL_ENGINE_SRC/BRIDGE_BUILD_DIR unset")


@pytest.fixture(scope="session")
def backend():
    if _KERNEL_SRC not in sys.path:
        sys.path.insert(0, _KERNEL_SRC)
    mod = importlib.import_module("kernel_engine.inference.finite_regime")
    return mod.FiniteRegimeKernel(_BUILD)


def _oracle_log_evidence(regime):
    cells, _w, F, _post, logp = regime._solve()
    lp = np.array(logp, dtype=float)
    for x, sg, rel, wt in regime.probes:
        c = min(int(np.searchsorted(cells[:, 1], x, side="left")), len(cells) - 1)
        fp = F[:, c]
        f = fp if sg > 0 else 1.0 - fp
        lp = lp + wt * np.log(f * rel + (1.0 - f) * (1.0 - rel))
    m = lp.max()
    return float(m + np.log(np.exp(lp - m).sum()))


def _rand_regime(seed, Q=16, p_two=0.05, nc=3, npr=8):
    rng = np.random.default_rng(seed)
    r = RegimePosterior(0.0, 1.0, n_grid=Q, p_two=p_two, p_flip=0.3)
    for _ in range(nc):
        a = float(rng.uniform(0.0, 0.7))
        b = float(rng.uniform(a + 0.05, 1.0))
        r.add_claim(a, b, 1 if rng.random() < 0.5 else -1,
                    n_eff=float(rng.uniform(0.5, 4.0)),
                    reliability=float(rng.uniform(0.6, 0.95)))
    for _ in range(npr):
        r.add_probe(float(rng.uniform(0.0, 1.0)), 1 if rng.random() < 0.5 else -1,
                    reliability=float(rng.uniform(0.6, 0.95)),
                    weight=float(rng.uniform(0.5, 2.0)))
    return r


def _assert_matches(regime, backend):
    comp = CompiledRegime.from_regime(regime, backend)
    oc, _ow, oF, opost = regime._with_probes()
    pk = np.array([opost[:2].sum(),
                   opost[2:regime._n_one].sum() if regime._n_one > 2 else 0.0,
                   opost[regime._n_one:].sum() if len(opost) > regime._n_one else 0.0])
    assert regime.potential_value() == pytest.approx(comp.potential_value(), abs=1e-8)
    assert regime.expected_error() == pytest.approx(comp.expected_error(), abs=1e-8)
    assert _oracle_log_evidence(regime) == pytest.approx(comp.log_evidence(), abs=1e-8)
    assert float(np.max(np.abs(comp.p_plus_cells() - (opost @ oF)))) < 1e-8
    assert float(np.max(np.abs(comp.family_mass() - pk))) < 1e-8
    assert abs(comp.collision()["p_two_transitions"]
               - float(opost[regime._n_one:].sum())) < 1e-8
    for x in (0.0, 0.13, 0.5, 0.77, 1.0):
        assert comp.p_plus(x) == pytest.approx(regime.p_plus(x), abs=1e-8)
    return comp


def test_parity_random_weighted(backend):
    for seed in range(12):
        _assert_matches(_rand_regime(seed), backend)


def test_parity_two_transitions_and_no_collapse(backend):
    a = _rand_regime(3, Q=20, p_two=0.20)
    b = _rand_regime(3, Q=20, p_two=0.0)
    ca, cb = _assert_matches(a, backend), _assert_matches(b, backend)
    assert ca.collision()["p_two_transitions"] > 0.05
    assert cb.collision()["p_two_transitions"] == pytest.approx(0.0, abs=1e-12)
    assert abs(ca.log_evidence() - cb.log_evidence()) > 1e-6  # family is not collapsed


def test_no_dense_path_after_patch(backend, monkeypatch):
    """RegimePosterior._solve/_with_probes must never be used by the bridge."""
    regime = _rand_regime(5, Q=12, p_two=0.05, nc=2, npr=4)

    def boom(*_a, **_k):
        raise AssertionError("dense RegimePosterior path was called")

    monkeypatch.setattr(RegimePosterior, "_solve", boom)
    monkeypatch.setattr(RegimePosterior, "_with_probes", boom)
    comp = CompiledRegime.from_regime(regime, backend)
    assert 0.0 <= comp.p_plus(0.4) <= 1.0
    assert math.isfinite(comp.potential_value())
    assert math.isfinite(comp.log_evidence())
    assert sum(comp.family_mass()) == pytest.approx(1.0, abs=1e-9)
    comp.add_probe(0.33, -1, reliability=0.9, weight=1.0, evidence_id="e1")
    x, g = comp.best_probe(reliability=0.9)
    assert math.isfinite(g) and 0.0 <= x <= 1.0


def test_uninformative_and_boundaries(backend):
    r = RegimePosterior(0.0, 1.0, n_grid=6, p_two=0.05)
    r.add_claim(0.2, 0.4, +1, n_eff=3.0, reliability=0.8)
    before = r.potential_value()
    r.add_probe(0.5, +1, reliability=0.5)     # zero information
    assert r.potential_value() == pytest.approx(before, abs=1e-12)
    r.add_probe(0.0, -1, reliability=0.99)
    r.add_probe(1.0, +1, reliability=0.99)
    _assert_matches(r, backend)


def test_modified_reliabilities(backend):
    r = _rand_regime(9, Q=14)
    comp = CompiledRegime.from_regime(r, backend)
    new = [0.9 - 0.01 * i for i in range(len(r.claims))]
    r.set_claim_reliabilities(new)
    comp.set_claim_reliabilities(new)
    assert r.potential_value() == pytest.approx(comp.potential_value(), abs=1e-8)
    assert _oracle_log_evidence(r) == pytest.approx(comp.log_evidence(), abs=1e-8)


def test_strong_conflict_and_tails(backend):
    r = RegimePosterior(0.0, 1.0, n_grid=8, p_two=0.05)
    r.add_claim(0.1, 0.3, +1, n_eff=8.0, reliability=0.95)
    r.add_claim(0.55, 0.75, -1, n_eff=8.0, reliability=0.95)  # strong conflict
    comp = _assert_matches(r, backend)
    pp = comp.p_plus_cells()
    assert pp.min() < 0.05 and pp.max() > 0.95                  # near the tails
    assert 0.0 <= comp.collision()["p_two_transitions"] <= 1.0


def test_gradient_sufficiency(backend):
    """d logZ / d(A_q)=E[S_q] and d/d(B_q)=E[S_q^2]: natural stats are the true sufficient state."""
    comp = CompiledRegime.from_regime(_rand_regime(11, Q=10, nc=2, npr=5), backend)
    st = comp.natural_statistics()
    mass, logtot, _ = comp._effective_mass()
    const = logtot + comp._probe_weight * math.log(0.5) + comp._claim_const
    eps = 1e-6
    for q in range(comp.n_cells):
        a = st["kernel_a"].copy()
        a[q] += eps
        up = backend.evaluate(a, st["kernel_b"], mass)["log_partition"] + const
        a[q] -= 2 * eps
        dn = backend.evaluate(a, st["kernel_b"], mass)["log_partition"] + const
        assert (up - dn) / (2 * eps) == pytest.approx(comp.mean_s_cells()[q], abs=1e-6)
        b = st["kernel_b"].copy()
        b[q] += eps
        up = backend.evaluate(st["kernel_a"], b, mass)["log_partition"] + const
        b[q] -= 2 * eps
        dn = backend.evaluate(st["kernel_a"], b, mass)["log_partition"] + const
        assert (up - dn) / (2 * eps) == pytest.approx(comp.mean_s2_cells()[q], abs=1e-6)


def test_best_probe_exact_vs_oracle(backend):
    r = _rand_regime(4, Q=14, p_two=0.1, nc=2, npr=4)
    comp = CompiledRegime.from_regime(r, backend)
    x_ora, g_ora = r.best_probe(reliability=0.9)
    assert comp.expected_gain(x_ora, reliability=0.9)["gain"] == pytest.approx(g_ora, abs=1e-8)
    x_c, g_c = comp.best_probe(reliability=0.9)
    assert g_c >= g_ora - 1e-9


def test_guards_and_validation(backend):
    with pytest.raises(NotImplementedError):
        CompiledRegime(0.0, 1.0, backend=backend, claim_model="majority")
    c = CompiledRegime(0.0, 1.0, backend=backend, n_grid=6)
    with pytest.raises(ValueError):
        c.add_probe(0.5, 1, reliability=1.0, evidence_id="hard")   # r=1 refused
    with pytest.raises(ValueError):
        c.add_probe(1.5, 1, reliability=0.9, evidence_id="oob")    # outside domain
    with pytest.raises(ValueError):
        c.add_probe(0.5, 0, reliability=0.9, evidence_id="sgn")    # sign 0
    with pytest.raises(ValueError):
        c.add_probe(0.5, 1, reliability=0.9, weight=0.0, evidence_id="w")
    with pytest.raises(ValueError):
        c.add_probe(0.5, 1, evidence_id="")
    c.add_probe(0.5, 1, reliability=0.9, evidence_id="dup")
    with pytest.raises(ValueError):
        c.add_probe(0.4, -1, reliability=0.9, evidence_id="dup")   # declared duplicate
    with pytest.raises(ValueError):
        c.add_claim(0.4, 0.2, +1)                                   # reversed
    with pytest.raises(ValueError):
        c.add_claim(0.4, 0.4, +1)                                   # zero width
    with pytest.raises(ValueError):
        c.add_claim(-0.5, 0.5, +1)                                  # outside domain


def test_frozen_partition_contract(backend):
    c = CompiledRegime(0.0, 1.0, backend=backend, n_grid=4)
    with pytest.raises(FrozenPartitionError):
        c.add_claim(0.1, 0.3, +1)                 # new edges -> refuse
    existing = float(c.cells[1, 1])               # snap to an existing edge
    c.add_claim(existing, float(c.cells[-1, 1]), +1, n_eff=2.0, reliability=0.8)
    assert c.n_cells == len(c.cells)
    assert math.isfinite(c.potential_value())


def test_majority_message_is_not_a_model_swap():
    with pytest.raises(NotImplementedError) as exc:
        majority_unsupported("majority")
    assert "RegimeMarkov" in str(exc.value)   # named, and explicitly not a substitute


def test_parity_from_regime_reads_only_public_fields(backend):
    regime = _rand_regime(2, Q=10, p_two=0.05, nc=2, npr=3)
    # Snapshot the dense internals, then corrupt them; from_regime must not care.
    comp = CompiledRegime.from_regime(regime, backend)
    saved = regime._cache
    regime._cache = ("corrupt", "corrupt", "corrupt", "corrupt", "corrupt")
    try:
        comp2 = CompiledRegime.from_regime(regime, backend)
        assert comp2.p_plus(0.5) == pytest.approx(comp.p_plus(0.5), abs=1e-12)
    finally:
        regime._cache = saved
