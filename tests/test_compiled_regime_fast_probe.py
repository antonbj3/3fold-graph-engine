"""best_probe screened paths ("batch", "covariance") against the direct per-candidate loop.

Two backends. (1) An in-file NumPy enumeration of every world of the 0/1/2 family (small Q);
it needs no Kernel, so decision parity and every fallback are tested in the plain suite, and
its covariance is an independent oracle (world enumeration, not the cut-state recursion).
(2) The native Kernel (``KERNEL_ENGINE_SRC`` + ``BRIDGE_BUILD_DIR``); covariance tests there need
``kernel_engine.inference.finite_laplacian`` and skip when that module is absent.
"""
from __future__ import annotations

import importlib
import itertools
import math
import os
import sys

import numpy as np
import pytest

from graph_engine.compiled_regime import CompiledRegime
from graph_engine.instruments import ContactSolverSweep, SweepFamily, _reliability_of

_KERNEL_SRC = os.environ.get("KERNEL_ENGINE_SRC")
_BUILD = os.environ.get("BRIDGE_BUILD_DIR")
_REL = 1e-12


# ---------------------------------------------------------------- NumPy enumeration backend
def _worlds(q, mass):
    out = []                                            # (log prior, S vector)
    lp = [(math.log(m) - math.log(2) - math.log(math.comb(q, k))) if m > 0 else None
          for k, m in enumerate(mass)]
    for s in (1.0, -1.0):
        if lp[0] is not None:
            out.append((lp[0], np.full(q, s)))
        if lp[1] is not None:
            for i in range(q):
                v = np.full(q, s)
                v[i] = 0.0
                v[i + 1:] = -s
                out.append((lp[1], v))
        if lp[2] is not None and q >= 2:
            for i, j in itertools.combinations(range(q), 2):
                v = np.full(q, s)
                v[i] = 0.0
                v[i + 1:j] = -s
                v[j] = 0.0
                out.append((lp[2], v))
    return out


class EnumBackend:
    """Brute-force world enumeration with the Kernel's evaluate/evaluate_batch/cov_rows contract."""

    def __init__(self, batch=True, cov=True, mean_shift=0.0):
        if batch:
            self.evaluate_batch = self._evaluate_batch
        if cov:
            self.cov_rows = self._cov_rows
        self.mean_shift = mean_shift
        self.calls = {"evaluate": 0, "batch": 0, "cov": 0}

    @staticmethod
    def _moments(a, b, mass):
        q = len(a)
        W = _worlds(q, np.asarray(mass, float))
        S = np.array([v for _, v in W])
        T = np.concatenate([S, S * S], axis=1)
        logw = np.array([lp for lp, _ in W]) + T @ np.concatenate([a, b])
        m = logw.max()
        w = np.exp(logw - m)
        z = w.sum()
        w = w / z
        k = np.array([0 if not np.any(v == 0) else (1 if np.sum(v == 0) == 1 else 2) for _, v in W])
        fam = np.array([w[k == i].sum() for i in range(3)])
        mean = w @ T
        cov = (T * w[:, None]).T @ T - np.outer(mean, mean)
        return mean[:q], mean[q:], fam, float(m + math.log(z)), cov

    def evaluate(self, a, b, family_mass, *, method="transfer"):
        self.calls["evaluate"] += 1
        ms, m2, fam, lz, _ = self._moments(np.asarray(a, float), np.asarray(b, float), family_mass)
        return {"mean_s": ms, "mean_s2": m2, "p_plus": np.clip((1 + ms) / 2, 0, 1),
                "family_mass": fam, "log_partition": lz}

    def _evaluate_batch(self, a, b, family_mass):
        self.calls["batch"] += 1
        rows = [self.evaluate(ai, bi, family_mass) for ai, bi in zip(a, b)]
        self.calls["evaluate"] -= len(rows)
        return {"p_plus": np.array([r["p_plus"] for r in rows])}

    def _cov_rows(self, a, b, rows, family_mass):
        self.calls["cov"] += 1
        ms, m2, _, lz, cov = self._moments(np.asarray(a, float), np.asarray(b, float), family_mass)
        return {"rows": cov[np.asarray(rows)], "mean_s": ms + self.mean_shift, "mean_s2": m2,
                "log_partition": lz}


def _rand_regime(backend, seed, q, *, cov=None, potential="entropy", lo=0.0, hi=1.0,
                 r_probe=(0.6, 0.95), n_probes=5, p_two=0.05):
    rng = np.random.default_rng(seed)
    claims = []
    for _ in range(int(rng.integers(0, 4))):
        a, b = sorted(rng.uniform(lo, hi, 2))
        if b - a < 1e-3 * (hi - lo):
            continue
        claims.append((float(a), float(b), int(rng.choice([-1, 1])), float(rng.uniform(0.5, 3)),
                       float(rng.uniform(0.55, 0.95))))
    reg = CompiledRegime(lo, hi, backend=backend, covariance_backend=cov, n_grid=q,
                         potential=potential, p_two=p_two, claims=claims)
    for i in range(n_probes):
        reg.add_probe(float(rng.uniform(lo, hi)), int(rng.choice([-1, 1])),
                      float(rng.uniform(*r_probe)), float(rng.choice([1.0, 0.5, 2.0])),
                      evidence_id=f"t:{seed}:{i}")
    return reg


def _check_parity(reg, r, methods=("batch", "covariance"), expect_path=None):
    ref = reg.best_probe(r)
    assert reg.last_probe_path["path"] == "direct"
    d = reg.probe_losses(r)
    for m in methods:
        got = reg.best_probe(r, method=m)
        assert got[0] == ref[0], (m, got, ref)
        path = reg.last_probe_path
        if expect_path:
            assert path["path"] == expect_path[m]
        if path["resolved"]:
            assert got[1] == ref[1]                      # multi-cell tie re-decided by the direct path
        # Value scale: the loss is in bits x length, at most (hi - lo). Relative to the loss itself
        # neither path is 1e-12-accurate on nearly resolved regimes (an 80-bit reference:
        # direct is 1e-8 relative off at before = 2e-4 bits), so the tolerance is on that scale.
        scale = _REL * (reg.hi - reg.lo)
        assert abs(got[1] - ref[1]) <= scale
        s = reg.probe_losses(r, method=m)
        assert np.all(np.abs(s["after"] - d["after"]) <= scale + _REL * d["after"])
    return ref


# ---------------------------------------------------------------- no Kernel needed
def test_enum_covariance_matches_hypothetical_update():
    be = EnumBackend()
    reg = _rand_regime(be, 3, 6)
    d = reg.probe_losses(0.9)
    c = reg.probe_losses(0.9, method="covariance")
    assert c["path"] == "covariance"
    assert np.max(np.abs(c["after"] - d["after"]) / d["after"]) < 1e-13


@pytest.mark.parametrize("seed", range(40))
def test_decision_parity_enum_random(seed):
    rng = np.random.default_rng(10_000 + seed)
    q = int(rng.integers(1, 10))
    be = EnumBackend()
    pot = "error" if seed % 3 == 0 else "entropy"
    lo = float(rng.uniform(-5, 5)) if seed % 2 else 0.0
    reg = _rand_regime(be, seed, q, potential=pot, lo=lo, hi=lo + (7.0 if seed % 2 else 1.0),
                       p_two=0.0 if seed % 5 == 0 else 0.05)
    for r in (0.5, 0.51, 0.9, 0.999999):
        _check_parity(reg, r, expect_path={"batch": "batch", "covariance": "covariance"})


@pytest.mark.parametrize("q", [2, 4, 7])
@pytest.mark.parametrize("potential", ["entropy", "error"])
def test_symmetric_ties_resolved_by_direct_rule(q, potential):
    # No evidence: mirror cells are exact ties in exact arithmetic; the first maximum must win.
    be = EnumBackend()
    reg = CompiledRegime(0.0, 1.0, backend=be, covariance_backend=None, n_grid=q, potential=potential)
    _check_parity(reg, 0.95)
    for m in ("batch", "covariance"):
        got = reg.best_probe(0.95, method=m)
        if q % 2 == 0:                                   # two mirror cells share the maximum
            assert reg.last_probe_path["resolved"] >= 2
            assert got == reg.best_probe(0.95)


def test_r_half_all_gains_zero_first_candidate():
    be = EnumBackend()
    reg = _rand_regime(be, 5, 6)
    ref = reg.best_probe(0.5)
    for m in ("batch", "covariance"):
        assert reg.best_probe(0.5, method=m)[0] == ref[0]


def test_fallback_without_capabilities_is_bitwise_direct():
    be = EnumBackend(batch=False, cov=False)
    reg = _rand_regime(be, 11, 5)
    ref = reg.best_probe(0.9)
    for m in ("batch", "covariance"):
        assert reg.best_probe(0.9, method=m) == ref
        assert reg.last_probe_path["path"] == "direct"
        assert reg.last_probe_path["fallback_reason"]


def test_covariance_falls_back_to_batch_when_only_batch_exists():
    be = EnumBackend(batch=True, cov=False)
    reg = _rand_regime(be, 12, 5)
    got = reg.best_probe(0.9, method="covariance")
    assert reg.last_probe_path["path"] == "batch"
    assert "cov_rows" in reg.last_probe_path["fallback_reason"]
    assert got[0] == reg.best_probe(0.9)[0]


def test_backend_capability_discovered_without_explicit_covariance_backend():
    be = EnumBackend()
    reg = _rand_regime(be, 13, 5)
    reg.best_probe(0.9, method="covariance")
    assert reg.last_probe_path["path"] == "covariance" and be.calls["cov"] == 1


def test_disagreeing_covariance_kernel_is_refused_not_used():
    be = EnumBackend(batch=True, cov=False)
    bad = EnumBackend(mean_shift=1e-6)
    reg = _rand_regime(be, 14, 5, cov=bad)
    ref = reg.best_probe(0.9)
    assert reg.best_probe(0.9, method="covariance") == ref
    assert reg.last_probe_path["path"] == "direct"
    assert "disagrees" in reg.last_probe_path["fallback_reason"]


def test_default_is_unchanged_direct_loop_and_bad_method_refused():
    be = EnumBackend()
    reg = _rand_regime(be, 15, 4)
    reg.best_probe(0.9)
    assert reg.last_probe_path == {"path": "direct", "fallback_reason": None, "resolved": 0}
    assert be.calls["cov"] == 0 and be.calls["batch"] == 0
    with pytest.raises(ValueError):
        reg.best_probe(0.9, method="fast")
    with pytest.raises(TypeError):
        CompiledRegime(0.0, 1.0, backend=be, covariance_backend=object())


def test_screen_does_not_mutate_state():
    be = EnumBackend()
    reg = _rand_regime(be, 16, 6)
    before = (reg.p_plus_cells(), reg.log_evidence(), reg.evidence_ledger())
    reg.best_probe(0.9, method="covariance")
    reg.best_probe(0.9, method="batch")
    after = (reg.p_plus_cells(), reg.log_evidence(), reg.evidence_ledger())
    assert np.array_equal(before[0], after[0]) and before[1:] == after[1:]


# ---------------------------------------------------------------- native Kernel
_native = pytest.mark.skipif(not (_KERNEL_SRC and _BUILD), reason="KERNEL_ENGINE_SRC/BRIDGE_BUILD_DIR unset")


@pytest.fixture(scope="module")
def kernels():
    if _KERNEL_SRC not in sys.path:
        sys.path.insert(0, _KERNEL_SRC)
    R = importlib.import_module("kernel_engine.inference.finite_regime").FiniteRegimeKernel(_BUILD)
    try:
        lap = importlib.import_module("kernel_engine.inference.finite_laplacian")
    except ImportError:
        return R, None
    return R, lap.LaplacianKernel(_BUILD)


def _need_cov(kernels):
    if kernels[1] is None:
        pytest.skip("this Kernel checkout has no finite_laplacian (covariance capability)")


@_native
def test_native_enum_oracle_agrees(kernels):
    R, LK = kernels
    _need_cov(kernels)
    be = EnumBackend()
    for seed in range(5):
        a = _rand_regime(be, 700 + seed, 6)
        st = a.natural_statistics()
        mass = a.transition_prior()["mass"]
        n = R.evaluate(st["kernel_a"], st["kernel_b"], mass)
        e = be.evaluate(st["kernel_a"], st["kernel_b"], mass)
        assert np.max(np.abs(n["mean_s"] - e["mean_s"])) < 1e-12
        c = LK.cov_rows(st["kernel_a"], st["kernel_b"], np.arange(12), mass)["rows"]
        ce = be._cov_rows(st["kernel_a"], st["kernel_b"], np.arange(12), mass)["rows"]
        assert np.max(np.abs(c - ce)) < 1e-12


@_native
@pytest.mark.parametrize("seed", range(24))
def test_native_decision_and_value_parity_random(kernels, seed):
    R, LK = kernels
    _need_cov(kernels)
    rng = np.random.default_rng(20_000 + seed)
    q = int(rng.choice([1, 2, 3, 5, 16, 48, 97, 260]))
    pot = "error" if seed % 4 == 0 else "entropy"
    reg = _rand_regime(R, seed, q, cov=LK, potential=pot, n_probes=int(rng.integers(0, 30)),
                       r_probe=(0.5, 0.999) if seed % 3 == 0 else (0.6, 0.95))
    for r in (0.5 + 1e-9, 0.7, 0.95, 0.999999):
        _check_parity(reg, r, expect_path={"batch": "batch", "covariance": "covariance"})


@_native
def test_native_chunked_covariance_rows(kernels):
    R, LK = kernels
    _need_cov(kernels)
    reg = _rand_regime(R, 99, 300, cov=LK, n_probes=40)
    old = CompiledRegime._COV_ROW_CHUNK
    try:
        full = reg.probe_losses(0.9, method="covariance")["after"]
        CompiledRegime._COV_ROW_CHUNK = 37
        part = reg.probe_losses(0.9, method="covariance")["after"]
    finally:
        CompiledRegime._COV_ROW_CHUNK = old
    assert np.max(np.abs(full - part) / full) < 1e-14         # BLAS order may differ per chunk


# ---------------------------------------------------------------- self-contained instrument fixture
_FIXTURE_RELIABILITY = 0.75


def _fixture_rows(seed, n, x_lo, x_hi, tilt):
    """Deterministic synthetic scenes for a ContactSolverSweep.

    Method A's log-advantage over B varies across the condition axis (``tilt`` is the slope), so
    the family has both easy and hard regions and the instrument's MEASURED spread drives p.
    """
    rng = np.random.default_rng(seed)
    xs = np.sort(rng.uniform(x_lo, x_hi, n))
    center = 0.5 * (x_lo + x_hi)
    rows = []
    for i, x in enumerate(xs):
        a = float(rng.uniform(50.0, 5000.0))
        l = tilt * (x - center) + float(rng.normal(0.0, 0.35))
        b = float(a * math.exp(l))
        rows.append(SweepFamily(f"scene{i:03d}", float(x), a, b,
                                float(rng.uniform(1e-3, 5.0)), tag="fixture"))
    return rows


def _fixture_regimes(R, LK):
    """Native regimes built from a self-contained instrument fixture.

    Same shape as the contact-solver sweep integration: three named families, each reduced by
    ``ContactSolverSweep`` to a sign/probability at every scene and entered under one lineage root
    per family. No repository-external data or example script is read.
    """
    windows = {"router": 0.14, "rho": 0.30, "channels": 0.22}
    signs = {"router": -1, "rho": -1, "channels": 1}
    spec = (("router", 0, 24, -2.0, 2.0, 0.30),
            ("rho", 1, 30, -1.5, 1.5, 0.45),
            ("channels", 2, 40, 0.0, 4.0, 0.25))
    out = []
    for name, seed, n, x_lo, x_hi, tilt in spec:
        rows = _fixture_rows(seed, n, x_lo, x_hi, tilt)
        lo = min(r.x for r in rows) - 1e-9
        hi = max(r.x for r in rows) + 1e-9
        pair = (f"{name}:scene", "method_advantage_over_baseline")
        sign = signs[name]
        ins = ContactSolverSweep(f"{name}:scene", rows, f"campaign:{name}:scene", window=windows[name])
        reg = CompiledRegime(lo, hi, backend=R, covariance_backend=LK,
                             reliability=_FIXTURE_RELIABILITY, p_flip=0.3,
                             claims=[(lo, hi, sign, 1.0, _FIXTURE_RELIABILITY)])
        for r in rows:
            rd = ins.probe(pair, r.x)
            reg.add_probe(float(r.x), rd.sign, _reliability_of(rd.p), 1.0,
                          evidence_id=f"campaign:{name}:scene:{r.label}")
        out.append((name, reg))
    return out


@_native
def test_native_parity_closed_loop_fixture(kernels):
    R, LK = kernels
    _need_cov(kernels)
    for name, reg in _fixture_regimes(R, LK):
        rng = np.random.default_rng(len(name))
        for t in range(6):
            x, _ = _check_parity(reg, 0.95)
            reg.add_probe(x, int(rng.choice([-1, 1])), 0.95, evidence_id=f"loop:{name}:{t}")
