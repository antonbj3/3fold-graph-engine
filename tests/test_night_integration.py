"""Night integration: evidence federation + continuum backend + fast probe choice together.

The three features were developed on separate branches. These tests exercise the seams
between them on real ``CompiledRegime``/``ContinuumRegime`` objects:

* a federated native regime keeps the covariance probe path of its inputs and chooses the
  same probe as the direct loop and as a regime built from the evidence-ID union;
* continuum regimes federate by the same ID-union rule, the result equals the continuum view
  of the federated native regime (federation commutes with taking the continuum limit), and
  mixing a native with a continuum regime is refused;
* with the Kernel (KERNEL_ENGINE_SRC/BRIDGE_BUILD_DIR), the native federated regime at a
  fine grid matches the federated continuum within the declared O(1/n) tolerance of
  ``test_continuum_regime.py`` and its covariance probe choice equals the direct choice.

The no-Kernel backend is the independent brute-force enumeration of
``test_compiled_regime_fast_probe.EnumBackend``.
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

from graph_engine.compiled_regime import CompiledRegime
from graph_engine.continuum_regime import ContinuumRegime
from graph_engine.evidence_federation import IncompatibleFederationError, federate

from test_compiled_regime_fast_probe import EnumBackend

_KERNEL_SRC = os.environ.get("KERNEL_ENGINE_SRC")
_BUILD = os.environ.get("BRIDGE_BUILD_DIR")
needs_kernel = pytest.mark.skipif(not (_KERNEL_SRC and _BUILD),
                                  reason="KERNEL_ENGINE_SRC/BRIDGE_BUILD_DIR unset")

CLAIMS = ((0.1, 0.4, 1, 1.5, 0.7), (0.55, 0.9, -1, 2.0, 0.8))
# Two producers with one shared observation ("s1"); each has private evidence.
PROBES_A = (((0.22, 1, 0.9, 1.0), "a1"), ((0.61, -1, 0.8, 1.0), "s1"), ((0.47, 1, 0.7, 2.0), "a2"))
PROBES_B = (((0.61, -1, 0.8, 1.0), "s1"), ((0.83, -1, 0.85, 0.5), "b1"))
UNION = PROBES_A + PROBES_B[1:]


def _native(probes, backend, cov=None, n=24):
    return CompiledRegime(0.0, 1.0, backend=backend, covariance_backend=cov, n_grid=n, p_flip=0.3,
                          p_two=0.05, claims=CLAIMS, probes=[p for p, _ in probes],
                          probe_ids=[i for _, i in probes])


def _continuum(probes):
    return CompiledRegime(0.0, 1.0, n_grid=None, p_flip=0.3, p_two=0.05, claims=CLAIMS,
                          probes=[p for p, _ in probes], probe_ids=[i for _, i in probes])


def _ids(reg):
    return sorted(p[4] for p in reg.evidence_ledger()["probes"])


def test_federated_native_keeps_covariance_probe_path_and_equals_union():
    be = EnumBackend(batch=False, cov=False)        # evaluate only: covariance must come from cov
    cov = EnumBackend()
    a, b = _native(PROBES_A, be, cov), _native(PROBES_B, be, cov)
    joint, report = federate(a, b)
    assert report["shared_probe_ids"] == ("s1",)
    union = _native(UNION, be, cov)
    assert _ids(joint) == _ids(union)
    assert np.max(np.abs(joint.p_plus_cells() - union.p_plus_cells())) < 1e-12
    for r in (0.7, 0.95):
        ref = joint.best_probe(r)
        got = joint.best_probe(r, method="covariance")
        assert joint.last_probe_path["path"] == "covariance", joint.last_probe_path
        assert got[0] == ref[0] == union.best_probe(r, method="covariance")[0]
        assert abs(got[1] - ref[1]) <= 1e-12


def test_continuum_federation_is_id_union_and_commutes_with_continuum_view():
    ca, cb = _continuum(PROBES_A), _continuum(PROBES_B)
    assert isinstance(ca, ContinuumRegime)
    joint, report = federate(ca, cb)
    assert isinstance(joint, ContinuumRegime)
    assert report["shared_probe_ids"] == ("s1",) and report["n_probes_joint"] == 4
    xs = np.linspace(0.0, 1.0, 41)
    union = _continuum(UNION)
    assert np.max(np.abs(joint.p_plus_at(xs) - union.p_plus_at(xs))) < 1e-12
    # Idempotence and commutativity of the ID-union.
    same, _ = federate(ca, ca)
    assert np.max(np.abs(same.p_plus_at(xs) - ca.p_plus_at(xs))) < 1e-12
    ba, _ = federate(cb, ca)
    assert np.max(np.abs(ba.p_plus_at(xs) - joint.p_plus_at(xs))) < 1e-12
    # Continuum view of the federated native regime == federation of the continuum views.
    be = EnumBackend(batch=False, cov=False)
    nj, _ = federate(_native(PROBES_A, be), _native(PROBES_B, be))
    view = ContinuumRegime.from_compiled(nj)
    assert np.max(np.abs(view.p_plus_at(xs) - joint.p_plus_at(xs))) < 1e-12
    x, gain = joint.best_probe(0.9)
    assert 0.0 <= x <= 1.0 and gain > 0.0


def test_mixing_native_and_continuum_is_refused():
    be = EnumBackend(batch=False, cov=False)
    with pytest.raises(IncompatibleFederationError):
        federate(_native(PROBES_A, be), _continuum(PROBES_B))
    with pytest.raises(IncompatibleFederationError):
        federate(_continuum(PROBES_A), _native(PROBES_B, be))


@needs_kernel
def test_native_federation_fine_grid_matches_continuum_and_fast_probe():
    if _KERNEL_SRC not in sys.path:
        sys.path.insert(0, _KERNEL_SRC)
    kern = importlib.import_module("kernel_engine.inference.finite_regime").FiniteRegimeKernel(_BUILD)
    try:
        lap = importlib.import_module("kernel_engine.inference.finite_laplacian").LaplacianKernel(_BUILD)
    except ImportError:
        pytest.skip("this Kernel checkout has no finite_laplacian (covariance capability)")
    cj, _ = federate(_continuum(PROBES_A), _continuum(PROBES_B))
    xs = [0.1, 0.45, 0.61, 0.9]
    errs = []
    for n in (768, 3072):
        nj, _ = federate(_native(PROBES_A, kern, lap, n), _native(PROBES_B, kern, lap, n))
        err = max(abs(nj.p_plus(x) - cj.p_plus(x)) for x in xs)
        errs.append(err)
        assert err <= 40.0 / n                          # declared O(1/n) tolerance (continuum tests)
        assert np.max(np.abs(nj.family_mass() - cj.family_mass())) <= 40.0 / n
    assert errs[1] < errs[0] or errs[0] < 1e-9
    nj, _ = federate(_native(PROBES_A, kern, lap, 256), _native(PROBES_B, kern, lap, 256))
    for r in (0.7, 0.95):
        ref = nj.best_probe(r)
        got = nj.best_probe(r, method="covariance")
        assert nj.last_probe_path["path"] == "covariance", nj.last_probe_path
        assert got[0] == ref[0]
        assert abs(got[1] - ref[1]) <= 1e-12 * (nj.hi - nj.lo)
