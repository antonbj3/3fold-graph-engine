"""Contracts for exact evidence-ledger federation.

The default backend here is an independent dense world enumeration (not the code under
test). With KERNEL_ENGINE_SRC and BRIDGE_BUILD_DIR set, the native parity test also runs.
"""
from __future__ import annotations

import importlib
import math
import os
import sys
from itertools import combinations

import numpy as np
import pytest

from graph_engine.compiled_regime import CompiledRegime
from graph_engine.evidence_federation import (
    EvidenceConflictError, FederationError, IncompatibleFederationError,
    federate, federation_context, merge_evidence_ledgers)


class DenseBackend:
    """Enumerates all 0/1/2-transition sign paths; prior uniform inside each family."""

    def evaluate(self, a, b, family_mass, *, method="transfer"):
        a, b = np.asarray(a, float), np.asarray(b, float)
        q = len(a)
        rows, fam = [], []
        for k in range(3):
            if family_mass[k] <= 0:
                continue
            for flips in combinations(range(q), k):
                for init in (-1, 1):
                    s, row = init, []
                    for c in range(q):
                        row.append(0 if c in flips else s)
                        if c in flips:
                            s = -s
                    rows.append(row)
                    fam.append(k)
        S = np.array(rows, float)
        fam = np.array(fam)
        logprior = np.array([math.log(family_mass[k]) - math.log(2 * math.comb(q, k)) for k in fam])
        v = logprior + S @ a + (S * S) @ b
        m = v.max()
        logz = m + math.log(np.exp(v - m).sum())
        p = np.exp(v - logz)
        ms = p @ S
        return {"mean_s": ms, "mean_s2": p @ (S * S), "p_plus": (1 + ms) / 2,
                "family_mass": np.array([p[fam == k].sum() for k in range(3)]),
                "log_partition": logz}


BACKEND = DenseBackend()


def regime(probes=(), ids=(), claims=(), **kw):
    kw.setdefault("n_grid", 6)
    kw.setdefault("p_two", 0.1)
    return CompiledRegime(0.0, 1.0, backend=BACKEND, claims=claims, probes=probes,
                          probe_ids=list(ids), **kw)


def dense_union_p_plus(reg_template, probes):
    """Independent path: per-world log-likelihood sum, no natural coordinates."""
    cells = reg_template.cells
    q = len(cells)
    mass = reg_template.transition_prior()["mass"]
    rows, fam = [], []
    for k in range(3):
        for flips in combinations(range(q), k):
            for init in (-1, 1):
                s, row = init, []
                for c in range(q):
                    row.append(0 if c in flips else s)
                    if c in flips:
                        s = -s
                rows.append(row)
                fam.append(k)
    S = np.array(rows, float)
    lp = np.array([math.log(mass[k]) - math.log(2 * math.comb(q, k)) for k in fam])
    for x, y, r, w in probes:
        c = min(int(np.searchsorted(cells[:, 1], x, side="left")), q - 1)
        lp = lp + w * np.log(0.5 + y * (r - 0.5) * S[:, c])
    p = np.exp(lp - lp.max())
    p /= p.sum()
    return (1 + p @ S) / 2


P1 = (0.30, 1, 0.8, 1.0)
P2 = (0.70, -1, 0.9, 1.0)
P3 = (0.55, 1, 0.7, 1.0)


def test_identical_posteriors_different_joints():
    a = regime([P1], ["obs-1"])
    b_shared = regime([P1], ["obs-1"])
    b_indep = regime([P1], ["obs-2"])
    assert np.array_equal(b_shared.p_plus_cells(), b_indep.p_plus_cells())
    j_shared, rep_s = federate(a, b_shared)
    j_indep, rep_i = federate(a, b_indep)
    assert rep_s["shared_probe_ids"] == ("obs-1",)
    assert rep_i["shared_probe_ids"] == ()
    np.testing.assert_allclose(j_shared.p_plus_cells(), a.p_plus_cells(), rtol=0, atol=1e-15)
    np.testing.assert_allclose(j_indep.p_plus_cells(),
                               dense_union_p_plus(a, [P1, P1]), rtol=0, atol=1e-12)
    assert np.max(np.abs(j_shared.p_plus_cells() - j_indep.p_plus_cells())) > 0.05


def test_partial_overlap_equals_union_and_natural_sum():
    a = regime([P1, P2], ["e1", "e2"])
    b = regime([P2, P3], ["e2", "e3"])
    joint, rep = federate(a, b)
    assert rep["shared_probe_ids"] == ("e2",) and rep["novel_probe_ids_from_b"] == ("e3",)
    np.testing.assert_allclose(joint.p_plus_cells(), dense_union_p_plus(a, [P1, P2, P3]),
                               rtol=0, atol=1e-12)
    only_b = regime([P3], ["e3"])
    na, nb, nj = a.natural_statistics(), only_b.natural_statistics(), joint.natural_statistics()
    np.testing.assert_allclose(nj["A"], na["A"] + nb["A"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(nj["B"], na["B"] + nb["B"], rtol=0, atol=1e-15)
    # naive product of the two posteriors double counts e2
    naive_a = na["A"] + b.natural_statistics()["A"]
    assert not np.allclose(naive_a, nj["A"])


def test_order_symmetric():
    a = regime([P1, P2], ["e1", "e2"])
    b = regime([P2, P3], ["e2", "e3"])
    np.testing.assert_allclose(federate(a, b)[0].p_plus_cells(), federate(b, a)[0].p_plus_cells(),
                               rtol=0, atol=1e-14)


def test_log_evidence_counts_shared_once():
    a = regime([P1, P2], ["e1", "e2"])
    b = regime([P2, P3], ["e2", "e3"])
    ref = regime([P1, P2, P3], ["e1", "e2", "e3"])
    assert federate(a, b)[0].log_evidence() == pytest.approx(ref.log_evidence(), abs=1e-12)


def test_refuses_incompatible_prior_and_partition():
    a = regime([P1], ["e1"])
    with pytest.raises(IncompatibleFederationError):
        federate(a, regime([P3], ["e3"], p_two=0.2))
    with pytest.raises(IncompatibleFederationError):
        federate(a, regime([P3], ["e3"], n_grid=7))
    with pytest.raises(IncompatibleFederationError):
        federate(a, regime([P3], ["e3"], p_flip=0.4))
    claimed = regime([P3], ["e3"], claims=[(0.25, 0.6, 1, 1.0, 0.8)])
    with pytest.raises(IncompatibleFederationError):  # claim edges change the partition
        federate(a, claimed, claim_policy="disjoint")


def test_refuses_conflicting_record_for_same_id():
    a = regime([P1], ["e1"])
    b = regime([(0.30, 1, 0.75, 1.0)], ["e1"])
    with pytest.raises(EvidenceConflictError):
        federate(a, b)


def test_refuses_positional_legacy_ids():
    a = CompiledRegime(0.0, 1.0, backend=BACKEND, n_grid=6, probes=[P1])
    b = CompiledRegime(0.0, 1.0, backend=BACKEND, n_grid=6, probes=[P1])
    assert a.evidence_ids == b.evidence_ids == frozenset({"legacy:0"})
    with pytest.raises(FederationError):
        federate(a, b)


def test_claim_policies():
    claim = (0.0, 0.5, 1, 2.0, 0.8)
    other = (0.5, 1.0, -1, 1.0, 0.7)
    base = [claim, other]
    a = regime([P1], ["e1"], claims=base)
    b = regime([P3], ["e3"], claims=base)
    joint, rep = federate(a, b)
    assert rep["n_claims_joint"] == 2
    ref = regime([P1, P3], ["e1", "e3"], claims=base)
    np.testing.assert_allclose(joint.p_plus_cells(), ref.p_plus_cells(), rtol=0, atol=1e-14)
    a2 = regime([P1], ["e1"], claims=[claim, other])
    b2 = regime([P3], ["e3"], claims=[other, claim])  # same edges, same multiset
    federate(a2, b2)
    a3 = regime([P1], ["e1"], claims=[claim, other])
    b3 = regime([P3], ["e3"], claims=[claim, (0.5, 1.0, 1, 1.0, 0.7)])
    with pytest.raises(FederationError):
        federate(a3, b3)
    joint3, _ = federate(a3, b3, claim_policy="disjoint")
    ref3 = regime([P1, P3], ["e1", "e3"],
                  claims=[claim, other, claim, (0.5, 1.0, 1, 1.0, 0.7)])
    np.testing.assert_allclose(joint3.p_plus_cells(), ref3.p_plus_cells(), rtol=0, atol=1e-14)
    with pytest.raises(ValueError):
        federate(a3, b3, claim_policy="average")


def test_merge_does_not_mutate_inputs():
    a = regime([P1, P2], ["e1", "e2"])
    b = regime([P2, P3], ["e2", "e3"])
    la, lb = a.evidence_ledger(), b.evidence_ledger()
    pa = a.p_plus_cells()
    merged, _ = merge_evidence_ledgers(la, lb, context_a=federation_context(a),
                                       context_b=federation_context(b))
    assert a.evidence_ledger() == la and b.evidence_ledger() == lb
    np.testing.assert_array_equal(a.p_plus_cells(), pa)
    assert [r[4] for r in merged["probes"]] == ["e1", "e2", "e3"]


def test_rejects_foreign_schema():
    a = regime([P1], ["e1"])
    ctx = federation_context(a)
    with pytest.raises(IncompatibleFederationError):
        merge_evidence_ledgers({"schema": "other", "claims": (), "probes": ()}, a.evidence_ledger(),
                               context_a=ctx, context_b=ctx)


_KERNEL_SRC = os.environ.get("KERNEL_ENGINE_SRC")
_BUILD = os.environ.get("BRIDGE_BUILD_DIR")


@pytest.mark.skipif(not (_KERNEL_SRC and _BUILD), reason="KERNEL_ENGINE_SRC/BRIDGE_BUILD_DIR unset")
def test_native_kernel_parity_random():
    if _KERNEL_SRC not in sys.path:
        sys.path.insert(0, _KERNEL_SRC)
    native = importlib.import_module("kernel_engine.inference.finite_regime").FiniteRegimeKernel(_BUILD)
    rng = np.random.default_rng(20260922)
    for trial in range(20):
        pool = [(float(rng.uniform(0, 1)), int(rng.choice([-1, 1])), float(rng.choice([.6, .7, .8, .9])),
                 1.0, f"t{trial}-{i}") for i in range(12)]
        pick = rng.random((2, 12)) < 0.6
        sa = [p for p, k in zip(pool, pick[0]) if k]
        sb = [p for p, k in zip(pool, pick[1]) if k]
        mk = lambda s: CompiledRegime(0.0, 1.0, backend=native, n_grid=16, p_two=0.1,
                                      probes=[p[:4] for p in s], probe_ids=[p[4] for p in s])
        joint, _ = federate(mk(sa), mk(sb))
        union = {p[4]: p for p in sa + sb}
        np.testing.assert_allclose(joint.p_plus_cells(),
                                   dense_union_p_plus(joint, [p[:4] for p in union.values()]),
                                   rtol=0, atol=1e-10)
