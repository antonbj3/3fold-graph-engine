"""evidence_federation.py — exact merge of two CompiledRegime evidence ledgers.

Optional research API. In the finite native family a unit probe adds a fixed natural
increment (A_c, B_c) to its cell. Two regimes built on the SAME prior and partition
therefore combine exactly by counting every physical observation once:

    eta_joint = eta_a + sum over probes of b whose evidence_id is not in a.

Posteriors alone cannot do this. Two regimes can have identical individual posteriors
while the correct joint differs: one shared observation must be counted once, two
independent observations with the same content twice. The evidence identity decides.

Refusals (never silent):
* different schema, domain, partition edges or transition prior -> IncompatibleFederationError;
* the same evidence_id with a different record (position, sign, reliability or weight)
  -> EvidenceConflictError. The two nodes disagree about what that observation said;
* constructor fallback IDs ``legacy:<i>`` are positional, not physical identities.
  Two unrelated regimes both hold ``legacy:0``; merging on them would silently drop or
  conflict. They are refused;
* claims carry no IDs in the ledger schema. ``claim_policy="identical"`` (default)
  requires the same claim multiset in both ledgers and counts it once (shared background);
  ``"disjoint"`` declares the claims independent and keeps both; anything else is refused.

Scope: probes are declared conditionally independent given the world. Unknown overlap
(no IDs), a persistent unknown reliability shared across nodes, or different priors do
not reduce to this additive merge; see the Lane D research report for the exact
mixture forms. ID equality is trusted as provenance; this module does not authenticate it.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

__all__ = ["FederationError", "IncompatibleFederationError", "EvidenceConflictError",
           "federation_context", "merge_evidence_ledgers", "federate"]

_SCHEMA = "compiled-regime-evidence-v1"
_LEGACY_PREFIX = "legacy:"


class FederationError(Exception):
    """Base class for refused federation."""


class IncompatibleFederationError(FederationError):
    """Prior, partition, domain or schema differ; natural coordinates are not comparable."""


class EvidenceConflictError(FederationError):
    """One evidence_id carries different likelihood records in the two ledgers."""


def federation_context(regime) -> dict:
    """Everything that must be equal before two ledgers can share natural coordinates.

    The partition edges and the effective transition prior define the base measure that
    each eta is relative to. ``potential``/``query_weights`` are task readouts and are not
    part of the posterior, so they are not compared.
    """
    prior = regime.transition_prior()
    if regime.n_grid is None:
        # Continuum backend (n_grid=None): no partition; the base measure is fixed by the
        # domain and the transition prior alone. "resolution" keeps it apart from native.
        return {"schema": _SCHEMA, "resolution": "continuum", "lo": float(regime.lo),
                "hi": float(regime.hi), "claim_model": regime.claim_model, "cells": None,
                "prior_mass": tuple(float(x) for x in prior["mass"]),
                "prior_normalizer": float(prior["normalizer"]),
                "two_family_enabled": bool(prior["two_family_enabled"])}
    return {"schema": _SCHEMA, "lo": float(regime.lo), "hi": float(regime.hi),
            "claim_model": regime.claim_model,
            "cells": tuple(map(tuple, np.asarray(regime.cells, dtype=float).tolist())),
            "prior_mass": tuple(float(x) for x in prior["mass"]),
            "prior_normalizer": float(prior["normalizer"]),
            "two_family_enabled": bool(prior["two_family_enabled"])}


def _check_contexts(ca, cb):
    for key in ("schema", "resolution", "lo", "hi", "claim_model", "cells", "prior_mass",
                "prior_normalizer", "two_family_enabled"):
        if ca.get(key) != cb.get(key):
            raise IncompatibleFederationError(
                f"federation context differs in {key!r}; natural states are relative to "
                "different base measures or partitions and cannot be added. Rebuild both on a "
                "common prior/partition from their raw evidence instead.")


def _probe_index(ledger, name):
    if not isinstance(ledger, dict) or ledger.get("schema") != _SCHEMA:
        raise IncompatibleFederationError(f"{name} is not a {_SCHEMA} ledger")
    index = {}
    for rec in ledger["probes"]:
        if len(rec) != 5:
            raise ValueError(f"{name}: probe records are (x, sign, reliability, weight, id)")
        pid = rec[4]
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"{name}: probe evidence_id must be a nonempty string")
        if pid.startswith(_LEGACY_PREFIX):
            raise FederationError(
                f"{name}: evidence_id {pid!r} is a positional constructor fallback, not a "
                "physical observation identity; supply probe_ids before federating")
        if pid in index:
            raise EvidenceConflictError(f"{name}: evidence_id {pid!r} occurs twice")
        index[pid] = tuple(rec)
    return index


def merge_evidence_ledgers(ledger_a, ledger_b, *, context_a, context_b,
                           claim_policy: str = "identical"):
    """Return (merged_ledger, report). Ledger a's order is kept; b's novel probes follow."""
    _check_contexts(context_a, context_b)
    ia = _probe_index(ledger_a, "ledger_a")
    ib = _probe_index(ledger_b, "ledger_b")
    shared, novel = [], []
    for pid, rec in ib.items():
        if pid in ia:
            if ia[pid] != rec:
                raise EvidenceConflictError(
                    f"evidence_id {pid!r} has different records: {ia[pid]!r} vs {rec!r}")
            shared.append(pid)
        else:
            novel.append(rec)
    ca = [tuple(c) for c in ledger_a["claims"]]
    cb = [tuple(c) for c in ledger_b["claims"]]
    if claim_policy == "identical":
        if Counter(ca) != Counter(cb):
            raise FederationError(
                "claims differ and carry no identities; declare claim_policy='disjoint' if "
                "they are independent evidence, or reconcile them first")
        claims = ca
    elif claim_policy == "disjoint":
        claims = ca + cb
    else:
        raise ValueError("claim_policy must be 'identical' or 'disjoint'")
    merged = {"schema": _SCHEMA, "claims": tuple(claims),
              "probes": tuple(ledger_a["probes"]) + tuple(novel)}
    report = {"shared_probe_ids": tuple(shared), "novel_probe_ids_from_b": tuple(r[4] for r in novel),
              "n_probes_a": len(ia), "n_probes_b": len(ib), "n_probes_joint": len(ia) + len(novel),
              "claim_policy": claim_policy, "n_claims_joint": len(claims)}
    return merged, report


def federate(regime_a, regime_b, *, backend=None, claim_policy: str = "identical"):
    """Build the joint CompiledRegime of two regimes, each observation counted once.

    ``backend`` defaults to regime_a's injected backend; the optional covariance backend
    (fast ``best_probe``) is also taken from regime_a. Task settings (potential, query
    weights, method) are taken from regime_a. Two continuum regimes (``n_grid=None``)
    federate by the same ID-union into a ``ContinuumRegime``; mixing a native and a
    continuum regime is refused. Returns (joint_regime, report).
    """
    from .compiled_regime import CompiledRegime
    ca, cb = federation_context(regime_a), federation_context(regime_b)
    merged, report = merge_evidence_ledgers(regime_a.evidence_ledger(), regime_b.evidence_ledger(),
                                            context_a=ca, context_b=cb, claim_policy=claim_policy)
    if ca.get("resolution") == "continuum":
        from .continuum_regime import ContinuumRegime
        joint = ContinuumRegime(
            regime_a.lo, regime_a.hi, reliability=regime_a.reliability, p_flip=regime_a.p_flip,
            p_two=regime_a.p_two, potential=regime_a.potential, claim_model=regime_a.claim_model,
            claims=merged["claims"], probes=[r[:4] for r in merged["probes"]],
            probe_ids=[r[4] for r in merged["probes"]])
        if federation_context(joint) != ca:
            raise IncompatibleFederationError("joint prior differs from the federated inputs")
        return joint, report
    joint = CompiledRegime(
        regime_a.lo, regime_a.hi, backend=backend if backend is not None else regime_a._backend,
        reliability=regime_a.reliability, p_flip=regime_a.p_flip, p_two=regime_a.p_two,
        potential=regime_a.potential, n_grid=regime_a.n_grid, claim_model=regime_a.claim_model,
        claims=merged["claims"], probes=[r[:4] for r in merged["probes"]],
        probe_ids=[r[4] for r in merged["probes"]], query_weights=regime_a._query_weights,
        method=regime_a._method,
        covariance_backend=getattr(regime_a, "_covariance_backend", None))
    if federation_context(joint) != ca:
        # Claims under 'disjoint' may add edges only if they were not already edges; the
        # contexts were equal, so this would indicate an internal inconsistency.
        raise IncompatibleFederationError("joint partition differs from the federated inputs")
    return joint, report
