#!/usr/bin/env python3
"""admission_bands.py — THREE-BAND admission for video-ingest segments (an agent worktree).

WHY (bottleneck #1, reports/probes/bottleneck_census.json rank 1): ingest yield is 0% —
34/34 real segments ABSTAIN because an agent worktree's video_admission_cert (correctly, null-safely)
refuses to rule with <2 geometry proxies, and this pipeline today produces exactly ONE
proxy leg (stream_eyes optical flow). G's sync-composition (the would-be 2nd leg) was
retracted (anti-correlates at n=24), so BINARY admit/abstain stays pinned at 0% yield
indefinitely.

INNOVATION (same three-band logic as scripts/bayes_qc_gate.py): replace binary
admit/abstain with three bands —

  AUTO_ADMIT: >=2 usable geometry proxies AND the D cert says ADMIT (synced, no vetoes).
               Segment is clean; processed unflagged.
  QUARANTINE: exactly 1 usable proxy (cert cannot rule — evidence is one-legged, not
               absent), OR >=2 proxies with WEAK evidence (median sync below ADMIT floor
               but no pair below the weakest-link floor and no veto). Segment IS processed
               but carries a QUARANTINE flag that propagates into the segment sidecar
               (existing 3-level metadata format) such that NO downstream read path can
               classify it as clean (flag-must-propagate rule).
  REJECT: 0 usable signals (missing / non-finite / degenerate / too short = corrupt),
               OR positive evidence of badness: a method-pair below the weakest-link floor,
               or any complementary veto (non-rigid, invalid motion, unpredictable motion,
               pluggable veto legs).

Band semantics vs the D cert (read-only cross-worktree import, never modified):
  cert ADMIT                      -> AUTO_ADMIT
  cert ABSTAIN, cert_valid=False  -> band decided by USABLE-LEG COUNT (1 -> QUARANTINE,
                                     0 -> REJECT); the cert's null-safety is a statement
                                     about decidability, not about the segment being bad.
  cert ABSTAIN, cert_valid=True:
      any veto / weakest-link pair-disagreement -> REJECT   (named failure detected)
      low_sync only (median < SYNC_THRESH, min_pair ok)     -> QUARANTINE (weak evidence)

The QUARANTINE flag's NEVER-CLEAN invariant (enforced by apply_band_to_sidecar +
downstream_clean_view, selftested adversarially):
  - top-level  sidecar["quarantine"] = True
  - top-level  sidecar["admission_band"] = {band, quarantine, band_reason,...}
  - legacy     sidecar["agreement_cert"]["verdict"] is REWRITTEN to "quarantine"
               (never "admit") and its reason is prefixed "QUARANTINE:" — so an old
               consumer testing verdict == "admit" can NEVER see a quarantined segment
               as clean, and a reason-reader sees the flag inline.
  - original cert output preserved verbatim under sidecar["agreement_cert_prebands"].
LEVEL-3 note: the mention index stores only {term -> segment_id}; every consumer must
load the LEVEL-2 sidecar for content, which is exactly where the flag lives — so the
flag rides every downstream read path of the 3-level format.

Opt-in only: scripts/video_ingest.py grows a --bands flag that routes through
band_admission; default ingest behaviour is UNCHANGED.

gate: python3 scripts/admission_bands.py --selftest
run: python3 scripts/admission_bands.py --run-real   (bands the 34 real sidecars,
      writes flags into data/video_segments/*.json, emits
      reports/probes/admission_bands_v0.json)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SEGMENTS_DIR = REPO_ROOT / "data" / "video_segments"
SOURCES_DIR = REPO_ROOT / "data" / "video_sources"
REPORT_PATH = REPO_ROOT / "reports" / "probes" / "admission_bands_v0.json"


GOVERNOR_STATE = REPO_ROOT / "data" / "GOVERNOR_STATE.json"   # resource_governor.py


def load_governor_admission():
    """OPT-IN read of a resource-governor state file: when compute is the binding resource the
    governor recommends a sample floor for QUARANTINE processing (weakest evidence = lowest value
    per compute-second => deferred, processed when compute frees up). Returns a dict, or None when
    the file is absent; without --governor nothing is read and the default behaviour is unchanged."""
    try:
        with open(GOVERNOR_STATE) as f:
            st = json.load(f)
        av = st["recommended_thresholds"]["admission_video"]
        return dict(defer_below_samples=int(av["quarantine_defer_below_samples"]),
                    binding=st["decisive"]["binding_resource"],
                    state_generated=st.get("generated"))
    except Exception:
        return None

BANDS = ("AUTO_ADMIT", "QUARANTINE", "REJECT")
# legacy-verdict string per band (what agreement_cert.verdict is rewritten to)
BAND_VERDICT = {"AUTO_ADMIT": "admit", "QUARANTINE": "quarantine", "REJECT": "reject"}


def _load_cert_module():
    """Return the video_admission_cert module, or None if it is unavailable."""
    try:
        from . import video_admission_cert as vac
        return vac
    except ImportError:
        try:
            import video_admission_cert as vac
            return vac
        except Exception:
            return None


# Metadata PREFILTER leg. Kind-decorrelated: metadata (listing text, cheap, no video decode) runs
# before the signal certs (expensive). A metadata REJECT drops the segment WITHOUT extraction =
# throughput. The composed band is weakest-link. The prefilter module is optional.

def band_admission_prefiltered(segment_context_text, method_signals, temporal=None,
                               veto_legs=None, metadata_optional=False, **kw):
    """Metadata-prefiltered admission (compose_admission + a consumer-side authenticity rider).

    The composed authenticity guard is NOMINAL (it checks field NAMES, not a verified signature) and
    load-bearing on the consume_fields whitelist. Until a real provenance leg (signature verification
    with a positive RESULT) exists, this wrapper caps grounding_kind at 'consistency-only' whatever
    the composition returns, so a self-declared c2pa_verified field can never buy a provenance stamp.
    metadata_optional=True = throughput mode (only a hard REJECT drops the segment).

    Requires an external `compose_admission(segment_context_text, band, method_signals, ...)`
    prefilter module; if it is not importable the signal certs run alone and the result is tagged."""
    try:
        from gate_prefilter_kind_composition import compose_admission  # type: ignore
    except Exception as e:
        # loud degradation: run the signal certs alone, tag that the prefilter was unavailable
        res = band_admission(method_signals, temporal=temporal, veto_legs=veto_legs, **kw)
        res["prefilter"] = f"UNAVAILABLE ({type(e).__name__}) — signal cert only"
        return res
    res = compose_admission(segment_context_text, "band", method_signals,
                            temporal=temporal, veto_legs=veto_legs,
                            metadata_optional=metadata_optional)
    # authenticity rider (see docstring): cap until a verified provenance leg exists
    if res.get("grounding_kind") == "provenance-corroborated":
        res["grounding_kind"] = "consistency-only"
        res["grounding_capped_by"] = "consumer rider: authenticity guard nominal until a verified C2PA leg exists"
    return res


def band_admission_complementary_and(leg_verdicts: dict) -> dict:
    """SIBLING composition path to band_admission — the SYNC-path above is untouched and stays
    the correct mode for REDUNDANT estimators (same target quantity, cross-checked by agreement).
    This path is for COMPLEMENTARY legs: each leg already carries its OWN calibrated PREREG-band
    verdict (PASS/FAIL) plus which of its internal sub-signals ("mechanisms") actually carried
    that PASS. Two legs whose PASS is carried by the SAME detection mechanism are not effectively
    independent evidence — see docs/... flow_leg.py/appearance_leg.py's declared C4 shared-order-
    mechanism risk (measured verdict-correlation 0.546 on the calibration pool, entirely traceable
    to both legs' temporal-continuity axis reusing the identical order_gain/splice_dip statistic).

    leg_verdicts: {leg_name: {"verdict": "PASS"|"FAIL", "mechanisms": iterable[str]}}
      `mechanisms` = the mechanism CLASSES that actually carried THIS leg's PASS (empty/ignored
      if the leg is FAIL). Mechanism-class naming is the caller's contract (e.g.
      "magnitude-physicality", "spatial-coherence", "order-gain", "residual-stability").

    Admission rule (effective-independence, not just leg-count):
      >=2 legs PASS  AND  the UNION of mechanism classes across those PASSing legs has >=2
      DISTINCT members. Set union is what enforces "the shared mechanism counts in exactly
      one ben": if leg A's PASS is carried by {"order-gain"} and leg B's PASS is ALSO carried
      only by {"order-gain"}, the union stays size 1 -- the composed verdict correctly refuses
      to treat two order-gain-only PASSes as 2 independent votes. If either leg ALSO carries an
      unrelated mechanism (leg-unique axis), the union grows and the composition passes.

    CALLER CONTRACT: the function TRUSTS the caller's mechanism labels and cannot see (a)
    pool-degenerate legs (a leg whose verdict is CONSTANT over the whole candidate pool carries zero
    discriminating information in that pool yet satisfies the mechanism quorum vacuously) or (b)
    shared SOURCE LINEAGE (two legs that are functions of the same input variable are not independent
    evidence even when their mechanism names differ). RULE: the caller MUST exclude pool-constant legs
    and lineage-collapsed leg pairs BEFORE composition (measure variance + a lineage matrix). Two
    always-PASS legs with different labels otherwise AUTO_ADMIT everything (verified counterexample).

    Returns dict(band, quarantine, band_reason, n_legs_pass, mechanism_union, per_leg) — same
    band vocabulary (AUTO_ADMIT/QUARANTINE/REJECT) as band_admission so apply_band_to_sidecar
    and the never-clean invariant apply unmodified to this path too."""
    n_total = len(leg_verdicts)
    passing = {name: lv for name, lv in leg_verdicts.items() if lv.get("verdict") == "PASS"}
    n_pass = len(passing)
    mech_union: set[str] = set()
    per_leg = {}
    for name, lv in leg_verdicts.items():
        mechs = sorted(set(lv.get("mechanisms") or [])) if lv.get("verdict") == "PASS" else []
        per_leg[name] = dict(verdict=lv.get("verdict"), mechanisms=mechs)
        mech_union |= set(mechs)

    if n_pass == 0:
        band = "REJECT"
        reason = f"no_leg_pass: 0/{n_total} complementary legs PASS"
        quarantine = False
    elif n_pass == 1:
        band = "QUARANTINE"
        reason = (f"single_leg_pass: only 1/{n_total} legs PASS — complementary-AND cannot rule "
                  "with one leg (needs >=2 PASS + >=2 distinct mechanism classes)")
        quarantine = True
    elif len(mech_union) >= 2:
        band = "AUTO_ADMIT"
        reason = (f"{n_pass}/{n_total} legs PASS, mechanism union={sorted(mech_union)} "
                  "(>=2 distinct classes -- effective independence satisfied)")
        quarantine = False
    else:
        band = "QUARANTINE"
        reason = (f"{n_pass}/{n_total} legs PASS but mechanism union={sorted(mech_union)} is NOT "
                  ">=2 distinct — PASSes are carried by the SAME mechanism class (effective "
                  "independence unmet: the legs agree, but on a shared detection axis, not on "
                  "genuinely decorrelated evidence); processed WITH flag, never as clean")
        quarantine = True

    return dict(band=band, quarantine=quarantine, band_reason=reason,
               n_legs_pass=n_pass, n_legs_total=n_total,
               mechanism_union=sorted(mech_union), per_leg=per_leg)


def _usable_legs(method_signals, degen_std):
    """Replicates the cert's own usability criteria (T>=4, all finite, std>DEGEN_STD)
    so the band's leg-count agrees with what the cert itself would keep."""
    usable = []
    for name, sig in (method_signals or {}).items():
        s = np.asarray(sig, dtype=float)
        s = s.reshape(s.shape[0], -1) if s.ndim > 1 else s.reshape(-1, 1)
        if s.shape[0] >= 4 and np.all(np.isfinite(s)) and float(s.std()) > degen_std:
            usable.append(name)
    return usable


def band_admission(method_signals, temporal=None, veto_legs=None, method_families=None,
                   family_error_corr=None, family_corr_validated=False, family_corr_strata=None):
    """Three-band wrapper around an agent worktree's video_admission_cert.

    Returns dict(band, quarantine, band_reason, usable_legs, cert=<raw cert dict|None>).
    Never raises on bad input: unloadable cert -> QUARANTINE (cert unavailable is weak
    evidence, not proof of badness, and must never silently ADMIT).

    family_error_corr (D ROND-2 design-call): the MEASURED per-family-pair residual
    error-correlation. Passing it (<FAMILY_DECORR_THRESH=0.50) is the ONLY way to reach
    guard_basis='effective-measured' -> AUTO_ADMIT. Declared families alone stay
    'nominal-declared' -> coherent_fake_band DEMAND -> QUARANTINE-with-flag (two
    declared-different families can share a hidden coherent bias). AUTO_ADMIT is earned by
    MEASURING decorrelation, never by declaring it — a 0-auto-admit pipeline is a SOURCE
    problem (doesn't measure the quantity), fixed source-side, not by weakening the band.

    family_corr_validated (D cert hardening): a family_error_corr that is only
    synthetic/offline-calibrated is MEASURED-but-UNVALIDATED -> guard_basis
    'measured-unvalidated' -> booked NOMINAL -> DEMAND -> QUARANTINE. It must NOT hard-drive
    the effective path (AUTO_ADMIT *or* hard-QUARANTINE) on real data whose calibration->
    deployment gap is undischarged (both directions inherit the synthetic->real gap). Pass
    family_corr_validated=True ONLY after discharging that gap on the deployment domain."""
    vac = _load_cert_module()
    if vac is None:
        return dict(band="QUARANTINE", quarantine=True,
                    band_reason="cert_unavailable: graph_engine.video_admission_cert import "
                                "failed — cannot rule, process only under flag",
                    usable_legs=[], cert=None)

    usable = _usable_legs(method_signals, vac.DEGEN_STD)

    if len(usable) == 0:
        # Distinguish SUB-FLOOR weak signal from truly absent/corrupt signal: a leg with
        # >=2 finite, varying samples but fewer than the cert's >=4 floor (e.g. a short
        # segment under a coarse per-chunk extractor) carries WEAK evidence, not none —
        # band doctrine says QUARANTINE, not REJECT. REJECT is reserved for genuinely
        # no/corrupt signal: empty, <2 samples, non-finite, or (near-)constant.
        subfloor = []
        for name, sig in (method_signals or {}).items():
            s = np.asarray(sig, dtype=float)
            if s.size == 0:
                continue
            s = s.reshape(s.shape[0], -1) if s.ndim > 1 else s.reshape(-1, 1)
            if 2 <= s.shape[0] < 4 and np.all(np.isfinite(s)) and float(s.std()) > vac.DEGEN_STD:
                subfloor.append(dict(name=name, n_samples=int(s.shape[0])))
        if subfloor:
            desc = ", ".join(f"{d['name']}({d['n_samples']} samples)" for d in subfloor)
            return dict(band="QUARANTINE", quarantine=True,
                        band_reason=f"subfloor_signal: finite varying signal below the cert's "
                                    f">=4-sample floor ({desc}) — weak evidence, processed "
                                    "WITH flag, never as clean",
                        usable_legs=[], cert=None)
        return dict(band="REJECT", quarantine=False,
                    band_reason="no_signal: 0 usable proxies and no sub-floor signal "
                                "(missing/<2-sample/non-finite/degenerate = corrupt input)",
                    usable_legs=[], cert=None)

    cert = vac.video_admission_cert(method_signals, temporal=temporal,
                                    veto_legs=veto_legs, method_families=method_families,
                                    family_error_corr=family_error_corr,
                                    family_corr_validated=family_corr_validated,
                                    family_corr_strata=family_corr_strata)

    if len(usable) == 1:
        return dict(band="QUARANTINE", quarantine=True,
                    band_reason="single_proxy: 1 usable geometry proxy "
                                f"({usable[0]}) — cert cannot rule (needs >=2); "
                                "processed WITH flag, never as clean",
                    usable_legs=usable, cert=cert)

    # >=2 usable legs: the cert actually ruled.
    if cert["verdict"] == "ADMIT":
        # WIRE (D 68b9b10f2): gate AUTO_ADMIT on the coherent-fake band, not raw cert.
        # Closes the 11-surface dye-test leak (design was folded but not in the live path).
        cf = vac.coherent_fake_band(cert)   # guard_basis + coverage-{geom} + CI
        if cf == "AUTO_ADMIT":
            return dict(band="AUTO_ADMIT", quarantine=False,
                        band_reason=f"cert ADMIT + coherent-fake-clear: {cert['reason']}",
                        usable_legs=usable, cert=cert)
        # my scheme has no DEMAND band -> map to QUARANTINE-with-flag (D's own guidance:
        # processed WITH review flag, NEVER auto-clean). Preserves never-clean invariant.
        # D remediation refinement (QC on 6e21e7a63): the two non-clean outcomes have
        # DISTINCT remediations and must be distinguishable downstream:
        # unvalidated corr -> QUARANTINE + remediation=re-measure (recoverable:
        # validate the corr on the deployment/confusion mode)
        # VALIDATED corr >= bar -> REJECT (PROVEN shared bias on the deployment domain:
        # the ensemble's agreement is actively misleading, not
        # merely undecided — positive evidence of badness) +
        # remediation=replace-method-pair.
        if cf == "DEMAND":
            return dict(band="QUARANTINE", quarantine=True,
                        band_reason=f"coherent-fake nominal/uncovered -> review (demand->quarantine): {cert['reason']}",
                        remediation="re-measure: validate family_error_corr on the deployment "
                                    "domain's ADVERSARIAL-MINIMUM confusion mode (or add coverage)",
                        usable_legs=usable, cert=cert)
        if (cert.get("family_corr_validated") and cert.get("guard_basis") == "effective-measured"
                and cert.get("coherent_fake_guarded") is False):
            return dict(band="REJECT", quarantine=False,
                        band_reason=f"proven shared bias: VALIDATED family_error_corr="
                                    f"{cert.get('family_error_corr')} >= {vac.FAMILY_DECORR_THRESH} — "
                                    f"agreement is misleading, not merely undecided: {cert['reason']}",
                        remediation="replace-method-pair: this ensemble cannot certify this "
                                    "segment; re-admit only with a decorrelated/orthogonal pair",
                        usable_legs=usable, cert=cert)
        return dict(band="QUARANTINE", quarantine=True,
                    band_reason=f"coherent-fake UNGUARDED: {cert['reason']}",
                    remediation="re-measure: guard basis insufficient (declare families / "
                                "measure family_error_corr)",
                    usable_legs=usable, cert=cert)

    if not cert.get("cert_valid", False):
        # defensive: cert filtered differently than we did -> weak evidence, not badness
        return dict(band="QUARANTINE", quarantine=True,
                    band_reason=f"cert_invalid_with_signal: {cert['reason']}",
                    usable_legs=usable, cert=cert)

    reason = cert.get("reason", "")
    if cert.get("vetoes") or "method_disagreement" in reason:
        return dict(band="REJECT", quarantine=False,
                    band_reason=f"cert named a failure: {reason}",
                    usable_legs=usable, cert=cert)

    # remaining ABSTAIN cause is low_sync (median below ADMIT floor, no pair below
    # weakest-link floor, no veto) = weak evidence
    return dict(band="QUARANTINE", quarantine=True,
                band_reason=f"weak_sync: {reason} — processed WITH flag",
                usable_legs=usable, cert=cert)


# --------------------------------------------------------------------------
# Provenance masking (D ROND-2 finding 2): the served packet must speak with ONE
# voice = the band's. Raw sync-cert fields (cert.verdict="ADMIT", the preserved
# pre-band verdict, and cert.reason which contains English clean-words like "passes")
# are SERVED in the sidecar, so a naive value-level consumer that greps for an
# admit/clean token would read a raw SYNC verdict as the admission decision. D's rule:
# a served packet must never carry a raw "admit" that contradicts the band. When the
# band is NOT AUTO_ADMIT (i.e. never-clean), we rewrite any clean-reading string leaf
# in the raw-cert provenance subtrees so it (a) preserves the original text for
# provenance, (b) carries the authoritative band decision, and (c) carries the agent pool
# "quarantine" never-clean marker. Self-contained token set (no dependency on D's
# private tracer); the marker itself is what a value-grep keys on.
import re as _re
# Word-boundary matcher (shim-QC catch): the earlier padded-substring set
# (" ok", "ok ") missed a leaf that is EXACTLY "ok" (no padding), while a bare substring
# "ok" would false-positive inside e.g. Swedish "protokollet". \b-anchored prefix classes
# cover admit/admitted, cleaned, accepted, verified, validity, passes, ok — and nothing
# embedded mid-word.
_PROV_CLEAN_RE = _re.compile(
    r"\b(admit\w*|clean\w*|accept\w*|green|verif\w*|valid\w*|pass\w*|ok)\b")


def _mask_provenance_clean_tokens(obj, band):
    if isinstance(obj, dict):
        return {k: _mask_provenance_clean_tokens(v, band) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask_provenance_clean_tokens(v, band) for v in obj]
    if isinstance(obj, str):
        low = obj.lower()
        if _PROV_CLEAN_RE.search(low) and "quarantine" not in low:
            return (f"{obj} [PROVENANCE raw-sync-only — admission={BAND_VERDICT[band]} "
                    f"(band={band}); NOT-clean; quarantine-invariant/see admission_band.band]")
    return obj


# --------------------------------------------------------------------------
# Flag propagation into the existing 3-level metadata format (LEVEL 2 sidecar)
# --------------------------------------------------------------------------
def apply_band_to_sidecar(sidecar: dict, band_result: dict, ts: str | None = None) -> dict:
    """Injects the band into a segment sidecar IN PLACE (and returns it), enforcing the
    NEVER-CLEAN invariant for QUARANTINE/REJECT on every read path:
      new path: sidecar["admission_band"]/["quarantine"]
      legacy path: sidecar["agreement_cert"]["verdict"] rewritten (never "admit" unless
                    band == AUTO_ADMIT); reason prefixed with the band flag.
    The pre-band cert output is preserved under agreement_cert_prebands (write-once)."""
    band = band_result["band"]
    assert band in BANDS, f"unknown band {band}"
    quarantine = bool(band_result.get("quarantine", band == "QUARANTINE"))
    # QUARANTINE implies the flag; a quarantine flag without QUARANTINE band is forbidden
    assert quarantine == (band == "QUARANTINE"), "quarantine flag must equal band==QUARANTINE"

    if "agreement_cert_prebands" not in sidecar and "agreement_cert" in sidecar:
        sidecar["agreement_cert_prebands"] = sidecar["agreement_cert"]

    verdict = BAND_VERDICT[band]
    reason = band_result.get("band_reason", "")
    if band != "AUTO_ADMIT":
        reason = f"{band}: {reason}"  # flag inline in the reason string too

    cert_served = band_result.get("cert")
    if band != "AUTO_ADMIT":
        # D ROND-2 finding 2: mask every raw-sync clean-token in the SERVED provenance
        # subtrees so the packet speaks with one voice (the band's). Original text kept,
        # band decision + never-clean marker injected.
        if "agreement_cert_prebands" in sidecar:
            sidecar["agreement_cert_prebands"] = _mask_provenance_clean_tokens(
                sidecar["agreement_cert_prebands"], band)
        cert_served = _mask_provenance_clean_tokens(cert_served, band)

    sidecar["admission_band"] = dict(
        band=band, quarantine=quarantine, band_reason=reason,
        usable_legs=band_result.get("usable_legs", []),
        banded_ts=ts or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        cert=cert_served,
    )
    sidecar["quarantine"] = quarantine
    sidecar["agreement_cert"] = dict(verdict=verdict, reason=reason, band=band)
    return sidecar


def downstream_clean_view(sidecar: dict) -> dict:
    """The consumer-side read: a segment is CLEAN only if EVERY read path agrees.
    Any quarantine/reject marker on ANY path -> not clean. This is the function a
    downstream consumer should call (and what the selftest attacks adversarially)."""
    band = (sidecar.get("admission_band") or {}).get("band")
    q_flag = bool(sidecar.get("quarantine", False)) or \
        bool((sidecar.get("admission_band") or {}).get("quarantine", False))
    legacy_verdict = (sidecar.get("agreement_cert") or {}).get("verdict")
    clean = (band == "AUTO_ADMIT") and (not q_flag) and (legacy_verdict == "admit")
    return dict(segment_id=sidecar.get("segment_id"), clean=clean,
                quarantine=q_flag, band=band, legacy_verdict=legacy_verdict)


# --------------------------------------------------------------------------
# Real-corpus run: the 34 sidecars in data/video_segments/
# --------------------------------------------------------------------------
def _seeded_flow_series(segment_id: str, n: int):
    """CONSTRUCTED stand-in for the discarded stream_eyes optical-flow series.

    REFRAME (declared in the report): the raw videos were deleted by the
    stream-process-discard contract and the sidecars store only frame windows, so the
    flow VALUES are not reconstructible. What IS measured from surviving metadata is
    leg AVAILABILITY at the pipeline's actual sampling: stream_eyes ran on every source
    (ingest_first_real_run.json) and emits one [median,p90] flow sample per fixed
    CHUNK; n = the number of chunks overlapping the segment's real frame window. The
    series below (seeded by segment_id, correct length n) exists ONLY to drive the
    cert's code path; banding at the 0/1-proxy level depends on availability + sample
    count, not values."""
    seed = int(hashlib.sha256(segment_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    med = np.abs(np.cumsum(rng.standard_normal(n))) * 0.5 + 0.1
    p90 = med * (1.5 + 0.1 * rng.standard_normal(n))
    return np.stack([med, p90], axis=1)


def _n_chunk_samples(frame_start: int, frame_end: int, chunk_frames: int) -> int:
    """Number of stream_eyes fixed chunks [k*cs, (k+1)*cs) overlapping [frame_start,
    frame_end) — replicates video_ingest._optical_flow_method_signal's overlap rule."""
    if frame_end <= frame_start:
        return 0
    return len([c0 for c0 in range(0, frame_end + chunk_frames, chunk_frames)
                if c0 < frame_end and c0 + chunk_frames > frame_start])


def run_real(segments_dir: Path = SEGMENTS_DIR, write: bool = True,
             chunk_frames: int = 30, governor: dict | None = None) -> dict:
    """Bands the real segment sidecars at the pipeline-ACTUAL signal granularity:
    stream_eyes emits one flow sample per `chunk_frames` fixed chunk (the real run used
    the default 30), so a segment's single available leg carries n_chunk samples — NOT
    one per frame. A sensitivity ladder over chunk sizes is reported alongside."""
    sidecar_paths = sorted(segments_dir.glob("*.json"))
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rows = []
    for path in sidecar_paths:
        with open(path) as f:
            sidecar = json.load(f)
        seg = sidecar.get("triage_segment", {})
        a, b = int(seg.get("frame_start", 0)), int(seg.get("frame_end", 0))
        T = b - a
        n = _n_chunk_samples(a, b, chunk_frames)
        # a segment shorter than 4 frames has no cert-grade signal at ANY chunking
        if T < 4:
            n = min(n, 1)
        sigs = {}
        if n >= 1:
            sigs["optical_flow_stream_eyes"] = _seeded_flow_series(sidecar["segment_id"], n)
        band_result = band_admission(sigs)
        apply_band_to_sidecar(sidecar, band_result, ts=ts)
        # OPT-IN governor defer: an ANNOTATION on top of the band (band + NEVER-CLEAN
        # invariant untouched -- QUARANTINE already never reads clean). Weakest-evidence
        # segments (fewest samples) have the lowest value per compute-second; when compute
        # is the binding resource their expensive processing is deferred.
        gov_defer = False
        if governor is not None and band_result["band"] == "QUARANTINE" \
                and n < governor["defer_below_samples"]:
            gov_defer = True
            sidecar["admission_band"]["governor_defer"] = True
            sidecar["admission_band"]["governor_note"] = (
                f"deferred: n_flow_samples={n} < defer_floor="
                f"{governor['defer_below_samples']} (binding={governor['binding']}, "
                f"state {governor['state_generated']})")
        if write:
            with open(path, "w") as f:
                json.dump(sidecar, f, indent=2, default=_np_default)
        rows.append(dict(segment_id=sidecar["segment_id"], T_frames=T,
                         n_flow_samples=n, band=band_result["band"],
                         quarantine=band_result["quarantine"],
                         governor_defer=gov_defer,
                         band_reason=band_result["band_reason"]))
    dist = {b: sum(1 for r in rows if r["band"] == b) for b in BANDS}
    # sensitivity ladder: how the distribution moves with the instrument's chunk size
    ladder = {}
    for cs in (30, 10, 1):
        counts = {b: 0 for b in BANDS}
        for path in sidecar_paths:
            with open(path) as f:
                sc = json.load(f)
            seg = sc.get("triage_segment", {})
            a, b = int(seg.get("frame_start", 0)), int(seg.get("frame_end", 0))
            n_ = _n_chunk_samples(a, b, cs)
            if (b - a) < 4:
                n_ = min(n_, 1)
            if n_ >= 4:
                counts["QUARANTINE"] += 1          # single usable proxy
            elif n_ >= 2:
                counts["QUARANTINE"] += 1          # subfloor weak signal
            else:
                counts["REJECT"] += 1
        ladder[f"chunk_frames={cs}"] = counts
    return dict(n_segments=len(rows), chunk_frames=chunk_frames,
                band_distribution=dist, sensitivity_ladder=ladder, rows=rows,
                segments_dir=str(segments_dir), written=write, banded_ts=ts)


def _np_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# --------------------------------------------------------------------------
# Selftest: synthetic battery hitting all three bands + adversarial invariant check
# --------------------------------------------------------------------------
def selftest() -> int:
    rng = np.random.default_rng(0)
    T = 60
    truth = np.cumsum(rng.standard_normal((T, 3)), axis=0)
    noisy = lambda s: truth + s * rng.standard_normal((T, 3))
    ok = True
    cases = []

    def check(name, band_result, expect_band):
        nonlocal ok
        passed = band_result["band"] == expect_band
        ok = ok and passed
        cases.append((name, passed, band_result["band"], expect_band,
                      band_result["band_reason"][:70]))
        return band_result

    # AUTO_ADMIT needs BOTH axes effective (coherent_fake_band = min of family+coverage):
    # family: family_error_corr < FAMILY_DECORR_THRESH -> effective-measured
    # coverage: every non-geometry FAKE_DIM CI-covered by a veto leg
    _vac0 = _load_cert_module()
    # D 57223c717 (coverage meta-mirage): a dim is VALIDATED-covered only via coverage_strata
    # CONTAINING the adversarial_minimum stratum with worst ci_lo >= EFF_COVER_MAG.
    _covlegs = [_vac0.veto_leg(d, covers=[d], score_effective=0.9, score_effective_ci_lo=0.8,
                               coverage_strata={"adversarial_minimum": 0.75, "easy": 0.9})
                for d in ("motion", "deformation", "acquisition", "appearance", "semantic")]
    # 1) two synced + VALIDATED measured decorr + full coverage -> AUTO_ADMIT (the earned path).
    # family_corr_validated=True = the calibration->deployment gap is discharged.
    check("2 synced + VALIDATED decorr + full coverage", band_admission(
        {"sfm": noisy(0.05), "flow": noisy(0.05)},
        method_families={"sfm": "classical-geo", "flow": "learned-mono"},
        family_error_corr=0.05, family_corr_validated=True, veto_legs=_covlegs), "AUTO_ADMIT")
    # 1c) measured-but-UNVALIDATED corr BELOW the bar (D hardening; B's synthetic 0.563 class):
    # an offline/synthetic-only corr must NOT hard-drive the effective path on real data ->
    # guard_basis='measured-unvalidated' -> NOMINAL -> DEMAND -> QUARANTINE. Even a
    # below-bar value does not auto-admit until validated on the deployment domain.
    check("measured-but-UNVALIDATED decorr (synthetic)", band_admission(
        {"sfm": noisy(0.05), "flow": noisy(0.05)},
        method_families={"sfm": "classical-geo", "flow": "learned-mono"},
        family_error_corr=0.05, family_corr_validated=False, veto_legs=_covlegs), "QUARANTINE")
    # 1d) VALIDATED corr >= bar = PROVEN shared bias on the deployment domain -> REJECT
    # (D remediation refinement: agreement is actively misleading, not merely
    # undecided; remediation=replace-method-pair, distinct from 1c's re-measure).
    r_bias = check("VALIDATED shared bias (corr>=bar)", band_admission(
        {"sfm": noisy(0.05), "flow": noisy(0.05)},
        method_families={"sfm": "classical-geo", "flow": "learned-mono"},
        family_error_corr=0.99, family_corr_validated=True, veto_legs=_covlegs), "REJECT")
    if not (r_bias.get("remediation", "").startswith("replace-method-pair")
            and "proven shared bias" in r_bias["band_reason"]):
        ok = False
        cases.append(("1d remediation flag", False, "-", "-", "missing replace-method-pair remediation"))
    # 1b) -hardening (D ROND-2 finding 1): declared families WITHOUT a measured
    # error-corr = nominal-declared -> coherent_fake_band DEMAND -> QUARANTINE-with-flag,
    # NEVER AUTO_ADMIT. Two declared-different families can share a hidden coherent bias;
    # auto-admitting on the nominal count re-opens the leak the band exists to close.
    check("declared families, NO measured decorr (nominal)", band_admission(
        {"sfm": noisy(0.05), "flow": noisy(0.05)},
        method_families={"sfm": "classical-geo", "flow": "learned-mono"}), "QUARANTINE")
    # 2) single proxy -> QUARANTINE (the 34-segment reality)
    r_q = check("1 proxy (pipeline today)", band_admission(
        {"optical_flow_stream_eyes": np.abs(noisy(0.05))}), "QUARANTINE")
    # 3) zero usable signal (constant series = degenerate) -> REJECT
    check("0 usable (degenerate)", band_admission(
        {"flat": np.ones((T, 2))}), "REJECT")
    # 4) non-finite input -> REJECT
    bad = noisy(0.05); bad[3] = np.nan
    check("0 usable (non-finite only)", band_admission({"bad": bad}), "REJECT")
    # 4b) SUB-FLOOR signal: 2-3 finite varying samples (< cert's 4-floor) = weak
    # evidence, NOT corrupt -> QUARANTINE (the coarse-chunk short-segment case)
    check("subfloor signal (3 samples)", band_admission(
        {"optical_flow_stream_eyes": np.array([[0.2, 0.5], [0.9, 1.4], [0.4, 0.8]])}),
        "QUARANTINE")
    # 4c) a single sample cannot even show variation -> REJECT
    check("1 sample only", band_admission(
        {"optical_flow_stream_eyes": np.array([[0.2, 0.5]])}), "REJECT")
    # 5) hard disagreement (one garbage method, weakest-link) -> REJECT
    check("pair disagreement", band_admission(
        {"sfm": noisy(0.05), "flow": rng.standard_normal((T, 3))}), "REJECT")
    # 6) veto (deformation) despite sync -> REJECT
    check("deformation veto", band_admission(
        {"sfm": noisy(0.05), "flow": noisy(0.05)},
        temporal={"deformation_flagged": True}), "REJECT")
    # 7) weak sync (correlated-but-noisy: median in [MIN_PAIR, SYNC) band) -> QUARANTINE
    vac = _load_cert_module()
    weak = None
    for scale in np.linspace(0.5, 6.0, 40):  # find a noise level landing in the weak band
        a, b = truth + scale * rng.standard_normal((T, 3)), truth + scale * rng.standard_normal((T, 3))
        c = vac.video_admission_cert({"sfm": a, "flow": b})
        if c["cert_valid"] and c["verdict"] == "ABSTAIN" and "low_sync" in c["reason"]:
            weak = {"sfm": a, "flow": b}
            break
    if weak is None:
        cases.append(("weak-sync search", False, "-", "QUARANTINE", "no weak-band pair found"))
        ok = False
    else:
        check("weak sync (low_sync only)", band_admission(weak), "QUARANTINE")

    # 8) NEVER-CLEAN invariant, adversarially: take a quarantined band result, apply it
    # to a sidecar that PREVIOUSLY claimed admit, then attack every read path.
    sidecar = dict(segment_id="selftest_seg", triage_segment=dict(frame_start=0, frame_end=T),
                   agreement_cert=dict(verdict="admit", reason="stale pre-band admit"))
    apply_band_to_sidecar(sidecar, r_q)
    view = downstream_clean_view(sidecar)
    attacks = [
        ("consumer view clean flag", view["clean"] is False),
        ("legacy verdict path", sidecar["agreement_cert"]["verdict"] != "admit"),
        ("top-level quarantine flag", sidecar["quarantine"] is True),
        ("band block flag", sidecar["admission_band"]["quarantine"] is True),
        ("flag inline in reason string", "QUARANTINE" in sidecar["agreement_cert"]["reason"]),
        # provenance preserved BUT masked: original text kept, never-clean marker injected
        # (finding 2 — a served raw 'admit' must carry the quarantine marker)
        ("pre-band provenance preserved+masked",
         "stale pre-band admit" in sidecar["agreement_cert_prebands"]["reason"]
         and "quarantine" in sidecar["agreement_cert_prebands"]["reason"].lower()),
        ("pre-band raw verdict masked never-clean",
         "quarantine" in str(sidecar["agreement_cert_prebands"]["verdict"]).lower()),
        ("json round-trip keeps flag", json.loads(json.dumps(sidecar, default=_np_default))["quarantine"] is True),
    ]
    for name, passed in attacks:
        ok = ok and passed
        cases.append((f"invariant: {name}", passed, "-", "-", ""))
    # 8c) governor-defer (opt-in) annotates but NEVER changes band/clean-view;
    # default path (governor=None) leaves sidecars annotation-free.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for sid, t_frames in (("gov_short", 45), ("gov_long", 300)):   # 30-chunks: 2 vs 10 samples
            with open(Path(td) / f"{sid}.json", "w") as f:
                json.dump(dict(segment_id=sid,
                               triage_segment=dict(frame_start=0, frame_end=t_frames)), f)
        res_gov = run_real(Path(td), write=True, chunk_frames=30,
                           governor=dict(defer_below_samples=4, binding="compute",
                                         state_generated="selftest"))
        rows_g = {r["segment_id"]: r for r in res_gov["rows"]}
        with open(Path(td) / "gov_short.json") as f:
            sc_short = json.load(f)
        res_def = run_real(Path(td), write=False, chunk_frames=30)     # default: no governor
        passed = (rows_g["gov_short"]["governor_defer"] is True
                  and rows_g["gov_long"]["governor_defer"] is False
                  and rows_g["gov_short"]["band"] == "QUARANTINE"      # band untouched
                  and downstream_clean_view(sc_short)["clean"] is False
                  and sc_short["admission_band"]["governor_defer"] is True
                  and all(r["governor_defer"] is False for r in res_def["rows"]))
        ok = ok and passed
        cases.append(("governor defer: opt-in annotation only, band/clean untouched",
                      passed, "-", "-", ""))

    # 8b) an AUTO_ADMIT sidecar DOES read clean (the invariant is not vacuous)
    sidecar_a = dict(segment_id="selftest_admit", agreement_cert=dict(verdict="abstain", reason="stub"))
    r_a = band_admission({"sfm": noisy(0.05), "flow": noisy(0.05)},
                         method_families={"sfm": "classical-geo", "flow": "learned-mono"},
                         family_error_corr=0.05, family_corr_validated=True, veto_legs=_covlegs)
    apply_band_to_sidecar(sidecar_a, r_a)
    passed = downstream_clean_view(sidecar_a)["clean"] is True
    ok = ok and passed
    cases.append(("invariant not vacuous: AUTO_ADMIT reads clean", passed, "-", "-", ""))

    # 9) complementary-AND sibling path -- independent of the
    # SYNC-path above; exercises the mechanism-diversity gate directly.
    def check_and(name, leg_verdicts, expect_band):
        nonlocal ok
        r = band_admission_complementary_and(leg_verdicts)
        passed = r["band"] == expect_band
        ok = ok and passed
        cases.append((f"AND: {name}", passed, r["band"], expect_band, r["band_reason"][:70]))
        return r

    # 9a) 2 legs PASS, leg-unique mechanisms (no overlap) -> AUTO_ADMIT
    check_and("2 legs, disjoint mechanisms", {
        "flow": {"verdict": "PASS", "mechanisms": {"magnitude-physicality", "spatial-coherence"}},
        "appearance": {"verdict": "PASS", "mechanisms": {"residual-stability"}},
    }, "AUTO_ADMIT")
    # 9b) 2 legs PASS, but BOTH PASSes carried ONLY by the shared order-gain mechanism -> QUARANTINE
    # (the exact C4 hazard: flow's continuity-only PASS + appearance's clip/depth-only PASS)
    r_shared = check_and("2 legs, BOTH pass via shared order-gain only", {
        "flow": {"verdict": "PASS", "mechanisms": {"order-gain"}},
        "appearance": {"verdict": "PASS", "mechanisms": {"order-gain"}},
    }, "QUARANTINE")
    if r_shared["mechanism_union"] != ["order-gain"]:
        ok = False
        cases.append(("AND: shared-mechanism union collapses to size 1", False, "-", "-",
                      f"got {r_shared['mechanism_union']}"))
    # 9c) 2 legs PASS, ONE leg's PASS carries order-gain PLUS a unique axis -> the unique axis
    # alone is enough to diversify the union -> AUTO_ADMIT
    check_and("2 legs, one leg contributes an extra unique axis", {
        "flow": {"verdict": "PASS", "mechanisms": {"order-gain", "magnitude-physicality"}},
        "appearance": {"verdict": "PASS", "mechanisms": {"order-gain"}},
    }, "AUTO_ADMIT")
    # 9d) only 1 leg PASS -> QUARANTINE (cannot rule with one leg, mirrors single_proxy band)
    check_and("1/2 legs pass", {
        "flow": {"verdict": "PASS", "mechanisms": {"magnitude-physicality", "spatial-coherence"}},
        "appearance": {"verdict": "FAIL", "mechanisms": []},
    }, "QUARANTINE")
    # 9e) 0 legs PASS -> REJECT
    check_and("0/2 legs pass", {
        "flow": {"verdict": "FAIL", "mechanisms": []},
        "appearance": {"verdict": "FAIL", "mechanisms": []},
    }, "REJECT")
    # 9f) apply_band_to_sidecar / downstream_clean_view work unmodified on this path's output too
    sidecar_and = dict(segment_id="selftest_and_admit", agreement_cert=dict(verdict="abstain", reason="stub"))
    r_and_admit = band_admission_complementary_and({
        "flow": {"verdict": "PASS", "mechanisms": {"magnitude-physicality", "spatial-coherence"}},
        "appearance": {"verdict": "PASS", "mechanisms": {"residual-stability"}},
    })
    apply_band_to_sidecar(sidecar_and, r_and_admit)
    passed = downstream_clean_view(sidecar_and)["clean"] is True
    ok = ok and passed
    cases.append(("AND: apply_band_to_sidecar/clean-view compatible (AUTO_ADMIT)", passed, "-", "-", ""))
    sidecar_and_q = dict(segment_id="selftest_and_q", agreement_cert=dict(verdict="admit", reason="stale"))
    apply_band_to_sidecar(sidecar_and_q, r_shared)
    passed = downstream_clean_view(sidecar_and_q)["clean"] is False
    ok = ok and passed
    cases.append(("AND: never-clean invariant holds on shared-mechanism QUARANTINE", passed, "-", "-", ""))

    print("admission_bands selftest:")
    for name, passed, got, want, note in cases:
        tag = "PASS" if passed else "FAIL"
        extra = f"-> {got} (want {want}) {note}" if want != "-" else ""
        print(f"  [{tag}] {name:45s} {extra}")
    print(f"  => {'PASS: all cases' if ok else 'FAIL: see above'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="three-band admission wrapper (AUTO_ADMIT/QUARANTINE/REJECT)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run-real", action="store_true",
                    help="band the real sidecars in data/video_segments/ (writes flags in place)")
    ap.add_argument("--no-write", action="store_true", help="with --run-real: dry pass, no sidecar writes")
    ap.add_argument("--governor", action="store_true",
                    help="opt-in: annotate governor_defer per resource_governor's "
                         "admission_video floor (data/GOVERNOR_STATE.json); band/clean unchanged")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.run_real:
        gov = load_governor_admission() if args.governor else None
        if args.governor:
            print(f"governor: {gov if gov else 'ingen GOVERNOR_STATE.json — defer av'}")
        result = run_real(write=not args.no_write, governor=gov)
        print(json.dumps(dict(n_segments=result["n_segments"],
                              band_distribution=result["band_distribution"],
                              written=result["written"]), indent=2))
        for r in result["rows"]:
            print(f"  {r['segment_id']}  T={r['T_frames']:4d}  {r['band']:10s} {r['band_reason'][:60]}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
