#!/usr/bin/env python3
"""cert_grade_engine.py -- grade a delivered claim in milliseconds.

Most cert claims are MECHANICALLY decidable (a number, a threshold-provenance tag, a lineage pair,
a scope match) and never needed a language-model read in the first place. This module is the fast,
pure-function verdict layer: it decides what CAN be decided by machine and honestly ESCALATEs the
residue a human or model must still read. It never silently passes an undecidable claim, and a flag
always propagates to the aggregate.

It consumes existing tools rather than re-implementing them:
  * C4 calls `leg_decorrelation_lineage_gate` (layer-1 structural pair verdict, and the
    2-layer `compose_decorrelation_verdict` when empirical evidence is present).
  * The threshold-provenance lesson (C3) is the  tautology-gate lesson: a gate
    whose anchor_class is its own self-quantile is not evidence of anything (it always
    "passes" by construction) -> automatic FLAG, never silently trusted.

======================================================================================
CLAIM CONTRACT (minimal, documented here since no docs/CLAIM_CONTRACT.md existed at
build time -- if a sibling agent later publishes one, reconcile field names against it):

claim = {
  "id":                str, optional, for logging only,

  "decisive_number":   float, the number the claim asserts is the outcome,
  "decisive_number_path": str, dotted key-path into the artifact JSON where the
                         REAL number lives, e.g. "metrics.auc" or "0.rho" (int index
                         for list levels). Required for C2 to recompute-and-compare.
  "tolerance":         float, absolute tolerance for the C2 compare (default 1e-6).

  "reproducer": {                      # C1 -- optional; omitted => C1 escalates
      "cmd":          list[str] or str, subprocess argv/shell command,
      "timeout_s":    float, default 30,
      "expect_regex": str, optional, stdout must match,
      "expect_json_path": str, optional, dotted path into stdout-parsed-as-JSON,
      "expect_json_value": any, required if expect_json_path given,
  },

  "gate_def": {
      "anchor_class": str, one of:
          "self-quantile"        -> C3 automatic FLAG (tautological; lesson)
          "known-fake-anchored"  -> C3 PASS (a deliberately-labeled synthetic anchor,
                                     acceptable for calibration claims)
          "external-reference"   -> C3 PASS (anchored to data outside the claim's
                                     own construction)
          anything else / missing -> C3 ESCALATE (undetermined provenance)
  },

  "legs": [ {"lineage_id": str, "empirical_verdict": str optional,
             "empirical_neff": float optional,
             "empirical_min_stratum_n": int optional},... ],
          # C4 -- pairwise decorrelation over every leg pair. 0 or 1 legs => trivial PASS.

  "claimed_numbers": [float,...],     # C5 -- every number narrated in the claim;
                                          each must be FOUND (within tolerance) somewhere
                                          in the artifact's flattened numeric values.

  "claimed_scope": {"n": int, "substrate": str},   # C6
  "artifacts": {"path": str},           # a JSON file on disk holding the real numbers/
                                          scope this claim is graded against. Required
                                          for C2/C5/C6; its own scope fields are read
                                          from artifact["scope"]["n"] / ["substrate"]
                                          (or top-level "n"/"substrate" as a fallback).
}

VERDICT:
  PASS-MACHINE   -- every clause resolved PASS.
  FAIL-MACHINE   -- at least one clause hard-FAILed (deterministically wrong/dishonest).
  ESCALATE-LLM   -- no hard FAIL, but >=1 clause could not be mechanically decided;
                    `escalate_reason` names the precise open question (frontier residue).
  FAIL beats ESCALATE beats PASS in priority (a determined problem must surface even
  if another clause is merely undecided -- never let an escalate mask a fail).

USAGE:
  python3 scripts/cert_grade_engine.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any

_LINEAGE_GATE_IMPORT_ERROR = None
try:
    from . import leg_decorrelation_lineage_gate as _lineage_gate
except ImportError:
    try:
        import leg_decorrelation_lineage_gate as _lineage_gate
    except Exception as e:  # pragma: no cover - environment-dependent
        _lineage_gate = None
        _LINEAGE_GATE_IMPORT_ERROR = str(e)

# Optional extra clause plugins (C7-). Point $CERT_GRADE_CLAUSE_DIR at a directory of modules
# exposing the function names below; absent, the clauses are skipped with a notice and the engine
# still runs (graceful degradation, never a crash on a missing neighbour).
_CLAUSE_DIR = os.environ.get("CERT_GRADE_CLAUSE_DIR", "")
_B_CLAUSES = []
if _CLAUSE_DIR and os.path.isdir(_CLAUSE_DIR):
    if _CLAUSE_DIR not in sys.path:
        sys.path.insert(0, _CLAUSE_DIR)
    for _mod, _fn in [
        ("rider_clause_C7_decisive_number_verified_vs_reproducer_output", "clause_c7_decisive_number_verified"),
        ("rider_clause_C8_ci_lower_not_point_clears_the_bar", "clause_c8_ci_lower_not_point"),
        ("rider_clauses_C9_C11_adversarial_minimum_and_registry_verified_kind", "clause_c9_adversarial_minimum"),
        ("rider_clause_C10_selection_inflation_beat_the_null_max_over_K", "clause_c10_selection_inflation"),
        ("rider_clauses_C9_C11_adversarial_minimum_and_registry_verified_kind", "clause_c11_registry_verified_kind"),
    ]:
        try:
            _m = __import__(_mod)
            _B_CLAUSES.append(getattr(_m, _fn))
        except Exception as _e:
            _B_CLAUSES.append(("unavailable", _fn, str(_e)[:60]))

PASS, FAIL, ESCALATE = "PASS", "FAIL", "ESCALATE"

ANCHOR_TAUTOLOGICAL = {"self-quantile", "self_quantile"}
ANCHOR_ACCEPTABLE = {"known-fake-anchored", "external-reference", "derived-floor"}
# C3 REFERENT: these two classes require an ACTUAL anchor (path or dataset token) in
# gate_def.anchor_ref -- without it "external-reference" is just a trusted STRING, a tautology in
# disguise. "derived-floor" does NOT require anchor_ref (an analytically derived floor has no
# external dataset to point at).
ANCHOR_REQUIRES_REF = {"known-fake-anchored", "external-reference"}

# ALIAS: short-form anchor_class tokens seen in the live claim corpus (36-38 of 88+ claims). Alias-normalization ONLY -- these map onto the SAME two long-form classes
# already in ANCHOR_REQUIRES_REF, subject to the identical anchor_ref/self-reference/bare-label
# checks (see FP7 fix below). Never add a class here without also covering it in ANCHOR_REQUIRES_REF
# or ANCHOR_ACCEPTABLE, and never land this alone (bundle invariant, see clause_c3 docstring note).
ANCHOR_CLASS_ALIASES = {"ext-ref": "external-reference", "known-fake": "known-fake-anchored"}

# VACUOUS-TOLERANCE GUARD (sibling of the isfinite guard): a FINITE but OVERSIZED tolerance means
# C2/C5 can never falsify the claim -- the claimant picks the band that defines "matches". Rule
# (derived from the real claim population):
# bound = NULL_BAND_FLOOR if |claimed/real| ~ 0 (near-zero exception:
# legitimate null/calibration claims with
# decisive_number ~ 0; bands up to 2.0 were
# observed in the corpus by design)
# bound = max(REL_CAP * |claimed/real|, ABS_FLOOR) otherwise
# tolerance > bound => ESCALATE (never a silent PASS; ESCALATE rather than FAIL because vacuity
# makes the verdict UNDEFINED, not necessarily WRONG).
_VACUOUS_REL_CAP = 0.5          # tolerance may be at most half the magnitude of the number
_VACUOUS_ABS_FLOOR = 0.3        # floor for bounded measures (correlation [-1,1], p-value [0,1],
 # percentages) where "relative to the number" misses plausibility
_VACUOUS_NULL_BAND_FLOOR = 2.0  # floor for a near-zero decisive_number (legitimate wide-band
                                 # null-hypothesis claims; max observed in the corpus = 2.0)
_VACUOUS_NEAR_ZERO_EPS = 1e-9

# ★VERDICT/VALUE-BLIND (FP8, C→red-team): reversible word-bounded flip of the
# boolean/verdict tokens an honest reproducer prints. A single atomic sub swaps each pair in
# BOTH directions (dict maps a->b AND b->a), so no placeholder round-trip is needed.
_VERDICT_FLIP = {
    "True": "False", "False": "True", "TRUE": "FALSE", "FALSE": "TRUE",
    "true": "false", "false": "true",
    "PASS": "FAIL", "FAIL": "PASS", "Pass": "Fail", "Fail": "Pass",
    "pass": "fail", "fail": "pass", "passed": "failed", "failed": "passed",
    "PASSED": "FAILED", "FAILED": "PASSED", "yes": "no", "no": "yes",
    "YES": "NO", "NO": "YES",
}
_VERDICT_FLIP_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(t) for t in _VERDICT_FLIP), key=len, reverse=True)) + r")\b")


def _hostile_rewrite(text: str, dn_value) -> str:
    """Build a HOSTILE copy of reproducer stdout for the C1 verdict/value-blindness probe:
    (1) replace the claimed decisive_number with a far decoy (common float formats), and
    (2) flip verdict tokens (True<->False, PASS<->FAIL,...). A C1 expect_regex that STILL
    matches this copy binds to NEITHER the claimed value NOR the gate verdict -- it is printed
    unconditionally and would PASS a failing-gate reproducer just as readily (FP2/FP8 class).
    Returns the (possibly-unchanged) rewritten text; caller only escalates when it differs."""
    out = text
    if isinstance(dn_value, bool):
        pass  # bool is an int subclass but has no meaningful numeric decoy; verdict flip covers it
    elif isinstance(dn_value, (int, float)):
        decoy = dn_value + 84213.991827 + abs(dn_value) * 7.3
        decoy_s = f"{decoy:.6f}"
        forms = {repr(dn_value), str(dn_value), f"{float(dn_value)}",
                 f"{float(dn_value):.6f}", f"{float(dn_value):g}"}
        for f in sorted((f for f in forms if f), key=len, reverse=True):
            if f in out:
                out = out.replace(f, decoy_s)
    out = _VERDICT_FLIP_RE.sub(lambda m: _VERDICT_FLIP[m.group(0)], out)
    return out


def _vacuous_tolerance_bound(scale: float) -> float:
    """Given the magnitude scale (max(|claimed|, |real|) or similar), return the largest
    tolerance that is still considered to have discriminating power. tolerance > bound is a
    vacuous band -- ESCALATE, never a silent PASS."""
    scale = abs(scale)
    if scale < _VACUOUS_NEAR_ZERO_EPS:
        return _VACUOUS_NULL_BAND_FLOOR
    return max(_VACUOUS_REL_CAP * scale, _VACUOUS_ABS_FLOOR)


# --------------------------------------------------------------------------------- #
# small pure helpers
# --------------------------------------------------------------------------------- #

def _dig(obj: Any, dotted_path: str):
    """Walk a dotted path (dict keys or list indices) into a nested structure.
    Raises KeyError/IndexError/TypeError on miss -- callers catch and ESCALATE/FAIL.

    ★AMBIGUITY-FIX (adversarial bughunt, false-FAIL #1, confirmed live in
    I-CRYSTAL-GROWTH-PURIFICATION-V0.json path='2_falsification.PF_Boron_fs0.5' and
    I-HAIKU-TIER-VERIFY-AB-V0.json): a dotted_path segment can itself contain a literal '.'
    (a decimal embedded in an identifier, e.g. a fraction-of-solid key 'PF_Boron_fs0.5').
    Naive split('.') treats EVERY dot as a path separator and false-FAILs any claim whose
    real key contains one (KeyError on a truncated fragment like 'PF_Boron_fs0'). Fix: at
    each dict level, try the LONGEST remaining run of parts joined by '.' that is an
    ACTUAL key of the current dict first, shrinking one part at a time; only fall back to
    single-part traversal (original behavior, identical KeyError semantics) when no
    literal-joined key matches. This is a greedy prefer-what-actually-exists rule: it
    cannot resolve a maliciously-ambiguous dict that has BOTH a literal 'a.b' key AND a
    nested a->b route to a DIFFERENT value (no such case observed in the claims corpus,
    and escaping the separator would require a claim-format change), but it is strictly
    conservative for every case that worked before -- a path with no dots-inside-keys has
    no shorter literal match to prefer, so the longest match IS still the single part and
    resolution is byte-identical to the old code."""
    cur = obj
    parts = dotted_path.split(".")
    n = len(parts)
    i = 0
    while i < n:
        if isinstance(cur, list):
            cur = cur[int(parts[i])]
            i += 1
            continue
        if isinstance(cur, dict):
            matched = False
            for j in range(n, i, -1):
                candidate = ".".join(parts[i:j])
                if candidate in cur:
                    cur = cur[candidate]
                    i = j
                    matched = True
                    break
            if matched:
                continue
            cur = cur[parts[i]]  # no literal-joined key -- fall back, same error as before
            i += 1
            continue
        cur = cur[parts[i]]  # neither list nor dict -- preserve original TypeError behavior
        i += 1
    return cur


def _load_artifact(path: str):
    with open(path) as f:
        return json.load(f)


# ★PATH-FILE-PREFIX RESOLUTION (sibling-pattern normalization, FF5 + generalizes the
# existing colon-form fix): the live corpus embeds a FILE-then-DEEP-PATH convention in front of a
# dotted key path three different ways -- '::' (double colon, the majority convention), '#'
# (URL-fragment style, FF5: I-TENSION-RESOLUTIONS-V0.json 'file.json#summary.n_resolved_fully'),
# and ':' (single colon, oldest/defensive form). This is NOT full RFC 6901 JSON Pointer (consulted
# as the canonical path-escaping reference for this fix): RFC 6901 pointers are '/'-separated with
# '~0'/'~1' escapes for '~'/'/', but the observed live convention keeps the SAME dot-separated key
# grammar C2/_dig already use on the right of the separator -- only the FILE-vs-KEY boundary is
# URL-fragment-flavored, so the correct fix is a 3rd/4th separator token, not a rewrite to slash
# pointers (would break every existing dotted-path claim in the corpus). Shared by C1 and C2 so
# both clauses recognize all three conventions identically (previously C1 had '::'/':' and C2 had
# only ':' -- neither had '#').
_PATH_FILE_SEPARATORS = ("::", "#", ":")


def _resolve_file_prefixed_path(path_str: str):
    """If `path_str` embeds a FILE<sep>deep.dotted.key prefix (sep in '::', '#', ':', tried in
    that order -- longest/most-specific convention first) and the left-of-separator part names a
    REAL FILE ON DISK (checked at call time, so a freshly-emitted artifact is visible), return
    (loaded_json, right_of_separator_key). Returns (None, path_str) unchanged when no separator
    resolves to a real file, so callers fall through to their original behavior -- an unrecognized
    separator-bearing path fails exactly as before, never silently passes."""
    for sep in _PATH_FILE_SEPARATORS:
        if sep in path_str:
            fpart, _, kpart = path_str.partition(sep)
            if kpart and os.path.isfile(fpart):
                try:
                    with open(fpart) as fh:
                        return json.load(fh), kpart
                except Exception:
                    pass  # fall through -- try the next separator / original behavior
    return None, path_str


def _coerce_float_or_none(x):
    try:
        if isinstance(x, bool):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _normalize_decisive_number(claim: dict):
    """Canonicalize decisive_number across every live schema variant found in the
    sibling-pattern sweep:
      1. documented bare-float form: decisive_number=float, top-level decisive_number_path,
         top-level tolerance (unchanged -- this is the byte-identical default path).
      2. dict-form (FF4 + real-corpus siblings, 9 live claims): decisive_number={'value': float,
         'decisive_number_path'|'name'|'path': <path-string>, 'tolerance': <float-or-prose>} --
         the path-key name varies live (decisive_number_path/name/path all observed); the dict's
         OWN tolerance wins over the claim's top-level tolerance when present and numeric.
      3. tolerance field-name alias (FF6, 2 live claims): top-level 'decisive_number_tolerance'
         instead of 'tolerance' -- trusted ONLY when it parses as a float; both live instances are
         PROSE ("abs(a-b)<0.01 (agree to 4sf)", gate-requirement text), so silently defaulting to
         1e-6 is exactly the false-FAIL FF6 reports -- returns tol_escalate instead of guessing.
    Returns (value: float|None, path: str|None, tolerance: float, tol_escalate: str|None).
    tol_escalate is None normally; when set, the caller ESCALATEs on that exact text rather than
    using `tolerance` (which is still populated with the 1e-6 placeholder default so downstream
    arithmetic never crashes on None, but callers must check tol_escalate FIRST)."""
    dn = claim.get("decisive_number")
    tol_escalate = None

    if isinstance(dn, dict):
        value = _coerce_float_or_none(dn.get("value"))
        path = dn.get("decisive_number_path") or dn.get("name") or dn.get("path")
        dict_tol = _coerce_float_or_none(dn.get("tolerance"))
        if dict_tol is not None:
            return value, path, dict_tol, None
        if "tolerance" in dn:
            tol_escalate = (f"decisive_number.tolerance={dn['tolerance']!r} is not numeric -- "
                             "cannot mechanically resolve a tolerance band (FF6-class gap, one "
                             "level down inside the dict-form decisive_number)")
    else:
        value = _coerce_float_or_none(dn) if dn is not None else None
        path = claim.get("decisive_number_path")

    tol = _coerce_float_or_none(claim.get("tolerance"))
    if tol is None and "decisive_number_tolerance" in claim:
        # ★FF6-ALIAS: field-name variant 'decisive_number_tolerance' instead of the
        # contract's 'tolerance' -- trusted ONLY when numeric. Both live corpus instances
        # (I-LEADFREE-TRIPLET-FIELD-V1, I-GRASP-IMPEDANCE-V2) are prose, not a number; C2 used to
        # silently fall back to the hardcoded default (1e-6), producing a false-FAIL on a value
        # the claim explicitly documents agreeing to 4 significant figures. ESCALATE instead of
        # guessing OR false-FAILing on an over-tight default.
        alias_tol = _coerce_float_or_none(claim.get("decisive_number_tolerance"))
        if alias_tol is not None:
            tol = alias_tol
        elif tol_escalate is None:
            tol_escalate = (f"decisive_number_tolerance={claim['decisive_number_tolerance']!r} is "
                             "not numeric -- cannot mechanically resolve a tolerance band (FF6 "
                             "field-name alias present but non-numeric; refusing to silently fall "
                             "back to the default 1e-6, which would false-FAIL a claim whose real "
                             "intended tolerance is wider)")
    if tol is None:
        tol = 1e-6  # unchanged default -- identical to the pre-fix float(claim.get('tolerance', 1e-6))

    return value, path, tol, tol_escalate


def _flatten_numbers(obj: Any, out: list[float] | None = None) -> list[float]:
    """Collect every numeric leaf value in a nested dict/list structure."""
    if out is None:
        out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numbers(v, out)
    return out


def _artifact_scope(artifact: dict) -> dict:
    scope = artifact.get("scope") if isinstance(artifact.get("scope"), dict) else {}
    n = scope.get("n", artifact.get("n"))
    substrate = scope.get("substrate", artifact.get("substrate"))
    return {"n": n, "substrate": substrate}


def _claim_artifact_paths(claim: dict) -> list[str]:
    """Every on-disk path the claim itself names as ITS OWN artifact (normalizes the
    artifacts: [str,...] / {"path":...} / [{"path":...}] forms grade already tolerates)."""
    art = claim.get("artifacts")
    paths: list[str] = []
    if isinstance(art, dict):
        p = art.get("path")
        if isinstance(p, str) and p:
            paths.append(p)
    elif isinstance(art, list):
        for item in art:
            if isinstance(item, str) and item:
                paths.append(item)
            elif isinstance(item, dict) and isinstance(item.get("path"), str) and item["path"]:
                paths.append(item["path"])
    return paths


def _looks_like_path(s: str) -> bool:
    return "/" in s or s.lower().endswith((".json", ".jsonl", ".csv", ".txt", ".md"))


# Trusted artifact root: artifacts must live under $CERT_ARTIFACT_ROOT (default: the current
# working directory) and inside a reports/ tree.
_TRUSTED_ARTIFACT_ROOT = os.environ.get("CERT_ARTIFACT_ROOT", os.getcwd())
_DECORR_EVIDENCE_KEY_TOKENS = ("n_eff", "neff", "family_error_corr", "decorrelat", "worst_pair_corr")


def _artifact_path_trusted(p) -> bool:
    """Trusted-root containment: the path must live under $CERT_ARTIFACT_ROOT and in a reports/ tree. Shared
    by C4's leg-evidence rule (below) and grade's own artifacts.path loading (FP5 fix,
: previously C2/C5/C6 trusted ANY on-disk artifacts.path unconditionally, asymmetric
    with this exact rule C4 already applies to leg evidence)."""
    pp = str(p).replace("\\", "/")
    root = str(_TRUSTED_ARTIFACT_ROOT).replace("\\", "/").rstrip("/")
    return pp.startswith(root + "/") and "/reports/" in pp


def _has_numeric_decorrelation_evidence(obj, _depth: int = 0) -> bool:
    """★FP3-FIX (keyword-substring evidence): the old check was
    `any(k in json.dumps(d).lower for k in (...))` -- a bare SUBSTRING match against the WHOLE
    serialized text, satisfied by prose mentioning the word with zero actual measurement (e.g.
    {'note': 'n_eff was measured and everything looked decorrelated, trust me'} has no real
    number anywhere, confirmed false-PASS via false_pass_qc_v0 FP3). Require instead that at
    least one of the known decorrelation-evidence KEYS resolves to an ACTUAL FINITE NUMBER
    somewhere in the structure (any nesting depth) -- a real key with a real value, not a word
    appearing inside an unrelated string."""
    if _depth > 8:
        return False
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if any(tok in kl for tok in _DECORR_EVIDENCE_KEY_TOKENS):
                fv = _coerce_float_or_none(v)
                if fv is not None:
                    import math
                    if math.isfinite(fv):
                        return True
            if _has_numeric_decorrelation_evidence(v, _depth + 1):
                return True
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            if _has_numeric_decorrelation_evidence(item, _depth + 1):
                return True
    return False


def _empirical_evidence_ok(leg: dict) -> bool:
    """A self-declared leg empirical_verdict is only trustworthy if it points at a PARSEABLE
    on-disk decorrelation-evidence artifact under a trusted lane reports/ path -- 'file exists'
    is not enough (/etc/hostname passed an earlier, weaker version of this check). Shared,
    module-level so EVERY C4 branch with a self-declared empirical_verdict (not just the
    classical-formula same-lineage-id branch) is held to the same bar."""
    p = str(leg.get("empirical_evidence_path", ""))
    if not p:
        return False
    if not (os.path.isfile(p) and _artifact_path_trusted(p)):
        return False
    try:
        d = json.load(open(p))
    except Exception:
        return False
    return _has_numeric_decorrelation_evidence(d)


# --------------------------------------------------------------------------------- #
# C1 -- reproducer (seconds-scale; timed SEPARATELY from the verdict-logic budget)
# --------------------------------------------------------------------------------- #

def _normalize_reproducer_aliases(rep: dict) -> dict:
    # ALIAS NORMALIZATION: claim authors write expected_output_regex / expected_regex instead of
    # expect_regex, so the engine normalizes mechanically. Canonical fields win when both exist.
    if not isinstance(rep, dict):
        return rep
    for canon, aliases in (("expect_regex", ("expected_output_regex", "expected_regex", "expect_stdout_regex")),
                           ("expect_json_path", ("expected_json_path",)),
                           ("expect_json_value", ("expected_json_value",)),
                           ("expected_exit", ("expect_exit", "expected_returncode"))):
        if canon not in rep:
            for a in aliases:
                if a in rep:
                    rep[canon] = rep[a]
                    break
    return rep


def clause_c1_reproducer(claim: dict) -> dict:
    rep = claim.get("reproducer")
    rep = _normalize_reproducer_aliases(rep)
    if not rep or not rep.get("cmd"):
        return {"clause": "C1_reproducer", "result": ESCALATE,
                "evidence": "no reproducer.cmd given -- cannot mechanically re-run; "
                            "an LLM (or operator) must judge the claim without a live repro"}
    # C1 BINDING (path resolution also covers dict-form decisive_number via
    # _normalize_decisive_number): decisive_number_path means the claim stands or falls on a number
    # C2 will pull out of the artifact -- the reproducer MUST then be bound to that number
    # (expect_regex or expect_json_path), otherwise "exit=0" only proves the script RAN, not that it
    # COMPUTED THE RIGHT THING (a smoke test, not a substance test). ESCALATE, never a silent PASS.
    _dn_value_c1, _dn_path_c1, _dn_tol_c1, _ = _normalize_decisive_number(claim)
    if _dn_path_c1 and not (rep.get("expect_regex") or rep.get("expect_json_path")):
        return {"clause": "C1_reproducer", "result": ESCALATE,
                "evidence": "decisive_number_path is set but the reproducer has NEITHER expect_regex "
                            "NOR expect_json_path -- the reproducer is unbound to the claim "
                            "(did not crash != computed correctly)"}
    # ★EXPECT_JSON_VALUE-DEFAULT: the live corpus (5 claims:
    # I-FRAME0024-GHOST-HUNT-V0 + 4 others, confirmed via /tmp/claims/I-*.json + direct
    # verify_shadow_compare disagreement measurement) sets expect_json_path but NEVER sets
    # expect_json_value at all -- relying on decisive_number(.value) itself as the implied
    # expectation, the same number C2/C7 already check. Without this default, `got !=
    # rep.get('expect_json_value')` compares the REAL extracted number against None and ALWAYS
    # FAILs regardless of correctness -- a deterministic false-FAIL for every claim using this
    # (widespread, legitimate) convention. Safe direction: this only ever ADDS a comparison target
    # where none existed before (a bare expect_json_path with no value at all could never
    # discriminate anything either way); a genuine mismatch still correctly FAILs below.
    if rep.get("expect_json_path") and "expect_json_value" not in rep and _dn_value_c1 is not None:
        rep = {**rep, "expect_json_value": _dn_value_c1}
    # VACUOUS-REGEX GUARD (same defect class as the vacuous-tolerance hole): an expect_regex that
    # matches EVERYTHING (e.g. '.*', '.*?', '' or other trivially universal patterns) satisfies the
    # LETTER of C1 binding but not its SPIRIT -- it binds the reproducer to NOTHING specific. Three
    # ORTHOGONAL empirical probes, ESCALATE if ANY matches:
    # (a) letters-only nonce: the pattern matches SOMETHING that cannot occur in a real answer
    # -> vacuous in the "matches everything" direction.
    # (b) empty-string probe (catches expect_regex='^$' against intentionally empty stdout): if
    # the pattern matches TOTAL SILENCE it proves nothing about ANY number -- more extreme
    # than the nonce, catches patterns that only require absence of content.
    # (c) numeric-decoy probe (NEW, catches FP2: expect_regex=r'\d+\.\d+' matching ANY float):
    # when a specific decisive_number is claimed, the pattern must NOT also match a
    # DIFFERENT number formatted the same way -- probed with a decoy value far from the
    # claimed one, never assuming any particular surrounding-text FORMAT (no false-FAIL
    # risk on legitimately format-specific real-corpus regexes, which never accidentally
    # match an unrelated synthetic decoy string).
    _er = rep.get("expect_regex")
    if _er:
        _nonce = "ZQXJVKPNONCEPROBE_NOT_REAL_REPRODUCER_OUTPUT_WKPZQXJV"
        try:
            _matches_nonce = bool(re.search(_er, _nonce))
        except re.error:
            _matches_nonce = False  # invalid regex -- caught later when actually applied (FAIL)
        try:
            _matches_empty = bool(re.search(_er, ""))
        except re.error:
            _matches_empty = False
        if _matches_nonce or _matches_empty:
            return {"clause": "C1_reproducer", "result": ESCALATE,
                    "evidence": f"vacuous expect_regex={_er!r}: matches "
                                f"{'total silence (the empty string)' if _matches_empty else 'a random nonce'} "
                                "and so can prove nothing about the claimed number -- the pattern has "
                                "no discriminating power (same one-sided-guard class as the vacuous "
                                "tolerance hole: C1 binding is met to the letter, not to the spirit)"}
        if isinstance(_dn_value_c1, (int, float)):
            _decoy = _dn_value_c1 + 84213.991827 + abs(_dn_value_c1) * 7.3
            _decoy_s = f"computed value: {_decoy}"
            try:
                _matches_decoy = bool(re.search(_er, _decoy_s))
            except re.error:
                _matches_decoy = False
            if _matches_decoy:
                return {"clause": "C1_reproducer", "result": ESCALATE,
                        "evidence": f"vacuous expect_regex={_er!r}: matches {_decoy_s!r} -- a NUMBER "
                                    f"THAT IS NOT the claimed decisive_number={_dn_value_c1!r} -- "
                                    "the pattern does not bind specifically to the claimed number "
                                    "(digit-matching but content-free)"}
    cmd = rep["cmd"]
    timeout_s = rep.get("timeout_s", 30)
    try:
        proc = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                               text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"clause": "C1_reproducer", "result": FAIL,
                "evidence": f"reproducer timed out after {timeout_s}s"}
    except Exception as e:
        return {"clause": "C1_reproducer", "result": FAIL,
                "evidence": f"reproducer raised {type(e).__name__}: {e}"}
    if proc.returncode != 0:
        _err = (proc.stderr or "") + (proc.stdout or "")
        # ENV-FAIL != substance-FAIL: a reproducer that does not even START in the grader's
        # environment proves nothing about the claim -> ESCALATE with the contract rule, never FAIL.
        if any(k in _err for k in ("ModuleNotFoundError", "Error while finding module",
                                   "No module named", "command not found", "No such file or directory")):
            return {"clause": "C1_reproducer", "result": ESCALATE,
                    "evidence": f"ENV-FAIL (not substance): the reproducer did not start in the grader's "
                                f"environment (exit={proc.returncode}) -- contract rule: cmd must be "
                                f"environment-self-contained (absolute interpreter + PYTHONPATH); {_err[-120:]!r}"}
        return {"clause": "C1_reproducer", "result": FAIL,
                "evidence": f"exit={proc.returncode} stderr={proc.stderr[:300]!r}"}
    if rep.get("expect_regex") and not re.search(rep["expect_regex"], proc.stdout):
        return {"clause": "C1_reproducer", "result": FAIL,
                "evidence": f"stdout did not match expect_regex={rep['expect_regex']!r} "
                            f"stdout={proc.stdout[:300]!r}"}
    # ★VERDICT/VALUE-BLIND-GUARD (FP8, C→red-team): the expect_regex matched the
    # REAL stdout above. But C1's job is to bind the reproducer to the CLAIMED result. The three
    # PRE-execution probes (nonce/empty/decoy) catch UNIVERSAL and any-number regexes but MISS a
    # specific token printed UNCONDITIONALLY ('machine_claim_ok:' without True, 'measured_value:')
    # -- it survives all three yet matches whether the gate PASSED or FAILED (4/4 false-PASS on a
    # failing-gate probe, measured by C). Post-execution probe: rebuild stdout HOSTILE (claimed
    # number -> decoy AND verdict tokens flipped); if the regex STILL matches, it binds to neither
    # the value nor the verdict -> ESCALATE. Only fires when the rewrite actually changed something
    # (a value/verdict was present to break) -> a genuine number- or verdict-binding regex stops
    # matching and correctly PASSES; regression-safe (subsumes the pre-exec decoy probe).
    if rep.get("expect_regex"):
        _hostile = _hostile_rewrite(proc.stdout, _dn_value_c1)
        if _hostile != proc.stdout:
            try:
                _still_matches = bool(re.search(rep["expect_regex"], _hostile))
            except re.error:
                _still_matches = False
            if _still_matches:
                return {"clause": "C1_reproducer", "result": ESCALATE,
                        "evidence": f"verdict/value-blind expect_regex={rep['expect_regex']!r}: it ALSO "
                                    "matches a hostile rewrite of stdout (claimed decisive_number -> "
                                    "decoy AND verdict tokens flipped True<->False/PASS<->FAIL) -- "
                                    "the pattern binds neither to the claimed number nor to the gate "
                                    "verdict; it is printed unconditionally and would PASS a FAILING-gate "
                                    "reproducer just as easily"}
    if rep.get("expect_json_path"):
        _ejp = rep["expect_json_path"]
        # ★COLON/HASH-FORM-NORMALISERING (C1, bughunt fix false-FAIL #2 + FF5 sibling
        # generalization): live corpus usage writes expect_json_path as "file.json::a.b.c"
        # (double colon, majority convention) or "file.json#a.b.c" (URL-fragment style, FF5).
        # Mirrors C2's convention (checked AFTER the reproducer ran, so a freshly-written
        # artifact is visible) via the shared _resolve_file_prefixed_path helper -- falls through
        # unchanged to stdout-parsing when no separator resolves to a real file (safe: an
        # unrecognized separator-bearing path just fails exactly as before).
        _loaded_ejp, _ejp = _resolve_file_prefixed_path(_ejp)
        try:
            if _loaded_ejp is not None:
                parsed = _loaded_ejp
            else:
                parsed = json.loads(proc.stdout)
            got = _dig(parsed, _ejp)
        except Exception as e:
            return {"clause": "C1_reproducer", "result": FAIL,
                    "evidence": f"could not extract expect_json_path from stdout: {e}"}
        if got != rep.get("expect_json_value"):
            return {"clause": "C1_reproducer", "result": FAIL,
                    "evidence": f"expect_json_path={rep['expect_json_path']!r} got={got!r} "
                                f"want={rep.get('expect_json_value')!r}"}
    return {"clause": "C1_reproducer", "result": PASS,
            "evidence": f"exit=0, {len(proc.stdout)} bytes stdout, all expectations met"}


# --------------------------------------------------------------------------------- #
# C2 -- decisive-number recompute/extract vs claim, within tolerance
# --------------------------------------------------------------------------------- #

def clause_c2_decisive_number(claim: dict, artifact) -> dict:
    if "decisive_number" not in claim:
        return {"clause": "C2_decisive_number", "result": ESCALATE,
                "evidence": "claim has no decisive_number to check"}
    # ★SCHEMA-NORMALISERING: decisive_number can be a
    # bare float (documented contract, top-level decisive_number_path/tolerance) OR a dict
    # carrying its own path (key name varies live: decisive_number_path/name/path) and its own
    # tolerance (numeric, or in 2 live cases prose -- never silently substituted, see
    # _normalize_decisive_number's docstring).
    claimed, path, tol, tol_escalate = _normalize_decisive_number(claim)
    if claimed is None:
        return {"clause": "C2_decisive_number", "result": ESCALATE,
                "evidence": f"decisive_number present but its value could not be parsed as a "
                            f"float ({claim.get('decisive_number')!r})"}
    if not path:
        return {"clause": "C2_decisive_number", "result": ESCALATE,
                "evidence": "no decisive_number_path -- cannot locate the number in the "
                            "artifact mechanically"}
    # ★KOLON/HASH-FORM-NORMALISERING (D curation "recurring" + FF5 sibling
    # generalization): live convention embeds a FILE<sep>deep.path prefix with ':',
    # '::', or '#' (I-TENSION-RESOLUTIONS-V0.json) -- shared helper with C1, tries all three.
    _loaded, _kpart = _resolve_file_prefixed_path(path)
    if _loaded is not None:
        artifact, path = _loaded, _kpart
    if artifact is None:
        return {"clause": "C2_decisive_number", "result": ESCALATE,
                "evidence": "no artifact loaded -- cannot recompute/extract the number"}
    try:
        real = float(_dig(artifact, path))
    except Exception as e:
        return {"clause": "C2_decisive_number", "result": FAIL,
                "evidence": f"decisive_number_path={path!r} not found in artifact ({e}) "
                            "-- narrated-but-absent figure"}
    import math  # NaN on EITHER side made abs(NaN)>tol False = fail-OPEN;
    if not (math.isfinite(real) and math.isfinite(claimed) and math.isfinite(tol)):
        return {"clause": "C2_decisive_number", "result": FAIL,
                "evidence": f"non-finite value (claimed={claimed}, artifact={real}, tol={tol}) "
                            "-- NaN/inf is never a verified figure (isfinite guard on both sides)"}
    if tol_escalate:
        # ★FF6-FIX: tolerance couldn't be mechanically resolved (non-numeric alias field) --
        # ESCALATE for a human read rather than silently using the 1e-6 placeholder default,
        # which previously produced a false-FAIL on claims with a genuinely wider intended band.
        return {"clause": "C2_decisive_number", "result": ESCALATE, "evidence": tol_escalate}
    # VACUOUS-TOLERANCE GUARD: a FINITE tolerance that is large relative to the magnitude of the
    # number makes the comparison NON-DISCRIMINATING -- the claimant can assert a decisive_number
    # that their OWN artifact contradicts simply by inflating the tolerance. ESCALATE (never a
    # silent PASS) BEFORE the comparison, whichever way it would fall: a non-discriminating gate is
    # unreliable in BOTH directions.
    scale = max(abs(real), abs(claimed))
    bound = _vacuous_tolerance_bound(scale)
    if tol > bound:
        return {"clause": "C2_decisive_number", "result": ESCALATE,
                "evidence": f"vacuous tolerance: the claim cannot be falsified by its own gate -- "
                            f"tol={tol:g} > bound={bound:g} (rel_cap={_VACUOUS_REL_CAP}x scale={scale:g}, "
                            f"abs_floor/null_band={_VACUOUS_ABS_FLOOR}/{_VACUOUS_NULL_BAND_FLOOR}) "
                            f"-- the tolerance is too wide to decide whether claimed={claimed} "
                            f"matches artifact value={real}"}
    # REVERSE VACUITY (tolerance=0 on a float): tol==0 demands EXACT bitwise equality -- legitimate
    # for deterministic integer/boolean-like decisive_numbers (e.g. 1.0 = "all gates passed", count
    # fields; in the corpus every existing tol=0 claim has an integer value). But on a GENUINELY
    # continuous float (non-integer) tol=0 is "never passes" -- floating-point noise means the claim
    # can NEVER be verified even when it is true. ESCALATE (not FAIL) so a human decides whether the
    # exactness is really intended.
    if tol == 0.0 and (abs(claimed - round(claimed)) > 1e-12 or abs(real - round(real)) > 1e-12):
        return {"clause": "C2_decisive_number", "result": ESCALATE,
                "evidence": f"reverse vacuity: tolerance=0 against a non-integer float "
                            f"(claimed={claimed}, artifact={real}) -- floating-point noise can make "
                            "the claim UNVERIFIABLE forever even when it is true; confirm that exact "
                            "bitwise equality is really intended (otherwise set a real tolerance)"}
    if abs(real - claimed) > tol:
        return {"clause": "C2_decisive_number", "result": FAIL,
                "evidence": f"claimed={claimed} artifact-value={real} |delta|="
                            f"{abs(real - claimed):.6g} > tol={tol}"}
    return {"clause": "C2_decisive_number", "result": PASS,
            "evidence": f"claimed={claimed} matches artifact-value={real} within tol={tol}"}


# --------------------------------------------------------------------------------- #
# C3 -- threshold-provenance (the tautology-gate lesson)
# --------------------------------------------------------------------------------- #

def clause_c3_threshold_provenance(claim: dict) -> dict:
    gate_def = claim.get("gate_def") or {}
    anchor_class_raw = str(gate_def.get("anchor_class", "")).strip().lower()
    # ★FF7-ALIAS (BUNDLED with FP7's bare-label hardening below -- landing this alone
    # would convert an inert exposure into a LIVE one, see cert_grader_fix_bundle_v0's bundle
    # invariant): repository-wide short-form anchor_class convention ('ext-ref'/'known-fake', 36-38/88+
    # of the live corpus, and this bughunt's own claim) was previously REJECTED as 'unrecognized'
    # (a false-ESCALATE, FF7) purely because only the long-form tokens were in
    # ANCHOR_ACCEPTABLE/ANCHOR_REQUIRES_REF. Pure alias normalization -- NOT a new anchor class,
    # NOT a relaxation of any check below (FP7's bare-label ESCALATE applies identically to
    # short-form-resolved claims after this alias resolves).
    anchor_class = ANCHOR_CLASS_ALIASES.get(anchor_class_raw, anchor_class_raw)
    if not anchor_class_raw:
        return {"clause": "C3_threshold_provenance", "result": ESCALATE,
                "evidence": "gate_def.anchor_class missing -- provenance undetermined"}
    if anchor_class in ANCHOR_TAUTOLOGICAL:
        return {"clause": "C3_threshold_provenance", "result": FAIL,
                "evidence": f"anchor_class={anchor_class!r} is a SELF-QUANTILE anchor "
                            "(L317 lesson): the gate is measured against its own "
                            "distribution and always 'passes' by construction -- "
                            "tautological, not evidence"}
    if anchor_class in ANCHOR_REQUIRES_REF:
        # C3 REFERENT: anchor_class is a TRUSTED STRING today -- a claim can write
        # 'external-reference' without pointing at anything external at all (a tautology in
        # disguise). Require gate_def.anchor_ref (path or dataset token) MECHANICALLY, and when it
        # is a path, require that it does not live among the claim's OWN artifacts (self-reference
        # hides the same tautology one step away).
        anchor_ref = gate_def.get("anchor_ref")
        if not anchor_ref or not str(anchor_ref).strip():
            return {"clause": "C3_threshold_provenance", "result": ESCALATE,
                    "evidence": f"anchor_class={anchor_class!r} requires gate_def.anchor_ref "
                                "(path/dataset token) -- anchor referent missing"}
        anchor_ref_s = str(anchor_ref).strip()
        if not _looks_like_path(anchor_ref_s):
            # ★FP7-FIX (self-quantile-smyg via bare-label anchor_ref,, BUNDLED with
            # FF7's alias above -- see bundle invariant): a bare descriptive label ('my-totally-
            # external-benchmark-v1') satisfied the anchor_ref-present check to the LETTER but not
            # the SPIRIT -- nothing ties it to anything actually external (confirmed false-PASS,
            # false_pass_qc_v0 FP7). No anchor-registry exists yet to verify a label mechanically
            # (documented out-of-scope in cert_grader_bughunt_v0) -- ESCALATE, never PASS, until
            # one does. Applies to BOTH classes in ANCHOR_REQUIRES_REF (broader-than-stated per
            # false_pass_qc_v0's own net_count: the same bypass affects known-fake-anchored too,
            # not only external-reference as the original bug-table 'case' field named).
            return {"clause": "C3_threshold_provenance", "result": ESCALATE,
                    "evidence": f"anchor_class={anchor_class!r} anchor_ref={anchor_ref_s!r} is a "
                                "bare label, not a path/dataset-token -- cannot mechanically "
                                "verify anything external was actually used (no anchor-registry "
                                "exists yet); ESCALATE, not a silent PASS (FP7 fix)"}
        try:
            anchor_ref_norm = os.path.normpath(os.path.abspath(anchor_ref_s))
        except Exception:
            anchor_ref_norm = anchor_ref_s
        for p in _claim_artifact_paths(claim):
            try:
                p_norm = os.path.normpath(os.path.abspath(p))
            except Exception:
                p_norm = p
            if p_norm == anchor_ref_norm:
                return {"clause": "C3_threshold_provenance", "result": ESCALATE,
                        "evidence": f"anchor_class={anchor_class!r} anchor_ref={anchor_ref_s!r} "
                                    f"is IDENTICAL to the claim's own artifact ({p!r}) -- "
                                    "the anchor referent is self-referencing, not external"}
        return {"clause": "C3_threshold_provenance", "result": PASS,
                "evidence": f"anchor_class={anchor_class!r} anchored to anchor_ref={anchor_ref_s!r} "
                            "(non-self-referencing, path-form verified)"}
    if anchor_class in ANCHOR_ACCEPTABLE:
        return {"clause": "C3_threshold_provenance", "result": PASS,
                "evidence": f"anchor_class={anchor_class!r} is externally/known-fake "
                            "anchored -- non-tautological"}
    return {"clause": "C3_threshold_provenance", "result": ESCALATE,
            "evidence": f"anchor_class={anchor_class!r} unrecognized -- cannot classify "
                        "as tautological or externally anchored without a human read"}


# --------------------------------------------------------------------------------- #
# C4 -- lineage / decorrelation, delegated to D's gate (never re-implemented)
# --------------------------------------------------------------------------------- #

def clause_c4_lineage_decorrelation(claim: dict) -> dict:
    # VACUITY SWEEP: is a trivial PASS on 0/1 legs a hole? Reviewed against a 469-claim corpus: no
    # case of legs=[] together with a claimed_scope implying multi-source fusion (no claim hides a
    # multi-leg composition behind an empty legs list). Conclusion: BY DESIGN, not vacuous -- C4 is
    # SPECIFICALLY about PAIRWISE decorrelation, and a claim with 0/1 legs mechanically has no pair
    # to decorrelate (PASS is the honest figure, not a guess). The underlying number
    # (decisive_number) is checked INDEPENDENTLY by C2/C5 whatever the leg count, so an empty legs
    # list does NOT open a path past the actual claim substance -- it only switches off ONE clause
    # that is meaningless for a single-leg claim. A claim that deliberately EMPTIED its legs list to
    # escape a FENCE would be a hole, but that requires the claimant to already KNOW legs would be
    # fenced; no such signature was observed in the corpus.
    legs = claim.get("legs") or []
    if len(legs) < 2:
        return {"clause": "C4_lineage_decorrelation", "result": PASS,
                "evidence": f"{len(legs)} leg(s) -- nothing to decorrelate"}
    if _lineage_gate is None:
        return {"clause": "C4_lineage_decorrelation", "result": ESCALATE,
                "evidence": f"could not import leg_decorrelation_lineage_gate "
                            f"({_LINEAGE_GATE_IMPORT_ERROR}) -- cannot mechanically "
                            "check lineage"}
    verdicts = []
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            a, b = legs[i], legs[j]
            # lineage_kind distinction: shared TRAINED WEIGHTS = proven common mode -> the FENCE
            # applies. A shared CLASSICAL FORMULA (fisher-z, RANSAC, ...) on different data is NOT
            # common mode -- two confidence intervals on different strata are independent. An
            # identical formula lineage without empirical evidence -> ESCALATE, never FAIL.
            # REGISTRY-VERIFIED kind: a self-declared lineage_kind='classical-formula' could DODGE
            # the weights fence (label a trained model as a formula). Rule: a field that selects the
            # safe-vs-unsafe branch is VERIFIED from an identity registry, never taken on the
            # producer's word. verify_lineage_kind: the registry WINS; a mislabel = weights fence;
            # unknown = ESCALATE. Residual: a self-declared empirical ADMIT needs an on-disk
 # EVIDENCE POINTER.
            def _verified_kind(leg):
                if _lineage_gate is not None and hasattr(_lineage_gate, "verify_lineage_kind"):
                    v = _lineage_gate.verify_lineage_kind(leg.get("lineage_id"),
                                                          leg.get("lineage_kind", "trained-weights"))
                    return v.get("verified_kind"), v.get("mislabel"), v.get("basis")
                return leg.get("lineage_kind", "trained-weights"), False, "registry unavailable (declared kind)"
            kind_a, mis_a, basis_a = _verified_kind(a)
            kind_b, mis_b, basis_b = _verified_kind(b)
            same_id = a.get("lineage_id") and a.get("lineage_id") == b.get("lineage_id")
            if mis_a or mis_b:
                verdicts.append((i, j, {"verdict": "FENCE",
                                        "reason": f"lineage_kind MISLABEL (registry wins): {basis_a if mis_a else basis_b}"}))
                continue
            if same_id and (kind_a is None or kind_b is None):
                verdicts.append((i, j, {"verdict": "UNDECIDABLE",
                                        "reason": "shared lineage_id, kind unverifiable in registry -- ESCALATE"}))
                continue
            if same_id and kind_a == kind_b == "classical-formula":
                # SPOOF RECURSION: lineage_id is SELF-DECLARED -- a trained-model leg can claim
                # id='fisher-z' and reach this branch. A name allowlist can NEVER close that
                # (identity != name). The mechanically raisable bar: the evidence must be a PARSEABLE
                # decorrelation measurement (n_eff/corr fields) under a trusted reports path -- not
                # "the file exists" (/etc/hostname passed an earlier version!). OPEN RESIDUAL: real
                # identity verification is what the reproducer ACTUALLY ran (weight hash / behaviour
                # fingerprint); until that exists, a PASS here trusts the reports path, it does not
                # prove identity.
                # _empirical_evidence_ok is now the
 # module-level helper shared by EVERY branch below (was previously wired only here).
                emp_ok = ((a.get("empirical_verdict") == "ADMIT" or b.get("empirical_verdict") == "ADMIT")
                          and any(_empirical_evidence_ok(l) for l in (a, b)))
                if emp_ok:
                    verdicts.append((i, j, {"verdict": "DECORRELATED",
                                            "reason": "shared classical formula, registry-consistent, empirical ADMIT "
                                                      "with on-disk evidence pointer"}))
                else:
                    verdicts.append((i, j, {"verdict": "UNDECIDABLE",
                                            "reason": "shared classical-formula lineage: empirical decorrelation needs "
                                                      "an ON-DISK evidence pointer (self-declared ADMIT insufficient -- "
                                                      "L339 residual), else ESCALATE"}))
                continue
            emp_verdict_raw = a.get("empirical_verdict") or b.get("empirical_verdict")
            emp_neff = a.get("empirical_neff", b.get("empirical_neff"))
            emp_n = a.get("empirical_min_stratum_n", b.get("empirical_min_stratum_n"))
            # C4 UNIFICATION: the evidence-pointer requirement used to apply ONLY to the
            # same-id classical-formula branch above -- EVERY branch with a SELF-DECLARED
            # empirical_verdict (e.g. a distinct-lineage pair's ADMIT) was taken on its word here, and
            # a bare declaration was enough to lift NOT-DETERMINED to DECORRELATED. The same bar now
            # applies everywhere: no on-disk evidence pointer -> the declaration counts AS ABSENT
            # (compose_decorrelation_verdict is called WITHOUT it) and the pair escalates explicitly
            # instead of silently falling back to layer 1 (transparent, not a silent downgrade).
            if emp_verdict_raw and not any(_empirical_evidence_ok(l) for l in (a, b)):
                verdicts.append((i, j, {"verdict": "UNDECIDABLE",
                                        "reason": f"self-declared empirical_verdict={emp_verdict_raw!r} without "
                                                  "an on-disk evidence pointer (the classical-formula rule "
                                                  "applies to ALL branches) -- ESCALATE"}))
                continue
            emp_verdict = emp_verdict_raw
            v = _lineage_gate.compose_decorrelation_verdict(
                a.get("lineage_id"), b.get("lineage_id"),
                empirical_verdict=emp_verdict, empirical_neff=emp_neff,
                empirical_min_stratum_n=emp_n)
            verdicts.append((i, j, v))
    fences = [(i, j, v) for i, j, v in verdicts if v["verdict"] == "FENCE"]
    if fences:
        i, j, v = fences[0]
        return {"clause": "C4_lineage_decorrelation", "result": FAIL,
                "evidence": f"leg pair ({i},{j}) FENCE: {v['reason']}"}
    undetermined = [(i, j, v) for i, j, v in verdicts
                    if v["verdict"] in ("NOT-DETERMINED", "UNDECIDABLE")]
    if undetermined:
        i, j, v = undetermined[0]
        return {"clause": "C4_lineage_decorrelation", "result": ESCALATE,
                "evidence": f"leg pair ({i},{j}) {v['verdict']}: {v['reason']}"}
    return {"clause": "C4_lineage_decorrelation", "result": PASS,
            "evidence": f"all {len(verdicts)} leg pair(s) DECORRELATED/bookable"}


# --------------------------------------------------------------------------------- #
# C5 -- claim-artifact consistency (no narrated-but-absent figures)
# --------------------------------------------------------------------------------- #

def clause_c5_claim_artifact_consistency(claim: dict, artifact, extra_pool: list[float] | None = None) -> dict:
    claimed_numbers = claim.get("claimed_numbers")
    if not claimed_numbers:
        return {"clause": "C5_claim_artifact_consistency", "result": ESCALATE,
                "evidence": "no claimed_numbers listed -- nothing to cross-check"}
    # ★DICT-CLAIMED-NUMBERS ASYMMETRY: decisive_number has
    # a documented dict-form (_normalize_decisive_number above) but claimed_numbers never did --
    # every clause below does `float(n) for n in claimed_numbers`, which HARD-CRASHES with
    # ValueError when claimed_numbers is a dict (iterating a dict yields its string KEYS, not its
    # values). 115 live C claims hit this. Normalize the same way dict-form decisive_number does.
    if isinstance(claimed_numbers, dict):
        claimed_numbers = list(claimed_numbers.values())
    # ★NON-NUMERIC-ELEMENT RESIDUAL: dict->values
    # normalization above still crashes float when a VALUE itself is non-numeric (a verdict/mode/
    # bool string misplaced in claimed_numbers, e.g. claimed_numbers=[1.0, "PASS-MACHINE"]) -- C hit
    # 11 of these post-fix and worked around them by editing C's own claim files, not the tool, so
    # the crash risk remains live for any future claim (any lane). ESCALATE (not FAIL, not a silent
    # drop) matches this function's own established convention for malformed-but-not-provably-wrong
    # input (see the vacuous-tolerance / reverse-vacuity guards above) -- a non-numeric entry means the
    # claim itself needs fixing, not that the underlying claim is false.
    _non_numeric = [n for n in claimed_numbers if isinstance(n, bool) or not isinstance(n, (int, float))]
    if _non_numeric:
        return {"clause": "C5_claim_artifact_consistency", "result": ESCALATE,
                "evidence": f"claimed_numbers contains non-numeric element(s) {_non_numeric!r} -- "
                            "cannot cross-check against the artifact's numeric pool; fix the claim's "
                            "claimed_numbers to contain only numbers"}
    if artifact is None:
        return {"clause": "C5_claim_artifact_consistency", "result": ESCALATE,
                "evidence": "no artifact loaded -- cannot verify claimed numbers exist"}
    tol = float(claim.get("tolerance", 1e-6))
    # VACUOUS-TOLERANCE GUARD: C5 reads the SAME claimant-set `tolerance` as C2 for its pool
    # comparison (abs(n-p) <= tol) -- an oversized tol means EVERY claimed_number is "found"
    # whatever the artifact actually says (any number in the pool matches within a vacuous band).
    # Same bound function; the scale is the magnitude of the largest claimed number (the near-zero
    # exception applies equally).
    scale = max((abs(float(n)) for n in claimed_numbers), default=0.0)
    bound = _vacuous_tolerance_bound(scale)
    if tol > bound:
        return {"clause": "C5_claim_artifact_consistency", "result": ESCALATE,
                "evidence": f"vacuous tolerance: the claim cannot be falsified by its own gate -- "
                            f"tol={tol:g} > bound={bound:g} (rel_cap={_VACUOUS_REL_CAP}x scale={scale:g}, "
                            f"abs_floor/null_band={_VACUOUS_ABS_FLOOR}/{_VACUOUS_NULL_BAND_FLOOR}) "
                            "-- the tolerance is too wide to decide whether the claimed_numbers are "
                            "really present in the artifact (C2/C5 share the same one-sided-guard class)"}
    # REVERSE VACUITY (same as C2): tol==0 against a GENUINELY non-integer claimed_number risks a
    # claim that can NEVER be found in the pool because of floating-point noise (e.g. 0.873 vs
    # 0.8730000000000001) even when it is true -- ESCALATE, not a silent FAIL.
    if tol == 0.0:
        _non_integer = [n for n in claimed_numbers if abs(float(n) - round(float(n))) > 1e-12]
        if _non_integer:
            return {"clause": "C5_claim_artifact_consistency", "result": ESCALATE,
                    "evidence": f"reverse vacuity: tolerance=0 against non-integer claimed_numbers "
                                f"{_non_integer} -- floating-point noise can make the claim "
                                "UNVERIFIABLE forever even when it is true; confirm that exact bitwise "
                                "equality is intended (otherwise set a real tolerance)"}
    # ★FP6/FF3-DUAL POOL UNION: grade now graded ALL readable artifacts (not just
    # artifacts[0]); a claimed number legitimately living in artifacts[1] (a 2nd JSON in the same
    # citation bundle) should still be found -- extra_pool is the union of every OTHER readable
    # artifact's flattened numbers (never the representative one twice).
    pool = _flatten_numbers(artifact)
    if extra_pool:
        pool = pool + list(extra_pool)
    missing = [n for n in claimed_numbers
               if not any(abs(float(n) - p) <= tol for p in pool)]
    if missing:
        return {"clause": "C5_claim_artifact_consistency", "result": FAIL,
                "evidence": f"claimed number(s) {missing} not found anywhere in artifact "
                            f"({len(pool)} numeric leaves scanned) -- narrated-but-absent"}
    return {"clause": "C5_claim_artifact_consistency", "result": PASS,
            "evidence": f"all {len(claimed_numbers)} claimed number(s) present in artifact"}


# --------------------------------------------------------------------------------- #
# C6 -- scope honesty (claimed n/substrate vs artifact's actual scope)
# --------------------------------------------------------------------------------- #

def clause_c6_scope_honesty(claim: dict, artifact) -> dict:
    claimed_scope = claim.get("claimed_scope")
    if not claimed_scope:
        return {"clause": "C6_scope_honesty", "result": ESCALATE,
                "evidence": "no claimed_scope given -- cannot check for scope inflation"}
    if artifact is None:
        return {"clause": "C6_scope_honesty", "result": ESCALATE,
                "evidence": "no artifact loaded -- cannot verify actual scope"}
    real_scope = _artifact_scope(artifact)
    if real_scope["n"] is None and real_scope["substrate"] is None:
        return {"clause": "C6_scope_honesty", "result": ESCALATE,
                "evidence": "artifact carries no scope.n/scope.substrate (or top-level "
                            "n/substrate) fields -- cannot verify"}
    # ★FP4-FIX (partial-scope blind spot): the old loop only ever compared a field
    # when BOTH claimed and real sides were not-None, then returned unconditional PASS whenever
    # `mismatches` was empty -- indistinguishable from "every claimed field was actually checked
    # and matched." A claimed_scope field with NO artifact counterpart (e.g. claimed
    # substrate='real-world-validated' but the artifact's own scope carries no substrate at all)
    # was silently treated as matching (confirmed false-PASS, false_pass_qc_v0 FP4). Track
    # unchecked fields SEPARATELY: a real detected mismatch still FAILs (worse than unchecked, and
    # unaffected by this fix -- negative control unchanged), but a field that was NEVER CHECKED
    # ESCALATEs instead of silently PASSing.
    mismatches = []
    unchecked = []
    if claimed_scope.get("n") is not None:
        if real_scope["n"] is not None:
            if int(claimed_scope["n"]) != int(real_scope["n"]):
                mismatches.append(f"n: claimed={claimed_scope['n']} actual={real_scope['n']}")
        else:
            unchecked.append("n")
    if claimed_scope.get("substrate") is not None:
        if real_scope["substrate"] is not None:
            if str(claimed_scope["substrate"]).strip().lower() != \
               str(real_scope["substrate"]).strip().lower():
                mismatches.append(f"substrate: claimed={claimed_scope['substrate']!r} "
                                   f"actual={real_scope['substrate']!r}")
        else:
            unchecked.append("substrate")
    if mismatches:
        return {"clause": "C6_scope_honesty", "result": FAIL,
                "evidence": "scope mismatch: " + "; ".join(mismatches)}
    if unchecked:
        return {"clause": "C6_scope_honesty", "result": ESCALATE,
                "evidence": f"claimed_scope field(s) {unchecked} have no counterpart in the "
                            f"artifact's own scope ({real_scope}) -- NEVER CHECKED, not confirmed "
                            "matching (FP4 fix: partial-scope blind spot)"}
    return {"clause": "C6_scope_honesty", "result": PASS,
            "evidence": f"claimed scope matches artifact scope ({real_scope})"}


# --------------------------------------------------------------------------------- #
# -- unit sanity guard: units are not cosmetic -- a 10x cm/mm error can be written silently.
# Fires only when the claim carries the measurement_claim_sync "target"
# convention (target.unit + target.value + target.field_path); a claim without that
# convention has nothing for to check -- trivial PASS, never a new ESCALATE burden on
# every existing claim.
# --------------------------------------------------------------------------------- #

_UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
_UNIT_SUFFIX_RE = re.compile(r"(?:^|_)(mm|cm|m)(?:$|[_.])")


def _flatten_paths(obj: Any, prefix: str = "", out: dict[str, float] | None = None) -> dict[str, float]:
    """Like _flatten_numbers but keeps the dotted PATH for each numeric leaf (needed to find a
    sibling field sharing a basename under a different unit suffix, e.g. dimensions_cm.outer vs
    dimensions_mm.outer)."""
    if out is None:
        out = {}
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_paths(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            _flatten_paths(v, f"{prefix}.{idx}" if prefix else str(idx), out)
    return out


def _field_unit_signature(path_str: str) -> tuple[str | None, str]:
    """Scan EVERY dotted segment of a field path (not just the last -- the real convention is
    'dimensions_mm.outer', the unit suffix lives on a PARENT segment, not the leaf) for a
    mm/cm/m unit token. Returns (unit_or_None, signature) where `signature` is the path with
    that segment's unit token stripped, so 'dimensions_mm.outer' and 'dimensions_cm.outer' -- the
    SAME quantity in two units -- collapse onto the same signature 'dimensions.outer' and can be
    cross-checked."""
    segs = str(path_str or "").lower().split(".")
    for idx, seg in enumerate(segs):
        m = _UNIT_SUFFIX_RE.search(seg)
        if m:
            base = seg[: m.start(1)].rstrip("_.")
            sig_segs = list(segs)
            sig_segs[idx] = base
            return m.group(1), ".".join(sig_segs)
    return None, ".".join(segs)


def clause_c12_unit_guard(claim: dict, artifact) -> dict:
    target = claim.get("target") or {}
    unit = target.get("unit")
    field_path = target.get("field_path")
    value = target.get("value")
    if not unit or value is None or not isinstance(unit, str):
        return {"clause": "C12_unit_guard", "result": PASS,
                "evidence": "no target.unit/target.value declared -- nothing to size-check"}
    unit_norm = unit.strip().lower()
    if unit_norm not in _UNIT_TO_MM:
        return {"clause": "C12_unit_guard", "result": PASS,
                "evidence": f"unit={unit_norm!r} outside the mm/cm/m length family -- C12 is "
                            "scoped to that family only (mechanical band check needs a known "
                            "conversion factor)"}
    try:
        value_f = float(value)
    except Exception:
        return {"clause": "C12_unit_guard", "result": ESCALATE,
                "evidence": f"target.value={value!r} not numeric -- cannot size-check against unit={unit_norm!r}"}

    implied_unit, sig = _field_unit_signature(str(field_path or ""))
    if implied_unit and implied_unit != unit_norm:
        return {"clause": "C12_unit_guard", "result": ESCALATE,
                "evidence": f"target.unit={unit_norm!r} but field_path={field_path!r} implies "
                            f"unit={implied_unit!r} -- unit/field-name mismatch (orimlig storleksordning)"}

    # cm-vs-mm-10x-detektorn: if the artifact carries the SAME quantity (same signature) under a
    # DIFFERENT unit suffix somewhere in its own field name, the two must agree (within 10%)
    # after conversion to mm -- a bare 10x slip (135mm claimed, artifact's own
    # dimensions_cm.outer says 13.5, i.e. 135mm too -> fine; but if artifact says 1.35 (13.5mm)
    # or 135 (1350mm) that is a 10x/100x unit error) is caught mechanically instead of cosmetic.
    if implied_unit and artifact is not None:
        value_mm = value_f * _UNIT_TO_MM[unit_norm]
        for path, other_val in _flatten_paths(artifact).items():
            other_unit, other_sig = _field_unit_signature(path)
            if not other_unit or other_sig != sig or other_unit == unit_norm:
                continue
            other_mm = other_val * _UNIT_TO_MM[other_unit]
            if other_mm == 0:
                continue
            ratio = value_mm / other_mm
            if not (0.9 <= ratio <= 1.1):
                return {"clause": "C12_unit_guard", "result": ESCALATE,
                        "evidence": f"target {field_path}={value_f}{unit_norm} ({value_mm:g}mm) vs "
                                    f"artifact {path}={other_val}{other_unit} ({other_mm:g}mm) -- "
                                    f"ratio={ratio:.3g}, implausible order of magnitude (suspected unit "
                                    "error, e.g. cm/mm 10x)"}

    return {"clause": "C12_unit_guard", "result": PASS,
            "evidence": f"target.unit={unit_norm!r} consistent with field name"
                        + (" and artifact cross-check" if implied_unit else "")}


# --------------------------------------------------------------------------------- #
# grade-level helpers: B-clause schema adapter, NA-downgrade, C7 wiring (FP1 fix)
# --------------------------------------------------------------------------------- #

def _normalize_claim_for_b_clauses(claim: dict) -> dict:
    """B's rider clauses (C7-) are coded against the OFFICIAL CLAIM_CONTRACT (decisive_number
    as a dict, artifacts as a list) -- this engine's internal/documented form also allows
    decisive_number as a bare float (+ top-level decisive_number_path/tolerance) and artifacts as
    {'path':...}. Normalize into B's expected shape using the SAME _normalize_decisive_number
    canonicalization C2 uses, so C7 sees decisive_number_path under the EXACT key it reads
    (dn.get('decisive_number_path')) regardless of which live alias (decisive_number_path/name/
    path) the claim actually used.

    ★FP1-FIX: the PREVIOUS version of this adapter did `{'decisive_number':
    {'value': float(_dn)}}` for the bare-float branch, DROPPING decisive_number_path entirely --
    C7's `dn.get('decisive_number_path')` was therefore always None, and C7 ALWAYS ESCALATEd
    'no decisive_number_path...', silently NA-downgraded, making C7 structurally dead for every
    documented-contract claim (confirmed false-PASS, false_pass_qc_v0 FP1: nothing in the whole
    pipeline required the reproducer to have PRODUCED the claimed number). This version also
    resolves dict-form decisive_number (FF4 'name'/'path' key aliases), extending the fix to
    those claims too -- previously they were passed through with whatever key they originally
    used, which C7 also could not read."""
    claim_b = claim
    value, path, tol, _tol_escalate = _normalize_decisive_number(claim)
    if value is not None:
        claim_b = {**claim_b, "decisive_number": {"value": value, "decisive_number_path": path,
                                                    "tolerance": tol}}
    art = claim_b.get("artifacts")
    if isinstance(art, dict):
        claim_b = {**claim_b, "artifacts": [art.get("path")]}
    return claim_b


def _na_downgrade(r: dict) -> dict:
    """A missing field = NOT APPLICABLE (informative, does not gate) -- otherwise the rider clauses
    contaminate every claim that does not carry their fields. Present-but-wrong still gates fully.
    Factored out so both the C8- loop and C7's separately-timed call share one rule."""
    if r.get("result") == ESCALATE:
        ev = str(r.get("evidence", "")).lower()
        if ev.startswith(("no ", "not a ")) or any(k in ev for k in
                ("missing", "absent", "not provided", "needs ", "cannot tell",
                 "not present", "do not declare", "no legs declare")):
            return {**r, "result": "NA", "evidence": "n/a (field not in the claim): " + str(r.get("evidence"))[:80]}
    return r


def _c7_artifact_path(claim_b: dict):
    """Mirror B's C7 art_path resolution (read-only trace of
    b_rider_clause_C7_decisive_number_verified_vs_reproducer_output.clause_c7_decisive_number_verified,
    NEVER a copy -- B's module is cross-worktree and this engine never modifies it). Needed here
    so the FP1 freshness guard below can snapshot the artifact's mtime BEFORE the reproducer runs."""
    rep = claim_b.get("reproducer") or {}
    art_path = rep.get("emits_artifact")
    if not art_path:
        arts = claim_b.get("artifacts")
        if isinstance(arts, list) and arts:
            art_path = arts[0]
    return art_path if isinstance(art_path, str) and art_path else None


def _safe_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


_C7_FN_NAME = "clause_c7_decisive_number_verified"
_c7_fn = next((f for f in _B_CLAUSES if callable(f) and getattr(f, "__name__", "") == _C7_FN_NAME), None)
_c7_unavailable = next((bc for bc in _B_CLAUSES
                         if not callable(bc) and bc[1] == _C7_FN_NAME), None)


# --------------------------------------------------------------------------------- #
# top-level grade
# --------------------------------------------------------------------------------- #

def grade(claim: dict, run_reproducer: bool = True) -> dict:
    """Grade one delivered claim. Returns:
      {verdict, ms, ms_verdict, ms_repro, checks: [...], escalate, escalate_reason}
    `ms` = total wall time. `ms_verdict` = C2-C6 only (the <9ms-target budget).
    `ms_repro` = C1 subprocess time, reported separately (seconds-scale, not counted
    against the 9ms verdict-logic target).
    """
    t_total0 = time.perf_counter()

    checks = []
    claim_b = _normalize_claim_for_b_clauses(claim)

    # C1 (+ C7, FP1 fix) are the only I/O-heavy, seconds-scale clauses -- timed together, both
    # gated by run_reproducer (previously C7 ran unconditionally inside the <9ms ms_verdict
    # bucket, which is exactly the SECOND hazard cert_grader_bughunt_v0 flagged: naively wiring
    # C7's decisive_number_path through would make it re-run the reproducer subprocess a second
    # time inside the sub-9ms budget window).
    t_repro0 = time.perf_counter()
    if run_reproducer:
        # ★FP1-FRESHNESS-GUARD: snapshot the C7-relevant artifact's mtime BEFORE C1
        # runs the reproducer (MUST happen before the clause_c1_reproducer call below, not
        # after -- snapshotting post-run would see C1's own fresh write and always read as
        # "unchanged"), then call C7 with run=False -- C1 JUST ran the identical reproducer.cmd
        # moments ago as a side effect, so re-running it a second time inside C7 would double the
        # real (seconds-scale) subprocess cost for no benefit; run=False makes C7 check the
        # artifact C1's run just (potentially) produced instead. A C7 PASS is only trustworthy if
        # the artifact was ACTUALLY (re)written during this run -- not a pre-existing
        # self-authored file the reproducer never touches (the exact FP1 construction:
        # self-declared decisive_number written into an artifact the claimant also controls, with
        # a reproducer whose stdout matches an unrelated pattern). A PASS resting on an unchanged
        # mtime is downgraded to ESCALATE, never silently trusted.
        _c7_art_path = _c7_artifact_path(claim_b) if _c7_fn is not None else None
        _c7_mtime_before = _safe_mtime(_c7_art_path) if _c7_art_path else None
        checks.append(clause_c1_reproducer(claim))
        if _c7_fn is not None:
            try:
                _r = _c7_fn(claim_b, run=False)
                if _r.get("result") == PASS:
                    _mtime_after = _safe_mtime(_c7_art_path) if _c7_art_path else None
                    _fresh = ((_c7_mtime_before is None and _mtime_after is not None) or
                              (_c7_mtime_before is not None and _mtime_after is not None
                               and _mtime_after > _c7_mtime_before))
                    if not _fresh:
                        _r = {**_r, "result": ESCALATE,
                              "evidence": "C7 reported PASS but the emitted artifact's mtime did "
                                          f"NOT change during this reproducer run (path="
                                          f"{_c7_art_path!r}) -- cannot verify the number was "
                                          "ACTUALLY produced by the reproducer, not a pre-existing "
                                          "self-authored file (FP1 freshness guard)"}
                checks.append(_na_downgrade(_r))
            except Exception as _e:
                checks.append({"clause": _C7_FN_NAME, "result": ESCALATE,
                               "evidence": f"C7 raised {type(_e).__name__}: {str(_e)[:80]}"})
        elif _c7_unavailable is not None:
            checks.append({"clause": _c7_unavailable[1], "result": ESCALATE,
                           "evidence": f"B worktree clause unavailable: {_c7_unavailable[2]}"})
    ms_repro = (time.perf_counter() - t_repro0) * 1000.0

    # C2-C6: the verdict-logic budget.
    t_verdict0 = time.perf_counter()
    artifact = None
    extra_numeric_pool: list = []
    # ★FP6/FF3/FP5-FIX: grade ALL of the claim's own readable artifacts, not just
    # index 0 (FP6: a contradicting 2nd artifact was previously silently ignored regardless of
    # list order; FF3 dual: 13 real claims false-ESCALATE today because entry[0] happens to be a
    # non-JSON companion file while a later entry IS the real data -- same 'first=art[0]' root
    # cause, both directions). Also applies FP5's repo/lane-containment check (symmetric with C4's
    # own _empirical_evidence_ok rule) before trusting any artifact as ground truth.
    art_paths = _claim_artifact_paths(claim)
    loaded: list = []       # [(path, parsed_json)]
    load_errors: list = []  # [(path, error_str)]
    for _p in art_paths:
        try:
            loaded.append((_p, _load_artifact(_p)))
        except Exception as e:
            load_errors.append((_p, str(e)))

    if art_paths and not loaded:
        # every named artifact was unreadable -- every clause that needed one escalates,
        # explicitly, rather than silently skipping.
        _artifact_error = "; ".join(f"{p!r}: {e}" for p, e in load_errors)
        for clause_name in ("C2_decisive_number", "C5_claim_artifact_consistency",
                            "C6_scope_honesty"):
            checks.append({"clause": clause_name, "result": ESCALATE,
                            "evidence": f"no artifact could be read ({len(art_paths)} path(s) "
                                        f"tried): {_artifact_error}"})
    elif loaded:
        _dn_value, _dn_path, _dn_tol, _ = _normalize_decisive_number(claim)
        _resolved = []
        if _dn_path:
            for _p, _a in loaded:
                try:
                    _resolved.append((_p, float(_dig(_a, _dn_path))))
                except Exception:
                    pass
        _resolved_paths = {p for p, _v in _resolved}
        _conflict = None
        for _i in range(len(_resolved)):
            for _j in range(_i + 1, len(_resolved)):
                if abs(_resolved[_i][1] - _resolved[_j][1]) > max(_dn_tol or 1e-6, 1e-9):
                    _conflict = (_resolved[_i], _resolved[_j])
                    break
            if _conflict:
                break

        # representative-artifact ranking: prefer (trusted AND resolves decisive_number_path) >
        # (resolves it, even if untrusted -- a trusted-but-IRRELEVANT artifact must not shadow
        # the one that actually carries the number, or C2 spuriously FAILs 'not found' against
        # the wrong file) > (trusted, doesn't resolve) > (first loaded). Stable sort preserves
        # original claim list order among ties.
        def _artifact_rank(entry):
            _p, _a = entry
            _trust = _artifact_path_trusted(_p)
            _resolves = _p in _resolved_paths
            return (_trust and _resolves, _resolves, _trust)

        rep_path, artifact = sorted(loaded, key=_artifact_rank, reverse=True)[0]
        artifact_trusted = _artifact_path_trusted(rep_path)
        extra_numeric_pool = [n for p, a in loaded if a is not artifact for n in _flatten_numbers(a)]

        if _conflict:
            (p1, v1), (p2, v2) = _conflict
            checks.append({"clause": "C2_decisive_number", "result": FAIL,
                            "evidence": f"multi-artifact CONTRADICTION at decisive_number_path="
                                        f"{_dn_path!r}: {p1!r}={v1!r} vs {p2!r}={v2!r} (|delta|="
                                        f"{abs(v1 - v2):.6g}) -- claim names {len(loaded)} "
                                        "artifacts that disagree; never silently pick one by list "
                                        "order (FP6 fix)"})
            checks.append(clause_c5_claim_artifact_consistency(claim, artifact, extra_pool=extra_numeric_pool))
            checks.append(clause_c6_scope_honesty(claim, artifact))
        else:
            _c2r = clause_c2_decisive_number(claim, artifact)
            _c5r = clause_c5_claim_artifact_consistency(claim, artifact, extra_pool=extra_numeric_pool)
            _c6r = clause_c6_scope_honesty(claim, artifact)
            if not artifact_trusted:
                # ★FP5-FIX: containment only needs to gate the PASS direction -- a
                # FAIL or ESCALATE derived from an untrusted artifact is still a SAFE, honest
                # outcome (a real detected defect, or an honest 'cannot decide', doesn't become
                # less real for living in a scratch path -- and blanket-ESCALATE-ing a genuine
                # FAIL would have thrown away a correct catch just because of where the file
                # lives). Only a PASS is the actual FP5 threat model (an arbitrary self-authored
                # /tmp file forging a match) -- downgrade PASS-only to ESCALATE, never silently
                # trust an unanchored positive result.
                _suffix = (f" (FP5 fix: representative artifact {rep_path!r} is outside any "
                          "trusted reports/ tree -- unanchored provenance, a PASS cannot "
                          "be trusted from here)")
                if _c2r.get("result") == PASS:
                    _c2r = {**_c2r, "result": ESCALATE, "evidence": str(_c2r.get("evidence", "")) + _suffix}
                if _c5r.get("result") == PASS:
                    _c5r = {**_c5r, "result": ESCALATE, "evidence": str(_c5r.get("evidence", "")) + _suffix}
                if _c6r.get("result") == PASS:
                    _c6r = {**_c6r, "result": ESCALATE, "evidence": str(_c6r.get("evidence", "")) + _suffix}
            checks.append(_c2r)
            checks.append(_c5r)
            checks.append(_c6r)
    else:
        # no artifacts named at all -- unchanged: clauses called with artifact=None, each
        # ESCALATEs on its own "no artifact loaded" path.
        checks.append(clause_c2_decisive_number(claim, None))
        checks.append(clause_c5_claim_artifact_consistency(claim, None))
        checks.append(clause_c6_scope_honesty(claim, None))

    checks.append(clause_c12_unit_guard(claim, artifact))
    checks.append(clause_c3_threshold_provenance(claim))
    # Optional plugin clauses C8- (optional claim fields -- a clause without its fields returns
    # ESCALATE/skip by design). C7 is handled separately above (ms_repro bucket).
    for _bc in _B_CLAUSES:
        if callable(_bc):
            if getattr(_bc, "__name__", "") == _C7_FN_NAME:
                continue  # already handled in the ms_repro-timed block above
            try:
                _r = _bc(claim_b)
                checks.append(_na_downgrade(_r))
            except Exception as _e:
                checks.append({"clause": getattr(_bc, "__name__", "B_clause"), "result": ESCALATE,
                               "evidence": f"B-clause raised {type(_e).__name__}: {str(_e)[:60]}"})
        else:
            if _bc[1] == _C7_FN_NAME:
                continue  # already handled above (unavailable case reported there too)
            checks.append({"clause": _bc[1], "result": ESCALATE,
                           "evidence": f"B worktree clause unavailable: {_bc[2]}"})
    checks.append(clause_c4_lineage_decorrelation(claim))

    ms_verdict = (time.perf_counter() - t_verdict0) * 1000.0
    ms_total = (time.perf_counter() - t_total0) * 1000.0

    fails = [c for c in checks if c["result"] == FAIL]
    escalates = [c for c in checks if c["result"] == ESCALATE]

    if fails:
        verdict = "FAIL-MACHINE"
        escalate = False
        escalate_reason = None
    elif escalates:
        verdict = "ESCALATE-LLM"
        escalate = True
        escalate_reason = "; ".join(f"{c['clause']}: {c['evidence']}" for c in escalates)
    else:
        verdict = "PASS-MACHINE"
        escalate = False
        escalate_reason = None

    return {
        "verdict": verdict,
        "ms": round(ms_total, 4),
        "ms_verdict": round(ms_verdict, 4),
        "ms_repro": round(ms_repro, 4),
        "checks": checks,
        "escalate": escalate,
        "escalate_reason": escalate_reason,
    }


# --------------------------------------------------------------------------------- #
# selftest -- >=6 synthetic claims: 2 clean-pass, 2 hard-fail (self-quantile gate +
# shared-lineage pair), 2 escalate. All timed; verdict-logic (excl C1) must be <9ms.
# --------------------------------------------------------------------------------- #

def _write_tmp_artifact(tmp_dir: str, name: str, payload: dict) -> str:
    path = os.path.join(tmp_dir, name)
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def _selftest() -> bool:
    import tempfile

    ok = True
    results = []

    # ★C4-unifiering selftest fixture: a REAL on-disk evidence pointer under a trusted
    # lane reports/ path (repo-relative reports/probes/), so claim1's distinct-lineage-pair
    # empirical ADMIT is legitimately backed, not a bare self-declaration. Written just for
    # this selftest run and removed in `finally` -- never left behind.
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _evidence_path = os.path.join(_repo_root, "reports", "probes", "_selftest_c4_evidence_tmp.json")
    os.makedirs(os.path.dirname(_evidence_path), exist_ok=True)
    with open(_evidence_path, "w") as _f:
        json.dump({"n_eff": 2.05, "note": "cert_grade_engine selftest fixture -- safe to delete"}, _f)

    # ★FP5-FIX-COMPATIBLE: artifacts must now live under a trusted
    # trusted reports/ tree (see _artifact_path_trusted) -- system /tmp no longer
    # qualifies. Create the selftest's scratch dir INSIDE reports/probes/ instead (still a real
    # tempdir, still auto-cleaned on context exit, just repo-contained).
    with tempfile.TemporaryDirectory(prefix="cert_grade_engine_selftest_",
                                      dir=os.path.join(_repo_root, "reports", "probes")) as tmp:

        # ---- 1: clean PASS -- everything checks out ----
        art1 = _write_tmp_artifact(tmp, "clean1.json", {
            "metrics": {"auc": 0.842}, "scope": {"n": 120, "substrate": "euroc-mono"},
        })
        claim1 = {
            "id": "clean-pass-1",
            "decisive_number": 0.842, "decisive_number_path": "metrics.auc", "tolerance": 1e-6,
            # ★FP1-FIX-COMPATIBLE: the reproducer must ACTUALLY (re-)emit the
            # artifact for C7's freshness guard to trust a PASS (the whole point of the FP1 fix --
            # a self-authored artifact the reproducer never touches must NOT verify).
            "reproducer": {"cmd": ["python3", "-c",
                                    f"import json; json.dump({{'metrics': {{'auc': 0.842}}, "
                                    f"'scope': {{'n': 120, 'substrate': 'euroc-mono'}}}}, "
                                    f"open({art1!r}, 'w')); print('OK 0.842')"],
                            "expect_regex": r"OK 0\.842"},
            # ★FP7-FIX-COMPATIBLE: anchor_ref must be path-form now that a bare
            # label ESCALATEs (see clause_c3's FP7 fix) -- a real external benchmark reference,
            # not one of this claim's own artifacts.
            "gate_def": {"anchor_class": "external-reference",
                         "anchor_ref": "external_benchmarks/euroc_mono_v1.json"},
            "legs": [{"lineage_id": "kubric-cnn-transformer"},
                     {"lineage_id": "whisper conv+transformer audio",
                      "empirical_verdict": "ADMIT", "empirical_evidence_path": _evidence_path}],
            "claimed_numbers": [0.842, 120],
            "claimed_scope": {"n": 120, "substrate": "euroc-mono"},
            "artifacts": {"path": art1},
        }
        results.append(("clean-pass-1", "PASS-MACHINE", grade(claim1)))
        try:
            os.remove(_evidence_path)
        except OSError:
            pass

        # ---- 2: clean PASS -- known-fake-anchored, single leg (trivial C4) ----
        art2 = _write_tmp_artifact(tmp, "clean2.json", {
            "rho": 0.913, "n": 64, "substrate": "kubric-synth",
        })
        claim2 = {
            "id": "clean-pass-2",
            "decisive_number": 0.913, "decisive_number_path": "rho", "tolerance": 1e-6,
            # ★FP1-FIX-COMPATIBLE: also (re-)emit art2 so C7's freshness guard sees
            # a real write, not just a stdout print.
            "reproducer": {"cmd": ["python3", "-c",
                                    f"import json; json.dump({{'rho': 0.913, 'n': 64, "
                                    f"'substrate': 'kubric-synth'}}, open({art2!r}, 'w')); "
                                    f"print(json.dumps({{'rho': 0.913}}))"],
                            "expect_json_path": "rho", "expect_json_value": 0.913},
            "gate_def": {"anchor_class": "known-fake-anchored",
                         "anchor_ref": "external_benchmarks/kubric_synth_known_fake_negative_control.json"},
            "legs": [{"lineage_id": "dinov2-vit-s (Metric3D)"}],
            "claimed_numbers": [0.913, 64],
            "claimed_scope": {"n": 64, "substrate": "kubric-synth"},
            "artifacts": {"path": art2},
        }
        results.append(("clean-pass-2", "PASS-MACHINE", grade(claim2)))

        # ---- 3: hard FAIL -- self-quantile tautological gate (lesson) ----
        art3 = _write_tmp_artifact(tmp, "fail_selfq.json", {
            "score": 0.99, "n": 50, "substrate": "self-quantile-set",
        })
        claim3 = {
            "id": "hard-fail-self-quantile",
            "decisive_number": 0.99, "decisive_number_path": "score", "tolerance": 1e-6,
            "reproducer": {"cmd": ["python3", "-c", "print('OK 0.99')"],
                            "expect_regex": r"OK 0\.99"},
            "gate_def": {"anchor_class": "self-quantile"},
            "legs": [],
            "claimed_numbers": [0.99, 50],
            "claimed_scope": {"n": 50, "substrate": "self-quantile-set"},
            "artifacts": {"path": art3},
        }
        results.append(("hard-fail-self-quantile", "FAIL-MACHINE", grade(claim3)))

        # ---- 4: hard FAIL -- shared-lineage leg pair (identical weights lineage) ----
        art4 = _write_tmp_artifact(tmp, "fail_lineage.json", {
            "auc": 0.77, "n": 30, "substrate": "real-video",
        })
        claim4 = {
            "id": "hard-fail-shared-lineage",
            "decisive_number": 0.77, "decisive_number_path": "auc", "tolerance": 1e-6,
            "reproducer": {"cmd": ["python3", "-c", "print('OK 0.77')"],
                            "expect_regex": r"OK 0\.77"},
            "gate_def": {"anchor_class": "external-reference", "anchor_ref": "real-video-external-baseline"},
            "legs": [{"lineage_id": "dinov2-vit-s (Metric3D)"},
                     {"lineage_id": "dinov2-vit-s (VDA)"}],
            "claimed_numbers": [0.77, 30],
            "claimed_scope": {"n": 30, "substrate": "real-video"},
            "artifacts": {"path": art4},
        }
        results.append(("hard-fail-shared-lineage", "FAIL-MACHINE", grade(claim4)))

        # ---- 5: ESCALATE -- distinct-lineage leg pair with NO empirical evidence yet
        # (layer-1 alone can't decide -- NOT-DETERMINED, per D's gate) ----
        art5 = _write_tmp_artifact(tmp, "escalate_lineage.json", {
            "auc": 0.81, "n": 45, "substrate": "real-video",
        })
        claim5 = {
            "id": "escalate-lineage-not-determined",
            "decisive_number": 0.81, "decisive_number_path": "auc", "tolerance": 1e-6,
            "reproducer": {"cmd": ["python3", "-c", "print('OK 0.81')"],
                            "expect_regex": r"OK 0\.81"},
            "gate_def": {"anchor_class": "external-reference", "anchor_ref": "real-video-external-baseline"},
            "legs": [{"lineage_id": "dinov2-vit-s (Metric3D)"},
                     {"lineage_id": "kubric-cnn-transformer (CoTracker)"}],
            "claimed_numbers": [0.81, 45],
            "claimed_scope": {"n": 45, "substrate": "real-video"},
            "artifacts": {"path": art5},
        }
        results.append(("escalate-lineage-not-determined", "ESCALATE-LLM", grade(claim5)))

        # ---- 6: ESCALATE -- unrecognized anchor_class (provenance genuinely unknown) ----
        art6 = _write_tmp_artifact(tmp, "escalate_anchor.json", {
            "auc": 0.65, "n": 20, "substrate": "mixed",
        })
        claim6 = {
            "id": "escalate-unknown-anchor",
            "decisive_number": 0.65, "decisive_number_path": "auc", "tolerance": 1e-6,
            "reproducer": {"cmd": ["python3", "-c", "print('OK 0.65')"],
                            "expect_regex": r"OK 0\.65"},
            "gate_def": {"anchor_class": "reviewer-eyeballed"},
            "legs": [],
            "claimed_numbers": [0.65, 20],
            "claimed_scope": {"n": 20, "substrate": "mixed"},
            "artifacts": {"path": art6},
        }
        results.append(("escalate-unknown-anchor", "ESCALATE-LLM", grade(claim6)))

        # ---- 7 (bonus): hard FAIL -- narrated-but-absent decisive number (C2) ----
        art7 = _write_tmp_artifact(tmp, "fail_absent.json", {
            "auc": 0.50, "n": 20, "substrate": "mixed",
        })
        claim7 = {
            "id": "hard-fail-absent-number",
            "decisive_number": 0.95, "decisive_number_path": "auc", "tolerance": 1e-6,
            "reproducer": {"cmd": ["python3", "-c", "print('OK')"], "expect_regex": r"OK"},
            "gate_def": {"anchor_class": "external-reference", "anchor_ref": "mixed-external-baseline"},
            "legs": [],
            "claimed_numbers": [0.95],
            "claimed_scope": {"n": 20, "substrate": "mixed"},
            "artifacts": {"path": art7},
        }
        results.append(("hard-fail-absent-number", "FAIL-MACHINE", grade(claim7)))

        # ---- 8 (bonus): hard FAIL -- scope inflation (C6) ----
        art8 = _write_tmp_artifact(tmp, "fail_scope.json", {
            "auc": 0.80, "n": 12, "substrate": "sim-only",
        })
        claim8 = {
            "id": "hard-fail-scope-inflation",
            "decisive_number": 0.80, "decisive_number_path": "auc", "tolerance": 1e-6,
            "reproducer": {"cmd": ["python3", "-c", "print('OK')"], "expect_regex": r"OK"},
            "gate_def": {"anchor_class": "external-reference", "anchor_ref": "sim-only-external-baseline"},
            "legs": [],
            "claimed_numbers": [0.80, 12],
            "claimed_scope": {"n": 500, "substrate": "real-world"},
            "artifacts": {"path": art8},
        }
        results.append(("hard-fail-scope-inflation", "FAIL-MACHINE", grade(claim8)))

        print(f"{'claim':<32} {'expected':<14} {'got':<14} {'ms_verdict':>10} "
              f"{'ms_repro':>9} {'ms_total':>9}  ok")
        for name, expected, g in results:
            passed = g["verdict"] == expected
            ok = ok and passed
            print(f"{name:<32} {expected:<14} {g['verdict']:<14} {g['ms_verdict']:>10.4f} "
                  f"{g['ms_repro']:>9.4f} {g['ms']:>9.4f}  {'PASS' if passed else 'FAIL'}")
            if not passed or "--verbose" in sys.argv:
                for c in g["checks"]:
                    print(f"     [{c['result']:<8}] {c['clause']:<28} {c['evidence']}")
            if g["escalate"]:
                print(f"     escalate_reason: {g['escalate_reason']}")

        verdict_times = [g["ms_verdict"] for _, _, g in results]
        max_ms = max(verdict_times)
        mean_ms = sum(verdict_times) / len(verdict_times)
        budget_ok = max_ms < 9.0
        print(f"\nverdict-logic latency (C2-C6, excl. C1 subprocess): "
              f"mean={mean_ms:.4f}ms max={max_ms:.4f}ms target<9ms/claim -> "
              f"{'MET' if budget_ok else 'MISSED'}")
        if not budget_ok:
            slowest = max(results, key=lambda r: r[2]["ms_verdict"])
            print(f"  slowest claim: {slowest[0]} at {slowest[2]['ms_verdict']:.4f}ms "
                  "-- profile before trusting the 9ms target further "
                  "(likely dominated by JSON artifact I/O, not clause logic).")
        ok = ok and budget_ok

    print(f"\ncert_grade_engine selftest: {'ALL PASS' if ok else 'FAILURES'} "
          f"({sum(1 for n, e, g in results if g['verdict'] == e)}/{len(results)} verdicts correct, "
          f"latency {'MET' if budget_ok else 'MISSED'})")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.selftest or True:
        ok = _selftest()
        sys.exit(0 if ok else 1)
