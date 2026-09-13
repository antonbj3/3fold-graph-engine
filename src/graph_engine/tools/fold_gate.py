#!/usr/bin/env python3
"""fold_gate.py — THE HARVEST GATE. No agent/wave result gets booked into a ledger without a
RE-RUN RECEIPT — and the gate applies SYMMETRICALLY to PASS and FAIL.

Receipt validation applies equally to positive and negative results. The gate opens
  the cited artifact and checks its recorded metrics against the submitted values.

CONTRACT (adapter doctrine: thin/dumb mapping, hard round-trip gate, loud misfit):
  A fold-submission (the small record an agent produces to book a result) MUST contain:
    (a) verdict-type: one of {positive, negative, partial}          [unless HYPOTHESIS-tagged]
    (b) RE-RUN RECEIPT: rerun.command   -> a runnable command line (its.py token must exist)
                                    rerun.artifact  -> a file that EXISTS on disk
                                    rerun.checks[]  -> >=1 key metric, each looked up in the artifact and
                                                       VERIFIED to match within tolerance (not just claimed)
    (c) for a NEGATIVE verdict: mechanism       -> WHY the signal is absent / the honest mechanism
                                    control         -> the control that separates 'method wrong' from
                                                       'signal doesn't exist' (a positive control that PASSES
                                                       proves the method can see the signal when present)
    (d) HYPOTHESIS-tag exception: a submission tagged HYPOTHESIS is ALLOWED but only AS A HYPOTHESIS
                                    (decision=ALLOW, as='hypothesis') — never booked as a result.

  decision = ALLOW iff missing == [].  Otherwise BLOCK, and missing[] names every unmet requirement,
  including per-check number MISMATCHES with an explicit claimed-vs-actual diff.

VERIFICATION STRENGTH:
  Level 1 (default, cheap, always-on): artifact exists + every cited metric matches the artifact on disk.
    This alone catches the two harvest failure modes: no-artifact-cited, and report-number != artifact-number.
  Level 2 (--execute): actually RE-RUN rerun.command (bounded by --timeout, default 110s), then re-verify the
    cited metrics against the freshly-written artifact. This is the full OMKORNING. If the command exceeds the
    timeout the gate BLOCKs with a 'restructure re-run to a subset' instruction (never a silent pass).

Callable:  fold_gate(report_text_or_json, base_dir=None, execute=False, timeout=110) -> dict with
           keys {decision: 'ALLOW'|'BLOCK', missing: [...], as, verdict, checks, artifact,...}
CLI:       python3 scripts/fold_gate.py <submission.json | ->            # gate one submission
           python3 scripts/fold_gate.py --execute <submission.json>      # Level-2: also re-run the command
           python3 scripts/fold_gate.py --selftest                       # run the 4 real cases, write JSON
"""
import json, os, re, sys, subprocess

# repo root = parent of scripts/ (this file lives in scripts/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_HYP = re.compile(r'\bHYPOTHES(IS|ES|IZED)?\b|\bawaiting[-\s]?qc\b|\bspeculativ', re.I)
_VERDICTS = ("positive", "negative", "partial")


# ---------------------------------------------------------------- helpers
def _resolve(path, base_dir):
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


def _dig(obj, dotted):
    """Traverse a dotted key path through nested dict/list. Returns (found, value).

    ★AMBIGUITY-FIX (SEED-FAILOPEN-AUDIT-THE-JUDGES, ported from cert_grade_engine.py's
    already-fixed twin _dig: both files implement the same dotted-path walk, but this copy still
    had the naive bug that fix closed): a dotted path segment can itself contain a literal '.' (a
    decimal embedded in an identifier, e.g. 'PF_Boron_fs0.5') -- naive split('.') treats EVERY dot
    as a path separator and silently returns (False, None) even when the real key exists (confirmed
    live: _dig({'2_falsification': {'PF_Boron_fs0.5': 42.0}}, '2_falsification.PF_Boron_fs0.5')
    returned found=False before this fix -- a false-negative feeding directly into fold_gate's own
    admission check at its call site). Fix: at each dict level, try the LONGEST remaining run of
    parts joined by '.' that is an ACTUAL key first, shrinking one part at a time; fall back to
    single-part traversal (original behavior) only when no literal-joined key matches -- a path with
    no dots-inside-keys has no shorter literal match to prefer, so resolution is unchanged for every
    case that worked before."""
    cur = obj
    parts = dotted.split('.')
    n = len(parts)
    i = 0
    while i < n:
        if isinstance(cur, list):
            try:
                cur = cur[int(parts[i])]
            except (ValueError, IndexError):
                return False, None
            i += 1
            continue
        if isinstance(cur, dict):
            matched = False
            for j in range(n, i, -1):
                candidate = '.'.join(parts[i:j])
                if candidate in cur:
                    cur = cur[candidate]
                    i = j
                    matched = True
                    break
            if matched:
                continue
            seg = parts[i]
            if seg not in cur:
                return False, None
            cur = cur[seg]
            i += 1
            continue
        return False, None
    return True, cur


def _match(claimed, actual, tol):
    """True iff claimed matches actual. Numbers within tol (abs); bool/str/None exact."""
    if isinstance(claimed, bool) or isinstance(actual, bool):
        return claimed == actual
    if isinstance(claimed, (int, float)) and isinstance(actual, (int, float)):
        t = tol if tol is not None else (1e-6 + 1e-6 * abs(actual))
        return abs(claimed - actual) <= t
    return claimed == actual


def _script_token(command):
    """Extract the first.py path token from a command line, if any."""
    for tok in re.split(r'\s+', command.strip()):
        tok = tok.strip('"\'')
        if tok.endswith('.py'):
            return tok
    return None


def _coerce(report):
    """Accept a dict, a JSON string, a path to a JSON file, or free text -> (submission_dict, raw_text)."""
    if isinstance(report, dict):
        return report, json.dumps(report)
    if isinstance(report, str):
        s = report.strip()
        # a path to a json file?
        if len(s) < 4096 and os.path.exists(s):
            try:
                with open(s) as f:
                    txt = f.read()
                return json.loads(txt), txt
            except Exception:
                pass
        # a json string?
        if s[:1] in ('{', '['):
            try:
                return json.loads(s), s
            except Exception:
                pass
        # free text: not a structured submission
        return None, s
    return None, str(report)


# ---------------------------------------------------------------- the gate
def fold_gate(report_text_or_json, base_dir=None, execute=False, timeout=110):
    """Return the harvest-gate decision for one fold-submission.

    ALLOW iff the submission carries a verified re-run receipt (and, for a negative, a mechanism + a
    signal-vs-method control) — OR it is HYPOTHESIS-tagged (then ALLOW as hypothesis, never as result).
    """
    base_dir = base_dir or BASE_DIR
    sub, raw = _coerce(report_text_or_json)
    missing = []
    checks_out = []
    warnings = []

    # --- (d) HYPOTHESIS exception (checked first: tag wins over any result claim) -----------------
    hyp = False
    if isinstance(sub, dict):
        hyp = bool(sub.get("hypothesis")) or bool(_HYP.search(
            " ".join(str(sub.get(k, "")) for k in ("verdict", "tags", "status", "tag", "note"))))
    if _HYP.search(raw or ""):
        hyp = True
    if hyp:
        return {"decision": "ALLOW", "as": "hypothesis", "verdict": "hypothesis",
                "missing": [], "checks": [], "warnings": [],
                "note": "HYPOTHESIS-tagged: folded as a hypothesis only, NOT booked as a result."}

    # A non-hypothesis result MUST be a structured submission (free text cannot carry a verifiable receipt).
    if not isinstance(sub, dict):
        return {"decision": "BLOCK", "as": "result", "verdict": None,
                "missing": ["structured fold-submission required (dict/JSON with verdict + rerun receipt); "
                            "free-text results are not bookable — provide a re-run receipt or tag HYPOTHESIS"],
                "checks": [], "warnings": []}

    # --- (a) verdict-type -------------------------------------------------------------------------
    verdict = str(sub.get("verdict", "")).strip().lower()
    verdict = verdict.split("(")[0].strip()  # tolerate 'negative(codec_confounded)' style
    if verdict not in _VERDICTS:
        missing.append(f"verdict: must be one of {_VERDICTS} (got {sub.get('verdict')!r})")

    # --- (b) RE-RUN RECEIPT -----------------------------------------------------------------------
    rerun = sub.get("rerun") or {}
    if not isinstance(rerun, dict):
        rerun = {}
    command = str(rerun.get("command", "")).strip()
    artifact = rerun.get("artifact")
    raw_checks = rerun.get("checks") or []

    if not command:
        missing.append("rerun.command: a runnable command line is required (the re-run recipe)")
    else:
        tok = _script_token(command)
        if tok is not None and not os.path.exists(_resolve(tok, base_dir)):
            missing.append(f"rerun.command: script not found on disk: {tok}")

    art_path = _resolve(artifact, base_dir) if artifact else None
    art_obj = None

    # --- Level 2: actually re-run the command FIRST (it may (re)generate the artifact) ------------
    pre_mtime = os.path.getmtime(art_path) if (art_path and os.path.exists(art_path)) else None
    if execute and command:
        try:
            proc = subprocess.run(command, shell=True, cwd=base_dir, timeout=timeout,
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                missing.append(f"rerun EXECUTE: command exited {proc.returncode} "
                               f"(stderr tail: {proc.stderr.strip()[-200:]!r})")
            else:
                warnings.append("rerun EXECUTE: command re-ran clean (exit 0)")
                if pre_mtime is not None and art_path and os.path.exists(art_path) \
                        and os.path.getmtime(art_path) <= pre_mtime:
                    warnings.append("rerun EXECUTE: artifact mtime unchanged after re-run "
                                    "(command may not rewrite it) — numbers still verified below")
        except subprocess.TimeoutExpired:
            missing.append(f"rerun EXECUTE: command exceeded {timeout}s — RESTRUCTURE the re-run to a "
                           f"cheap subset/oracle that regenerates the cited metrics under the budget "
                           f"(never book on an un-run receipt)")

    # --- artifact must EXIST on disk (Level-1 receipt) or have been (re)generated by execute ------
    if not artifact:
        missing.append("rerun.artifact: an on-disk artifact file must be cited")
    elif not os.path.exists(art_path):
        missing.append(f"rerun.artifact: file not found on disk: {artifact}  (resolved: {art_path})")
    else:
        try:
            with open(art_path) as f:
                art_obj = json.load(f)
        except Exception as e:
            missing.append(f"rerun.artifact: exists but is not readable JSON ({type(e).__name__}: {e})")

    # verify each cited metric against the artifact
    if not raw_checks:
        missing.append("rerun.checks: at least one key metric must be cited AND looked up in the artifact")
    for i, chk in enumerate(raw_checks):
        if not isinstance(chk, dict):
            missing.append(f"rerun.checks[{i}]: must be an object {{key, expected[, tol]}}")
            continue
        key = chk.get("key") or chk.get("key_path") or chk.get("path")
        expected = chk.get("expected", chk.get("value"))
        tol = chk.get("tol")
        rec = {"key": key, "claimed": expected}
        if not key:
            rec.update(ok=False, actual=None, reason="no key")
            missing.append(f"rerun.checks[{i}]: missing 'key' (dotted path into the artifact)")
            checks_out.append(rec)
            continue
        if art_obj is None:
            rec.update(ok=False, actual=None, reason="artifact unavailable")
            checks_out.append(rec)
            continue  # artifact-missing already recorded above
        found, actual = _dig(art_obj, key)
        rec["actual"] = actual
        if not found:
            rec.update(ok=False, reason="key not present in artifact")
            missing.append(f"rerun.checks: key {key!r} not found in artifact")
        elif not _match(expected, actual, tol):
            try:
                diff = abs(expected - actual)
            except Exception:
                diff = f"{expected!r} != {actual!r}"
            rec.update(ok=False, reason="mismatch", diff=diff)
            missing.append(f"rerun.checks: MISMATCH at {key!r}: claimed={expected!r} "
                           f"actual={actual!r} diff={diff} (tol={tol if tol is not None else 'default'})")
        else:
            rec.update(ok=True)
        checks_out.append(rec)

    # --- (c) negative verdict needs mechanism + signal-vs-method control ---------------------------
    if verdict == "negative":
        if not str(sub.get("mechanism", "")).strip():
            missing.append("mechanism: a NEGATIVE must state the honest mechanism (why the signal is absent)")
        if not str(sub.get("control", "")).strip():
            missing.append("control: a NEGATIVE must name the control separating 'method wrong' from "
                           "'signal absent' (a positive control that PASSES proves the method can see it)")

    # --- soft freshness signal (non-blocking): stale artifact vs its script ----------------------
    if art_path and os.path.exists(art_path) and command:
        tok = _script_token(command)
        if tok:
            sp = _resolve(tok, base_dir)
            if os.path.exists(sp) and os.path.getmtime(art_path) < os.path.getmtime(sp):
                warnings.append(f"freshness: artifact is OLDER than its script ({tok}) — possibly stale; "
                                f"consider --execute to re-run (non-blocking)")

    decision = "ALLOW" if not missing else "BLOCK"
    return {"decision": decision, "as": "result", "verdict": verdict or None,
            "missing": missing, "checks": checks_out, "warnings": warnings,
            "artifact": art_path, "artifact_exists": bool(art_obj is not None)}


# ---------------------------------------------------------------- self-test on TODAY's real cases
def _synthetic_artifact(tmpdir):
    """Write the small artifact the selftest cases cite, so the selftest is self-contained."""
    art_rel = os.path.join("reports", "probes", "example_validation.json")
    art_abs = os.path.join(tmpdir, art_rel)
    os.makedirs(os.path.dirname(art_abs), exist_ok=True)
    with open(art_abs, "w") as f:
        json.dump({
            "panel": {"case_a": {"slope": 0.03946}},
            "FINDING": {"naive_method_false_positive": True,
                        "conjunction_gate_fences_case_a": True,
                        "frozen_gate_both_captures_native": False},
            "controls": {"genuine_positive": {"reads_native": True}},
        }, f)
    return art_rel


def _selftest(tmpdir=None):
    """Four cases against a synthetic artifact written into a temp directory:
    a valid receipt ALLOWs, a clipped artifact path BLOCKs, a number mismatch BLOCKs with a diff,
    and a HYPOTHESIS-tagged submission ALLOWs as a hypothesis."""
    import tempfile
    tmp = tmpdir or tempfile.mkdtemp(prefix="fold_gate_selftest_")
    art_rel = _synthetic_artifact(tmp)
    cell_rel = "example_cell.py"
    with open(os.path.join(tmp, cell_rel), "w") as f:
        f.write("# example cell: regenerates the artifact cited by the receipt\n")
    cmd = f"python3 {cell_rel}"

 # (i) a result with a verified receipt -> ALLOW
    valid_negative = {
        "cell": "example_cell_validation",
        "verdict": "negative",
        "rerun": {
            "command": cmd,
            "artifact": art_rel,
            "checks": [
                {"key": "panel.case_a.slope", "expected": 0.03946, "tol": 1e-4},
                {"key": "FINDING.naive_method_false_positive", "expected": True},
                {"key": "FINDING.conjunction_gate_fences_case_a", "expected": True},
                {"key": "FINDING.frozen_gate_both_captures_native", "expected": False},
                {"key": "controls.genuine_positive.reads_native", "expected": True},
            ],
        },
        "mechanism": ("The measured statistic is confounded by the encoder: at block level it injects "
                      "intensity-correlated residual noise that mimics the law under test, so the leg "
                      "ABSTAINS (OUT_OF_REGIME) rather than issue the verdict it would wrongly give."),
        "control": ("The positive control reads NATIVE through the same encoder, so the method DOES detect a "
                    "genuine signal; the negative control is fenced. The abstention is signal-absent-in-regime, "
                    "NOT method failure."),
    }

    # (ii) same report with the artifact path clipped -> BLOCK
    clipped = json.loads(json.dumps(valid_negative))
    clipped["rerun"].pop("artifact")

    # (iii) fabricated report whose number does NOT match the artifact -> BLOCK with a diff
    fabricated = json.loads(json.dumps(valid_negative))
    fabricated["rerun"]["checks"] = [
        {"key": "panel.case_a.slope", "expected": 0.5, "tol": 1e-4},  # real is 0.03946
    ]

    # (iv) HYPOTHESIS-tagged -> ALLOW as hypothesis
    hypothesis = {
        "cell": "next_idea",
        "verdict": "positive",  # claimed, but the tag downgrades it
        "hypothesis": True,
        "note": "HYPOTHESIS awaiting-QC: an unverified idea for a follow-up measurement.",
    }

    cases = {
        "i_valid_receipt": (valid_negative, "ALLOW"),
        "ii_artifact_path_clipped": (clipped, "BLOCK"),
        "iii_number_mismatch": (fabricated, "BLOCK"),
        "iv_hypothesis_tagged": (hypothesis, "ALLOW"),
    }

    results = {}
    all_pass = True
    for name, (sub, expect) in cases.items():
        res = fold_gate(sub, base_dir=tmp)
        ok = (res["decision"] == expect)
        detail_ok = True
        if name == "i_valid_receipt":
            detail_ok = (res.get("as") == "result" and all(c["ok"] for c in res["checks"]))
        if name == "iv_hypothesis_tagged":
            detail_ok = (res.get("as") == "hypothesis")
        if name == "iii_number_mismatch":
            detail_ok = any("MISMATCH" in m for m in res["missing"])
        if name == "ii_artifact_path_clipped":
            detail_ok = any("rerun.artifact" in m for m in res["missing"])
        passed = ok and detail_ok
        all_pass = all_pass and passed
        results[name] = {
            "expected_decision": expect,
            "got_decision": res["decision"],
            "as": res.get("as"),
            "PASS": passed,
            "missing": res["missing"],
            "checks": res["checks"],
            "warnings": res.get("warnings", []),
        }

    return {
        "gate": "fold_gate (harvest gate)",
        "premise": ("no result -- PASS or FAIL -- is booked into a ledger without a verified re-run receipt; "
                    "negatives need a mechanism plus a signal-vs-method control; HYPOTHESIS folds as "
                    "hypothesis only."),
        "artifact_under_test": art_rel,
        "all_pass": all_pass,
        "cases": results,
    }


# ---------------------------------------------------------------- CLI
def _print(res):
    print(f"decision: {res['decision']}  (as {res.get('as')})  verdict={res.get('verdict')}")
    for m in res.get("missing", []):
        print(f"  MISSING: {m}")
    for c in res.get("checks", []):
        flag = "ok" if c.get("ok") else "FAIL"
        print(f"  check[{flag}] {c.get('key')}: claimed={c.get('claimed')!r} actual={c.get('actual')!r}"
              + (f" diff={c.get('diff')}" if c.get("diff") is not None else ""))
    for w in res.get("warnings", []):
        print(f"  warn: {w}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--selftest":
        out = _selftest()
        outdir = os.environ.get("FOLD_GATE_OUT", os.path.join(os.getcwd(), "reports", "probes"))
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "fold_gate_selftest.json")
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2)
        for name, r in out["cases"].items():
            print(f"[{'PASS' if r['PASS'] else 'FAIL'}] {name}: expected {r['expected_decision']} "
                  f"got {r['got_decision']} (as {r['as']})")
            for m in r["missing"]:
                print(f"        MISSING: {m}")
        print(f"ALL_PASS={out['all_pass']}  -> {outpath}")
        sys.exit(0 if out["all_pass"] else 1)

    execute = False
    if argv and argv[0] in ("--execute", "-x"):
        execute = True
        argv = argv[1:]

    if not argv:
        print(__doc__)
        print("usage: fold_gate.py [--execute] <submission.json | ->   |   --selftest")
        sys.exit(2)

    src = argv[0]
    if src == "-":
        report = sys.stdin.read()
    else:
        report = src  # path or JSON string; fold_gate coerces
    res = fold_gate(report, execute=execute)
    _print(res)
    sys.exit(0 if res["decision"] == "ALLOW" else 1)
