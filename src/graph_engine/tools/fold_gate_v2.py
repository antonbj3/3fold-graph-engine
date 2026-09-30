#!/usr/bin/env python3
"""fold_gate_v2.py — v1's harvest gate PLUS the ATOMS coverage/triviality gate.

BUILDS ON (does not replace) scripts/fold_gate.py:
  v1 (fold_gate) verifies the fold-submission carries a re-run receipt whose cited numbers actually match
     the artifact on disk; negatives need mechanism + a signal-vs-method control; HYPOTHESIS folds as
     hypothesis only.
  v2 additionally requires the SAME report to carry an ATOMS block (docs/ATOMS_STANDARD.md) that
     atoms_runner passes: every atom PASS, every load-bearing claim COVERED by >=1 non-weak passing atom,
     and no atom flagged TRIVIAL by the anti-gaming null-pass-rate check.

  decision = ALLOW iff  v1.decision == ALLOW  AND  atoms.decision == ALLOW.
  (A HYPOTHESIS-tagged submission short-circuits to ALLOW-as-hypothesis exactly as in v1 -- a hypothesis
   is not required to carry a full ATOMS receipt; it is not being booked as a result.)

INPUT: one report that is BOTH a fold-submission and an ATOMS carrier. Two supported shapes:
   * a dict/JSON with a 'rerun' receipt (for v1) and an 'ATOMS' key (for atoms_runner), or
   * a dict/JSON fold-submission + a separate `atoms_report` argument (text or dict) holding the block.

Callable:  fold_gate_v2(submission, atoms_report=None, base_dir=None, execute=False, timeout=110) -> dict
CLI:       python3 scripts/fold_gate_v2.py <submission.json>
           python3 scripts/fold_gate_v2.py --execute <submission.json>
           python3 scripts/fold_gate_v2.py --selftest
"""
import json, os, sys, math, shlex, copy, uuid, hashlib

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from fold_gate import fold_gate, BASE_DIR, _coerce, _HYP        # noqa: E402
from atoms_runner import run_atoms, parse_atoms_block            # noqa: E402


def fold_gate_v2(submission, atoms_report=None, base_dir=None, execute=False, timeout=110,
                 admission_guards=(), _calibration_cache=None, verification_registry=None):
    base_dir = base_dir or BASE_DIR
    v1 = fold_gate(submission, base_dir=base_dir, execute=execute, timeout=timeout)

    # HYPOTHESIS short-circuit: v1 already ALLOWed-as-hypothesis; do not demand a full ATOMS receipt.
    if v1.get("as") == "hypothesis":
        return {"decision": "ALLOW", "as": "hypothesis", "v1": v1, "atoms": None,
                "reason_codes": ["HYPOTHESIS_NOT_RESULT"],
                "missing": [], "note": "HYPOTHESIS: folded as hypothesis; ATOMS receipt not required."}

    # locate the ATOMS block: explicit arg > submission['ATOMS'] > submission itself (fenced text/dict)
    sub, _raw = _coerce(submission)
    src = atoms_report
    if src is None and isinstance(sub, dict):
        src = {"ATOMS": sub["ATOMS"]} if "ATOMS" in sub else sub
    atoms = run_atoms(src if src is not None else submission,
                      base_dir=base_dir, execute=execute, timeout=min(timeout, 30))

    missing = list(v1.get("missing", []))
    if atoms.get("decision") != "ALLOW":
        missing = missing + [f"ATOMS: {m}" for m in atoms.get("missing", [])]

    decision = "ALLOW" if (v1["decision"] == "ALLOW" and atoms.get("decision") == "ALLOW") else "BLOCK"
    out = {"decision": decision, "as": "result",
            "verdict": v1.get("verdict"),
            "v1_decision": v1["decision"], "atoms_decision": atoms.get("decision"),
            "missing": missing, "v1": v1, "atoms": atoms}
    out["reason_codes"] = []
    for check in v1.get("checks", []):
        if check.get("reason") == "key not present in artifact":
            out["reason_codes"].append("METRIC_ABSENT")
        elif check.get("reason") == "mismatch":
            out["reason_codes"].append("METRIC_MISMATCH")
    if isinstance(sub, dict):
        for chk in (sub.get("rerun") or {}).get("checks", []):
            tol = chk.get("tol") if isinstance(chk, dict) else None
            if tol is not None and (isinstance(tol, bool) or not isinstance(tol, (int, float))
                                     or not math.isfinite(tol) or tol < 0):
                out["reason_codes"].append("INVALID_TOLERANCE")
    guards = set(admission_guards)
    if not guards <= {"a", "b", "c"}:
        raise ValueError("unknown admission guard")
    if not guards:
        return out
    artifact = None
    try:
        with open(v1["artifact"]) as f:
            artifact = json.load(f)
    except (OSError, KeyError, TypeError, ValueError):
        pass
    artifact = artifact if isinstance(artifact, dict) else {}
    obligations = {}
    if "a" in guards:
        from graph_engine.agreement_admission_gate import equal_selectivity_control
        spec = artifact.get("agreement_calibration")
        if not isinstance(spec, dict):
            obligations["a"] = dict(validated=False, reason_codes=["CALIBRATION_MISSING"])
        else:
            key = json.dumps(spec, sort_keys=True)
            cache = _calibration_cache if _calibration_cache is not None else {}
            if key not in cache:
                try:
                    cache[key] = equal_selectivity_control(**spec)
                except (TypeError, ValueError):
                    cache[key] = dict(validated=False, reason_codes=["CALIBRATION_INVALID"])
            obligations["a"] = cache[key]
    if "b" in guards:
        # These typed witnesses are read from the receipt artifact, not the proposal
        # or the sentinel's expected class. Truthfulness still requires external audit.
        codes = []; witness = artifact.get("validation")
        required = {"instrument_family", "reference_family", "event_basis", "test_relation"}
        if (not isinstance(witness, dict) or not required <= witness.keys()
                or any(not isinstance(witness[k], str) or not witness[k] for k in required)):
            codes.append("VALIDATION_WITNESS_MISSING")
        else:
            if witness["instrument_family"] == witness["reference_family"]:
                codes.append("SELF_REFERENCE")
            if witness["event_basis"] != "external_observation":
                codes.append("EVENT_DEFINED_BY_THRESHOLD")
            if witness["test_relation"] != "independent_oracle":
                codes.append("TAUTOLOGICAL_CONTROL")
        command = str((sub.get("rerun") or {}).get("command", "")) if isinstance(sub, dict) else ""
        try:
            argv = shlex.split(command)
        except ValueError:
            argv = []
        if not argv or argv[0] in {"true", ":", "/bin/true", "/usr/bin/true"}:
            codes.append("ALWAYS_ZERO_COMMAND")
        # Reject the registered Python no-op recipe; do not claim general code semantics.
        tok = next((x for x in argv if x.endswith(".py")), None)
        if tok:
            path = tok if os.path.isabs(tok) else os.path.join(base_dir, tok)
            try:
                import ast
                tree = ast.parse(open(path).read())
                body = [x for x in tree.body if not (isinstance(x, ast.Expr)
                        and isinstance(x.value, ast.Constant) and isinstance(x.value.value, str))]
                if not body or all(isinstance(x, ast.Pass) for x in body):
                    codes.append("ALWAYS_ZERO_COMMAND")
            except (OSError, SyntaxError):
                codes.append("VALIDATION_WITNESS_MISSING")
        obligations["b"] = dict(validated=not codes, reason_codes=codes)
    if "c" in guards:
        from graph_engine.coupling_admission_precheck import source_family_admission
        reports = artifact.get("source_reports")
        if not isinstance(reports, list):
            obligations["c"] = dict(validated=False, reason_codes=["PROVENANCE_MISSING"])
        else:
            r = source_family_admission(reports, artifact.get("inherited_support", 0),
                                        verification_registry=verification_registry, require_registry=True)
            obligations["c"] = dict(validated=r["admit"], **r)
    from graph_engine.admission_bands import apply_admission_obligations
    return apply_admission_obligations(out, obligations)


def planted_fold_catalogue(submission, base_dir):
    """Fresh artifact mutations for every registered fault; labels stay out of payload.

    The operator mutates a valid receipt, not gate answers. The returned expected
    reason is only visible to the round supervisor. The caller owns base_dir.
    """
    sub = copy.deepcopy(submission)
    path = sub.get("rerun", {}).get("artifact")
    path = path if path and os.path.isabs(path) else os.path.join(base_dir, path or "")
    with open(path) as f:
        original = json.load(f)
    nonce = uuid.uuid4().hex
    recipes = [
        ("metric_absent", "METRIC_ABSENT"), ("metric_mismatch", "METRIC_MISMATCH"),
        ("self_reference", "SELF_REFERENCE"), ("event_threshold", "EVENT_DEFINED_BY_THRESHOLD"),
        ("tautology", "TAUTOLOGICAL_CONTROL"), ("zero_command", "ALWAYS_ZERO_COMMAND"),
        ("selection_only", "NO_EXCESS_AGREEMENT_GAIN"),
        ("duplicate_family", "SOURCE_FAMILY_DUPLICATION"), ("cascade", "COUPLING_CASCADE"),
        ("missing_calibration", "CALIBRATION_MISSING"), ("missing_provenance", "PROVENANCE_MISSING"),
        ("hypothesis_laundering", "HYPOTHESIS_NOT_RESULT"), ("infinite_tolerance", "INVALID_TOLERANCE"),
        ("missing_semantic_witness", "VALIDATION_WITNESS_MISSING"),
        ("unregistered_source", "PROVENANCE_MISSING")]
    probes = []
    for i, (kind, reason) in enumerate(recipes):
        probe = copy.deepcopy(sub); art = copy.deepcopy(original)
        # opaque identities prevent evaluators keying their answer on the fault name
        probe["cell"] = nonce + "-" + str(i)
        if kind == "metric_absent": probe["rerun"]["checks"][0]["key"] = nonce + ".absent"
        elif kind == "metric_mismatch": probe["rerun"]["checks"][0]["expected"] = -987654.0
        elif kind == "self_reference": art["validation"]["reference_family"] = art["validation"]["instrument_family"]
        elif kind == "event_threshold": art["validation"]["event_basis"] = "score_threshold"
        elif kind == "tautology": art["validation"]["test_relation"] = "identity"
        elif kind == "zero_command": probe["rerun"]["command"] = "/bin/true"
        elif kind == "selection_only":
            cal = art["agreement_calibration"]
            # Identical constituent rankings to agreement's selection: zero excess.
            cal["scores"] = [[2.0 if x else float(j)/len(cal["selected"])
                              for j, x in enumerate(cal["selected"])]] * 2
        elif kind == "duplicate_family":
            art["source_reports"] = [copy.deepcopy(art["source_reports"][0])] * 2
            art["inherited_support"] = 0
        elif kind == "cascade": art["source_reports"] = []; art["inherited_support"] = 2
        elif kind == "missing_calibration": art.pop("agreement_calibration", None)
        elif kind == "missing_provenance": art.pop("source_reports", None)
        elif kind == "missing_semantic_witness": art.pop("validation", None)
        elif kind == "unregistered_source": art["source_reports"] = [{"source_id": nonce + "-unknown"}]
        elif kind == "hypothesis_laundering": probe["hypothesis"] = True
        elif kind == "infinite_tolerance":
            probe["rerun"]["checks"][0].update(expected=-987654.0, tol=float("inf"))
        target = os.path.join(base_dir, "probe-" + nonce + "-" + str(i) + ".json")
        with open(target, "w") as f: json.dump(art, f)
        old = probe["rerun"]["artifact"]; probe["rerun"]["artifact"] = target
        for atom in probe.get("ATOMS", {}).get("atoms", []):
            if atom.get("artifact") == old: atom["artifact"] = target
        probes.append(dict(id=probe["cell"], fault_class=kind, required_reason=reason,
                           submission=probe, artifact_path=target))
    return probes


def fold_gate_v2_round(submissions, probe_template, base_dir, admission_guards=("a", "b", "c"),
                       evaluator=None, round_id=None, verification_registry=None):
    """Evaluate all proposals and fresh sentinels before exposing any accepted result.

    b blocks the WHOLE round if even one sentinel is accepted or misdiagnosed.
    An ALLOW-as-hypothesis never counts as result admission. No caller-supplied
    previous green round receipt can bypass rerunning these probes.
    """
    guards = tuple(admission_guards); cache = {}
    verification_registry = copy.deepcopy(verification_registry)
    if evaluator is None:
        evaluator = lambda s: fold_gate_v2(s, base_dir=base_dir, admission_guards=guards,
                                           _calibration_cache=cache, verification_registry=verification_registry)
    def evaluate(s):
        try:
            result = evaluator(copy.deepcopy(s))
            if not isinstance(result, dict): raise ValueError("non-object gate result")
            return result
        except Exception as exc:
            return dict(decision="BLOCK", reason_codes=["EVALUATOR_ERROR"], error=type(exc).__name__)
    results = [evaluate(s) for s in submissions]
    probes = []; failed = []
    if "b" in guards:
        try:
            catalogue = planted_fold_catalogue(probe_template, base_dir)
        except (OSError, KeyError, TypeError, ValueError):
            catalogue = []; failed.append("catalogue unavailable")
        try:
            for p in catalogue:
                r = evaluate(p["submission"])
                rejected = not (r.get("decision") == "ALLOW" and r.get("as") == "result")
                passed = rejected and p["required_reason"] in r.get("reason_codes", [])
                probes.append(dict(id=p["id"], fault_class=p["fault_class"], required_reason=p["required_reason"],
                                   passed=passed, evaluation=r))
                if not passed: failed.append(p["fault_class"])
        finally:
            for p in catalogue:
                os.unlink(p["artifact_path"])
    valid = not failed
    accepted = [i for i, r in enumerate(results) if valid and r.get("decision") == "ALLOW" and r.get("as") == "result"]
    payload_hash = hashlib.sha256(json.dumps(submissions, sort_keys=True).encode()).hexdigest()
    return dict(round_id=round_id or uuid.uuid4().hex, admission_guards=list(guards),
                fence_passed=valid, failed_fault_classes=failed, accepted_indices=accepted,
                proposals=results, probes=probes, submission_sha256=payload_hash,
                verification_registry_sha256=hashlib.sha256(json.dumps(verification_registry, sort_keys=True).encode()).hexdigest())


# ---------------------------------------------------------------- self-test: v1 cases carried into v2
def _selftest():
    """v1 receipt + ATOMS coverage on a synthetic artifact in a temp directory."""
    import tempfile
    from fold_gate import _synthetic_artifact

    tmp = tempfile.mkdtemp(prefix="fold_gate_v2_selftest_")
    art = _synthetic_artifact(tmp)
    cell_rel = "example_cell.py"
    with open(os.path.join(tmp, cell_rel), "w") as f:
        f.write("# example cell: regenerates the artifact cited by the receipt\n")

 # a well-formed v2 submission: v1 receipt AND an ATOMS block that passes + covers its claim
    good = {
        "cell": "example_cell_v2gate",
        "verdict": "negative",
        "rerun": {
            "command": f"python3 {cell_rel}",
            "artifact": art,
            "checks": [{"key": "panel.case_a.slope", "expected": 0.03946, "tol": 1e-4}],
        },
        "mechanism": "the statistic is encoder-confounded; the captures lack the white-noise fingerprint.",
        "control": "the positive control reads native through the same encoder -> the method can see the signal.",
        "ATOMS": {
            "report": "example_cell_negative",
            "claims": [{"id": "naive_fp", "text": "the naive statistic gives a false positive on case A",
                        "load_bearing": True}],
            "atoms": [
                {"id": "case_a_slope", "claim": "naive_fp", "type": "value-in-artifact", "artifact": art,
                 "key": "panel.case_a.slope", "expected": 0.03946, "tol": 1e-4},
                {"id": "naive_flag", "claim": "naive_fp", "type": "value-in-artifact", "artifact": art,
                 "key": "FINDING.naive_method_false_positive", "expected": True},
            ],
        },
    }
    # v1 ok but ATOMS number poisoned -> v2 BLOCK
    v1ok_atomsbad = json.loads(json.dumps(good))
    v1ok_atomsbad["ATOMS"]["atoms"][0]["expected"] = 0.5   # real is 0.03946
    # ATOMS ok but v1 receipt missing artifact -> v2 BLOCK
    atomsok_v1bad = json.loads(json.dumps(good))
    atomsok_v1bad["rerun"].pop("artifact")
    # HYPOTHESIS-tagged -> ALLOW as hypothesis (no ATOMS required)
    hyp = {"cell": "idea", "verdict": "positive", "hypothesis": True, "note": "HYPOTHESIS awaiting-QC"}

    cases = {
        "i_v1ok_atomsok": (good, "ALLOW"),
        "ii_v1ok_atoms_poisoned": (v1ok_atomsbad, "BLOCK"),
        "iii_atomsok_v1_missing_artifact": (atomsok_v1bad, "BLOCK"),
        "iv_hypothesis": (hyp, "ALLOW"),
    }
    results, all_pass = {}, True
    for name, (sub, expect) in cases.items():
        r = fold_gate_v2(sub, base_dir=tmp)
        ok = (r["decision"] == expect)
        detail = True
        if name == "ii_v1ok_atoms_poisoned":
            detail = r["v1_decision"] == "ALLOW" and r["atoms_decision"] == "BLOCK"
        if name == "iii_atomsok_v1_missing_artifact":
            detail = r["v1_decision"] == "BLOCK" and r["atoms_decision"] == "ALLOW"
        if name == "iv_hypothesis":
            detail = r.get("as") == "hypothesis"
        if name == "i_v1ok_atomsok":
            detail = r["v1_decision"] == "ALLOW" and r["atoms_decision"] == "ALLOW"
        passed = ok and detail
        all_pass = all_pass and passed
        results[name] = {"expected": expect, "got": r["decision"], "PASS": passed,
                         "v1_decision": r.get("v1_decision"), "atoms_decision": r.get("atoms_decision"),
                         "missing": r["missing"]}
    return {"gate": "fold_gate_v2 (v1 receipt + ATOMS coverage/triviality)",
            "all_pass": all_pass, "cases": results}


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--selftest":
        out = _selftest()
        outdir = os.environ.get("FOLD_GATE_OUT", os.path.join(os.getcwd(), "reports", "probes"))
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "fold_gate_v2_selftest.json")
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2)
        for name, r in out["cases"].items():
            print(f"[{'PASS' if r['PASS'] else 'FAIL'}] {name}: expected {r['expected']} got {r['got']} "
                  f"(v1={r['v1_decision']} atoms={r['atoms_decision']})")
        print(f"ALL_PASS={out['all_pass']} -> {outpath}")
        sys.exit(0 if out["all_pass"] else 1)
    execute = False
    if argv and argv[0] in ("--execute", "-x"):
        execute, argv = True, argv[1:]
    if not argv:
        print(__doc__)
        sys.exit(2)
    src = sys.stdin.read() if argv[0] == "-" else argv[0]
    res = fold_gate_v2(src, execute=execute)
    print(json.dumps({k: v for k, v in res.items() if k not in ("v1", "atoms")}, indent=2))
    sys.exit(0 if res["decision"] == "ALLOW" else 1)
